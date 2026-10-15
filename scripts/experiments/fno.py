import copy
import logging
import math
import os
from datetime import date, timedelta
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
import utils
import yaml
from config import Config
from neuralop.models import FNO
from reproducibility import init_reproducibility
from torch.utils.data import DataLoader
from tqdm import tqdm

from aiice import AIICE
from aiice.loader import Loader
from aiice.preprocess import SlidingWindowDataset


# ---------------------------------------------------------------------------
# Model: FNO as time-as-channel forecaster (B, T_in, H, W) -> (B, T_out, H, W)
# ---------------------------------------------------------------------------
class FNOForecaster(nn.Module):
    """Wraps `neuralop.models.FNO` for the AIICE contract.

    Aiice hands models tensors of shape ``(B, T_in, H, W)``, which already
    matches FNO2d's ``(B, C, H, W)`` layout: time is treated as the channel
    dimension. ``in_channels = T_in`` and ``out_channels = T_out`` so a single
    forward pass maps the entire input window to the entire forecast horizon.

    At evaluation time the output is clamped to the physical SIC range
    ``[0, 1]``. During training the clamp is disabled so the soft bound
    penalty in the loss can keep gradients flowing on out-of-range samples.
    """

    def __init__(
        self,
        t_in: int,
        t_out: int,
        n_modes: tuple[int, int],
        hidden_channels: int,
        n_layers: int,
    ) -> None:
        super().__init__()
        self.fno = FNO(
            n_modes=n_modes,
            in_channels=t_in,
            out_channels=t_out,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fno(x)
        if not self.training:
            out = out.clamp(0.0, 1.0)
        return out


def coerce_modes(h: int, w: int, target: int) -> tuple[int, int]:
    """Clamp ``n_modes`` to the Nyquist limit on each spatial axis.

    ``neuralop.FNO`` requires ``n_modes[i] <= spatial_dim[i] // 2``. Some
    AIICE sea crops are narrow enough (Sea of Japan, Chukchi Sea) that the
    target value of 16 has to be reduced.
    """
    return (min(target, h // 2), min(target, w // 2))


def safe_state_dict(model: nn.Module) -> dict[str, Any]:
    """Deep-copy the state dict and drop the ``_metadata`` key.

    ``neuralop.FNO`` stores tensorised weights via ``tltorch``, which leaves
    a ``_metadata`` entry as a dict in ``state_dict()``. Standard PyTorch
    keeps ``_metadata`` as an attribute on the OrderedDict, so a strict
    ``load_state_dict`` rejects the extra key. Removing it here keeps later
    reload code (and any downstream tooling) compatible.
    """
    sd = copy.deepcopy(model.state_dict())
    sd.pop("_metadata", None)
    return sd


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def _make_loss(bound_weight: float):
    """L1 plus a soft penalty pulling predictions back into ``[0, 1]``.

    The bound term is a physically motivated regulariser for SIC, which is a
    fraction by definition. Its weight is small (default 0.1) so the model
    is nudged toward feasibility rather than constrained against fitting.
    """

    def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l_data = (pred - target).abs().mean()
        l_bounds = (torch.relu(-pred) + torch.relu(pred - 1.0)).mean()
        return l_data + bound_weight * l_bounds

    return loss_fn


# ---------------------------------------------------------------------------
# Data: train and validation from non-overlapping date ranges
# ---------------------------------------------------------------------------
def _build_loaders(
    cfg: Config,
    sea: str | None,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, tuple[int, int]]:
    """Build loaders for [train_start, val_start) and [val_start, test_start)."""
    loader = Loader()
    val_start = date.fromisoformat(cfg.aiice.val_start_date)
    test_start = date.fromisoformat(cfg.aiice.test_start_date)

    train_data = loader.get(
        start=cfg.aiice.train_start_date,
        end=(val_start - timedelta(days=1)).isoformat(),
        sea=sea,
        step=cfg.aiice.step,
        tensor_out=True,
    )
    val_data = loader.get(
        start=cfg.aiice.val_start_date,
        end=(test_start - timedelta(days=1)).isoformat(),
        sea=sea,
        step=cfg.aiice.step,
        tensor_out=True,
    )

    train_ds = SlidingWindowDataset(
        data=train_data,
        pre_history_len=cfg.aiice.pre_history_len,
        forecast_len=cfg.aiice.forecast_len,
    )
    val_ds = SlidingWindowDataset(
        data=val_data,
        pre_history_len=cfg.aiice.pre_history_len,
        forecast_len=cfg.aiice.forecast_len,
    )

    generator = None
    if cfg.run.seed is not None:
        generator = torch.Generator()
        generator.manual_seed(cfg.run.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, pin_memory=True
    )

    h, w = train_data.shape[-2:]
    return train_loader, val_loader, (int(h), int(w))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(
    logger: logging.Logger,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    experiment_path: str,
    in_time_points: int,
    out_time_points: int,
    spatial_shape: tuple[int, int],
    args: dict[str, Any],
    device: str,
) -> tuple[float, nn.Module]:
    """Train one FNO with cosine LR, gradient clipping, and val early stopping."""
    h, w = spatial_shape
    n_modes = coerce_modes(h, w, target=args["target_modes"])
    if n_modes != (args["target_modes"], args["target_modes"]):
        logger.info(
            f"-- Reduced n_modes from "
            f"({args['target_modes']}, {args['target_modes']}) to {n_modes} "
            f"for crop {h}x{w} (Nyquist)"
        )

    model = FNOForecaster(
        t_in=in_time_points,
        t_out=out_time_points,
        n_modes=n_modes,
        hidden_channels=args["hidden_channels"],
        n_layers=args["n_layers"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"-- FNO params: {n_params / 1e6:.2f}M")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args["lr"],
        weight_decay=args["weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args["max_epoch"])
    loss_fn = _make_loss(bound_weight=args["bound_weight"])

    loss_history: list[float] = []
    val_loss_history: list[float] = []
    best_val_loss = math.inf
    best_state: dict[str, Any] | None = None
    epochs_no_improve = 0

    for epoch in range(args["max_epoch"]):
        model.train()
        train_loss_sum = 0.0
        for x, y in tqdm(train_dataloader, leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args["grad_clip"])
            optimizer.step()
            train_loss_sum += loss.item()
        train_loss = train_loss_sum / len(train_dataloader)
        scheduler.step()

        model.eval()
        v_total, v_n = 0.0, 0
        with torch.no_grad():
            for x, y in val_dataloader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                v_total += loss_fn(model(x), y).item() * x.size(0)
                v_n += x.size(0)
        val_loss = v_total / max(v_n, 1)

        loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"-- epoch : {epoch + 1}/{args['max_epoch']}, "
            f"{train_loss=:.5f}, {val_loss=:.5f}, {current_lr=:.2e}"
        )

        if val_loss < best_val_loss - args["min_delta"]:
            best_val_loss = val_loss
            best_state = safe_state_dict(model)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args["patience"]:
                logger.warning("EARLY STOPPING TRIGGERED")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    logger.info(f"- End of training (best val loss: {best_val_loss:.5f})")

    torch.save(safe_state_dict(model), f"{experiment_path}/model.pt")
    utils.plot_history(loss_history, f"{experiment_path}/loss_history.png", show=False)
    utils.plot_history(
        val_loss_history, f"{experiment_path}/val_loss_history.png", show=False
    )

    logger.info("- All savings are done!")
    return best_val_loss, model


# ---------------------------------------------------------------------------
# Driver registered in cli.py via match cfg.run.model_name
# ---------------------------------------------------------------------------
def run(
    logger: logging.Logger,
    cfg: Config,
    sea: str | None,
) -> None:
    """Train and validate FNO using explicit dates, then benchmark on test."""
    experiment_path = f"{cfg.output_path}/fno/{sea}"
    os.makedirs(experiment_path, exist_ok=True)

    best_loss_value = math.inf
    best_iteration = 0
    best_model: nn.Module | None = None

    train_loader, val_loader, spatial_shape = _build_loaders(
        cfg=cfg,
        sea=sea,
        batch_size=cfg.aiice.batch_size,
    )

    for i, experiment in enumerate(cfg.run.experiments):
        i_experiment_path = f"{experiment_path}/{i}"
        os.makedirs(i_experiment_path, exist_ok=True)

        init_reproducibility(
            seed=cfg.run.seed,
            deterministic=cfg.run.deterministic,
            logger=logger,
        )
        if cfg.run.seed is not None and train_loader.generator is not None:
            train_loader.generator.manual_seed(cfg.run.seed)

        loss_value, model = train(
            logger=logger,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            experiment_path=i_experiment_path,
            in_time_points=cfg.aiice.pre_history_len,
            out_time_points=cfg.aiice.forecast_len,
            spatial_shape=spatial_shape,
            args=experiment,
            device=cfg.device,
        )

        if loss_value < best_loss_value:
            best_iteration = i
            best_model = model
            best_loss_value = loss_value

    logger.info(f"Best loss model is here: {experiment_path}/{best_iteration}")

    aiice = AIICE(
        pre_history_len=cfg.aiice.pre_history_len,
        forecast_len=cfg.aiice.forecast_len,
        batch_size=cfg.aiice.batch_size,
        start=cfg.aiice.test_start_date,
        step=cfg.aiice.step,
        sea=sea,
        device=cfg.device,
        threads=cfg.aiice.threads,
    )
    report = aiice.bench(model=best_model, plot_workers=8)
    with open(f"{experiment_path}/best-model-{best_iteration}-report.yaml", "w") as f:
        yaml.safe_dump(report, f)

    logger.info("Eval is done!")
