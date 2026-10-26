import copy
import logging
import math
import os
import time
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
import utils
import yaml
from checkpoints import (
    checkpoint_config,
    load_training_checkpoint,
    save_training_checkpoint,
)
from config import Config
from neuralop.models import FNO
from reproducibility import init_reproducibility
from torch.utils.data import DataLoader
from tqdm import tqdm

from aiice import AIICE


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

    def state_dict(self, *args, **kwargs):
        """Keep neuraloperator constructor metadata out of PyTorch checkpoints."""
        state = super().state_dict(*args, **kwargs)
        state.pop("_metadata", None)
        return state

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
    seed: int | None,
    deterministic: bool,
    checkpoint_metadata: dict[str, Any],
    resume_from: str | None = None,
) -> tuple[float, nn.Module]:
    """Train one FNO with cosine LR, gradient clipping, and val early stopping, and resumable checkpoints."""
    
    cuda_active = torch.cuda.is_available() and torch.device(device).type == "cuda"
    if cuda_active:
        torch.cuda.reset_peak_memory_stats(device)

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
    criterion = _make_loss(bound_weight=args["bound_weight"])

    resume_state = (
        load_training_checkpoint(
            path=resume_from,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=checkpoint_metadata,
            device=device,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
        )
        if resume_from is not None
        else {}
    )
    start_epoch = resume_state.get("epoch", 0)
    training_seconds_before = resume_state.get("training_seconds", 0.0)
    previous_peak_allocated_gib = resume_state.get("peak_gpu_allocated_gib")
    previous_peak_reserved_gib = resume_state.get("peak_gpu_reserved_gib")
    resume_stopped_early = resume_state.get("stopped_early", False)
    loss_history: list[float] = resume_state.get("loss_history", [])
    val_loss_history: list[float] = resume_state.get("val_loss_history", [])
    epoch_seconds = resume_state.get("epoch_seconds", [])
    best_val_loss = resume_state.get("best_val_loss", math.inf)
    early_stopping_val_loss = resume_state.get("early_stopping_val_loss", math.inf)
    best_train_loss = resume_state.get("best_train_loss", math.inf)
    best_epoch: int | None = resume_state.get("best_epoch")
    best_state_dict: dict[str, Any] | None = resume_state.get("best_state_dict")
    epochs_no_improve = resume_state.get("epochs_no_improve", 0)
    del resume_state
    if resume_from is not None:
        logger.info(f"Resumed training from {resume_from} after epoch {start_epoch}")

    if cuda_active:
        torch.cuda.synchronize(device)
    training_started = time.perf_counter()

    end_epoch = start_epoch if resume_stopped_early else args["max_epoch"]
    for epoch in range(start_epoch, end_epoch):
        epoch_started = time.perf_counter()

        model.train()
        loss = 0
        for x, y in tqdm(train_dataloader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()

            outputs = model(x)
            train_loss = criterion(outputs, y)
            train_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args["grad_clip"])
            optimizer.step()
            loss += train_loss.item()

        loss = loss / len(train_dataloader)
        scheduler.step()
        loss_history.append(loss)

        model.eval()
        val_loss_sum = 0.0
        val_samples = 0
        with torch.no_grad():
            for x, y in val_dataloader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                batch_size = x.size(0)
                val_loss_sum += criterion(model(x), y).item() * batch_size
                val_samples += batch_size

        if val_samples == 0:
            raise ValueError("validation dataloader must contain at least one sample")
        val_loss = val_loss_sum / val_samples
        val_loss_history.append(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f'-- epoch : {epoch + 1}/{args["max_epoch"]}, '
            f"train_loss={loss:.6f}, val_loss={val_loss:.6f}, {current_lr=}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_train_loss = loss
            best_epoch = epoch + 1
            best_state_dict = copy.deepcopy(model.state_dict())

        if val_loss < early_stopping_val_loss - args["min_delta"]:
            early_stopping_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        should_stop = epochs_no_improve >= args["patience"]
        if should_stop:
            logger.warning("EARLY STOPPING TRIGGERED")

        if cuda_active:
            torch.cuda.synchronize(device)
        epoch_seconds.append(time.perf_counter() - epoch_started)

        save_training_checkpoint(
            path=f"{experiment_path}/last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            training_state={
                "epoch": epoch + 1,
                "loss_history": loss_history,
                "val_loss_history": val_loss_history,
                "epoch_seconds": epoch_seconds,
                "best_val_loss": best_val_loss,
                "early_stopping_val_loss": early_stopping_val_loss,
                "best_train_loss": best_train_loss,
                "best_epoch": best_epoch,
                "best_state_dict": best_state_dict,
                "epochs_no_improve": epochs_no_improve,
                "stopped_early": should_stop,
                "training_seconds": (
                    training_seconds_before + time.perf_counter() - training_started
                ),
                "peak_gpu_allocated_gib": (
                    max(
                        previous_peak_allocated_gib or 0.0,
                        torch.cuda.max_memory_allocated(device) / 1024**3,
                    )
                    if cuda_active
                    else None
                ),
                "peak_gpu_reserved_gib": (
                    max(
                        previous_peak_reserved_gib or 0.0,
                        torch.cuda.max_memory_reserved(device) / 1024**3,
                    )
                    if cuda_active
                    else None
                ),
            },
            config=checkpoint_metadata,
            device=device,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
        )

        if should_stop:
            break

    training_seconds = training_seconds_before + time.perf_counter() - training_started
    peak_allocated_gib = None
    peak_reserved_gib = None
    if cuda_active:
        peak_allocated_gib = max(
            previous_peak_allocated_gib or 0.0,
            torch.cuda.max_memory_allocated(device) / 1024**3,
        )
        peak_reserved_gib = max(
            previous_peak_reserved_gib or 0.0,
            torch.cuda.max_memory_reserved(device) / 1024**3,
        )
        logger.info(
            "-- peak GPU memory: "
            f"allocated={peak_allocated_gib:.2f} GiB, "
            f"reserved={peak_reserved_gib:.2f} GiB"
        )

    if best_state_dict is None or best_epoch is None:
        raise RuntimeError("training finished without a validation checkpoint")

    model.load_state_dict(best_state_dict)
    torch.save(model.state_dict(), f"{experiment_path}/model.pt")
    utils.plot_loss_history(
        train_history=loss_history,
        val_history=val_loss_history,
        save_path=f"{experiment_path}/loss_history.png",
        best_epoch=best_epoch,
    )
    logger.info(
        f"- End of training (best validation loss: {best_val_loss:.6f}, "
        f"epoch: {best_epoch})"
    )

    epochs_completed = len(epoch_seconds)
    training_summary = {
        "epochs_completed": epochs_completed,
        "training_seconds": round(training_seconds, 3),
        "mean_epoch_seconds": round(training_seconds / epochs_completed, 3)
        if epochs_completed
        else None,
        "epoch_seconds": [round(seconds, 3) for seconds in epoch_seconds],
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "train_loss_at_best_epoch": best_train_loss,
        "stopped_early": epochs_completed < args["max_epoch"],
        "seed": seed,
        "deterministic": deterministic,
        "peak_gpu_allocated_gib": round(peak_allocated_gib, 2)
        if peak_allocated_gib is not None
        else None,
        "peak_gpu_reserved_gib": round(peak_reserved_gib, 2)
        if peak_reserved_gib is not None
        else None,
    }
    with open(f"{experiment_path}/training-summary.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(training_summary, f, sort_keys=False)

    logger.info("- All savings are done!")
    return best_val_loss, model


# ---------------------------------------------------------------------------
# Driver registered in cli.py via match cfg.run.model_name
# ---------------------------------------------------------------------------
def run(
    logger: logging.Logger,
    cfg: Config,
    sea: str | None,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
) -> None:
    """Train and validate FNO using explicit dates, then benchmark on test."""
    experiment_path = f"{cfg.output_path}/fno/{sea}"
    os.makedirs(experiment_path, exist_ok=True)

    best_loss_value = math.inf
    best_iteration = 0
    best_model: nn.Module | None = None

    spatial_shape = tuple(train_dataloader.dataset[0][0].shape[-2:])

    for i, experiment in enumerate(cfg.run.experiments):
        i_experiment_path = f"{experiment_path}/{i}"
        os.makedirs(i_experiment_path, exist_ok=True)

        init_reproducibility(
            seed=cfg.run.seed,
            deterministic=cfg.run.deterministic,
            logger=logger,
        )
        if cfg.run.seed is not None and train_dataloader.generator is not None:
            train_dataloader.generator.manual_seed(cfg.run.seed)

        loss_value, model = train(
            logger=logger,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            experiment_path=i_experiment_path,
            in_time_points=cfg.aiice.pre_history_len,
            out_time_points=cfg.aiice.forecast_len,
            spatial_shape=spatial_shape,
            args=experiment,
            device=cfg.device,
            seed=cfg.run.seed,
            deterministic=cfg.run.deterministic,
            resume_from=cfg.run.resume_from,
            checkpoint_metadata=checkpoint_config(cfg, sea, experiment),
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
