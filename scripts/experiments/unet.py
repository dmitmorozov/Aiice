import copy
import logging
import math
import os
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
import yaml
from config import Config
from reproducibility import init_reproducibility
from torch.utils.data import DataLoader
from tqdm import tqdm

from aiice import AIICE


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int = 8):
        super().__init__()
        norm_groups = min(groups, out_channels)
        while out_channels % norm_groups != 0:
            norm_groups -= 1

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(norm_groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(norm_groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels, groups=groups),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        groups: int = 8,
    ):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )
        self.conv = DoubleConv(
            out_channels + skip_channels,
            out_channels,
            groups=groups,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_y != 0 or diff_x != 0:
            x = F.pad(
                x,
                [
                    diff_x // 2,
                    diff_x - diff_x // 2,
                    diff_y // 2,
                    diff_y - diff_y // 2,
                ],
            )

        return self.conv(torch.cat([skip, x], dim=1))


class UNetForecast(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 16,
        depth: int = 3,
        norm_groups: int = 8,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")

        channels = [base_channels * 2**i for i in range(depth + 1)]

        self.depth = depth
        self.inc = DoubleConv(in_channels, channels[0], groups=norm_groups)
        self.down_blocks = nn.ModuleList(
            DownBlock(channels[i], channels[i + 1], groups=norm_groups)
            for i in range(depth)
        )
        self.up_blocks = nn.ModuleList(
            UpBlock(
                in_channels=channels[i + 1],
                skip_channels=channels[i],
                out_channels=channels[i],
                groups=norm_groups,
            )
            for i in reversed(range(depth))
        )
        self.outc = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        x = self._pad_to_stride(x)

        skips = [self.inc(x)]
        for down in self.down_blocks:
            skips.append(down(skips[-1]))

        x = skips[-1]
        for up, skip in zip(self.up_blocks, reversed(skips[:-1])):
            x = up(x, skip)

        x = self.outc(x)
        x = x[..., :height, :width]
        return torch.sigmoid(x)

    def _pad_to_stride(self, x: torch.Tensor) -> torch.Tensor:
        stride = 2**self.depth
        height, width = x.shape[-2:]
        pad_h = (stride - height % stride) % stride
        pad_w = (stride - width % stride) % stride
        if pad_h == 0 and pad_w == 0:
            return x

        return F.pad(x, [0, pad_w, 0, pad_h], mode="replicate")


def run(
    logger: logging.Logger,
    cfg: Config,
    sea: str | None,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
):
    experiment_path = f"{cfg.output_path}/unet/{sea}"
    os.makedirs(experiment_path, exist_ok=True)

    best_loss_value = math.inf
    best_iteration = 0
    best_model: nn.Module | None = None

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
            in_channels=cfg.aiice.pre_history_len,
            out_channels=cfg.aiice.forecast_len,
            args=experiment,
            device=cfg.device,
            seed=cfg.run.seed,
            deterministic=cfg.run.deterministic,
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
    report = aiice.bench(
        model=best_model,
        # path=f"{experiment_path}/gif/",
        plot_workers=8,
    )

    with open(f"{experiment_path}/best-model-{best_iteration}-report.yaml", "w") as f:
        yaml.safe_dump(report, f)

    logger.info("Eval is done!")


def train(
    logger: logging.Logger,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    experiment_path: str,
    in_channels: int,
    out_channels: int,
    args: dict[str, Any],
    device: str,
    seed: int | None,
    deterministic: bool,
) -> tuple[float, nn.Module]:

    cuda_active = torch.cuda.is_available() and torch.device(device).type == "cuda"
    if cuda_active:
        torch.cuda.reset_peak_memory_stats(device)

    model = UNetForecast(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=args.get("base_channels", 16),
        depth=args.get("depth", 3),
        norm_groups=args.get("norm_groups", 8),
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=args["lr"], weight_decay=args.get("weight_decay", 0.0)
    )
    scheduler = optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=args["lr"],
        max_lr=0.005,
        step_size_up=30,
        mode="triangular2",
        cycle_momentum=False,
    )

    criterion = nn.L1Loss()
    loss_history: list[float] = []
    val_loss_history: list[float] = []
    epoch_seconds = []
    best_val_loss = math.inf
    early_stopping_val_loss = math.inf
    best_train_loss = math.inf
    best_epoch: int | None = None
    best_state_dict: dict[str, Any] | None = None
    epochs_no_improve = 0

    if cuda_active:
        torch.cuda.synchronize(device)
    training_started = time.perf_counter()

    for epoch in range(args["max_epoch"]):
        epoch_started = time.perf_counter()

        model.train()
        loss = 0
        for x, y in tqdm(train_dataloader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            outputs = model(x)
            train_loss = criterion(outputs, y)
            train_loss.backward()
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
                x = x.to(device)
                y = y.to(device)
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

        if should_stop:
            break

    training_seconds = time.perf_counter() - training_started
    peak_allocated_gib = None
    peak_reserved_gib = None
    if cuda_active:
        peak_allocated_gib = torch.cuda.max_memory_allocated(device) / 1024**3
        peak_reserved_gib = torch.cuda.max_memory_reserved(device) / 1024**3
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
