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
from config import Config
from reproducibility import init_reproducibility
from torch.utils.data import DataLoader
from torchcnnbuilder.models import ForecasterBase
from tqdm import tqdm

from aiice import AIICE


class Conv3dModel(nn.Module):
    def __init__(self, forecaster_model: nn.Module, kernel_size):
        super().__init__()

        padding = tuple(k // 2 for k in kernel_size)

        self.forecaster = forecaster_model
        self.conv = nn.Sequential(
            nn.Conv3d(
                in_channels=1,
                out_channels=4,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                in_channels=4,
                out_channels=1,
                kernel_size=kernel_size,
                padding=padding,
            ),
        )

    def forward(self, x):
        # add channel dim (B, T, H, W) -> (B, 1, T, H, W) for conv3d
        x = x.unsqueeze(1)
        x = self.forecaster(x)
        x = self.conv(x)
        return x.squeeze(1)


def run(
    logger: logging.Logger,
    cfg: Config,
    sea: str | None,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
):
    experiment_path = f"{cfg.output_path}/conv3d/{sea}"
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
            data_shape=train_dataloader.dataset[0][0].shape[-2:],
            in_time_points=cfg.aiice.pre_history_len,
            out_time_point=cfg.aiice.forecast_len,
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
    data_shape: int,
    in_time_points: int,
    out_time_point: int,
    args: dict[str, Any],
    device: str,
    seed: int | None,
    deterministic: bool,
) -> tuple[float, nn.Module]:

    cuda_active = torch.cuda.is_available() and torch.device(device).type == "cuda"
    if cuda_active:
        torch.cuda.reset_peak_memory_stats(device)

    forecaster_params = {
        "input_size": data_shape,
        "n_layers": 5,
        "in_time_points": in_time_points,
        "out_time_points": out_time_point,
        "convolve_params": {"kernel_size": args["kernel_size"]},
        "transpose_convolve_params": {"kernel_size": args["kernel_size"]},
        "conv_dim": 3,
    }
    forecaster_model = ForecasterBase(**forecaster_params)
    model = Conv3dModel(forecaster_model, args["kernel_size"]).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args["lr"])
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
