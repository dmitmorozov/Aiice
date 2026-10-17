import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from config import Config
from torch.utils.data import DataLoader



def checkpoint_config(cfg: Config, sea: str | None, experiment: dict) -> dict:
    return {
        "model_name": cfg.run.model_name,
        "args": experiment.copy(),
        "seed": cfg.run.seed,
        "deterministic": cfg.run.deterministic,
        "device_type": torch.device(cfg.device).type,
        "context": {
            "data": cfg.aiice.model_dump(exclude={"sea", "threads"}),
            "sea": sea,
        },
    }


def _rng_state(device: str, train: DataLoader, val: DataLoader) -> dict:
    numpy_state = list(np.random.get_state())
    numpy_state[1] = numpy_state[1].tolist()
    return {
        "python": random.getstate(),
        "numpy": numpy_state,
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device)
        if torch.device(device).type == "cuda"
        else None,
        "train_generator": train.generator.get_state()
        if train.generator is not None
        else None,
        "val_generator": val.generator.get_state()
        if val.generator is not None
        else None,
    }


def _restore_rng_state(state: dict, train: DataLoader, val: DataLoader, device: str):
    random.setstate(state["python"])
    numpy_state = list(state["numpy"])
    numpy_state[1] = np.asarray(numpy_state[1], dtype=np.uint32)
    np.random.set_state(tuple(numpy_state))
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device=device)
    for key, loader in (("train_generator", train), ("val_generator", val)):
        if (state[key] is None) != (loader.generator is None):
            raise ValueError(f"Checkpoint does not match {key}")
        if loader.generator is not None:
            loader.generator.set_state(state[key])


def save_training_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    training_state: dict,
    config: dict,
    device: str,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
) -> None:
    payload = {
        "config": config,
        "torch_version": str(torch.__version__),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "training_state": training_state,
        "rng": _rng_state(device, train_dataloader, val_dataloader),
    }
    target = Path(path)
    temporary = target.with_name(target.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def load_training_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    config: dict,
    device: str,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
) -> dict:
    checkpoint = torch.load(
        Path(path).expanduser(), map_location="cpu", weights_only=True
    )
    required = {"model", "optimizer", "scheduler", "training_state", "rng"}
    if (
        not isinstance(checkpoint, dict)
        or not required.issubset(checkpoint)
        or not ("config" in checkpoint or "signature" in checkpoint)
    ):
        raise ValueError("resume_from must point to a last.pt training checkpoint")

    saved = checkpoint.get("config", checkpoint.get("signature"))
    saved_config = {key: saved[key] for key in config}
    current_config = config.copy()
    saved_config["args"] = saved_config["args"].copy()
    current_config["args"] = current_config["args"].copy()
    saved_max_epoch = saved_config["args"].pop("max_epoch")
    current_max_epoch = current_config["args"].pop("max_epoch")
    if saved_config != current_config:
        differences = [
            key for key in config if saved_config[key] != current_config[key]
        ]
        raise ValueError("Checkpoint is incompatible with: " + ", ".join(differences))

    state = checkpoint["training_state"]
    if not isinstance(state, dict):
        raise ValueError("Checkpoint has no training state")
    completed = state.get("epoch")
    if not isinstance(completed, int) or completed < 1 or any(
        len(state[key]) != completed
        for key in ("loss_history", "val_loss_history", "epoch_seconds")
    ):
        raise ValueError("Checkpoint epoch and histories do not match")
    if current_max_epoch < max(saved_max_epoch, completed):
        raise ValueError("max_epoch cannot be decreased when resuming training")

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    rng = checkpoint["rng"]
    if isinstance(rng["cuda"], list):
        rng["cuda"] = rng["cuda"][saved.get("device_index") or 0]
    _restore_rng_state(rng, train_dataloader, val_dataloader, device)
    return state
