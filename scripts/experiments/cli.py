import argparse
import logging
from datetime import date, timedelta

import baseline_mean
import baseline_repeat
import config
import conv2d
import conv3d
import convlstm
import fno
import torch
import unet
import yaml
from reproducibility import init_reproducibility
from torch.utils.data import DataLoader

from aiice.loader import Loader
from aiice.preprocess import SlidingWindowDataset


def init_logger() -> logging.Logger:
    logger = logging.getLogger("logger")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    return logger


def init_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def init_config() -> config.Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        data = yaml.safe_load(f)

    return config.Config(**data)


def init_dataloader(
    cfg: config.Aiice,
    device: str,
    sea: str | None,
    start_date: str | None = None,
    end_date: str | None = None,
    shuffle: bool = True,
    seed: int | None = None,
) -> DataLoader:
    loader = Loader()
    train_data = loader.get(
        start=start_date or cfg.train_start_date,
        end=end_date
        or (date.fromisoformat(cfg.val_start_date) - timedelta(days=1)).isoformat(),
        sea=sea,
        step=cfg.step,
        tensor_out=True,
        threads=cfg.threads,
    )
    train_dataset = SlidingWindowDataset(
        data=train_data,
        pre_history_len=cfg.pre_history_len,
        forecast_len=cfg.forecast_len,
        device=device,
    )
    generator = None
    if shuffle and seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        generator=generator,
    )
    return train_dataloader


def init_train_val_dataloaders(
    cfg: config.Aiice,
    device: str,
    sea: str | None,
    seed: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation loaders from the configured split dates."""
    test_start = date.fromisoformat(cfg.test_start_date)
    train_dataloader = init_dataloader(
        cfg,
        device=device,
        sea=sea,
        seed=seed,
    )
    val_dataloader = init_dataloader(
        cfg,
        device=device,
        sea=sea,
        start_date=cfg.val_start_date,
        end_date=(test_start - timedelta(days=1)).isoformat(),
        shuffle=False,
    )
    return train_dataloader, val_dataloader


def main():
    logger = init_logger()
    cfg = init_config()

    init_reproducibility(
        seed=cfg.run.seed,
        deterministic=cfg.run.deterministic,
        logger=logger,
    )

    if cfg.device is None:
        cfg.device = init_device()

    seas: list[str | None] = []
    if isinstance(cfg.aiice.sea, list):
        seas = cfg.aiice.sea
    else:
        seas.append(cfg.aiice.sea)

    for sea in seas:
        logger.info(f"=== Running for sea: {sea} ===")

        match cfg.run.model_name:
            case "conv2d":
                train_dataloader = init_dataloader(
                    cfg.aiice,
                    device=cfg.device,
                    sea=sea,
                )

                conv2d.run(
                    logger=logger, cfg=cfg, sea=sea, train_dataloader=train_dataloader
                )
            case "conv3d":
                train_dataloader = init_dataloader(
                    cfg.aiice,
                    device=cfg.device,
                    sea=sea,
                )

                conv3d.run(
                    logger=logger,
                    cfg=cfg,
                    sea=sea,
                    train_dataloader=train_dataloader,
                )
            case "convlstm":
                train_dataloader, val_dataloader = init_train_val_dataloaders(
                    cfg.aiice,
                    device=cfg.device,
                    sea=sea,
                    seed=cfg.run.seed,
                )

                convlstm.run(
                    logger=logger,
                    cfg=cfg,
                    sea=sea,
                    train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader,
                )
            case "fno":
                # FNO builds its loaders from the same explicit split dates.
                fno.run(
                    logger=logger,
                    cfg=cfg,
                    sea=sea,
                )
            case "unet":
                train_dataloader = init_dataloader(
                    cfg.aiice,
                    device=cfg.device,
                    sea=sea,
                )

                unet.run(
                    logger=logger,
                    cfg=cfg,
                    sea=sea,
                    train_dataloader=train_dataloader,
                )
            case "baseline_mean":
                baseline_mean.run(
                    logger=logger,
                    cfg=cfg,
                    sea=sea,
                )
            case "baseline_repeat":
                baseline_repeat.run(
                    logger=logger,
                    cfg=cfg,
                    sea=sea,
                )
            case _:
                raise ValueError("unknown experiment run type")


if __name__ == "__main__":
    main()
