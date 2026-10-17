import logging
import os
import random

import numpy as np
import torch


def init_reproducibility(
    seed: int | None,
    deterministic: bool,
    logger: logging.Logger,
) -> None:
    if seed is None:
        if deterministic:
            raise ValueError("run.deterministic=true requires run.seed")
        logger.info("Reproducibility seed is not configured")
        return

    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)
    logger.info(f"Reproducibility configured: seed={seed}, {deterministic=}")
