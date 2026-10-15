from typing import Any

from pydantic import BaseModel


class Aiice(BaseModel):
    sea: str | list | None = None
    start_date: str  # YYYY-mm-dd
    end_date: str  # YYYY-mm-dd
    val_start_date: str | None = None  # YYYY-mm-dd
    val_end_date: str | None = None  # YYYY-mm-dd
    pre_history_len: int
    forecast_len: int
    step: int | str
    batch_size: int
    threads: int = 16


class Run(BaseModel):
    model_name: str
    seed: int | None = None
    deterministic: bool = False
    experiments: list[dict[str, Any]] = []


class Config(BaseModel):
    aiice: Aiice
    run: Run
    output_path: str
    device: str | None = None
