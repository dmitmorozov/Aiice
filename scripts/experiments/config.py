from datetime import date
from typing import Any

from pydantic import BaseModel, model_validator


class Aiice(BaseModel):
    sea: str | list | None = None
    train_start_date: str  # inclusive, YYYY-mm-dd
    val_start_date: str  # inclusive; exclusive end of train, YYYY-mm-dd
    test_start_date: str  # inclusive; exclusive end of validation, YYYY-mm-dd
    pre_history_len: int
    forecast_len: int
    step: int | str
    batch_size: int
    threads: int = 16

    @model_validator(mode="after")
    def validate_date_order(self) -> "Aiice":
        train_start = date.fromisoformat(self.train_start_date)
        val_start = date.fromisoformat(self.val_start_date)
        test_start = date.fromisoformat(self.test_start_date)
        if not train_start < val_start < test_start:
            raise ValueError(
                "Date ranges must satisfy train_start_date < "
                "val_start_date < test_start_date"
            )
        return self


class Run(BaseModel):
    model_name: str
    seed: int | None = None
    deterministic: bool = False
    resume_from: str | None = None
    experiments: list[dict[str, Any]] = []


class Config(BaseModel):
    aiice: Aiice
    run: Run
    output_path: str
    device: str | None = None

    @model_validator(mode="after")
    def validate_resume(self) -> "Config":
        if self.run.resume_from is None:
            return self
        if len(self.run.experiments) != 1:
            raise ValueError("run.resume_from requires exactly one experiment")
        if isinstance(self.aiice.sea, list) and len(self.aiice.sea) != 1:
            raise ValueError("run.resume_from requires exactly one sea")
        return self
