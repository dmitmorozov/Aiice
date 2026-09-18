# API Reference

This section documents the public API of the `aiice-bench` package.

| Module | Description |
|---|---|
| [`aiice.AIICE`](aiice.md) | High-level `AIICE` class — entry point for loading data and running benchmarks |
| [`aiice.metrics`](metrics.md) | Metric functions and the `Evaluator` accumulator |
| [`aiice.loader`](loader.md) | `Loader` — downloads and masks OSI-SAF data from Hugging Face |
| [`aiice.preprocess`](preprocess.md) | `SlidingWindowDataset` and tensor utilities |
| [`aiice.core.huggingface`](core/huggingface.md) | `HfDatasetClient` - downloads raw data from Hugging Face |
