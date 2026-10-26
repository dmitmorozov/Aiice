
This section describes how to run experiments and update the AIICE benchmark leaderboard.

The repository provides multiple experimental setups for different models. Each model is defined by a separate configuration file located in the [`configs`](scripts/experiments/configs) directory.

---

To run an experiment, use the following command:

```bash
uv run scripts/experiments/cli.py \
    --config scripts/experiments/configs/baseline_mean.yaml
````

This will execute the selected model configuration and produce evaluation reports for the specified sea regions.

---

After an experiment is completed, you can add its results for a specific sea to the global leaderboard using:

```bash
uv run --project=scripts scripts/experiments/convert.py \
    --model baseline_mean \
    --sea "Kara Sea" \
    --forecast_len 54 \
    --step 1w \
    --report "outputs/baseline_mean/Kara Sea/report.yaml" \
    --csv docs/assets/leaderboard.csv
```

To add results for all seas at once, point `--report` to the output directory (subdirectories are treated as sea names, each should contain a YAML file):

```bash
uv run --project=scripts scripts/experiments/convert.py \
    --model baseline_mean \
    --forecast_len 54 \
    --step 1w \
    --report outputs/baseline_mean \
    --csv docs/assets/leaderboard.csv
```

This command:

* extracts metrics from the YAML report(s)
* associates the results with the given forecast length and step
* updates (or creates) the central CSV leaderboard, deduplicating by model, sea, metric, forecast length, and step

## Dataset split

All experiment configs use three explicit start dates:

```yaml
aiice:
  train_start_date: "1980-01-01"
  val_start_date: "2016-01-01"
  test_start_date: "2020-01-01"
```

The periods are `[train_start_date, val_start_date)`,
`[val_start_date, test_start_date)`, and `[test_start_date, dataset_end]`.
The first two upper bounds are excluded; `dataset_end` is included.
The dates must satisfy `train_start_date < val_start_date < test_start_date`.
Because `Loader.get()` includes both bounds, training and validation are loaded
through the day before the next period starts. Each input/target pair is built
entirely inside its own period. With the dates above, training covers 1980–2015,
validation covers 2016–2019, and testing starts in 2020.

## Resuming training

U-Net, Conv2D, Conv3D, ConvLSTM, and FNO save a full `last.pt` after each completed
epoch, beside the best-validation `model.pt` saved at normal completion. Set
`run.resume_from` to its path and select one sea and one experiment to resume.
Model, sea, dates/windows, batch size, seed, deterministic flag, CPU/CUDA type,
and training parameters must match; the total `max_epoch` may be increased.
Compatibility is checked from the configuration. Older `last.pt` formats are
supported; GPU index/count and `pin_memory` may change. An interrupted epoch
is repeated; a checkpoint stopped by patience is finalized without more training.
FNO restores its cosine scheduler, including the original `T_max`; increasing
`max_epoch` extends training without rebuilding the saved learning-rate schedule.

## Explanations

Here you can find some models' setup explanations

---

### FNO baseline

A Fourier Neural Operator (FNO) baseline is provided via `fno.py`, with one
config per scenario from Table 1 of the AIICE paper. The model wraps
`neuralop.models.FNO` as a "time-as-channel" forecaster
(`(B, T_in, H, W) -> (B, T_out, H, W)`) and is trained per sea.

FNO uses the shared CLI train/validation loaders and the explicit periods above.
The best weights are selected by any decrease in validation loss; `min_delta`
only controls the separate early-stopping counter. Training retains L1 plus
the soft bounds penalty; validation uses clamped predictions.
Its test period starts at `test_start_date`. Each experiment saves
`training-summary.yaml`, a combined train/validation loss plot, `model.pt`,
and the resumable `last.pt`.
To run all seven paper scenarios on the five leaderboard seas:

```bash
for sc in L1 L2 L3 L4 S1 S2 S3; do
    uv run --project=scripts scripts/experiments/cli.py \
        --config scripts/experiments/configs/fno_${sc}.yaml
done
```

The current configs pin `seed: 42`, `target_modes: 16`,
`val_start_date: "2016-01-01"`, and `test_start_date: "2020-01-01"`. The
spectral mode count is automatically clamped to the Nyquist limit
`spatial_dim // 2` for narrow sea crops (Sea of Japan, Chukchi Sea).

Full-Arctic configs are in `configs/global_artic/fno/`, one per scenario,
with `sea: null`, `seed: 42`, and `deterministic: true`. They retain the
64-channel, four-layer, 16-mode architecture and FP32 computation.

--

### HF-Mamba baseline

See scripts [here](./FH-Mamba/Dockerfile). This environment was tested on NVIDIA Tesla V100 GPUs.

The causal-conv1d and mamba-ssm extensions are compiled for compute capability sm_70; if you use a different GPU, rebuild the image with `TORCH_CUDA_ARCH_LIST` set to your architecture.
If training fails with a `CUDNN_STATUS_NOT_INITIALIZED` error, comment out the `total_flops = flops.total()` line in `trainer.py` to disable FLOPs estimation. This does not affect training results.
In our experiments, we increased `hid_S` to 32 and `hid_T_channels` to 24 in `config.py`. 

Sea ice concentration images were resized to 56x56, and the temporal-spatial patch was set to (1, 2, 2) in the `HilbertScan3DMambaBlock` initialization within `FH_Mamba.py`.
