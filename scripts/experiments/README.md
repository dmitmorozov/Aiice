
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
