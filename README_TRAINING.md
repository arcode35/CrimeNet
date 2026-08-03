# CrimeNet XGBoost training

These files train a weighted binary XGBoost baseline where:

- target: `is_observed_event`
- sample weight: `importance_weight`
- categorical predictor: `offense_mark`
- default split: train before 2024, validate on 2024, test from 2025 onward

## Copy into the repository

Merge the directories in this archive into `/Users/xor/crimenet_ml`.

Do not replace your existing `feature_sets.py` unless it differs from the included version.

## Install

From the repository root:

```bash
python -m pip install -r requirements-ml.txt
python -m pip install -e .
chmod +x scripts/train_history.sh scripts/train_core.sh
```

## Smoke test

```bash
./scripts/train_history.sh --limit-per-split 100000
```

## Full history baseline

```bash
./scripts/train_history.sh
```

## Full core model

```bash
./scripts/train_core.sh
```

## RTX 3080

Run in the environment where CUDA-enabled XGBoost is installed:

```bash
./scripts/train_history.sh --device cuda
```

The macOS build should use `--device cpu`.

## Output

Each run writes a new directory under `artifacts/xgboost/` containing:

- `model.ubj`
- `metrics.json`
- `evals_result.json`
- `metadata.json`
- `feature_importance.csv`
- `test_predictions.parquet`
- `config.yaml`

## Existing split partitions

The included configs use chronological cutoffs because the displayed schema did
not show a `dataset_split` column. To use Hive partitions such as
`dataset_split=train`, change the split block to:

```yaml
split:
  mode: existing_column
  column: dataset_split
  values:
    train: train
    validation: validation
    test: test
```
