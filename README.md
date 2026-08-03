# CrimeNet ML

CrimeNet trains the established single-node XGBoost point-process model. Its objective is
`count:poisson`, evaluated with `poisson-nloglik`. For row `i`, the Berman–Turner response is

```text
y_i = event_multiplicity_i / importance_weight_i
sample_weight_i = importance_weight_i
```

This makes XGBoost optimize (up to response-only constants) the point-process loss
`sum(w_i * lambda_i - event_i * log(lambda_i))`. It is not a binary classifier.

The training-split constant intensity `sum(events) / sum(weights)` initializes `base_score`.
Native XGBoost categorical handling is used for `offense_mark`, with vocabularies learned only
from training. Early stopping uses validation. After fitting, one calibration multiplier is
computed as validation observed event mass divided by validation raw predicted event mass. The
same fixed multiplier is applied to validation and test. The production `model` MLflow artifact
includes this multiplier; `raw_xgboost_model` is intentionally uncalibrated and is for debugging.

## Local setup and training

```bash
uv lock
uv sync --extra local --extra dev
uv run crimenet-train --config configs/local.yml
uv run crimenet-train --config configs/local.yml --limit-per-split 100000 --device cpu --run-name smoke-test
uv run crimenet-train --config configs/local.yml --device cuda
```

The default local tracking store is `sqlite:///mlflow.db`; artifacts and convenient copies are
written locally. To use a server, start one and override configuration without editing source:

```bash
uv run mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts
MLFLOW_TRACKING_URI=http://127.0.0.1:5000 uv run crimenet-train --config configs/local.yml
```

Every run records the resolved config, dataset identity/fingerprint, split totals, Git state,
XGBoost parameters, raw and calibrated validation/test metrics, iteration history, feature
importance, native model, and calibrated pyfunc model. The CLI prints the run ID, tracking URI,
artifact URI, and model URI.

Registration is a separate, explicit action and does not assign a Champion alias:

```bash
uv run crimenet-register --run-id RUN_ID --model-artifact model --registered-model-name crimenet_xgb_history_poisson
```

## Databricks

The Databricks adapter uses Spark only to read, filter, and project a Unity Catalog table. It
collects each split to the driver and trains the same single-node `XGBRegressor`; it does not use
distributed or Spark XGBoost. A row-count guard prevents accidental oversized collection unless
explicitly overridden.

```bash
databricks bundle validate --target dev
databricks bundle deploy --target dev
databricks bundle run --target dev train_xgb_history_poisson
```

Use `--target staging` or `--target prod` for the corresponding config and catalog. Override
`CRIMENET_DATA_TABLE`, `MLFLOW_TRACKING_URI`, `MLFLOW_REGISTRY_URI`, or
`MLFLOW_EXPERIMENT_NAME` as needed. Databricks authentication is supplied through the normal CLI
profile or service-principal environment; no credentials or personal cluster IDs are embedded.

## Troubleshooting

- Missing PySpark locally is expected for the local backend. Install `uv sync --extra databricks`
  only when exercising the Databricks adapter.
- For MLflow server errors, verify the server is reachable and that its artifact root is writable.
- For Databricks authentication failures, run `databricks auth profiles` and validate the selected
  profile before bundle commands.
