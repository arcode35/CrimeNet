# CrimeNet machine learning

## Layout

```text
machine_learning/
├── artifacts/
│   ├── experiments/
│   └── mlflow/
├── experiments/
│   ├── experiment_logging.py
│   ├── mlflow_config.py
│   ├── orchestrator.py
│   └── log/
│       └── experiments.jsonl
└── models/
    └── xgboost/
        ├── model.py
        └── configs/
            └── baseline_v1.yaml
```

## Run

From `src/`:

```bash
python -m machine_learning.experiments.orchestrator \
  --config machine_learning/models/xgboost/configs/baseline_v1.yaml
```

## MLflow UI

From `src/`:

```bash
mlflow server \
  --backend-store-uri sqlite:////ABSOLUTE/PATH/TO/src/machine_learning/artifacts/mlflow/mlflow.db \
  --default-artifact-root file:///ABSOLUTE/PATH/TO/src/machine_learning/artifacts/mlflow/artifacts \
  --port 5000
```

The Python experiment runner does not need the server running; it writes directly to
the SQLite tracking store.
