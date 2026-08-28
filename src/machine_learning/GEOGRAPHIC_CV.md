# CrimeNet geographic model-selection contract

## Geographic OOF CV

HPO uses the frozen `crimenet_geocv_v1` mapping. One Optuna trial owns one GPU
and runs five fold models sequentially. Each fold excludes its three cities
from `split=train` and evaluates only those cities in `split=validation`.
Every modeling city therefore receives exactly one zero-shot OOF evaluation.
The primary objective is the equal-city, 15-city macro OOF NLL/event; pooled
OOF point-process metrics are computed by summing the original fold sufficient
statistics without changing Monte Carlo integration weights.

## Final production training

After tournament selection, HPO exports `best_config.yaml` with the winning
hyperparameters and median winning boosting-round count. That config trains one
model on 100% of `split=train` for all 15 cities. Geographic exclusions are
removed and fixed geographic folds remain only as model-selection provenance.

## Final in-domain temporal validation

The final production model evaluates all 15 cities in `split=validation`. This
is labeled `final_in_domain_temporal_validation`; it is a diagnostic and is not
the zero-shot selection estimate. The five-fold `geographic_oof_validation`
report is persisted alongside it.

## Test

The test partition is not exposed to HPO, caching, final fitting, category
discovery, or validation. Test evaluation requires a future distinct entrypoint.

Production HPO:

The default runner pins the canonical remote snapshot, stages only `train` and
`validation` beside the global Arrow stage cache, and resumes both snapshot
downloads and Optuna journals. Use `--snapshot-stage-dir` and
`--stage-cache-dir` to place both on the production NVMe. `split=test` is never
listed or staged.

```bash
uv run python -m machine_learning.experiments.xgb_hpo \
  --family intensity \
  --config src/machine_learning/models/xgboost/configs/intensity_transfer_prod_v1.yaml \
  --study-name intensity_transfer_prod_v1 \
  --output-dir hpo
```

Final fit, using the directly runnable config exported by HPO:

```bash
uv run python -m machine_learning.experiments.orchestrator \
  --config hpo/intensity_transfer_prod_v1_crimenet_geocv_v1_transfer_v2_<snapshot-prefix>_<feature-hash-prefix>/best_config.yaml
```
