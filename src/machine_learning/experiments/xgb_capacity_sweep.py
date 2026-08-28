from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import yaml

from machine_learning.experiments import xgb_hpo as hpo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a deterministic fixed XGBoost intensity capacity sweep using the "
            "same CrimeNet CUDA trainer, five-fold geographic CV, snapshot staging, "
            "Arrow cache, and macro-city objective as xgb_hpo.py."
        )
    )
    parser.add_argument("--sweep-config", type=Path, required=True)
    parser.add_argument("--snapshot-stage-dir", type=Path, default=None)
    parser.add_argument("--stage-cache-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--threads-per-worker", type=int, default=None)
    parser.add_argument("--monitor-seconds", type=float, default=None)
    parser.add_argument("--no-snapshot-stage", action="store_true")
    parser.add_argument("--rebuild-stage-cache", action="store_true")
    parser.add_argument("--keep-trial-artifacts", action="store_true")
    return parser.parse_args()


def load_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        value = yaml.safe_load(fh)
    if not isinstance(value, dict):
        raise TypeError(f"Expected mapping at YAML root: {path}")
    return value


def resolve_repo_path(value: str | Path, *, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (repo_root / path).resolve()


def required_params(params: dict[str, Any]) -> dict[str, Any]:
    required = {
        "max_depth", "max_bin", "max_cat_to_onehot", "learning_rate",
        "subsample", "colsample_bytree", "min_child_weight", "reg_lambda",
        "reg_alpha", "max_delta_step", "gamma",
    }
    missing = sorted(required - set(params))
    if missing:
        raise ValueError(f"Candidate missing required parameter(s): {missing}")
    out = dict(params)
    out["max_depth"] = int(out["max_depth"])
    out["max_bin"] = int(out["max_bin"])
    out["max_cat_to_onehot"] = int(out["max_cat_to_onehot"])
    for key in required - {"max_depth", "max_bin", "max_cat_to_onehot"}:
        out[key] = float(out[key])
    return out


def build_candidates(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    common = dict(cfg["common_params"])
    depths = [int(x) for x in cfg["depths"]]
    regimes = cfg["regularization_regimes"]
    if not isinstance(regimes, dict) or not regimes:
        raise ValueError("regularization_regimes must be a non-empty mapping")

    candidates: list[dict[str, Any]] = []
    for regime_name, regime_params in regimes.items():
        for depth in depths:
            params = {**common, **dict(regime_params), "max_depth": depth}
            candidates.append({
                "name": f"{regime_name}_d{depth}",
                "group": str(regime_name),
                "params": required_params(params),
            })

    for probe in cfg.get("interaction_probes", []):
        params = {**common, **dict(probe.get("params", {}))}
        candidates.append({
            "name": str(probe["name"]),
            "group": "interaction_probe",
            "params": required_params(params),
        })

    names = [item["name"] for item in candidates]
    if len(names) != len(set(names)):
        raise ValueError("Candidate names must be unique")
    seen: set[str] = set()
    for item in candidates:
        key = json.dumps(item["params"], sort_keys=True, separators=(",", ":"))
        if key in seen:
            raise ValueError(f"Duplicate parameter set detected at {item['name']!r}")
        seen.add(key)
    return candidates


def fingerprint(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def validate_or_write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != manifest["fingerprint"]:
            raise RuntimeError(
                "Sweep definition changed while reusing an existing resume directory. "
                f"Use a new sweep_name/output directory or delete {path.parent}."
            )
        return
    atomic_write_json(path, manifest)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "rank", "candidate_name", "group", "score", "best_iteration",
        "max_depth", "max_bin", "max_cat_to_onehot", "learning_rate",
        "subsample", "colsample_bytree", "min_child_weight", "reg_lambda",
        "reg_alpha", "max_delta_step", "gamma",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd().resolve()
    sweep_cfg = load_mapping(args.sweep_config)
    sweep_name = str(sweep_cfg.get("sweep_name", "xgb_intensity_capacity_sweep"))
    family = str(sweep_cfg.get("family", "intensity"))
    if family != "intensity":
        raise ValueError("This runner is intentionally limited to family='intensity'.")

    base_config_path = resolve_repo_path(sweep_cfg["base_config"], repo_root=repo_root)
    base_config = hpo.load_yaml(base_config_path)

    data_cfg = dict(base_config.get("data", {}))
    canonical = hpo.resolve_model_table_from_config(data_cfg)
    base_config = hpo.enrich_config_with_lineage(base_config, canonical)
    contract = hpo.resolve_feature_contract(
        base_config["features"],
        available_columns=list(canonical.manifest["columns"]),
    )
    base_config["features"]["resolved_numeric"] = list(contract.numeric)
    base_config["features"]["resolved_categorical"] = list(contract.categorical)
    base_config["features"]["feature_contract_hash"] = contract.contract_hash

    folds = hpo.resolve_geographic_folds(base_config)
    if len(folds) != 5:
        raise RuntimeError(f"Expected canonical five-fold GeoCV, got {len(folds)} folds")

    stage_cfg = dict(sweep_cfg["stage"])
    stage = hpo.Stage(
        name="capacity_sweep",
        train_fraction=float(stage_cfg.get("train_fraction", 0.25)),
        validation_fraction=float(stage_cfg.get("validation_fraction", 0.25)),
        num_boost_round=int(stage_cfg.get("num_boost_round", 2500)),
        early_stopping_rounds=int(stage_cfg.get("early_stopping_rounds", 100)),
    )

    data_seed = int(base_config["data"]["seed"])
    configured_seed = sweep_cfg.get("seed")
    if configured_seed is not None and int(configured_seed) != data_seed:
        raise ValueError(
            f"Sweep seed={configured_seed} differs from base data seed={data_seed}. "
            "At partial fractions this creates a different deterministic sample/cache."
        )
    seed = data_seed
    candidates = build_candidates(sweep_cfg)

    runtime = dict(sweep_cfg.get("runtime", {}))
    gpus = hpo.parse_gpus(args.gpus or str(runtime.get("gpus", "0,1,2,3,4,5,6,7,8,9")))
    workers = hpo.resolved_worker_count(
        args.workers if args.workers is not None else int(runtime.get("workers", len(gpus))),
        gpus,
    )
    threads = args.threads_per_worker if args.threads_per_worker is not None else int(runtime.get("threads_per_worker", 0))
    monitor = args.monitor_seconds if args.monitor_seconds is not None else float(runtime.get("monitor_seconds", 30.0))

    output_root = args.output_dir or resolve_repo_path(runtime.get("output_dir", "hpo/capacity_sweeps"), repo_root=repo_root)
    tag = hpo.sanitize_name(
        f"{sweep_name}_{hpo.CANONICAL_GEOCV_VERSION}_{canonical.snapshot_id[:10]}_{contract.contract_hash[:10]}"
    )
    root = (Path(output_root) / tag).resolve()
    root.mkdir(parents=True, exist_ok=True)
    base_config["hpo_runtime"] = {
        "enabled": True,
        "report_root": str((root / "geocv_trial_reports").resolve()),
    }

    cache_dir = (args.stage_cache_dir or resolve_repo_path(runtime.get("stage_cache_dir", "hpo_nvme/stage_cache"), repo_root=repo_root)).resolve()
    snapshot_dir = (args.snapshot_stage_dir or resolve_repo_path(runtime.get("snapshot_stage_dir", "hpo_nvme/snapshot_stage"), repo_root=repo_root)).resolve()

    module = hpo.resolve_training_module(base_config, family=family, explicit=None)
    module_name = module.__name__
    del module

    if canonical.local_root:
        hpo_table_root = str(Path(canonical.local_root).resolve())
        hpo.assert_test_split_not_staged(Path(hpo_table_root))
    elif canonical.snapshot_uri.startswith("s3://"):
        plan = hpo.plan_hpo_snapshot_stage(
            snapshot_uri=canonical.snapshot_uri,
            snapshot_id=canonical.snapshot_id,
            lake=canonical.lake,
        )
        disk = hpo.preflight_hpo_disk(
            plan=plan,
            stage_dir=snapshot_dir,
            cache_dir=cache_dir,
            stage_enabled=not args.no_snapshot_stage,
        )
        print(f"remote parquet bytes: {disk['remote_parquet_bytes']:,}")
        print(f"free disk bytes:      {disk['free_disk_bytes']:,}")
        if args.no_snapshot_stage:
            hpo_table_root = canonical.snapshot_uri
        else:
            started = time.perf_counter()
            local_root = hpo.stage_hpo_snapshot(plan=plan, stage_dir=snapshot_dir, lake=canonical.lake, workers=12)
            print(f"[timing] snapshot staging/reuse: {time.perf_counter() - started:.3f}s")
            hpo.assert_test_split_not_staged(local_root)
            local_ref = hpo.resolve_model_table(local_root=str(local_root))
            if local_ref.lineage != canonical.lineage:
                raise RuntimeError("Local HPO snapshot lineage differs from canonical source")
            hpo_table_root = str(local_root)
            base_config["data"]["local_snapshot_root"] = hpo_table_root
    else:
        hpo_table_root = str(canonical.snapshot_uri)

    cache = hpo.prepare_stage_sample_cache(
        module_name=module_name,
        base_config=base_config,
        family=family,
        stage=stage,
        snapshot_source=hpo_table_root,
        cache_dir=cache_dir,
        rebuild=args.rebuild_stage_cache,
    )

    manifest_payload = {
        "sweep_name": sweep_name,
        "base_config": str(base_config_path),
        "snapshot_id": canonical.snapshot_id,
        "feature_contract_hash": contract.contract_hash,
        "fold_version": hpo.CANONICAL_GEOCV_VERSION,
        "folds": {name: list(cities) for name, cities in folds.items()},
        "seed": seed,
        "stage": stage.__dict__,
        "candidates": candidates,
        "test_split_used": False,
    }
    manifest = {**manifest_payload, "fingerprint": fingerprint(manifest_payload)}
    validate_or_write_manifest(root / "sweep_manifest.json", manifest)

    print("\n" + "=" * 80)
    print("FIXED INTENSITY CAPACITY SWEEP")
    print("=" * 80)
    print(f"Candidates:          {len(candidates)}")
    print(f"Objective:           {hpo.objective_metric(family)}")
    print(f"Sampling seed:       {seed}")
    print(f"Train/validation:    {stage.train_fraction:g}/{stage.validation_fraction:g}")
    print(f"GPUs/workers:        {len(gpus)}/{workers}")
    print(f"Snapshot:            {hpo_table_root}")
    print(f"Stage cache:         {cache_dir}")
    print(f"Output:              {root}")
    print("TEST SPLIT WILL NOT BE ACCESSED.\n")

    results = hpo.run_parallel_tournament(
        base_config=base_config,
        family=family,
        finalists=[item["params"] for item in candidates],
        seeds=[seed],
        stage=stage,
        device="cuda",
        study_tag=tag,
        keep_artifacts=args.keep_trial_artifacts,
        snapshot_source=hpo_table_root,
        gpus=gpus,
        worker_count=workers,
        threads_per_worker=threads,
        module_name=module_name,
        monitor_seconds=monitor,
        state_path=root / "sweep_state.json",
        sample_cache_entries=cache,
        cache_prepared_xy=False,
    )

    candidate_by_key = {
        json.dumps(item["params"], sort_keys=True, separators=(",", ":")): item
        for item in candidates
    }
    ranked: list[dict[str, Any]] = []
    for result in results:
        key = json.dumps(result["params"], sort_keys=True, separators=(",", ":"))
        candidate = candidate_by_key[key]
        seed_result = result["seed_results"][0]
        ranked.append({
            "rank": int(result["rank"]),
            "candidate_name": candidate["name"],
            "group": candidate["group"],
            "score": float(result["mean_score"]),
            "best_iteration": int(seed_result["best_iteration"]),
            **result["params"],
            "metrics": seed_result["metrics"],
        })

    ranked.sort(key=lambda row: row["score"])
    atomic_write_json(root / "ranked_results.json", ranked)
    write_csv(root / "ranked_results.csv", ranked)

    print("\n" + "=" * 80)
    print("CAPACITY SWEEP COMPLETE")
    print("=" * 80)
    for row in ranked[:10]:
        print(
            f"#{row['rank']:>2} {row['candidate_name']:<28} "
            f"depth={row['max_depth']:>2} score={row['score']:.12f} "
            f"best_iter={row['best_iteration']}"
        )
    print(f"\nJSON: {root / 'ranked_results.json'}")
    print(f"CSV:  {root / 'ranked_results.csv'}")
    print("TEST SPLIT HAS NOT BEEN ACCESSED.")


if __name__ == "__main__":
    main()
