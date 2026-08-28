from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from machine_learning.data.features import resolve_feature_contract
from machine_learning.data.geographic_cv import (
    CANONICAL_GEOCV_VERSION,
    resolve_geographic_folds,
)
from machine_learning.data.model_table import (
    enrich_config_with_lineage,
    resolve_model_table,
)
from machine_learning.experiments.xgb_hpo import (
    Stage,
    load_yaml,
    resolve_training_module,
    run_train_once,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one fixed CrimeNet XGBoost 5-fold geographic-CV baseline."
    )
    parser.add_argument(
        "--local-snapshot-root",
        type=Path,
        default=Path("/workspace/crimenet_final_model"),
        help="Local directory used to stage the immutable final-model snapshot once.",
    )

    parser.add_argument(
        "--rclone-remote",
        default="b2",
        help="Configured rclone remote name for Backblaze B2.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--device",
        default="cuda",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--early-stop",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
    )
    return parser.parse_args()
def stage_snapshot_once(
    *,
    remote_uri: str,
    snapshot_id: str,
    local_stage_root: Path,
    rclone_remote: str,
) -> Path:
    """
    Download one immutable final-model snapshot to local disk exactly once.

    Returns the local snapshot root expected by resolve_model_table(local_root=...).
    """

    local_snapshot_root = (
        local_stage_root.expanduser().resolve() / f"snapshot_id={snapshot_id}"
    )

    ready_marker = local_snapshot_root / "_LOCAL_DOWNLOAD_COMPLETE.json"

    if ready_marker.exists():
        print(f"Local snapshot already staged: {local_snapshot_root}")
        return local_snapshot_root

    local_snapshot_root.mkdir(parents=True, exist_ok=True)

    if not remote_uri.startswith("s3://"):
        raise ValueError(
            f"Expected immutable S3-compatible snapshot URI, got: {remote_uri}"
        )

    # Example:
    # s3://crimenet-data/gold/final_model_table/snapshot_id=XYZ
    #
    # Convert to the path understood by the configured rclone B2 remote:
    # b2:crimenet-data/gold/final_model_table/snapshot_id=XYZ
    relative = remote_uri.removeprefix("s3://")
    rclone_source = f"{rclone_remote}:{relative}"

    print("=" * 80)
    print("STAGING IMMUTABLE MODEL TABLE LOCALLY")
    print("=" * 80)
    print(f"Remote: {remote_uri}")
    print(f"Rclone: {rclone_source}")
    print(f"Local:  {local_snapshot_root}")

    subprocess.run(
        [
            "rclone",
            "copy",
            rclone_source,
            str(local_snapshot_root),
            "--progress",
            "--transfers",
            "16",
            "--checkers",
            "32",
            "--multi-thread-streams",
            "4",
            "--fast-list",
        ],
        check=True,
    )

    parquet_files = list(local_snapshot_root.rglob("*.parquet"))

    if not parquet_files:
        raise RuntimeError(
            f"Local snapshot staging produced no Parquet files: {local_snapshot_root}"
        )

    total_bytes = sum(path.stat().st_size for path in parquet_files)

    marker = {
        "snapshot_id": snapshot_id,
        "source_uri": remote_uri,
        "local_root": str(local_snapshot_root),
        "parquet_file_count": len(parquet_files),
        "bytes": total_bytes,
    }

    temp_marker = ready_marker.with_suffix(".tmp")
    temp_marker.write_text(
        json.dumps(marker, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temp_marker.replace(ready_marker)

    print(
        f"Local snapshot ready: files={len(parquet_files):,}, "
        f"size={total_bytes / (1024 ** 3):.2f} GiB"
    )

    return local_snapshot_root

def params_from_config(config: dict) -> dict:
    arch = config["architecture"]
    opt = config["optimization"]

    return {
        "max_depth": int(arch["max_depth"]),
        "max_bin": int(arch["max_bin"]),
        "max_cat_to_onehot": int(arch["max_cat_to_onehot"]),
        "learning_rate": float(opt["learning_rate"]),
        "subsample": float(opt["subsample"]),
        "colsample_bytree": float(opt["colsample_bytree"]),
        "min_child_weight": float(opt["min_child_weight"]),
        "reg_lambda": float(opt["reg_lambda"]),
        "reg_alpha": float(opt.get("reg_alpha", 0.0)),
        "max_delta_step": float(opt["max_delta_step"]),
        "gamma": float(opt.get("gamma", 0.0)),
    }


def main() -> None:
    args = parse_args()

    config = load_yaml(args.config)

# -------------------------------------------------------------------------
# 1. Resolve the canonical immutable REMOTE snapshot once.
# -------------------------------------------------------------------------

    data_cfg = config.get("data", {})

    remote_table_ref = resolve_model_table(
        snapshot_override_uri=data_cfg.get("snapshot_override_uri"),
        local_root=None,
    )

    remote_uri = str(remote_table_ref.uri)
    snapshot_id = str(remote_table_ref.snapshot_id)

    print(f"Resolved immutable remote snapshot: {snapshot_id}")
    print(f"Remote URI: {remote_uri}")


    # -------------------------------------------------------------------------
    # 2. Stage that exact immutable snapshot onto local disk ONCE.
    # -------------------------------------------------------------------------

    local_snapshot_root = stage_snapshot_once(
        remote_uri=remote_uri,
        snapshot_id=snapshot_id,
        local_stage_root=args.local_snapshot_root,
        rclone_remote=args.rclone_remote,
    )


    # -------------------------------------------------------------------------
    # 3. From this point onward, ONLY use local Parquet.
    # -------------------------------------------------------------------------

    table_ref = resolve_model_table(
        local_root=str(local_snapshot_root),
    )

    if table_ref.snapshot_id != remote_table_ref.snapshot_id:
        raise RuntimeError(
            "Local staged snapshot identity does not match resolved remote snapshot: "
            f"remote={remote_table_ref.snapshot_id}, "
            f"local={table_ref.snapshot_id}"
        )

    config = enrich_config_with_lineage(config, table_ref)

    # Force every downstream trainer/fold to the local snapshot.
    config["data"]["local_snapshot_root"] = str(local_snapshot_root)

    # Prevent downstream code from accidentally preferring a remote override.
    config["data"].pop("snapshot_override_uri", None)
    config["data"]["final_model_snapshot_uri"] = str(local_snapshot_root)

    # Resolve the corrected ZERO-SHOT feature contract before training.
    train_split = str(config["data"].get("train_split", "train"))
    schema = table_ref.scan_split(train_split).collect_schema().names()

    contract = resolve_feature_contract(
        config["features"],
        available_columns=schema,
    )

    config["features"]["resolved_numeric"] = list(contract.numeric)
    config["features"]["resolved_categorical"] = list(contract.categorical)
    config["features"]["feature_contract_hash"] = contract.contract_hash

    # Freeze/validate the five canonical folds.
    folds = resolve_geographic_folds(config)

    print("=" * 80)
    print("CRIMENET GEOGRAPHIC-CV BASELINE")
    print("=" * 80)
    print(f"Snapshot:            {table_ref.snapshot_id}")
    print(f"Feature hash:        {contract.contract_hash}")
    print(f"Numeric features:    {len(contract.numeric)}")
    print(f"Categorical features:{len(contract.categorical)}")
    print(f"GeoCV version:       {CANONICAL_GEOCV_VERSION}")
    print(f"Folds:               {len(folds)}")
    print(f"Train fraction:      {args.train_fraction}")
    print(f"Validation fraction: {args.validation_fraction}")
    print(f"Rounds:              {args.rounds}")
    print(f"Device:              {args.device}")
    print()

    print("NUMERIC FEATURES:")
    for name in contract.numeric:
        print(f"  {name}")

    print("\nCATEGORICAL FEATURES:")
    for name in contract.categorical:
        print(f"  {name}")

    # Strong zero-shot sanity check.
    history_prefixes = (
        "cell_crime_",
        "cell_violent_",
        "cell_property_",
        "city_crime_",
        "k1_crime_",
    )
    history_exact = {
        "has_crime_cell_28d",
        "has_crime_city_28d",
        "hours_since_last_crime_cell_capped_28d",
        "hours_since_last_crime_city_capped_28d",
        "cell_crime_24h_vs_28d_ratio",
        "cell_share_of_k1_crime_24h",
    }

    resolved = list(contract.numeric) + list(contract.categorical)

    bad = [
        feature
        for feature in resolved
        if feature.startswith(history_prefixes)
        or feature in history_exact
    ]

    if bad:
        raise RuntimeError(
            "Zero-shot feature contract contains crime-history predictors: "
            + ", ".join(sorted(bad))
        )

    module = resolve_training_module(
        config,
        family="intensity",
        explicit=None,
    )

    params = params_from_config(config)

    stage = Stage(
        name="baseline",
        train_fraction=float(args.train_fraction),
        validation_fraction=float(args.validation_fraction),
        num_boost_round=int(args.rounds),
        early_stopping_rounds=int(args.early_stop),
    )

    score, metrics, best_iteration = run_train_once(
        module=module,
        base_config=config,
        family="intensity",
        params=params,
        stage=stage,
        device=args.device,
        run_label="zero_shot_geocv_baseline",
        seed=int(args.seed),
        keep_artifacts=args.keep_artifacts,
        snapshot_source=str(local_snapshot_root),
    )

    print("\n" + "=" * 80)
    print("BASELINE COMPLETE")
    print("=" * 80)
    print(f"Macro OOF NLL/event: {score:.12f}")
    print(f"Best iteration:      {best_iteration}")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print("\nTEST SPLIT HAS NOT BEEN ACCESSED.")


if __name__ == "__main__":
    main()