from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from machine_learning.data.features import (
    resolve_feature_contract,
    validate_zero_shot_feature_contract,
)
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
        default=4000,
    )

    parser.add_argument(
        "--early-stop",
        type=int,
        default=150,
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

    parser.add_argument(
        "--local-snapshot-root",
        type=Path,
        default=Path("/workspace/crimenet_final_model"),
        help=(
            "Local staging directory for the immutable final-model snapshot. "
            "The snapshot is downloaded once and all folds read from this local copy."
        ),
    )

    parser.add_argument(
        "--rclone-remote",
        default="b2",
        help="Configured rclone remote name for Backblaze B2.",
    )

    return parser.parse_args()


def _snapshot_local_path(
    *,
    local_stage_root: Path,
    snapshot_id: str,
) -> Path:
    return (
        local_stage_root.expanduser().resolve()
        / f"snapshot_id={snapshot_id}"
    )


def _ready_marker_path(local_snapshot_root: Path) -> Path:
    return local_snapshot_root / "_LOCAL_DOWNLOAD_COMPLETE.json"


def _local_snapshot_looks_complete(
    *,
    local_snapshot_root: Path,
    snapshot_id: str,
) -> bool:
    """
    Cheap integrity check used before trusting the local completion marker.

    This deliberately does not replace resolve_model_table() validation.
    It only determines whether the expensive B2 copy can be skipped.
    """

    marker_path = _ready_marker_path(local_snapshot_root)
    manifest_path = local_snapshot_root / "manifest.json"

    if not marker_path.is_file():
        return False

    if not manifest_path.is_file():
        return False

    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    if not isinstance(marker, dict):
        return False

    if not isinstance(manifest, dict):
        return False

    if str(marker.get("snapshot_id", "")) != snapshot_id:
        return False

    if str(manifest.get("snapshot_id", "")) != snapshot_id:
        return False

    parquet_files = list(
        local_snapshot_root.glob("split=*/source_city=*/*.parquet")
    )

    if not parquet_files:
        return False

    expected_count = marker.get("parquet_file_count")

    if expected_count is not None:
        try:
            expected_count = int(expected_count)
        except (TypeError, ValueError):
            return False

        if len(parquet_files) != expected_count:
            return False

    return True


def stage_snapshot_once(
    *,
    remote_uri: str,
    snapshot_id: str,
    local_stage_root: Path,
    rclone_remote: str,
) -> Path:
    """
    Download one immutable final-model snapshot to local disk once.

    The local copy retains the canonical manifest and immutable snapshot
    identity. A completion marker is written only after rclone exits
    successfully and the expected partitioned Parquet structure exists.

    Re-running against the same completed immutable snapshot skips B2.
    """

    if not remote_uri.startswith("s3://"):
        raise ValueError(
            "Expected immutable S3-compatible snapshot URI, "
            f"got: {remote_uri}"
        )

    if not snapshot_id:
        raise ValueError("snapshot_id must be non-empty")

    if shutil.which("rclone") is None:
        raise RuntimeError(
            "rclone is required for local snapshot staging but was not found "
            "on PATH."
        )

    local_snapshot_root = _snapshot_local_path(
        local_stage_root=local_stage_root,
        snapshot_id=snapshot_id,
    )

    ready_marker = _ready_marker_path(local_snapshot_root)

    if _local_snapshot_looks_complete(
        local_snapshot_root=local_snapshot_root,
        snapshot_id=snapshot_id,
    ):
        print("=" * 80)
        print("LOCAL SNAPSHOT ALREADY PRESENT")
        print("=" * 80)
        print(f"Snapshot: {snapshot_id}")
        print(f"Local:    {local_snapshot_root}")
        print("Skipping B2 download.")
        print()

        return local_snapshot_root

    local_snapshot_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # If an earlier interrupted run wrote a completion marker before some
    # external corruption, remove it. rclone will safely resume/reconcile the
    # directory instead of downloading valid existing objects again.
    ready_marker.unlink(missing_ok=True)

    # Example:
    #
    # s3://crimenet-data/gold/final_model_table/snapshot_id=XYZ
    #
    # becomes:
    #
    # b2:crimenet-data/gold/final_model_table/snapshot_id=XYZ
    #
    relative_path = remote_uri.removeprefix("s3://")
    rclone_source = f"{rclone_remote}:{relative_path}"

    print("=" * 80)
    print("STAGING IMMUTABLE MODEL TABLE LOCALLY")
    print("=" * 80)
    print(f"Snapshot: {snapshot_id}")
    print(f"Remote:   {remote_uri}")
    print(f"Rclone:   {rclone_source}")
    print(f"Local:    {local_snapshot_root}")
    print()

    subprocess.run(
        [
            "rclone",
            "copy",
            rclone_source,
            str(local_snapshot_root),
            "--progress",
            "--stats",
            "10s",
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

    manifest_path = local_snapshot_root / "manifest.json"

    if not manifest_path.is_file():
        raise RuntimeError(
            "Local snapshot staging completed but manifest.json is missing: "
            f"{manifest_path}"
        )

    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Downloaded manifest is invalid JSON: {manifest_path}"
        ) from exc

    if not isinstance(manifest, dict):
        raise RuntimeError(
            "Downloaded final-model manifest must be a JSON object."
        )

    downloaded_snapshot_id = str(
        manifest.get("snapshot_id", "")
    )

    if downloaded_snapshot_id != snapshot_id:
        raise RuntimeError(
            "Downloaded snapshot identity mismatch: "
            f"expected={snapshot_id}, "
            f"manifest={downloaded_snapshot_id}"
        )

    parquet_files = list(
        local_snapshot_root.glob(
            "split=*/source_city=*/*.parquet"
        )
    )

    if not parquet_files:
        raise RuntimeError(
            "Local snapshot staging produced no partitioned Parquet files: "
            f"{local_snapshot_root}"
        )

    total_bytes = sum(
        path.stat().st_size
        for path in parquet_files
    )

    marker = {
        "snapshot_id": snapshot_id,
        "source_uri": remote_uri,
        "local_root": str(local_snapshot_root),
        "parquet_file_count": len(parquet_files),
        "parquet_bytes": total_bytes,
    }

    temporary_marker = ready_marker.with_name(
        ready_marker.name + ".tmp"
    )

    temporary_marker.write_text(
        json.dumps(
            marker,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    temporary_marker.replace(ready_marker)

    print()
    print("=" * 80)
    print("LOCAL SNAPSHOT READY")
    print("=" * 80)
    print(f"Snapshot:      {snapshot_id}")
    print(f"Parquet files: {len(parquet_files):,}")
    print(
        f"Parquet size:  "
        f"{total_bytes / (1024 ** 3):.2f} GiB"
    )
    print(f"Local root:    {local_snapshot_root}")
    print()

    return local_snapshot_root


def params_from_config(
    config: dict,
) -> dict:
    arch = config["architecture"]
    opt = config["optimization"]

    return {
        "max_depth": int(
            arch["max_depth"]
        ),
        "max_bin": int(
            arch["max_bin"]
        ),
        "max_cat_to_onehot": int(
            arch["max_cat_to_onehot"]
        ),
        "learning_rate": float(
            opt["learning_rate"]
        ),
        "subsample": float(
            opt["subsample"]
        ),
        "colsample_bytree": float(
            opt["colsample_bytree"]
        ),
        "min_child_weight": float(
            opt["min_child_weight"]
        ),
        "reg_lambda": float(
            opt["reg_lambda"]
        ),
        "reg_alpha": float(
            opt.get(
                "reg_alpha",
                0.0,
            )
        ),
        "max_delta_step": float(
            opt["max_delta_step"]
        ),
        "gamma": float(
            opt.get(
                "gamma",
                0.0,
            )
        ),
    }


def main() -> None:
    args = parse_args()

    config = load_yaml(
        args.config
    )

    # ----------------------------------------------------------------------
    # 1. Resolve and pin the canonical immutable REMOTE snapshot.
    #
    # This does only the metadata/pointer resolution needed to establish the
    # immutable identity. We retain that canonical remote URI in lineage.
    # ----------------------------------------------------------------------

    data_cfg = config.get(
        "data",
        {},
    )

    remote_table_ref = resolve_model_table(
        snapshot_override_uri=data_cfg.get(
            "snapshot_override_uri"
        ),
        local_root=None,
    )

    snapshot_id = str(
        remote_table_ref.snapshot_id
    )

    remote_uri = str(
        remote_table_ref.snapshot_uri
    )

    print("=" * 80)
    print("RESOLVED CANONICAL MODEL TABLE")
    print("=" * 80)
    print(f"Snapshot: {snapshot_id}")
    print(f"Remote:   {remote_uri}")
    print()

    # ----------------------------------------------------------------------
    # 2. Download that exact immutable snapshot ONCE.
    #
    # If a completed local copy already exists for this snapshot ID, this
    # stage performs zero B2 data downloads.
    # ----------------------------------------------------------------------

    local_snapshot_root = stage_snapshot_once(
        remote_uri=remote_uri,
        snapshot_id=snapshot_id,
        local_stage_root=args.local_snapshot_root,
        rclone_remote=args.rclone_remote,
    )

    # ----------------------------------------------------------------------
    # 3. Re-resolve from local disk.
    #
    # resolve_model_table(local_root=...) validates the copied manifest and
    # retains the canonical immutable snapshot identity from that manifest.
    # ----------------------------------------------------------------------

    table_ref = resolve_model_table(
        local_root=str(
            local_snapshot_root
        )
    )

    if (
        table_ref.snapshot_id
        != remote_table_ref.snapshot_id
    ):
        raise RuntimeError(
            "Local staged snapshot identity does not match "
            "the resolved remote snapshot: "
            f"remote={remote_table_ref.snapshot_id}, "
            f"local={table_ref.snapshot_id}"
        )

    if (
        table_ref.snapshot_uri
        != remote_table_ref.snapshot_uri
    ):
        raise RuntimeError(
            "Local staged snapshot canonical URI does not match "
            "the resolved remote snapshot: "
            f"remote={remote_table_ref.snapshot_uri}, "
            f"local_manifest={table_ref.snapshot_uri}"
        )

    # Preserve canonical lineage from the immutable manifest.
    config = enrich_config_with_lineage(
        config,
        table_ref,
    )

    # Tell every downstream HPO/training access path to physically scan the
    # local snapshot.
    config.setdefault(
        "data",
        {},
    )

    config["data"]["local_snapshot_root"] = str(
        local_snapshot_root
    )

    # A remote override must not take precedence later.
    config["data"].pop(
        "snapshot_override_uri",
        None,
    )

    # IMPORTANT:
    # Do NOT replace final_model_snapshot_uri with the local path.
    #
    # enrich_config_with_lineage() intentionally records the canonical,
    # immutable remote snapshot URI for provenance. local_snapshot_root is
    # the physical read location only.

    # ----------------------------------------------------------------------
    # 4. Resolve feature contract using LOCAL Parquet metadata.
    # ----------------------------------------------------------------------

    train_split = str(
        config["data"].get(
            "train_split",
            "train",
        )
    )

    schema = (
        table_ref
        .scan_split(train_split)
        .collect_schema()
        .names()
    )

    contract = resolve_feature_contract(
        config["features"],
        available_columns=schema,
    )

    config["features"]["resolved_numeric"] = list(
        contract.numeric
    )

    config["features"]["resolved_categorical"] = list(
        contract.categorical
    )

    config["features"]["feature_contract_hash"] = (
        contract.contract_hash
    )

    # ----------------------------------------------------------------------
    # 5. Fail closed on the zero-shot feature contract.
    # ----------------------------------------------------------------------

    validate_zero_shot_feature_contract(
        numeric=list(
            contract.numeric
        ),
        categorical=list(
            contract.categorical
        ),
    )

    # ----------------------------------------------------------------------
    # 6. Freeze/validate canonical geographic folds.
    # ----------------------------------------------------------------------

    folds = resolve_geographic_folds(
        config
    )

    print("=" * 80)
    print("CRIMENET GEOGRAPHIC-CV BASELINE")
    print("=" * 80)
    print(
        f"Snapshot:             "
        f"{table_ref.snapshot_id}"
    )
    print(
        f"Canonical URI:        "
        f"{table_ref.snapshot_uri}"
    )
    print(
        f"Physical read root:   "
        f"{table_ref.scan_root}"
    )
    print(
        f"Feature hash:         "
        f"{contract.contract_hash}"
    )
    print(
        f"Numeric features:     "
        f"{len(contract.numeric)}"
    )
    print(
        f"Categorical features: "
        f"{len(contract.categorical)}"
    )
    print(
        f"GeoCV version:        "
        f"{CANONICAL_GEOCV_VERSION}"
    )
    print(
        f"Folds:                "
        f"{len(folds)}"
    )
    print(
        f"Train fraction:       "
        f"{args.train_fraction}"
    )
    print(
        f"Validation fraction:  "
        f"{args.validation_fraction}"
    )
    print(
        f"Rounds:               "
        f"{args.rounds}"
    )
    print(
        f"Device:               "
        f"{args.device}"
    )
    print()

    print("NUMERIC FEATURES:")

    for name in contract.numeric:
        print(
            f"  {name}"
        )

    print()
    print("CATEGORICAL FEATURES:")

    for name in contract.categorical:
        print(
            f"  {name}"
        )

    # ----------------------------------------------------------------------
    # 7. Resolve production training implementation.
    # ----------------------------------------------------------------------

    module = resolve_training_module(
        config,
        family="intensity",
        explicit=None,
    )

    params = params_from_config(
        config
    )

    stage = Stage(
        name="baseline",
        train_fraction=float(
            args.train_fraction
        ),
        validation_fraction=float(
            args.validation_fraction
        ),
        num_boost_round=int(
            args.rounds
        ),
        early_stopping_rounds=int(
            args.early_stop
        ),
    )

    # ----------------------------------------------------------------------
    # 8. One candidate, five frozen geographic folds.
    #
    # snapshot_source is explicitly the local immutable snapshot root.
    # run_train_once() performs the five folds sequentially.
    # ----------------------------------------------------------------------

    score, metrics, best_iteration = run_train_once(
        module=module,
        base_config=config,
        family="intensity",
        params=params,
        stage=stage,
        device=args.device,
        run_label="zero_shot_geocv_baseline",
        seed=int(
            args.seed
        ),
        keep_artifacts=args.keep_artifacts,
        snapshot_source=str(
            local_snapshot_root
        ),
    )

    print()
    print("=" * 80)
    print("BASELINE COMPLETE")
    print("=" * 80)
    print(
        f"Macro OOF NLL/event: "
        f"{score:.12f}"
    )
    print(
        f"Best iteration:      "
        f"{best_iteration}"
    )
    print(
        json.dumps(
            metrics,
            indent=2,
            sort_keys=True,
        )
    )
    print()
    print(
        "TEST SPLIT HAS NOT BEEN ACCESSED."
    )


if __name__ == "__main__":
    main()
