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

    # Resolve and pin immutable model table exactly as production HPO does.
    data_cfg = config.get("data", {})
    table_ref = resolve_model_table(
        snapshot_override_uri=data_cfg.get("snapshot_override_uri"),
        local_root=data_cfg.get("local_snapshot_root"),
    )

    config = enrich_config_with_lineage(config, table_ref)

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
        snapshot_source=None,
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