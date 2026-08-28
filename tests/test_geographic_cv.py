from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import yaml

from machine_learning.data.cache import cache_identity
from machine_learning.data.geographic_cv import (
    CANONICAL_GEOGRAPHIC_FOLDS,
    CANONICAL_MODELING_CITIES,
    aggregate_intensity_oof,
    resolve_geographic_folds,
    validate_geographic_folds,
)
from machine_learning.data.geography import (
    deterministic_sample,
    geographic_frames,
    validate_holdout_membership,
)
from machine_learning.experiments.xgb_hpo import (
    Stage,
    aggregate_tournament_geocv_metrics,
    build_final_production_config,
    prepare_stage_sample_cache,
    run_train_once,
)
from machine_learning.models.xgboost import model as intensity_module


def _config() -> dict:
    path = Path(
        "src/machine_learning/models/xgboost/configs/intensity_transfer_prod_v1.yaml"
    )
    return yaml.safe_load(path.read_text())


def _city_metric(city: str, *, nll_per_event: float, events: float = 1.0) -> dict:
    exposure = 10.0 + events
    expected = events * 1.1
    constant = nll_per_event + 0.5
    return {
        "source_city": city,
        "rows": int(events) + 1,
        "observed_events": events,
        "integration_rows": 1,
        "total_exposure": exposure,
        "exposure": exposure,
        "expected_events": expected,
        "expected_observed_ratio": expected / events,
        "calibration_error_pct": 10.0,
        "nll": nll_per_event * events,
        "nll_per_event": nll_per_event,
        "constant_nll_per_event": constant,
        "nll_gain_per_event": constant - nll_per_event,
        "bits_per_event": (constant - nll_per_event) / np.log(2.0),
    }


def _fold_reports() -> dict[str, dict]:
    reports: dict[str, dict] = {}
    index = 1
    for fold_name, cities in CANONICAL_GEOGRAPHIC_FOLDS.items():
        rows = []
        for city in cities:
            rows.append(_city_metric(city, nll_per_event=float(index), events=float(index)))
            index += 1
        observed = sum(row["observed_events"] for row in rows)
        nll = sum(row["nll"] for row in rows)
        expected = sum(row["expected_events"] for row in rows)
        exposure = sum(row["exposure"] for row in rows)
        reports[fold_name] = {
            "per_city": rows,
            "global": {
                "observed_events": observed,
                "expected_events": expected,
                "total_exposure": exposure,
                "nll": nll,
                "nll_per_event": nll / observed,
                "expected_observed_ratio": expected / observed,
                "calibration_error_pct": 10.0,
                "bits_per_event": 0.0,
            },
            "macro_city": {
                "mean_nll_per_event": float(np.mean([r["nll_per_event"] for r in rows])),
                "mean_bits_per_event": float(np.mean([r["bits_per_event"] for r in rows])),
            },
        }
    return reports


def test_frozen_fold_contract_and_order() -> None:
    folds = resolve_geographic_folds(_config())
    assert tuple(folds) == (
        "bay_area",
        "mid_atlantic",
        "dfw_southwest",
        "major_urban",
        "western_mixed",
    )
    assert len(folds) == 5
    assert all(len(cities) == 3 for cities in folds.values())
    assert {city for cities in folds.values() for city in cities} == CANONICAL_MODELING_CITIES


@pytest.mark.parametrize("mutation", ["duplicate", "omitted", "unknown"])
def test_fold_contract_fails_closed(mutation: str) -> None:
    folds = {name: list(cities) for name, cities in CANONICAL_GEOGRAPHIC_FOLDS.items()}
    if mutation == "duplicate":
        folds["western_mixed"][2] = "atlanta"
    elif mutation == "omitted":
        folds["western_mixed"].pop()
    else:
        folds["western_mixed"][2] = "unknown_city"
    with pytest.raises(ValueError):
        validate_geographic_folds(folds)


def test_each_fold_excludes_three_and_oof_covers_every_city_once() -> None:
    all_rows = pl.DataFrame({"source_city": sorted(CANONICAL_MODELING_CITIES)}).lazy()
    evaluated: list[str] = []
    for held_out in CANONICAL_GEOGRAPHIC_FOLDS.values():
        train, validation, _ = geographic_frames(
            train=all_rows,
            validation=all_rows,
            holdout_cities=held_out,
            report_in_domain=False,
        )
        train_cities = set(train.collect()["source_city"])
        validation_cities = set(validation.collect()["source_city"])
        assert len(train_cities) == 12
        assert not (train_cities & set(held_out))
        assert validation_cities == set(held_out)
        evaluated.extend(validation_cities)
    assert len(evaluated) == len(set(evaluated)) == 15
    assert set(evaluated) == CANONICAL_MODELING_CITIES


def test_geocv_training_membership_requires_all_other_twelve_cities() -> None:
    holdouts = CANONICAL_GEOGRAPHIC_FOLDS["bay_area"]
    expected_training = sorted(CANONICAL_MODELING_CITIES - set(holdouts))
    validation = pl.DataFrame({"source_city": list(holdouts)})
    validate_holdout_membership(
        training=pl.DataFrame({"source_city": expected_training}),
        validation=validation,
        holdout_cities=holdouts,
        expected_modeling_cities=CANONICAL_MODELING_CITIES,
    )
    with pytest.raises(ValueError, match="training city mismatch"):
        validate_holdout_membership(
            training=pl.DataFrame({"source_city": expected_training[:-1]}),
            validation=validation,
            holdout_cities=holdouts,
            expected_modeling_cities=CANONICAL_MODELING_CITIES,
        )


def test_oof_macro_and_integral_aggregation_are_not_renormalized() -> None:
    reports = _fold_reports()
    result = aggregate_intensity_oof(reports)
    cities = result["cities"]
    expected_macro = np.mean([row["nll_per_event"] for row in cities])
    total_nll = sum(report["global"]["nll"] for report in reports.values())
    total_events = sum(
        report["global"]["observed_events"] for report in reports.values()
    )
    total_exposure = sum(report["global"]["total_exposure"] for report in reports.values())
    assert result["metrics"]["geocv_macro_nll_per_event"] == pytest.approx(expected_macro)
    assert result["metrics"]["geocv_pooled_nll_per_event"] == pytest.approx(
        total_nll / total_events
    )
    assert result["metrics"]["geocv_total_exposure"] == pytest.approx(total_exposure)
    assert result["metrics"]["total_oof_nll"] == pytest.approx(total_nll)
    assert result["metrics"]["total_oof_observed_events"] == pytest.approx(total_events)
    assert result["metrics"]["total_oof_exposure"] == pytest.approx(total_exposure)
    assert expected_macro != pytest.approx(total_nll / total_events)


def test_cache_identity_is_global_and_reused_across_folds_and_stages() -> None:
    base = {
        "cache_version": "5",
        "snapshot_id": "snapshot",
        "schema_version": "schema",
        "feature_contract_hash": "features",
        "model_family": "intensity",
        "model_module": "module",
        "split": "train",
        "fraction": 0.5,
        "seed": 42,
    }
    explore = cache_identity(**base)
    refine_validation = cache_identity(**base)
    assert explore == refine_validation

    full = {**base, "fraction": 1.0, "seed": 0}
    refine_train = cache_identity(**full)
    tournament_train = cache_identity(**full)
    assert refine_train == tournament_train


@pytest.mark.parametrize("fraction", [0.25, 1.0])
def test_global_sample_then_fold_filter_matches_old_fold_sampling(
    fraction: float,
) -> None:
    rows = pl.DataFrame(
        {
            "model_row_id": [
                f"{city}-{index}"
                for city in sorted(CANONICAL_MODELING_CITIES)
                for index in range(20)
            ],
            "source_city": [
                city
                for city in sorted(CANONICAL_MODELING_CITIES)
                for _ in range(20)
            ],
        }
    ).lazy()
    global_sample = deterministic_sample(rows, fraction=fraction, seed=42)
    for held_out in CANONICAL_GEOGRAPHIC_FOLDS.values():
        old_train, old_validation, _ = geographic_frames(
            train=rows,
            validation=rows,
            holdout_cities=held_out,
            report_in_domain=False,
        )
        old_train_ids = set(
            deterministic_sample(old_train, fraction=fraction, seed=42)
            .collect()["model_row_id"]
            .to_list()
        )
        old_validation_ids = set(
            deterministic_sample(old_validation, fraction=fraction, seed=42)
            .collect()["model_row_id"]
            .to_list()
        )
        new_train, new_validation, _ = geographic_frames(
            train=global_sample,
            validation=global_sample,
            holdout_cities=held_out,
            report_in_domain=False,
        )
        assert set(new_train.collect()["model_row_id"]) == old_train_ids
        assert set(new_validation.collect()["model_row_id"]) == old_validation_ids


def test_one_candidate_invokes_five_fits_and_fold_failure_fails_trial(tmp_path: Path) -> None:
    calls: list[tuple[str, ...]] = []

    def train(config, *, run_id, config_hash):
        held = tuple(config["validation"]["geographic_holdout_cities"])
        calls.append(held)
        fold = str(getattr(module, "_hpo_active_fold_name"))
        rows = [_city_metric(city, nll_per_event=1.0) for city in held]
        report = {
            "per_city": rows,
            "global": {
                "observed_events": 3.0,
                "expected_events": 3.3,
                "total_exposure": sum(row["exposure"] for row in rows),
                "nll": 3.0,
                "nll_per_event": 1.0,
                "expected_observed_ratio": 1.1,
                "calibration_error_pct": 10.0,
                "bits_per_event": 0.0,
            },
            "macro_city": {"mean_nll_per_event": 1.0, "mean_bits_per_event": 0.0},
        }
        return {
            "metrics": {"best_iteration": 2.0},
            "geographic_validation": report,
            "artifacts": [],
            "summary": {"fold": fold},
        }

    module = SimpleNamespace(train=train)
    config = _minimal_hpo_config(tmp_path)
    score, metrics, best_iteration = run_train_once(
        module=module,
        base_config=config,
        family="intensity",
        params=_params(),
        stage=Stage("smoke", 1.0, 1.0, 2, 1),
        device="cpu",
        run_label="five-fold",
        seed=42,
        keep_artifacts=False,
    )
    assert len(calls) == 5
    assert calls == list(CANONICAL_GEOGRAPHIC_FOLDS.values())
    assert score == pytest.approx(1.0)
    assert metrics["geocv_macro_nll_per_event"] == pytest.approx(1.0)
    assert best_iteration == 2

    calls.clear()

    def failing_train(config, *, run_id, config_hash):
        if getattr(failing_module, "_hpo_active_fold_name") == "dfw_southwest":
            raise RuntimeError("synthetic fold failure")
        return train(config, run_id=run_id, config_hash=config_hash)

    failing_module = SimpleNamespace(train=failing_train)
    with pytest.raises(RuntimeError, match="dfw_southwest"):
        run_train_once(
            module=failing_module,
            base_config=config,
            family="intensity",
            params=_params(),
            stage=Stage("smoke", 1.0, 1.0, 2, 1),
            device="cpu",
            run_label="failure",
            seed=42,
            keep_artifacts=False,
        )


def _minimal_hpo_config(tmp_path: Path) -> dict:
    config = _config()
    config["features"] = {"feature_set": "synthetic", "numeric": ["feature"], "categorical": []}
    config["artifacts"]["output_root"] = str(tmp_path / "artifacts")
    config["hpo_runtime"] = {"report_root": str(tmp_path / "reports")}
    return config


def _params() -> dict:
    return {
        "max_depth": 2,
        "max_bin": 16,
        "max_cat_to_onehot": 4,
        "learning_rate": 0.1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "min_child_weight": 0.0,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "max_delta_step": 1.0,
        "gamma": 0.0,
    }


def test_hpo_export_becomes_pinned_all_city_production_config(tmp_path: Path) -> None:
    base = _minimal_hpo_config(tmp_path)
    base["data"].update(
        {
            "final_model_snapshot_id": "snapshot-1",
            "final_model_snapshot_uri": "s3://bucket/final/snapshot_id=snapshot-1",
            "final_model_schema_version": "v1",
            "local_snapshot_root": "/tmp/staged-copy",
        }
    )
    base["validation"]["geographic_holdout_cities"] = ["dallas"]
    exported = build_final_production_config(
        base_config=base,
        best_params=_params(),
        family="intensity",
        device="cuda",
        seed=42,
        winning_rounds=37,
        hpo_metadata={"winning_refine_trial_number": 9},
    )
    assert exported["data"]["final_model_snapshot_id"] == "snapshot-1"
    assert exported["data"]["final_model_snapshot_uri"].endswith("snapshot_id=snapshot-1")
    assert "local_snapshot_root" not in exported["data"]
    assert "geographic_holdout_cities" not in exported["validation"]
    assert exported["final_training"] == {"use_all_cities": True, "train_fraction": 1.0}
    assert exported["training"]["num_boost_round"] == 37
    assert exported["training"]["fixed_num_boost_round"] is True
    assert resolve_geographic_folds(exported) == CANONICAL_GEOGRAPHIC_FOLDS
    assert exported["hpo"]["winning_refine_trial_number"] == 9


def test_tournament_geocv_metrics_are_averaged_equally_across_seeds() -> None:
    seed_records = [
        {
            "metrics": {
                "geocv_macro_nll_per_event": score,
                "geocv_pooled_nll_per_event": score + 1.0,
                "geocv_report_path": f"seed-{seed}.json",
                "best_iteration": 10.0,
            }
        }
        for seed, score in ((42, 1.0), (1337, 4.0), (2026, 7.0))
    ]
    aggregated = aggregate_tournament_geocv_metrics(seed_records)
    assert aggregated == {
        "geocv_macro_nll_per_event": 4.0,
        "geocv_pooled_nll_per_event": 5.0,
    }


@pytest.fixture
def fifteen_city_snapshot(tmp_path: Path) -> Path:
    root = tmp_path / "snapshot"
    columns = [
        "model_row_id",
        "row_type",
        "event_indicator",
        "is_observed_event",
        "event_count",
        "integration_weight_cell_seconds",
        "source_city",
        "split",
        "feature",
    ]
    for split in ("train", "validation"):
        for index, city in enumerate(sorted(CANONICAL_MODELING_CITIES)):
            directory = root / f"split={split}" / f"source_city={city}"
            directory.mkdir(parents=True)
            pl.DataFrame(
                {
                    "model_row_id": [f"{split}-{city}-e", f"{split}-{city}-i"],
                    "row_type": ["event", "integration"],
                    "event_indicator": [1, 0],
                    "is_observed_event": [True, False],
                    "event_count": [1, 0],
                    "integration_weight_cell_seconds": [None, float(index + 1)],
                    "feature": [float(index), float(index + 1)],
                },
                schema_overrides={"integration_weight_cell_seconds": pl.Float64},
            ).write_parquet(directory / "part.parquet")
    sealed = root / "split=test" / "source_city=sealed"
    sealed.mkdir(parents=True)
    (sealed / "must-not-read.parquet").write_bytes(b"not parquet")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "snapshot_id": "fifteen-city",
                "snapshot_uri": "s3://canonical/final_model_table/snapshot_id=fifteen-city",
                "schema_version": "v1",
                "columns": columns,
            }
        )
    )
    return root


def test_hpo_fast_path_evaluates_only_validation_macro_metric(
    fifteen_city_snapshot: Path, tmp_path: Path
) -> None:
    config = _minimal_hpo_config(tmp_path)
    config["data"].update(
        {
            "local_snapshot_root": str(fifteen_city_snapshot),
            "train_fraction": 1.0,
            "validation_fraction": 1.0,
        }
    )
    config["features"]["categorical"] = []
    config["architecture"].update({"device": "cpu", "max_bin": 16, "max_depth": 2})
    config["training"].update(
        {"num_boost_round": 1, "early_stopping_rounds": 1, "verbose_eval": False}
    )
    config["final_training"]["use_all_cities"] = False
    config["validation"].update(
        {
            "geographic_holdout_cities": list(
                CANONICAL_GEOGRAPHIC_FOLDS["bay_area"]
            ),
            "report_in_domain_validation": False,
        }
    )
    config["hpo_runtime"]["enabled"] = True
    result = intensity_module.train(config, run_id="hpo-fast", config_hash="hash")
    assert result["artifacts"] == []
    assert result["summary"]["hpo_fast_path"] is True
    assert set(result["history"]) == {"validation"}
    assert set(result["history"]["validation"]) == {
        "macro_city_pp_nll_per_event"
    }
    assert not (tmp_path / "artifacts").exists()


def test_tiny_five_fold_then_final_all_city_smoke(
    fifteen_city_snapshot: Path, tmp_path: Path
) -> None:
    config = _minimal_hpo_config(tmp_path)
    config["data"].update(
        {
            "local_snapshot_root": str(fifteen_city_snapshot),
            "train_fraction": 1.0,
            "validation_fraction": 1.0,
        }
    )
    config["features"]["categorical"] = []
    config["architecture"].update({"device": "cpu", "max_bin": 16, "max_depth": 2})
    config["training"].update(
        {"num_boost_round": 1, "early_stopping_rounds": 1, "verbose_eval": False}
    )
    score, metrics, _ = run_train_once(
        module=intensity_module,
        base_config=config,
        family="intensity",
        params=_params(),
        stage=Stage("smoke", 1.0, 1.0, 1, 1),
        device="cpu",
        run_label="real-five-fold",
        seed=42,
        keep_artifacts=False,
    )
    report = json.loads(Path(metrics["geocv_report_path"]).read_text())
    assert np.isfinite(score)
    assert len(report["folds"]) == 5
    assert len(report["cities"]) == 15
    assert {row["source_city"] for row in report["cities"]} == CANONICAL_MODELING_CITIES
    artifact_root = tmp_path / "artifacts"
    assert not list(artifact_root.rglob("model.json"))
    assert not list(artifact_root.rglob("metadata.json"))
    assert not list(artifact_root.rglob("feature_importance.json"))
    assert not list(artifact_root.rglob("training_history.json"))

    final_config = copy.deepcopy(config)
    final_config.pop("hpo_runtime", None)
    final_config["final_training"] = {"use_all_cities": True, "train_fraction": 1.0}
    final_config["validation"].pop("geographic_holdout_cities", None)
    final_config["training"]["fixed_num_boost_round"] = True
    final_config["model"]["name"] = "synthetic-final"
    result = intensity_module.train(
        final_config, run_id="final", config_hash="synthetic-hash"
    )
    metadata = json.loads(
        (tmp_path / "artifacts" / "synthetic-final" / "final" / "metadata.json").read_text()
    )
    assert result["summary"]["test_split_used"] is False
    assert metadata["training_strategy"] == "final_all_city_train"
    assert metadata["geographic_validation"] is None
    assert metadata["validation"] == {
        "selection_metric_source": "geographic_oof_cv",
        "final_diagnostic_source": "full_validation_split_in_domain",
    }
    assert metadata["final_training"]["city_count"] == 15
    assert metadata["final_training"]["excluded_cities"] == []
    assert len(metadata["final_in_domain_temporal_validation"]["per_city"]) == 15
    assert (artifact_root / "synthetic-final" / "final" / "model.json").is_file()
    assert (artifact_root / "synthetic-final" / "final" / "training_history.json").is_file()


def test_global_arrow_cache_reuses_cross_stage_fraction_pairs(
    fifteen_city_snapshot: Path, tmp_path: Path
) -> None:
    config = _minimal_hpo_config(tmp_path)
    config["data"].update(
        {
            "local_snapshot_root": str(fifteen_city_snapshot),
            "final_model_snapshot_id": "fifteen-city",
            "final_model_snapshot_uri": (
                "s3://canonical/final_model_table/snapshot_id=fifteen-city"
            ),
            "final_model_schema_version": "v1",
            "seed": 42,
        }
    )
    cache_dir = tmp_path / "cache"
    common = {
        "module_name": "machine_learning.models.xgboost.model",
        "base_config": config,
        "family": "intensity",
        "snapshot_source": str(fifteen_city_snapshot),
        "cache_dir": cache_dir,
        "rebuild": False,
    }
    explore = prepare_stage_sample_cache(
        **common,
        stage=Stage("explore", 0.25, 0.25, 2, 1),
    )
    refine = prepare_stage_sample_cache(
        **common,
        stage=Stage("refine", 1.0, 0.25, 2, 1),
    )
    tournament = prepare_stage_sample_cache(
        **common,
        stage=Stage("tournament", 1.0, 1.0, 2, 1),
    )
    assert explore["validation|0.25|42"] == refine["validation|0.25|42"]
    assert refine["train|1|0"] == tournament["train|1|0"]
    assert len(list(cache_dir.glob("*.arrow"))) == 4
    manifests = [json.loads(path.read_text()) for path in cache_dir.glob("*.json")]
    assert all("fold_name" not in manifest for manifest in manifests)
    assert all("held_out_cities" not in manifest for manifest in manifests)
