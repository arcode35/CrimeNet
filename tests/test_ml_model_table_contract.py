from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from machine_learning.data.cache import cache_identity
from machine_learning.data.features import (
    DEFAULT_TRANSFERABLE_CATEGORICAL,
    DEFAULT_TRANSFERABLE_NUMERIC,
    LOCAL_HISTORY_ABLATION,
    resolve_feature_contract,
)
from machine_learning.data.geography import geographic_frames
from machine_learning.data.metrics import geographic_point_process_metrics
from machine_learning.data.model_table import (
    ResolvedModelTable,
    resolve_model_table,
    resolve_model_table_from_config,
)
from machine_learning.data.point_process import prepare_target_exposure
from machine_learning.models.xgboost.model import (
    _point_process_eval_values,
    train as train_intensity,
)
from machine_learning.experiments.xgb_hpo import (
    Stage,
    build_space,
    enqueue_trial_params,
    objective_metric,
    prepared_xy_cache_enabled,
)
from crimenet_data.resources.crime_lake import CrimeLakeResources


def _rows(split: str, city: str) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "model_row_id": [f"{split}-{city}-event", f"{split}-{city}-integration"],
            "row_type": ["event", "integration"],
            "event_indicator": [1, 0],
            "is_observed_event": [True, False],
            "event_count": [1, 0],
            "integration_weight_cell_seconds": [None, 10.0],
            "source_city": [city, city],
            "feature": [1.0, 2.0],
            "lighting_condition": ["daylight", "night"],
            "row_year": [1999, 2035],
        },
        schema_overrides={"integration_weight_cell_seconds": pl.Float64},
    )


@pytest.fixture
def local_snapshot(tmp_path: Path) -> Path:
    root = tmp_path / "staged-final-model"
    for split, city in (("train", "alpha"), ("validation", "beta"), ("test", "sealed")):
        directory = root / f"split={split}" / f"source_city={city}"
        directory.mkdir(parents=True)
        _rows(split, city).drop("source_city").write_parquet(directory / "part.parquet")
    manifest = {
        "snapshot_id": "fixture-001",
        "snapshot_uri": "s3://canonical/gold/final_model_table/snapshot_id=fixture-001",
        "schema_version": "final_model_table_v1",
        "columns": [
            "model_row_id",
            "row_type",
            "event_indicator",
            "is_observed_event",
            "event_count",
            "integration_weight_cell_seconds",
            "source_city",
            "split",
            "feature",
            "lighting_condition",
            "row_year",
        ],
        "event_spine_snapshot_id": "events-1",
        "integration_snapshot_id": "integration-1",
        "environmental_snapshot_id": "environment-1",
        "temporal_history_snapshot_id": "history-1",
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def test_local_parquet_override_prunes_split_and_seals_test(local_snapshot: Path) -> None:
    table = resolve_model_table(local_root=str(local_snapshot))
    train = table.scan_split("train").collect()
    validation = table.scan_split("validation").collect()
    assert set(train["source_city"]) == {"alpha"}
    assert set(validation["source_city"]) == {"beta"}
    assert set(train["snapshot_id"]) == {"fixture-001"}
    with pytest.raises(ValueError, match="sealed"):
        table.scan_split("test")


def test_current_snapshot_is_resolved_once() -> None:
    class Lake:
        calls = 0

        def resolve_final_model_table_snapshot(self, *, snapshot_override_uri=None):
            self.calls += 1
            assert snapshot_override_uri is None
            return "s3://bucket/gold/final_model_table/snapshot_id=one", {
                "snapshot_id": "one",
                "snapshot_uri": "s3://bucket/gold/final_model_table/snapshot_id=one",
                "schema_version": "v1",
                "columns": [
                    "model_row_id", "row_type", "event_indicator",
                    "is_observed_event", "event_count",
                    "integration_weight_cell_seconds", "source_city", "split",
                ],
            }

    lake = Lake()
    resolved = resolve_model_table(lake=lake)  # type: ignore[arg-type]
    assert resolved.snapshot_id == "one"
    assert lake.calls == 1


def test_configured_snapshot_uri_and_id_are_both_enforced() -> None:
    uri = "s3://bucket/gold/final_model_table/snapshot_id=pinned"

    class Lake:
        seen_override = None

        def resolve_final_model_table_snapshot(self, *, snapshot_override_uri=None):
            self.seen_override = snapshot_override_uri
            return uri, {
                "snapshot_id": "pinned",
                "snapshot_uri": uri,
                "schema_version": "v1",
                "columns": [
                    "model_row_id", "row_type", "event_indicator",
                    "is_observed_event", "event_count",
                    "integration_weight_cell_seconds", "source_city", "split",
                ],
            }

    lake = Lake()
    resolved = resolve_model_table_from_config(
        {
            "final_model_snapshot_uri": uri,
            "final_model_snapshot_id": "pinned",
        },
        lake=lake,  # type: ignore[arg-type]
    )
    assert resolved.snapshot_id == "pinned"
    assert lake.seen_override == uri
    with pytest.raises(RuntimeError, match="differs from pinned config"):
        resolve_model_table_from_config(
            {
                "final_model_snapshot_uri": uri,
                "final_model_snapshot_id": "other",
            },
            lake=Lake(),  # type: ignore[arg-type]
        )


def test_explicit_canonical_snapshot_override_is_identity_checked(tmp_path: Path) -> None:
    lake = CrimeLakeResources(bucket=str(tmp_path / "lake"))
    snapshot_uri = lake.final_model_table_snapshot_uri("fixed")
    partition = Path(snapshot_uri) / "split=train" / "source_city=alpha"
    partition.mkdir(parents=True)
    _rows("train", "alpha").drop("source_city").write_parquet(
        partition / "part.parquet"
    )
    Path(lake.snapshot_success_uri(snapshot_uri)).write_bytes(b"")
    Path(lake.snapshot_manifest_uri(snapshot_uri)).write_text(
        json.dumps({"snapshot_id": "fixed", "snapshot_uri": snapshot_uri})
    )
    resolved_uri, manifest = lake.resolve_final_model_table_snapshot(
        snapshot_override_uri=snapshot_uri
    )
    assert resolved_uri == snapshot_uri
    assert manifest["snapshot_id"] == "fixed"
    with pytest.raises(ValueError, match="outside the canonical root"):
        lake.resolve_final_model_table_snapshot(
            snapshot_override_uri=str(tmp_path / "latest")
        )


def test_b2_scan_uses_storage_options_and_partition_glob(monkeypatch) -> None:
    calls: dict[str, object] = {}
    fixture = _rows("train", "alpha").lazy().with_columns(
        pl.lit("train").alias("split")
    )

    class Lake:
        def storage_options_for(self, uri: str):
            calls["storage_uri"] = uri
            return {"endpoint_url": "https://s3.example.invalid"}

    def fake_scan(path: str, **kwargs):
        calls["path"] = path
        calls["kwargs"] = kwargs
        return fixture

    monkeypatch.setattr(pl, "scan_parquet", fake_scan)
    table = ResolvedModelTable(
        snapshot_id="s1",
        snapshot_uri="s3://bucket/gold/final_model_table/snapshot_id=s1",
        schema_version="v1",
        manifest={},
        lake=Lake(),  # type: ignore[arg-type]
    )
    table.scan_split("train")
    assert calls["path"] == (
        "s3://bucket/gold/final_model_table/snapshot_id=s1/"
        "split=train/source_city=*/*.parquet"
    )
    assert calls["kwargs"]["storage_options"]["endpoint_url"].startswith("https://")


@pytest.mark.parametrize(
    "forbidden",
    [
        "source_city", "osm_h3_cell_id", "latitude", "longitude",
        "event_count", "integration_weight_cell_seconds", "canonical_subtype_code",
        "future_provenance_id",
    ],
)
def test_feature_contract_rejects_leaky_predictors(forbidden: str) -> None:
    with pytest.raises(ValueError, match="Forbidden"):
        resolve_feature_contract(
            {"feature_set": "test", "numeric": [forbidden], "categorical": []},
            available_columns=[forbidden],
        )


def test_feature_contract_order_hash_duplicates_and_missing() -> None:
    config = {"feature_set": "test", "numeric": ["b", "a"], "categorical": ["c"]}
    one = resolve_feature_contract(config, available_columns=["a", "b", "c"])
    two = resolve_feature_contract(config, available_columns=["c", "b", "a"])
    assert one.all_features == ("b", "a", "c")
    assert one.contract_hash == two.contract_hash
    with pytest.raises(ValueError, match="duplicates"):
        resolve_feature_contract(
            {"numeric": ["a", "a"], "categorical": []}, available_columns=["a"]
        )
    with pytest.raises(ValueError, match="missing"):
        resolve_feature_contract(
            {"numeric": ["absent"], "categorical": []}, available_columns=[]
        )


def test_transferable_defaults_are_zero_shot_safe_and_guard_is_centralized() -> None:
    available = [*DEFAULT_TRANSFERABLE_NUMERIC, *DEFAULT_TRANSFERABLE_CATEGORICAL]
    contract = resolve_feature_contract(
        {"feature_set": "transferable_v2", "zero_shot_geography": True},
        available_columns=available,
    )
    assert not (set(contract.numeric) & set(LOCAL_HISTORY_ABLATION))
    with pytest.raises(ValueError, match="Zero-shot feature contract violation"):
        resolve_feature_contract(
            {
                "feature_set": "unsafe",
                "zero_shot_geography": True,
                "numeric": ["cell_crime_count_24h"],
                "categorical": [],
            },
            available_columns=["cell_crime_count_24h"],
        )


def test_event_null_exposure_becomes_zero_and_integration_is_preserved() -> None:
    y, exposure = prepare_target_exposure(_rows("train", "alpha"))
    np.testing.assert_array_equal(y, [1.0, 0.0])
    np.testing.assert_array_equal(exposure, [0.0, 10.0])


@pytest.mark.parametrize("event_weight,integration_weight", [(1.0, 10.0), (None, 0.0), (None, -1.0), (None, float("nan"))])
def test_invalid_exposure_contract_fails(event_weight, integration_weight) -> None:
    frame = _rows("train", "alpha").with_columns(
        pl.Series("integration_weight_cell_seconds", [event_weight, integration_weight])
    )
    with pytest.raises(ValueError):
        prepare_target_exposure(frame)


def test_geographic_holdout_precedes_sampling_and_keeps_in_domain() -> None:
    train = pl.concat([_rows("train", "alpha"), _rows("train", "beta")]).lazy()
    validation = pl.concat(
        [_rows("validation", "alpha"), _rows("validation", "beta")]
    ).lazy()
    training, geographic, in_domain = geographic_frames(
        train=train,
        validation=validation,
        holdout_cities=["beta"],
        report_in_domain=True,
    )
    assert set(training.collect()["source_city"]) == {"alpha"}
    assert set(geographic.collect()["source_city"]) == {"beta"}
    assert in_domain is not None
    assert set(in_domain.collect()["source_city"]) == {"alpha"}


def _identity(**changes) -> str:
    args = {
        "cache_version": "3",
        "snapshot_id": "snapshot-a",
        "schema_version": "schema-a",
        "feature_contract_hash": "features-a",
        "model_family": "intensity",
        "model_module": "module",
        "split": "train",
        "fraction": 0.5,
        "seed": 42,
    }
    args.update(changes)
    return cache_identity(**args)


def test_cache_identity_uses_immutable_contract() -> None:
    assert _identity() == _identity()
    assert _identity() != _identity(snapshot_id="snapshot-b")
    assert _identity() != _identity(feature_contract_hash="features-b")
    assert _identity() != _identity(target_column="other_target")
    with pytest.raises(ValueError, match="Test split"):
        _identity(split="test")


def test_transfer_hpo_uses_macro_city_objective_and_reset_depth_space() -> None:
    config = {"architecture": {"max_bin": 256, "max_cat_to_onehot": 4}}
    space = build_space(config, "intensity")
    assert (space["depth_low"], space["depth_high"]) == (4, 12)
    assert space["max_bin_choices"] == [128, 256, 512]
    assert objective_metric("intensity") == "geocv_macro_nll_per_event"


def test_intensity_gamma_is_preserved_when_enqueueing_refinement_seed() -> None:
    class Study:
        enqueued = None

        def enqueue_trial(self, params):
            self.enqueued = params

    study = Study()
    enqueue_trial_params(
        study,  # type: ignore[arg-type]
        {"max_depth": 8, "reg_alpha": 0.0, "gamma": 2.75},
        family="intensity",
    )
    assert study.enqueued["use_gamma"] is True
    assert study.enqueued["gamma_nonzero"] == pytest.approx(2.75)
    assert "gamma" not in study.enqueued


def test_prepared_xy_cache_is_disabled_if_either_stage_fraction_is_full() -> None:
    assert prepared_xy_cache_enabled(
        requested=True, stage=Stage("explore", 0.25, 0.25, 10, 2)
    )
    assert not prepared_xy_cache_enabled(
        requested=True, stage=Stage("refine", 1.0, 0.25, 10, 2)
    )
    assert not prepared_xy_cache_enabled(
        requested=True, stage=Stage("tournament", 1.0, 1.0, 10, 2)
    )


def test_macro_city_early_stopping_metric_is_equal_city_not_pooled() -> None:
    pooled, macro = _point_process_eval_values(
        y=np.asarray([1.0, 0.0, 1.0, 1.0, 0.0]),
        exposure=np.asarray([0.0, 1.0, 0.0, 0.0, 10.0]),
        margin=np.zeros(5),
        city_codes=np.asarray([0, 0, 1, 1, 1]),
        city_count=2,
        min_log_intensity=-30.0,
        max_log_intensity=15.0,
    )
    assert pooled == pytest.approx(11.0 / 3.0)
    assert macro == pytest.approx((1.0 + 5.0) / 2.0)
    assert macro != pytest.approx(pooled)


def test_geographic_metrics_are_unweighted_city_means() -> None:
    frame = pl.concat(
        [
            _rows("validation", "alpha"),
            _rows("validation", "beta").with_columns(
                pl.col("integration_weight_cell_seconds").fill_null(0.0) * 2
            ),
        ]
    )
    report = geographic_point_process_metrics(
        frame,
        log_intensity=np.log(np.array([0.1, 0.1, 0.2, 0.2])),
        constant_log_intensity=np.log(0.1),
    )
    city_values = [row["nll_per_event"] for row in report["per_city"]]
    assert report["macro_city"]["mean_nll_per_event"] == pytest.approx(
        float(np.mean(city_values))
    )
    assert report["global"]["rows"] == 4


def test_tiny_intensity_training_smoke_never_reads_test(
    local_snapshot: Path, tmp_path: Path
) -> None:
    config = {
        "model": {"name": "tiny", "family": "xgboost"},
        "data": {
            "local_snapshot_root": str(local_snapshot),
            "train_split": "train",
            "validation_split": "validation",
            "train_fraction": 1.0,
            "validation_fraction": 1.0,
            "seed": 42,
        },
        "features": {
            "feature_set": "synthetic",
            "numeric": ["feature"],
            "categorical": ["lighting_condition"],
        },
        "architecture": {
            "tree_method": "hist",
            "device": "cpu",
            "max_bin": 16,
            "max_depth": 2,
            "max_cat_to_onehot": 4,
        },
        "optimization": {
            "learning_rate": 0.1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "min_child_weight": 0.0,
            "max_delta_step": 1.0,
            "gamma": 0.0,
            "reg_lambda": 1.0,
            "reg_alpha": 0.0,
        },
        "training": {
            "num_boost_round": 2,
            "early_stopping_rounds": 1,
            "verbose_eval": False,
        },
        "validation": {
            "geographic_holdout_cities": ["beta"],
            "report_in_domain_validation": False,
        },
        "numerics": {
            "min_log_intensity": -30.0,
            "max_log_intensity": 15.0,
            "hessian_floor": 1e-6,
            "event_exposure_tolerance": 1e-12,
        },
        "artifacts": {"output_root": str(tmp_path / "artifacts")},
    }
    result = train_intensity(config, run_id="run", config_hash="hash")
    assert result["summary"]["test_split_used"] is False
    assert result["metrics"]["sample_train_rows"] == 2
    assert result["metrics"]["sample_validation_rows"] == 2
    assert result["metrics"]["geographic_macro_nll_per_event"] == pytest.approx(
        result["metrics"]["sample_validation_nll_per_event"]
    )
    assert "macro_city_pp_nll_per_event" in result["history"]["validation"]
    assert (
        tmp_path / "artifacts" / "tiny" / "run" / "geographic_validation_by_city.csv"
    ).is_file()
