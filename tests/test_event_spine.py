from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from crimenet_data.assets.crime.canonical import CANONICAL_MAPPING_VERSION
from crimenet_data.assets.event_spine.build import (
    attach_selected_features,
    build_event_spine,
    localize_occurrence_times,
    prepare_event_index,
    select_temporal_matches,
)
from crimenet_data.assets.event_spine.gold import event_spine_gold_assets
from crimenet_data.assets.event_spine.publishing import (
    event_spine_root,
    event_spine_snapshot_uri,
    publish_event_spine_snapshot,
)
from crimenet_data.assets.event_spine.schema import (
    EVENT_SPINE_LATEST_POINTER,
    EVENT_SPINE_SCHEMA_VERSION,
)
from crimenet_data.assets.event_spine.temporal import (
    history_root,
    load_selected_feature_rows,
    load_temporal_index,
    prune_history_to_h3,
    selected_history_keys,
    validate_temporal_history,
)
from crimenet_data.assets.event_spine.validation import validate_event_spine
from crimenet_data.definitions import defs
from crimenet_data.resources.crime_lake import CrimeLakeResources


def _lake(tmp_path: Path) -> CrimeLakeResources:
    return CrimeLakeResources(bucket=str(tmp_path / "object-store"))


def _history(*, duplicate: bool = False) -> pl.DataFrame:
    available = [
        datetime(2023, 12, 1, tzinfo=UTC),
        datetime(2024, 1, 2, tzinfo=UTC),
    ]
    if duplicate:
        available[1] = available[0]
    return pl.DataFrame(
        {
            "osm_h3_cell_id": [10, 10],
            "feature_available_at": available,
            "feature_version_id": ["v1", "v2"],
            "osm_available_at": [
                datetime(2023, 11, 30, tzinfo=UTC),
                datetime(2024, 1, 1, tzinfo=UTC),
            ],
            "feature_value": [1.0, 2.0],
        }
    )


def _publication_history_summary() -> dict[str, object]:
    summary = validate_temporal_history(_history())
    summary.update(
        {
            "history_scope": "modeled_event_h3_footprint",
            "unique_relevant_h3_cells": 1,
            "skinny_history_columns": [
                "osm_h3_cell_id",
                "feature_available_at",
                "feature_version_id",
                "osm_available_at",
            ],
            "skinny_history_column_count": 4,
            "filtered_skinny_history_rows": 2,
            "filtered_history_h3_cells": 1,
            "unique_selected_history_keys": 2,
            "full_feature_rows_retrieved": 2,
            "full_feature_column_count": len(_history().columns),
        }
    )
    return summary


def _write_history(lake: CrimeLakeResources, history: pl.DataFrame) -> None:
    path = (
        Path(history_root(lake))
        / "feature_available_date=2024-01-01"
        / "version_id=test"
        / "part-0.parquet"
    )
    path.parent.mkdir(parents=True)
    history.write_parquet(path)


def _events() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "crime_id": ["city:a", "city:b"],
            "source_city": ["city", "city"],
            "occurrence_year": pl.Series([2024, 2024], dtype=pl.Int16),
            "occurrence_timestamp_utc": [
                datetime(2024, 1, 1, 12, tzinfo=UTC),
                datetime(2024, 1, 3, 12, tzinfo=UTC),
            ],
            "osm_h3_cell_id": [10, 10],
        }
    )


def _temporal_edge_history() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "osm_h3_cell_id": [10, 10, 10, 20, 99],
            "feature_available_at": [
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 3, tzinfo=UTC),
                datetime(2024, 1, 5, tzinfo=UTC),
                datetime(2024, 1, 4, tzinfo=UTC),
                datetime(2024, 1, 1, tzinfo=UTC),
            ],
            "feature_version_id": ["v1", "v2", "v3", "v20", "irrelevant"],
            "osm_available_at": [
                datetime(2023, 12, 31, tzinfo=UTC),
                datetime(2024, 1, 2, tzinfo=UTC),
                datetime(2024, 1, 4, tzinfo=UTC),
                datetime(2024, 1, 3, tzinfo=UTC),
                datetime(2023, 12, 31, tzinfo=UTC),
            ],
            "feature_value": [1.0, 2.0, 3.0, 20.0, 99.0],
        }
    )


def _temporal_edge_events() -> pl.DataFrame:
    event_specs = [
        ("latest", 10, datetime(2024, 1, 4, tzinfo=UTC)),
        ("exact", 10, datetime(2024, 1, 3, tzinfo=UTC)),
        ("future", 10, datetime(2024, 1, 2, tzinfo=UTC)),
        ("before", 10, datetime(2023, 1, 1, tzinfo=UTC)),
        ("other-before", 20, datetime(2024, 1, 2, tzinfo=UTC)),
        ("other-match", 20, datetime(2024, 1, 5, tzinfo=UTC)),
    ]
    return pl.DataFrame(
        {
            "crime_id": [f"city:{name}" for name, _, _ in event_specs],
            "source_city": ["city"] * len(event_specs),
            "occurrence_year": pl.Series(
                [timestamp.year for _, _, timestamp in event_specs],
                dtype=pl.Int16,
            ),
            "occurrence_timestamp_utc": [timestamp for _, _, timestamp in event_specs],
            "osm_h3_cell_id": [cell for _, cell, _ in event_specs],
        }
    )


def _built_spine() -> tuple[pl.DataFrame, dict[str, object]]:
    spine, build_summary = build_event_spine(
        events=_events(),
        history=_history(),
    )
    return spine, validate_event_spine(spine, build_summary)


def test_history_root_is_exact_production_history_not_annual() -> None:
    root = history_root(CrimeLakeResources())

    assert root == (
        "s3://crimenet-data/gold/national_feature_store/temporal/h3_r9/history"
    )
    assert "/annual" not in root


def test_localize_occurrence_times_uses_deterministic_dst_policy() -> None:
    localized = localize_occurrence_times(
        pl.DataFrame(
            {
                "crime_id": ["ambiguous", "nonexistent"],
                "source_timezone": ["America/New_York", "America/New_York"],
                "occurrence_timestamp": [
                    datetime(2024, 11, 3, 1, 30),  # noqa: DTZ001 - local wall time
                    datetime(2024, 3, 10, 2, 30),  # noqa: DTZ001 - local wall time
                ],
            }
        )
    ).sort("crime_id")

    assert localized["occurrence_timestamp_utc"].to_list() == [
        datetime(2024, 11, 3, 5, 30, tzinfo=UTC),
        None,
    ]

    indexed = localized.with_columns(pl.lit(10).alias("osm_h3_cell_id"))
    event_index, _, summary = prepare_event_index(indexed)
    assert event_index["crime_id"].to_list() == ["ambiguous"]
    assert summary["invalid_event_utc_rows"] == 1


def test_temporal_history_rejects_duplicate_logical_keys() -> None:
    with pytest.raises(RuntimeError, match="violates unique"):
        validate_temporal_history(_history(duplicate=True))


def test_temporal_history_rejects_component_availability_leakage() -> None:
    history = _history().with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(pl.lit(datetime(2024, 1, 1, tzinfo=UTC)))
        .otherwise(pl.col("osm_available_at"))
        .alias("osm_available_at")
    )

    with pytest.raises(RuntimeError, match="component availability leakage"):
        validate_temporal_history(history)


def test_h3_pruning_projects_skinny_history_before_collection() -> None:
    pruned = prune_history_to_h3(
        _temporal_edge_history().lazy(),
        relevant_h3_cells=pl.DataFrame({"osm_h3_cell_id": [10]}),
        columns=[
            "osm_h3_cell_id",
            "feature_available_at",
            "feature_version_id",
            "osm_available_at",
        ],
    ).collect(engine="streaming")

    assert pruned.height == 3
    assert pruned["osm_h3_cell_id"].unique().to_list() == [10]
    assert "feature_value" not in pruned.columns


def test_skinny_asof_preserves_exact_temporal_and_h3_semantics() -> None:
    events = _temporal_edge_events()
    event_index, relevant_cells, event_summary = prepare_event_index(events)
    temporal_index = prune_history_to_h3(
        _temporal_edge_history().lazy(),
        relevant_h3_cells=relevant_cells,
        columns=[
            "osm_h3_cell_id",
            "feature_available_at",
            "feature_version_id",
            "osm_available_at",
        ],
    ).collect()
    matched, summary = select_temporal_matches(
        event_index=event_index,
        temporal_index=temporal_index,
        event_summary=event_summary,
    )
    selected = {
        row["crime_id"]: row["feature_available_at"]
        for row in matched.iter_rows(named=True)
    }

    assert selected == {
        "city:latest": datetime(2024, 1, 3, tzinfo=UTC),
        "city:exact": datetime(2024, 1, 3, tzinfo=UTC),
        "city:future": datetime(2024, 1, 1, tzinfo=UTC),
        "city:other-match": datetime(2024, 1, 4, tzinfo=UTC),
    }
    assert "city:before" not in selected
    assert "city:other-before" not in selected
    assert summary["no_legal_history_match_rows"] == 2
    assert summary["selected_temporal_match_rows"] == 4


def test_full_feature_reattachment_preserves_key_and_event_grain() -> None:
    events = _temporal_edge_events()
    event_index, relevant_cells, event_summary = prepare_event_index(events)
    history = _temporal_edge_history()
    temporal_index = prune_history_to_h3(
        history.lazy(),
        relevant_h3_cells=relevant_cells,
        columns=[
            "osm_h3_cell_id",
            "feature_available_at",
            "feature_version_id",
            "osm_available_at",
        ],
    ).collect()
    matched, _ = select_temporal_matches(
        event_index=event_index,
        temporal_index=temporal_index,
        event_summary=event_summary,
    )
    keys = selected_history_keys(matched)
    full_features = history.join(keys, on=keys.columns, how="semi")
    spine = attach_selected_features(
        events=events,
        matched_event_keys=matched,
        full_feature_rows=full_features,
    ).sort("crime_id")

    assert spine.height == matched.height
    assert spine["crime_id"].n_unique() == spine.height
    assert "feature_value" in spine.columns
    assert dict(spine.select("crime_id", "feature_value").iter_rows()) == {
        "city:exact": 2.0,
        "city:future": 1.0,
        "city:latest": 2.0,
        "city:other-match": 20.0,
    }


def test_two_stage_history_loaders_prune_and_retrieve_exact_rows(
    tmp_path: Path,
) -> None:
    lake = _lake(tmp_path)
    _write_history(lake, _temporal_edge_history())
    relevant = pl.DataFrame({"osm_h3_cell_id": [10]})

    temporal_index, history_summary = load_temporal_index(
        lake,
        relevant_h3_cells=relevant,
    )
    assert temporal_index.height == 3
    assert "feature_value" not in temporal_index.columns
    assert history_summary["filtered_skinny_history_rows"] == 3
    assert history_summary["unique_relevant_h3_cells"] == 1

    selected = temporal_index.filter(pl.col("feature_version_id") == "v2").select(
        "osm_h3_cell_id", "feature_available_at"
    )
    full_features, retrieval_summary = load_selected_feature_rows(
        lake,
        relevant_h3_cells=relevant,
        selected_keys=selected,
    )
    assert full_features.height == 1
    assert full_features["feature_value"].to_list() == [2.0]
    assert retrieval_summary["unique_selected_history_keys"] == 1
    assert retrieval_summary["full_feature_rows_retrieved"] == 1


def test_backward_asof_selects_latest_legal_feature_without_multiplication() -> None:
    spine, summary = _built_spine()

    assert spine.sort("crime_id")["feature_version_id"].to_list() == ["v1", "v2"]
    assert summary["row_count"] == 2
    assert summary["unique_crime_ids"] == 2
    assert summary["future_feature_leaks"] == 0
    assert summary["coverage_pct"] == 100.0


def test_event_spine_validation_rejects_future_features() -> None:
    spine, build_summary = build_event_spine(
        events=_events(),
        history=_history(),
    )
    invalid = spine.with_columns(
        (pl.col("occurrence_timestamp_utc") + pl.duration(days=1)).alias(
            "feature_available_at"
        )
    )

    with pytest.raises(RuntimeError, match="future_feature_leaks"):
        validate_event_spine(invalid, build_summary)


def test_event_spine_snapshot_publication_and_pointer_order(tmp_path: Path) -> None:
    lake = _lake(tmp_path)
    spine, join_summary = _built_spine()
    manifest = publish_event_spine_snapshot(
        crime_lake=lake,
        spine=spine,
        snapshot_id="gold-one",
        created_at_utc=datetime(2026, 8, 25, 12, tzinfo=UTC),
        silver_snapshot_uri="s3://silver/snapshot_id=silver-one",
        silver_manifest={
            "snapshot_id": "silver-one",
            "mapping_version": CANONICAL_MAPPING_VERSION,
            "schema_version": "crime_silver_v1",
        },
        join_summary=join_summary,
        history_summary=_publication_history_summary(),
    )

    snapshot = Path(event_spine_snapshot_uri(lake, "gold-one"))
    pointer_path = Path(event_spine_root(lake), EVENT_SPINE_LATEST_POINTER)
    assert (snapshot / "_SUCCESS").is_file()
    assert (snapshot / "manifest.json").is_file()
    assert (snapshot / "source_city=city" / "occurrence_year=2024").is_dir()
    assert not (snapshot / "source_city%3Dcity").exists()
    assert manifest["schema_version"] == EVENT_SPINE_SCHEMA_VERSION
    assert manifest["row_count"] == 2
    assert manifest["silver_snapshot_id"] == "silver-one"
    assert manifest["history_root"].endswith(
        "/gold/national_feature_store/temporal/h3_r9/history"
    )
    assert json.loads(pointer_path.read_text())["snapshot_id"] == "gold-one"

    invalid = pl.concat([spine, spine.head(1)])
    with pytest.raises(RuntimeError, match="post-write quality gate"):
        publish_event_spine_snapshot(
            crime_lake=lake,
            spine=invalid,
            snapshot_id="gold-invalid",
            created_at_utc=datetime(2026, 8, 25, 13, tzinfo=UTC),
            silver_snapshot_uri="s3://silver/snapshot_id=silver-one",
            silver_manifest={"snapshot_id": "silver-one"},
            join_summary=join_summary,
            history_summary=_publication_history_summary(),
        )
    assert json.loads(pointer_path.read_text())["snapshot_id"] == "gold-one"
    assert not Path(event_spine_snapshot_uri(lake, "gold-invalid"), "_SUCCESS").exists()


def test_gold_event_spine_is_registered_in_definitions() -> None:
    keys = {
        key.to_user_string() for key in defs.resolve_asset_graph().get_all_asset_keys()
    }

    assert event_spine_gold_assets[0].key.to_user_string() == "gold_event_spine"
    assert "gold_event_spine" in keys
