from __future__ import annotations

import pytest

from crimenet.config.resources import CrimeNetTables
from crimenet.config.validation import (
    QualityThresholds,
    normalize_pipeline_run_id,
    validate_identifier,
    validate_qualified_table_name,
)


def test_validates_safe_unity_catalog_identifiers() -> None:
    assert validate_identifier("crime_net_2026") == "crime_net_2026"
    assert (
        validate_qualified_table_name("crime_net.silver.crime_offenses")
        == "crime_net.silver.crime_offenses"
    )


@pytest.mark.parametrize(
    "value",
    ["", "two-parts.table", "catalog.schema.table.extra", "a.b.drop table"],
)
def test_rejects_unsafe_table_names(value: str) -> None:
    with pytest.raises(ValueError):
        validate_qualified_table_name(value)


@pytest.mark.parametrize(
    "component_name",
    [
        "catalog",
        "bronze_schema",
        "silver_schema",
        "gold_schema",
        "operations_schema",
        "data_quality_schema",
    ],
)
def test_crimenet_tables_rejects_unsafe_components(
    component_name: str,
) -> None:
    arguments = {"catalog": "crime"}
    arguments[component_name] = "unsafe-name"

    with pytest.raises(ValueError, match=component_name):
        CrimeNetTables(**arguments)


def test_pipeline_run_id_is_safe_and_bounded() -> None:
    assert normalize_pipeline_run_id("job/42 retry 1") == "job_42_retry_1"
    assert len(normalize_pipeline_run_id("a" * 200)) == 96


def test_quality_thresholds_reject_invalid_rates() -> None:
    with pytest.raises(ValueError, match="maximum_quarantine_rate"):
        QualityThresholds(maximum_quarantine_rate=1.1).validate()
    with pytest.raises(ValueError, match="maximum_row_count"):
        QualityThresholds(
            minimum_row_count=10,
            maximum_row_count=9,
        ).validate()
