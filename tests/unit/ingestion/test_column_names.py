from __future__ import annotations

import pytest
from pyspark.sql import DataFrame

from crimenet.ingestion.column_names import (
    normalize_column_name,
    normalize_column_names,
    normalized_column_names,
)
from crimenet.jobs.bronze_ingestion import COLUMN_OVERRIDES

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("source_name", "expected"),
    [
        ("Call (911) Problem", "call_911_problem"),
        ("\ufeff ZIP\xa0Code ", "zip_code"),
        ("123 field", "column_123_field"),
        ("", "unnamed_column"),
        ("Crème brûlée", "cre_me_bru_le_e"),
        ("Y Cordinate", "y_cordinate"),
    ],
)
def test_normalize_column_name(
    source_name: str,
    expected: str,
) -> None:
    assert normalize_column_name(source_name) == expected


def test_unknown_collisions_are_suffixed_deterministically() -> None:
    source_names = ["A-B", "A B", "a_b", "!!!", ""]

    assert normalized_column_names(source_names) == [
        "a_b",
        "a_b_2",
        "a_b_3",
        "unnamed_column",
        "unnamed_column_2",
    ]
    assert normalized_column_names(source_names) == (
        normalized_column_names(source_names)
    )


def test_fort_worth_coordinate_overrides_preserve_both_sources(
    fort_worth_raw: DataFrame,
) -> None:
    normalized = normalize_column_names(
        fort_worth_raw,
        overrides=COLUMN_OVERRIDES["fort_worth"],
    )

    assert "latitude" in normalized.columns
    assert "longitude" in normalized.columns
    assert "alternate_latitude" in normalized.columns
    assert "alternate_longitude" in normalized.columns
    assert len(normalized.columns) == len(set(normalized.columns))

    row = normalized.filter(
        normalized.case_no_offense == "190085144-23C"
    ).first()
    assert row is not None
    assert row.latitude == pytest.approx(32.740739801747495)
    assert row.longitude == pytest.approx(-97.26158046529801)
    assert row.alternate_latitude == pytest.approx(
        32.740745512247045
    )
    assert row.alternate_longitude == pytest.approx(
        -97.26158810460899
    )
