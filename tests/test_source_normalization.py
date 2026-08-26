from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from crimenet_data.assets.crime.ingestion import prepare_bronze_source
from crimenet_data.assets.crime.normalization import (
    normalization_type,
    normalize_sonoma_coordinates,
    normalize_source,
    normalize_unix_ms_timestamp,
)
from crimenet_data.assets.crime.silver import build_silver
from crimenet_data.assets.crime.sources import AdapterContext, get_source
from crimenet_data.resources.crime_lake import CrimeLakeResources
from crimenet_data.resources.duckdb import DuckDBResource


@pytest.mark.parametrize(
    ("source_key", "source_column"),
    [
        ("washington_dc", "occurred_at_raw"),
        ("baltimore", "crime_date_time"),
    ],
)
def test_unix_milliseconds_dispatch_preserves_timestamp_precision(
    source_key: str,
    source_column: str,
) -> None:
    lf = pl.LazyFrame(
        {
            source_column: [
                "1654289340000",
                None,
                "invalid",
                "-942710384000",
            ]
        }
    )

    result = normalize_source(lf, source_key).collect()

    timestamps = result["occurrence_timestamp"].dt.replace_time_zone("UTC")
    assert timestamps.to_list() == [
        datetime(2022, 6, 3, 20, 49, tzinfo=UTC),
        None,
        None,
        datetime(1940, 2, 17, 0, 0, 16, tzinfo=UTC),
    ]
    assert result["occurrence_year"].to_list() == [2022, None, None, 1940]
    assert normalization_type(source_key) == "unix_ms_timestamp"


def test_unix_millisecond_helper_uses_the_requested_source_column() -> None:
    result = normalize_unix_ms_timestamp(
        pl.LazyFrame({"event_ms": ["1262322085000"]}),
        "event_ms",
    ).collect()

    timestamp = result["occurrence_timestamp"].dt.replace_time_zone("UTC").item()
    assert timestamp == datetime(2010, 1, 1, 5, 1, 25, tzinfo=UTC)
    assert result["occurrence_year"].item() == 2010


def test_washington_dc_start_date_normalizes_without_unix_ms_column() -> None:
    result = normalize_source(
        pl.LazyFrame({"start_date": ["2024-01-02 03:04:05"]}),
        "washington_dc",
    ).collect()

    assert result["occurrence_timestamp"].item() == datetime(2024, 1, 2, 3, 4, 5)
    assert result["occurrence_year"].item() == 2024


def test_sonoma_location_coordinates_are_parsed_without_geocoding() -> None:
    result = normalize_sonoma_coordinates(
        pl.LazyFrame(
            {
                "location": [
                    "(38.511152, -122.781156)",
                    "( 38.511152 , -122.781156 )",
                    "\n,  \n(38.605279, -122.871845)",
                    None,
                    "",
                    "malformed",
                ]
            }
        )
    ).collect()

    assert result["latitude"].to_list() == [
        38.511152,
        38.511152,
        38.605279,
        None,
        None,
        None,
    ]
    assert result["longitude"].to_list() == [
        -122.781156,
        -122.781156,
        -122.871845,
        None,
        None,
        None,
    ]


def test_sources_without_special_handling_pass_through_unchanged() -> None:
    lf = pl.LazyFrame({"value": [1]})

    assert normalize_source(lf, "new_york") is lf
    assert normalization_type("new_york") == "none"


def test_dallas_fixture_coordinates_convert_to_wgs84_without_axis_swap() -> None:
    lake = CrimeLakeResources(bucket="/tmp/crimenet-normalization-test")
    fixture = (
        lake.get_source_fixture("dallas")
        .filter(
            pl.col("x_coordinate").is_not_null() & pl.col("y_cordinate").is_not_null()
        )
        .head(1)
    )

    with DuckDBResource(enable_spatial=True).get_connection() as connection:
        result = (
            normalize_source(
                fixture,
                "dallas",
                connection=connection,
            )
            .select("latitude", "longitude")
            .collect()
        )

    latitude = result["latitude"].item()
    longitude = result["longitude"].item()
    assert 32.61341314 <= latitude <= 33.02346757
    assert -97.00063702 <= longitude <= -96.46270666
    assert latitude > 0
    assert longitude < 0


def test_invalid_dallas_coordinates_become_null() -> None:
    lf = pl.LazyFrame(
        {
            "x_coordinate": ["invalid", "NaN", None],
            "y_cordinate": ["6972558.28815", "6972558.28815", None],
        }
    )

    with DuckDBResource(enable_spatial=True).get_connection() as connection:
        result = normalize_source(
            lf,
            "dallas",
            connection=connection,
        ).collect()

    assert result["latitude"].null_count() == 3
    assert result["longitude"].null_count() == 3


def test_bronze_preserves_duplicates_and_silver_deduplicates(tmp_path: Path) -> None:
    source = get_source("baltimore")
    rows = pl.DataFrame(
        {
            "RowID": ["duplicate", "duplicate"],
            "CrimeDateTime": ["1654289340000", "1654289340000"],
            "CrimeCode": ["4E", "4E"],
            "Description": ["THEFT", "THEFT"],
            "Latitude": ["39.29", "39.29"],
            "Longitude": ["-76.61", "-76.61"],
            "Location": ["Main", "Main"],
            "PremiseType": ["Street", "Street"],
            "New_District": ["C", "C"],
            "Neighborhood": ["N", "N"],
            "_source_file_uri": ["/landing/a", "/landing/b"],
        }
    )
    prepared = prepare_bronze_source(
        rows.lazy(),
        source,
        run_id="test-run",
        ingested_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    lake = CrimeLakeResources(
        bucket=str(tmp_path / "lake"),
        local_fixture_root=str(Path(__file__).parent / "fixtures"),
    )

    assert prepared.collect().height == 2
    silver = build_silver(
        prepared,
        lake.get_crosswalk_fixture(),
        source_key="baltimore",
        adapter_context=AdapterContext(),
    ).collect()
    assert silver.height == 1


def test_sonoma_normalized_coordinates_reach_source_projection() -> None:
    source = get_source("sonoma_county_sheriff_ca")
    raw = pl.LazyFrame(
        {
            "id": ["1"],
            "incident_number": ["I1"],
            "date_time": ["2024-01-02 03:04:00"],
            "incident_type": ["THEFT"],
            "location_type": ["Street"],
            "city": ["Sonoma"],
            "location": ["(38.511152, -122.781156)"],
            "agency": ["Sheriff"],
            "source_file_uri": ["/landing/sonoma.csv"],
            "ingestion_run_id": ["test-run"],
            "ingested_at_utc": [datetime(2026, 1, 1, tzinfo=UTC)],
        }
    )

    normalized = normalize_source(raw, "sonoma_county_sheriff_ca")
    result = source.adapt_to_silver(normalized, AdapterContext()).collect()

    assert result["latitude"].item() == pytest.approx(38.511152)
    assert result["longitude"].item() == pytest.approx(-122.781156)
