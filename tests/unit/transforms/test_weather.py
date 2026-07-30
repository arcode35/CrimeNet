from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType

from crimenet.ingestion.column_names import (
    normalize_column_names,
)
from crimenet.ingestion.metadata import (
    add_ingestion_metadata,
)
from crimenet.silver.weather import (
    WEATHER_MERGE_KEYS,
    deduplicate_weather_records,
    transform_open_meteo_weather,
)

REPRESENTATIVE_REQUEST_ID = (
    "41410836a2e5f05f11a9390f72a9ca3fd22359dbb6e2c789d936a44441307736"
)
DUPLICATE_REQUEST_ID = (
    "6fd5e7c27d6a06f5ecfbbdd2b436a1d78d3eba20ccad0efccf8fc157ad08168b"
)


def _bronze_weather(
    dataframe: DataFrame,
) -> DataFrame:
    return add_ingestion_metadata(
        normalize_column_names(dataframe),
        source_system="open_meteo",
    )


def test_weather_fixture_is_json_lines_with_nested_structs(
    weather_raw: DataFrame,
) -> None:
    assert weather_raw.count() == 94
    assert isinstance(
        weather_raw.schema["hourly"].dataType,
        StructType,
    )
    assert {
        field.name
        for field in weather_raw.schema[
            "hourly"
        ].dataType.fields
    } == {
        "temperature_2m",
        "time",
    }


def test_weather_transform_accepts_struct_and_maps_exact_values(
    weather_raw: DataFrame,
) -> None:
    response = weather_raw.filter(
        F.col("request_id")
        == REPRESENTATIVE_REQUEST_ID
    )
    transformed = transform_open_meteo_weather(
        _bronze_weather(response)
    ).cache()

    try:
        assert transformed.count() == 4800

        row = (
            transformed
            .filter(
                F.col("weather_timestamp").cast("string")
                == "2026-01-01 00:00:00"
            )
            .first()
        )

        assert row is not None
        assert row["provider"] == "open_meteo"
        assert row["model"] == "era5_land"
        assert row["weather_query_cell_id"] == 604164855133372415
        assert row["h3_resolution"] == 6
        assert row["query_latitude"] == 32.604817481
        assert row["query_longitude"] == -97.299776319
        assert row["grid_latitude"] == 32.6
        assert row["grid_longitude"] == -97.299995
        assert row["grid_elevation"] == 211.0
        assert str(row["weather_date"]) == "2026-01-01"
        assert row["temperature_2m_c"] == 15.3
        assert row["temperature_unit"] == "°C"
        assert row["timezone"] == "GMT"
        assert row["utc_offset_seconds"] == 0
    finally:
        transformed.unpersist()


def test_weather_transform_accepts_string_json(
    weather_raw: DataFrame,
) -> None:
    response = (
        weather_raw
        .filter(
            F.col("request_id")
            == REPRESENTATIVE_REQUEST_ID
        )
        .withColumn(
            "hourly",
            F.to_json("hourly"),
        )
        .withColumn(
            "hourly_units",
            F.to_json("hourly_units"),
        )
    )

    first = (
        transform_open_meteo_weather(
            _bronze_weather(response)
        )
        .orderBy("weather_timestamp")
        .select(
            F.col("weather_timestamp")
            .cast("string")
            .alias("weather_timestamp"),
            "temperature_2m_c",
        )
        .first()
    )

    assert first is not None
    assert first["weather_timestamp"] == (
        "2026-01-01 00:00:00"
    )
    assert first["temperature_2m_c"] == 15.3


def test_weather_parsing_is_safe_and_enforces_hour_grain(
    weather_raw: DataFrame,
) -> None:
    malformed = (
        weather_raw
        .filter(
            F.col("request_id")
            == REPRESENTATIVE_REQUEST_ID
        )
        .limit(1)
        .withColumn(
            "hourly",
            F.lit(
                '{"time":["2026-01-01T00:00","not-a-time",'
                '"2026-01-01T00:30"],'
                '"temperature_2m":["bad","2.0","3.0"]}'
            ),
        )
        .withColumn(
            "hourly_units",
            F.to_json("hourly_units"),
        )
    )

    rows = (
        transform_open_meteo_weather(
            _bronze_weather(malformed)
        )
        .select(
            F.col("weather_timestamp")
            .cast("string")
            .alias("weather_timestamp"),
            "temperature_2m_c",
        )
        .collect()
    )

    assert len(rows) == 1
    assert rows[0]["weather_timestamp"] == (
        "2026-01-01 00:00:00"
    )
    assert rows[0]["temperature_2m_c"] is None


def test_weather_fixture_duplicate_is_deduplicated_deterministically(
    weather_raw: DataFrame,
) -> None:
    duplicated_response = weather_raw.filter(
        F.col("request_id")
        == DUPLICATE_REQUEST_ID
    )
    transformed = transform_open_meteo_weather(
        _bronze_weather(duplicated_response)
    ).cache()

    try:
        assert transformed.count() == 17_520

        deduplicated = deduplicate_weather_records(
            transformed
        )
        repartitioned = deduplicate_weather_records(
            transformed.repartition(5)
        )

        assert deduplicated.count() == 8_760
        assert (
            deduplicated
            .select(*WEATHER_MERGE_KEYS)
            .exceptAll(
                repartitioned.select(
                    *WEATHER_MERGE_KEYS
                )
            )
            .isEmpty()
        )
        assert (
            repartitioned
            .select(*WEATHER_MERGE_KEYS)
            .exceptAll(
                deduplicated.select(
                    *WEATHER_MERGE_KEYS
                )
            )
            .isEmpty()
        )
    finally:
        transformed.unpersist()

