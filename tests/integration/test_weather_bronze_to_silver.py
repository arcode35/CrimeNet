from __future__ import annotations

import pytest
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

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

pytestmark = pytest.mark.integration

def test_weather_fixture_bronze_to_silver_contract(
    weather_raw: DataFrame,
) -> None:
    bronze = add_ingestion_metadata(
        normalize_column_names(weather_raw),
        source_system="open_meteo",
    )
    silver = transform_open_meteo_weather(
        bronze
    ).cache()

    try:
        assert bronze.count() == 94
        assert silver.count() == 784_368

        summary = (
            silver
            .agg(
                F.countDistinct(
                    "weather_query_cell_id"
                ).alias("cell_count"),
                F.sum(
                    F.col(
                        "temperature_2m_c"
                    )
                    .isNull()
                    .cast("long")
                ).alias("null_temperature_count"),
                F.min(
                    "temperature_2m_c"
                ).alias("minimum_temperature"),
                F.max(
                    "temperature_2m_c"
                ).alias("maximum_temperature"),
                F.sum(
                    (
                        F.col("weather_timestamp")
                        != F.date_trunc(
                            "hour",
                            F.col(
                                "weather_timestamp"
                            ),
                        )
                    ).cast("long")
                ).alias("misaligned_count"),
            )
            .first()
        )

        assert summary is not None
        assert summary["cell_count"] == 68
        assert summary[
            "null_temperature_count"
        ] == 52_608
        assert summary[
            "minimum_temperature"
        ] == -30.1
        assert summary[
            "maximum_temperature"
        ] == 43.5
        assert summary["misaligned_count"] == 0

        deduplicated = (
            deduplicate_weather_records(
                silver
            )
            .cache()
        )
        try:
            assert deduplicated.count() == 775_608
            assert (
                deduplicated
                .groupBy(*WEATHER_MERGE_KEYS)
                .count()
                .filter(F.col("count") > 1)
                .isEmpty()
            )
        finally:
            deduplicated.unpersist()
    finally:
        silver.unpersist()

