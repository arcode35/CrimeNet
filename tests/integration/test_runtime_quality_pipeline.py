from __future__ import annotations

import pytest
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from crimenet.ingestion.column_names import normalize_column_names
from crimenet.ingestion.metadata import add_ingestion_metadata
from crimenet.quality import (
    validate_silver_crime,
    validate_socioeconomic,
    validate_weather,
)
from crimenet.silver.socioeconomic import (
    deduplicate_socioeconomic_records,
    transform_acs5_tracts,
)
from crimenet.silver.weather import (
    deduplicate_weather_records,
    transform_open_meteo_weather,
)

pytestmark = pytest.mark.integration

_WEATHER_REQUEST_ID = (
    "c52a2df6ddadcaf4376426a0bf6bc2a03"
    "fa24d08c4b1166adbfa4eee274c220a"
)


def test_fixture_materializations_pass_structured_runtime_quality(
    deduplicated_crimes: DataFrame,
    socioeconomic_bronze: DataFrame,
    weather_raw: DataFrame,
) -> None:
    crime = deduplicated_crimes.localCheckpoint(eager=True)
    crime_report = validate_silver_crime(
        crime,
        minimum_occurred_at_coverage=0.95,
    )

    socioeconomic = deduplicate_socioeconomic_records(
        transform_acs5_tracts(
            socioeconomic_bronze
        ).localCheckpoint(eager=True)
    ).localCheckpoint(eager=True)
    socioeconomic_report = validate_socioeconomic(socioeconomic)

    weather_bronze = add_ingestion_metadata(
        normalize_column_names(
            weather_raw.filter(
                F.col("request_id") == _WEATHER_REQUEST_ID
            )
        ),
        source_system="open_meteo",
    ).localCheckpoint(eager=True)
    weather = deduplicate_weather_records(
        transform_open_meteo_weather(weather_bronze)
    ).localCheckpoint(eager=True)
    weather_report = validate_weather(weather)

    assert crime_report.passed
    assert socioeconomic_report.passed
    assert weather_report.passed
    assert {
        check.check_name for check in crime_report.checks
    } >= {
        "canonical_schema",
        "unique_crime_offense_id",
        "occurred_at_coverage",
    }
