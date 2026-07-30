from __future__ import annotations

import pytest
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

pytestmark = pytest.mark.unit


def _only_source_file(dataframe: DataFrame) -> str:
    files = [
        row["_source_file"]
        for row in dataframe.select("_source_file").distinct().collect()
    ]
    assert len(files) == 1
    source_file = files[0]
    assert isinstance(source_file, str)
    return source_file


def test_dallas_reader_loads_multiline_csv_fixture(
    dallas_raw: DataFrame,
) -> None:
    assert dallas_raw.count() == 119
    assert len(dallas_raw.columns) == 87
    assert {
        "Incident Number w/year",
        "Service Number ID",
        "Date1 of Occurrence",
        "Time1 of Occurrence",
        "NIBRS Code",
        "Y Cordinate",
        "Location1",
        "_source_file",
    }.issubset(dallas_raw.columns)

    row = (
        dallas_raw
        .filter(F.col("Service Number ID") == "095681-2022-01")
        .select(
            F.col("Incident Number w/year").alias("incident"),
            F.col("Date1 of Occurrence").alias("occurred_date"),
            F.col("Time1 of Occurrence").alias("occurred_time"),
            F.col("Location1").alias("location"),
        )
        .first()
    )
    assert row is not None
    assert row.incident == "095681-2022"
    assert row.occurred_date == "2022-05-29 00:00:00.0000000"
    assert row.occurred_time == "05:11"
    assert row.location.endswith("(32.877, -96.75812)")
    assert _only_source_file(dallas_raw).endswith(
        "dallas_fixture.csv"
    )


def test_houston_reader_loads_csv_fixture(
    houston_raw: DataFrame,
) -> None:
    expected_columns = [
        "Incident",
        "RMSOccurrenceDate",
        "RMSOccurrenceHour",
        "NIBRSClass",
        "NIBRSDescription",
        "OffenseCount",
        "Beat",
        "Premise",
        "StreetNo",
        "StreetName",
        "StreetType",
        "Suffix",
        "City",
        "ZIPCode",
        "MapLongitude",
        "MapLatitude",
        "_source_file",
    ]
    assert houston_raw.columns == expected_columns
    assert houston_raw.count() == 157

    row = houston_raw.filter(F.col("Incident") == "126259321").first()
    assert row is not None
    assert row.RMSOccurrenceDate == "9/18/2021"
    assert row.RMSOccurrenceHour == "19"
    assert row.NIBRSClass == "90J"
    assert row.MapLongitude == "-95.391398"
    assert row.MapLatitude == "29.744681"
    assert _only_source_file(houston_raw).endswith(
        "houston_fixture.csv"
    )


def test_fort_worth_reader_accepts_actual_json_filename(
    fort_worth_raw: DataFrame,
) -> None:
    assert fort_worth_raw.count() == 163
    assert len(fort_worth_raw.columns) == 30
    assert {
        "Case_No",
        "Case_No_Offense",
        "From_Date",
        "Latitude",
        "Longitude",
        "_latitude",
        "_longitude",
        "_source_file",
    }.issubset(fort_worth_raw.columns)

    row = fort_worth_raw.filter(
        F.col("Case_No_Offense") == "190085144-23C"
    ).first()
    assert row is not None
    assert row.Case_No == "190085144"
    assert row.Offense == "23C"
    assert row.From_Date == 1569880135000
    assert row.Latitude == 32.740739801747495
    assert row._latitude == 32.740745512247045
    assert _only_source_file(fort_worth_raw).endswith(
        "fort_worth_fixture.json"
    )


def test_socioeconomic_batch_reader_loads_json_lines_fixture(
    socioeconomic_raw: DataFrame,
) -> None:
    assert socioeconomic_raw.count() == 117
    assert len(socioeconomic_raw.columns) == 40
    assert {
        "B01003_001E",
        "B19013_001E",
        "NAME",
        "acs_vintage",
        "geoid",
        "_source_file",
    }.issubset(socioeconomic_raw.columns)

    row = socioeconomic_raw.filter(
        F.col("geoid") == "48085031812"
    ).filter(F.col("acs_vintage") == 2021).first()
    assert row is not None
    assert row.NAME == "Census Tract 318.12, Collin County, Texas"
    assert row.B01003_001E == "1416"
    assert row.B19013_001E == "95257"
    assert _only_source_file(socioeconomic_raw).endswith(
        "socioeconomic_fixture.json"
    )


def test_weather_batch_reader_loads_nested_json_lines_fixture(
    weather_raw: DataFrame,
) -> None:
    assert weather_raw.count() == 94
    assert len(weather_raw.columns) == 19
    assert {
        "provider",
        "model",
        "weather_query_cell_id",
        "hourly",
        "hourly_units",
        "_source_file",
    }.issubset(weather_raw.columns)

    row = (
        weather_raw
        .filter(
            F.col("request_id")
            == (
                "31817c77769ed3618c504937973cbccb8d7916b40468f"
                "57798113c993b00a09b"
            )
        )
        .select(
            "provider",
            "model",
            "weather_query_cell_id",
            "grid_elevation",
            F.size("hourly.time").alias("hour_count"),
            F.col("hourly.time")[0].alias("first_time"),
            F.col("hourly.temperature_2m")[0].alias(
                "first_temperature"
            ),
        )
        .first()
    )
    assert row is not None
    assert row.provider == "open_meteo"
    assert row.model == "era5_land"
    assert row.weather_query_cell_id == 604156793613975551
    assert row.grid_elevation == 287.0
    assert row.hour_count == 8760
    assert row.first_time == "2019-01-01T00:00"
    assert row.first_temperature == 2.4
    assert _only_source_file(weather_raw).endswith(
        "weather_fixture.json"
    )
