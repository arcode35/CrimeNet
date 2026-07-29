from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType

from crimenet.contracts.bronze import (
    DALLAS_SCHEMA,
    FORT_WORTH_SCHEMA,
    HOUSTON_SCHEMA,
)
from crimenet.ingestion.column_names import normalize_column_names
from crimenet.ingestion.metadata import add_ingestion_metadata
from crimenet.jobs.bronze_ingestion import COLUMN_OVERRIDES
from crimenet.transforms.dallas import to_canonical as dallas_to_canonical
from crimenet.transforms.fort_worth import (
    to_canonical as fort_worth_to_canonical,
)
from crimenet.transforms.houston import to_canonical as houston_to_canonical


def _bronze(
    spark: SparkSession,
    *,
    schema: StructType,
    values: dict[str, object],
    source: str,
) -> DataFrame:
    raw = spark.createDataFrame([values], schema=schema).withColumn(
        "_source_file",
        F.lit("/landing/original"),
    )
    normalized = normalize_column_names(
        raw,
        overrides=COLUMN_OVERRIDES.get(source),
    )
    return add_ingestion_metadata(
        normalized,
        source,
        contract_version="municipal_crime_v1",
    )


def test_dallas_timestamp_coordinate_and_identity_normalization(
    spark: SparkSession,
) -> None:
    row = {
        "Service Number ID": "OFF-1",
        "Incident Number w/year": "INC-1",
        "Date1 of Occurrence": "2024-05-03",
        "Time1 of Occurrence": "14:37",
        "Date of Report": "2024-05-03 15:00:00",
        "Update Date": "2024-05-04 10:00:00",
        "Location1": "(32.7767, -96.7970)",
    }
    result_frame = dallas_to_canonical(
        _bronze(
            spark,
            schema=DALLAS_SCHEMA,
            values=row,
            source="dallas",
        )
    )
    result = result_frame.first()
    assert result is not None
    assert (
        result_frame.select(
            F.date_format("occurred_at", "yyyy-MM-dd HH:mm:ss").alias("value")
        ).first()["value"]
        == "2024-05-03 19:37:00"
    )
    assert result["latitude"] == 32.7767
    assert result["longitude"] == -96.797
    assert result["source_incident_id"] == "INC-1"
    assert result["source_offense_id"] == "OFF-1"
    assert len(result["business_identity"]) == 64


def test_houston_builds_stable_offense_id_from_incident_and_class(
    spark: SparkSession,
) -> None:
    row = {
        "Incident": "H-1",
        "NIBRSClass": "23A",
        "NIBRSDescription": "Theft",
        "RMSOccurrenceDate": "5/3/2024",
        "RMSOccurrenceHour": "7",
        "MapLatitude": "29.7604",
        "MapLongitude": "-95.3698",
    }
    original = _bronze(
        spark,
        schema=HOUSTON_SCHEMA,
        values=row,
        source="houston",
    )
    moved = original.withColumn("source_file", F.lit("/other/path.csv"))
    first = houston_to_canonical(original).first()
    second = houston_to_canonical(moved).first()
    assert first is not None and second is not None
    assert (
        houston_to_canonical(original)
        .select(F.hour("occurred_at").alias("hour"))
        .first()["hour"]
        == 12
    )
    assert first["source_offense_id"] == second["source_offense_id"]
    assert first["business_identity"] == second["business_identity"]


def test_houston_corrections_keep_identity_and_missing_keys_quarantine(
    spark: SparkSession,
) -> None:
    original_values = {
        "Incident": " H-1 ",
        "NIBRSClass": "23a",
        "NIBRSDescription": "Theft",
        "RMSOccurrenceDate": "5/3/2024",
        "RMSOccurrenceHour": "7",
        "StreetNo": "100",
        "StreetName": "Main",
        "MapLatitude": "29.7604",
        "MapLongitude": "-95.3698",
    }
    corrected_values = {
        **original_values,
        "RMSOccurrenceHour": "8",
        "StreetNo": "200",
        "MapLatitude": "29.7700",
    }

    original = houston_to_canonical(
        _bronze(
            spark,
            schema=HOUSTON_SCHEMA,
            values=original_values,
            source="houston",
        )
    ).first()
    corrected = houston_to_canonical(
        _bronze(
            spark,
            schema=HOUSTON_SCHEMA,
            values=corrected_values,
            source="houston",
        )
    ).first()
    missing_class = houston_to_canonical(
        _bronze(
            spark,
            schema=HOUSTON_SCHEMA,
            values={**original_values, "NIBRSClass": "  "},
            source="houston",
        )
    ).first()

    assert original is not None and corrected is not None
    assert missing_class is not None
    assert original["source_incident_id"] == "H-1"
    assert original["offense_code"] == "23A"
    assert original["source_offense_id"] == corrected["source_offense_id"]
    assert original["business_identity"] == corrected["business_identity"]
    assert missing_class["source_offense_id"] is None


def test_fort_worth_epoch_millis_and_alternate_coordinates(
    spark: SparkSession,
) -> None:
    row = {
        "Case_No_Offense": "FW-1-01",
        "Case_No": "FW-1",
        "From_Date": "1714747042000",
        "Latitude": "32.75",
        "_Latitude": "32.76",
        "Longitude": "-97.33",
        "_Longitude": "-97.34",
    }
    result = fort_worth_to_canonical(
        _bronze(
            spark,
            schema=FORT_WORTH_SCHEMA,
            values=row,
            source="fort_worth",
        )
    ).first()
    assert result is not None
    assert result["source_offense_id"] == "FW-1-01"
    assert result["occurred_at"] is not None
    assert result["latitude"] == 32.75
    assert result["alternate_latitude"] == 32.76


def test_ambiguous_and_nonexistent_texas_wall_times_are_not_guessed(
    spark: SparkSession,
) -> None:
    dallas = dallas_to_canonical(
        _bronze(
            spark,
            schema=DALLAS_SCHEMA,
            values={
                "Service Number ID": "DST-SPRING",
                "Incident Number w/year": "DST-SPRING",
                "Date1 of Occurrence": "2024-03-10",
                "Time1 of Occurrence": "02:30",
            },
            source="dallas",
        )
    ).first()
    houston = houston_to_canonical(
        _bronze(
            spark,
            schema=HOUSTON_SCHEMA,
            values={
                "Incident": "DST-FALL",
                "NIBRSClass": "23A",
                "RMSOccurrenceDate": "11/3/2024",
                "RMSOccurrenceHour": "1",
            },
            source="houston",
        )
    ).first()

    assert dallas is not None and dallas["occurred_at"] is None
    assert houston is not None and houston["occurred_at"] is None
