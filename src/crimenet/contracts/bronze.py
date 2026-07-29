"""Versioned schemas for externally supplied municipal crime files."""

from __future__ import annotations

from dataclasses import dataclass

from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

CRIME_CONTRACT_VERSION = "municipal_crime_v1"
CORRUPT_RECORD_COLUMN = "_corrupt_record"


def _string_schema(*column_names: str) -> StructType:
    return StructType(
        [StructField(name, StringType(), True) for name in column_names]
        + [StructField(CORRUPT_RECORD_COLUMN, StringType(), True)]
    )


DALLAS_SCHEMA = _string_schema(
    "Service Number ID",
    "Incident Number w/year",
    "NIBRS Code",
    "NIBRS Crime",
    "UCR Offense Name",
    "Type of Incident",
    "UCR Offense Description",
    "Date1 of Occurrence",
    "Time1 of Occurrence",
    "Date of Report",
    "Update Date",
    "Incident Address",
    "City",
    "State",
    "Zip Code",
    "Beat",
    "Type Location",
    "Location1",
    "X Coordinate",
    "Y Cordinate",
)

HOUSTON_SCHEMA = _string_schema(
    "Incident",
    "NIBRSClass",
    "NIBRSDescription",
    "RMSOccurrenceDate",
    "RMSOccurrenceHour",
    "OffenseCount",
    "StreetNo",
    "StreetName",
    "StreetType",
    "Suffix",
    "City",
    "ZIPCode",
    "Beat",
    "Premise",
    "MapLatitude",
    "MapLongitude",
)

# ArcGIS exports have used both underscored and title-cased coordinate names.
# The explicit contract deliberately names each supported variant.
FORT_WORTH_SCHEMA = _string_schema(
    "Case_No_Offense",
    "OBJECTID",
    "Case_No",
    "Offense",
    "Nature_Of_Call",
    "Offense_Desc",
    "From_Date",
    "Reported_Date",
    "LastUpdated",
    "Address",
    "Block_Address",
    "City",
    "State",
    "Beat",
    "LocationTypeDescription",
    "Latitude",
    "_Latitude",
    "Longitude",
    "_Longitude",
    "X_Coordinate",
    "Y_Coordinate",
)

OPEN_METEO_HOURLY_SCHEMA = StructType(
    [
        StructField(
            "time",
            ArrayType(StringType(), containsNull=True),
            True,
        ),
        StructField(
            "temperature_2m",
            ArrayType(DoubleType(), containsNull=True),
            True,
        ),
    ]
)

OPEN_METEO_UNITS_SCHEMA = StructType(
    [
        StructField("time", StringType(), True),
        StructField("temperature_2m", StringType(), True),
    ]
)

OPEN_METEO_RESPONSE_SCHEMA = StructType(
    [
        StructField("request_id", StringType(), True),
        StructField("provider", StringType(), True),
        StructField("model", StringType(), True),
        StructField("weather_query_cell_id", LongType(), True),
        StructField("h3_resolution", IntegerType(), True),
        StructField("query_latitude", DoubleType(), True),
        StructField("query_longitude", DoubleType(), True),
        StructField("grid_latitude", DoubleType(), True),
        StructField("grid_longitude", DoubleType(), True),
        StructField("grid_elevation", DoubleType(), True),
        StructField("start_date", StringType(), True),
        StructField("end_date", StringType(), True),
        StructField("timezone", StringType(), True),
        StructField("utc_offset_seconds", IntegerType(), True),
        StructField("cell_selection", StringType(), True),
        StructField(
            "hourly_variables",
            ArrayType(StringType(), containsNull=False),
            True,
        ),
        StructField("hourly_units", OPEN_METEO_UNITS_SCHEMA, True),
        StructField("hourly", OPEN_METEO_HOURLY_SCHEMA, True),
        StructField("_rescued_data", StringType(), True),
    ]
)

_ACS5_VARIABLES = tuple(
    name.upper()
    for name in (
        "b01003_001e",
        "b01003_001m",
        "b01002_001e",
        "b01002_001m",
        "b19013_001e",
        "b19013_001m",
        "b17001_001e",
        "b17001_001m",
        "b17001_002e",
        "b17001_002m",
        "b23025_003e",
        "b23025_003m",
        "b23025_005e",
        "b23025_005m",
        "b25001_001e",
        "b25001_001m",
        "b25002_002e",
        "b25002_002m",
        "b25002_003e",
        "b25002_003m",
        "b25003_001e",
        "b25003_001m",
        "b25003_003e",
        "b25003_003m",
        "b08201_001e",
        "b08201_001m",
        "b08201_002e",
        "b08201_002m",
    )
)

ACS5_TRACT_RESPONSE_SCHEMA = StructType(
    [
        StructField("NAME", StringType(), True),
        *[StructField(name, StringType(), True) for name in _ACS5_VARIABLES],
        StructField("state", StringType(), True),
        StructField("county", StringType(), True),
        StructField("tract", StringType(), True),
        StructField("geoid", StringType(), True),
        StructField("acs_vintage", IntegerType(), True),
        StructField("period_start_year", IntegerType(), True),
        StructField("period_end_year", IntegerType(), True),
        StructField("dataset", StringType(), True),
        StructField("geography_type", StringType(), True),
        StructField("retrieved_at", StringType(), True),
        StructField("_rescued_data", StringType(), True),
    ]
)


@dataclass(frozen=True)
class SourceContract:
    source_system: str
    version: str
    file_format: str
    schema: StructType
    required_normalized_columns: frozenset[str]


SOURCE_CONTRACTS = {
    "dallas": SourceContract(
        source_system="dallas",
        version=CRIME_CONTRACT_VERSION,
        file_format="csv",
        schema=DALLAS_SCHEMA,
        required_normalized_columns=frozenset(
            {
                "service_number_id",
                "incident_number_w_year",
                "date1_of_occurrence",
                "time1_of_occurrence",
            }
        ),
    ),
    "houston": SourceContract(
        source_system="houston",
        version=CRIME_CONTRACT_VERSION,
        file_format="csv",
        schema=HOUSTON_SCHEMA,
        required_normalized_columns=frozenset(
            {
                "incident",
                "nibrsclass",
                "rmsoccurrencedate",
                "rmsoccurrencehour",
            }
        ),
    ),
    "fort_worth": SourceContract(
        source_system="fort_worth",
        version=CRIME_CONTRACT_VERSION,
        file_format="json",
        schema=FORT_WORTH_SCHEMA,
        required_normalized_columns=frozenset(
            {
                "case_no_offense",
                "case_no",
                "from_date",
            }
        ),
    ),
}


def get_source_contract(source: str) -> SourceContract:
    try:
        return SOURCE_CONTRACTS[source]
    except KeyError as exc:
        raise ValueError(f"No municipal source contract for {source!r}.") from exc


def validate_contract_columns(
    column_names: list[str],
    contract: SourceContract,
) -> None:
    """Fail clearly when a required contract column is absent."""
    missing = contract.required_normalized_columns - set(column_names)
    if missing:
        raise ValueError(
            f"{contract.source_system} contract {contract.version} is "
            "missing required normalized columns: "
            + ", ".join(sorted(missing))
        )
