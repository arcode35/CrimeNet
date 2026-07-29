"""Source-specific raw-file readers."""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from crimenet.contracts.bronze import (
    ACS5_TRACT_RESPONSE_SCHEMA,
    CORRUPT_RECORD_COLUMN,
    OPEN_METEO_RESPONSE_SCHEMA,
    get_source_contract,
)


def _with_source_file(
    dataframe: DataFrame,
) -> DataFrame:
    return dataframe.select(
        "*",
        F.col("_metadata.file_path").alias(
            "_source_file"
        ),
    )


def read_dallas_raw(
    spark: SparkSession,
    input_path: str,
) -> DataFrame:
    contract = get_source_contract("dallas")
    dataframe = (
        spark.read
        .format("csv")
        .schema(contract.schema)
        .option("header", "true")
        .option("enforceSchema", "false")
        .option("mode", "PERMISSIVE")
        .option("columnNameOfCorruptRecord", CORRUPT_RECORD_COLUMN)
        .option("multiLine", "true")
        .option("quote", '"')
        .option("escape", '"')
        .option("recursiveFileLookup", "true")
        .option("pathGlobFilter", "*.csv")
        .load(input_path)
    )

    return _with_source_file(dataframe)


def read_houston_raw(
    spark: SparkSession,
    input_path: str,
) -> DataFrame:
    contract = get_source_contract("houston")
    dataframe = (
        spark.read
        .format("csv")
        .schema(contract.schema)
        .option("header", "true")
        .option("enforceSchema", "false")
        .option("mode", "PERMISSIVE")
        .option("columnNameOfCorruptRecord", CORRUPT_RECORD_COLUMN)
        .option("recursiveFileLookup", "true")
        .option("pathGlobFilter", "*.csv")
        .load(input_path)
    )

    return _with_source_file(dataframe)


def read_fort_worth_raw(
    spark: SparkSession,
    input_path: str,
) -> DataFrame:
    contract = get_source_contract("fort_worth")
    dataframe = (
        spark.read
        .format("json")
        .schema(contract.schema)
        .option("mode", "PERMISSIVE")
        .option("columnNameOfCorruptRecord", CORRUPT_RECORD_COLUMN)
        .option("recursiveFileLookup", "true")
        .option("pathGlobFilter", "*.jsonl")
        .load(input_path)
    )

    return _with_source_file(dataframe)


def read_weather_raw(
    spark: SparkSession,
    input_path: str,
    *,
    schema_path: str,
) -> DataFrame:
    dataframe = (
        spark.readStream
        .format("cloudFiles")
        .schema(OPEN_METEO_RESPONSE_SCHEMA)
        .option(
            "cloudFiles.format",
            "json",
        )
        .option(
            "cloudFiles.schemaLocation",
            schema_path,
        )
        .option(
            "cloudFiles.schemaEvolutionMode",
            "rescue",
        )
        .option(
            "rescuedDataColumn",
            "_rescued_data",
        )
        .option(
            "pathGlobFilter",
            "*.json",
        )
        .load(input_path)
    )

    return _with_source_file(dataframe)

def read_acs5_tract_raw(
    spark: SparkSession,
    input_path: str,
    *,
    schema_path: str,
) -> DataFrame:
    """Incrementally read landed ACS tract JSON Lines files."""
    dataframe = (
        spark.readStream
        .format("cloudFiles")
        .schema(ACS5_TRACT_RESPONSE_SCHEMA)
        .option(
            "cloudFiles.format",
            "json",
        )
        .option(
            "cloudFiles.schemaLocation",
            schema_path,
        )
        .option(
            "cloudFiles.schemaEvolutionMode",
            "rescue",
        )
        .option(
            "rescuedDataColumn",
            "_rescued_data",
        )
        .option(
            "pathGlobFilter",
            "*.jsonl",
        )
        .load(input_path)
    )

    return _with_source_file(dataframe)
