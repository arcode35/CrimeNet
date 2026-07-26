"""Source-specific raw-file readers."""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


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
    dataframe = (
        spark.read
        .format("csv")
        .option("header", "true")
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
    dataframe = (
        spark.read
        .format("csv")
        .option("header", "true")
        .option("recursiveFileLookup", "true")
        .option("pathGlobFilter", "*.csv")
        .load(input_path)
    )

    return _with_source_file(dataframe)


def read_fort_worth_raw(
    spark: SparkSession,
    input_path: str,
) -> DataFrame:
    dataframe = (
        spark.read
        .format("json")
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