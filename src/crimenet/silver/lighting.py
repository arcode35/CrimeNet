"""Compute solar position and lighting conditions using pvlib."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from importlib import import_module

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pvlib  # type: ignore[import-untyped]
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from crimenet.contracts.lighting import (
    LIGHTING_DEFINITION_VERSION,
    LIGHTING_KEYS,
)
from crimenet.observability.logging import get_logger

LOGGER = get_logger(__name__)

CENTER_SCHEMA = """
    type STRING,
    coordinates ARRAY<DOUBLE>
"""

OUTPUT_SCHEMA = StructType(
    [
        StructField(
            "weather_query_cell_id",
            LongType(),
            False,
        ),
        StructField(
            "solar_timestamp_hour",
            TimestampType(),
            False,
        ),
        StructField(
            "query_latitude",
            DoubleType(),
            False,
        ),
        StructField(
            "query_longitude",
            DoubleType(),
            False,
        ),
        StructField(
            "solar_elevation_deg",
            DoubleType(),
            True,
        ),
        StructField(
            "apparent_solar_elevation_deg",
            DoubleType(),
            True,
        ),
        StructField(
            "solar_zenith_deg",
            DoubleType(),
            True,
        ),
        StructField(
            "solar_azimuth_deg",
            DoubleType(),
            True,
        ),
        StructField(
            "lighting_condition",
            StringType(),
            False,
        ),
        StructField(
            "is_daylight",
            BooleanType(),
            False,
        ),
        StructField(
            "pvlib_version",
            StringType(),
            False,
        ),
        StructField(
            "lighting_definition_version",
            StringType(),
            False,
        ),
    ]
)


def classify_lighting_condition(
    elevation: pd.Series,
) -> np.ndarray:
    return np.select(
        [
            elevation >= 0.0,
            elevation >= -6.0,
            elevation >= -12.0,
            elevation >= -18.0,
        ],
        [
            "daylight",
            "civil_twilight",
            "nautical_twilight",
            "astronomical_twilight",
        ],
        default="night",
    )


def extract_lighting_keys(
    crime_dataframe: DataFrame,
) -> DataFrame:
    """Extract cell-hour keys and attach Databricks H3 cell centers."""
    unique_keys = extract_lighting_key_grain(
        crime_dataframe
    )

    return attach_lighting_cell_centers(
        unique_keys
    )


def extract_lighting_key_grain(
    crime_dataframe: DataFrame,
) -> DataFrame:
    """Extract the deterministic active-version cell-hour grain."""
    valid_condition = (
        F.col("weather_query_cell_id").isNotNull()
        & F.col("occurred_at").isNotNull()
    )

    return (
        crime_dataframe
        .filter(valid_condition)
        .select(
            F.col("weather_query_cell_id").cast("long"),
            F.date_trunc(
                "hour",
                F.col("occurred_at"),
            ).alias("solar_timestamp_hour"),
            F.lit(
                LIGHTING_DEFINITION_VERSION
            ).alias("lighting_definition_version"),
        )
        .dropDuplicates(list(LIGHTING_KEYS))
    )


def attach_lighting_cell_centers(
    lighting_keys: DataFrame,
) -> DataFrame:
    """Attach H3 centers using Databricks-only SQL functions."""
    dbf = import_module(
        "pyspark.databricks.sql.functions"
    )

    return (
        lighting_keys
        .withColumn(
            "_center",
            F.from_json(
                dbf.h3_centerasgeojson(
                    "weather_query_cell_id"
                ),
                CENTER_SCHEMA,
            ),
        )
        .withColumn(
            "query_longitude",
            F.col("_center.coordinates")[0],
        )
        .withColumn(
            "query_latitude",
            F.col("_center.coordinates")[1],
        )
        .drop("_center")
        .filter(
            F.col("query_latitude").isNotNull()
            & F.col("query_longitude").isNotNull()
            & F.col("query_latitude").between(
                -90.0,
                90.0,
            )
            & F.col("query_longitude").between(
                -180.0,
                180.0,
            )
        )
    )


def select_missing_lighting_keys(
    candidate_keys: DataFrame,
    existing_keys: DataFrame,
) -> DataFrame:
    """Return active candidate keys absent from an existing key set."""
    for dataset_name, dataframe in (
        ("candidate", candidate_keys),
        ("existing", existing_keys),
    ):
        missing_columns = [
            column_name
            for column_name in LIGHTING_KEYS
            if column_name
            not in dataframe.columns
        ]

        if missing_columns:
            raise ValueError(
                f"{dataset_name.title()} lighting keys are missing "
                f"columns: {missing_columns}"
            )

    return (
        candidate_keys
        .join(
            existing_keys
            .select(*LIGHTING_KEYS)
            .dropDuplicates(
                list(LIGHTING_KEYS)
            ),
            on=list(LIGHTING_KEYS),
            how="left_anti",
        )
    )


def calculate_solar_positions(
    batches: Iterable[pd.DataFrame],
) -> Iterator[pd.DataFrame]:
    output_columns = [
        field.name
        for field in OUTPUT_SCHEMA.fields
    ]

    for batch in batches:
        if batch.empty:
            yield pd.DataFrame(
                columns=output_columns
            )
            continue

        output_batches: list[
            pd.DataFrame
        ] = []

        grouped = batch.groupby(
            [
                "weather_query_cell_id",
                "query_latitude",
                "query_longitude",
            ],
            sort=False,
            dropna=False,
        )

        for (
            _weather_query_cell_id,
            query_latitude,
            query_longitude,
        ), group in grouped:
            timestamps = pd.DatetimeIndex(
                pd.to_datetime(
                    group["solar_timestamp_hour"],
                    utc=True,
                )
            )

            solar_position = (
                pvlib.solarposition
                .get_solarposition(
                    time=timestamps,
                    latitude=float(
                        query_latitude
                    ),
                    longitude=float(
                        query_longitude
                    ),
                    method="nrel_numpy",
                )
            )

            result = group.copy()

            elevation = pd.Series(
                solar_position[
                    "elevation"
                ].to_numpy(),
                index=result.index,
                dtype="float64",
            )

            apparent_elevation = pd.Series(
                solar_position[
                    "apparent_elevation"
                ].to_numpy(),
                index=result.index,
                dtype="float64",
            )

            result[
                "solar_elevation_deg"
            ] = elevation

            result[
                "apparent_solar_elevation_deg"
            ] = apparent_elevation

            result[
                "solar_zenith_deg"
            ] = (
                solar_position[
                    "zenith"
                ].to_numpy()
            )

            result[
                "solar_azimuth_deg"
            ] = (
                solar_position[
                    "azimuth"
                ].to_numpy()
            )

            result[
                "lighting_condition"
            ] = classify_lighting_condition(
                elevation
            )

            result["is_daylight"] = (
                elevation >= 0.0
            )

            result["pvlib_version"] = (
                pvlib.__version__
            )

            result[
                "lighting_definition_version"
            ] = LIGHTING_DEFINITION_VERSION

            output_batches.append(
                result[output_columns]
            )

        yield pd.concat(
            output_batches,
            ignore_index=True,
        )


def compute_lighting_conditions(
    lighting_keys: DataFrame,
) -> DataFrame:
    return (
        lighting_keys
        .repartition(
            "weather_query_cell_id"
        )
        .sortWithinPartitions(
            "weather_query_cell_id",
            "solar_timestamp_hour",
            "lighting_definition_version",
        )
        .mapInPandas(
            calculate_solar_positions,
            schema=OUTPUT_SCHEMA,
        )
        .withColumn(
            "computed_at",
            F.current_timestamp(),
        )
    )


def validate_lighting_results(
    dataframe: DataFrame,
) -> None:
    missing_columns = [
        field.name
        for field in OUTPUT_SCHEMA.fields
        if field.name not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Lighting results are missing required columns: "
            f"{missing_columns}"
        )

    invalid_keys = dataframe.filter(
        F.col("weather_query_cell_id").isNull()
        | F.col("solar_timestamp_hour").isNull()
        | F.col(
            "lighting_definition_version"
        ).isNull()
        | (
            F.col(
                "lighting_definition_version"
            )
            != LIGHTING_DEFINITION_VERSION
        )
        | (
            F.col("solar_timestamp_hour")
            != F.date_trunc(
                "hour",
                F.col("solar_timestamp_hour"),
            )
        )
    )

    if not invalid_keys.isEmpty():
        raise RuntimeError(
            "Lighting results contain null, misaligned, or "
            "unsupported active-version keys."
        )

    duplicate_keys = (
        dataframe
        .groupBy(*LIGHTING_KEYS)
        .count()
        .filter(
            F.col("count") > 1
        )
    )

    if not duplicate_keys.isEmpty():
        raise RuntimeError(
            "Lighting results contain duplicate "
            "H3-cell/timestamp keys."
        )

    valid_conditions = (
        "daylight",
        "civil_twilight",
        "nautical_twilight",
        "astronomical_twilight",
        "night",
    )

    expected_condition = (
        F.when(
            F.col("solar_elevation_deg") >= 0.0,
            F.lit("daylight"),
        )
        .when(
            F.col("solar_elevation_deg") >= -6.0,
            F.lit("civil_twilight"),
        )
        .when(
            F.col("solar_elevation_deg") >= -12.0,
            F.lit("nautical_twilight"),
        )
        .when(
            F.col("solar_elevation_deg") >= -18.0,
            F.lit("astronomical_twilight"),
        )
        .otherwise(
            F.lit("night")
        )
    )

    invalid_rows = (
        dataframe
        .filter(
            F.col(
                "query_latitude"
            ).isNull()
            | F.col(
                "query_longitude"
            ).isNull()
            | F.isnan(
                F.col("query_latitude")
            )
            | F.isnan(
                F.col("query_longitude")
            )
            | ~F.col(
                "query_latitude"
            ).between(
                -90.0,
                90.0,
            )
            | ~F.col(
                "query_longitude"
            ).between(
                -180.0,
                180.0,
            )
            | F.col(
                "solar_elevation_deg"
            ).isNull()
            | F.col(
                "apparent_solar_elevation_deg"
            ).isNull()
            | F.col(
                "solar_zenith_deg"
            ).isNull()
            | F.col(
                "solar_azimuth_deg"
            ).isNull()
            | F.isnan(
                F.col("solar_elevation_deg")
            )
            | F.isnan(
                F.col(
                    "apparent_solar_elevation_deg"
                )
            )
            | F.isnan(
                F.col("solar_zenith_deg")
            )
            | F.isnan(
                F.col("solar_azimuth_deg")
            )
            | ~F.col(
                "solar_elevation_deg"
            ).between(
                -90.0,
                90.0,
            )
            | ~F.col(
                "apparent_solar_elevation_deg"
            ).between(
                -90.0,
                90.0,
            )
            | ~F.col(
                "solar_zenith_deg"
            ).between(
                0.0,
                180.0,
            )
            | ~F.col(
                "solar_azimuth_deg"
            ).between(
                0.0,
                360.0,
            )
            | ~F.col(
                "lighting_condition"
            ).isin(*valid_conditions)
            | (
                F.col("lighting_condition")
                != expected_condition
            )
            | F.col("is_daylight").isNull()
            | (
                F.col("is_daylight")
                != (
                    F.col(
                        "solar_elevation_deg"
                    ) >= 0.0
                )
            )
            | F.col("pvlib_version").isNull()
            | (
                F.length(
                    F.trim(
                        F.col("pvlib_version")
                    )
                )
                == 0
            )
        )
    )

    if not invalid_rows.isEmpty():
        raise RuntimeError(
            "Lighting results contain invalid "
            "solar-position or classification values."
        )


def materialize_lighting_conditions(
    spark: SparkSession,
    *,
    crime_table: str,
    target_table: str,
    full_rebuild: bool,
) -> None:
    crime_dataframe = spark.table(
        crime_table
    )

    candidate_keys = extract_lighting_keys(
        crime_dataframe
    )

    table_exists = spark.catalog.tableExists(
        target_table
    )

    rebuild_required = (
        full_rebuild
        or not table_exists
    )

    if rebuild_required:
        keys_to_compute = candidate_keys

        LOGGER.info(
            "lighting_full_rebuild_started",
            source_table=crime_table,
            target_table=target_table,
        )
    else:
        existing_keys = (
            spark.table(target_table)
            .select(*LIGHTING_KEYS)
            .dropDuplicates(
                list(LIGHTING_KEYS)
            )
        )

        keys_to_compute = (
            select_missing_lighting_keys(
                candidate_keys,
                existing_keys,
            )
        )

        LOGGER.info(
            "lighting_incremental_started",
            source_table=crime_table,
            target_table=target_table,
        )

    # This action evaluates only the Spark key extraction and
    # anti-join. It does not execute pvlib.
    if keys_to_compute.isEmpty():
        if table_exists:
            validate_lighting_results(
                spark.table(target_table)
            )

        LOGGER.info(
            "lighting_no_new_keys",
            target_table=target_table,
        )
        return

    lighting_results = (
        compute_lighting_conditions(
            keys_to_compute
        )
    )

    if rebuild_required:
        (
            lighting_results.write
            .format("delta")
            .mode("overwrite")
            .option(
                "overwriteSchema",
                "true",
            )
            .saveAsTable(
                target_table
            )
        )

        write_mode = "overwrite"
    else:
        merge_condition = """
            target.weather_query_cell_id
                = source.weather_query_cell_id
            AND target.solar_timestamp_hour
                = source.solar_timestamp_hour
            AND target.lighting_definition_version
                = source.lighting_definition_version
        """

        DeltaTable = import_module(
            "delta.tables"
        ).DeltaTable

        (
            DeltaTable.forName(
                spark,
                target_table,
            )
            .alias("target")
            .merge(
                lighting_results.alias(
                    "source"
                ),
                merge_condition,
            )
            .whenNotMatchedInsertAll()
            .execute()
        )

        write_mode = "merge"

    # Read the committed Delta data. This does not rerun pvlib.
    validate_lighting_results(
        spark.table(target_table)
    )

    LOGGER.info(
        "lighting_conditions_materialized",
        target_table=target_table,
        write_mode=write_mode,
    )
