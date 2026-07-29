"""Compute solar position and lighting conditions using pvlib."""

from __future__ import annotations

from collections.abc import Iterator

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
    VALID_LIGHTING_CONDITIONS,
)
from crimenet.observability.logging import get_logger
from crimenet.spatial.h3 import extract_h3_centers
from crimenet.utils.promotion import (
    drop_staging_table,
    promote_staged_delta_table,
    staging_table_name,
)

LOGGER = get_logger(__name__)

OUTPUT_SCHEMA = StructType(
    [
        StructField(
            "lighting_query_cell_id",
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
    *,
    definition_version: str = LIGHTING_DEFINITION_VERSION,
) -> DataFrame:
    if not definition_version.strip():
        raise ValueError("Lighting definition version must not be blank.")

    valid_condition = (
        F.col("weather_query_cell_id").isNotNull() & F.col("occurred_at").isNotNull()
    )

    unique_keys = (
        crime_dataframe.filter(valid_condition)
        .select(
            F.col("weather_query_cell_id").cast("long").alias("lighting_query_cell_id"),
            F.date_trunc(
                "hour",
                F.col("occurred_at"),
            ).alias("solar_timestamp_hour"),
            F.lit(definition_version).alias("lighting_definition_version"),
        )
        .dropDuplicates(list(LIGHTING_KEYS))
    )

    return extract_h3_centers(
        unique_keys,
        cell_column="lighting_query_cell_id",
    ).filter(F.col("query_latitude").isNotNull() & F.col("query_longitude").isNotNull())


def calculate_solar_positions(
    batches: Iterator[pd.DataFrame],
) -> Iterator[pd.DataFrame]:
    output_columns = [field.name for field in OUTPUT_SCHEMA.fields]

    for batch in batches:
        if batch.empty:
            yield pd.DataFrame(columns=output_columns)
            continue

        output_batches: list[pd.DataFrame] = []

        grouped = batch.groupby(
            [
                "lighting_query_cell_id",
                "query_latitude",
                "query_longitude",
                "lighting_definition_version",
            ],
            sort=False,
            dropna=False,
        )

        for (
            _lighting_query_cell_id,
            query_latitude,
            query_longitude,
            lighting_definition_version,
        ), group in grouped:
            definition_version_value = str(lighting_definition_version)
            timestamps = pd.DatetimeIndex(
                pd.to_datetime(
                    group["solar_timestamp_hour"],
                    utc=True,
                )
            )

            solar_position = pvlib.solarposition.get_solarposition(
                time=timestamps,
                latitude=float(str(query_latitude)),
                longitude=float(str(query_longitude)),
                method="nrel_numpy",
            )

            result = group.copy()

            elevation = pd.Series(
                solar_position["elevation"].to_numpy(),
                index=result.index,
                dtype="float64",
            )

            apparent_elevation = pd.Series(
                solar_position["apparent_elevation"].to_numpy(),
                index=result.index,
                dtype="float64",
            )

            result["solar_elevation_deg"] = elevation

            result["apparent_solar_elevation_deg"] = apparent_elevation

            result["solar_zenith_deg"] = solar_position["zenith"].to_numpy()

            result["solar_azimuth_deg"] = solar_position["azimuth"].to_numpy()

            result["lighting_condition"] = classify_lighting_condition(elevation)

            result["is_daylight"] = elevation >= 0.0

            result["pvlib_version"] = pvlib.__version__

            result["lighting_definition_version"] = definition_version_value

            output_batches.append(result[output_columns])

        yield pd.concat(
            output_batches,
            ignore_index=True,
        )


def compute_lighting_conditions(
    lighting_keys: DataFrame,
) -> DataFrame:
    return (
        lighting_keys.repartition("lighting_query_cell_id")
        .sortWithinPartitions(
            *LIGHTING_KEYS,
        )
        .mapInPandas(
            calculate_solar_positions,  # type: ignore[arg-type]
            schema=OUTPUT_SCHEMA,
        )
        .withColumn(
            "computed_at",
            F.current_timestamp(),
        )
    )


def _validate_lighting_schema(
    dataframe: DataFrame,
    *,
    enforce_schema_nullability: bool,
) -> None:
    required_fields = {field.name: field for field in OUTPUT_SCHEMA.fields}
    actual_fields = {field.name: field for field in dataframe.schema.fields}
    schema_errors = [
        (
            f"{name} expected {expected_field.dataType.simpleString()}, "
            f"found {actual_fields[name].dataType.simpleString()}"
        )
        for name, expected_field in required_fields.items()
        if name in actual_fields
        and actual_fields[name].dataType != expected_field.dataType
    ]
    if enforce_schema_nullability:
        schema_errors.extend(
            (
                f"{name} expected nullable={expected_field.nullable}, "
                f"found nullable={actual_fields[name].nullable}"
            )
            for name, expected_field in required_fields.items()
            if name in actual_fields
            and not expected_field.nullable
            and actual_fields[name].nullable
        )
    missing_columns = sorted(set(required_fields) - set(actual_fields))

    if missing_columns or schema_errors:
        details = [
            *([f"missing columns: {missing_columns}"] if missing_columns else []),
            *schema_errors,
        ]
        raise RuntimeError(
            "Lighting result schema is incompatible: " + "; ".join(details)
        )


def validate_lighting_results(
    dataframe: DataFrame,
    *,
    expected_keys: DataFrame | None = None,
    definition_version: str | None = None,
    allow_other_definition_versions: bool = False,
    enforce_schema_nullability: bool = True,
) -> None:
    _validate_lighting_schema(
        dataframe,
        enforce_schema_nullability=enforce_schema_nullability,
    )

    duplicate_keys = (
        dataframe.groupBy(*LIGHTING_KEYS).count().filter(F.col("count") > 1)
    )

    if not duplicate_keys.isEmpty():
        raise RuntimeError(
            "Lighting results contain duplicate H3-cell/hour/definition-version keys."
        )

    required_value_is_null = F.lit(False)
    for field in OUTPUT_SCHEMA.fields:
        if not field.nullable:
            required_value_is_null |= F.col(field.name).isNull()

    invalid_rows = dataframe.filter(
        required_value_is_null
        | (F.trim(F.col("lighting_definition_version")) == "")
        | (F.trim(F.col("pvlib_version")) == "")
        | (
            F.col("solar_timestamp_hour")
            != F.date_trunc(
                "hour",
                F.col("solar_timestamp_hour"),
            )
        )
        | (
            (F.col("lighting_definition_version") != F.lit(definition_version))
            if (definition_version is not None and not allow_other_definition_versions)
            else F.lit(False)
        )
        | F.col("query_latitude").isNull()
        | F.col("query_longitude").isNull()
        | ~F.col("query_latitude").between(
            -90.0,
            90.0,
        )
        | ~F.col("query_longitude").between(
            -180.0,
            180.0,
        )
        | F.col("solar_elevation_deg").isNull()
        | F.col("apparent_solar_elevation_deg").isNull()
        | F.col("solar_zenith_deg").isNull()
        | F.col("solar_azimuth_deg").isNull()
        | ~F.col("solar_elevation_deg").between(
            -90.0,
            90.0,
        )
        | ~F.col("apparent_solar_elevation_deg").between(
            -90.0,
            90.0,
        )
        | ~F.col("solar_zenith_deg").between(
            0.0,
            180.0,
        )
        | ~F.col("solar_azimuth_deg").between(
            0.0,
            360.0,
        )
        | ~F.col("lighting_condition").isin(*VALID_LIGHTING_CONDITIONS)
        | (F.col("is_daylight") != (F.col("solar_elevation_deg") >= 0.0))
    )

    if not invalid_rows.isEmpty():
        raise RuntimeError(
            "Lighting results contain invalid solar-position or classification values."
        )

    if expected_keys is None:
        return

    expected = expected_keys.select(*LIGHTING_KEYS).dropDuplicates(list(LIGHTING_KEYS))
    actual = dataframe.select(*LIGHTING_KEYS)

    if definition_version is not None:
        actual = actual.filter(
            F.col("lighting_definition_version") == definition_version
        )

    missing_keys = expected.join(
        actual,
        on=list(LIGHTING_KEYS),
        how="left_anti",
    )
    unexpected_keys = actual.join(
        expected,
        on=list(LIGHTING_KEYS),
        how="left_anti",
    )

    if not missing_keys.isEmpty() or not unexpected_keys.isEmpty():
        raise RuntimeError(
            "Lighting candidate business-key set does not match "
            "the requested H3-cell/hour/definition-version set: "
            f"missing={missing_keys.count()}, "
            f"unexpected={unexpected_keys.count()}."
        )


def materialize_lighting_conditions(
    spark: SparkSession,
    *,
    crime_table: str,
    target_table: str,
    full_rebuild: bool,
    definition_version: str = LIGHTING_DEFINITION_VERSION,
    pipeline_run_id: str | None = None,
) -> None:
    crime_dataframe = spark.table(crime_table)

    candidate_keys = extract_lighting_keys(
        crime_dataframe,
        definition_version=definition_version,
    )

    table_exists = spark.catalog.tableExists(target_table)

    target_has_current_contract = table_exists and set(LIGHTING_KEYS).issubset(
        spark.table(target_table).columns
    )

    rebuild_required = full_rebuild or not target_has_current_contract

    if target_has_current_contract and not full_rebuild:
        validate_lighting_results(
            spark.table(target_table),
            allow_other_definition_versions=True,
            enforce_schema_nullability=False,
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
            .dropDuplicates(list(LIGHTING_KEYS))
        )

        keys_to_compute = candidate_keys.join(
            existing_keys,
            on=list(LIGHTING_KEYS),
            how="left_anti",
        )

        LOGGER.info(
            "lighting_incremental_started",
            source_table=crime_table,
            target_table=target_table,
        )

    # This action evaluates only key extraction and the anti-join.
    # It does not execute pvlib.
    if not rebuild_required and keys_to_compute.isEmpty():
        LOGGER.info(
            "lighting_no_new_keys",
            target_table=target_table,
            lighting_definition_version=definition_version,
        )

        validate_lighting_results(
            spark.table(target_table),
            expected_keys=candidate_keys,
            definition_version=definition_version,
            allow_other_definition_versions=True,
            enforce_schema_nullability=False,
        )
        return

    lighting_results = compute_lighting_conditions(keys_to_compute)
    _validate_lighting_schema(
        lighting_results,
        enforce_schema_nullability=True,
    )
    staging_table = staging_table_name(
        target_table,
        pipeline_run_id,
    )

    try:
        (
            lighting_results.write.format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(staging_table)
        )

        staged_results = spark.table(staging_table)
        validate_lighting_results(
            staged_results,
            expected_keys=keys_to_compute,
            definition_version=definition_version,
            enforce_schema_nullability=False,
        )

        if rebuild_required:
            promote_staged_delta_table(
                spark,
                staging_table=staging_table,
                target_table=target_table,
            )
            write_mode = "staged_replace"
        else:
            from delta.tables import (  # type: ignore[import-not-found]
                DeltaTable,
            )

            merge_condition = """
                target.lighting_query_cell_id
                    = source.lighting_query_cell_id
                AND target.solar_timestamp_hour
                    = source.solar_timestamp_hour
                AND target.lighting_definition_version
                    = source.lighting_definition_version
            """

            (
                DeltaTable.forName(
                    spark,
                    target_table,
                )
                .alias("target")
                .merge(
                    staged_results.alias("source"),
                    merge_condition,
                )
                .whenNotMatchedInsertAll()
                .execute()
            )
            write_mode = "validated_merge"

        # Validate the committed table against every currently requested key.
        validate_lighting_results(
            spark.table(target_table),
            expected_keys=candidate_keys,
            definition_version=definition_version,
            allow_other_definition_versions=True,
            enforce_schema_nullability=False,
        )
    finally:
        drop_staging_table(
            spark,
            staging_table,
        )

    LOGGER.info(
        "lighting_conditions_materialized",
        target_table=target_table,
        write_mode=write_mode,
        lighting_definition_version=definition_version,
    )
