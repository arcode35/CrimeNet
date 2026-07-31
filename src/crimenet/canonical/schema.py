"""Canonical CrimeNet Silver schema."""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    LongType,
    ShortType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)


def _field(
    name: str,
    data_type: object,
) -> StructField:
    return StructField(
        name,
        data_type,
        nullable=True,
    )


CANONICAL_CRIME_SCHEMA = StructType(
    [
        # Canonical identity
        _field(
            "canonical_record_id",
            StringType(),
        ),
        _field("source_city", StringType()),
        _field("source_system", StringType()),
        _field("source_dataset", StringType()),
        _field(
            "source_dataset_id",
            StringType(),
        ),
        _field(
            "source_dataset_kind",
            StringType(),
        ),
        _field(
            "source_record_id",
            StringType(),
        ),
        _field(
            "source_record_id_is_generated",
            BooleanType(),
        ),
        _field(
            "source_incident_id",
            StringType(),
        ),
        _field(
            "source_row_hash",
            StringType(),
        ),

        # Time
        _field(
            "occurred_at_start",
            TimestampType(),
        ),
        _field(
            "occurred_at_end",
            TimestampType(),
        ),
        _field(
            "reported_at",
            TimestampType(),
        ),
        _field(
            "updated_at",
            TimestampType(),
        ),
        _field(
            "source_timezone",
            StringType(),
        ),
        _field(
            "occurrence_time_precision",
            StringType(),
        ),
        _field(
            "occurrence_time_status",
            StringType(),
        ),
        _field(
            "is_occurrence_interval",
            BooleanType(),
        ),
        _field(
            "occurrence_interval_minutes",
            LongType(),
        ),
        _field(
            "occurrence_year",
            ShortType(),
        ),

        # Source offense taxonomy
        _field("offense_code", StringType()),
        _field("offense_name", StringType()),
        _field(
            "offense_description",
            StringType(),
        ),
        _field(
            "offense_category",
            StringType(),
        ),
        _field(
            "offense_subcategory",
            StringType(),
        ),
        _field(
            "offense_group",
            StringType(),
        ),
        _field(
            "crime_against_category",
            StringType(),
        ),
        _field(
            "attempt_status",
            StringType(),
        ),
        _field("offense_count", LongType()),

        # Cross-city taxonomy
        _field(
            "canonical_offense_category",
            StringType(),
        ),
        _field(
            "canonical_offense_subcategory",
            StringType(),
        ),

        # Disposition and attributes
        _field("resolution", StringType()),
        _field("is_arrest", BooleanType()),
        _field("is_domestic", BooleanType()),
        _field("is_shooting", BooleanType()),
        _field("is_hate_crime", BooleanType()),
        _field(
            "is_gang_related",
            BooleanType(),
        ),
        _field(
            "is_drug_related",
            BooleanType(),
        ),
        _field(
            "is_family_offense",
            BooleanType(),
        ),
        _field("weapon", StringType()),

        # Coordinates
        _field("latitude", DoubleType()),
        _field("longitude", DoubleType()),
        _field(
            "alternate_latitude",
            DoubleType(),
        ),
        _field(
            "alternate_longitude",
            DoubleType(),
        ),
        _field(
            "source_x_coordinate",
            DoubleType(),
        ),
        _field(
            "source_y_coordinate",
            DoubleType(),
        ),
        _field("source_crs", StringType()),
        _field("target_crs", StringType()),
        _field(
            "coordinate_method",
            StringType(),
        ),
        _field(
            "coordinate_precision",
            StringType(),
        ),
        _field(
            "coordinate_validation_status",
            StringType(),
        ),
        _field(
            "coordinate_error_meters",
            DoubleType(),
        ),
        _field(
            "is_coordinate_within_city_bounds",
            BooleanType(),
        ),

        # Human-readable location
        _field("address", StringType()),
        _field(
            "block_address",
            StringType(),
        ),
        _field("intersection", StringType()),
        _field("premise_type", StringType()),
        _field(
            "location_description",
            StringType(),
        ),
        _field("city", StringType()),
        _field("state", StringType()),
        _field("postal_code", StringType()),

        # Administrative geography
        _field("neighborhood", StringType()),
        _field("district", StringType()),
        _field("precinct", StringType()),
        _field("beat", StringType()),
        _field("sector", StringType()),
        _field(
            "reporting_area",
            StringType(),
        ),
        _field("ward", StringType()),
        _field(
            "council_district",
            StringType(),
        ),
        _field(
            "census_tract",
            StringType(),
        ),
        _field(
            "census_block",
            StringType(),
        ),

        # Operational attributes
        _field("shift", StringType()),
        _field("jurisdiction", StringType()),

        # Lineage
        _field("source_file", StringType()),
        _field("source_url", StringType()),
        _field(
            "downloaded_at_utc",
            TimestampType(),
        ),
        _field(
            "ingested_at_utc",
            TimestampType(),
        ),
        _field(
            "canonicalized_at_utc",
            TimestampType(),
        ),
        _field(
            "canonical_schema_version",
            StringType(),
        ),
        _field(
            "key_definition_version",
            StringType(),
        ),
        _field(
            "offense_taxonomy_version",
            StringType(),
        ),
    ]
)


CANONICAL_COLUMNS = tuple(
    field.name
    for field in CANONICAL_CRIME_SCHEMA.fields
)


CANONICAL_NON_NULL_COLUMNS = frozenset(
    {
        "canonical_record_id",
        "source_city",
        "source_record_id",
        "source_record_id_is_generated",
        "occurred_at_start",
        "occurrence_year",
        "offense_count",
        "canonicalized_at_utc",
        "canonical_schema_version",
        "key_definition_version",
        "offense_taxonomy_version",
    }
)


def complete_canonical_schema(
    dataframe: DataFrame,
) -> DataFrame:
    """Add typed null columns, cast fields, and enforce order."""

    source_columns = set(
        dataframe.columns
    )

    expressions = []

    for field in CANONICAL_CRIME_SCHEMA.fields:
        if field.name in source_columns:
            expression = (
                F.col(field.name)
                .cast(field.dataType)
                .alias(field.name)
            )
        else:
            expression = (
                F.lit(None)
                .cast(field.dataType)
                .alias(field.name)
            )

        expressions.append(expression)

    return dataframe.select(
        *expressions
    )


def validate_canonical_schema(
    dataframe: DataFrame,
) -> None:
    expected = [
        (
            field.name,
            field.dataType.simpleString(),
        )
        for field
        in CANONICAL_CRIME_SCHEMA.fields
    ]

    actual = [
        (
            field.name,
            field.dataType.simpleString(),
        )
        for field
        in dataframe.schema.fields
    ]

    if actual != expected:
        raise ValueError(
            "Canonical schema mismatch.\n"
            f"Expected: {expected}\n"
            f"Actual: {actual}"
        )