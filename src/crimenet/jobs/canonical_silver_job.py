"""PySpark entry point for canonical CrimeNet Silver."""

from __future__ import annotations

import argparse
import importlib
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Callable

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
import uuid
from crimenet.canonical.common import (
    LOCAL_TIMEZONES,
)
from crimenet.canonical.deduplication import (
    deduplicate_canonical_records,
)
from crimenet.canonical.definitions import (
    CANONICAL_SCHEMA_VERSION,
    KEY_DEFINITION_VERSION,
    OFFENSE_TAXONOMY_VERSION,
)
from crimenet.canonical.quality import (
    CityBuildMetrics,
    audit_city,
    enforce_hard_gates,
)
from crimenet.canonical.schema import (
    CANONICAL_CRIME_SCHEMA,
    complete_canonical_schema,
    validate_canonical_schema,
)
from crimenet.config.resources import (
    CrimeNetTables,
)
from crimenet.observability.logging import (
    get_logger,
)


LOGGER = get_logger(__name__)


CITY_MODULES = {
    "dallas": "dallas",
    "fort_worth": "fort_worth",
    "new_york": "new_york",
    "chicago": "chicago",
    "san_francisco": "san_francisco",
    "seattle": "seattle",
    "baltimore": "baltimore",
    "washington_dc": "washington_dc",
}


SYSTEM_COLUMNS = {
    "source_city",
    "occurrence_year",
    "canonicalized_at_utc",
    "canonical_schema_version",
    "key_definition_version",
    "offense_taxonomy_version",
}


Adapter = Callable[
    [DataFrame],
    DataFrame,
]
def stage_city_dataframe(
    dataframe: DataFrame,
    *,
    catalog: str,
    silver_schema: str,
    city: str,
) -> str:
    run_suffix = uuid.uuid4().hex[:12]

    staging_table = (
        f"{catalog}."
        f"{silver_schema}."
        f"_canonical_staging_"
        f"{city}_{run_suffix}"
    )

    (
        dataframe.write
        .format("delta")
        .mode("overwrite")
        .saveAsTable(staging_table)
    )

    return staging_table

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--catalog",
        required=True,
    )

    parser.add_argument(
        "--bronze-schema",
        default="bronze",
    )

    parser.add_argument(
        "--silver-schema",
        default="silver",
    )

    parser.add_argument(
        "--minimum-year",
        type=int,
        default=2014,
    )

    parser.add_argument(
        "--city",
        required=True,
        choices=tuple(CITY_MODULES),
    )

    return parser.parse_args()


def load_adapter(
    city: str,
) -> Adapter:
    module_name = (
        "crimenet.canonical.cities."
        f"{CITY_MODULES[city]}"
    )

    module = importlib.import_module(
        module_name
    )

    builder = getattr(
        module,
        "build_canonical",
        None,
    )

    if builder is None or not callable(
        builder
    ):
        raise RuntimeError(
            f"{module_name} must expose "
            "build_canonical(bronze)"
        )

    return builder


def add_system_columns(
    dataframe: DataFrame,
    *,
    city: str,
    canonicalized_at: datetime,
) -> DataFrame:
    if (
        "occurred_at_start"
        not in dataframe.columns
    ):
        raise ValueError(
            f"{city} did not produce "
            "occurred_at_start"
        )

    local_timestamp = (
        F.from_utc_timestamp(
            F.col("occurred_at_start"),
            LOCAL_TIMEZONES[city],
        )
    )

    return (
        dataframe
        .withColumn(
            "source_city",
            F.lit(city),
        )
        .withColumn(
            "occurrence_year",
            F.year(local_timestamp)
            .cast("smallint"),
        )
        .withColumn(
            "canonicalized_at_utc",
            F.lit(canonicalized_at)
            .cast("timestamp"),
        )
        .withColumn(
            "canonical_schema_version",
            F.lit(
                CANONICAL_SCHEMA_VERSION
            ),
        )
        .withColumn(
            "key_definition_version",
            F.lit(
                KEY_DEFINITION_VERSION
            ),
        )
        .withColumn(
            "offense_taxonomy_version",
            F.lit(
                OFFENSE_TAXONOMY_VERSION
            ),
        )
    )


def build_city(
    spark: SparkSession,
    *,
    tables: CrimeNetTables,
    city: str,
    minimum_year: int,
    canonicalized_at: datetime,
) -> DataFrame:
    bronze_table = (
        tables.bronze_for_source(city)
    )

    LOGGER.info(
        "Reading Bronze city",
        city=city,
        bronze_table=bronze_table,
    )

    bronze = spark.read.table(
        bronze_table
    )

    adapter = load_adapter(city)

    city_frame = adapter(bronze)

    forbidden = (
        set(city_frame.columns)
        & SYSTEM_COLUMNS
    )

    if forbidden:
        raise ValueError(
            f"{city} adapter produced "
            "or overwrote system fields: "
            f"{sorted(forbidden)}"
        )

    canonical = add_system_columns(
        city_frame,
        city=city,
        canonicalized_at=(
            canonicalized_at
        ),
    )

    canonical = (
        complete_canonical_schema(
            canonical
        )
    )

    # The analytical cutoff belongs in Silver,
    # not acquisition or Bronze.
    canonical = canonical.filter(
        F.col("occurrence_year")
        >= F.lit(minimum_year)
    )

    canonical = (
        deduplicate_canonical_records(
            canonical
        )
    )

    validate_canonical_schema(
        canonical
    )

    return canonical


def ensure_target_table(
    spark: SparkSession,
    *,
    target_table: str,
) -> None:
    if spark.catalog.tableExists(
        target_table
    ):
        return

    empty = spark.createDataFrame(
        [],
        schema=CANONICAL_CRIME_SCHEMA,
    )

    (
        empty.write
        .format("delta")
        .mode("overwrite")
        .partitionBy(
            "source_city",
            "occurrence_year",
        )
        .saveAsTable(target_table)
    )


def replace_city(
    dataframe: DataFrame,
    *,
    target_table: str,
    city: str,
) -> None:
    escaped_city = city.replace(
        "'",
        "''",
    )

    (
        dataframe.write
        .format("delta")
        .mode("overwrite")
        .option(
            "replaceWhere",
            (
                "source_city = "
                f"'{escaped_city}'"
            ),
        )
        .saveAsTable(target_table)
    )

def drop_staging_table(
    spark: SparkSession,
    *,
    staging_table: str,
) -> None:
    """Remove a temporary canonical staging table."""

    escaped_identifier = ".".join(
        f"`{part.replace('`', '``')}`"
        for part in staging_table.split(".")
    )

    spark.sql(
        f"DROP TABLE IF EXISTS "
        f"{escaped_identifier}"
    )

    LOGGER.info(
        "Dropped canonical staging table",
        staging_table=staging_table,
    )

def run(
    spark: SparkSession,
    *,
    catalog: str,
    bronze_schema: str,
    silver_schema: str,
    minimum_year: int,
    city: str,
) -> CityBuildMetrics:
    # Local strings are initially parsed as wall-clock
    # values under UTC and then shifted from each city's
    # source timezone into UTC.
    spark.conf.set(
        "spark.sql.session.timeZone",
        "UTC",
    )

    # Canonical contract casts must fail rather than
    # silently discard malformed adapter values.
    spark.conf.set(
        "spark.sql.ansi.enabled",
        "true",
    )

    tables = CrimeNetTables(
        catalog=catalog,
        bronze_schema=bronze_schema,
        silver_schema=silver_schema,
    )

    target_table = (
        tables.crime_offenses_silver
    )

    ensure_target_table(
        spark,
        target_table=target_table,
    )

    canonicalized_at = datetime.now(
        UTC
    )

    metrics: list[
        CityBuildMetrics
    ] = []

    LOGGER.info(
        "Building canonical city",
        city=city,
        minimum_year=minimum_year,
    )

    dataframe = build_city(
        spark,
        tables=tables,
        city=city,
        minimum_year=minimum_year,
        canonicalized_at=canonicalized_at,
    )

    staging_table = (
        f"{catalog}."
        f"{silver_schema}."
        f"_canonical_city_{city}"
    )

    (
        dataframe.write
        .format("delta")
        .mode("overwrite")
        .option(
            "overwriteSchema",
            "true",
        )
        .saveAsTable(staging_table)
    )

    staged_dataframe = spark.read.table(
        staging_table
    )

    city_metrics = audit_city(
        staged_dataframe,
        city=city,
    )

    enforce_hard_gates(
        city_metrics
    )

    LOGGER.info(
        "Canonical city staged",
        staging_table=staging_table,
        **asdict(city_metrics),
    )

    return city_metrics

def main() -> None:
    args = parse_args()

    spark = (
        SparkSession.getActiveSession()
        or SparkSession.builder.getOrCreate()
    )

    try:
        metric = run(
            spark,
            catalog=args.catalog,
            bronze_schema=args.bronze_schema,
            silver_schema=args.silver_schema,
            minimum_year=args.minimum_year,
            city=args.city,
        )

    except Exception:
        LOGGER.exception(
            "Canonical Silver city failed",
            city=args.city,
        )
        raise

    LOGGER.info(
        "Canonical Silver city completed",
        city=args.city,
        rows=metric.rows,
    )


if __name__ == "__main__":
    main()