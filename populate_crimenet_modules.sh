#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
PKG_DIR="${ROOT}/src/crimenet"

if [[ ! -d "${ROOT}" ]]; then
  echo "Project root does not exist: ${ROOT}" >&2
  exit 1
fi

mkdir -p \
  "${PKG_DIR}/ingestion" \
  "${PKG_DIR}/transforms" \
  "${PKG_DIR}/contracts" \
  "${PKG_DIR}/config" \
  "${PKG_DIR}/jobs" \
  "${PKG_DIR}/quality" \
  "${ROOT}/tests/unit"

for package_dir in \
  "${PKG_DIR}" \
  "${PKG_DIR}/ingestion" \
  "${PKG_DIR}/transforms" \
  "${PKG_DIR}/contracts" \
  "${PKG_DIR}/config" \
  "${PKG_DIR}/jobs" \
  "${PKG_DIR}/quality"
do
  touch "${package_dir}/__init__.py"
done

cat > "${PKG_DIR}/ingestion/column_names.py" <<'PY'
"""Column-name normalization for source datasets."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping

from pyspark.sql import DataFrame


def normalize_column_name(name: str) -> str:
    """Convert a source column name to a Delta-safe snake_case identifier."""
    normalized = (
        name.replace("\ufeff", "")
        .replace("\xa0", " ")
    )
    normalized = unicodedata.normalize("NFKD", normalized).lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")

    if not normalized:
        normalized = "unnamed_column"

    if normalized[0].isdigit():
        normalized = f"column_{normalized}"

    return normalized


def normalized_column_names(
    column_names: list[str],
    overrides: Mapping[str, str] | None = None,
) -> list[str]:
    """
    Normalize names and deterministically suffix unknown collisions.

    Known source collisions should use explicit overrides so their meaning does
    not depend on source-column order.
    """
    overrides = overrides or {}
    counts: defaultdict[str, int] = defaultdict(int)
    output: list[str] = []

    for original_name in column_names:
        requested_name = overrides.get(
            original_name,
            normalize_column_name(original_name),
        )
        base_name = normalize_column_name(requested_name)
        counts[base_name] += 1
        occurrence = counts[base_name]

        output.append(
            base_name if occurrence == 1 else f"{base_name}_{occurrence}"
        )

    return output


def normalize_column_names(
    dataframe: DataFrame,
    overrides: Mapping[str, str] | None = None,
) -> DataFrame:
    """Return a DataFrame with Delta-safe, collision-free column names."""
    return dataframe.toDF(
        *normalized_column_names(dataframe.columns, overrides=overrides)
    )
PY

cat > "${PKG_DIR}/ingestion/readers.py" <<'PY'
"""Source-specific raw-file readers."""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


def _with_source_file(dataframe: DataFrame) -> DataFrame:
    return dataframe.select(
        "*",
        F.col("_metadata.file_path").alias("_source_file"),
    )


def read_dallas_raw(
    spark: SparkSession,
    input_path: str,
) -> DataFrame:
    """Read Dallas CSV files, including multiline quoted location fields."""
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
    """Read Houston NIBRS CSV files."""
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
    """Read Fort Worth JSON Lines files."""
    dataframe = (
        spark.read
        .format("json")
        .option("recursiveFileLookup", "true")
        .option("pathGlobFilter", "*.jsonl")
        .load(input_path)
    )
    return _with_source_file(dataframe)
PY

cat > "${PKG_DIR}/ingestion/metadata.py" <<'PY'
"""Reusable ingestion metadata expressions."""

from __future__ import annotations

from collections.abc import Iterable

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

DEFAULT_HASH_EXCLUSIONS = frozenset(
    {
        "source_file",
        "source_row_hash",
        "source_system",
        "ingested_at",
    }
)


def source_row_hash(
    dataframe: DataFrame,
    excluded_columns: Iterable[str] = DEFAULT_HASH_EXCLUSIONS,
) -> Column:
    """
    Build a deterministic SHA-256 hash from normalized source values.

    The source file is excluded so an identical source record keeps the same
    identity if the file is copied into a different landing subdirectory.
    """
    excluded = set(excluded_columns)
    source_columns = [
        F.col(column_name)
        for column_name in sorted(dataframe.columns)
        if column_name not in excluded
    ]

    if not source_columns:
        raise ValueError("Cannot hash a DataFrame with no eligible columns.")

    return F.sha2(
        F.to_json(
            F.struct(*source_columns),
            options={"ignoreNullFields": "false"},
        ),
        256,
    )


def add_ingestion_metadata(
    dataframe: DataFrame,
    source_system: str,
) -> DataFrame:
    """Add stable row identity and operational ingestion metadata."""
    return (
        dataframe
        .withColumn("source_row_hash", source_row_hash(dataframe))
        .withColumn("source_system", F.lit(source_system))
        .withColumn("ingested_at", F.current_timestamp())
    )
PY

cat > "${PKG_DIR}/contracts/silver.py" <<'PY'
"""Canonical Silver offense-level contract."""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql.types import (
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

SILVER_SCHEMA = StructType(
    [
        StructField("source_city", StringType(), False),
        StructField("source_record_id", StringType(), True),
        StructField("source_incident_id", StringType(), True),
        StructField("offense_code", StringType(), True),
        StructField("offense_name", StringType(), True),
        StructField("offense_description", StringType(), True),
        StructField("occurred_at", TimestampType(), True),
        StructField("reported_at", TimestampType(), True),
        StructField("updated_at", TimestampType(), True),
        StructField("offense_count", LongType(), True),
        StructField("address", StringType(), True),
        StructField("city", StringType(), True),
        StructField("state", StringType(), True),
        StructField("postal_code", StringType(), True),
        StructField("beat", StringType(), True),
        StructField("premise_type", StringType(), True),
        StructField("latitude", DoubleType(), True),
        StructField("longitude", DoubleType(), True),
        StructField("alternate_latitude", DoubleType(), True),
        StructField("alternate_longitude", DoubleType(), True),
        StructField("source_x_coordinate", DoubleType(), True),
        StructField("source_y_coordinate", DoubleType(), True),
        StructField("source_file", StringType(), True),
        StructField("source_row_hash", StringType(), False),
    ]
)

SILVER_COLUMNS = tuple(field.name for field in SILVER_SCHEMA.fields)


def assert_silver_contract(dataframe: DataFrame) -> None:
    """Fail early when a source transform violates the canonical contract."""
    actual_fields = dataframe.schema.fields
    expected_fields = SILVER_SCHEMA.fields

    if len(actual_fields) != len(expected_fields):
        raise ValueError(
            "Silver schema has the wrong number of columns: "
            f"expected={len(expected_fields)}, actual={len(actual_fields)}"
        )

    mismatches: list[str] = []

    for expected, actual in zip(expected_fields, actual_fields):
        if expected.name != actual.name or expected.dataType != actual.dataType:
            mismatches.append(
                f"expected {expected.name}:{expected.dataType.simpleString()}, "
                f"got {actual.name}:{actual.dataType.simpleString()}"
            )

    if mismatches:
        raise ValueError(
            "Silver schema contract mismatch:\n- " + "\n- ".join(mismatches)
        )
PY

cat > "${PKG_DIR}/transforms/common.py" <<'PY'
"""Common PySpark expressions used by source transformations."""

from __future__ import annotations

from pyspark.sql import Column
from pyspark.sql import functions as F


def null_string() -> Column:
    return F.lit(None).cast("string")


def null_long() -> Column:
    return F.lit(None).cast("long")


def null_double() -> Column:
    return F.lit(None).cast("double")


def null_timestamp() -> Column:
    return F.lit(None).cast("timestamp")


def try_cast(column_name: str, data_type: str) -> Column:
    """ANSI-safe cast for a trusted internal column name."""
    return F.expr(f"try_cast(`{column_name}` AS {data_type})")


def timestamp_millis(column_name: str) -> Column:
    """Convert a possibly string/long epoch-millisecond field to timestamp."""
    return F.expr(
        f"timestamp_millis(try_cast(`{column_name}` AS BIGINT))"
    )


def trimmed_address(*column_names: str) -> Column:
    """Join nullable address components with one space."""
    return F.trim(
        F.concat_ws(
            " ",
            *(F.col(column_name) for column_name in column_names),
        )
    )
PY

cat > "${PKG_DIR}/transforms/dallas.py" <<'PY'
"""Dallas Bronze-to-canonical-Silver transformation."""

from __future__ import annotations

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from crimenet.contracts.silver import SILVER_COLUMNS, assert_silver_contract
from crimenet.transforms.common import null_double, try_cast

_COORDINATE_PATTERN = (
    r"\(\s*"
    r"(-?\d+(?:\.\d+)?)"
    r"\s*,\s*"
    r"(-?\d+(?:\.\d+)?)"
    r"\s*\)"
)


def _source_timestamp(column_name: str) -> Column:
    """Parse Dallas timestamps with optional fractional seconds."""
    return F.coalesce(
        F.expr(
            f"try_to_timestamp(`{column_name}`, "
            "'yyyy-MM-dd HH:mm:ss.SSSSSSS')"
        ),
        F.expr(
            f"try_to_timestamp(`{column_name}`, "
            "'yyyy-MM-dd HH:mm:ss.SSSSSS')"
        ),
        F.expr(
            f"try_to_timestamp(`{column_name}`, "
            "'yyyy-MM-dd HH:mm:ss')"
        ),
    )


def _occurrence_timestamp() -> Column:
    return F.expr(
        "try_to_timestamp("
        "concat(substring(`date1_of_occurrence`, 1, 10), "
        "' ', `time1_of_occurrence`), "
        "'yyyy-MM-dd HH:mm'"
        ")"
    )


def to_canonical(dataframe: DataFrame) -> DataFrame:
    latitude_text = F.regexp_extract(
        F.col("location1"),
        _COORDINATE_PATTERN,
        1,
    )
    longitude_text = F.regexp_extract(
        F.col("location1"),
        _COORDINATE_PATTERN,
        2,
    )

    result = dataframe.select(
        F.lit("dallas").alias("source_city"),
        F.col("service_number_id")
        .cast("string")
        .alias("source_record_id"),
        F.col("incident_number_w_year")
        .cast("string")
        .alias("source_incident_id"),
        F.col("nibrs_code").cast("string").alias("offense_code"),
        F.coalesce(
            F.col("nibrs_crime"),
            F.col("ucr_offense_name"),
            F.col("type_of_incident"),
        )
        .cast("string")
        .alias("offense_name"),
        F.coalesce(
            F.col("ucr_offense_description"),
            F.col("type_of_incident"),
        )
        .cast("string")
        .alias("offense_description"),
        _occurrence_timestamp().alias("occurred_at"),
        _source_timestamp("date_of_report").alias("reported_at"),
        _source_timestamp("update_date").alias("updated_at"),
        F.lit(1).cast("long").alias("offense_count"),
        F.col("incident_address").cast("string").alias("address"),
        F.col("city").cast("string").alias("city"),
        F.col("state").cast("string").alias("state"),
        F.col("zip_code").cast("string").alias("postal_code"),
        F.col("beat").cast("string").alias("beat"),
        F.col("type_location").cast("string").alias("premise_type"),
        F.when(
            latitude_text != "",
            latitude_text.cast("double"),
        ).alias("latitude"),
        F.when(
            longitude_text != "",
            longitude_text.cast("double"),
        ).alias("longitude"),
        null_double().alias("alternate_latitude"),
        null_double().alias("alternate_longitude"),
        try_cast("x_coordinate", "double").alias("source_x_coordinate"),
        try_cast("y_cordinate", "double").alias("source_y_coordinate"),
        F.col("source_file").cast("string").alias("source_file"),
        F.col("source_row_hash").cast("string").alias("source_row_hash"),
    ).select(*SILVER_COLUMNS)

    assert_silver_contract(result)
    return result
PY

cat > "${PKG_DIR}/transforms/houston.py" <<'PY'
"""Houston Bronze-to-canonical-Silver transformation."""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from crimenet.contracts.silver import SILVER_COLUMNS, assert_silver_contract
from crimenet.transforms.common import (
    null_double,
    null_timestamp,
    trimmed_address,
    try_cast,
)


def to_canonical(dataframe: DataFrame) -> DataFrame:
    result = dataframe.select(
        F.lit("houston").alias("source_city"),
        F.col("source_row_hash").cast("string").alias("source_record_id"),
        F.col("incident").cast("string").alias("source_incident_id"),
        F.col("nibrsclass").cast("string").alias("offense_code"),
        F.col("nibrsdescription").cast("string").alias("offense_name"),
        F.col("nibrsdescription")
        .cast("string")
        .alias("offense_description"),
        F.expr(
            "try_to_timestamp("
            "concat(`rmsoccurrencedate`, ' ', "
            "lpad(`rmsoccurrencehour`, 2, '0')), "
            "'M/d/yyyy HH'"
            ")"
        ).alias("occurred_at"),
        null_timestamp().alias("reported_at"),
        null_timestamp().alias("updated_at"),
        try_cast("offensecount", "long").alias("offense_count"),
        trimmed_address(
            "streetno",
            "streetname",
            "streettype",
            "suffix",
        ).alias("address"),
        F.col("city").cast("string").alias("city"),
        F.lit("TX").cast("string").alias("state"),
        F.col("zipcode").cast("string").alias("postal_code"),
        F.col("beat").cast("string").alias("beat"),
        F.col("premise").cast("string").alias("premise_type"),
        try_cast("maplatitude", "double").alias("latitude"),
        try_cast("maplongitude", "double").alias("longitude"),
        null_double().alias("alternate_latitude"),
        null_double().alias("alternate_longitude"),
        null_double().alias("source_x_coordinate"),
        null_double().alias("source_y_coordinate"),
        F.col("source_file").cast("string").alias("source_file"),
        F.col("source_row_hash").cast("string").alias("source_row_hash"),
    ).select(*SILVER_COLUMNS)

    assert_silver_contract(result)
    return result
PY

cat > "${PKG_DIR}/transforms/fort_worth.py" <<'PY'
"""Fort Worth Bronze-to-canonical-Silver transformation."""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from crimenet.contracts.silver import SILVER_COLUMNS, assert_silver_contract
from crimenet.transforms.common import (
    null_string,
    timestamp_millis,
)


def to_canonical(dataframe: DataFrame) -> DataFrame:
    result = dataframe.select(
        F.lit("fort_worth").alias("source_city"),
        F.coalesce(
            F.col("case_no_offense").cast("string"),
            F.col("objectid").cast("string"),
        ).alias("source_record_id"),
        F.col("case_no").cast("string").alias("source_incident_id"),
        F.col("offense").cast("string").alias("offense_code"),
        F.col("nature_of_call").cast("string").alias("offense_name"),
        F.col("offense_desc")
        .cast("string")
        .alias("offense_description"),
        timestamp_millis("from_date").alias("occurred_at"),
        timestamp_millis("reported_date").alias("reported_at"),
        timestamp_millis("lastupdated").alias("updated_at"),
        F.lit(1).cast("long").alias("offense_count"),
        F.coalesce(
            F.col("address"),
            F.col("block_address"),
        )
        .cast("string")
        .alias("address"),
        F.col("city").cast("string").alias("city"),
        F.col("state").cast("string").alias("state"),
        null_string().alias("postal_code"),
        F.col("beat").cast("string").alias("beat"),
        F.col("locationtypedescription")
        .cast("string")
        .alias("premise_type"),
        F.col("latitude").cast("double").alias("latitude"),
        F.col("longitude").cast("double").alias("longitude"),
        F.col("alternate_latitude")
        .cast("double")
        .alias("alternate_latitude"),
        F.col("alternate_longitude")
        .cast("double")
        .alias("alternate_longitude"),
        F.col("x_coordinate")
        .cast("double")
        .alias("source_x_coordinate"),
        F.col("y_coordinate")
        .cast("double")
        .alias("source_y_coordinate"),
        F.col("source_file").cast("string").alias("source_file"),
        F.col("source_row_hash").cast("string").alias("source_row_hash"),
    ).select(*SILVER_COLUMNS)

    assert_silver_contract(result)
    return result
PY

cat > "${PKG_DIR}/transforms/canonical.py" <<'PY'
"""Composition of all source-specific canonical transformations."""

from __future__ import annotations

from pyspark.sql import DataFrame

from crimenet.contracts.silver import assert_silver_contract
from crimenet.transforms.dallas import to_canonical as transform_dallas
from crimenet.transforms.fort_worth import (
    to_canonical as transform_fort_worth,
)
from crimenet.transforms.houston import to_canonical as transform_houston


def build_crime_offenses(
    dallas_bronze: DataFrame,
    houston_bronze: DataFrame,
    fort_worth_bronze: DataFrame,
) -> DataFrame:
    """Create the unified offense-grain Silver DataFrame."""
    result = (
        transform_dallas(dallas_bronze)
        .unionByName(transform_houston(houston_bronze))
        .unionByName(transform_fort_worth(fort_worth_bronze))
    )

    assert_silver_contract(result)
    return result
PY

cat > "${PKG_DIR}/config/resources.py" <<'PY'
"""Central construction of Unity Catalog object names."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CrimeNetTables:
    catalog: str
    bronze_schema: str = "bronze"
    silver_schema: str = "silver"

    @property
    def dallas_bronze(self) -> str:
        return f"{self.catalog}.{self.bronze_schema}.dallas_crime"

    @property
    def houston_bronze(self) -> str:
        return f"{self.catalog}.{self.bronze_schema}.houston_crime"

    @property
    def fort_worth_bronze(self) -> str:
        return f"{self.catalog}.{self.bronze_schema}.fort_worth_crime"

    @property
    def crime_offenses_silver(self) -> str:
        return f"{self.catalog}.{self.silver_schema}.crime_offenses"

    def bronze_for_city(self, city: str) -> str:
        table_by_city = {
            "dallas": self.dallas_bronze,
            "houston": self.houston_bronze,
            "fort_worth": self.fort_worth_bronze,
        }

        try:
            return table_by_city[city]
        except KeyError as exc:
            raise ValueError(f"Unsupported city: {city!r}") from exc
PY

cat > "${PKG_DIR}/jobs/bronze_ingestion.py" <<'PY'
"""Python-wheel entry point for one city's Bronze ingestion."""

from __future__ import annotations

import argparse
from collections.abc import Callable

from pyspark.sql import DataFrame, SparkSession

from crimenet.config.resources import CrimeNetTables
from crimenet.ingestion.column_names import normalize_column_names
from crimenet.ingestion.metadata import add_ingestion_metadata
from crimenet.ingestion.readers import (
    read_dallas_raw,
    read_fort_worth_raw,
    read_houston_raw,
)

Reader = Callable[[SparkSession, str], DataFrame]

READERS: dict[str, Reader] = {
    "dallas": read_dallas_raw,
    "houston": read_houston_raw,
    "fort_worth": read_fort_worth_raw,
}

COLUMN_OVERRIDES: dict[str, dict[str, str]] = {
    "fort_worth": {
        "Latitude": "latitude",
        "latitude": "latitude",
        "_Latitude": "alternate_latitude",
        "_latitude": "alternate_latitude",
        "Longitude": "longitude",
        "longitude": "longitude",
        "_Longitude": "alternate_longitude",
        "_longitude": "alternate_longitude",
    }
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--bronze-schema", default="bronze")
    parser.add_argument(
        "--city",
        required=True,
        choices=tuple(READERS),
    )
    parser.add_argument("--input-path", required=True)
    parser.add_argument(
        "--write-mode",
        default="overwrite",
        choices=("overwrite", "append"),
    )
    return parser.parse_args()


def run(
    spark: SparkSession,
    *,
    catalog: str,
    bronze_schema: str,
    city: str,
    input_path: str,
    write_mode: str,
) -> None:
    tables = CrimeNetTables(
        catalog=catalog,
        bronze_schema=bronze_schema,
    )

    raw_dataframe = READERS[city](spark, input_path)
    normalized_dataframe = normalize_column_names(
        raw_dataframe,
        overrides=COLUMN_OVERRIDES.get(city),
    )
    bronze_dataframe = add_ingestion_metadata(
        normalized_dataframe,
        source_system=city,
    )

    writer = (
        bronze_dataframe.write
        .format("delta")
        .mode(write_mode)
    )

    if write_mode == "overwrite":
        writer = writer.option("overwriteSchema", "true")

    writer.saveAsTable(tables.bronze_for_city(city))


def main() -> None:
    args = parse_args()
    spark = SparkSession.getActiveSession() or (
        SparkSession.builder.getOrCreate()
    )

    run(
        spark,
        catalog=args.catalog,
        bronze_schema=args.bronze_schema,
        city=args.city,
        input_path=args.input_path,
        write_mode=args.write_mode,
    )


if __name__ == "__main__":
    main()
PY

cat > "${PKG_DIR}/jobs/silver_transform.py" <<'PY'
"""Python-wheel entry point for canonical Silver transformation."""

from __future__ import annotations

import argparse

from pyspark.sql import SparkSession

from crimenet.config.resources import CrimeNetTables
from crimenet.transforms.canonical import build_crime_offenses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--bronze-schema", default="bronze")
    parser.add_argument("--silver-schema", default="silver")
    # Accepted for compatibility with the bundle task. Quarantine writes can
    # use this in the next iteration without changing the task interface.
    parser.add_argument(
        "--data-quality-schema",
        default="data_quality",
    )
    parser.add_argument(
        "--write-mode",
        default="overwrite",
        choices=("overwrite", "append"),
    )
    return parser.parse_args()


def run(
    spark: SparkSession,
    *,
    catalog: str,
    bronze_schema: str,
    silver_schema: str,
    write_mode: str,
) -> None:
    tables = CrimeNetTables(
        catalog=catalog,
        bronze_schema=bronze_schema,
        silver_schema=silver_schema,
    )

    silver_dataframe = build_crime_offenses(
        dallas_bronze=spark.table(tables.dallas_bronze),
        houston_bronze=spark.table(tables.houston_bronze),
        fort_worth_bronze=spark.table(tables.fort_worth_bronze),
    )

    writer = (
        silver_dataframe.write
        .format("delta")
        .mode(write_mode)
    )

    if write_mode == "overwrite":
        writer = writer.option("overwriteSchema", "true")

    writer.saveAsTable(tables.crime_offenses_silver)


def main() -> None:
    args = parse_args()
    spark = SparkSession.getActiveSession() or (
        SparkSession.builder.getOrCreate()
    )

    run(
        spark,
        catalog=args.catalog,
        bronze_schema=args.bronze_schema,
        silver_schema=args.silver_schema,
        write_mode=args.write_mode,
    )


if __name__ == "__main__":
    main()
PY

cat > "${PKG_DIR}/quality/rules.py" <<'PY'
"""Reusable canonical Silver quality-rule expressions."""

from __future__ import annotations

from pyspark.sql import Column
from pyspark.sql import functions as F


def has_source_identity() -> Column:
    return (
        F.col("source_city").isNotNull()
        & F.col("source_record_id").isNotNull()
    )


def coordinates_are_valid() -> Column:
    return (
        (
            F.col("latitude").isNull()
            & F.col("longitude").isNull()
        )
        |
        (
            F.col("latitude").between(-90.0, 90.0)
            & F.col("longitude").between(-180.0, 180.0)
        )
    )


def occurred_at_is_valid() -> Column:
    return F.col("occurred_at").isNotNull()
PY

cat > "${ROOT}/tests/unit/test_column_names.py" <<'PY'
from crimenet.ingestion.column_names import (
    normalize_column_name,
    normalized_column_names,
)


def test_normalize_column_name() -> None:
    assert normalize_column_name("Call (911) Problem") == "call_911_problem"
    assert normalize_column_name("Date/Time") == "date_time"
    assert normalize_column_name("\ufeff ZIP Code ") == "zip_code"


def test_collision_suffixes_are_deterministic() -> None:
    assert normalized_column_names(
        ["Latitude", "_latitude", "Longitude", "_longitude"],
        overrides={
            "Latitude": "latitude",
            "_latitude": "alternate_latitude",
            "Longitude": "longitude",
            "_longitude": "alternate_longitude",
        },
    ) == [
        "latitude",
        "alternate_latitude",
        "longitude",
        "alternate_longitude",
    ]
PY

cat > "${PKG_DIR}/transforms/__init__.py" <<'PY'
from crimenet.transforms.canonical import build_crime_offenses

__all__ = ["build_crime_offenses"]
PY

cat > "${PKG_DIR}/ingestion/__init__.py" <<'PY'
from crimenet.ingestion.column_names import normalize_column_names
from crimenet.ingestion.metadata import add_ingestion_metadata

__all__ = [
    "add_ingestion_metadata",
    "normalize_column_names",
]
PY

echo "Populated reusable CrimeNet modules under ${ROOT}."
echo
echo "Run:"
echo "  uv run pytest tests/unit/test_column_names.py"
echo "  uv build"
echo "  databricks bundle validate -t dev"
