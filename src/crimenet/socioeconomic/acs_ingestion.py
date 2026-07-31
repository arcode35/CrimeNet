"""PySpark landing and validation for ACS tract partitions."""

from __future__ import annotations

import json
import os
import shutil
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    IntegerType,
    StringType,
    StructField,
    StructType,
)
from requests import Session

from crimenet.observability.logging import (
    get_logger,
)
from crimenet.socioeconomic.acs_client import (
    ACS5_DATASET,
    ACS5_TRACT_VARIABLES,
    METRO_GEOGRAPHIES,
    MetroGeography,
    fetch_acs5_tracts,
    utc_now,
)


LOGGER = get_logger(__name__)


ACS5_TRACT_RAW_SCHEMA = StructType(
    [
        StructField(
            "NAME",
            StringType(),
            True,
        ),
        *[
            StructField(
                variable,
                StringType(),
                True,
            )
            for variable in ACS5_TRACT_VARIABLES
        ],
        StructField(
            "state",
            StringType(),
            False,
        ),
        StructField(
            "county",
            StringType(),
            False,
        ),
        StructField(
            "tract",
            StringType(),
            False,
        ),
        StructField(
            "geoid",
            StringType(),
            False,
        ),
        StructField(
            "acs_vintage",
            IntegerType(),
            False,
        ),
        StructField(
            "period_start_year",
            IntegerType(),
            False,
        ),
        StructField(
            "period_end_year",
            IntegerType(),
            False,
        ),
        StructField(
            "dataset",
            StringType(),
            False,
        ),
        StructField(
            "geography_type",
            StringType(),
            False,
        ),
        StructField(
            "retrieved_at",
            StringType(),
            False,
        ),
    ]
)


@dataclass(frozen=True)
class AcsAcquisitionSummary:
    """Summary of one ACS landing execution."""

    examined: int
    attempted: int
    downloaded: int
    cached: int
    failed: int
    failures: tuple[str, ...]


def get_partition_directory(
    output_root: Path,
    *,
    metro: MetroGeography,
    vintage: int,
    county_fips: str,
) -> Path:
    """Return the deterministic output directory for one partition."""
    return (
        output_root
        / f"metro={metro.key}"
        / f"vintage={vintage}"
        / f"state_fips={metro.state_fips}"
        / f"county_fips={county_fips}"
    )


def records_to_dataframe(
    spark: SparkSession,
    records: Sequence[dict[str, Any]],
) -> DataFrame:
    """Create an explicitly typed Spark DataFrame."""
    if not records:
        raise ValueError(
            "Cannot create an ACS DataFrame from "
            "an empty record collection"
        )

    return spark.createDataFrame(
        list(records),
        schema=ACS5_TRACT_RAW_SCHEMA,
    )


def _find_parquet_part_file(
    directory: Path,
) -> Path:
    part_files = sorted(
        path
        for path in directory.iterdir()
        if (
            path.is_file()
            and path.name.startswith("part-")
            and path.suffix == ".parquet"
        )
    )

    if len(part_files) != 1:
        raise RuntimeError(
            "Expected exactly one Spark Parquet "
            f"part file, found {len(part_files)}: "
            f"path={directory}"
        )

    return part_files[0]


def write_parquet_atomic(
    spark: SparkSession,
    records: Sequence[dict[str, Any]],
    destination: Path,
) -> int:
    """Write one county/vintage partition with PySpark.

    Spark writes to a temporary directory. After validation, the single
    Parquet part file is atomically moved into the final raw-layer path.
    """
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_directory = (
        destination.parent
        / (
            f".{destination.stem}."
            f"{uuid4().hex}.spark"
        )
    )

    dataframe = records_to_dataframe(
        spark,
        records,
    )

    expected_rows = len(records)

    try:
        (
            dataframe
            .coalesce(1)
            .write
            .format("parquet")
            .mode("overwrite")
            .option(
                "compression",
                "zstd",
            )
            .save(
                str(temporary_directory)
            )
        )

        actual_rows = (
            spark.read
            .parquet(
                str(temporary_directory)
            )
            .count()
        )

        if actual_rows != expected_rows:
            raise RuntimeError(
                "Temporary ACS Parquet validation failed: "
                f"expected_rows={expected_rows}, "
                f"actual_rows={actual_rows}, "
                f"path={temporary_directory}"
            )

        part_file = _find_parquet_part_file(
            temporary_directory
        )

        os.replace(
            part_file,
            destination,
        )

    finally:
        shutil.rmtree(
            temporary_directory,
            ignore_errors=True,
        )

    return expected_rows


def write_json_atomic(
    payload: dict[str, Any],
    destination: Path,
) -> None:
    """Atomically write a JSON partition manifest."""
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = destination.with_name(
        f".{destination.name}.{uuid4().hex}.tmp"
    )

    try:
        with temporary_path.open(
            mode="w",
            encoding="utf-8",
        ) as file:
            json.dump(
                payload,
                file,
                indent=2,
                sort_keys=True,
            )

            file.write("\n")
            file.flush()
            os.fsync(
                file.fileno()
            )

        os.replace(
            temporary_path,
            destination,
        )

    finally:
        temporary_path.unlink(
            missing_ok=True,
        )


def validate_existing_partition(
    spark: SparkSession,
    *,
    parquet_path: Path,
    manifest_path: Path,
    metro: MetroGeography,
    vintage: int,
    county_fips: str,
) -> bool:
    """Return whether an existing ACS partition is complete and valid."""
    if not parquet_path.is_file():
        return False

    if not manifest_path.is_file():
        return False

    try:
        manifest = json.loads(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )

        if manifest.get("metro") != metro.key:
            return False

        if manifest.get("dataset") != ACS5_DATASET:
            return False

        if (
            manifest.get("geography_type")
            != "tract"
        ):
            return False

        if int(
            manifest["vintage"]
        ) != vintage:
            return False

        if (
            str(manifest["state_fips"])
            != metro.state_fips
        ):
            return False

        if (
            str(manifest["county_fips"])
            != county_fips
        ):
            return False

        expected_variables = (
            "NAME",
            *ACS5_TRACT_VARIABLES,
        )

        actual_variables = tuple(
            str(variable)
            for variable in manifest.get(
                "variables",
                [],
            )
        )

        if actual_variables != expected_variables:
            return False

        expected_rows = int(
            manifest["record_count"]
        )

        if expected_rows <= 0:
            return False

        existing_dataframe = (
            spark.read
            .parquet(
                str(parquet_path)
            )
        )

        required_columns = set(
            ACS5_TRACT_RAW_SCHEMA.fieldNames()
        )

        if not required_columns.issubset(
            set(existing_dataframe.columns)
        ):
            return False

        metrics = (
            existing_dataframe
            .agg(
                F.count(
                    F.lit(1)
                ).alias(
                    "row_count"
                ),
                F.countDistinct(
                    "geoid"
                ).alias(
                    "distinct_geoids"
                ),
                F.sum(
                    F.when(
                        F.col("geoid").isNull(),
                        F.lit(1),
                    ).otherwise(
                        F.lit(0)
                    )
                ).alias(
                    "null_geoids"
                ),
            )
            .first()
        )

        actual_rows = int(
            metrics["row_count"]
        )

        distinct_geoids = int(
            metrics["distinct_geoids"]
        )

        null_geoids = int(
            metrics["null_geoids"] or 0
        )

        return (
            actual_rows == expected_rows
            and distinct_geoids == actual_rows
            and null_geoids == 0
        )

    except Exception as exc:
        LOGGER.warning(
            "Existing ACS partition failed validation",
            metro=metro.key,
            vintage=vintage,
            state_fips=metro.state_fips,
            county_fips=county_fips,
            parquet_path=str(parquet_path),
            error=str(exc),
        )

        return False


def build_partition_manifest(
    *,
    metro: MetroGeography,
    vintage: int,
    county_fips: str,
    record_count: int,
    parquet_path: Path,
) -> dict[str, Any]:
    """Build the sidecar manifest for one landed partition."""
    return {
        "metro": metro.key,
        "dataset": ACS5_DATASET,
        "geography_type": "tract",
        "vintage": vintage,
        "period_start_year": vintage - 4,
        "period_end_year": vintage,
        "state_fips": metro.state_fips,
        "county_fips": county_fips,
        "record_count": record_count,
        "variables": [
            "NAME",
            *ACS5_TRACT_VARIABLES,
        ],
        "retrieved_at": utc_now(),
        "parquet_file": parquet_path.name,
        "compression": "zstd",
    }


def ingest_acs5_tracts(
    spark: SparkSession,
    *,
    output_root: Path,
    metros: Sequence[str],
    start_vintage: int,
    end_vintage: int,
    api_key: str,
    overwrite: bool,
    pause_seconds: float,
    maximum_partitions: int | None,
    session: Session,
) -> AcsAcquisitionSummary:
    """Acquire all configured ACS county/vintage partitions."""
    normalized_metros = tuple(
        dict.fromkeys(
            metro.strip().lower()
            for metro in metros
            if metro.strip()
        )
    )

    if not normalized_metros:
        raise ValueError(
            "At least one metro is required"
        )

    unsupported_metros = (
        set(normalized_metros)
        - set(METRO_GEOGRAPHIES)
    )

    if unsupported_metros:
        raise ValueError(
            "Unsupported metros: "
            + ", ".join(
                sorted(unsupported_metros)
            )
        )

    examined_count = 0
    attempted_count = 0
    downloaded_count = 0
    cached_count = 0
    failed_count = 0
    failures: list[str] = []

    for metro_key in normalized_metros:
        metro = METRO_GEOGRAPHIES[
            metro_key
        ]

        for vintage in range(
            start_vintage,
            end_vintage + 1,
        ):
            for county_fips in metro.county_fips:
                partition_directory = (
                    get_partition_directory(
                        output_root,
                        metro=metro,
                        vintage=vintage,
                        county_fips=county_fips,
                    )
                )

                parquet_path = (
                    partition_directory
                    / "part-00000.parquet"
                )

                manifest_path = (
                    partition_directory
                    / "_manifest.json"
                )

                if (
                    not overwrite
                    and validate_existing_partition(
                        spark,
                        parquet_path=parquet_path,
                        manifest_path=manifest_path,
                        metro=metro,
                        vintage=vintage,
                        county_fips=county_fips,
                    )
                ):
                    examined_count += 1
                    cached_count += 1

                    LOGGER.info(
                        "Skipping verified ACS partition",
                        metro=metro.key,
                        vintage=vintage,
                        state_fips=metro.state_fips,
                        county_fips=county_fips,
                        parquet_path=str(
                            parquet_path
                        ),
                    )

                    continue

                if (
                    maximum_partitions is not None
                    and attempted_count
                    >= maximum_partitions
                ):
                    return AcsAcquisitionSummary(
                        examined=examined_count,
                        attempted=attempted_count,
                        downloaded=downloaded_count,
                        cached=cached_count,
                        failed=failed_count,
                        failures=tuple(
                            failures
                        ),
                    )

                examined_count += 1
                attempted_count += 1

                try:
                    LOGGER.info(
                        "Fetching ACS tract partition",
                        metro=metro.key,
                        vintage=vintage,
                        state_fips=metro.state_fips,
                        county_fips=county_fips,
                    )

                    records = fetch_acs5_tracts(
                        vintage=vintage,
                        state_fips=(
                            metro.state_fips
                        ),
                        county_fips=(
                            county_fips
                        ),
                        api_key=api_key,
                        session=session,
                    )

                    record_count = (
                        write_parquet_atomic(
                            spark,
                            records,
                            parquet_path,
                        )
                    )

                    manifest = (
                        build_partition_manifest(
                            metro=metro,
                            vintage=vintage,
                            county_fips=(
                                county_fips
                            ),
                            record_count=(
                                record_count
                            ),
                            parquet_path=(
                                parquet_path
                            ),
                        )
                    )

                    write_json_atomic(
                        manifest,
                        manifest_path,
                    )

                    downloaded_count += 1

                    LOGGER.info(
                        "Landed ACS tract partition",
                        metro=metro.key,
                        vintage=vintage,
                        state_fips=metro.state_fips,
                        county_fips=county_fips,
                        record_count=record_count,
                        parquet_path=str(
                            parquet_path
                        ),
                    )

                except Exception as exc:
                    failed_count += 1

                    failure_message = (
                        f"metro={metro.key}, "
                        f"vintage={vintage}, "
                        f"state={metro.state_fips}, "
                        f"county={county_fips}, "
                        f"error={type(exc).__name__}: "
                        f"{exc}"
                    )

                    failures.append(
                        failure_message
                    )

                    LOGGER.exception(
                        "ACS tract partition failed",
                        metro=metro.key,
                        vintage=vintage,
                        state_fips=metro.state_fips,
                        county_fips=county_fips,
                    )

                finally:
                    if pause_seconds > 0:
                        time.sleep(
                            pause_seconds
                        )

    return AcsAcquisitionSummary(
        examined=examined_count,
        attempted=attempted_count,
        downloaded=downloaded_count,
        cached=cached_count,
        failed=failed_count,
        failures=tuple(
            failures
        ),
    )