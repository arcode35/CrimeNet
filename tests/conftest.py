from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from pyspark.sql import DataFrame, SparkSession

from crimenet.ingestion.column_names import normalize_column_names
from crimenet.ingestion.metadata import add_ingestion_metadata
from crimenet.ingestion.readers import (
    read_acs5_tract_batch_raw,
    read_dallas_raw,
    read_fort_worth_raw,
    read_houston_raw,
    read_weather_batch_raw,
)
from crimenet.jobs.bronze_ingestion import COLUMN_OVERRIDES
from crimenet.transforms.canonical import (
    add_crime_offense_id,
    build_crime_offenses,
    deduplicate_crime_offenses,
)
from crimenet.transforms.dallas import to_canonical as transform_dallas
from crimenet.transforms.fort_worth import (
    to_canonical as transform_fort_worth,
)
from crimenet.transforms.houston import to_canonical as transform_houston


@pytest.fixture(scope="session")
def fixture_path() -> Callable[[str], Path]:
    fixture_root = Path(__file__).resolve().parent / "fixtures"

    def resolve(relative_path: str) -> Path:
        path = fixture_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(
                f"CrimeNet test fixture does not exist: {relative_path}"
            )
        return path

    return resolve


@pytest.fixture(scope="session")
def spark(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[SparkSession]:
    runtime_root = tmp_path_factory.mktemp("spark-runtime")
    previous_timezone = os.environ.get("TZ")
    previous_local_dirs = os.environ.get("SPARK_LOCAL_DIRS")
    previous_worker_python = os.environ.get("PYSPARK_PYTHON")
    previous_driver_python = os.environ.get(
        "PYSPARK_DRIVER_PYTHON"
    )
    os.environ["TZ"] = "UTC"
    os.environ["SPARK_LOCAL_DIRS"] = str(runtime_root / "local")
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    if hasattr(time, "tzset"):
        time.tzset()

    session = (
        SparkSession.builder
        .master("local[2]")
        .appName("crimenet-tests")
        .config("spark.ui.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false")
        .config(
            "spark.sql.warehouse.dir",
            str(runtime_root / "warehouse"),
        )
        .config("spark.sql.catalogImplementation", "in-memory")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.default.parallelism", "2")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    session.sparkContext.setCheckpointDir(
        str(runtime_root / "checkpoints")
    )
    session.conf.set("spark.sql.session.timeZone", "UTC")

    try:
        yield session
    finally:
        session.catalog.clearCache()
        session.stop()
        if previous_timezone is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = previous_timezone
        if previous_local_dirs is None:
            os.environ.pop("SPARK_LOCAL_DIRS", None)
        else:
            os.environ["SPARK_LOCAL_DIRS"] = previous_local_dirs
        if previous_worker_python is None:
            os.environ.pop("PYSPARK_PYTHON", None)
        else:
            os.environ["PYSPARK_PYTHON"] = previous_worker_python
        if previous_driver_python is None:
            os.environ.pop("PYSPARK_DRIVER_PYTHON", None)
        else:
            os.environ[
                "PYSPARK_DRIVER_PYTHON"
            ] = previous_driver_python
        if hasattr(time, "tzset"):
            time.tzset()


def _bronze_dataframe(
    raw_dataframe: DataFrame,
    *,
    source_system: str,
    overrides: dict[str, str] | None = None,
) -> DataFrame:
    return add_ingestion_metadata(
        normalize_column_names(
            raw_dataframe,
            overrides=overrides,
        ),
        source_system=source_system,
    )


@pytest.fixture(scope="session")
def dallas_raw(
    spark: SparkSession,
    fixture_path: Callable[[str], Path],
) -> DataFrame:
    return read_dallas_raw(
        spark,
        str(fixture_path("dallas/dallas_fixture.csv")),
    )


@pytest.fixture(scope="session")
def houston_raw(
    spark: SparkSession,
    fixture_path: Callable[[str], Path],
) -> DataFrame:
    return read_houston_raw(
        spark,
        str(fixture_path("houston/houston_fixture.csv")),
    )


@pytest.fixture(scope="session")
def fort_worth_raw(
    spark: SparkSession,
    fixture_path: Callable[[str], Path],
) -> DataFrame:
    return read_fort_worth_raw(
        spark,
        str(fixture_path("fort_worth/fort_worth_fixture.json")),
    )


@pytest.fixture(scope="session")
def socioeconomic_raw(
    spark: SparkSession,
    fixture_path: Callable[[str], Path],
) -> DataFrame:
    return read_acs5_tract_batch_raw(
        spark,
        str(
            fixture_path(
                "socioeconomic/socioeconomic_fixture.json"
            )
        ),
    )


@pytest.fixture(scope="session")
def weather_raw(
    spark: SparkSession,
    fixture_path: Callable[[str], Path],
) -> DataFrame:
    return read_weather_batch_raw(
        spark,
        str(fixture_path("weather/weather_fixture.json")),
    )


@pytest.fixture(scope="session")
def dallas_bronze(dallas_raw: DataFrame) -> DataFrame:
    return _bronze_dataframe(
        dallas_raw,
        source_system="dallas",
    ).cache()


@pytest.fixture(scope="session")
def houston_bronze(houston_raw: DataFrame) -> DataFrame:
    return _bronze_dataframe(
        houston_raw,
        source_system="houston",
    ).cache()


@pytest.fixture(scope="session")
def fort_worth_bronze(
    fort_worth_raw: DataFrame,
) -> DataFrame:
    return _bronze_dataframe(
        fort_worth_raw,
        source_system="fort_worth",
        overrides=COLUMN_OVERRIDES["fort_worth"],
    ).cache()


@pytest.fixture(scope="session")
def socioeconomic_bronze(
    socioeconomic_raw: DataFrame,
) -> DataFrame:
    return _bronze_dataframe(
        socioeconomic_raw,
        source_system="census_acs5",
    ).cache()


@pytest.fixture(scope="session")
def weather_bronze(weather_raw: DataFrame) -> DataFrame:
    return _bronze_dataframe(
        weather_raw,
        source_system="open_meteo",
    ).cache()


@pytest.fixture(scope="session")
def dallas_canonical(dallas_bronze: DataFrame) -> DataFrame:
    return transform_dallas(dallas_bronze).cache()


@pytest.fixture(scope="session")
def houston_canonical(houston_bronze: DataFrame) -> DataFrame:
    return transform_houston(houston_bronze).cache()


@pytest.fixture(scope="session")
def fort_worth_canonical(
    fort_worth_bronze: DataFrame,
) -> DataFrame:
    return transform_fort_worth(fort_worth_bronze).cache()


@pytest.fixture(scope="session")
def canonical_crimes(
    dallas_bronze: DataFrame,
    houston_bronze: DataFrame,
    fort_worth_bronze: DataFrame,
) -> DataFrame:
    return build_crime_offenses(
        dallas_bronze,
        houston_bronze,
        fort_worth_bronze,
    ).cache()


@pytest.fixture(scope="session")
def crime_offenses_with_ids(
    canonical_crimes: DataFrame,
) -> DataFrame:
    return add_crime_offense_id(canonical_crimes).cache()


@pytest.fixture(scope="session")
def deduplicated_crimes(
    crime_offenses_with_ids: DataFrame,
) -> DataFrame:
    return deduplicate_crime_offenses(
        crime_offenses_with_ids
    ).cache()
