from __future__ import annotations

from collections.abc import Iterator

import pytest
from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[SparkSession]:
    warehouse = tmp_path_factory.mktemp("unit-spark-warehouse")
    session = (
        SparkSession.builder.master("local[2]")
        .appName("crimenet-tests")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.warehouse.dir", str(warehouse))
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


@pytest.fixture(scope="module")
def delta_spark(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[SparkSession]:
    warehouse = tmp_path_factory.mktemp("delta-spark-warehouse")
    builder = (
        SparkSession.builder.master("local[2]")
        .appName("crimenet-delta-integration-tests")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.sql.session.timeZone", "UTC")
        .config(
            "spark.sql.extensions",
            "io.delta.sql.DeltaSparkSessionExtension",
        )
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.sql.warehouse.dir", str(warehouse))
        .config("spark.databricks.delta.snapshotPartitions", "2")
    )
    session = configure_spark_with_delta_pip(builder).getOrCreate()
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()
