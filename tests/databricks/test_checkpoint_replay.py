from __future__ import annotations

from uuid import uuid4

import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    LongType,
    StringType,
    StructField,
    StructType,
)

pytestmark = pytest.mark.databricks


def test_autoloader_checkpoint_replay_is_absorbed_by_keyed_delta_merge(
    spark: SparkSession,
) -> None:
    from delta.tables import DeltaTable
    from pyspark.dbutils import DBUtils

    dbutils = DBUtils(spark)
    test_root = f"dbfs:/tmp/crimenet-tests/{uuid4().hex}"
    source_path = f"{test_root}/source"
    schema_path = f"{test_root}/schema"
    checkpoint_path = f"{test_root}/checkpoint"
    target_path = f"{test_root}/target"
    schema = StructType(
        [
            StructField("id", LongType(), False),
            StructField("value", StringType(), True),
        ]
    )

    def merge_batch(batch: DataFrame, _batch_id: int) -> None:
        source = batch.select("id", "value").dropDuplicates(["id"])
        (
            DeltaTable.forPath(spark, target_path)
            .alias("target")
            .merge(source.alias("source"), "target.id = source.id")
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .execute()
        )

    def process_available_files() -> None:
        stream = (
            spark.readStream
            .format("cloudFiles")
            .option("cloudFiles.format", "json")
            .option("cloudFiles.schemaLocation", schema_path)
            .schema(schema)
            .load(source_path)
        )
        query = (
            stream.writeStream
            .option("checkpointLocation", checkpoint_path)
            .trigger(availableNow=True)
            .foreachBatch(merge_batch)
            .start()
        )
        query.awaitTermination()

    try:
        dbutils.fs.mkdirs(source_path)
        dbutils.fs.put(
            f"{source_path}/records.json",
            '{"id":1,"value":"first"}\n{"id":2,"value":"second"}\n',
            True,
        )
        spark.createDataFrame([], schema).write.format("delta").save(
            target_path
        )

        process_available_files()
        assert spark.read.format("delta").load(target_path).count() == 2

        dbutils.fs.rm(checkpoint_path, True)
        process_available_files()

        rows = {
            row.id: row.value
            for row in (
                spark.read.format("delta")
                .load(target_path)
                .collect()
            )
        }
        assert rows == {1: "first", 2: "second"}
    finally:
        dbutils.fs.rm(test_root, True)
