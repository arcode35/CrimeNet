from __future__ import annotations

from pyspark.sql import SparkSession

from crimenet.ingestion.metadata import (
    add_ingestion_metadata,
    source_row_hash,
)


def test_source_hash_ignores_file_path_and_retrieval_metadata(
    spark: SparkSession,
) -> None:
    rows = [
        ("same", "/landing/a.csv", "2026-01-01T00:00:00Z"),
        ("same", "/copied/b.csv", "2026-02-01T00:00:00Z"),
    ]
    base = spark.createDataFrame(
        rows,
        "value STRING, source_file STRING, retrieved_at STRING",
    )
    dataframe = base.withColumn("source_row_hash", source_row_hash(base))
    hashes = [row["source_row_hash"] for row in dataframe.collect()]
    assert len(set(hashes)) == 1


def test_ingestion_metadata_attaches_contract_and_stable_identity(
    spark: SparkSession,
) -> None:
    source = spark.createDataFrame(
        [("one", "/landing/a.csv")],
        "value STRING, source_file STRING",
    )
    result = add_ingestion_metadata(
        source,
        "dallas",
        contract_version="municipal_crime_v1",
    ).first()
    assert result is not None
    assert result["source_system"] == "dallas"
    assert result["source_contract_version"] == "municipal_crime_v1"
    assert len(result["source_row_hash"]) == 64
    assert result["ingested_at"] is not None


def test_hash_is_independent_of_column_order(spark: SparkSession) -> None:
    first = spark.createDataFrame([("a", "b")], "left STRING, right STRING")
    second = first.select("right", "left")
    assert (
        first.select(source_row_hash(first).alias("hash")).first()["hash"]
        == second.select(source_row_hash(second).alias("hash")).first()["hash"]
    )
