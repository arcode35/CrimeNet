from __future__ import annotations

import sys
from datetime import datetime
from typing import Any

import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from crimenet.config.resources import CrimeNetTables
from crimenet.jobs import (
    acs5_landing_job,
    acs_vintage_calendar_job,
    bronze_ingestion,
    gold_crime_features_job,
    location_tract_mapping_job,
    preflight,
    quality_checks,
    silver_lighting_job,
    silver_socioeconomic_job,
    silver_transform,
    silver_weather_job,
    tiger_line_boundaries_job,
    weather_request_planner_job,
    weather_retrieval_job,
)
from crimenet.observability.run_context import resolve_pipeline_run_id
from crimenet.quality.quarantine import merge_quarantine
from crimenet.quality.rules import (
    coordinates_are_valid,
    has_source_identity,
    occurred_at_is_valid,
)
from crimenet.silver.socioeconomic import SOCIOECONOMIC_DEFINITION_VERSION
from crimenet.silver.weather import WEATHER_DEFINITION_VERSION


@pytest.mark.parametrize(
    ("module", "arguments", "expected"),
    [
        (
            acs5_landing_job,
            ["--landing-path", "/landing"],
            ("landing_path", "/landing"),
        ),
        (
            acs_vintage_calendar_job,
            ["--catalog", "crime"],
            ("catalog", "crime"),
        ),
        (
            bronze_ingestion,
            [
                "--catalog",
                "crime",
                "--source",
                "dallas",
                "--input-path",
                "/landing",
            ],
            ("write_mode", "merge"),
        ),
        (
            gold_crime_features_job,
            ["--catalog", "crime"],
            ("weather_h3_resolution", 6),
        ),
        (
            location_tract_mapping_job,
            ["--catalog", "crime"],
            ("maximum_ambiguous_matches", 0),
        ),
        (
            preflight,
            ["--catalog", "crime", "--validate-only"],
            ("validate_only", True),
        ),
        (
            quality_checks,
            [
                "--catalog",
                "crime",
                "--silver-schema",
                "silver",
                "--data-quality-schema",
                "dq",
            ],
            ("maximum_quarantine_rate", 0.05),
        ),
        (
            silver_lighting_job,
            ["--catalog", "crime", "--full-rebuild"],
            ("full_rebuild", True),
        ),
        (
            silver_socioeconomic_job,
            ["--catalog", "crime", "--full-rebuild"],
            ("checkpoint_path", None),
        ),
        (
            silver_transform,
            ["--catalog", "crime"],
            ("minimum_row_count", 1),
        ),
        (
            silver_weather_job,
            ["--catalog", "crime", "--checkpoint-path", "/state"],
            ("checkpoint_path", "/state"),
        ),
        (
            tiger_line_boundaries_job,
            ["--catalog", "crime", "--landing-path", "/landing"],
            ("state_fips", "48"),
        ),
        (
            weather_request_planner_job,
            ["--catalog", "crime"],
            ("hourly_variables", "temperature_2m"),
        ),
        (
            weather_retrieval_job,
            ["--catalog", "crime", "--cache-directory", "/cache"],
            ("max_retries", 5),
        ),
    ],
)
def test_job_argument_contracts(
    module: Any,
    arguments: list[str],
    expected: tuple[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", [module.__name__, *arguments])
    parsed = module.parse_args()
    assert getattr(parsed, expected[0]) == expected[1]


def test_job_argument_cross_field_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bronze",
            "--catalog",
            "crime",
            "--source",
            "open_meteo_weather",
            "--input-path",
            "/landing",
        ],
    )
    with pytest.raises(SystemExit):
        bronze_ingestion.parse_args()

    monkeypatch.setattr(
        sys,
        "argv",
        ["silver-acs", "--catalog", "crime"],
    )
    with pytest.raises(SystemExit):
        silver_socioeconomic_job.parse_args()


def test_table_names_cover_every_pipeline_resource() -> None:
    tables = CrimeNetTables(
        catalog="crime",
        bronze_schema="raw",
        silver_schema="clean",
        gold_schema="features",
        operations_schema="operations",
        data_quality_schema="quality",
    )
    assert tables.dallas_bronze == "crime.raw.dallas_crime"
    assert tables.houston_bronze == "crime.raw.houston_crime"
    assert tables.fort_worth_bronze == "crime.raw.fort_worth_crime"
    assert tables.open_meteo_weather_bronze == "crime.raw.open_meteo_weather"
    assert tables.acs5_tract_bronze == "crime.raw.acs5_tract_socioeconomic"
    assert tables.weather_hourly_silver == "crime.clean.weather_hourly"
    assert tables.crime_offenses_silver == "crime.clean.crime_offenses"
    assert tables.tract_socioeconomic_silver == (
        "crime.clean.tract_socioeconomic"
    )
    assert tables.crime_quarantine == "crime.quality.crime_quarantine"
    assert tables.quality_results == "crime.quality.quality_results"
    assert tables.pipeline_failures == "crime.operations.pipeline_failures"
    assert tables.bronze_for_source("acs5_tract") == tables.acs5_tract_bronze
    with pytest.raises(ValueError, match="Unsupported source"):
        tables.bronze_for_source("invalid")


def test_pipeline_run_id_precedence_and_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRIMENET_PIPELINE_RUN_ID", " env-run ")
    monkeypatch.setenv("DATABRICKS_JOB_RUN_ID", "db-run")
    assert resolve_pipeline_run_id(" explicit-run ") == "explicit_run"
    assert resolve_pipeline_run_id() == "env_run"

    monkeypatch.delenv("CRIMENET_PIPELINE_RUN_ID")
    assert resolve_pipeline_run_id() == "db_run"
    monkeypatch.delenv("DATABRICKS_JOB_RUN_ID")
    generated = resolve_pipeline_run_id()
    assert len(generated) == 32


def test_reusable_quality_rules(
    spark: SparkSession,
) -> None:
    dataframe = spark.createDataFrame(
        [
            ("dallas", "1", 32.8, -96.8, datetime(2024, 1, 1)),
            ("dallas", None, 32.8, -196.8, None),
            ("dallas", "3", None, None, datetime(2024, 1, 1)),
        ],
        """
        source_city string,
        source_record_id string,
        latitude double,
        longitude double,
        occurred_at timestamp
        """,
    )
    assert dataframe.filter(has_source_identity()).count() == 2
    assert dataframe.filter(coordinates_are_valid()).count() == 2
    assert dataframe.filter(occurred_at_is_valid()).count() == 2


def test_bronze_deduplication_and_run_dispatch(
    spark: SparkSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = spark.createDataFrame(
        [
            ("dallas", "hash", "/z.csv", datetime(2024, 1, 2)),
            ("dallas", "hash", "/a.csv", datetime(2024, 1, 3)),
            ("dallas", "other", "/x.csv", datetime(2024, 1, 1)),
        ],
        """
        source_system string,
        source_row_hash string,
        source_file string,
        ingested_at timestamp
        """,
    )
    selected = bronze_ingestion._deduplicate_bronze_batch(rows)
    assert {
        (row["source_row_hash"], row["source_file"])
        for row in selected.collect()
    } == {("hash", "/a.csv"), ("other", "/x.csv")}

    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(
        bronze_ingestion,
        "_run_batch_ingestion",
        lambda _spark, **kwargs: calls.append(("batch", kwargs)),
    )
    monkeypatch.setattr(
        bronze_ingestion,
        "_run_streaming_ingestion",
        lambda _spark, **kwargs: calls.append(("stream", kwargs)),
    )
    bronze_ingestion.run(
        spark,
        catalog="crime",
        bronze_schema="bronze",
        source="dallas",
        input_path="/landing",
        write_mode="merge",
    )
    assert calls[-1][0] == "batch"

    with pytest.raises(ValueError, match="schema_path"):
        bronze_ingestion.run(
            spark,
            catalog="crime",
            bronze_schema="bronze",
            source="open_meteo_weather",
            input_path="/landing",
            write_mode="merge",
        )
    with pytest.raises(ValueError, match="checkpoint_path"):
        bronze_ingestion.run(
            spark,
            catalog="crime",
            bronze_schema="bronze",
            source="open_meteo_weather",
            input_path="/landing",
            write_mode="merge",
            schema_path="/schema",
        )
    bronze_ingestion.run(
        spark,
        catalog="crime",
        bronze_schema="bronze",
        source="open_meteo_weather",
        input_path="/landing",
        write_mode="merge",
        schema_path="/schema",
        checkpoint_path="/checkpoint",
    )
    assert calls[-1][0] == "stream"


class _CatalogProxy:
    def __init__(
        self,
        session: SparkSession,
        *,
        table_exists: bool = True,
    ) -> None:
        self.session = session
        self.table_exists = table_exists

    def tableExists(self, _table: str) -> bool:
        return self.table_exists

    def dropTempView(self, name: str) -> None:
        self.session.catalog.dropTempView(name)


class _SqlRecorder:
    def __init__(self, session: SparkSession) -> None:
        self.session = session
        self.catalog = _CatalogProxy(session)
        self.statements: list[str] = []

    def sql(self, statement: str) -> None:
        self.statements.append(statement)


def test_weather_ddl_quotes_and_validates_target_table(
    spark: SparkSession,
) -> None:
    recorder = _SqlRecorder(spark)
    silver_weather_job.ensure_weather_hourly_table(
        recorder,  # type: ignore[arg-type]
        table_name="crime.silver.weather_hourly",
    )
    assert (
        "CREATE TABLE IF NOT EXISTS "
        "`crime`.`silver`.`weather_hourly`"
    ) in recorder.statements[0]

    with pytest.raises(ValueError, match="table name component"):
        silver_weather_job.ensure_weather_hourly_table(
            recorder,  # type: ignore[arg-type]
            table_name="crime.silver.weather_hourly; DROP TABLE victims",
        )
    assert len(recorder.statements) == 1


def _weather_merge_frame(spark: SparkSession) -> DataFrame:
    return spark.createDataFrame(
        [
            (
                "open_meteo",
                "era5_land",
                123,
                datetime(2024, 1, 1),
                "hash-a",
                datetime(2024, 1, 2),
                datetime(2024, 1, 3),
            ),
            (
                "open_meteo",
                "era5_land",
                123,
                datetime(2024, 1, 1),
                "hash-b",
                datetime(2024, 1, 2),
                datetime(2024, 1, 3),
            ),
        ],
        """
        provider string,
        model string,
        weather_query_cell_id long,
        weather_timestamp timestamp,
        source_row_hash string,
        bronze_ingested_at timestamp,
        silver_processed_at timestamp
        """,
    )


def test_weather_merge_builds_deterministic_delta_statement(
    spark: SparkSession,
) -> None:
    recorder = _SqlRecorder(spark)
    silver_weather_job.merge_weather_batch(
        _weather_merge_frame(spark),
        7,
        spark=recorder,  # type: ignore[arg-type]
        target_table="crime.silver.weather_hourly",
    )
    assert len(recorder.statements) == 1
    statement = recorder.statements[0]
    assert (
        "MERGE INTO `crime`.`silver`.`weather_hourly`"
        in statement
    )
    assert "target.weather_query_cell_id = source.weather_query_cell_id" in (
        statement
    )
    assert "source.source_row_hash" in statement

    empty = _weather_merge_frame(spark).limit(0)
    silver_weather_job.merge_weather_batch(
        empty,
        8,
        spark=recorder,  # type: ignore[arg-type]
        target_table="crime.silver.weather_hourly",
    )
    assert len(recorder.statements) == 1

    with pytest.raises(ValueError, match="table name component"):
        silver_weather_job.merge_weather_batch(
            empty,
            9,
            spark=recorder,  # type: ignore[arg-type]
            target_table="crime.silver.weather_hourly; DROP TABLE victims",
        )
    assert len(recorder.statements) == 1


def test_socioeconomic_merge_quotes_and_validates_target_table(
    spark: SparkSession,
) -> None:
    batch = spark.createDataFrame(
        [
            (
                "48113000100",
                2023,
                "acs-v1",
                "hash-a",
                "/landing/acs.jsonl",
            )
        ],
        """
        geoid string,
        acs_vintage int,
        socioeconomic_definition_version string,
        source_row_hash string,
        source_file string
        """,
    )
    recorder = _SqlRecorder(spark)
    silver_socioeconomic_job.merge_socioeconomic_batch(
        batch,
        7,
        spark=recorder,  # type: ignore[arg-type]
        target_table="crime.silver.tract_socioeconomic",
    )
    assert (
        "MERGE INTO `crime`.`silver`.`tract_socioeconomic`"
        in recorder.statements[0]
    )

    unsafe_target = "crime.silver.tract_socioeconomic; DROP TABLE victims"
    with pytest.raises(ValueError, match="table name component"):
        silver_socioeconomic_job.merge_socioeconomic_batch(
            batch.limit(0),
            8,
            spark=recorder,  # type: ignore[arg-type]
            target_table=unsafe_target,
        )
    with pytest.raises(ValueError, match="table name component"):
        silver_socioeconomic_job.ensure_target_table(
            spark,
            target_table=unsafe_target,
            schema=batch.schema,
        )
    assert len(recorder.statements) == 1


def test_quarantine_persistence_uses_entity_and_observation_merges(
    spark: SparkSession,
) -> None:
    quarantine = spark.createDataFrame(
        [
            (
                "reject-1",
                "dallas",
                "/source.csv",
                "row-hash",
                "{}",
                "INVALID_COORDINATES",
                "coordinates invalid",
                "run-1",
                datetime(2024, 1, 1),
                "{}",
            )
        ],
        """
        quarantine_id string,
        source_system string,
        source_file string,
        source_row_hash string,
        raw_payload string,
        quarantine_reason_code string,
        quarantine_reason string,
        pipeline_run_id string,
        quarantined_at timestamp,
        validation_fields string
        """,
    )
    recorder = _SqlRecorder(spark)
    merge_quarantine(
        recorder,  # type: ignore[arg-type]
        quarantine=quarantine,
        target_table="crime.quality.crime_quarantine",
    )
    assert len(recorder.statements) == 2
    assert "quarantine_id" in recorder.statements[0]
    assert "quarantine_observation_id" in recorder.statements[1]
    assert "<=>" in recorder.statements[0]
    assert "MERGE INTO `crime`.`quality`.`crime_quarantine`" in (
        recorder.statements[0]
    )

    merge_quarantine(
        recorder,  # type: ignore[arg-type]
        quarantine=quarantine.limit(0),
        target_table="crime.quality.crime_quarantine",
    )
    assert len(recorder.statements) == 2

    with pytest.raises(RuntimeError, match="must be non-null and non-blank"):
        merge_quarantine(
            recorder,  # type: ignore[arg-type]
            quarantine=quarantine.withColumn(
                "quarantine_id",
                F.lit(None).cast("string"),
            ),
            target_table="crime.quality.crime_quarantine",
        )
    assert len(recorder.statements) == 2


def test_socioeconomic_deduplication_validation_and_dispatch(
    spark: SparkSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = spark.createDataFrame(
        [
            (
                "48113000100",
                2023,
                35.0,
                datetime(2024, 1, 1),
                "hash-a",
                "/a",
                "acs-v1",
            ),
            (
                "48113000100",
                2023,
                36.0,
                datetime(2024, 1, 2),
                "hash-b",
                "/b",
                "acs-v1",
            ),
        ],
        """
        geoid string,
        acs_vintage int,
        median_age double,
        bronze_ingested_at timestamp,
        source_row_hash string,
        source_file string,
        socioeconomic_definition_version string
        """,
    )
    deduplicated = (
        silver_socioeconomic_job.deduplicate_socioeconomic_records(source)
    )
    assert deduplicated.first()["median_age"] == 36.0
    silver_socioeconomic_job.validate_socioeconomic_dataframe(deduplicated)

    with pytest.raises(RuntimeError, match="duplicate"):
        silver_socioeconomic_job.validate_socioeconomic_dataframe(source)
    with pytest.raises(RuntimeError, match="median_age"):
        silver_socioeconomic_job.validate_socioeconomic_dataframe(
            deduplicated.withColumn("median_age", F.lit(121.0))
        )

    calls: list[str] = []
    monkeypatch.setattr(
        silver_socioeconomic_job,
        "rebuild_socioeconomic_table",
        lambda *_args, **_kwargs: calls.append("rebuild"),
    )
    monkeypatch.setattr(
        silver_socioeconomic_job,
        "run_incremental_stream",
        lambda *_args, **_kwargs: calls.append("incremental"),
    )
    monkeypatch.setattr(
        silver_socioeconomic_job,
        "validate_rebuilt_table",
        lambda *_args, **_kwargs: calls.append("validate"),
    )
    silver_socioeconomic_job.run(
        spark,
        catalog="crime",
        bronze_schema="bronze",
        silver_schema="silver",
        data_quality_schema="quality",
        checkpoint_path=None,
        full_rebuild=True,
        pipeline_run_id="run-1",
    )
    assert calls == ["rebuild", "validate"]
    with pytest.raises(ValueError, match="checkpoint_path"):
        silver_socioeconomic_job.run(
            spark,
            catalog="crime",
            bronze_schema="bronze",
            silver_schema="silver",
            data_quality_schema="quality",
            checkpoint_path=None,
            full_rebuild=False,
        )


def test_definition_changes_require_full_derived_table_rebuild(
    spark: SparkSession,
) -> None:
    bronze = spark.range(1)
    current_weather = spark.createDataFrame(
        [(WEATHER_DEFINITION_VERSION,)],
        "weather_definition_version string",
    )
    stale_weather = spark.createDataFrame(
        [("retired-weather-definition",)],
        "weather_definition_version string",
    )
    assert not silver_weather_job.weather_rebuild_required(
        bronze_dataframe=bronze,
        target_dataframe=current_weather,
    )
    assert silver_weather_job.weather_rebuild_required(
        bronze_dataframe=bronze,
        target_dataframe=stale_weather,
    )
    assert silver_weather_job.weather_rebuild_required(
        bronze_dataframe=bronze,
        target_dataframe=current_weather.limit(0),
    )

    current_acs = spark.createDataFrame(
        [(SOCIOECONOMIC_DEFINITION_VERSION,)],
        "socioeconomic_definition_version string",
    )
    stale_acs = spark.createDataFrame(
        [("retired-acs-definition",)],
        "socioeconomic_definition_version string",
    )
    assert not silver_socioeconomic_job.socioeconomic_rebuild_required(
        bronze_dataframe=bronze,
        target_dataframe=current_acs,
    )
    assert silver_socioeconomic_job.socioeconomic_rebuild_required(
        bronze_dataframe=bronze,
        target_dataframe=stale_acs,
    )
    assert not silver_socioeconomic_job.socioeconomic_rebuild_required(
        bronze_dataframe=bronze.limit(0),
        target_dataframe=stale_acs,
    )
