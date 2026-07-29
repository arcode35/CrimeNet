from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pyspark.sql import DataFrame, Row, SparkSession

from crimenet.config.resources import CrimeNetTables
from crimenet.config.validation import QualityThresholds
from crimenet.jobs import (
    preflight,
    quality_checks,
    silver_transform,
    weather_retrieval_job,
)
from crimenet.quality.checks import QualityCheck
from crimenet.weather.open_meteo_client import OpenMeteoClientConfig


class _PreflightCatalog:
    def __init__(self, missing: set[str] | None = None) -> None:
        self.missing = missing or set()

    def databaseExists(self, name: str) -> bool:
        return name not in self.missing


class _CollectedRows:
    def __init__(self, rows: list[Row]) -> None:
        self.rows = rows

    def collect(self) -> list[Row]:
        return self.rows


class _PreflightSpark:
    def __init__(
        self,
        *,
        volumes: dict[str, set[str]],
        missing_schemas: set[str] | None = None,
    ) -> None:
        self.catalog = _PreflightCatalog(missing_schemas)
        self.volumes = volumes
        self.statements: list[str] = []

    def sql(self, statement: str) -> _CollectedRows:
        self.statements.append(statement)
        prefix = "SHOW VOLUMES IN "
        if statement.startswith(prefix):
            schema = statement.removeprefix(prefix)
            return _CollectedRows(
                [
                    Row(volume_name=name)
                    for name in sorted(self.volumes.get(schema, set()))
                ]
            )
        return _CollectedRows([])


def test_preflight_creates_and_verifies_exact_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spark = _PreflightSpark(
        volumes={
            "crime.raw": {"landing"},
            "crime.ops": {"checkpoints"},
        }
    )
    secrets: list[tuple[str | None, str | None]] = []
    monkeypatch.setattr(
        preflight,
        "_validate_secret",
        lambda _spark, *, secret_scope, census_api_key_secret: secrets.append(
            (secret_scope, census_api_key_secret)
        ),
    )
    preflight.run(
        spark,  # type: ignore[arg-type]
        catalog="crime",
        schemas=("raw", "ops"),
        volumes=(("raw", "landing"), ("ops", "checkpoints")),
        secret_scope="scope",
        census_api_key_secret="key",
        validate_only=False,
    )
    assert spark.statements[:3] == [
        "CREATE CATALOG IF NOT EXISTS crime",
        "CREATE SCHEMA IF NOT EXISTS crime.raw",
        "CREATE SCHEMA IF NOT EXISTS crime.ops",
    ]
    assert "CREATE VOLUME IF NOT EXISTS crime.raw.landing" in spark.statements
    assert secrets == [("scope", "key")]


def test_preflight_reports_missing_schema_volume_and_secret_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_schema = _PreflightSpark(
        volumes={},
        missing_schemas={"crime.raw"},
    )
    with pytest.raises(RuntimeError, match="Missing required schemas"):
        preflight.run(
            missing_schema,  # type: ignore[arg-type]
            catalog="crime",
            schemas=("raw",),
            volumes=(),
            secret_scope=None,
            census_api_key_secret=None,
            validate_only=True,
        )

    missing_volume = _PreflightSpark(volumes={"crime.raw": set()})
    with pytest.raises(RuntimeError, match="Missing required Unity Catalog"):
        preflight.run(
            missing_volume,  # type: ignore[arg-type]
            catalog="crime",
            schemas=("raw",),
            volumes=(("raw", "landing"),),
            secret_scope=None,
            census_api_key_secret=None,
            validate_only=True,
        )

    with pytest.raises(ValueError, match="supplied together"):
        preflight._validate_secret(
            object(),  # type: ignore[arg-type]
            secret_scope="scope",
            census_api_key_secret=None,
        )
    preflight._validate_secret(
        object(),  # type: ignore[arg-type]
        secret_scope=None,
        census_api_key_secret=None,
    )

    class _Secrets:
        def get(self, *, scope: str, key: str) -> str:
            assert (scope, key) == ("scope", "key")
            return ""

    class _Dbutils:
        def __init__(self, _spark: object) -> None:
            self.secrets = _Secrets()

    monkeypatch.setattr(
        preflight.importlib,
        "import_module",
        lambda _name: SimpleNamespace(DBUtils=_Dbutils),
    )
    with pytest.raises(RuntimeError, match="exists but is empty"):
        preflight._validate_secret(
            object(),  # type: ignore[arg-type]
            secret_scope="scope",
            census_api_key_secret="key",
        )


class _TableCatalog:
    def __init__(self, existing: set[str]) -> None:
        self.existing = existing

    def tableExists(self, name: str) -> bool:
        return name in self.existing


class _MappedSpark:
    def __init__(
        self,
        session: SparkSession,
        tables: dict[str, DataFrame],
        *,
        existing: set[str] | None = None,
    ) -> None:
        self.session = session
        self.tables = tables
        self.catalog = _TableCatalog(existing or set())
        self.conf = session.conf

    def table(self, name: str) -> DataFrame:
        return self.tables[name]


def _bronze_frame(
    spark: SparkSession,
    source: str,
    identity: str,
) -> DataFrame:
    return spark.createDataFrame(
        [(source, f"hash-{source}", identity)],
        """
        source_system string,
        source_row_hash string,
        business_identity string
        """,
    )


def test_silver_transform_orchestrates_validation_before_promotion(
    spark: SparkSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tables = CrimeNetTables(catalog="crime")
    bronze_frames = {
        tables.dallas_bronze: _bronze_frame(spark, "dallas", "identity-d"),
        tables.houston_bronze: _bronze_frame(spark, "houston", "identity-h"),
        tables.fort_worth_bronze: _bronze_frame(
            spark,
            "fort_worth",
            "identity-f",
        ),
    }
    valid = bronze_frames[tables.dallas_bronze].unionByName(
        bronze_frames[tables.houston_bronze]
    ).unionByName(bronze_frames[tables.fort_worth_bronze])
    quarantine = spark.createDataFrame(
        [],
        "quarantine_id string, quarantine_reason_code string",
    )
    candidate = valid.withColumn(
        "weather_query_cell_id",
        valid.source_row_hash.cast("long"),
    )
    mapped = _MappedSpark(
        spark,
        {
            **bronze_frames,
            tables.crime_offenses_silver: candidate,
        },
    )
    calls: list[str] = []
    monkeypatch.setattr(
        silver_transform,
        "build_crime_offenses",
        lambda **_kwargs: valid,
    )
    monkeypatch.setattr(
        silver_transform,
        "split_crime_quarantine",
        lambda _frame, **_kwargs: (valid, quarantine),
    )
    monkeypatch.setattr(
        silver_transform,
        "merge_quarantine",
        lambda *_args, **_kwargs: calls.append("quarantine"),
    )
    monkeypatch.setattr(
        silver_transform,
        "deduplicate_crime_offenses",
        lambda _frame: valid,
    )
    monkeypatch.setattr(
        silver_transform,
        "add_weather_query_cell",
        lambda _frame, **_kwargs: candidate,
    )
    monkeypatch.setattr(
        silver_transform,
        "assert_silver_contract",
        lambda _frame: calls.append("contract"),
    )
    monkeypatch.setattr(
        silver_transform,
        "evaluate_crime_quality",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        silver_transform,
        "quality_results_dataframe",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        silver_transform,
        "merge_quality_results",
        lambda *_args, **_kwargs: calls.append("quality"),
    )

    def promote(
        _spark: object,
        *,
        candidate: DataFrame,
        validate: Any,
        **_kwargs: object,
    ) -> None:
        calls.append("stage")
        validate(candidate)
        calls.append("promote")

    monkeypatch.setattr(silver_transform, "promote_staged_table", promote)
    silver_transform.run(
        mapped,  # type: ignore[arg-type]
        catalog="crime",
        bronze_schema="bronze",
        silver_schema="silver",
        data_quality_schema="data_quality",
        pipeline_run_id="run/1",
        thresholds=QualityThresholds(
            minimum_row_count=0,
            minimum_silver_to_bronze_ratio=0.0,
        ),
    )
    assert calls == [
        "quarantine",
        "stage",
        "contract",
        "quality",
        "promote",
    ]


def test_silver_transform_persists_failed_candidate_evidence(
    spark: SparkSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tables = CrimeNetTables(catalog="crime")
    bronze = _bronze_frame(spark, "dallas", "identity")
    valid = bronze
    quarantine = spark.createDataFrame(
        [],
        "quarantine_id string, quarantine_reason_code string",
    )
    candidate = valid.withColumn(
        "weather_query_cell_id",
        valid.source_row_hash.cast("long"),
    )
    mapped = _MappedSpark(
        spark,
        {
            tables.dallas_bronze: bronze,
            tables.houston_bronze: bronze,
            tables.fort_worth_bronze: bronze,
            tables.crime_offenses_silver: candidate,
        },
    )
    persisted: list[object] = []
    monkeypatch.setattr(
        silver_transform,
        "build_crime_offenses",
        lambda **_kwargs: valid,
    )
    monkeypatch.setattr(
        silver_transform,
        "split_crime_quarantine",
        lambda *_args, **_kwargs: (valid, quarantine),
    )
    monkeypatch.setattr(
        silver_transform,
        "merge_quarantine",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        silver_transform,
        "deduplicate_crime_offenses",
        lambda _frame: valid,
    )
    monkeypatch.setattr(
        silver_transform,
        "add_weather_query_cell",
        lambda *_args, **_kwargs: candidate,
    )
    monkeypatch.setattr(
        silver_transform,
        "assert_silver_contract",
        lambda _frame: (_ for _ in ()).throw(RuntimeError("bad contract")),
    )
    monkeypatch.setattr(
        silver_transform,
        "quality_results_dataframe",
        lambda *_args, **kwargs: kwargs["checks"],
    )
    monkeypatch.setattr(
        silver_transform,
        "merge_quality_results",
        lambda _spark, *, results, **_kwargs: persisted.extend(results),
    )

    def rejected_promotion(
        _spark: object,
        *,
        candidate: DataFrame,
        validate: Any,
        **_kwargs: object,
    ) -> None:
        validate(candidate)

    monkeypatch.setattr(
        silver_transform,
        "promote_staged_table",
        rejected_promotion,
    )
    with pytest.raises(RuntimeError, match="bad contract"):
        silver_transform.run(
            mapped,  # type: ignore[arg-type]
            catalog="crime",
            bronze_schema="bronze",
            silver_schema="silver",
            data_quality_schema="data_quality",
            pipeline_run_id="failed",
        )
    assert persisted[0].check_name == "silver_candidate_validation"  # type: ignore[union-attr]


def test_quality_stage_counts_current_run_quarantine_and_blocks(
    spark: SparkSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tables = CrimeNetTables(catalog="crime")
    bronze = spark.createDataFrame(
        [("dallas", "hash")],
        "source_system string, source_row_hash string",
    )
    crime = spark.createDataFrame([(1,)], "id long")
    observations = spark.createDataFrame(
        [
            ("run-1", "reject-1", "INVALID_COORDINATES"),
            ("other", "reject-2", "OTHER"),
        ],
        """
        pipeline_run_id string,
        quarantine_id string,
        quarantine_reason_code string
        """,
    )
    mapped = _MappedSpark(
        spark,
        {
            tables.crime_offenses_silver: crime,
            tables.dallas_bronze: bronze,
            tables.houston_bronze: bronze,
            tables.fort_worth_bronze: bronze,
            f"{tables.crime_quarantine}_observations": observations,
        },
        existing={f"{tables.crime_quarantine}_observations"},
    )
    observed: dict[str, object] = {}

    def evaluate(
        _crime: DataFrame,
        **kwargs: object,
    ) -> list[QualityCheck]:
        observed.update(kwargs)
        return [
            QualityCheck(
                check_name="forced_failure",
                severity="BLOCKING",
                passed=False,
                observed_value=1,
                expected_threshold=0,
            )
        ]

    monkeypatch.setattr(quality_checks, "evaluate_crime_quality", evaluate)
    monkeypatch.setattr(
        quality_checks,
        "quality_results_dataframe",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        quality_checks,
        "merge_quality_results",
        lambda *_args, **_kwargs: None,
    )
    with pytest.raises(RuntimeError, match="forced_failure"):
        quality_checks.run(
            mapped,  # type: ignore[arg-type]
            tables=tables,
            pipeline_run_id="run-1",
            thresholds=QualityThresholds(),
        )
    assert observed["bronze_distinct_count"] == 3
    assert observed["quarantine_count"] == 1
    assert observed["quarantine_reason_counts"] == {
        "INVALID_COORDINATES": 1
    }


def test_weather_retrieval_persists_audit_even_when_fetch_fails(
    spark: SparkSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = spark.createDataFrame([(1,)], "id long")
    mapped = _MappedSpark(
        spark,
        {"crime.ops.weather_request_manifest": manifest},
    )
    persisted: list[dict[str, object]] = []
    monkeypatch.setattr(
        weather_retrieval_job,
        "fetch_weather_manifest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    monkeypatch.setattr(
        weather_retrieval_job,
        "_persist_audit_events",
        lambda _spark, **kwargs: persisted.append(kwargs),
    )
    with pytest.raises(RuntimeError, match="offline"):
        weather_retrieval_job.run(
            mapped,  # type: ignore[arg-type]
            catalog="crime",
            ops_schema="ops",
            cache_directory="/cache",
            client_config=OpenMeteoClientConfig(),
            pipeline_run_id="run/1",
        )
    assert persisted[0]["pipeline_run_id"] == "run_1"
    assert persisted[0]["target_table"] == (
        "crime.ops.weather_request_failures"
    )
