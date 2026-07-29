from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from crimenet.boundaries.tiger_line import BOUNDARY_DEFINITION_VERSION
from crimenet.config.resources import CrimeNetTables
from crimenet.config.validation import QualityThresholds
from crimenet.contracts.gold import crime_offense_id_from_business_identity
from crimenet.contracts.lighting import LIGHTING_DEFINITION_VERSION
from crimenet.gold import crime_features
from crimenet.ingestion.metadata import add_ingestion_metadata
from crimenet.jobs.bronze_ingestion import merge_bronze_batch
from crimenet.jobs.bronze_ingestion import run as run_bronze
from crimenet.jobs.gold_crime_features_job import run as run_gold
from crimenet.jobs.silver_transform import run as run_silver
from crimenet.quality.quarantine import (
    merge_quarantine,
    split_crime_quarantine,
)
from crimenet.spatial.tract_mapping import (
    MAPPING_DEFINITION_VERSION,
    location_tract_key,
)
from crimenet.utils.promotion import (
    promote_staged_table,
    staging_table_name,
)

pytestmark = pytest.mark.delta

FIXTURES = Path(__file__).parents[1] / "fixtures" / "logical_e2e"


@pytest.fixture
def delta_schemas(
    delta_spark: SparkSession,
) -> Iterator[tuple[str, str, str]]:
    suffix = uuid4().hex[:12]
    schemas = (
        f"bronze_it_{suffix}",
        f"silver_it_{suffix}",
        f"quality_it_{suffix}",
    )
    for schema in schemas:
        delta_spark.sql(f"CREATE DATABASE spark_catalog.{schema}")
    yield schemas
    for schema in reversed(schemas):
        delta_spark.sql(
            f"DROP DATABASE IF EXISTS spark_catalog.{schema} CASCADE"
        )


@pytest.fixture
def gold_schema(delta_spark: SparkSession) -> Iterator[str]:
    schema = f"gold_it_{uuid4().hex[:12]}"
    delta_spark.sql(f"CREATE DATABASE spark_catalog.{schema}")
    yield schema
    delta_spark.sql(
        f"DROP DATABASE IF EXISTS spark_catalog.{schema} CASCADE"
    )


def _table(schema: str, name: str) -> str:
    return f"spark_catalog.{schema}.{name}"


def _bronze_rows(
    spark: SparkSession,
    *,
    source_file: str,
    values: tuple[str, ...],
) -> DataFrame:
    raw = (
        spark.createDataFrame(
            [(value, source_file) for value in values],
            "value STRING, source_file STRING",
        )
    )
    return add_ingestion_metadata(
        raw,
        "dallas",
        contract_version="municipal_crime_v1",
    )


def test_bronze_merge_is_stable_after_batch_reset_and_moved_path(
    delta_spark: SparkSession,
    delta_schemas: tuple[str, str, str],
) -> None:
    bronze_schema, _, _ = delta_schemas
    target = _table(bronze_schema, "bronze_replay")

    merge_bronze_batch(
        _bronze_rows(
            delta_spark,
            source_file="/landing/original/crime.csv",
            values=("same-record",),
        ),
        batch_id=17,
        spark=delta_spark,
        target_table=target,
    )
    # Resetting the batch ID models a lost streaming checkpoint. The identical
    # logical record is also presented from a different landing path.
    merge_bronze_batch(
        _bronze_rows(
            delta_spark,
            source_file="/landing/recovered/crime.csv",
            values=("same-record", "new-record"),
        ),
        batch_id=0,
        spark=delta_spark,
        target_table=target,
    )

    rows = {
        row["value"]: row.asDict()
        for row in delta_spark.table(target).collect()
    }
    assert set(rows) == {"same-record", "new-record"}
    assert rows["same-record"]["source_file"] == (
        "/landing/original/crime.csv"
    )
    assert len({row["source_row_hash"] for row in rows.values()}) == 2


def _single_reason_reject(spark: SparkSession) -> DataFrame:
    return spark.createDataFrame(
        [
            (
                "dallas",
                None,
                "offense-1",
                "business-1",
                "source-hash-1",
                datetime(2024, 5, 3, 19, 37),
                32.7767,
                -96.7970,
                None,
                "/landing/crime.csv",
            )
        ],
        """
        source_system STRING,
        source_incident_id STRING,
        source_offense_id STRING,
        business_identity STRING,
        source_row_hash STRING,
        occurred_at TIMESTAMP,
        latitude DOUBLE,
        longitude DOUBLE,
        source_corrupt_record STRING,
        source_file STRING
        """,
    )


def test_quarantine_retry_and_new_run_observations_are_idempotent(
    delta_spark: SparkSession,
    delta_schemas: tuple[str, str, str],
) -> None:
    _, _, quality_schema = delta_schemas
    target = _table(quality_schema, "crime_quarantine")
    observations = f"{target}_observations"
    _, first_run = split_crime_quarantine(
        _single_reason_reject(delta_spark),
        pipeline_run_id="run-1",
    )

    merge_quarantine(
        delta_spark,
        quarantine=first_run,
        target_table=target,
    )
    merge_quarantine(
        delta_spark,
        quarantine=first_run,
        target_table=target,
    )

    assert delta_spark.table(target).count() == 1
    assert delta_spark.table(observations).count() == 1
    first_entity = delta_spark.table(target).first()
    assert first_entity is not None
    assert first_entity["first_seen_pipeline_run_id"] == "run-1"

    _, second_run = split_crime_quarantine(
        _single_reason_reject(delta_spark),
        pipeline_run_id="run-2",
    )
    merge_quarantine(
        delta_spark,
        quarantine=second_run,
        target_table=target,
    )

    assert delta_spark.table(target).count() == 1
    sightings = delta_spark.table(observations).groupBy(
        "pipeline_run_id"
    ).count().collect()
    assert {row["pipeline_run_id"]: row["count"] for row in sightings} == {
        "run-1": 1,
        "run-2": 1,
    }


def test_staged_promotion_preserves_last_good_then_promotes_success(
    delta_spark: SparkSession,
    delta_schemas: tuple[str, str, str],
) -> None:
    _, silver_schema, _ = delta_schemas
    target = _table(silver_schema, "promotion_target")
    delta_spark.createDataFrame(
        [(1, "last-good")],
        "id LONG, value STRING",
    ).write.format("delta").saveAsTable(target)

    def reject_candidate(_: DataFrame) -> None:
        raise RuntimeError("candidate failed validation")

    with pytest.raises(RuntimeError, match="failed validation"):
        promote_staged_table(
            delta_spark,
            candidate=delta_spark.createDataFrame(
                [(2, "invalid")],
                "id LONG, value STRING",
            ),
            target_table=target,
            pipeline_run_id="failed-run",
            validate=reject_candidate,
        )

    assert delta_spark.table(target).collect()[0].asDict() == {
        "id": 1,
        "value": "last-good",
    }
    failed_stage = staging_table_name(target, "failed-run")
    assert not delta_spark.catalog.tableExists(failed_stage)

    promote_staged_table(
        delta_spark,
        candidate=delta_spark.createDataFrame(
            [(2, "promoted"), (3, "also-promoted")],
            "id LONG, value STRING",
        ),
        target_table=target,
        pipeline_run_id="successful-run",
        validate=lambda candidate: (
            None
            if candidate.count() == 2
            else pytest.fail("unexpected candidate")
        ),
    )

    promoted = {
        row["id"]: row["value"]
        for row in delta_spark.table(target).collect()
    }
    assert promoted == {2: "promoted", 3: "also-promoted"}
    successful_stage = staging_table_name(target, "successful-run")
    assert not delta_spark.catalog.tableExists(successful_stage)


def _ingest_fixtures(
    spark: SparkSession,
    *,
    bronze_schema: str,
    root: Path,
) -> None:
    for source, extension in (
        ("dallas", "csv"),
        ("houston", "csv"),
        ("fort_worth", "jsonl"),
    ):
        run_bronze(
            spark,
            catalog="spark_catalog",
            bronze_schema=bronze_schema,
            source=source,
            input_path=str(root / source / f"crime.{extension}"),
            write_mode="merge",
        )


def _silver_snapshot(
    spark: SparkSession,
    table: str,
) -> list[dict[str, object]]:
    return [
        row.asDict(recursive=True)
        for row in spark.table(table).orderBy("business_identity").collect()
    ]


def _save_delta(dataframe: DataFrame, target_table: str) -> None:
    (
        dataframe.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(target_table)
    )


def _load_gold_dependency_fixtures(
    spark: SparkSession,
    *,
    silver_schema: str,
    crime_table: str,
) -> None:
    dependencies = FIXTURES / "dependencies"
    calendar = spark.read.json(
        str(dependencies / "acs_vintage_calendar.jsonl")
    ).select(
        F.col("acs_vintage").cast("int").alias("acs_vintage"),
        F.to_date("acs_release_date").alias("acs_release_date"),
        F.col("tiger_line_year").cast("int").alias("tiger_line_year"),
        F.col("tract_definition_vintage")
        .cast("int")
        .alias("tract_definition_vintage"),
    )
    _save_delta(
        calendar,
        _table(silver_schema, "acs_vintage_calendar"),
    )

    socioeconomic = spark.read.json(
        str(dependencies / "tract_socioeconomic.jsonl")
    ).select(
        F.col("geoid").cast("string").alias("geoid"),
        F.col("acs_vintage").cast("int").alias("acs_vintage"),
        F.col("geography_name").cast("string").alias("geography_name"),
        F.col("population").cast("long").alias("population"),
        F.col("population_moe").cast("long").alias("population_moe"),
        F.col("median_age").cast("double").alias("median_age"),
        F.col("median_age_moe").cast("double").alias("median_age_moe"),
        F.col("median_household_income")
        .cast("double")
        .alias("median_household_income"),
        F.col("median_household_income_moe")
        .cast("double")
        .alias("median_household_income_moe"),
        *[
            F.col(name).cast("double").alias(name)
            for name in (
                "poverty_rate",
                "unemployment_rate",
                "vacancy_rate",
                "renter_occupied_rate",
                "no_vehicle_rate",
            )
        ],
    )
    _save_delta(
        socioeconomic,
        _table(silver_schema, "tract_socioeconomic"),
    )

    weather = spark.read.json(
        str(dependencies / "weather_hourly.jsonl")
    ).select(
        F.col("provider").cast("string").alias("provider"),
        F.col("model").cast("string").alias("model"),
        F.col("h3_resolution").cast("int").alias("h3_resolution"),
        F.col("weather_query_cell_id")
        .cast("long")
        .alias("weather_query_cell_id"),
        F.to_timestamp("weather_timestamp").alias("weather_timestamp"),
        F.to_date("weather_date").alias("weather_date"),
        F.col("temperature_2m_c")
        .cast("double")
        .alias("temperature_2m_c"),
        F.col("grid_elevation").cast("double").alias("grid_elevation"),
    )
    _save_delta(weather, _table(silver_schema, "weather_hourly"))

    lighting = spark.read.json(
        str(dependencies / "solar_lighting_conditions.jsonl")
    ).select(
        F.col("lighting_query_cell_id")
        .cast("long")
        .alias("lighting_query_cell_id"),
        F.to_timestamp("solar_timestamp_hour").alias(
            "solar_timestamp_hour"
        ),
        F.col("lighting_definition_version")
        .cast("string")
        .alias("lighting_definition_version"),
        *[
            F.col(name).cast("double").alias(name)
            for name in (
                "solar_elevation_deg",
                "apparent_solar_elevation_deg",
                "solar_zenith_deg",
                "solar_azimuth_deg",
            )
        ],
        F.col("lighting_condition")
        .cast("string")
        .alias("lighting_condition"),
        F.col("is_daylight").cast("boolean").alias("is_daylight"),
    )
    _save_delta(
        lighting,
        _table(silver_schema, "solar_lighting_conditions"),
    )

    boundary_path = dependencies / "tract_boundaries.geojson"
    boundary_document = json.loads(boundary_path.read_text(encoding="utf-8"))
    boundary_features = boundary_document["features"]
    archive_sha256 = hashlib.sha256(boundary_path.read_bytes()).hexdigest()
    boundary_rows = [
        (
            int(feature["properties"]["boundary_vintage"]),
            str(feature["properties"]["geoid"]),
            str(feature["properties"]["name"]),
            json.dumps(
                feature["geometry"],
                sort_keys=True,
                separators=(",", ":"),
            ),
            BOUNDARY_DEFINITION_VERSION,
            archive_sha256,
        )
        for feature in boundary_features
    ]
    boundaries = spark.createDataFrame(
        boundary_rows,
        """
        boundary_vintage INT,
        geoid STRING,
        tract_name STRING,
        tract_geometry_geojson STRING,
        boundary_definition_version STRING,
        source_archive_sha256 STRING
        """,
    )
    _save_delta(
        boundaries,
        _table(silver_schema, "census_tract_boundaries"),
    )

    mapping_rows: list[tuple[object, ...]] = []
    locations = spark.table(crime_table).select(
        "latitude",
        "longitude",
    ).collect()
    for location in locations:
        latitude = float(location["latitude"])
        longitude = float(location["longitude"])
        matches = []
        for feature in boundary_features:
            ring = feature["geometry"]["coordinates"][0]
            longitudes = [float(point[0]) for point in ring]
            latitudes = [float(point[1]) for point in ring]
            if (
                min(longitudes) <= longitude <= max(longitudes)
                and min(latitudes) <= latitude <= max(latitudes)
            ):
                matches.append(feature)
        assert len(matches) == 1
        feature = matches[0]
        tiger_line_year = int(
            feature["properties"]["boundary_vintage"]
        )
        mapping_rows.append(
            (
                tiger_line_year,
                latitude,
                longitude,
                str(feature["properties"]["geoid"]),
                "matched_contains",
                1,
                BOUNDARY_DEFINITION_VERSION,
                archive_sha256,
                MAPPING_DEFINITION_VERSION,
                location_tract_key(
                    tiger_line_year=tiger_line_year,
                    latitude=latitude,
                    longitude=longitude,
                    boundary_definition_version=(
                        BOUNDARY_DEFINITION_VERSION
                    ),
                    source_archive_sha256=archive_sha256,
                    mapping_definition_version=(
                        MAPPING_DEFINITION_VERSION
                    ),
                ),
                "fixture-boundary-map",
                datetime(2024, 6, 1),
            )
        )
    mapping = spark.createDataFrame(
        mapping_rows,
        """
        tiger_line_year INT,
        latitude DOUBLE,
        longitude DOUBLE,
        tract_geoid STRING,
        match_status STRING,
        candidate_match_count INT,
        boundary_definition_version STRING,
        source_archive_sha256 STRING,
        mapping_definition_version STRING,
        location_tract_key STRING,
        pipeline_run_id STRING,
        mapped_at TIMESTAMP
        """,
    )
    _save_delta(
        mapping,
        _table(silver_schema, "crime_location_tract_mapping"),
    )


def _gold_snapshot(
    spark: SparkSession,
    table: str,
) -> list[dict[str, object]]:
    stable_columns = (
        "crime_offense_id",
        "business_identity",
        "source_system",
        "selected_acs_vintage",
        "tract_geoid",
        "median_household_income",
        "temperature_2m_c",
        "grid_elevation",
        "lighting_condition",
        "is_daylight",
        "weather_match_found",
        "lighting_match_found",
        "socioeconomic_match_found",
        "boundary_definition_version",
        "mapping_definition_version",
        "location_tract_key",
    )
    return [
        row.asDict(recursive=True)
        for row in spark.table(table)
        .select(*stable_columns)
        .orderBy("crime_offense_id")
        .collect()
    ]


def _promote_gold_for_local_delta(
    spark: SparkSession,
    *,
    staging_table: str,
    target_table: str,
) -> None:
    # Databricks supports DEEP CLONE, while OSS Delta does not parse it.
    # This adapter keeps the local test on real Delta tables and replaces only
    # the Databricks-specific final promotion statement.
    spark.sql(
        f"CREATE OR REPLACE TABLE {target_table} "
        f"USING DELTA AS SELECT * FROM {staging_table}"
    )


def test_logical_fixture_pipeline_is_stable_across_reingestion_and_rerun(
    delta_spark: SparkSession,
    delta_schemas: tuple[str, str, str],
    gold_schema: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bronze_schema, silver_schema, quality_schema = delta_schemas
    tables = CrimeNetTables(
        catalog="spark_catalog",
        bronze_schema=bronze_schema,
        silver_schema=silver_schema,
        data_quality_schema=quality_schema,
    )
    thresholds = QualityThresholds(maximum_quarantine_rate=0.0)
    monkeypatch.setattr(
        crime_features,
        "promote_staged_delta_table",
        _promote_gold_for_local_delta,
    )

    _ingest_fixtures(
        delta_spark,
        bronze_schema=bronze_schema,
        root=FIXTURES,
    )
    run_silver(
        delta_spark,
        catalog="spark_catalog",
        bronze_schema=bronze_schema,
        silver_schema=silver_schema,
        data_quality_schema=quality_schema,
        pipeline_run_id="logical-run-1",
        thresholds=thresholds,
    )
    first_snapshot = _silver_snapshot(
        delta_spark,
        tables.crime_offenses_silver,
    )
    assert len(first_snapshot) == 3
    assert {row["source_system"] for row in first_snapshot} == {
        "dallas",
        "houston",
        "fort_worth",
    }
    quarantine_tables = (
        tables.crime_quarantine,
        f"{tables.crime_quarantine}_observations",
    )
    assert all(
        not delta_spark.catalog.tableExists(table)
        for table in quarantine_tables
    )
    _load_gold_dependency_fixtures(
        delta_spark,
        silver_schema=silver_schema,
        crime_table=tables.crime_offenses_silver,
    )
    assert delta_spark.table(
        _table(silver_schema, "census_tract_boundaries")
    ).count() == 3
    run_gold(
        delta_spark,
        catalog="spark_catalog",
        silver_schema=silver_schema,
        gold_schema=gold_schema,
        data_quality_schema=quality_schema,
        weather_provider="open_meteo",
        weather_model="era5_land",
        weather_h3_resolution=6,
        lighting_definition_version=LIGHTING_DEFINITION_VERSION,
        minimum_weather_coverage=1.0,
        minimum_lighting_coverage=1.0,
        minimum_acs_coverage=1.0,
        minimum_tract_coverage=1.0,
        pipeline_run_id="logical-run-1",
    )
    gold_table = _table(gold_schema, "crime_features")
    first_gold_snapshot = _gold_snapshot(delta_spark, gold_table)
    assert len(first_gold_snapshot) == 3
    expected_gold_ids = {
        crime_offense_id_from_business_identity(
            str(row["business_identity"])
        )
        for row in first_snapshot
    }
    assert {
        row["crime_offense_id"]
        for row in first_gold_snapshot
    } == expected_gold_ids
    assert all(
        row["weather_match_found"]
        and row["lighting_match_found"]
        and row["socioeconomic_match_found"]
        for row in first_gold_snapshot
    )
    assert (
        delta_spark.table(tables.quality_results)
        .filter(F.col("pipeline_run_id") == "logical_run_1")
        .filter(~F.col("passed"))
        .count()
        == 0
    )

    moved_root = tmp_path / "recovered_landing"
    shutil.copytree(FIXTURES, moved_root)
    _ingest_fixtures(
        delta_spark,
        bronze_schema=bronze_schema,
        root=moved_root,
    )
    assert {
        source: delta_spark.table(
            tables.bronze_for_source(source)
        ).count()
        for source in ("dallas", "houston", "fort_worth")
    } == {"dallas": 1, "houston": 1, "fort_worth": 1}

    run_silver(
        delta_spark,
        catalog="spark_catalog",
        bronze_schema=bronze_schema,
        silver_schema=silver_schema,
        data_quality_schema=quality_schema,
        pipeline_run_id="logical-run-2",
        thresholds=thresholds,
    )
    assert _silver_snapshot(
        delta_spark,
        tables.crime_offenses_silver,
    ) == first_snapshot
    run_gold(
        delta_spark,
        catalog="spark_catalog",
        silver_schema=silver_schema,
        gold_schema=gold_schema,
        data_quality_schema=quality_schema,
        weather_provider="open_meteo",
        weather_model="era5_land",
        weather_h3_resolution=6,
        lighting_definition_version=LIGHTING_DEFINITION_VERSION,
        minimum_weather_coverage=1.0,
        minimum_lighting_coverage=1.0,
        minimum_acs_coverage=1.0,
        minimum_tract_coverage=1.0,
        pipeline_run_id="logical-run-2",
    )
    assert _gold_snapshot(
        delta_spark,
        gold_table,
    ) == first_gold_snapshot
    assert all(
        not delta_spark.catalog.tableExists(table)
        for table in quarantine_tables
    )
    assert (
        delta_spark.table(tables.quality_results)
        .filter(~F.col("passed"))
        .count()
        == 0
    )
    quality_run_ids = {
        row["pipeline_run_id"]
        for row in delta_spark.table(tables.quality_results)
        .select("pipeline_run_id")
        .distinct()
        .collect()
    }
    assert quality_run_ids == {"logical_run_1", "logical_run_2"}
    gold_quality_run_ids = {
        row["pipeline_run_id"]
        for row in delta_spark.table(tables.quality_results)
        .filter(F.col("table_name") == gold_table)
        .select("pipeline_run_id")
        .distinct()
        .collect()
    }
    assert gold_quality_run_ids == {
        "logical_run_1",
        "logical_run_2",
    }
