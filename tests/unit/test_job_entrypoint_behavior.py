from __future__ import annotations

import sys
from typing import Any

from pyspark.sql import SparkSession

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


def _invoke_main(
    monkeypatch: Any,
    module: Any,
    arguments: list[str],
    *,
    target: str = "run",
) -> dict[str, object]:
    captured: dict[str, object] = {}

    def recorder(*args: object, **kwargs: object) -> None:
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr(sys, "argv", [module.__name__, *arguments])
    monkeypatch.setattr(module, target, recorder)
    module.main()
    return captured


def test_landing_and_core_medallion_entrypoints_wire_cli_values(
    spark: SparkSession,
    monkeypatch: Any,
) -> None:
    assert SparkSession.getActiveSession() is spark

    landing = _invoke_main(
        monkeypatch,
        acs5_landing_job,
        [
            "--landing-path",
            "/landing/acs",
            "--start-vintage",
            "2022",
            "--end-vintage",
            "2023",
            "--pipeline-run-id",
            "run/1",
        ],
        target="ingest_acs5_tract_vintages",
    )
    assert landing["kwargs"]["start_vintage"] == 2022  # type: ignore[index]
    assert landing["kwargs"]["api_key"] is None  # type: ignore[index]
    assert landing["kwargs"]["minimum_record_count"] == 1  # type: ignore[index]

    bronze = _invoke_main(
        monkeypatch,
        bronze_ingestion,
        [
            "--catalog",
            "crime",
            "--source",
            "dallas",
            "--input-path",
            "/landing/dallas",
        ],
    )
    assert bronze["kwargs"]["source"] == "dallas"  # type: ignore[index]

    silver = _invoke_main(
        monkeypatch,
        silver_transform,
        ["--catalog", "crime", "--pipeline-run-id", "run-1"],
    )
    assert silver["kwargs"]["catalog"] == "crime"  # type: ignore[index]
    assert silver["kwargs"]["thresholds"].minimum_row_count == 1  # type: ignore[index,union-attr]

    quality = _invoke_main(
        monkeypatch,
        quality_checks,
        [
            "--catalog",
            "crime",
            "--silver-schema",
            "silver",
            "--data-quality-schema",
            "quality",
            "--pipeline-run-id",
            "run-1",
        ],
    )
    assert quality["kwargs"]["pipeline_run_id"] == "run_1"  # type: ignore[index]
    assert quality["kwargs"]["tables"].quality_results == (  # type: ignore[index,union-attr]
        "crime.quality.quality_results"
    )


def test_dependency_and_enrichment_entrypoints_wire_versions(
    spark: SparkSession,
    monkeypatch: Any,
) -> None:
    calendar = _invoke_main(
        monkeypatch,
        acs_vintage_calendar_job,
        [
            "--catalog",
            "crime",
            "--start-vintage",
            "2021",
            "--end-vintage",
            "2023",
            "--pipeline-run-id",
            "calendar/1",
        ],
    )
    assert calendar["kwargs"]["pipeline_run_id"] == "calendar_1"  # type: ignore[index]

    tiger = _invoke_main(
        monkeypatch,
        tiger_line_boundaries_job,
        [
            "--catalog",
            "crime",
            "--landing-path",
            "/landing/tiger",
            "--pipeline-run-id",
            "tiger/1",
        ],
    )
    assert tiger["kwargs"]["state_fips"] == "48"  # type: ignore[index]
    assert tiger["kwargs"]["pipeline_run_id"] == "tiger_1"  # type: ignore[index]

    mapping = _invoke_main(
        monkeypatch,
        location_tract_mapping_job,
        [
            "--catalog",
            "crime",
            "--full-rebuild",
            "--pipeline-run-id",
            "map/1",
        ],
    )
    assert mapping["kwargs"]["full_rebuild"] is True  # type: ignore[index]

    lighting = _invoke_main(
        monkeypatch,
        silver_lighting_job,
        [
            "--catalog",
            "crime",
            "--full-rebuild",
            "--pipeline-run-id",
            "light/1",
        ],
    )
    assert lighting["kwargs"]["pipeline_run_id"] == "light_1"  # type: ignore[index]

    socioeconomic = _invoke_main(
        monkeypatch,
        silver_socioeconomic_job,
        [
            "--catalog",
            "crime",
            "--full-rebuild",
            "--pipeline-run-id",
            "acs/1",
        ],
    )
    assert socioeconomic["kwargs"]["full_rebuild"] is True  # type: ignore[index]

    weather = _invoke_main(
        monkeypatch,
        silver_weather_job,
        [
            "--catalog",
            "crime",
            "--checkpoint-path",
            "/state/weather",
            "--pipeline-run-id",
            "weather/1",
        ],
    )
    assert weather["kwargs"]["checkpoint_path"] == "/state/weather"  # type: ignore[index]


def test_weather_gold_and_preflight_entrypoints_parse_operational_settings(
    spark: SparkSession,
    monkeypatch: Any,
) -> None:
    preflight_call = _invoke_main(
        monkeypatch,
        preflight,
        [
            "--catalog",
            "crime",
            "--validate-only",
            "--pipeline-run-id",
            "preflight/1",
        ],
    )
    assert preflight_call["kwargs"]["validate_only"] is True  # type: ignore[index]
    assert len(preflight_call["kwargs"]["volumes"]) == 3  # type: ignore[arg-type,index]

    configured_preflight_call = _invoke_main(
        monkeypatch,
        preflight,
        [
            "--catalog",
            "crime",
            "--preflight-mode",
            "validate",
            "--pipeline-run-id",
            "preflight/2",
        ],
    )
    assert configured_preflight_call["kwargs"]["validate_only"] is True  # type: ignore[index]

    create_preflight_call = _invoke_main(
        monkeypatch,
        preflight,
        [
            "--catalog",
            "crime",
            "--preflight-mode",
            "create",
            "--pipeline-run-id",
            "preflight/3",
        ],
    )
    assert create_preflight_call["kwargs"]["validate_only"] is False  # type: ignore[index]

    planned = _invoke_main(
        monkeypatch,
        weather_request_planner_job,
        [
            "--catalog",
            "crime",
            "--hourly-variables",
            "temperature_2m, dew_point_2m",
            "--availability-cutoff",
            "2024-06-30",
        ],
    )
    assert planned["kwargs"]["hourly_variables"] == (  # type: ignore[index]
        "temperature_2m",
        "dew_point_2m",
    )
    assert str(planned["kwargs"]["availability_cutoff"]) == "2024-06-30"  # type: ignore[index]

    retrieval = _invoke_main(
        monkeypatch,
        weather_retrieval_job,
        [
            "--catalog",
            "crime",
            "--cache-directory",
            "/cache/weather",
            "--max-retries",
            "3",
            "--max-concurrent-requests",
            "4",
        ],
    )
    config = retrieval["kwargs"]["client_config"]  # type: ignore[index]
    assert config.max_retries == 3  # type: ignore[union-attr]
    assert config.max_concurrent_requests == 4  # type: ignore[union-attr]

    gold = _invoke_main(
        monkeypatch,
        gold_crime_features_job,
        [
            "--catalog",
            "crime",
            "--minimum-weather-coverage",
            "0.9",
            "--minimum-lighting-coverage",
            "0.8",
            "--minimum-acs-coverage",
            "0.7",
            "--minimum-tract-coverage",
            "0.6",
            "--pipeline-run-id",
            "gold/1",
        ],
    )
    assert gold["kwargs"]["minimum_weather_coverage"] == 0.9  # type: ignore[index]
    assert gold["kwargs"]["pipeline_run_id"] == "gold_1"  # type: ignore[index]
