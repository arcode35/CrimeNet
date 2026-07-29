from __future__ import annotations

import importlib
import re
import sys
import tomllib
from collections import defaultdict, deque
from pathlib import Path
from types import ModuleType

import yaml
from pytest import MonkeyPatch

ROOT = Path(__file__).resolve().parents[2]


def _configuration() -> tuple[dict[str, str], dict[str, object]]:
    with (ROOT / "pyproject.toml").open("rb") as file:
        scripts = tomllib.load(file)["project"]["scripts"]
    with (ROOT / "resources/jobs/crime_pipeline.job.yml").open() as file:
        job = yaml.safe_load(file)["resources"]["jobs"]["crime_pipeline"]
    return scripts, job


def _module_for_entry_point(scripts: dict[str, str], name: str) -> ModuleType:
    module_name = scripts[name].partition(":")[0]
    return importlib.import_module(module_name)


def _placeholder(value: object) -> str:
    text = str(value)
    if text == "{{job.run_id}}":
        return "test-run"
    match = re.fullmatch(r"\$\{var\.([A-Za-z0-9_]+)\}", text)
    if not match:
        return text
    variable = match.group(1)
    if any(
        token in variable
        for token in (
            "count",
            "coverage",
            "rate",
            "ratio",
            "resolution",
            "vintage",
            "requests",
            "seconds",
        )
    ):
        return "1"
    values = {
        "catalog": "catalog",
        "bronze_schema": "bronze",
        "silver_schema": "silver",
        "gold_schema": "gold",
        "ops_schema": "ops",
        "data_quality_schema": "data_quality",
        "raw_files_schema": "raw_files",
        "landing_volume": "landing",
        "autoloader_schemas_volume": "schemas",
        "checkpoints_volume": "checkpoints",
        "preflight_mode": "create",
        "state_fips": "48",
        "minimum_acs_tracts_per_vintage": "1",
        "minimum_tiger_tracts_per_vintage": "1",
        "maximum_boundary_quarantine_records": "0",
        "maximum_ambiguous_spatial_matches": "0",
        "weather_model": "era5_land",
        "weather_hourly_variables": "temperature_2m",
        "lighting_definition_version": "lighting-v1",
        "boundary_definition_version": "boundary-v1",
        "mapping_definition_version": "mapping-v1",
        "open_meteo_archive_url": "https://example.test/archive",
    }
    return values.get(variable, "configured-value")


def test_bundle_graph_is_acyclic_and_references_existing_tasks() -> None:
    _, job = _configuration()
    tasks = {task["task_key"]: task for task in job["tasks"]}
    indegree = {key: 0 for key in tasks}
    dependents: dict[str, list[str]] = defaultdict(list)
    for key, task in tasks.items():
        for dependency in task.get("depends_on", []):
            dependency_key = dependency["task_key"]
            assert dependency_key in tasks
            indegree[key] += 1
            dependents[dependency_key].append(key)

    queue = deque(key for key, count in indegree.items() if count == 0)
    visited: list[str] = []
    while queue:
        current = queue.popleft()
        visited.append(current)
        for dependent in dependents[current]:
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                queue.append(dependent)
    assert set(visited) == set(tasks)
    assert tasks["gold_crime_features"]["depends_on"]


def test_bundle_tasks_have_exact_direct_dependencies() -> None:
    _, job = _configuration()
    actual = {
        task["task_key"]: {
            dependency["task_key"]
            for dependency in task.get("depends_on", [])
        }
        for task in job["tasks"]
    }
    expected = {
        "preflight": set(),
        "bronze_dallas": {"preflight"},
        "bronze_houston": {"preflight"},
        "bronze_fort_worth": {"preflight"},
        "silver_crime": {
            "bronze_dallas",
            "bronze_houston",
            "bronze_fort_worth",
        },
        "crime_quality_checks": {"silver_crime"},
        "weather_request_plan": {"crime_quality_checks"},
        "weather_api_retrieval": {"weather_request_plan"},
        "bronze_open_meteo_weather": {"weather_api_retrieval"},
        "silver_weather_hourly": {"bronze_open_meteo_weather"},
        "acs_vintage_calendar": {"preflight"},
        "tiger_line_boundaries": {"acs_vintage_calendar"},
        "land_acs5_tracts": {"preflight"},
        "bronze_acs5_tracts": {"land_acs5_tracts"},
        "silver_acs5_tracts": {"bronze_acs5_tracts"},
        "silver_lighting_conditions": {"crime_quality_checks"},
        "location_tract_mapping": {
            "crime_quality_checks",
            "acs_vintage_calendar",
            "tiger_line_boundaries",
        },
        "gold_crime_features": {
            "crime_quality_checks",
            "silver_weather_hourly",
            "silver_acs5_tracts",
            "silver_lighting_conditions",
            "location_tract_mapping",
        },
    }
    assert actual == expected


def test_every_bundle_entry_point_exists_and_accepts_its_arguments(
    monkeypatch: MonkeyPatch,
) -> None:
    scripts, job = _configuration()
    for task in job["tasks"]:
        wheel_task = task["python_wheel_task"]
        entry_point = wheel_task["entry_point"]
        assert entry_point in scripts, task["task_key"]
        module = _module_for_entry_point(scripts, entry_point)
        parameters = [_placeholder(value) for value in wheel_task["parameters"]]
        monkeypatch.setattr(
            sys,
            "argv",
            [entry_point, *parameters],
        )
        parsed = module.parse_args()
        assert parsed is not None


def test_all_public_entry_point_modules_import_and_are_implemented() -> None:
    scripts, _ = _configuration()
    for path in scripts.values():
        module_name, _, callable_name = path.partition(":")
        module = importlib.import_module(module_name)
        assert callable(getattr(module, callable_name))

    production_source = "\n".join(
        path.read_text()
        for path in (ROOT / "src/crimenet").rglob("*.py")
    )
    assert "NotImplementedError" not in production_source


def test_bundle_variables_targets_and_serverless_environment_are_consistent() -> None:
    with (ROOT / "databricks.yml").open() as file:
        bundle = yaml.safe_load(file)
    target_documents = []
    for path in sorted((ROOT / "targets").glob("*.yml")):
        with path.open() as file:
            target_documents.append(yaml.safe_load(file))

    declared_variables = set(bundle["variables"])
    configured_targets = {
        target_name: target
        for document in target_documents
        for target_name, target in document["targets"].items()
    }
    for target_name, target in configured_targets.items():
        assert set(target.get("variables", {})) <= declared_variables, target_name

    _, job = _configuration()
    serialized_job = yaml.safe_dump(job)
    referenced_variables = set(
        re.findall(r"\$\{var\.([A-Za-z0-9_]+)\}", serialized_job)
    )
    assert referenced_variables <= declared_variables

    environments = {
        environment["environment_key"]: environment["spec"]
        for environment in job["environments"]
    }
    assert environments["default"]["dependencies"] == [
        "../../dist/*.whl",
        "-r /Workspace/${workspace.file_path}/requirements-prod.txt",
    ]
    for task in job["tasks"]:
        assert task["environment_key"] in environments

    production = configured_targets["prod"]
    assert "service_principal_name" in production["run_as"]
    assert "current_user" not in yaml.safe_dump(production)

    development = configured_targets["dev"]
    assert development["variables"]["preflight_mode"] == "create"
    assert production["variables"]["preflight_mode"] == "validate"
    for target in (development, production):
        assert {
            "minimum_acs_tracts_per_vintage",
            "minimum_tiger_tracts_per_vintage",
            "maximum_boundary_quarantine_records",
            "maximum_ambiguous_spatial_matches",
        } <= set(target["variables"])
