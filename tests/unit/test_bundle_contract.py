from __future__ import annotations

import importlib
import re
import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_every_project_entrypoint_imports_and_is_callable() -> None:
    with (_REPOSITORY_ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)

    for script_name, target in project["project"]["scripts"].items():
        module_name, attribute_name = target.split(":", maxsplit=1)
        module = importlib.import_module(module_name)
        entrypoint = getattr(module, attribute_name)

        assert callable(entrypoint), script_name


def test_bundle_wheel_tasks_reference_declared_project_scripts() -> None:
    with (_REPOSITORY_ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)
    declared_scripts = set(project["project"]["scripts"])

    bundle_text = (
        _REPOSITORY_ROOT
        / "resources"
        / "jobs"
        / "crime_pipeline.job.yml"
    ).read_text(encoding="utf-8")
    bundle_entrypoints = set(
        re.findall(r"^\s+entry_point:\s+(\S+)\s*$", bundle_text, re.MULTILINE)
    )

    assert bundle_entrypoints
    assert bundle_entrypoints <= declared_scripts
