"""Pipeline run identity shared by job entry points."""

from __future__ import annotations

import os
from uuid import uuid4

from crimenet.config.validation import normalize_pipeline_run_id

_RUN_ID_ENVIRONMENT_VARIABLES = (
    "CRIMENET_PIPELINE_RUN_ID",
    "DATABRICKS_JOB_RUN_ID",
    "DB_JOB_RUN_ID",
)


def resolve_pipeline_run_id(explicit_value: str | None = None) -> str:
    """Resolve a stable job-run identifier, with a UUID for local execution."""
    candidates = (
        explicit_value,
        *(os.environ.get(name) for name in _RUN_ID_ENVIRONMENT_VARIABLES),
    )
    for candidate in candidates:
        if candidate and candidate.strip():
            return normalize_pipeline_run_id(candidate)
    return uuid4().hex
