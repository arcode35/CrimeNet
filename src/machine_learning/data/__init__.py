"""Shared immutable model-data contracts."""

from machine_learning.data.model_table import (
    ResolvedModelTable,
    resolve_model_table,
    resolve_model_table_from_config,
)

__all__ = [
    "ResolvedModelTable",
    "resolve_model_table",
    "resolve_model_table_from_config",
]
