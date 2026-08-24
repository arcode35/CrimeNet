from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import polars as pl

Severity = Literal["error", "warn"]
ExpressionFactory = Callable[[], pl.Expr]


@dataclass(frozen=True)
class ContractRule:
    name: str
    rule: ExpressionFactory
    severity: Severity = "error"
    description: str = ""


@dataclass(frozen=True)
class DataContract:
    name: str
    version: str
    description: str
    grain: str

    schema: Mapping[str, Any]

    required_columns: frozenset[str]
    non_nullable_columns: frozenset[str]

    primary_key: tuple[str, ...] = ()
    partition_by: tuple[str, ...] = ()

    allow_extra_columns: bool = False

    rules: tuple[ContractRule, ...] = ()

    owner: str | None = None
