"""Column-name normalization for source datasets."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping

from pyspark.sql import DataFrame


def normalize_column_name(name: str) -> str:
    """Convert a source column name to a Delta-safe snake_case identifier."""
    normalized = (
        name.replace("\ufeff", "")
        .replace("\xa0", " ")
    )
    normalized = unicodedata.normalize("NFKD", normalized).lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")

    if not normalized:
        normalized = "unnamed_column"

    if normalized[0].isdigit():
        normalized = f"column_{normalized}"

    return normalized


def normalized_column_names(
    column_names: list[str],
    overrides: Mapping[str, str] | None = None,
) -> list[str]:
    """
    Normalize names and deterministically suffix unknown collisions.

    Known source collisions should use explicit overrides so their meaning does
    not depend on source-column order.
    """
    overrides = overrides or {}
    counts: defaultdict[str, int] = defaultdict(int)
    output: list[str] = []

    for original_name in column_names:
        requested_name = overrides.get(
            original_name,
            normalize_column_name(original_name),
        )
        base_name = normalize_column_name(requested_name)
        counts[base_name] += 1
        occurrence = counts[base_name]

        output.append(
            base_name if occurrence == 1 else f"{base_name}_{occurrence}"
        )

    return output


def normalize_column_names(
    dataframe: DataFrame,
    overrides: Mapping[str, str] | None = None,
) -> DataFrame:
    """Return a DataFrame with Delta-safe, collision-free column names."""
    return dataframe.toDF(
        *normalized_column_names(dataframe.columns, overrides=overrides)
    )
