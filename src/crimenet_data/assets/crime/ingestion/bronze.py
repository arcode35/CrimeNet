from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from crimenet_data.assets.crime.common.expressions import text_expr
from crimenet_data.assets.crime.sources.base import SourceDefinition


def attach_provenance(
    lf: pl.LazyFrame,
    *,
    source_key: str,
    run_id: str,
    ingested_at: datetime | None = None,
) -> pl.LazyFrame:
    """Attach the one canonical Bronze provenance representation."""

    timestamp = ingested_at or datetime.now(UTC)
    result = lf.with_columns(
        pl.lit(source_key).alias("source_city"),
        text_expr(
            lf,
            "landing_object_uri",
            "__landing_object_uri",
            "source_file_uri",
            "_source_file_uri",
            "source_file",
            "source_url",
        ).alias("source_file_uri"),
        pl.lit(run_id).alias("ingestion_run_id"),
        pl.lit(timestamp).cast(pl.Datetime("us", "UTC")).alias("ingested_at_utc"),
    )
    technical_columns = [
        name
        for name in ("landing_object_uri", "__landing_object_uri")
        if name in result.collect_schema().names()
    ]
    return result.drop(technical_columns) if technical_columns else result


def prepare_bronze_source(
    raw_lf: pl.LazyFrame,
    source: SourceDefinition,
    *,
    run_id: str,
    ingested_at: datetime | None = None,
) -> pl.LazyFrame:
    """Apply source-owned preparation and generic Bronze guarantees."""

    lf = source.prepare_bronze(raw_lf)
    lf = attach_provenance(
        lf,
        source_key=source.config.key,
        run_id=run_id,
        ingested_at=ingested_at,
    )
    occurrence_year = source.occurrence_timestamp(lf).dt.year().cast(pl.Int16)
    lf = lf.with_columns(occurrence_year.alias("occurrence_year"))

    keys = source.config.deduplication_keys
    if keys:
        missing = set(keys) - set(lf.collect_schema().names())
        if missing:
            raise KeyError(
                f"Cannot prepare Bronze for {source.config.key!r}; "
                f"deduplication keys are missing: {sorted(missing)}"
            )
        lf = lf.unique(subset=list(keys), keep="last", maintain_order=False)
    return lf
