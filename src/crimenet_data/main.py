#!/usr/bin/env python3
from __future__ import annotations

import polars as pl
import polars_h3 as plh3

from crimenet_data.resources.crime_lake import CrimeLakeResources


ANNUAL_FEATURE_ROOT = (
    "s3://crimenet-data/gold/national_feature_store/"
    "temporal/h3_r9/annual"
)

MODELED_ONLY = True

pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(80)


def pct(numerator: str, denominator: str, alias: str) -> pl.Expr:
    return (
        pl.when(pl.col(denominator) > 0)
        .then(
            pl.col(numerator).cast(pl.Float64)
            / pl.col(denominator).cast(pl.Float64)
            * 100.0
        )
        .otherwise(None)
        .alias(alias)
    )


def weighted_sum(condition: pl.Expr, weight: str, alias: str) -> pl.Expr:
    return (
        pl.when(condition)
        .then(pl.col(weight))
        .otherwise(0)
        .sum()
        .alias(alias)
    )


def main() -> None:
    lake = CrimeLakeResources()

    print("=" * 100)
    print("SILVER -> NATIONAL TEMPORAL FEATURE STORE COVERAGE (FAST)")
    print("=" * 100)

    # ------------------------------------------------------------------
    # 1. Resolve Silver once. Row counts come from the already-published
    #    manifest, avoiding an otherwise redundant full Silver scan.
    # ------------------------------------------------------------------
    silver = lake.scan_silver_snapshot()
    manifest = lake.read_silver_manifest()

    print("\nSILVER COUNTS")
    print(
        {
            "silver_rows": int(manifest["row_count"]),
            "modeled_rows": int(manifest["include_in_model_rows"]),
        }
    )

    events = silver
    if MODELED_ONLY:
        events = events.filter(
            pl.col("include_in_model").fill_null(False)
        )

    # ------------------------------------------------------------------
    # 2. Compute H3, then IMMEDIATELY collapse 16M+ events to one row per
    #    (city, year, H3 cell). The event_count column preserves exact
    #    event-weighted coverage statistics.
    # ------------------------------------------------------------------
    crime_cell_years = (
        events
        .select(
            "source_city",
            "occurrence_year",
            "latitude",
            "longitude",
        )
        .with_columns(
            plh3.latlng_to_cell(
                "latitude",
                "longitude",
                resolution=9,
                return_dtype=pl.UInt64,
            ).alias("osm_h3_cell_id")
        )
        .group_by(
            [
                "source_city",
                "occurrence_year",
                "osm_h3_cell_id",
            ]
        )
        .agg(
            pl.len().alias("event_count")
        )
    )

    # ------------------------------------------------------------------
    # 3. Read ONLY the annual columns needed to answer coverage.
    #
    # feature_available_at / feature_version_id are deliberately omitted:
    # they were projected in the old script but never used.
    # ------------------------------------------------------------------
    annual_keys = (
        pl.scan_parquet(
            f"{ANNUAL_FEATURE_ROOT}/as_of_year=*/part-*.parquet",
            storage_options=lake.storage_options,
            credential_provider=None,
            hive_partitioning=False,
        )
        .select(
            pl.col("osm_h3_cell_id")
            .cast(pl.UInt64, strict=False),

            pl.col("as_of_year")
            .cast(pl.Int16, strict=False)
            .alias("occurrence_year"),

            pl.col("_socioeconomic_matched")
            .fill_null(False),
        )
        .with_columns(
            pl.lit(True).alias("_feature_matched")
        )
    )

    # ------------------------------------------------------------------
    # 4. Do the expensive remote join ONCE, after event aggregation, then
    #    materialize the much smaller cell-year table in memory.
    #
    # Because crime_cell_years is unique by (city, year, H3) and annual is
    # expected to be unique by (year, H3), this should be m:1 overall due
    # to multiple cities potentially sharing a cell. validate="m:1"
    # enforces the annual-store uniqueness contract.
    # ------------------------------------------------------------------
    joined = (
        crime_cell_years
        .join(
            annual_keys,
            on=["osm_h3_cell_id", "occurrence_year"],
            how="left",
            validate="m:1",
        )
        .with_columns(
            pl.col("_feature_matched")
            .fill_null(False),

            pl.col("_socioeconomic_matched")
            .fill_null(False),
        )
        .collect(engine="streaming")
    )

    print(
        f"\nCOMPRESSED JOIN DOMAIN: {joined.height:,} "
        "unique city/H3/year rows"
    )

    feature_matched = pl.col("_feature_matched")
    socio_matched = (
        pl.col("_feature_matched")
        & pl.col("_socioeconomic_matched")
    )

    # ------------------------------------------------------------------
    # 5. Everything below runs against the already-materialized compact
    #    DataFrame. No S3 rescan, no repeated H3 calculation, no repeated
    #    join.
    # ------------------------------------------------------------------
    global_coverage = (
        joined.lazy()
        .select(
            pl.col("event_count").sum().alias("events"),

            weighted_sum(
                pl.col("osm_h3_cell_id").is_null(),
                "event_count",
                "null_h3_events",
            ),

            weighted_sum(
                feature_matched,
                "event_count",
                "feature_matched_events",
            ),

            weighted_sum(
                ~feature_matched,
                "event_count",
                "feature_missing_events",
            ),

            weighted_sum(
                socio_matched,
                "event_count",
                "socio_matched_events",
            ),

            weighted_sum(
                feature_matched & ~pl.col("_socioeconomic_matched"),
                "event_count",
                "feature_without_socio_events",
            ),
        )
        .with_columns(
            pct(
                "feature_matched_events",
                "events",
                "feature_coverage_pct",
            ),
            pct(
                "socio_matched_events",
                "events",
                "socio_coverage_pct",
            ),
        )
        .collect()
    )

    print("\nGLOBAL EVENT-WEIGHTED COVERAGE")
    print(global_coverage)

    by_city = (
        joined.lazy()
        .group_by("source_city")
        .agg(
            pl.col("event_count").sum().alias("events"),
            pl.col("osm_h3_cell_id").n_unique().alias("unique_h3_cells"),

            weighted_sum(
                feature_matched,
                "event_count",
                "feature_matched_events",
            ),

            weighted_sum(
                ~feature_matched,
                "event_count",
                "feature_missing_events",
            ),

            weighted_sum(
                socio_matched,
                "event_count",
                "socio_matched_events",
            ),
        )
        .with_columns(
            pct(
                "feature_matched_events",
                "events",
                "feature_coverage_pct",
            ),
            pct(
                "socio_matched_events",
                "events",
                "socio_coverage_pct",
            ),
        )
        .sort("feature_coverage_pct")
        .collect()
    )

    print("\nCOVERAGE BY CITY")
    print(by_city)

    by_year = (
        joined.lazy()
        .group_by("occurrence_year")
        .agg(
            pl.col("event_count").sum().alias("events"),
            pl.col("osm_h3_cell_id").n_unique().alias("unique_h3_cells"),

            weighted_sum(
                feature_matched,
                "event_count",
                "feature_matched_events",
            ),

            weighted_sum(
                ~feature_matched,
                "event_count",
                "feature_missing_events",
            ),

            weighted_sum(
                socio_matched,
                "event_count",
                "socio_matched_events",
            ),
        )
        .with_columns(
            pct(
                "feature_matched_events",
                "events",
                "feature_coverage_pct",
            ),
            pct(
                "socio_matched_events",
                "events",
                "socio_coverage_pct",
            ),
        )
        .sort("occurrence_year")
        .collect()
    )

    print("\nCOVERAGE BY YEAR")
    print(by_year)

    by_city_year = (
        joined.lazy()
        .group_by(
            ["source_city", "occurrence_year"]
        )
        .agg(
            pl.col("event_count").sum().alias("events"),
            pl.col("osm_h3_cell_id").n_unique().alias("unique_h3_cells"),

            weighted_sum(
                feature_matched,
                "event_count",
                "feature_matched_events",
            ),

            weighted_sum(
                ~feature_matched,
                "event_count",
                "feature_missing_events",
            ),

            weighted_sum(
                socio_matched,
                "event_count",
                "socio_matched_events",
            ),
        )
        .with_columns(
            pct(
                "feature_matched_events",
                "events",
                "feature_coverage_pct",
            ),
            pct(
                "socio_matched_events",
                "events",
                "socio_coverage_pct",
            ),
        )
        .sort(["source_city", "occurrence_year"])
        .collect()
    )

    print("\nCOVERAGE BY CITY/YEAR")
    print(by_city_year)

    # joined is already unique at city/year/H3 grain, so there is no need
    # for another .unique() pass.
    cell_year_coverage = (
        joined.lazy()
        .group_by("source_city")
        .agg(
            pl.len().alias("crime_cell_years"),
            pl.col("_feature_matched").sum().alias("matched_cell_years"),
            (~pl.col("_feature_matched"))
            .sum()
            .alias("missing_cell_years"),
        )
        .with_columns(
            pct(
                "matched_cell_years",
                "crime_cell_years",
                "cell_year_coverage_pct",
            )
        )
        .sort("cell_year_coverage_pct")
        .collect()
    )

    print("\nUNIQUE H3 CELL-YEAR COVERAGE BY CITY")
    print(cell_year_coverage)

    # joined is already aggregated, so no second group-by is needed.
    top_missing = (
        joined.lazy()
        .filter(~pl.col("_feature_matched"))
        .select(
            "source_city",
            "occurrence_year",
            "osm_h3_cell_id",
            pl.col("event_count").alias("missing_events"),
        )
        .sort("missing_events", descending=True)
        .head(100)
        .collect()
    )

    print("\nTOP MISSING H3 CELL-YEARS BY EVENT COUNT")
    print(top_missing)

    stats = global_coverage.row(0, named=True)

    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)
    print(
        f"Events checked: {int(stats['events']):,}\n"
        f"Feature matched: {int(stats['feature_matched_events']):,} "
        f"({float(stats['feature_coverage_pct']):.4f}%)\n"
        f"Feature missing: {int(stats['feature_missing_events']):,}\n"
        f"Socioeconomic matched: {int(stats['socio_matched_events']):,} "
        f"({float(stats['socio_coverage_pct']):.4f}%)"
    )


if __name__ == "__main__":
    main()
