from __future__ import annotations

import polars as pl

from crimenet_data.resources import CrimeLakeResources


CITIES = [
    "chicago",
    "new_york",
    "san_francisco",
    "seattle",
    "washington_dc",
    "baltimore",
]

# Silver occurrence_timestamp is Datetime(us) with NO timezone.
CUTOFF = pl.datetime(
    2026,
    1,
    1,
    23,
    59,
    59,
)


def main() -> None:
    resources = CrimeLakeResources()

    snapshot_uri = resources.resolve_current_silver_snapshot()

    print()
    print("=" * 100)
    print("CURRENT SILVER SNAPSHOT")
    print("=" * 100)
    print(snapshot_uri)
    print()

    silver = resources.scan_silver_snapshot(snapshot_uri)

    schema = silver.collect_schema()

    print("Timestamp columns available:")
    for name in schema.names():
        if "timestamp" in name.lower():
            print(f"  {name}: {schema[name]}")
    print()

    ts = pl.col("occurrence_timestamp")

    result = (
        silver
        .filter(pl.col("source_city").is_in(CITIES))
        .group_by("source_city")
        .agg(
            pl.len().alias("total_rows"),
            ts.min().alias("min_timestamp"),
            ts.max().alias("max_timestamp"),

            (ts > CUTOFF)
            .fill_null(False)
            .sum()
            .alias("rows_after_2026_01_01"),

            (ts.dt.year() == 2026)
            .fill_null(False)
            .sum()
            .alias("rows_in_2026"),

            ts
            .filter(ts > CUTOFF)
            .min()
            .alias("first_timestamp_after_2026_01_01"),

            ts
            .filter(ts > CUTOFF)
            .max()
            .alias("last_timestamp_after_2026_01_01"),
        )
        .with_columns(
            (pl.col("rows_after_2026_01_01") > 0)
            .alias("has_coverage_after_2026_01_01")
        )
        .sort("source_city")
        .collect(engine="streaming")
    )

    print("=" * 100)
    print("SILVER COVERAGE")
    print("=" * 100)
    print(result)
    print()

    found = set(result["source_city"].to_list())
    missing = sorted(set(CITIES) - found)

    if missing:
        print("WARNING: missing requested cities:")
        for city in missing:
            print(f"  - {city}")
        print()

    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)

    for row in result.iter_rows(named=True):
        status = "YES" if row["has_coverage_after_2026_01_01"] else "NO"

        print(
            f"{row['source_city']:24s} "
            f"after Jan 1: {status:3s} | "
            f"rows after: {row['rows_after_2026_01_01']:,} | "
            f"rows in 2026: {row['rows_in_2026']:,} | "
            f"max: {row['max_timestamp']}"
        )


if __name__ == "__main__":
    main()
PY
