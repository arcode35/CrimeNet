from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import polars as pl

from crimenet_data.resources.crime_lake import CrimeLakeResources


# ---------------------------------------------------------------------------
# Sources that currently qualify for CrimeNet after the known Bronze -> Silver
# fixes:
#
# - Baltimore / DC: Unix-ms timestamps
# - Dallas: EPSG:2276 -> EPSG:4326
# - Sonoma: parse `location` -> lat/lon
#
# Baton Rouge variants are intentionally excluded for now because the available
# timestamp field still needs occurrence-time semantic validation.
# Boston / Gainesville are excluded for insufficient timestamp precision.
# ---------------------------------------------------------------------------

CITIES = [
    "atlanta",
    "baltimore",
    "chandler_az",
    "chicago",
    "dallas",
    "denver",
    "fort_worth",
    "los_angeles_county_sheriff",
    "marin_county_sheriff_ca",
    "montgomery_county_md",
    "new_york",
    "san_francisco",
    "seattle",
    "sonoma_county_sheriff_ca",
    "washington_dc",
]

OUTPUT_ROOT = Path("artifacts/crime_crosswalk_evidence")
ZIP_PATH = Path("artifacts/crime_crosswalk_evidence.zip")

MAX_DISTINCT = 50_000
MAX_UNIQUENESS_RATIO = 0.20
MAX_SELECTED_COLUMNS = 10

TOP_VALUES_PER_COLUMN = 250
SAMPLE_ROWS = 200

SOURCE_TAXONOMY_OVERRIDES = {
    "atlanta": [
        "ucrliteral",
    ],

    "los_angeles_county_sheriff": [
        "stat",
        "stat_desc",
        "category",
        "part_category",
    ],

    "denver": [
        "offense_code",
        "offense_code_extension",
        "offense_type_id",
        "offense_category_id",
    ],
}
# ---------------------------------------------------------------------------
# Columns introduced by CrimeNet ingestion rather than the source taxonomy.
# ---------------------------------------------------------------------------

INGESTION_METADATA_COLUMNS = {
    "source_city",
    "source_file_uri",
    "ingestion_run_id",
    "ingested_at_utc",
    "snapshot_id",
    "occurrence_year",
}


# ---------------------------------------------------------------------------
# Taxonomy-column heuristics
#
# This does NOT map crimes.
#
# It only identifies columns likely to participate in the source's offense
# taxonomy so that we can inspect every distinct observed combination.
# ---------------------------------------------------------------------------

POSITIVE_PATTERNS: list[tuple[str, int]] = [
    # strongest
    (r"offen[sc]e", 15),
    (r"ofns", 15),
    (r"\bucr\b", 15),
    (r"iucr", 15),
    (r"nibrs", 15),
    (r"\bibr\b", 12),
    (r"statute", 12),
    (r"violation", 10),
    (r"charge", 10),
    (r"crime", 10),

    # structural taxonomy fields
    (r"category", 8),
    (r"\bcat\b", 6),
    (r"_cat_", 6),
    (r"description", 8),
    (r"\bdesc\b", 7),
    (r"_desc", 7),
    (r"descr", 7),
    (r"classification", 7),
    (r"\bclass\b", 5),
    (r"subtype", 8),
    (r"sub_type", 8),
    (r"subcategory", 8),
    (r"sub_category", 8),
    (r"type", 5),
    (r"code", 5),
    (r"nature", 6),

    # possible severity / disambiguation fields
    (r"severity", 7),
    (r"degree", 6),
    (r"felony", 5),
    (r"misdemeanor", 5),
    (r"law_cat", 8),
    (r"weapon", 3),
]


NEGATIVE_PATTERNS = [
    # record identifiers
    r"incident.*(?:id|number|num|no)",
    r"event.*(?:id|number|num|no)",
    r"case.*(?:id|number|num|no)",
    r"report.*(?:id|number|num|no)",
    r"record.*(?:id|number|num|no)",
    r"object.*id",
    r"global.*id",
    r"row.*id",

    # agency identifiers
    r"agency",
    r"department",

    # geography
    r"latitude",
    r"longitude",
    r"\blat\b",
    r"\blon\b",
    r"\blng\b",
    r"coordinate",
    r"location",
    r"address",
    r"street",
    r"intersection",
    r"premise",
    r"district",
    r"precinct",
    r"beat",
    r"sector",
    r"zone",
    r"zipcode",
    r"zip_code",

    # time
    r"date",
    r"time",
    r"year",
    r"month",
    r"day",
    r"hour",
    r"timestamp",

    # CrimeNet metadata
    r"source_file",
    r"source_city",
    r"snapshot",
    r"ingest",
    r"run_id",
    r"upload",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_filename(value: str) -> str:
    return re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        value,
    )


def is_nested_dtype(dtype: pl.DataType) -> bool:
    text = str(dtype)

    return any(
        text.startswith(prefix)
        for prefix in (
            "List",
            "Struct",
            "Array",
            "Object",
        )
    )


def taxonomy_name_score(column: str) -> tuple[int, list[str]]:
    """
    Score how likely a source column is to describe the offense taxonomy.
    """

    name = column.lower()

    if column in INGESTION_METADATA_COLUMNS:
        return -100, ["ingestion_metadata"]

    negative_matches = [
        pattern
        for pattern in NEGATIVE_PATTERNS
        if re.search(pattern, name)
    ]

    if negative_matches:
        return -100, [
            f"excluded:{pattern}"
            for pattern in negative_matches
        ]

    score = 0
    reasons: list[str] = []

    for pattern, weight in POSITIVE_PATTERNS:
        if re.search(pattern, name):
            score += weight
            reasons.append(
                f"{pattern}:{weight}"
            )

    return score, reasons


def scan_city(
    crime_lake: CrimeLakeResources,
    city: str,
) -> tuple[str, pl.LazyFrame]:

    snapshot_uri = (
        crime_lake.resolve_current_bronze_snapshot(
            city
        )
    )

    lf = crime_lake.scan_bronze_snapshot(
        city,
        snapshot_uri=snapshot_uri,
    )

    return snapshot_uri, lf


# ---------------------------------------------------------------------------
# Profile schema/cardinality
# ---------------------------------------------------------------------------

def profile_columns(
    lf: pl.LazyFrame,
) -> tuple[int, list[dict]]:

    schema = lf.collect_schema()

    profileable = [
        (name, dtype)
        for name, dtype in schema.items()
        if not is_nested_dtype(dtype)
    ]

    expressions: list[pl.Expr] = [
        pl.len().alias("__total_rows")
    ]

    aliases: dict[str, dict[str, str]] = {}

    for index, (name, dtype) in enumerate(
        profileable
    ):
        prefix = f"c{index}"

        aliases[name] = {
            "nulls": f"{prefix}_nulls",
            "unique": f"{prefix}_unique",
        }

        expressions.extend(
            [
                pl.col(name)
                .null_count()
                .alias(
                    aliases[name]["nulls"]
                ),

                pl.col(name)
                .n_unique()
                .alias(
                    aliases[name]["unique"]
                ),
            ]
        )

        if dtype == pl.String:
            aliases[name]["avg_length"] = (
                f"{prefix}_avg_length"
            )
            aliases[name]["max_length"] = (
                f"{prefix}_max_length"
            )

            expressions.extend(
                [
                    pl.col(name)
                    .str.len_chars()
                    .mean()
                    .alias(
                        aliases[name][
                            "avg_length"
                        ]
                    ),

                    pl.col(name)
                    .str.len_chars()
                    .max()
                    .alias(
                        aliases[name][
                            "max_length"
                        ]
                    ),
                ]
            )

    stats = (
        lf.select(expressions)
        .collect(engine="streaming")
        .row(0, named=True)
    )

    total_rows = int(
        stats["__total_rows"]
    )

    result: list[dict] = []

    for name, dtype in schema.items():
        score, reasons = (
            taxonomy_name_score(name)
        )

        if is_nested_dtype(dtype):
            result.append(
                {
                    "column": name,
                    "dtype": str(dtype),
                    "null_count": None,
                    "null_pct": None,
                    "n_unique": None,
                    "uniqueness_ratio": None,
                    "avg_string_length": None,
                    "max_string_length": None,
                    "taxonomy_name_score": score,
                    "taxonomy_name_reasons": (
                        "; ".join(reasons)
                    ),
                    "profile_status": (
                        "nested_dtype_skipped"
                    ),
                }
            )
            continue

        null_count = int(
            stats[
                aliases[name]["nulls"]
            ]
        )

        n_unique = int(
            stats[
                aliases[name]["unique"]
            ]
        )

        avg_length = None
        max_length = None

        if dtype == pl.String:
            avg_length = stats[
                aliases[name]["avg_length"]
            ]

            max_length = stats[
                aliases[name]["max_length"]
            ]

        non_null = (
            total_rows - null_count
        )

        uniqueness_ratio = (
            n_unique / non_null
            if non_null > 0
            else None
        )

        result.append(
            {
                "column": name,
                "dtype": str(dtype),
                "null_count": null_count,
                "null_pct": (
                    null_count
                    / total_rows
                    * 100.0
                    if total_rows
                    else None
                ),
                "n_unique": n_unique,
                "uniqueness_ratio": (
                    uniqueness_ratio
                ),
                "avg_string_length": (
                    avg_length
                ),
                "max_string_length": (
                    max_length
                ),
                "taxonomy_name_score": score,
                "taxonomy_name_reasons": (
                    "; ".join(reasons)
                ),
                "profile_status": "profiled",
            }
        )

    return total_rows, result


# ---------------------------------------------------------------------------
# Determine which columns should participate in the distinct offense-key dump.
#
# We deliberately make this conservative:
# - name must look taxonomy-related
# - cardinality cannot be enormous
# - near-row-unique fields are rejected
# - giant text/narrative columns are rejected
#
# Even rejected candidates remain visible in column_profile.csv, so nothing
# disappears from the evidence bundle.
# ---------------------------------------------------------------------------

def select_taxonomy_columns(
    profiles: list[dict],
    city: str,
) -> tuple[list[str], list[dict]]:

    candidates: list[dict] = []

    profile_by_name = {
        row["column"]: row
        for row in profiles
    }

    for profile in profiles:
        score = profile[
            "taxonomy_name_score"
        ]

        if score <= 0:
            continue

        accepted = True
        rejection_reason = None

        n_unique = profile["n_unique"]
        uniqueness_ratio = profile[
            "uniqueness_ratio"
        ]
        avg_length = profile[
            "avg_string_length"
        ]

        if profile["profile_status"] != "profiled":
            accepted = False
            rejection_reason = "nested_or_unprofiled"

        elif n_unique is None:
            accepted = False
            rejection_reason = "unknown_cardinality"

        elif n_unique == 0:
            accepted = False
            rejection_reason = "empty"

        elif n_unique > MAX_DISTINCT:
            accepted = False
            rejection_reason = (
                f"n_unique>{MAX_DISTINCT}"
            )

        elif (
            uniqueness_ratio is not None
            and uniqueness_ratio > MAX_UNIQUENESS_RATIO
            and n_unique > 2_000
        ):
            accepted = False
            rejection_reason = (
                "too_close_to_row_unique"
            )

        elif (
            avg_length is not None
            and avg_length > 200
            and n_unique > 1_000
        ):
            accepted = False
            rejection_reason = (
                "likely_free_text_narrative"
            )

        candidates.append(
            {
                **profile,
                "accepted": accepted,
                "rejection_reason": rejection_reason,
            }
        )

    candidates.sort(
        key=lambda row: (
            not row["accepted"],
            -row["taxonomy_name_score"],
            row["n_unique"]
            if row["n_unique"] is not None
            else 10**18,
            row["column"],
        )
    )
    override = SOURCE_TAXONOMY_OVERRIDES.get(city)

    if override is not None:
        selected = override

    selected = [
        row["column"]
        for row in candidates
        if row["accepted"]
    ][:MAX_SELECTED_COLUMNS]

    # --------------------------------------------------------------
    # Explicit per-source additions.
    # These are trusted source taxonomy fields that generic naming
    # heuristics may fail to recognize.
    # --------------------------------------------------------------

    for column in SOURCE_TAXONOMY_OVERRIDES.get(
        city,
        [],
    ):
        if column not in profile_by_name:
            raise KeyError(
                f"Configured taxonomy override "
                f"{column!r} is missing from {city!r}"
            )

        if column not in selected:
            selected.append(column)

        # Ensure it appears in taxonomy_candidates.csv too.
        existing = next(
            (
                row
                for row in candidates
                if row["column"] == column
            ),
            None,
        )

        if existing is None:
            candidates.append(
                {
                    **profile_by_name[column],
                    "accepted": True,
                    "rejection_reason": None,
                    "explicit_override": True,
                }
            )
        else:
            existing["accepted"] = True
            existing["rejection_reason"] = None
            existing["explicit_override"] = True

    for row in candidates:
        row.setdefault(
            "explicit_override",
            False,
        )

        row[
            "selected_for_combination"
        ] = row["column"] in selected

    return selected, candidates


# ---------------------------------------------------------------------------
# Distinct source taxonomy
# ---------------------------------------------------------------------------

def collect_taxonomy_combinations(
    lf: pl.LazyFrame,
    columns: list[str],
) -> pl.DataFrame:

    if not columns:
        return pl.DataFrame(
            {"row_count": []},
            schema={
                "row_count": pl.UInt64,
            },
        )

    expressions = [
        pl.col(column)
        .cast(pl.String, strict=False)
        .alias(column)
        for column in columns
    ]

    return (
        lf.select(expressions)
        .group_by(columns)
        .agg(
            pl.len().alias("row_count")
        )
        .sort(
            "row_count",
            descending=True,
        )
        .collect(engine="streaming")
    )


def write_marginal_value_counts(
    combinations: pl.DataFrame,
    columns: list[str],
    output_dir: Path,
) -> None:

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for column in columns:
        counts = (
            combinations
            .group_by(column)
            .agg(
                pl.col("row_count")
                .sum()
                .alias("row_count")
            )
            .sort(
                "row_count",
                descending=True,
            )
        )

        counts.write_csv(
            output_dir
            / f"{safe_filename(column)}.csv"
        )


# ---------------------------------------------------------------------------
# Representative source rows
# ---------------------------------------------------------------------------

def collect_sample_rows(
    lf: pl.LazyFrame,
    candidates: list[dict],
) -> pl.DataFrame:

    # Include every taxonomy-looking column, even if it was rejected from the
    # full combination key, so a human can detect heuristic mistakes.
    columns = [
        row["column"]
        for row in candidates
    ]

    schema_names = (
        lf.collect_schema().names()
    )

    # Keep source_file_uri as provenance if available.
    if (
        "source_file_uri"
        in schema_names
        and "source_file_uri"
        not in columns
    ):
        columns.append(
            "source_file_uri"
        )

    if not columns:
        # Fall back to a small schema-width sample.
        columns = schema_names[:20]

    return (
        lf.select(columns)
        .head(SAMPLE_ROWS)
        .collect(engine="streaming")
    )


# ---------------------------------------------------------------------------
# Existing crosswalk
# ---------------------------------------------------------------------------

CROSSWALK_SOURCE_FIELDS = [
    "source_file",
    "source_offense_code",
    "source_offense_category",
    "source_offense_description",
    "source_auxiliary",
    "source_severity",
]

CANONICAL_TAXONOMY_FIELDS = [
    "canonical_family_code",
    "canonical_offense_family",
    "canonical_subtype_code",
    "canonical_offense_subtype",
    "canonical_domain",
    "canonical_target",
    "is_criminal_event",
    "is_violent",
    "is_property",
]


def export_existing_crosswalk(
    crime_lake: CrimeLakeResources,
) -> dict:

    crosswalk = (
        crime_lake.resolve_crosswalk()
        .collect(engine="streaming")
    )

    crosswalk.write_csv(
        OUTPUT_ROOT
        / "existing_crosswalk_v1_3.csv"
    )

    taxonomy = (
        crosswalk
        .select(
            CANONICAL_TAXONOMY_FIELDS
        )
        .unique()
        .sort(
            [
                "canonical_family_code",
                "canonical_subtype_code",
            ]
        )
    )

    taxonomy.write_csv(
        OUTPUT_ROOT
        / "existing_canonical_taxonomy.csv"
    )

    source_profiles = []

    for city in sorted(
        crosswalk[
            "source_city"
        ]
        .drop_nulls()
        .unique()
        .to_list()
    ):
        city_df = crosswalk.filter(
            pl.col("source_city") == city
        )

        row = {
            "source_city": city,
            "mapping_rows": city_df.height,
        }

        for field in (
            CROSSWALK_SOURCE_FIELDS
        ):
            if field not in city_df.columns:
                row[
                    f"{field}_populated"
                ] = 0
                continue

            populated = (
                city_df[field]
                .cast(pl.String)
                .fill_null("")
                .str.strip_chars()
                != ""
            ).sum()

            row[
                f"{field}_populated"
            ] = int(populated)

        source_profiles.append(row)

    pl.DataFrame(
        source_profiles
    ).write_csv(
        OUTPUT_ROOT
        / "existing_crosswalk_key_profiles.csv"
    )

    return {
        "crosswalk_rows": (
            crosswalk.height
        ),
        "canonical_taxonomy_rows": (
            taxonomy.height
        ),
        "existing_source_cities": sorted(
            crosswalk[
                "source_city"
            ]
            .drop_nulls()
            .unique()
            .to_list()
        ),
    }


# ---------------------------------------------------------------------------
# Per-city evidence
# ---------------------------------------------------------------------------

def profile_city(
    crime_lake: CrimeLakeResources,
    city: str,
) -> dict:

    print()
    print("=" * 80)
    print(city)
    print("=" * 80)

    city_dir = OUTPUT_ROOT / city

    city_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    snapshot_uri, lf = scan_city(
        crime_lake,
        city,
    )

    print(
        f"[{city}] snapshot: "
        f"{snapshot_uri}"
    )

    schema = lf.collect_schema()

    schema_rows = [
        {
            "column": name,
            "dtype": str(dtype),
        }
        for name, dtype
        in schema.items()
    ]

    pl.DataFrame(
        schema_rows
    ).write_csv(
        city_dir / "schema.csv"
    )

    print(
        f"[{city}] profiling "
        f"{len(schema)} columns..."
    )

    total_rows, profiles = (
        profile_columns(lf)
    )

    selected, candidates = (
        select_taxonomy_columns(
            profiles,
            city=city,
        )
    )

    print(
        f"[{city}] rows="
        f"{total_rows:,}"
    )

    print(
        f"[{city}] selected taxonomy "
        f"columns: {selected}"
    )

    pl.DataFrame(
        profiles
    ).write_csv(
        city_dir
        / "column_profile.csv"
    )

    if candidates:
        pl.DataFrame(
            candidates
        ).write_csv(
            city_dir
            / "taxonomy_candidates.csv"
        )

    print(
        f"[{city}] collecting distinct "
        f"taxonomy combinations..."
    )

    combinations = (
        collect_taxonomy_combinations(
            lf,
            selected,
        )
    )

    combinations.write_csv(
        city_dir
        / "taxonomy_combinations.csv"
    )

    write_marginal_value_counts(
        combinations,
        selected,
        city_dir / "top_values",
    )

    print(
        f"[{city}] combinations="
        f"{combinations.height:,}"
    )

    sample = collect_sample_rows(
        lf,
        candidates,
    )

    sample.write_csv(
        city_dir / "sample_rows.csv"
    )

    summary = {
        "source_city": city,
        "snapshot_uri": snapshot_uri,
        "row_count": total_rows,
        "column_count": len(schema),
        "selected_taxonomy_columns": (
            selected
        ),
        "taxonomy_combination_count": (
            combinations.height
        ),
        "candidate_column_count": (
            len(candidates)
        ),
    }

    (
        city_dir / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            default=str,
        )
    )

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    if OUTPUT_ROOT.exists():
        shutil.rmtree(
            OUTPUT_ROOT
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    crime_lake = (
        CrimeLakeResources()
    )

    manifest = {
        "cities_requested": CITIES,
        "existing_crosswalk": None,
        "cities": {},
        "failures": {},
    }

    # ------------------------------------------------------------------
    # Bundle the old crosswalk and canonical taxonomy so this evidence
    # package is completely self-contained.
    # ------------------------------------------------------------------

    print(
        "Exporting existing canonical "
        "crosswalk..."
    )

    try:
        manifest[
            "existing_crosswalk"
        ] = export_existing_crosswalk(
            crime_lake
        )
    except Exception as error:
        print(
            "WARNING: failed to export "
            f"existing crosswalk: {error}"
        )

        manifest[
            "existing_crosswalk_error"
        ] = {
            "type": (
                type(error).__name__
            ),
            "message": str(error),
        }

    # ------------------------------------------------------------------
    # Profile every accepted source independently.
    # ------------------------------------------------------------------

    for city in CITIES:
        try:
            manifest["cities"][city] = (
                profile_city(
                    crime_lake,
                    city,
                )
            )

        except Exception as error:
            print(
                f"[{city}] FAILED: "
                f"{type(error).__name__}: "
                f"{error}"
            )

            manifest[
                "failures"
            ][city] = {
                "type": (
                    type(error).__name__
                ),
                "message": str(error),
            }

    (
        OUTPUT_ROOT / "manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
            default=str,
        )
    )

    # ------------------------------------------------------------------
    # Make one file that can be uploaded back to ChatGPT.
    # ------------------------------------------------------------------

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    archive = shutil.make_archive(
        str(
            ZIP_PATH.with_suffix("")
        ),
        "zip",
        root_dir=OUTPUT_ROOT,
    )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(
        f"Evidence directory: "
        f"{OUTPUT_ROOT}"
    )
    print(
        f"Upload this file: "
        f"{archive}"
    )

    if manifest["failures"]:
        print()
        print(
            "Some cities failed:"
        )

        for city, error in (
            manifest[
                "failures"
            ].items()
        ):
            print(
                f"  {city}: "
                f"{error['type']}: "
                f"{error['message']}"
            )


if __name__ == "__main__":
    main()