from __future__ import annotations

from datetime import datetime

import polars as pl
import polars_h3 as plh3


EXPECTED_H3_RESOLUTION = 9
OSM_FEATURE_DEFINITION_VERSION = "2.0.0"

OSM_H3_KEY = [
    "source_city",
    "snapshot_year",
    "osm_h3_cell_id",
]


COUNT_COLUMNS = [
    "poi_count",
    "nightlife_poi_count",
    "food_poi_count",
    "retail_poi_count",
    "transit_poi_count",
    "education_poi_count",
    "healthcare_poi_count",
    "public_safety_poi_count",
    "finance_poi_count",
    "parking_poi_count",
    "lodging_poi_count",
    "recreation_poi_count",
    "poi_category_count",
    "road_segment_count",
    "intersection_count",
    "dead_end_count",
    "building_count",
    "residential_area_feature_count",
    "commercial_area_feature_count",
    "industrial_area_feature_count",
    "green_space_feature_count",
    "water_feature_count",
    "institutional_area_feature_count",
]


LENGTH_COLUMNS = [
    "road_length_m",
    "major_road_length_m",
    "collector_road_length_m",
    "residential_road_length_m",
    "service_road_length_m",
    "pedestrian_road_length_m",
    "cycleway_length_m",
    "representative_building_area_m2",
]


POI_CATEGORY_COLUMNS = [
    "nightlife_poi_count",
    "food_poi_count",
    "retail_poi_count",
    "transit_poi_count",
    "education_poi_count",
    "healthcare_poi_count",
    "public_safety_poi_count",
    "finance_poi_count",
    "parking_poi_count",
    "lodging_poi_count",
    "recreation_poi_count",
]


LAND_USE_COLUMNS = [
    "residential_area_feature_count",
    "commercial_area_feature_count",
    "industrial_area_feature_count",
    "green_space_feature_count",
    "water_feature_count",
    "institutional_area_feature_count",
]


REQUIRED_SOURCE_COLUMNS = {
    "source_city",
    "snapshot_year",
    "osm_h3_cell_id",
    "osm_h3_resolution",
    "processed_at",
    "maximum_node_degree",
    "average_intersection_degree",
    "one_way_road_length_ratio",
    *COUNT_COLUMNS,
    *LENGTH_COLUMNS,
}


DERIVED_FLOAT_COLUMNS = [
    "osm_cell_area_km2",

    "major_road_length_ratio",
    "collector_road_length_ratio",
    "residential_road_length_ratio",
    "service_road_length_ratio",
    "pedestrian_road_length_ratio",
    "cycleway_length_ratio",

    "average_road_segment_length_m",
    "intersections_per_road_km",
    "dead_ends_per_road_km",
    "dead_end_to_intersection_ratio",

    "nightlife_poi_ratio",
    "food_poi_ratio",
    "retail_poi_ratio",
    "transit_poi_ratio",

    "tracked_poi_category_entropy",
    "land_use_category_entropy",

    "commercial_residential_mix_ratio",
    "commercial_share_of_commercial_residential",

    "average_representative_building_area_m2",

    "poi_density_per_km2",
    "nightlife_poi_density_per_km2",
    "food_poi_density_per_km2",
    "retail_poi_density_per_km2",
    "transit_poi_density_per_km2",

    "road_length_density_m_per_km2",
    "major_road_density_m_per_km2",

    "intersection_density_per_km2",
    "dead_end_density_per_km2",
    "building_density_per_km2",

    "representative_building_area_m2_per_km2",
]


def validate_source_schema(
    lf: pl.LazyFrame,
) -> None:
    schema = lf.collect_schema()

    missing = (
        REQUIRED_SOURCE_COLUMNS
        - set(schema.names())
    )

    if missing:
        raise ValueError(
            "OSM H3 source is missing required columns: "
            f"{sorted(missing)}"
        )


def safe_ratio(
    numerator: str,
    denominator: str,
) -> pl.Expr:
    """
    Preserve the old Spark semantics:
    denominator <= 0 -> 0.0
    """

    return (
        pl.when(pl.col(denominator) > 0)
        .then(
            pl.col(numerator).cast(pl.Float64)
            / pl.col(denominator).cast(pl.Float64)
        )
        .otherwise(pl.lit(0.0))
    )


def safe_ratio_expr(
    numerator: pl.Expr,
    denominator: pl.Expr,
) -> pl.Expr:
    return (
        pl.when(denominator > 0)
        .then(
            numerator.cast(pl.Float64)
            / denominator.cast(pl.Float64)
        )
        .otherwise(pl.lit(0.0))
    )


def entropy_from_counts(
    columns: list[str],
) -> pl.Expr:
    """
    Shannon entropy using natural logarithm.

    This reproduces the old Spark implementation:

        -sum(p_i * ln(p_i))

    An all-zero vector has entropy 0.0.
    """

    counts = [
        pl.col(column).cast(pl.Float64)
        for column in columns
    ]

    total = pl.sum_horizontal(*counts)

    terms = []

    for count in counts:
        probability = count / total

        terms.append(
            pl.when(
                (total > 0)
                & (count > 0)
            )
            .then(
                -probability
                * probability.log()
            )
            .otherwise(pl.lit(0.0))
        )

    entropy = pl.sum_horizontal(*terms)

    return (
        pl.when(total > 0)
        .then(entropy)
        .otherwise(pl.lit(0.0))
    )

def normalize_osm_h3_source(
    raw_lf: pl.LazyFrame,
) -> pl.LazyFrame:
    validate_source_schema(raw_lf)

    lf = (
        raw_lf
        .with_columns(
            pl.col("source_city")
            .cast(pl.String)
            .str.strip_chars()
            .str.to_lowercase(),

            pl.col("snapshot_year")
            .cast(pl.Int32, strict=False),

            pl.col("osm_h3_cell_id")
            .cast(pl.String)
            .str.strip_chars()
            .str.to_lowercase(),

            pl.col("osm_h3_resolution")
            .cast(pl.Int8, strict=False),

            pl.col("processed_at")
            .cast(
                pl.Datetime(
                    "us",
                    time_zone="UTC",
                ),
                strict=False,
            ),

            *[
                pl.col(column)
                .cast(pl.Int64, strict=False)
                for column in COUNT_COLUMNS
            ],

            *[
                pl.col(column)
                .cast(pl.Float64, strict=False)
                for column in LENGTH_COLUMNS
            ],

            pl.col("maximum_node_degree")
            .cast(pl.Int64, strict=False),

            pl.col("average_intersection_degree")
            .cast(pl.Float64, strict=False),

            pl.col("one_way_road_length_ratio")
            .cast(pl.Float64, strict=False),
        )

        # ============================================================
        # Structural network null normalization
        #
        # These are undefined when the corresponding network does
        # not exist. Canonicalize only those cases to zero.
        #
        # IMPORTANT:
        # We deliberately DO NOT globally fill these columns.
        # Null values with a positive denominator remain null and
        # will fail validation.
        # ============================================================

        .with_columns(
            pl.when(
                (pl.col("road_segment_count") == 0)
                & pl.col(
                    "maximum_node_degree"
                ).is_null()
            )
            .then(pl.lit(0, dtype=pl.Int64))
            .otherwise(
                pl.col("maximum_node_degree")
            )
            .alias("maximum_node_degree"),

            pl.when(
                (pl.col("intersection_count") == 0)
                & pl.col(
                    "average_intersection_degree"
                ).is_null()
            )
            .then(pl.lit(0.0))
            .otherwise(
                pl.col(
                    "average_intersection_degree"
                )
            )
            .alias(
                "average_intersection_degree"
            ),

            pl.when(
                (pl.col("road_length_m") == 0)
                & pl.col(
                    "one_way_road_length_ratio"
                ).is_null()
            )
            .then(pl.lit(0.0))
            .otherwise(
                pl.col(
                    "one_way_road_length_ratio"
                )
            )
            .alias(
                "one_way_road_length_ratio"
            ),
        )

        .with_columns(
            plh3.str_to_int(
                "osm_h3_cell_id"
            )
            .alias("__h3_int"),
        )

        .with_columns(
            plh3.get_resolution(
                "__h3_int"
            )
            .cast(pl.Int8)
            .alias("__encoded_h3_resolution"),
        )
    )

    return lf


def collect_source_validation(
    lf: pl.LazyFrame,
    *,
    minimum_snapshot_year: int = 2013,
    maximum_snapshot_year: int = 2026,
) -> dict[str, int]:
    """
    Validate without filtering.

    If source data violates the Silver contract, the caller should
    fail rather than silently discard rows.
    """

    invalid_count = pl.any_horizontal(
        *[
            (
                pl.col(column).is_null()
                | (pl.col(column) < 0)
            )
            for column in COUNT_COLUMNS
        ]
    )

    invalid_length = pl.any_horizontal(
        *[
            (
                pl.col(column).is_null()
                | (pl.col(column) < 0)
            )
            for column in LENGTH_COLUMNS
        ]
    )

    invalid_key = (
        pl.col("source_city").is_null()
        | (pl.col("source_city").str.len_chars() == 0)
        | pl.col("snapshot_year").is_null()
        | ~pl.col("snapshot_year").is_between(
            minimum_snapshot_year,
            maximum_snapshot_year,
        )
        | pl.col("osm_h3_cell_id").is_null()
        | (pl.col("osm_h3_cell_id").str.len_chars() == 0)
        | pl.col("__h3_int").is_null()
        | (
            pl.col("osm_h3_resolution")
            != EXPECTED_H3_RESOLUTION
        )
        | (
            pl.col("__encoded_h3_resolution")
            != EXPECTED_H3_RESOLUTION
        )
    )

    invalid_network = (
        (
            pl.col("maximum_node_degree")
            .is_not_null()
            & (pl.col("maximum_node_degree") < 0)
        )
        |
        (
            pl.col("average_intersection_degree")
            .is_not_null()
            & (
                pl.col("average_intersection_degree") < 0
            )
        )
        |
        (
            pl.col("one_way_road_length_ratio")
            .is_not_null()
            & ~pl.col(
                "one_way_road_length_ratio"
            ).is_between(0.0, 1.0)
        )
    )

    stats = (
        lf
        .select(
            pl.len().alias("rows"),

            invalid_key
            .sum()
            .alias("invalid_key_rows"),

            invalid_count
            .sum()
            .alias("invalid_count_rows"),

            invalid_length
            .sum()
            .alias("invalid_length_rows"),

            invalid_network
            .sum()
            .alias("invalid_network_rows"),

            pl.col("processed_at")
            .is_null()
            .sum()
            .alias(
                "missing_processed_at_rows"
            ),
            pl.col("maximum_node_degree")
            .is_null()
            .sum()
            .alias("null_maximum_node_degree"),

            pl.col("average_intersection_degree")
            .is_null()
            .sum()
            .alias("null_average_intersection_degree"),

            pl.col("one_way_road_length_ratio")
            .is_null()
            .sum()
            .alias("null_one_way_road_length_ratio"),

            (
                pl.col("maximum_node_degree").is_null()
                & (pl.col("road_segment_count") > 0)
            )
            .sum()
            .alias("unexpected_null_maximum_node_degree"),

            (
                pl.col("average_intersection_degree").is_null()
                & (pl.col("intersection_count") > 0)
            )
            .sum()
            .alias(
                "unexpected_null_average_intersection_degree"
            ),

            (
                pl.col("one_way_road_length_ratio").is_null()
                & (pl.col("road_length_m") > 0)
            )
            .sum()
            .alias(
                "unexpected_null_one_way_road_length_ratio"
            ),

            (
                pl.col("maximum_node_degree") < 0
            )
            .fill_null(False)
            .sum()
            .alias("negative_maximum_node_degree"),

            (
                pl.col("average_intersection_degree") < 0
            )
            .fill_null(False)
            .sum()
            .alias(
                "negative_average_intersection_degree"
            ),

            (
                (
                    pl.col("one_way_road_length_ratio") < 0
                )
                | (
                    pl.col("one_way_road_length_ratio") > 1
                )
            )
            .fill_null(False)
            .sum()
            .alias("out_of_range_one_way_ratio"),
        )
        .collect()
        .row(
            0,
            named=True,
        )
    )

    duplicate_keys = (
        lf
        .group_by(OSM_H3_KEY)
        .len()
        .filter(
            pl.col("len") > 1
        )
        .select(
            pl.len()
            .alias("duplicate_keys")
        )
        .collect()
        .item()
    )

    stats["duplicate_keys"] = (
        duplicate_keys
    )

    return stats
def assert_valid_source(
    stats: dict[str, int],
) -> None:
    fatal_metrics = (
        "invalid_key_rows",
        "invalid_count_rows",
        "invalid_length_rows",
        "invalid_network_rows",
        "missing_processed_at_rows",
        "duplicate_keys",
    )

    failures = {
        metric: stats.get(metric, 0)
        for metric in fatal_metrics
        if stats.get(metric, 0) != 0
    }

    if not failures:
        return

    details = ", ".join(
        f"{metric}={value:,}"
        for metric, value in sorted(
            failures.items()
        )
    )

    raise ValueError(
        "OSM H3 source validation failed: "
        f"{details}"
    )


def build_osm_h3_silver(
    normalized_lf: pl.LazyFrame,
    *,
    run_id: str,
    processed_at: datetime,
) -> pl.LazyFrame:
    """
    Build canonical H3-9 OSM Silver features.

    Grain:
        (source_city, snapshot_year, osm_h3_cell_id)

    This transformation performs no row filtering.
    """

    # ================================================================
    # Exact H3 cell area
    # ================================================================

    lf = normalized_lf.with_columns(
        plh3.cell_area(
            "__h3_int",
            unit="km^2",
        )
        .cast(pl.Float64)
        .alias("osm_cell_area_km2"),
    )

    network_diagnostic = (
        normalized_lf
        .select(
            pl.len()
            .alias("rows"),

            pl.col("maximum_node_degree")
            .is_null()
            .sum()
            .alias("null_maximum_node_degree"),

            pl.col("average_intersection_degree")
            .is_null()
            .sum()
            .alias("null_average_intersection_degree"),

            pl.col("one_way_road_length_ratio")
            .is_null()
            .sum()
            .alias("null_one_way_road_length_ratio"),

            (
                pl.col("maximum_node_degree") < 0
            )
            .fill_null(False)
            .sum()
            .alias("negative_maximum_node_degree"),

            (
                pl.col("average_intersection_degree") < 0
            )
            .fill_null(False)
            .sum()
            .alias("negative_average_intersection_degree"),

            (
                (
                    pl.col("one_way_road_length_ratio") < 0
                )
                | (
                    pl.col("one_way_road_length_ratio") > 1
                )
            )
            .fill_null(False)
            .sum()
            .alias("invalid_one_way_ratio"),
        )
        .collect()
    )

    print(network_diagnostic)
    print(
    normalized_lf
    .filter(
        pl.col(
            "maximum_node_degree"
        ).is_null()
    )
    .select(
        pl.len()
        .alias("rows"),

        (
            pl.col("intersection_count") == 0
        )
        .sum()
        .alias(
            "zero_intersection_rows"
        ),

        (
            pl.col("intersection_count") > 0
        )
        .sum()
        .alias(
            "positive_intersection_rows"
        ),

        (
            pl.col("road_segment_count") == 0
        )
        .sum()
        .alias(
            "zero_road_segment_rows"
        ),

        (
            pl.col("road_segment_count") > 0
        )
        .sum()
        .alias(
            "positive_road_segment_rows"
        ),
    )
    .collect()
    )

    # ================================================================
    # Presence / aggregate flags
    # ================================================================

    lf = lf.with_columns(
        (pl.col("poi_count") > 0)
        .alias("has_poi_features"),

        (pl.col("road_length_m") > 0)
        .alias("has_road_features"),

        (pl.col("intersection_count") > 0)
        .alias(
            "has_intersection_features"
        ),

        (pl.col("building_count") > 0)
        .alias("has_building_features"),

        pl.sum_horizontal(
            *[
                pl.col(column)
                for column
                in LAND_USE_COLUMNS
            ]
        )
        .cast(pl.Int64)
        .alias("land_use_feature_count"),
    )

    lf = lf.with_columns(
        (
            pl.col(
                "land_use_feature_count"
            ) > 0
        )
        .alias("has_land_use_features"),
    )

    # ================================================================
    # Road composition
    #
    # Exact formulas from old Spark implementation.
    # ================================================================

    lf = lf.with_columns(
        safe_ratio(
            "major_road_length_m",
            "road_length_m",
        )
        .alias(
            "major_road_length_ratio"
        ),

        safe_ratio(
            "collector_road_length_m",
            "road_length_m",
        )
        .alias(
            "collector_road_length_ratio"
        ),

        safe_ratio(
            "residential_road_length_m",
            "road_length_m",
        )
        .alias(
            "residential_road_length_ratio"
        ),

        safe_ratio(
            "service_road_length_m",
            "road_length_m",
        )
        .alias(
            "service_road_length_ratio"
        ),

        safe_ratio(
            "pedestrian_road_length_m",
            "road_length_m",
        )
        .alias(
            "pedestrian_road_length_ratio"
        ),

        safe_ratio(
            "cycleway_length_m",
            "road_length_m",
        )
        .alias(
            "cycleway_length_ratio"
        ),
    )

    # ================================================================
    # Road-network structure
    # ================================================================

    lf = lf.with_columns(
        safe_ratio(
            "road_length_m",
            "road_segment_count",
        )
        .alias(
            "average_road_segment_length_m"
        ),

        safe_ratio_expr(
            pl.col("intersection_count"),
            pl.col("road_length_m")
            / pl.lit(1000.0),
        )
        .alias(
            "intersections_per_road_km"
        ),

        safe_ratio_expr(
            pl.col("dead_end_count"),
            pl.col("road_length_m")
            / pl.lit(1000.0),
        )
        .alias(
            "dead_ends_per_road_km"
        ),

        safe_ratio(
            "dead_end_count",
            "intersection_count",
        )
        .alias(
            "dead_end_to_intersection_ratio"
        ),
    )

    # ================================================================
    # POI composition
    # ================================================================

    lf = lf.with_columns(
        safe_ratio(
            "nightlife_poi_count",
            "poi_count",
        )
        .alias("nightlife_poi_ratio"),

        safe_ratio(
            "food_poi_count",
            "poi_count",
        )
        .alias("food_poi_ratio"),

        safe_ratio(
            "retail_poi_count",
            "poi_count",
        )
        .alias("retail_poi_ratio"),

        safe_ratio(
            "transit_poi_count",
            "poi_count",
        )
        .alias("transit_poi_ratio"),

        entropy_from_counts(
            POI_CATEGORY_COLUMNS
        )
        .alias(
            "tracked_poi_category_entropy"
        ),
    )

    # ================================================================
    # Land-use composition
    # ================================================================

    lf = lf.with_columns(
        entropy_from_counts(
            LAND_USE_COLUMNS
        )
        .alias(
            "land_use_category_entropy"
        ),

        # Preserve old v1 formula despite the slightly misleading name:
        #
        # commercial / residential
        safe_ratio(
            "commercial_area_feature_count",
            "residential_area_feature_count",
        )
        .alias(
            "commercial_residential_mix_ratio"
        ),

        # This column existed in the old Silver schema but is not
        # created by the supplied Spark file. Its formula is
        # reconstructed explicitly from its name.
        safe_ratio_expr(
            pl.col(
                "commercial_area_feature_count"
            ),
            (
                pl.col(
                    "commercial_area_feature_count"
                )
                + pl.col(
                    "residential_area_feature_count"
                )
            ),
        )
        .alias(
            "commercial_share_of_commercial_residential"
        ),
    )

    # ================================================================
    # Building summary
    # ================================================================

    lf = lf.with_columns(
        safe_ratio(
            "representative_building_area_m2",
            "building_count",
        )
        .alias(
            "average_representative_building_area_m2"
        ),
    )

    # ================================================================
    # Density features
    #
    # These columns existed in the old Silver schema but were not
    # created by the supplied Spark transformation. They are
    # reconstructed as primitive value / exact H3 cell area.
    # ================================================================

    lf = lf.with_columns(
        safe_ratio(
            "poi_count",
            "osm_cell_area_km2",
        )
        .alias(
            "poi_density_per_km2"
        ),

        safe_ratio(
            "nightlife_poi_count",
            "osm_cell_area_km2",
        )
        .alias(
            "nightlife_poi_density_per_km2"
        ),

        safe_ratio(
            "food_poi_count",
            "osm_cell_area_km2",
        )
        .alias(
            "food_poi_density_per_km2"
        ),

        safe_ratio(
            "retail_poi_count",
            "osm_cell_area_km2",
        )
        .alias(
            "retail_poi_density_per_km2"
        ),

        safe_ratio(
            "transit_poi_count",
            "osm_cell_area_km2",
        )
        .alias(
            "transit_poi_density_per_km2"
        ),

        safe_ratio(
            "road_length_m",
            "osm_cell_area_km2",
        )
        .alias(
            "road_length_density_m_per_km2"
        ),

        safe_ratio(
            "major_road_length_m",
            "osm_cell_area_km2",
        )
        .alias(
            "major_road_density_m_per_km2"
        ),

        safe_ratio(
            "intersection_count",
            "osm_cell_area_km2",
        )
        .alias(
            "intersection_density_per_km2"
        ),

        safe_ratio(
            "dead_end_count",
            "osm_cell_area_km2",
        )
        .alias(
            "dead_end_density_per_km2"
        ),

        safe_ratio(
            "building_count",
            "osm_cell_area_km2",
        )
        .alias(
            "building_density_per_km2"
        ),

        safe_ratio(
            "representative_building_area_m2",
            "osm_cell_area_km2",
        )
        .alias(
            "representative_building_area_m2_per_km2"
        ),
    )

    # ================================================================
    # Lineage / definition metadata
    # ================================================================

    lf = (
        lf
        .rename(
            {
                "processed_at":
                    "osm_extracted_at",
            }
        )
        .with_columns(
            pl.lit(
                OSM_FEATURE_DEFINITION_VERSION
            )
            .alias(
                "osm_feature_definition_version"
            ),

            pl.lit("openstreetmap")
            .alias("_source_system"),

            pl.lit(run_id)
            .alias("_ingestion_run_id"),

            pl.lit(processed_at)
            .alias("_ingested_at_utc"),

            pl.lit(processed_at)
            .alias("silver_processed_at"),
        )
        .drop(
            "__h3_int",
            "__encoded_h3_resolution",
        )
    )

    return lf


def collect_silver_validation(
    lf: pl.LazyFrame,
) -> dict[str, int | float]:
    invalid_derived_float = (
        pl.any_horizontal(
            *[
                (
                    pl.col(column).is_null()
                    | ~pl.col(column).is_finite()
                )
                for column
                in DERIVED_FLOAT_COLUMNS
            ]
        )
    )

    stats = (
        lf
        .select(
            pl.len()
            .alias("rows"),

            pl.struct(OSM_H3_KEY)
            .n_unique()
            .alias("unique_keys"),

            invalid_derived_float
            .sum()
            .alias(
                "invalid_derived_float_rows"
            ),

            pl.col("osm_cell_area_km2")
            .min()
            .alias(
                "minimum_cell_area_km2"
            ),

            pl.col("osm_cell_area_km2")
            .max()
            .alias(
                "maximum_cell_area_km2"
            ),
        )
        .collect()
        .row(
            0,
            named=True,
        )
    )

    return stats