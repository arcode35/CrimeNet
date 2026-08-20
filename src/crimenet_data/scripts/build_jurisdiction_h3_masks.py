from __future__ import annotations

import shutil
import time
import urllib.request
import zipfile

from pathlib import Path

import duckdb
import polars as pl
import polars_h3 as plh3


# =============================================================================
# Configuration
# =============================================================================


H3_RESOLUTION = 9

OSM_ROOT = (
    "gs://crimenet/silver/osm_h3_features"
)

OUTPUT_ROOT = (
    "gs://crimenet/silver/"
    "spatial_support/"
    "jurisdiction_h3_mask"
)

# Persisted compatibility contract:
#
#   one row per OSM city/year/H3-9 support key
#   inside_observation_domain =
#       OSM feature coverage AND H3 polygon intersects TIGER jurisdiction
#
# Because the input universe is OSM support, osm_feature_covered is True
# for every row. asset.py and transformations.py consume
# inside_observation_domain as the canonical model-domain decision.

CACHE_ROOT = Path(
    ".cache/tiger_place"
)


# Latest currently available TIGER/Line boundary vintage.
#
# Support years after this use the latest available legal boundary.
#
# Therefore:
#
#   2026 support -> 2025 TIGER boundary
#
LATEST_TIGER_YEAR = 2025


# Census incorporated-place GEOIDs.
#
# GEOID =
#     state FIPS (2)
#     +
#     place code (5)
#
CITY_CONFIG = {
    "baltimore": {
        "state_fips": "24",
        "place_geoid": "2404000",
        "expected_name": "Baltimore city",
    },

    "chicago": {
        "state_fips": "17",
        "place_geoid": "1714000",
        "expected_name": "Chicago city",
    },

    "dallas": {
        "state_fips": "48",
        "place_geoid": "4819000",
        "expected_name": "Dallas city",
    },

    "fort_worth": {
        "state_fips": "48",
        "place_geoid": "4827000",
        "expected_name": "Fort Worth city",
    },

    "new_york": {
        "state_fips": "36",
        "place_geoid": "3651000",
        "expected_name": "New York city",
    },

    "san_francisco": {
        "state_fips": "06",
        "place_geoid": "0667000",
        "expected_name": "San Francisco city",
    },

    "seattle": {
        "state_fips": "53",
        "place_geoid": "5363000",
        "expected_name": "Seattle city",
    },

    "washington_dc": {
        "state_fips": "11",
        "place_geoid": "1150000",
        "expected_name": "Washington city",
    },
}


# =============================================================================
# Helpers
# =============================================================================


def sql_literal(
    value: str,
) -> str:
    return (
        "'"
        + value.replace(
            "'",
            "''",
        )
        + "'"
    )


def tiger_boundary_year(
    support_year: int,
) -> int:
    """
    Select the authoritative TIGER vintage for a support year.

    Historical support:
        use the matching year's legal boundary.

    Future/current support beyond available TIGER:
        use latest published boundary.
    """

    return min(
        support_year,
        LATEST_TIGER_YEAR,
    )


def tiger_place_url(
    *,
    year: int,
    state_fips: str,
) -> str:
    return (
        "https://www2.census.gov/"
        f"geo/tiger/TIGER{year}/PLACE/"
        f"tl_{year}_{state_fips}_place.zip"
    )


def download_with_retries(
    *,
    url: str,
    destination: Path,
    attempts: int = 5,
) -> None:
    if destination.exists():
        return

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    for attempt in range(
        1,
        attempts + 1,
    ):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent":
                        "CrimeNet jurisdiction-mask builder",
                },
            )

            with (
                urllib.request.urlopen(
                    request,
                    timeout=120,
                ) as response,
                destination.open("wb") as output,
            ):
                shutil.copyfileobj(
                    response,
                    output,
                )

            return

        except Exception:
            if destination.exists():
                destination.unlink()

            if attempt == attempts:
                raise

            time.sleep(
                2 ** attempt
            )


def get_tiger_place_shapefile(
    *,
    year: int,
    state_fips: str,
) -> Path:
    """
    Download/cache one annual state-level TIGER place shapefile.
    """

    directory = (
        CACHE_ROOT
        / str(year)
        / state_fips
    )

    archive = (
        directory
        / f"tl_{year}_{state_fips}_place.zip"
    )

    expected_shapefile = (
        directory
        / f"tl_{year}_{state_fips}_place.shp"
    )

    if expected_shapefile.exists():
        return expected_shapefile

    url = tiger_place_url(
        year=year,
        state_fips=state_fips,
    )

    print(
        f"Downloading TIGER {year} "
        f"state={state_fips}"
    )

    download_with_retries(
        url=url,
        destination=archive,
    )

    with zipfile.ZipFile(
        archive
    ) as zipped:
        zipped.extractall(
            directory
        )

    if not expected_shapefile.exists():
        candidates = list(
            directory.glob(
                "*.shp"
            )
        )

        if len(candidates) != 1:
            raise ValueError(
                "Could not identify TIGER place shapefile: "
                f"year={year}, "
                f"state={state_fips}, "
                f"candidates={candidates}"
            )

        return candidates[0]

    return expected_shapefile


# =============================================================================
# OSM support
# =============================================================================


def load_osm_support(
    *,
    credentials: pl.CredentialProviderGCP,
) -> pl.DataFrame:
    """
    Load every unique OSM H3-9 support key.

    The mask is intentionally generated for EVERY OSM support key,
    not only cells that we expect to retain.

    Therefore each source support cell receives an explicit boolean:
        True  -> inside jurisdiction
        False -> outside jurisdiction

    There are no implicit/missing decisions.
    """

    support = (
        pl.scan_delta(
            OSM_ROOT,
            credential_provider=
                credentials,
        )
        .filter(
            pl.col(
                "osm_h3_resolution"
            )
            == H3_RESOLUTION
        )
        .select(
            "source_city",

            pl.col(
                "snapshot_year"
            )
            .cast(pl.Int32)
            .alias(
                "support_year"
            ),

            plh3.str_to_int(
                "osm_h3_cell_id"
            )
            .cast(pl.Int64)
            .alias(
                "osm_h3_cell_id"
            ),
        )
        .unique(
            subset=[
                "source_city",
                "support_year",
                "osm_h3_cell_id",
            ]
        )
        .sort(
            [
                "source_city",
                "support_year",
                "osm_h3_cell_id",
            ]
        )
        .with_columns(
            plh3.cell_to_boundary(
                "osm_h3_cell_id"
            )
            .alias(
                "_h3_boundary"
            )
        )
        .with_columns(
            pl.col(
                "_h3_boundary"
            )
            .map_elements(
                boundary_to_wkt,
                return_dtype=pl.String,
            )
            .alias(
                "_h3_wkt"
            )
        )
        .drop(
            "_h3_boundary"
        )
        .collect()
    )

    unknown_cities = (
        set(
            support[
                "source_city"
            ]
            .unique()
            .to_list()
        )
        -
        set(
            CITY_CONFIG
        )
    )

    if unknown_cities:
        raise ValueError(
            "OSM support contains cities with no "
            "jurisdiction configuration: "
            f"{sorted(unknown_cities)}"
        )

    if support.height == 0:
        raise ValueError(
            "OSM H3-9 support is empty."
        )

    duplicate_count = (
        support
        .group_by(
            [
                "source_city",
                "support_year",
                "osm_h3_cell_id",
            ]
        )
        .len()
        .filter(
            pl.col("len") != 1
        )
        .height
    )

    if duplicate_count:
        raise ValueError(
            "OSM support is not unique by "
            "city/year/H3 cell."
        )

    return support


# =============================================================================
# Jurisdiction classification
# =============================================================================


def load_jurisdiction_polygon(
    *,
    con: duckdb.DuckDBPyConnection,
    city: str,
    boundary_year: int,
) -> tuple[str, str]:
    """
    Load one Census incorporated-place polygon into DuckDB.

    TIGER/Line shapefiles are NAD83.

    We transform to WGS84 because H3 cell polygons use
    longitude/latitude in WGS84 coordinates.
    """

    config = (
        CITY_CONFIG[
            city
        ]
    )

    state_fips = (
        config[
            "state_fips"
        ]
    )

    place_geoid = (
        config[
            "place_geoid"
        ]
    )

    expected_name = (
        config[
            "expected_name"
        ]
    )

    shapefile = (
        get_tiger_place_shapefile(
            year=
                boundary_year,
            state_fips=
                state_fips,
        )
    )

    shp_sql = sql_literal(
        str(
            shapefile.resolve()
        )
    )

    geoid_sql = sql_literal(
        place_geoid
    )

    # ---------------------------------------------------------------------
    # Validate the requested jurisdiction first.
    # ---------------------------------------------------------------------

    jurisdiction_rows = (
        con.execute(
            f"""
            SELECT
                GEOID,
                NAME

            FROM ST_Read(
                {shp_sql}
            )

            WHERE
                GEOID = {geoid_sql}
            """
        )
        .fetchall()
    )

    if len(jurisdiction_rows) != 1:
        raise ValueError(
            "Expected exactly one Census place boundary: "
            f"city={city}, "
            f"boundary_year={boundary_year}, "
            f"GEOID={place_geoid}, "
            f"rows={len(jurisdiction_rows)}"
        )

    actual_geoid = str(
        jurisdiction_rows[0][0]
    )

    actual_name = str(
        jurisdiction_rows[0][1]
    )

    if actual_geoid != place_geoid:
        raise ValueError(
            f"GEOID mismatch for {city}: "
            f"{actual_geoid} != {place_geoid}"
        )

    if actual_name != expected_name:
        print(
            f"NOTE: Census NAME differs for {city}: "
            f"{actual_name!r} "
            f"(configured label={expected_name!r})"
        )

    # ---------------------------------------------------------------------
    # Materialize one jurisdiction geometry.
    #
    # TIGER:
    #     NAD83 / EPSG:4269
    #
    # H3 polygon:
    #     WGS84 / EPSG:4326
    #
    # always_xy=True preserves conventional:
    #
    #     x = longitude
    #     y = latitude
    # ---------------------------------------------------------------------

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE
            jurisdiction
        AS

        SELECT
            ST_Transform(
                geom,
                'EPSG:4269',
                'EPSG:4326',
                always_xy := true
            ) AS geom

        FROM ST_Read(
            {shp_sql}
        )

        WHERE
            GEOID = {geoid_sql}
        """
    )

    invalid_geometries = (
        con.execute(
            """
            SELECT COUNT(*)
            FROM jurisdiction
            WHERE NOT ST_IsValid(geom)
            """
        )
        .fetchone()[0]
    )

    if invalid_geometries:
        raise ValueError(
            "Invalid jurisdiction geometry: "
            f"city={city}, "
            f"boundary_year={boundary_year}"
        )

    return (
        actual_geoid,
        actual_name,
    )
def boundary_to_wkt(
    boundary: object,
) -> str:
    """
    Convert polars_h3.cell_to_boundary output to WKT.

    H3 coordinates are:
        latitude, longitude

    WKT expects:
        longitude latitude
    """

    if hasattr(
        boundary,
        "to_list",
    ):
        boundary = (
            boundary.to_list()
        )

    if not boundary:
        raise ValueError(
            "Empty H3 boundary."
        )

    first = boundary[0]

    # Handle:
    #
    #   [[lat, lng], ...]
    #
    # if returned by installed polars-h3 version.
    if isinstance(
        first,
        (list, tuple),
    ):
        coordinates = [
            (
                float(point[1]),
                float(point[0]),
            )
            for point in boundary
        ]

    # Handle documented flattened representation:
    #
    #   [lat0, lng0, lat1, lng1, ...]
    else:
        if len(boundary) % 2:
            raise ValueError(
                "Invalid flattened H3 boundary."
            )

        coordinates = [
            (
                float(boundary[index + 1]),
                float(boundary[index]),
            )
            for index in range(
                0,
                len(boundary),
                2,
            )
        ]

    if (
        coordinates[0]
        != coordinates[-1]
    ):
        coordinates.append(
            coordinates[0]
        )

    coordinate_text = ", ".join(
        f"{lng:.15f} {lat:.15f}"
        for lng, lat
        in coordinates
    )

    return (
        f"POLYGON(({coordinate_text}))"
    )

def classify_partition(
    *,
    con: duckdb.DuckDBPyConnection,
    points: pl.DataFrame,
    city: str,
    support_year: int,
) -> pl.DataFrame:
    boundary_year = (
        tiger_boundary_year(
            support_year
        )
    )

    place_geoid, place_name = (
        load_jurisdiction_polygon(
            con=con,
            city=city,
            boundary_year=
                boundary_year,
        )
    )

    try:
        con.unregister(
            "support_points_raw"
        )
    except Exception:
        pass

    con.register(
        "support_points_raw",
        points.to_arrow(),
    )
    con.execute(
    """
    CREATE OR REPLACE TEMP TABLE
        support_cells
    AS

    SELECT
        source_city,
        support_year,
        osm_h3_cell_id,

        ST_GeomFromText(
            _h3_wkt
        ) AS h3_geometry

    FROM support_points_raw
    """
)
    # ---------------------------------------------------------------------
    # Polygon intersection is deliberate.
    #
    # The model domain is defined over OSM-feature-covered H3-9 cells.
    # Any such H3 cell whose polygon intersects the authoritative TIGER
    # jurisdiction belongs to the modeled spatial observation domain.
    #
    # We intentionally do not use centroid containment because legitimate
    # boundary cells can straddle the jurisdiction boundary.
    # ---------------------------------------------------------------------

    result = pl.from_arrow(
        con.execute(
            """
            SELECT
                p.source_city,
                p.support_year,
                p.osm_h3_cell_id,

                ST_Intersects(
                    j.geom,
                    p.h3_geometry
                ) AS inside_jurisdiction

            FROM support_cells AS p

            CROSS JOIN jurisdiction AS j
            """
        )
        .fetch_arrow_table()
    )

    if (
        result.height
        != points.height
    ):
        raise ValueError(
            "Jurisdiction classification changed "
            "support cardinality: "
            f"city={city}, "
            f"year={support_year}, "
            f"input={points.height:,}, "
            f"output={result.height:,}"
        )

    null_decisions = (
        result
        .select(
            pl.col(
                "inside_jurisdiction"
            )
            .is_null()
            .sum()
        )
        .item()
    )
    if null_decisions:
        raise ValueError(
            "Jurisdiction mask contains null decisions: "
            f"city={city}, "
            f"year={support_year}, "
            f"nulls={null_decisions:,}"
        )
    result = (
        result
        .with_columns(
            pl.col("support_year")
            .cast(pl.Int32),

            pl.col("osm_h3_cell_id")
            .cast(pl.Int64),

            pl.col("inside_jurisdiction")
            .cast(pl.Boolean),

            pl.lit(True)
            .alias("osm_feature_covered"),

            # Canonical pipeline contract consumed by:
            #   - validate_spatial_domain_mask()
            #   - prepare_spatial_support()
            pl.col("inside_jurisdiction")
            .alias("inside_observation_domain"),

            # Explicit semantic alias retained for provenance/readability.
            pl.col("inside_jurisdiction")
            .alias("inside_model_spatial_domain"),

            pl.lit(boundary_year)
            .cast(pl.Int32)
            .alias("jurisdiction_boundary_year"),

            pl.lit(place_geoid)
            .alias("jurisdiction_geoid"),

            pl.lit(place_name)
            .alias("jurisdiction_name"),

            pl.lit(
                "osm_h3_support_x_tiger_place_intersects_v1"
            )
            .alias("domain_rule"),
        )
    )

    inside_count = (
        result
        .select(
            pl.col(
                "inside_observation_domain"
            )
            .sum()
        )
        .item()
    )

    print(
        f"{city:15s} "
        f"{support_year}: "
        f"{inside_count:,}/"
        f"{result.height:,} "
        f"inside "
        f"(TIGER {boundary_year})"
    )

    return result


# =============================================================================
# Full mask
# =============================================================================


def build_mask(
    *,
    support: pl.DataFrame,
) -> pl.DataFrame:
    con = duckdb.connect(
        database=":memory:"
    )

    try:
        con.execute(
            "INSTALL spatial"
        )

        con.execute(
            "LOAD spatial"
        )

        partitions = (
            support
            .select(
                [
                    "source_city",
                    "support_year",
                ]
            )
            .unique()
            .sort(
                [
                    "source_city",
                    "support_year",
                ]
            )
        )

        outputs: list[
            pl.DataFrame
        ] = []

        for row in (
            partitions
            .iter_rows(
                named=True
            )
        ):
            city = str(
                row[
                    "source_city"
                ]
            )

            support_year = int(
                row[
                    "support_year"
                ]
            )

            points = (
                support
                .filter(
                    (
                        pl.col(
                            "source_city"
                        )
                        == city
                    )
                    &
                    (
                        pl.col(
                            "support_year"
                        )
                        == support_year
                    )
                )
            )

            outputs.append(
                classify_partition(
                    con=con,
                    points=points,
                    city=city,
                    support_year=
                        support_year,
                )
            )

    finally:
        con.close()

    mask = (
        pl.concat(
            outputs,
            how="vertical",
        )
        .sort(
            [
                "source_city",
                "support_year",
                "osm_h3_cell_id",
            ]
        )
    )

    # =========================================================================
    # Hard global invariants
    # =========================================================================

    if (
        mask.height
        != support.height
    ):
        raise ValueError(
            "Mask does not contain exactly one decision "
            "for every OSM support cell: "
            f"osm={support.height:,}, "
            f"mask={mask.height:,}"
        )

    duplicate_keys = (
        mask
        .group_by(
            [
                "source_city",
                "support_year",
                "osm_h3_cell_id",
            ]
        )
        .len()
        .filter(
            pl.col("len") != 1
        )
        .height
    )

    if duplicate_keys:
        raise ValueError(
            "Jurisdiction mask key uniqueness violated: "
            f"{duplicate_keys:,} duplicate keys"
        )

    null_decisions = (
        mask
        .select(
            pl.col(
                "inside_observation_domain"
            )
            .is_null()
            .sum()
        )
        .item()
    )

    if null_decisions:
        raise ValueError(
            "Jurisdiction mask contains null decisions."
        )

    # ---------------------------------------------------------------------
    # Semantic schema invariant.
    #
    # This artifact is generated from the OSM support universe, so every
    # row is OSM-feature-covered. The three domain flags must therefore
    # remain exactly equivalent.
    # ---------------------------------------------------------------------
    semantic_mismatches = (
        mask
        .filter(
            (~pl.col("osm_feature_covered"))
            |
            (
                pl.col("inside_jurisdiction")
                != pl.col("inside_observation_domain")
            )
            |
            (
                pl.col("inside_model_spatial_domain")
                != pl.col("inside_observation_domain")
            )
        )
        .height
    )

    if semantic_mismatches:
        raise ValueError(
            "Jurisdiction mask semantic flags disagree: "
            f"{semantic_mismatches:,} rows"
        )

    return mask


# =============================================================================
# Diagnostics
# =============================================================================


def print_diagnostics(
    mask: pl.DataFrame,
) -> None:
    summary = (
        mask
        .group_by(
            [
                "source_city",
                "support_year",
            ]
        )
        .agg(
            pl.len()
            .alias(
                "candidate_cells"
            ),

            pl.col(
                "inside_observation_domain"
            )
            .sum()
            .alias(
                "inside_cells"
            ),

            (
                ~pl.col(
                    "inside_observation_domain"
                )
            )
            .sum()
            .alias(
                "outside_cells"
            ),
        )
        .with_columns(
            (
                pl.col(
                    "inside_cells"
                )
                /
                pl.col(
                    "candidate_cells"
                )
            )
            .alias(
                "inside_fraction"
            )
        )
        .sort(
            [
                "source_city",
                "support_year",
            ]
        )
    )

    city_summary = (
        mask
        .group_by(
            "source_city"
        )
        .agg(
            pl.len()
            .alias(
                "candidate_cells"
            ),

            pl.col(
                "inside_observation_domain"
            )
            .sum()
            .alias(
                "inside_cells"
            ),
        )
        .with_columns(
            (
                pl.col(
                    "inside_cells"
                )
                /
                pl.col(
                    "candidate_cells"
                )
            )
            .alias(
                "inside_fraction"
            )
        )
        .sort(
            "source_city"
        )
    )

    print(
        "\nCITY/YEAR MASK COVERAGE\n"
    )

    print(
        summary
    )

    print(
        "\nCITY MASK COVERAGE\n"
    )

    print(
        city_summary
    )


# =============================================================================
# Persist
# =============================================================================


def write_mask(
    *,
    mask: pl.DataFrame,
    credentials: pl.CredentialProviderGCP,
) -> None:
    (
        mask
        .lazy()
        .sink_delta(
            OUTPUT_ROOT,
            mode="overwrite",
            credential_provider=
                credentials,
            delta_write_options={
                "partition_by": [
                    "source_city",
                    "support_year",
                ],

                "schema_mode":
                    "overwrite",
            },
        )
    )


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    credentials = (
        pl.CredentialProviderGCP()
    )

    print(
        "Loading canonical OSM H3-9 support..."
    )

    support = load_osm_support(
        credentials=credentials,
    )

    print(
        f"OSM support keys: "
        f"{support.height:,}"
    )

    print(
        "\nBuilding jurisdiction mask..."
    )

    mask = build_mask(
        support=support,
    )

    print_diagnostics(
        mask
    )

    print(
        "\nWriting jurisdiction mask..."
    )

    write_mask(
        mask=mask,
        credentials=credentials,
    )

    print(
        "\nDONE"
    )

    print(
        f"rows={mask.height:,}"
    )

    print(
        f"output={OUTPUT_ROOT}"
    )


if __name__ == "__main__":
    main()
