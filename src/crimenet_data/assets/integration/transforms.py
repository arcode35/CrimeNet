
"""Pure transformations for CrimeNet integration-domain construction and sampling.

The model integration measure remains discrete H3-r9 cell × continuous time.
Latitude/longitude are sampled inside the selected H3 cell as auxiliary
sub-cell coordinates. They do not change the Monte Carlo cell-hour weight.

If the model is later redefined as a continuous-space intensity per km²,
the proposal and importance weights must be changed to area-weighted spatial
integration rather than the cell-hour estimator implemented here.
"""

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import h3
import numpy as np
import polars as pl


EARTH_RADIUS_M = 6_371_008.8

# Canonical half-open model split boundaries. Integration sampling owns this
# temporal contract; downstream model-table code imports these values rather
# than defining parallel dates.
TRAIN_SPLIT_END_UTC = datetime(2024, 1, 1, tzinfo=UTC)
VALIDATION_SPLIT_END_UTC = datetime(2025, 1, 1, tzinfo=UTC)
MODEL_SPLITS = ("train", "validation", "test")

INTEGRATION_SAMPLE_SCHEMA = pl.Schema(
    {
        "source_city": pl.String,
        "split": pl.String,
        "sample_index": pl.Int64,
        "integration_timestamp_utc": pl.Datetime("us", time_zone="UTC"),
        "osm_h3_cell_id": pl.Int64,
        "latitude": pl.Float64,
        "longitude": pl.Float64,
        "mc_weight_cell_hours": pl.Float64,
    }
)

INTEGRATION_DOMAIN_SCHEMA = pl.Schema(
    {
        "source_city": pl.String,
        "osm_h3_cell_id": pl.Int64,
        "h3_r9": pl.String,
        "in_authoritative_domain": pl.Boolean,
        "observed_in_training": pl.Boolean,
        "domain_origin": pl.String,
    }
)

TEMPORAL_COVERAGE_COLUMNS = (
    "source_city",
    "source_timezone",
    "coverage_start_utc",
    "coverage_end_utc",
    "coverage_basis",
    "coverage_reference",
)


@dataclass(frozen=True)
class TemporalCoverageInterval:
    """One audited, outcome-independent, half-open source interval."""

    source_city: str
    source_timezone: str
    start_utc: datetime
    end_utc: datetime
    coverage_basis: str
    coverage_reference: str

    @property
    def start_us(self) -> int:
        delta = self.start_utc - datetime(1970, 1, 1, tzinfo=UTC)
        return (
            (delta.days * 86_400 + delta.seconds) * 1_000_000
            + delta.microseconds
        )

    @property
    def duration_us(self) -> int:
        delta = self.end_utc - self.start_utc
        return (
            (delta.days * 86_400 + delta.seconds) * 1_000_000
            + delta.microseconds
        )

    def as_manifest_record(self) -> dict[str, str]:
        return {
            "coverage_start_utc": self.start_utc.isoformat(),
            "coverage_end_utc": self.end_utc.isoformat(),
            "source_timezone": self.source_timezone,
            "coverage_basis": self.coverage_basis,
            "coverage_reference": self.coverage_reference,
        }


@dataclass(frozen=True)
class H3SamplingGeometry:
    """Cached local-tangent geometry for fast sub-cell point sampling."""

    cell_ids: np.ndarray
    center_lat_rad: np.ndarray
    center_lon_rad: np.ndarray
    vertex_count: np.ndarray
    vertex_x_m: np.ndarray
    vertex_y_m: np.ndarray
    triangle_cdf: np.ndarray

    @property
    def max_vertices(self) -> int:
        return int(self.vertex_x_m.shape[1])


def source_seed(base_seed: int, source: str) -> int:
    """Derive a deterministic, source-specific RNG seed."""
    payload = f"{base_seed}:{source}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _parse_utc_timestamp(value: object, *, field: str, source: str) -> datetime:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{source}: {field} must be non-empty")

    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(
            f"{source}: {field} must be an ISO-8601 timestamp with a UTC offset"
        ) from error

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(
            f"{source}: {field} must include a UTC offset; got {text!r}"
        )
    return parsed.astimezone(UTC)


def resolve_temporal_coverage(
    rows: Sequence[Mapping[str, object]],
    *,
    source: str,
    start_year: int,
    end_year: int,
) -> tuple[list[TemporalCoverageInterval], np.ndarray, np.ndarray]:
    """Validate and clip audited source coverage to a source-local year window.

    Coverage comes only from the supplied full frozen provenance catalog. Crime
    timestamps are never inspected to create or extend an interval. Multiple
    non-overlapping rows preserve explicit gaps and source-era changes. Every
    modeled split is required to intersect declared support; an empty intersection
    is treated as a provenance/catalog error rather than silently as zero exposure.
    """
    if start_year > end_year:
        raise ValueError("start_year must be <= end_year")

    source_rows = [
        row for row in rows if str(row.get("source_city", "")).strip() == source
    ]
    if not source_rows:
        raise RuntimeError(
            f"{source}: no outcome-independent temporal coverage is declared"
        )

    timezones = {
        str(row.get("source_timezone", "")).strip() for row in source_rows
    }
    if "" in timezones or len(timezones) != 1:
        raise ValueError(
            f"{source}: coverage rows must declare one non-empty source_timezone"
        )
    source_timezone = next(iter(timezones))
    try:
        local_zone = ZoneInfo(source_timezone)
    except ZoneInfoNotFoundError as error:
        raise ValueError(
            f"{source}: unknown source_timezone {source_timezone!r}"
        ) from error

    training_start = datetime(start_year, 1, 1, tzinfo=local_zone).astimezone(UTC)
    training_end = datetime(end_year + 1, 1, 1, tzinfo=local_zone).astimezone(UTC)

    intervals: list[TemporalCoverageInterval] = []
    for row in source_rows:
        basis = str(row.get("coverage_basis", "")).strip()
        reference = str(row.get("coverage_reference", "")).strip()
        if not basis or not reference:
            raise ValueError(
                f"{source}: every coverage row requires coverage_basis and "
                "coverage_reference"
            )

        declared_start = _parse_utc_timestamp(
            row.get("coverage_start_utc"),
            field="coverage_start_utc",
            source=source,
        )
        declared_end = _parse_utc_timestamp(
            row.get("coverage_end_utc"),
            field="coverage_end_utc",
            source=source,
        )
        if declared_start >= declared_end:
            raise ValueError(
                f"{source}: coverage intervals must be positive and half-open"
            )

        clipped_start = max(declared_start, training_start)
        clipped_end = min(declared_end, training_end)
        if clipped_start >= clipped_end:
            continue

        intervals.append(
            TemporalCoverageInterval(
                source_city=source,
                source_timezone=source_timezone,
                start_utc=clipped_start,
                end_utc=clipped_end,
                coverage_basis=basis,
                coverage_reference=reference,
            )
        )

    intervals.sort(key=lambda interval: interval.start_utc)
    if not intervals:
        raise RuntimeError(
            f"{source}: declared temporal coverage does not intersect the "
            f"{start_year}-{end_year} source-local window"
        )

    for previous, current in zip(intervals, intervals[1:], strict=False):
        if current.start_utc < previous.end_utc:
            raise ValueError(
                f"{source}: temporal coverage intervals overlap: "
                f"{previous.end_utc.isoformat()} > {current.start_utc.isoformat()}"
            )

    starts_us = np.asarray(
        [interval.start_us for interval in intervals], dtype=np.int64
    )
    durations_us = np.asarray(
        [interval.duration_us for interval in intervals], dtype=np.int64
    )
    return intervals, starts_us, durations_us


def effective_coverage_local_years(
    intervals: Sequence[TemporalCoverageInterval],
) -> list[int]:
    """Return local calendar years intersecting effective half-open coverage."""
    if not intervals:
        raise ValueError("intervals must be non-empty")

    sources = {interval.source_city for interval in intervals}
    timezones = {interval.source_timezone for interval in intervals}
    if len(sources) != 1 or len(timezones) != 1:
        raise ValueError("intervals must belong to one source and timezone")

    local_zone = ZoneInfo(next(iter(timezones)))
    years: set[int] = set()
    for interval in intervals:
        if interval.start_utc >= interval.end_utc:
            raise ValueError("coverage intervals must be positive")

        first_year = interval.start_utc.astimezone(local_zone).year
        # Intervals are half-open. Subtracting one microsecond prevents an end
        # exactly at local New Year from admitting the following boundary
        # vintage.
        last_year = (
            interval.end_utc - timedelta(microseconds=1)
        ).astimezone(local_zone).year
        years.update(range(first_year, last_year + 1))

    return sorted(years)


def validate_authoritative_boundary_years(
    *,
    source: str,
    effective_local_coverage_years: Sequence[int],
    authoritative_boundary_years: Sequence[int],
) -> list[int]:
    """Require one authoritative boundary vintage for every covered local year."""
    effective = {int(year) for year in effective_local_coverage_years}
    authoritative = {int(year) for year in authoritative_boundary_years}
    missing = sorted(effective - authoritative)
    if missing:
        raise RuntimeError(
            f"{source}: missing authoritative boundary vintages: {missing}"
        )
    return sorted(authoritative)


def select_training_events(
    events: pl.LazyFrame,
    *,
    source: str,
    source_timezone: str,
    starts_us: np.ndarray,
    durations_us: np.ndarray,
) -> pl.LazyFrame:
    """Select events inside declared coverage without feature gating."""
    available = set(events.collect_schema().names())
    required = {
        "source_city",
        "occurrence_timestamp_utc",
        "osm_h3_cell_id",
    }
    missing = required - available
    if missing:
        raise KeyError(f"Event spine is missing columns: {sorted(missing)}")

    starts = np.asarray(starts_us, dtype=np.int64)
    durations = np.asarray(durations_us, dtype=np.int64)
    if starts.size == 0 or starts.size != durations.size:
        raise ValueError("starts_us and durations_us must be non-empty and aligned")
    if np.any(durations <= 0):
        raise ValueError("durations_us must be positive")

    timestamp = pl.col("occurrence_timestamp_utc").cast(
        pl.Datetime("us", time_zone="UTC"), strict=False
    )
    h3_cell = pl.col("osm_h3_cell_id").cast(pl.Int64, strict=False)
    source_events = events.filter(pl.col("source_city") == source)
    required_field_quality = (
        source_events.select(
            pl.len().alias("source_rows"),
            timestamp.is_null().sum().alias("invalid_occurrence_timestamp_rows"),
            h3_cell.is_null().sum().alias("invalid_osm_h3_cell_id_rows"),
        )
        .collect(engine="streaming")
        .row(0, named=True)
    )
    invalid_required = {
        name: int(required_field_quality[name])
        for name in (
            "invalid_occurrence_timestamp_rows",
            "invalid_osm_h3_cell_id_rows",
        )
        if int(required_field_quality[name]) != 0
    }
    if invalid_required:
        raise RuntimeError(
            f"{source}: event spine contains invalid required fields: "
            f"{invalid_required}"
        )

    timestamp_us = timestamp.dt.epoch("us")
    in_declared_coverage = pl.any_horizontal(
        *[
            (timestamp_us >= int(start))
            & (timestamp_us < int(start + duration))
            for start, duration in zip(starts, durations, strict=True)
        ]
    )

    return (
        source_events.filter(in_declared_coverage)
        .select(
            timestamp
            .dt.convert_time_zone(source_timezone)
            .dt.year()
            .alias("event_year"),
            h3_cell.alias("osm_h3_cell_id"),
        )
    )


def integration_sample_count(
    *,
    observed_event_count: int,
    samples_per_event: int,
) -> int:
    """Return ``M_s = K * N_s`` with explicit contract validation."""
    if observed_event_count <= 0:
        raise ValueError("observed_event_count must be positive")
    if samples_per_event <= 0:
        raise ValueError("samples_per_event must be positive")
    return observed_event_count * samples_per_event


def validate_h3_r9_cells(
    cells_hex: list[str],
    *,
    resolution: int,
    label: str,
) -> None:
    invalid = [
        cell
        for cell in cells_hex
        if not h3.is_valid_cell(str(cell))
        or h3.get_resolution(str(cell)) != resolution
    ]
    if invalid:
        raise RuntimeError(
            f"{label}: invalid/non-r{resolution} H3 cells; sample={invalid[:10]}"
        )


def build_frozen_source_domain(
    *,
    source: str,
    official_cells: np.ndarray,
    event_cells: np.ndarray,
) -> tuple[np.ndarray, pl.DataFrame]:
    """Union authoritative support with all observed training-event cells."""
    official_cells = np.unique(
        np.asarray(official_cells, dtype=np.int64)
    )
    event_cells = np.unique(
        np.asarray(event_cells, dtype=np.int64)
    )

    domain_cells = np.union1d(
        official_cells,
        event_cells,
    ).astype(np.int64, copy=False)

    if domain_cells.size == 0:
        raise RuntimeError(f"{source}: frozen integration domain is empty")

    in_official = np.isin(
        domain_cells,
        official_cells,
        assume_unique=True,
    )
    observed_training = np.isin(
        domain_cells,
        event_cells,
        assume_unique=True,
    )

    domain_df = (
        pl.DataFrame(
            {
                "source_city": [source] * int(domain_cells.size),
                "osm_h3_cell_id": domain_cells,
                "h3_r9": [
                    h3.int_to_str(int(cell))
                    for cell in domain_cells
                ],
                "in_authoritative_domain": in_official,
                "observed_in_training": observed_training,
            }
        )
        .with_columns(
            pl.when(
                pl.col("in_authoritative_domain")
                & pl.col("observed_in_training")
            )
            .then(pl.lit("official_and_observed"))
            .when(pl.col("in_authoritative_domain"))
            .then(pl.lit("official_only"))
            .otherwise(pl.lit("training_event_extension"))
            .alias("domain_origin")
        )
        .cast(INTEGRATION_DOMAIN_SCHEMA)
    )

    return domain_cells, domain_df


def monte_carlo_cell_hour_weight(
    *,
    domain_cell_count: int,
    temporal_support_hours: float,
    sample_count: int,
) -> float:
    """Weight for uniform sampling over frozen H3 cells × training time."""
    if domain_cell_count <= 0:
        raise ValueError("domain_cell_count must be positive")
    if temporal_support_hours <= 0:
        raise ValueError("temporal_support_hours must be positive")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")

    return (
        float(domain_cell_count)
        * float(temporal_support_hours)
        / float(sample_count)
    )


def prepare_h3_sampling_geometry(
    domain_cells: np.ndarray,
) -> H3SamplingGeometry:
    """Precompute local-tangent triangle fans for each H3 cell.

    Each H3 cell is tiny at r9, so a local tangent plane is an excellent
    approximation to surface-area-uniform sampling. The triangle fan uses
    the H3 cell center plus consecutive boundary vertices.

    This is intentionally cached once per source, not recomputed per row.
    """
    cell_ids = np.asarray(domain_cells, dtype=np.int64)

    if cell_ids.size == 0:
        raise ValueError("Cannot prepare geometry for an empty H3 domain")

    centers_lat: list[float] = []
    centers_lon: list[float] = []
    boundaries: list[list[tuple[float, float]]] = []

    max_vertices = 0

    for cell_int in cell_ids:
        cell = h3.int_to_str(int(cell_int))
        center_lat, center_lon = h3.cell_to_latlng(cell)
        boundary = list(h3.cell_to_boundary(cell))

        if len(boundary) < 3:
            raise RuntimeError(
                f"H3 cell {cell} has invalid boundary vertex count={len(boundary)}"
            )

        centers_lat.append(float(center_lat))
        centers_lon.append(float(center_lon))
        boundaries.append(
            [(float(lat), float(lon)) for lat, lon in boundary]
        )
        max_vertices = max(max_vertices, len(boundary))

    center_lat_rad = np.radians(
        np.asarray(centers_lat, dtype=np.float64)
    )
    center_lon_rad = np.radians(
        np.asarray(centers_lon, dtype=np.float64)
    )

    n_cells = int(cell_ids.size)

    vertex_count = np.zeros(n_cells, dtype=np.int16)
    vertex_x_m = np.zeros(
        (n_cells, max_vertices),
        dtype=np.float64,
    )
    vertex_y_m = np.zeros(
        (n_cells, max_vertices),
        dtype=np.float64,
    )
    triangle_cdf = np.ones(
        (n_cells, max_vertices),
        dtype=np.float64,
    )

    for i, boundary in enumerate(boundaries):
        lat0 = center_lat_rad[i]
        lon0 = center_lon_rad[i]

        nv = len(boundary)
        vertex_count[i] = nv

        lats = np.radians(
            np.asarray([p[0] for p in boundary], dtype=np.float64)
        )
        lons = np.radians(
            np.asarray([p[1] for p in boundary], dtype=np.float64)
        )

        dlon = (lons - lon0 + np.pi) % (2.0 * np.pi) - np.pi

        x = EARTH_RADIUS_M * np.cos(lat0) * dlon
        y = EARTH_RADIUS_M * (lats - lat0)

        vertex_x_m[i, :nv] = x
        vertex_y_m[i, :nv] = y

        x_next = np.roll(x, -1)
        y_next = np.roll(y, -1)

        # Center is the triangle fan's origin, so triangle area is
        # 1/2 |v_i x v_{i+1}|.
        areas = 0.5 * np.abs(
            x * y_next - y * x_next
        )

        total_area = float(areas.sum())
        if not np.isfinite(total_area) or total_area <= 0.0:
            raise RuntimeError(
                f"H3 cell {h3.int_to_str(int(cell_ids[i]))} "
                "has non-positive local polygon area"
            )

        cdf = np.cumsum(areas / total_area)
        cdf[-1] = 1.0
        triangle_cdf[i, :nv] = cdf

    return H3SamplingGeometry(
        cell_ids=cell_ids,
        center_lat_rad=center_lat_rad,
        center_lon_rad=center_lon_rad,
        vertex_count=vertex_count,
        vertex_x_m=vertex_x_m,
        vertex_y_m=vertex_y_m,
        triangle_cdf=triangle_cdf,
    )


def sample_latlon_within_h3(
    *,
    selected_cell_indices: np.ndarray,
    geometry: H3SamplingGeometry,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample one random sub-cell latitude/longitude for each selected cell.

    Sampling is uniform over a local-tangent polygon approximation to each
    H3 cell. At H3-r9 scale the curvature error is negligible for this
    auxiliary coordinate.

    Returns
    -------
    latitude, longitude : float64 arrays in degrees
    """
    cell_idx = np.asarray(
        selected_cell_indices,
        dtype=np.int64,
    )

    n = int(cell_idx.size)
    if n == 0:
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )

    if cell_idx.min() < 0 or cell_idx.max() >= geometry.cell_ids.size:
        raise IndexError("selected_cell_indices are outside cached H3 domain")

    # Pick a triangle in the center-to-edge fan, weighted by triangle area.
    u_triangle = rng.random(n)
    triangle_idx = np.zeros(n, dtype=np.int16)

    # Avoid materializing an (n_rows × max_vertices) CDF matrix.
    for k in range(geometry.max_vertices - 1):
        triangle_idx += (
            u_triangle
            > geometry.triangle_cdf[cell_idx, k]
        ).astype(np.int16)

    nv = geometry.vertex_count[cell_idx]
    if np.any(triangle_idx >= nv):
        raise RuntimeError("Triangle selection exceeded H3 boundary vertex count")

    next_idx = triangle_idx + 1
    next_idx = np.where(next_idx >= nv, 0, next_idx)

    v1x = geometry.vertex_x_m[cell_idx, triangle_idx]
    v1y = geometry.vertex_y_m[cell_idx, triangle_idx]
    v2x = geometry.vertex_x_m[cell_idx, next_idx]
    v2y = geometry.vertex_y_m[cell_idx, next_idx]

    # Uniform point in triangle with vertices (0, v1, v2).
    sqrt_u = np.sqrt(rng.random(n))
    mix = rng.random(n)

    x = sqrt_u * (
        (1.0 - mix) * v1x
        + mix * v2x
    )
    y = sqrt_u * (
        (1.0 - mix) * v1y
        + mix * v2y
    )

    lat0 = geometry.center_lat_rad[cell_idx]
    lon0 = geometry.center_lon_rad[cell_idx]

    lat_rad = lat0 + y / EARTH_RADIUS_M
    lon_rad = lon0 + x / (
        EARTH_RADIUS_M * np.cos(lat0)
    )

    lon_rad = (
        lon_rad + np.pi
    ) % (2.0 * np.pi) - np.pi

    return (
        np.degrees(lat_rad),
        np.degrees(lon_rad),
    )


def sample_integration_chunk(
    *,
    source: str,
    split: str,
    start_row: int,
    n: int,
    domain_cells: np.ndarray,
    geometry: H3SamplingGeometry,
    starts_us: np.ndarray,
    durations_us: np.ndarray,
    mc_weight_cell_hours: float,
    rng: np.random.Generator,
) -> pl.DataFrame:
    """Draw time first, then H3, then an auxiliary sub-cell coordinate."""

    if split not in MODEL_SPLITS:
        raise ValueError(f"unknown model split: {split!r}")

    if n <= 0:
        raise ValueError("n must be positive")

    if domain_cells.size == 0:
        raise ValueError("domain_cells must be non-empty")

    if starts_us.size == 0 or starts_us.size != durations_us.size:
        raise ValueError(
            "starts_us and durations_us must be non-empty and aligned"
        )

    if np.any(durations_us <= 0):
        raise ValueError("durations_us must be positive")

    cumulative_us = np.cumsum(
        durations_us,
        dtype=np.int64,
    )

    previous_cumulative = np.concatenate(
        [
            np.asarray([0], dtype=np.int64),
            cumulative_us[:-1],
        ]
    )

    total_duration_us = int(cumulative_us[-1])

    # Uniform over the union of effective temporal intervals.
    time_offsets = rng.integers(
        0,
        total_duration_us,
        size=n,
        dtype=np.int64,
    )

    interval_idx = np.searchsorted(
        cumulative_us,
        time_offsets,
        side="right",
    )

    sampled_timestamp_us = (
        starts_us[interval_idx]
        + time_offsets
        - previous_cumulative[interval_idx]
    )

    # Frozen TRAINING domain for every split.
    selected_cell_indices = rng.integers(
        0,
        domain_cells.size,
        size=n,
        dtype=np.int64,
    )

    sampled_h3 = domain_cells[selected_cell_indices]

    latitude, longitude = sample_latlon_within_h3(
        selected_cell_indices=selected_cell_indices,
        geometry=geometry,
        rng=rng,
    )

    sample_indices = np.arange(
        start_row,
        start_row + n,
        dtype=np.int64,
    )

    return (
        pl.DataFrame(
            {
                "sample_index": sample_indices,
                "_timestamp_us": sampled_timestamp_us,
                "osm_h3_cell_id": sampled_h3,
                "latitude": latitude,
                "longitude": longitude,
            }
        )
        .with_columns(
            pl.lit(source).alias("source_city"),
            pl.lit(split).alias("split"),
            pl.col("_timestamp_us")
            .cast(pl.Datetime("us", time_zone="UTC"))
            .alias("integration_timestamp_utc"),
            pl.lit(
                mc_weight_cell_hours,
                dtype=pl.Float64,
            ).alias("mc_weight_cell_hours"),
        )
        .select(
            "source_city",
            "split",
            "sample_index",
            "integration_timestamp_utc",
            "osm_h3_cell_id",
            "latitude",
            "longitude",
            "mc_weight_cell_hours",
        )
        .cast(INTEGRATION_SAMPLE_SCHEMA)
    )


__all__ = [
    "H3SamplingGeometry",
    "INTEGRATION_DOMAIN_SCHEMA",
    "INTEGRATION_SAMPLE_SCHEMA",
    "MODEL_SPLITS",
    "TEMPORAL_COVERAGE_COLUMNS",
    "TRAIN_SPLIT_END_UTC",
    "TemporalCoverageInterval",
    "VALIDATION_SPLIT_END_UTC",
    "build_frozen_source_domain",
    "effective_coverage_local_years",
    "integration_sample_count",
    "monte_carlo_cell_hour_weight",
    "prepare_h3_sampling_geometry",
    "sample_integration_chunk",
    "sample_latlon_within_h3",
    "select_training_events",
    "source_seed",
    "validate_authoritative_boundary_years",
    "resolve_temporal_coverage",
    "validate_h3_r9_cells",
]
