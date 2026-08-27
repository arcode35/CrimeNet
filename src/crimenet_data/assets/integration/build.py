"""Dagster orchestration for CrimeNet training integration sampling.

Frozen training support, per source:

    D_train,s =
        UNION(authoritative reporting-boundary H3-r9 cells from local-year
              vintages intersecting effective audited coverage)
        UNION(all observed event-spine H3-r9 cells inside the source's
              declared temporal coverage, clipped to 2014..2023)

Integration proposal:

    t ~ Uniform(training temporal support)
    h ~ Uniform(D_train,s)
    (lat, lon) ~ Uniform-within-selected-H3 approximation

The likelihood integration measure remains discrete H3 cell × continuous time,
so the Monte Carlo weight is:

    |D_s| * T_s_hours / M_s

Latitude/longitude are auxiliary sub-cell coordinates and do not alter that
cell-hour weight.
"""

import json
import math
from datetime import UTC, datetime
from typing import Any

import dagster as dg
import h3
import numpy as np
import polars as pl

from crimenet_data.assets.event_spine.schema import (
    EVENT_SPINE_UNMATCHED_HISTORY_POLICY,
    H3_RESOLUTION,
)
from crimenet_data.assets.crime.sources.registry import SOURCES
from crimenet_data.assets.integration.transforms import (
    INTEGRATION_DOMAIN_SCHEMA,
    INTEGRATION_SAMPLE_SCHEMA,
    TEMPORAL_COVERAGE_COLUMNS,
    build_frozen_source_domain,
    effective_coverage_local_years,
    integration_sample_count,
    monte_carlo_cell_hour_weight,
    prepare_h3_sampling_geometry,
    resolve_temporal_coverage,
    sample_integration_chunk,
    select_training_events,
    source_seed,
    validate_authoritative_boundary_years,
    validate_h3_r9_cells,
)
from crimenet_data.resources.crime_lake import CrimeLakeResources


TRAIN_START_YEAR = 2014
TRAIN_END_YEAR = 2023
DEFAULT_SAMPLES_PER_EVENT = 5
DEFAULT_CHUNK_ROWS = 1_000_000
DEFAULT_SEED = 2026
INTEGRATION_SCHEMA_VERSION = "crime_integration_samples_v3"

integration_sampling_executor = dg.multiprocess_executor.configured(
    lambda config: {"max_concurrent": config["max_concurrent"]},
    name="integration_sampling_multiprocess_executor",
    config_schema={
        "max_concurrent": dg.Field(int, default_value=4),
    },
)


def _scan_base_domain(crime_lake: CrimeLakeResources) -> pl.LazyFrame:
    uri = crime_lake.base_domain_uri
    lf = pl.scan_csv(
        uri,
        storage_options=crime_lake.storage_options_for(uri),
        credential_provider=None,
    )

    required = {
        "source_city",
        "event_year",
        "h3_r9",
    }
    missing = required - set(
        lf.collect_schema().names()
    )
    if missing:
        raise RuntimeError(
            "Base integration-domain CSV is missing columns: "
            f"{sorted(missing)}"
        )

    return lf


def _scan_temporal_coverage(crime_lake: CrimeLakeResources) -> pl.LazyFrame:
    uri = crime_lake.temporal_coverage_uri
    lf = pl.scan_csv(
        uri,
        storage_options=crime_lake.storage_options_for(uri),
        credential_provider=None,
        schema_overrides={column: pl.String for column in TEMPORAL_COVERAGE_COLUMNS},
    )
    missing = set(TEMPORAL_COVERAGE_COLUMNS) - set(lf.collect_schema().names())
    if missing:
        raise RuntimeError(
            "Temporal-coverage CSV is missing columns: " f"{sorted(missing)}"
        )
    return lf.select(*TEMPORAL_COVERAGE_COLUMNS)


def _resolve_event_spine_snapshot(
    crime_lake: CrimeLakeResources,
    snapshot_override_uri: str,
) -> tuple[str, dict[str, Any]]:
    snapshot_uri, manifest = crime_lake.resolve_event_spine_snapshot(
        snapshot_override_uri=snapshot_override_uri or None
    )
    return snapshot_uri, dict(manifest)


def _event_spine_dropped_rows(
    manifest: dict[str, Any],
    *,
    require_zero: bool,
) -> int:
    """Enforce that feature matching did not shrink observed-event support."""
    if require_zero and "dropped_rows" not in manifest:
        raise RuntimeError(
            "The selected Gold event-spine manifest does not report dropped_rows; "
            "cannot prove that feature availability preserved the observed-event "
            "H3 footprint."
        )
    dropped_rows = int(manifest.get("dropped_rows", 0) or 0)
    if require_zero and dropped_rows != 0:
        raise RuntimeError(
            "The selected Gold event spine reports "
            f"dropped_rows={dropped_rows:,}. Feature availability must not shrink "
            "the observed-event H3 footprint. Rebuild a lossless event snapshot "
            "or explicitly disable this guard only for a documented audit run."
        )
    unmatched_policy = manifest.get("unmatched_history_policy")
    if require_zero and unmatched_policy != EVENT_SPINE_UNMATCHED_HISTORY_POLICY:
        raise RuntimeError(
            "The selected Gold event spine does not retain feature-unmatched "
            "events with null features: "
            f"unmatched_history_policy={unmatched_policy!r}. Rebuild the event "
            "spine with the lossless history policy before integration sampling."
        )
    return dropped_rows


def _write_parquet(
    df: pl.DataFrame,
    uri: str,
    crime_lake: CrimeLakeResources,
) -> None:
    df.lazy().sink_parquet(
        uri,
        compression="zstd",
        compression_level=3,
        statistics=True,
        maintain_order=False,
        storage_options=crime_lake.storage_options_for(uri),
        credential_provider=None,
        mkdir=True,
        engine="streaming",
    )


def _validate_parquet_readback(
    *,
    uris: str | list[str],
    expected_schema: pl.Schema,
    expected_rows: int,
    expected_source: str,
    label: str,
    crime_lake: CrimeLakeResources,
) -> int:
    """Validate persisted integration schema, row count, and source grain."""
    uri_list = [uris] if isinstance(uris, str) else uris
    if not uri_list:
        raise RuntimeError(f"{label}: no Parquet objects were supplied for read-back")
    scanned = pl.scan_parquet(
        uri_list,
        storage_options=crime_lake.storage_options_for(uri_list[0]),
        credential_provider=None,
        hive_partitioning=False,
    )
    actual_schema = scanned.collect_schema()
    if actual_schema != expected_schema:
        raise RuntimeError(
            f"{label}: persisted schema mismatch: "
            f"expected={dict(expected_schema)}, actual={dict(actual_schema)}"
        )
    summary = (
        scanned.select(
            pl.len().alias("row_count"),
            pl.col("source_city").n_unique().alias("source_count"),
            pl.col("source_city").first().alias("source_city"),
        )
        .collect(engine="streaming")
        .row(0, named=True)
    )
    actual_rows = int(summary["row_count"])
    if actual_rows != expected_rows:
        raise RuntimeError(
            f"{label}: persisted row-count mismatch: "
            f"expected={expected_rows:,}, actual={actual_rows:,}"
        )
    if int(summary["source_count"]) != 1 or summary["source_city"] != expected_source:
        raise RuntimeError(
            f"{label}: persisted source grain mismatch: "
            f"expected={expected_source!r}, actual={summary}"
        )
    return actual_rows


@dg.op(
    out=dg.DynamicOut(dict),
    required_resource_keys={"crime_lake"},
    config_schema={
        "event_spine_snapshot_override_uri": dg.Field(
            str,
            default_value="",
            description=(
                "Optional immutable canonical snapshot override for forensic "
                "reproduction. Empty resolves CrimeLake's current event spine."
            ),
        ),
        "train_start_year": dg.Field(
            int,
            default_value=TRAIN_START_YEAR,
        ),
        "train_end_year": dg.Field(
            int,
            default_value=TRAIN_END_YEAR,
        ),
        "samples_per_event": dg.Field(
            int,
            default_value=DEFAULT_SAMPLES_PER_EVENT,
        ),
        "chunk_rows": dg.Field(
            int,
            default_value=DEFAULT_CHUNK_ROWS,
        ),
        "seed": dg.Field(
            int,
            default_value=DEFAULT_SEED,
        ),
        "sources": dg.Field(
            [str],
            default_value=[],
        ),
        "require_zero_dropped_spine_rows": dg.Field(
            bool,
            default_value=True,
        ),
    },
)
def emit_integration_source_plans(
    context,
):
    """Resolve immutable inputs once and fan out one source per Dagster step."""
    crime_lake: CrimeLakeResources = (
        context.resources.crime_lake
    )
    cfg = context.op_config

    start_year = int(cfg["train_start_year"])
    end_year = int(cfg["train_end_year"])
    samples_per_event = int(
        cfg["samples_per_event"]
    )
    chunk_rows = int(cfg["chunk_rows"])

    if start_year > end_year:
        raise ValueError(
            "train_start_year must be <= train_end_year"
        )
    if samples_per_event <= 0:
        raise ValueError(
            "samples_per_event must be positive"
        )
    if chunk_rows <= 0:
        raise ValueError(
            "chunk_rows must be positive"
        )
    if H3_RESOLUTION != 9:
        raise RuntimeError(
            "This integration contract expects H3-r9; "
            f"repo has H3_RESOLUTION={H3_RESOLUTION}"
        )

    (
        snapshot_uri,
        spine_manifest,
    ) = _resolve_event_spine_snapshot(
        crime_lake,
        str(
            cfg["event_spine_snapshot_override_uri"]
        ).strip(),
    )

    dropped_rows = _event_spine_dropped_rows(
        spine_manifest,
        require_zero=bool(cfg["require_zero_dropped_spine_rows"]),
    )

    base = (
        _scan_base_domain(crime_lake)
        .filter(
            pl.col("event_year")
            .cast(
                pl.Int64,
                strict=False,
            )
            .is_between(
                start_year,
                end_year,
                closed="both",
            )
        )
        .select(
            pl.col("source_city").cast(
                pl.String
            )
        )
        .unique()
        .collect(engine="streaming")
    )

    available_sources = sorted(
        str(x)
        for x in (
            base.get_column("source_city")
            .drop_nulls()
            .to_list()
        )
    )

    requested_sources = [
        str(x)
        for x in cfg["sources"]
    ]

    if requested_sources:
        missing = sorted(
            set(requested_sources)
            - set(available_sources)
        )
        if missing:
            raise RuntimeError(
                "Requested sources absent from "
                f"base_domain_h3.csv: {missing}"
            )
        sources = sorted(
            set(requested_sources)
        )
    else:
        sources = available_sources

    if not sources:
        raise RuntimeError(
            "No integration-sampling sources resolved"
        )

    coverage_rows = (
        _scan_temporal_coverage(crime_lake)
        .filter(pl.col("source_city").is_in(sources))
        .collect(engine="streaming")
        .to_dicts()
    )
    coverage_by_source: dict[str, dict[str, Any]] = {}
    for source in sources:
        intervals, starts_us, durations_us = resolve_temporal_coverage(
            coverage_rows,
            source=source,
            start_year=start_year,
            end_year=end_year,
        )
        source_timezone = intervals[0].source_timezone
        registered = SOURCES.get(source)
        if registered is not None and registered.config.timezone != source_timezone:
            raise RuntimeError(
                f"{source}: temporal coverage timezone {source_timezone!r} "
                "does not match the registered source timezone "
                f"{registered.config.timezone!r}"
            )
        coverage_by_source[source] = {
            "source_timezone": source_timezone,
            "effective_local_coverage_years": effective_coverage_local_years(
                intervals
            ),
            "temporal_coverage_intervals": [
                interval.as_manifest_record() for interval in intervals
            ],
            "temporal_interval_starts_us": starts_us.tolist(),
            "temporal_interval_durations_us": durations_us.tolist(),
        }

    snapshot_root = crime_lake.integration_snapshot_uri(context.run_id)
    if crime_lake._object_exists(crime_lake.integration_success_uri(snapshot_root)):
        raise RuntimeError(
            f"Integration-sampling snapshot is already complete: {snapshot_root}"
        )

    context.log.info(
        "integration_sampling_plan "
        f"sources={len(sources)} "
        f"event_spine_snapshot={snapshot_uri} "
        f"temporal_coverage={crime_lake.temporal_coverage_uri} "
        f"training_years={start_year}-{end_year} "
        f"samples_per_event={samples_per_event} "
        f"snapshot_root={snapshot_root}"
    )

    for source in sources:
        source_coverage = coverage_by_source[source]
        yield dg.DynamicOutput(
            {
                "source_city": source,
                **source_coverage,
                "event_spine_snapshot_uri": (
                    snapshot_uri
                ),
                "event_spine_snapshot_id": (
                    spine_manifest.get(
                        "snapshot_id"
                    )
                ),
                "event_spine_dropped_rows": dropped_rows,
                "snapshot_root": snapshot_root,
                "train_start_year": start_year,
                "train_end_year": end_year,
                "samples_per_event": (
                    samples_per_event
                ),
                "chunk_rows": chunk_rows,
                "seed": int(cfg["seed"]),
            },
            mapping_key=source,
        )


@dg.op(
    required_resource_keys={"crime_lake"},
    retry_policy=dg.RetryPolicy(
        max_retries=2,
        delay=30,
    ),
)
def sample_source_integration(
    context,
    plan: dict,
) -> dict:
    """Build one source domain and materialize its integration samples."""
    crime_lake: CrimeLakeResources = (
        context.resources.crime_lake
    )

    source = str(plan["source_city"])
    start_year = int(
        plan["train_start_year"]
    )
    end_year = int(
        plan["train_end_year"]
    )
    snapshot_uri = str(
        plan["event_spine_snapshot_uri"]
    )
    snapshot_root = str(
        plan["snapshot_root"]
    )
    k = int(plan["samples_per_event"])
    chunk_rows = int(
        plan["chunk_rows"]
    )

    # --------------------------------------------------------------
    # 1. Authoritative support only from local calendar-year vintages
    #    that intersect effective audited temporal coverage.
    # --------------------------------------------------------------
    effective_local_coverage_years = [
        int(year) for year in plan["effective_local_coverage_years"]
    ]
    base_rows = (
        _scan_base_domain(crime_lake)
        .filter(
            (pl.col("source_city") == source)
            & pl.col("event_year")
            .cast(pl.Int64, strict=False)
            .is_in(effective_local_coverage_years)
        )
        .select(
            pl.col("event_year").cast(
                pl.Int64,
                strict=False,
            ),
            pl.col("h3_r9").cast(
                pl.String
            ),
        )
        .drop_nulls()
        .unique()
        .collect(engine="streaming")
    )

    authoritative_boundary_years = sorted(
        int(year) for year in base_rows.get_column("event_year").unique().to_list()
    )
    authoritative_boundary_years = validate_authoritative_boundary_years(
        source=source,
        effective_local_coverage_years=effective_local_coverage_years,
        authoritative_boundary_years=authoritative_boundary_years,
    )

    official_hex = (
        base_rows.get_column("h3_r9")
        .unique()
        .to_list()
    )

    validate_h3_r9_cells(
        [str(cell) for cell in official_hex],
        resolution=H3_RESOLUTION,
        label=source,
    )

    official_cells = np.asarray(
        sorted(
            {
                h3.str_to_int(
                    str(cell)
                )
                for cell in official_hex
            }
        ),
        dtype=np.int64,
    )

    # --------------------------------------------------------------
    # 2. Every observed training-event H3 cell.
    #    Source partition prefix prevents every worker from scanning
    #    the full national event spine.
    # --------------------------------------------------------------
    source_globs = [
        crime_lake.event_spine_source_year_glob(
            snapshot_uri,
            source_city=source,
            occurrence_year=year,
        )
        for year in effective_local_coverage_years
    ]

    spine = pl.scan_parquet(
        source_globs,
        storage_options=(
            crime_lake.storage_options
        ),
        credential_provider=None,
        hive_partitioning=True,
    )

    events = select_training_events(
        spine,
        source=source,
        source_timezone=str(plan["source_timezone"]),
        starts_us=np.asarray(plan["temporal_interval_starts_us"], dtype=np.int64),
        durations_us=np.asarray(
            plan["temporal_interval_durations_us"], dtype=np.int64
        ),
    ).collect(engine="streaming")

    if events.is_empty():
        raise RuntimeError(
            f"{source}: zero observed training "
            f"events in {start_year}-{end_year}"
        )

    event_cells = (
        events.select(
            "osm_h3_cell_id"
        )
        .unique()
        .get_column(
            "osm_h3_cell_id"
        )
        .to_numpy()
        .astype(
            np.int64,
            copy=False,
        )
    )
    validate_h3_r9_cells(
        [h3.int_to_str(int(cell)) for cell in event_cells],
        resolution=H3_RESOLUTION,
        label=f"{source} observed training events",
    )

    # --------------------------------------------------------------
    # 3. Frozen training domain.
    # --------------------------------------------------------------
    (
        domain_cells,
        domain_df,
    ) = build_frozen_source_domain(
        source=source,
        official_cells=official_cells,
        event_cells=event_cells,
    )

    domain_uri = crime_lake.integration_domain_uri(snapshot_root, source)

    _write_parquet(
        domain_df,
        domain_uri,
        crime_lake,
    )

    # Cache H3 polygons ONCE per source.
    sampling_geometry = (
        prepare_h3_sampling_geometry(
            domain_cells
        )
    )

    # --------------------------------------------------------------
    # 4. Time-first Monte Carlo sampling.
    # --------------------------------------------------------------
    starts_us = np.asarray(plan["temporal_interval_starts_us"], dtype=np.int64)
    durations_us = np.asarray(
        plan["temporal_interval_durations_us"], dtype=np.int64
    )

    total_duration_us = int(
        durations_us.sum()
    )
    if total_duration_us <= 0:
        raise RuntimeError(
            f"{source}: invalid temporal support"
        )

    temporal_support_hours = (
        total_duration_us
        / 3_600_000_000.0
    )

    observed_events = int(
        events.height
    )
    sample_rows = integration_sample_count(
        observed_event_count=observed_events,
        samples_per_event=k,
    )

    mc_weight = (
        monte_carlo_cell_hour_weight(
            domain_cell_count=int(
                domain_cells.size
            ),
            temporal_support_hours=(
                temporal_support_hours
            ),
            sample_count=sample_rows,
        )
    )

    seed = source_seed(
        int(plan["seed"]),
        source,
    )
    rng = np.random.default_rng(
        seed
    )

    samples_prefix = crime_lake.integration_samples_prefix(snapshot_root, source)
    part_count = math.ceil(
        sample_rows / chunk_rows
    )

    for part_idx, start_row in enumerate(
        range(
            0,
            sample_rows,
            chunk_rows,
        )
    ):
        n = min(
            chunk_rows,
            sample_rows - start_row,
        )

        chunk = sample_integration_chunk(
            source=source,
            start_row=start_row,
            n=n,
            domain_cells=domain_cells,
            geometry=sampling_geometry,
            starts_us=starts_us,
            durations_us=durations_us,
            mc_weight_cell_hours=(
                mc_weight
            ),
            rng=rng,
        )

        part_uri = crime_lake.integration_sample_part_uri(
            snapshot_root,
            source,
            part_idx,
        )

        _write_parquet(
            chunk,
            part_uri,
            crime_lake,
        )

        context.log.info(
            f"{source}: wrote integration "
            f"part {part_idx + 1}/"
            f"{part_count} rows={n:,}"
        )

    training_extension_cells = int(
        domain_df.filter(
            pl.col("domain_origin")
            == "training_event_extension"
        ).height
    )

    result = {
        "source_city": source,
        "event_spine_snapshot_uri": (
            snapshot_uri
        ),
        "event_spine_snapshot_id": (
            plan.get(
                "event_spine_snapshot_id"
            )
        ),
        "event_spine_dropped_rows": int(plan["event_spine_dropped_rows"]),
        "snapshot_root": snapshot_root,
        "train_start_year": (
            start_year
        ),
        "train_end_year": end_year,
        "source_timezone": str(plan["source_timezone"]),
        "effective_local_coverage_years": effective_local_coverage_years,
        "authoritative_boundary_years": authoritative_boundary_years,
        "temporal_coverage_intervals": plan["temporal_coverage_intervals"],
        "temporal_coverage_interval_count": len(
            plan["temporal_coverage_intervals"]
        ),
        "observed_training_events": (
            observed_events
        ),
        "authoritative_h3_cells": int(
            official_cells.size
        ),
        "observed_training_h3_cells": int(
            np.unique(
                event_cells
            ).size
        ),
        "integration_domain_h3_cells": int(
            domain_cells.size
        ),
        "training_event_extension_cells": (
            training_extension_cells
        ),
        "samples_per_event": k,
        "chunk_rows": chunk_rows,
        "integration_sample_rows": (
            sample_rows
        ),
        "sample_part_count": (
            part_count
        ),
        "temporal_support_hours": (
            temporal_support_hours
        ),
        "mc_weight_cell_hours": (
            mc_weight
        ),
        "subcell_coordinate_policy": (
            "random point inside selected H3-r9 "
            "cell via local-tangent triangle fan"
        ),
        "seed": seed,
        "domain_uri": domain_uri,
        "samples_prefix": (
            samples_prefix
        ),
    }

    context.log.info(
        f"{source}: "
        f"domain={domain_cells.size:,} "
        f"official={official_cells.size:,} "
        f"event_cells={np.unique(event_cells).size:,} "
        f"extensions={training_extension_cells:,} "
        f"events={observed_events:,} "
        f"samples={sample_rows:,}"
    )

    return result


@dg.op(
    required_resource_keys={"crime_lake"},
)
def publish_integration_sampling(
    context,
    source_results: list[dict],
) -> str:
    """Publish only after every dynamic source worker succeeds."""
    crime_lake: CrimeLakeResources = (
        context.resources.crime_lake
    )

    if not source_results:
        raise RuntimeError(
            "No source integration-sampling "
            "results were produced"
        )

    source_results = sorted(
        source_results,
        key=lambda x: str(
            x["source_city"]
        ),
    )

    roots = {
        str(x["snapshot_root"])
        for x in source_results
    }
    input_snapshots = {
        str(
            x[
                "event_spine_snapshot_uri"
            ]
        )
        for x in source_results
    }
    chunk_rows_values = {int(x["chunk_rows"]) for x in source_results}

    if len(roots) != 1:
        raise RuntimeError(
            "Mapped steps disagree on "
            f"output root: {roots}"
        )
    if len(input_snapshots) != 1:
        raise RuntimeError(
            "Mapped steps disagree on event "
            f"spine snapshot: {input_snapshots}"
        )
    if len(chunk_rows_values) != 1:
        raise RuntimeError(
            f"Mapped steps disagree on chunk_rows: {chunk_rows_values}"
        )

    snapshot_root = next(
        iter(roots)
    )
    event_spine_snapshot_uri = next(
        iter(input_snapshots)
    )
    chunk_rows = next(iter(chunk_rows_values))

    for result in source_results:
        domain_uri = str(result["domain_uri"])
        if not crime_lake._object_exists(domain_uri):
            raise RuntimeError(f"Missing integration domain output: {domain_uri}")
        samples_prefix = str(result["samples_prefix"])
        sample_uris: list[str] = []
        for part_index in range(int(result["sample_part_count"])):
            part_uri = crime_lake.integration_sample_part_uri(
                snapshot_root,
                str(result["source_city"]),
                part_index,
            )
            if not crime_lake._object_exists(part_uri):
                raise RuntimeError(f"Missing integration sample output: {part_uri}")
            sample_uris.append(part_uri)

        source = str(result["source_city"])
        result["domain_readback_rows"] = _validate_parquet_readback(
            uris=domain_uri,
            expected_schema=INTEGRATION_DOMAIN_SCHEMA,
            expected_rows=int(result["integration_domain_h3_cells"]),
            expected_source=source,
            label=f"{source} integration domain",
            crime_lake=crime_lake,
        )
        result["sample_readback_rows"] = _validate_parquet_readback(
            uris=sample_uris,
            expected_schema=INTEGRATION_SAMPLE_SCHEMA,
            expected_rows=int(result["integration_sample_rows"]),
            expected_source=source,
            label=f"{source} integration samples",
            crime_lake=crime_lake,
        )

    total_events = sum(
        int(
            x[
                "observed_training_events"
            ]
        )
        for x in source_results
    )
    total_samples = sum(
        int(
            x[
                "integration_sample_rows"
            ]
        )
        for x in source_results
    )
    total_domain_rows = sum(
        int(
            x[
                "integration_domain_h3_cells"
            ]
        )
        for x in source_results
    )
    total_extension_cells = sum(
        int(
            x[
                "training_event_extension_cells"
            ]
        )
        for x in source_results
    )

    manifest = {
        "schema_version": (
            INTEGRATION_SCHEMA_VERSION
        ),
        "snapshot_id": context.run_id,
        "snapshot_root": (
            snapshot_root
        ),
        "created_at_utc": (
            datetime.now(
                UTC
            ).isoformat()
        ),
        "event_spine_snapshot_uri": (
            event_spine_snapshot_uri
        ),
        "event_spine_dropped_rows": source_results[0][
            "event_spine_dropped_rows"
        ],
        "event_spine_snapshot_id": (
            source_results[0].get(
                "event_spine_snapshot_id"
            )
        ),
        "h3_resolution": (
            H3_RESOLUTION
        ),
        "base_domain_uri": crime_lake.base_domain_uri,
        "temporal_coverage_uri": crime_lake.temporal_coverage_uri,
        "temporal_coverage_schema": list(TEMPORAL_COVERAGE_COLUMNS),
        "temporal_coverage_policy": (
            "source-specific, provenance-bearing, half-open UTC intervals; "
            "clipped to source-local training-year bounds; no interval is "
            "inferred from crime timestamps"
        ),
        "chunk_rows": chunk_rows,
        "train_start_year": source_results[0]["train_start_year"],
        "train_end_year": source_results[0]["train_end_year"],
        "domain_schema": {
            name: str(dtype) for name, dtype in INTEGRATION_DOMAIN_SCHEMA.items()
        },
        "sample_schema": {
            name: str(dtype) for name, dtype in INTEGRATION_SAMPLE_SCHEMA.items()
        },
        "domain_policy": (
            "per source: union(authoritative reporting-boundary H3 cells from "
            "local calendar-year vintages intersecting effective audited "
            "coverage, all observed training-event H3 cells inside declared "
            "coverage); freeze that union across all effective intervals"
        ),
        "validation_test_domain_policy": (
            "training domain is frozen; "
            "validation/test outcomes must not "
            "expand it"
        ),
        "sampling_policy": (
            "time first: uniform over source "
            "training temporal support; then "
            "uniform over frozen source H3 "
            "domain; then random sub-cell lat/lon"
        ),
        "spatial_measure": (
            "discrete H3-r9 cell; latitude/"
            "longitude are auxiliary coordinates"
        ),
        "feature_availability_policy": (
            "feature availability does not gate "
            "integration sampling; enrichment "
            "must use left joins / missing-value "
            "handling"
        ),
        "monte_carlo_estimator": (
            "sum_source[(|D_s| * T_s_hours / "
            "M_s) * sum_j lambda_s(h_j,t_j)]"
        ),
        "source_count": len(
            source_results
        ),
        "observed_training_events": (
            total_events
        ),
        "integration_sample_rows": (
            total_samples
        ),
        "integration_domain_rows": (
            total_domain_rows
        ),
        "training_event_extension_cells": (
            total_extension_cells
        ),
        "sources": source_results,
    }

    crime_lake._write_object(
        crime_lake.integration_manifest_uri(snapshot_root),
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8"),
        content_type="application/json",
    )

    crime_lake._write_object(
        crime_lake.integration_success_uri(snapshot_root),
        b"",
        content_type=(
            "application/octet-stream"
        ),
    )

    pointer = {
        "snapshot_id": context.run_id,
        "snapshot_uri": snapshot_root,
        "created_at_utc": (
            manifest["created_at_utc"]
        ),
        "schema_version": (
            INTEGRATION_SCHEMA_VERSION
        ),
        "event_spine_snapshot_id": (
            manifest[
                "event_spine_snapshot_id"
            ]
        ),
    }

    crime_lake._write_object(
        crime_lake.integration_latest_pointer_uri,
        json.dumps(
            pointer,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8"),
        content_type="application/json",
    )

    context.log.info(
        "integration_sampling_published "
        f"sources={len(source_results)} "
        f"events={total_events:,} "
        f"samples={total_samples:,} "
        f"domain_rows={total_domain_rows:,} "
        f"extension_cells="
        f"{total_extension_cells:,} "
        f"snapshot_root={snapshot_root}"
    )

    return snapshot_root


@dg.job(
    executor_def=integration_sampling_executor,
)
def integration_sampling_job():
    plans = (
        emit_integration_source_plans()
    )
    source_results = plans.map(
        sample_source_integration
    )
    publish_integration_sampling(
        source_results.collect()
    )


__all__ = [
    "integration_sampling_job",
]
