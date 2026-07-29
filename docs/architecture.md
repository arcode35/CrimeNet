# Architecture

## Boundaries

CrimeNet keeps orchestration and business logic separate:

- `resources/jobs/crime_pipeline.job.yml` declares task dependencies and
  environment arguments.
- `src/crimenet/jobs` contains thin CLI entry points.
- source contracts, transforms, quality rules, API clients, spatial logic, and
  promotion helpers live in dedicated modules under `src/crimenet`.
- landing objects are immutable inputs; notebooks are optional exploration and
  are not called by the bundle.

The Databricks job uses a wheel installed into a serverless environment. Unity
Catalog holds managed schemas, volumes, and Delta tables.

## Deployed task graph

```text
preflight
├── bronze_dallas ─────┐
├── bronze_houston ────┼── silver_crime ── crime_quality_checks
├── bronze_fort_worth ─┘            │
│                                   ├── weather_request_plan
│                                   │   └── weather_api_retrieval
│                                   │       └── bronze_open_meteo_weather
│                                   │           └── silver_weather_hourly
│                                   ├── silver_lighting_conditions
│                                   └──────────────┐
├── acs_vintage_calendar            │
│   └── tiger_line_boundaries ──────┼── location_tract_mapping
└── land_acs5_tracts                │
    └── bronze_acs5_tracts          │
        └── silver_acs5_tracts      │
                                    │
crime_quality_checks + silver_weather_hourly
+ silver_lighting_conditions + silver_acs5_tracts
+ location_tract_mapping ─────────────── gold_crime_features
```

The job has `max_concurrent_runs: 1`. Retries are configured on external
retrieval and rebuild tasks. The logical replay guarantees do not rely on that
setting, but serial runs avoid two deployments intentionally promoting
different definition versions at the same time.

## Principal data products

All names below are relative to the configured catalog.

| Table | Grain/key | Write behavior |
|---|---|---|
| `bronze.dallas_crime` | source + stable raw-row hash | insert-only `MERGE` |
| `bronze.houston_crime` | source + stable raw-row hash | insert-only `MERGE` |
| `bronze.fort_worth_crime` | source + stable raw-row hash | insert-only `MERGE` |
| `bronze.open_meteo_weather` | source + stable response hash | insert-only `MERGE` |
| `bronze.acs5_tract_socioeconomic` | source + stable ACS-row hash | insert-only `MERGE` |
| `silver.crime_offenses` | `business_identity` | staged full replacement |
| `silver.weather_hourly` | provider/model/H3 cell/UTC hour | deterministic upsert |
| `silver.solar_lighting_conditions` | H3 cell/UTC hour/definition version | staged replace or insert |
| `silver.acs_vintage_calendar` | ACS vintage | staged replacement |
| `silver.tract_socioeconomic` | GEOID + ACS vintage | version-aware upsert or staged rebuild |
| `silver.census_tract_boundaries` | tract/vintage/boundary version | staged replacement |
| `silver.crime_location_tract_mapping` | versioned location/vintage | stale-aware merge or staged replacement |
| `gold.crime_features` | `crime_offense_id` | staged full replacement |
| `data_quality.quality_results` | run/table/check/source | upsert |
| `data_quality.*_quarantine` | stable reject identity | insert-only entity merge |
| `data_quality.*_quarantine_observations` | reject + run | insert-only observation merge |
| `ops.weather_request_manifest` | deterministic request ID | staged replacement |
| `ops.weather_request_failures` | deterministic run/request/error event | upsert |

## Time semantics

Spark sessions are set to UTC for canonical and feature processing. Dallas and
Houston source timestamps are interpreted as `America/Chicago` wall-clock time
and converted to UTC. Fort Worth epoch-millisecond values are treated as UTC
instants.

Weather observations are UTC timestamps. Lighting generation and Gold both use
`date_trunc("hour", occurred_at)` and the key
`lighting_query_cell_id + solar_timestamp_hour +
lighting_definition_version`. An offense at `14:37:22` therefore joins a
lighting feature at `14:00:00`.

ACS features are selected point-in-time using the vintage calendar. The
calendar represents known release dates; a vintage is only eligible after its
release date. Boundaries are selected using the corresponding TIGER/Line year.

## Identity and determinism

Bronze row hashes serialize logical fields in sorted order and exclude source
paths and operational timestamps. The first-seen path remains lineage only.

Canonical Silver identity is a SHA-256 hash of source system, incident ID, and
offense ID. Houston derives a deterministic offense ID from normalized incident
ID and NIBRS class because the supplied source lacks a reliable offense-level
key. Mutable occurrence and location corrections are excluded from identity.

Conflicting rows for one Silver identity are ranked using source update time,
record completeness, report time, and stable row hash. Ingestion time and path
only break ties after equal logical content, so they cannot change the chosen
business values.

Gold applies a separately versioned hash to the Silver business identity.
Catalog names, workspace paths, source paths, and ingestion timestamps are
absent from the Gold identity.

## Checkpoints and recovery

Auto Loader checkpoints record file-discovery progress and make ordinary runs
efficient. They are not a uniqueness mechanism. Bronze `foreachBatch` sinks
merge on stable logical hashes, and weather/ACS Silver sinks use deterministic
keys and version-aware updates. Replacing a checkpoint can cause files to be
read again, but a successful replay converges to the same logical tables.
When a persisted Weather or ACS target contains a stale definition version,
the corresponding job bypasses its advanced checkpoint, rebuilds from the
complete Bronze table, validates a stage, and promotes it.

External cache writes use a temporary file and atomic rename. Partial files fail
validation and are audited before refetch. The pipeline is intentionally
restartable after partial completion: already committed Delta tables and valid
download cache entries are reused.

## Staging and promotion

Full rebuilds write `<table>__staging__<normalized-run-id>`. Candidate
validators execute against that table. On failure, the stage is dropped and the
existing final table is not replaced. On success, CrimeNet uses one Delta
`CREATE OR REPLACE TABLE`/clone operation and then drops the stage.

This gives a single-table commit boundary. It is not a transaction across
quality results, quarantine, lookup tables, Silver, and Gold. A run can expose
some successfully committed upstream tables before a later task fails.
Rerunning is the recovery mechanism.

## Version-aware derivations

- municipal contracts carry a source-contract version;
- canonical offenses carry a transformation version;
- weather carries a weather-definition version;
- lighting includes its definition version in the physical key;
- ACS features carry a socioeconomic-definition version;
- boundaries and spatial mappings carry boundary/mapping versions and archive
  checksums;
- Gold identity has an explicit identity version.

When an algorithm changes, operators must bump its version. Full-rebuild
pipelines replace the current candidate; Weather and ACS automatically stage a
full Bronze rebuild when the current target contains a missing or stale
definition.

## Runtime boundary

Local Spark tests cover pure transformations, cardinality, keys, validation,
cache behavior, and generated merge/promotion logic. The following remain
Databricks integration boundaries:

- Unity Catalog grants and managed volume behavior;
- serverless environment dependency installation;
- native geometry SQL functions and supported runtime version;
- secret-scope access;
- external network routes and source quotas;
- service-principal job execution;
- authenticated Asset Bundle validation and deployment.
