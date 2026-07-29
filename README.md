# CrimeNet

CrimeNet is a Databricks lakehouse pipeline that standardizes municipal crime
records from Dallas, Houston, and Fort Worth and enriches them with historical
weather, solar-lighting conditions, Census ACS tract features, and TIGER/Line
tract geography.

The repository contains the production Python package, a Databricks Asset
Bundle, local Spark tests, and operational documentation. Production logic is
under `src/crimenet`; notebooks are exploratory and are not pipeline
dependencies.

CrimeNet is designed for replay safety and deterministic output. It does not
claim a cross-table transaction or universal exactly-once delivery. Each Delta
table is committed independently, external downloads can fail, and a complete
run requires a correctly configured Databricks workspace and network access.

## Data flow

```text
                                   ┌─ weather request plan
Dallas Bronze ─┐                   ├─ Open-Meteo retrieval/cache
Houston Bronze ├─ Silver crime ─ quality ─ Weather Bronze ─ Weather Silver ─┐
Fort Worth ────┘                   ├─ Lighting Silver ───────────────────────┤
                                   │                                        │
preflight ─ ACS vintage calendar ─ TIGER/Line boundaries ─ tract mapping ──┤
     └──── ACS landing ─ ACS Bronze ─ ACS Silver ───────────────────────────┤
                                                                            │
                                                               Gold features
```

The deployed dependency graph is defined in
`resources/jobs/crime_pipeline.job.yml`. Development preflight creates and
validates managed catalogs, schemas, and volumes. Production preflight only
validates resources that an administrator has already provisioned.

## Layer responsibilities

| Layer | Purpose | Principal grain or identity |
|---|---|---|
| Landing | Immutable municipal files and atomically cached API responses | External object |
| Bronze | Source-faithful records with contract and ingestion metadata | `source_system + source_row_hash` |
| Silver crime | Validated and deterministically deduplicated offenses | `business_identity` |
| Silver weather | One provider/model observation for an H3 cell and UTC hour | `provider + model + weather_query_cell_id + weather_timestamp` |
| Silver lighting | Solar condition for a query cell, UTC hour, and algorithm version | `lighting_query_cell_id + solar_timestamp_hour + lighting_definition_version` |
| Silver ACS | ACS 5-year tract feature record | `geoid + acs_vintage` |
| Silver boundaries | Normalized tract geometry for a TIGER/Line vintage | boundary record/version |
| Silver tract mapping | Versioned point-to-tract result | location, TIGER year, mapping and boundary versions |
| Gold | One point-in-time enriched crime offense | `crime_offense_id` |
| Quarantine | Stable rejected entity plus per-run observation | `quarantine_id`; observation adds `pipeline_run_id` |
| Data quality / operations | Checks, failures, manifests, and request audit | run/check or request event identity |

## Stable identities and replay behavior

`add_ingestion_metadata` hashes a sorted JSON representation of logical source
fields. It excludes file paths, ingestion timestamps, batch identifiers, and
other physical metadata. Bronze uses an insert-only Delta `MERGE` on
`source_system + source_row_hash`; Auto Loader checkpoints reduce scanning but
are not the correctness boundary. Replaying a copied file or replacing a
checkpoint therefore does not create another logical Bronze row.

Silver crime uses source incident and offense identifiers. Houston, whose
source does not expose a reliable offense identifier, derives one from the
normalized incident ID and NIBRS class; mutable correction fields are excluded.
Duplicate precedence is:

1. latest source update timestamp;
2. greatest canonical-field completeness;
3. latest report timestamp;
4. lexicographically greatest stable source-row hash;
5. ingestion timestamp and source path only as lineage tie-breakers after
   logical content is identical.

Gold hashes the stable Silver `business_identity` with an explicit identity
version. Moving a file or deploying to a different catalog does not change the
resulting `crime_offense_id`.

Weather and ACS merges use definition versions and deterministic source hashes,
so replay order within a deployed definition cannot churn the selected row.
If a persisted Weather or ACS target contains an older definition, the job
rebuilds the complete candidate from Bronze before promotion instead of relying
on an already-advanced streaming checkpoint.
Lighting includes its definition version in the physical key. Location mapping
recomputes missing or stale mapping/boundary versions.

## Validation and safe replacement

Rebuild-style jobs write a deterministic run-scoped staging Delta table first.
They validate the staged data and replace one final table only after validation
passes. Silver crime validates the exact business-key set, schema, row bounds,
source presence, timestamps, coordinates, critical null rates, Bronze-to-Silver
ratio, and quarantine rate before promotion. Gold validates:

- exact expected and produced offense-key equality;
- unique `crime_offense_id` values and unchanged cardinality;
- required schema, types, non-null fields, coordinates, and timestamps;
- duplicate lookup keys and unexpected many-to-many joins;
- ambiguous spatial matches;
- weather, lighting, tract, and ACS coverage;
- feature ranges.

Weather, lighting, and ACS coverage use usable feature predicates rather than
lookup-row presence; a matched row with missing required values fails
validation and does not count toward a threshold.

Quality evidence is merged into
`<catalog>.<data_quality_schema>.quality_results`. Blocking failures raise a
clear exception. The previous final table remains available if staging or
validation fails.

Promotion is atomic only for the single Delta table being replaced. CrimeNet
does not provide an atomic transaction across Silver, lookup, Gold, quarantine,
and quality tables. A failed run is resumed by rerunning it with the same
inputs; already committed tables and caches are reused.

## Quarantine

Crime, weather, ACS, boundary, and spatial mapping validation retain rejected
records where practical. A stable entity table prevents a replayed bad record
from multiplying. Its companion `<quarantine_table>_observations` table records
one sighting per `quarantine_id + pipeline_run_id`, so task retries are
idempotent while later runs remain auditable.

Quarantine records include source identity and lineage, a reason code and
message, validation fields or payload, the pipeline run, and observation time.
An invalid external response is not silently treated as a valid Silver record.

## External data behavior

### Open-Meteo

The request manifest is deterministic. For deployed GMT requests, a response
must contain the exact ordered inclusive hourly range and finite numeric
values. Responses are written with a unique temporary file, flushed and
`fsync`ed, validated, then atomically renamed into the persistent landing
cache. Valid cache hits avoid HTTP calls. Invalid cache files are audited and
refetched. HTTP timeouts, connection errors, 429s, and retryable 5xx statuses
use a bounded retry policy; non-retryable failures stop the task. Request
failures are merged into the operations schema without logging secrets or full
payloads.

### Census ACS

The ACS calendar selects eligible 5-year vintages without using future
releases. Landed JSON Lines files are written atomically and validated before
reuse. The production target requires at least 5,000 unique, valid state tract
GEOIDs per vintage before a response can be landed or reused. A Census API key
is optional and is read only from the configured Databricks secret scope/key.

### TIGER/Line

Official tract archives are downloaded with retry and size limits, written
atomically, checked for ZIP integrity, and normalized to WGS84. Geometry,
GEOID, duplicate, vintage, and minimum-tract-count checks run before boundary
promotion. Native Databricks geometry execution requires a runtime that
supports the geometry SQL functions used by the mapping job.

## Configuration

All environment names and paths are bundle variables in `databricks.yml`.
`targets/dev.yml` deliberately uses permissive enrichment thresholds.
`targets/prod.yml` supplies explicit blocking thresholds and runs the job as a
service principal supplied through `production_service_principal_name`; it
does not use the deploying interactive user as the intended production
identity. The targets also explicitly select preflight behavior and TIGER/Line
and spatial-mapping quality gates.

No secret value belongs in this repository. Set bundle variables with target
configuration, `--var`, or `BUNDLE_VAR_*`. The optional Census variables are:

- `census_secret_scope`
- `census_api_key_secret`

Production also requires `production_service_principal_name`.

## Local development

Install `uv`, Java 17 or another Spark-compatible JDK, and the Databricks CLI.
Then run:

```bash
uv lock --check
uv sync --locked --all-groups
uv run ruff check .
uv run mypy src
uv run pytest --cov=crimenet --cov-report=term-missing --cov-fail-under=80
uv build
databricks bundle validate -t dev
```

The final command requires a Databricks profile or `DATABRICKS_HOST` and an
authentication credential. Ordinary CI always checks YAML, entry-point
imports, CLI/bundle argument compatibility, the acyclic task graph, and the
absence of unimplemented production commands. Authenticated bundle validation
is a separate credential-gated CI step.

`scripts/check.sh` runs the complete validation sequence.
`scripts/deploy.sh dev|prod` adds bundle deployment after validation.

## Workspace prerequisites

The job identity needs:

- `USE CATALOG`, `USE SCHEMA`, and the ability to create, modify, and select
  the required Delta tables;
- read/write access to landing, Auto Loader schema, and checkpoint volumes;
- permission to run and manage the deployed job and its serverless
  environment;
- `READ` on the configured secret scope when an ACS key is used;
- external-location/storage permissions if a target replaces managed volumes
  with external storage;
- outbound HTTPS access to Census, Open-Meteo, and Census TIGER/Line hosts.

An identity running preflight in `create` mode additionally needs permission to
create the configured catalog, schemas, and managed volumes. The production
target uses `validate` mode and does not grant those provisioning permissions
to the job service principal.

See `docs/operations_runbook.md` for provisioning, migration, recovery, and
deployment commands.

## Known limitations

- Local tests exercise transformations and deterministic SQL generation, but
  Unity Catalog permissions, serverless networking, native geometry functions,
  and authenticated bundle deployment require a workspace.
- Delta commits are per table; a complete pipeline run is not a cross-table
  transaction.
- Source contracts intentionally fail on unsupported municipal header changes.
  A renamed or type-changed upstream field requires a new contract version and
  transform update.
- The pipeline assumes the configured source files represent offenses for all
  three supported cities. Production quality checks block if a city
  unexpectedly contributes no valid records.
- Point-in-polygon behavior depends on supported Databricks geometry functions
  and valid TIGER geometries. Ambiguous matches are measured and production
  defaults block them.
- External APIs and source archives have availability and quota limits.
  Persistent caches make retries recoverable but cannot make an unavailable
  source available.

## Claims and automated evidence

| Claim | Implementation | Automated evidence |
|---|---|---|
| Bronze replay and file relocation preserve logical identity | `ingestion/metadata.py`, `jobs/bronze_ingestion.py` | `test_ingestion_identity.py`, Delta integration replay tests |
| Silver business identities exclude physical metadata | `transforms/common.py`, city transforms | `test_crime_transforms.py`, `test_crime_deduplication_quarantine.py` |
| Duplicate resolution is deterministic | `transforms/deduplication.py` | `test_crime_deduplication_quarantine.py` |
| Rejected crime/weather/ACS rows have stable reasons | `quality/quarantine.py`, `quality/external.py` | quarantine and Silver transformation tests |
| Lighting uses a versioned UTC-hour grain | `contracts/lighting.py`, `silver/lighting.py` | `test_lighting_grain_version.py` |
| A 14:37 offense joins its 14:00 lighting row | `gold/crime_features.py` | `test_lighting_grain_version.py` |
| Gold IDs do not depend on paths | `contracts/gold.py`, `gold/crime_features.py` | `test_gold_identity_validation.py` |
| Gold validates exact keys and enrichment coverage | `gold/crime_features.py` | `test_gold_identity_validation.py` |
| Failed staged validation preserves the final table | `utils/promotion.py` and rebuild jobs | promotion unit/Delta integration tests |
| Weather retry/cache behavior is bounded and auditable | `weather/open_meteo_client.py`, `weather/cache.py`, `weather/weather_ingestion.py` | weather reliability/cache/resilience tests |
| ACS calendar and tract dependencies are production jobs | `socioeconomic/acs_calendar.py`, boundary and mapping jobs | ACS calendar, TIGER, and tract mapping tests |
| Bundle entry points and arguments are internally consistent | `pyproject.toml`, `resources/jobs/crime_pipeline.job.yml` | `test_bundle_contract.py` |

## More documentation

- [Architecture](docs/architecture.md)
- [Data contracts](docs/data_contracts.md)
- [Data quality](docs/data_quality.md)
- [Operations runbook](docs/operations_runbook.md)
