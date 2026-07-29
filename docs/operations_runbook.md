# Operations runbook

## Prerequisites

Local validation requires `uv`, Python 3.11+, a Spark-compatible JDK, and the
Databricks CLI. The workspace requires Unity Catalog and a serverless runtime
supporting the SQL and geometry functions used by CrimeNet.

The production service principal needs:

- workspace permission to run/manage the job;
- `USE CATALOG` on the configured catalog;
- `USE SCHEMA`, create-table, modify, and select privileges on Bronze, Silver,
  Gold, operations, data-quality, and raw-file schemas as appropriate;
- read/write volume privileges for landing, Auto Loader schema, and checkpoint
  volumes;
- external-location and underlying cloud-storage privileges when external
  locations replace managed volumes;
- `READ` on the optional Census secret scope;
- serverless compute/environment access;
- outbound HTTPS to `api.census.gov`, `www2.census.gov`, and the configured
  Open-Meteo archive host.

An identity using `preflight_mode: create` additionally needs the metastore,
catalog, and schema privileges required to create the configured catalog,
schemas, and managed volumes. The checked-in production target intentionally
uses `validate` and leaves infrastructure provisioning to an administrator.

Do not place secret values in bundle YAML or source control.

## Configure a target

Review `databricks.yml`, `targets/dev.yml`, and `targets/prod.yml`. Supply
environment values through target YAML outside source control, CLI `--var`, or
`BUNDLE_VAR_*`.

Required:

```text
catalog
```

Production also requires:

```text
production_service_principal_name
```

Optional Census authentication requires both:

```text
census_secret_scope
census_api_key_secret
```

An empty pair uses the public unauthenticated Census API. The preflight rejects
a half-configured pair.

## Validate before deployment

```bash
uv lock --check
uv export --locked --no-dev --no-emit-project --no-hashes --format requirements-txt | diff - requirements-prod.txt
uv sync --locked --all-groups
uv run ruff check .
uv run mypy src
uv run python -m compileall -q src tests
uv run pytest --cov=crimenet --cov-report=term-missing --cov-report=xml --cov-fail-under=80
uv build
git diff --check
databricks bundle validate -t dev
```

Bundle validation needs a configured CLI profile or environment credentials.
Without credentials, run the preceding commands and the static bundle tests,
then record authenticated validation as not executed—do not report it as
passed.

Deploy:

```bash
./scripts/deploy.sh dev
./scripts/deploy.sh prod
```

`deploy.sh` reruns all local validation before `bundle deploy`.

## Preflight

The first bundle task validates the catalog, schemas, and three managed
volumes:

- raw landing;
- Auto Loader schemas;
- checkpoints.

The `preflight_mode` target variable controls whether it first creates missing
managed resources. The checked-in development target uses `create`; the
production target uses `validate`, so an administrator must provision the
production catalog, schemas, volumes, and grants before the complete job runs.
The legacy `crimenet_preflight --validate-only` flag remains available for
manual validation and is equivalent to `--preflight-mode validate`.

TIGER/Line row-count and quarantine gates and the maximum ambiguous spatial
match count are also explicit target variables. Review them when changing the
configured state or source scope; the production defaults remain blocking.

Secret scopes, workspace service principals, job entitlement, network policy,
external locations, and cloud IAM are workspace/account resources and are not
created by this bundle.

## First deployment and migration

1. Back up or clone any existing final tables.
2. Provision the configured catalog/schemas/volumes and grants.
3. Upload immutable source files under each municipal landing directory.
4. Supply the service principal and optional secret variables.
5. Validate and deploy the bundle.
6. Run preflight alone if workspace permissions are uncertain.
7. Run the complete job.
8. Review `data_quality.quality_results`, all quarantine tables, and
   `ops.weather_request_failures`.

This hardening changes identities, grains, and table schemas. Existing
installations should perform a controlled one-time rebuild:

- rebuild municipal Bronze if its prior row identity included file metadata;
- full-rebuild canonical Silver to populate `business_identity`, contract, and
  transformation versions;
- full-rebuild lighting to the three-column versioned UTC-hour key;
- full-rebuild ACS and TIGER/Line dependencies;
- run location mapping with `--full-rebuild` when upgrading a legacy mapping
  table;
- rebuild Gold to obtain path-independent `crime_offense_id` values;
- archive or migrate legacy quarantine tables that lack companion observation
  tables.

Downstream consumers must be notified that legacy Gold identifiers may change
once during this migration. After migration, their stability is the enforced
contract.

## Normal operation

Monitor:

- bundle task state and retry count;
- run ID and task logs;
- input/output/duplicate/quarantine counts;
- quality failures and coverage metrics;
- weather cache hits/downloads and failure events;
- ACS/TIGER download freshness;
- staging table names and promotion status;
- definition versions in current Silver and Gold tables.

The bundle permits one concurrent full run. Do not run an old wheel and a new
wheel concurrently while changing a definition version.

## Recovery scenarios

### Municipal file replay or checkpoint loss

Rerun Bronze. The content hash and insert-only merge are the correctness
boundary. Replacing or deleting an Auto Loader checkpoint may increase work but
does not require deleting Bronze rows. Never erase a final table to force
rediscovery.

### Interrupted weather or ACS download

Rerun the task. Temporary files are not accepted as cache hits. Invalid final
cache entries are audited and fetched again; valid entries are reused.

### API 429, timeout, or 5xx

Inspect the request failure table and logs. The HTTP adapter honors bounded
retry/backoff and `Retry-After`. If exhausted, wait for the provider or lower
concurrency/increase request spacing, then rerun. Do not place API keys in
logs or task parameters.

### Contract drift

Stop the pipeline. Preserve the source file. Compare its header/payload with the
versioned contract, add a new explicit version and transform, add regression
fixtures, and deploy through normal change control. Do not turn inference on as
an emergency workaround.

### Quarantine-rate failure

Query the current run's observation table grouped by reason code. Correct the
source/contract or explicitly approve a threshold change with evidence. A same
run retry does not duplicate observations.

### Staged validation failure

The previous final table remains in place. Query quality results and inspect the
run-scoped stage if the failure occurred before cleanup; jobs normally drop it
in `finally`. Correct the cause and rerun. Do not manually promote an
unvalidated stage.

### Failure after an upstream promotion

There is no cross-table rollback. Identify the first failed task and rerun the
job. Upstream merge/rebuild tasks and caches are replay-safe. If a rollback is
required for business reasons, use Delta time travel or an administrator-owned
clone procedure for each affected table and record the versions restored.

### Spatial ambiguity or unmatched spike

Check the selected ACS/TIGER vintage, boundary version, geometry validity,
coordinate order/ranges, and mapping version. Production blocks ambiguous
matches. A large unmatched rate may indicate bad coordinates or an unsupported
geographic area rather than a boundary download failure.

## Staging cleanup

Run-scoped stages are named:

```text
<final_table>__staging__<normalized_pipeline_run_id>
```

Jobs drop their own stage after success or failure. Before manually dropping an
abandoned stage:

1. resolve the exact three-part name;
2. verify no active run owns it;
3. confirm the final table is intact;
4. record the cleanup action.

Do not use broad wildcards or recursive storage deletion.

## Definition upgrades

When behavior changes, bump the relevant contract/definition/mapping version
and add a test proving stale rows are eligible for recomputation. Deploy
serially. Weather and ACS detect a stale persisted definition and stage a full
Bronze rebuild; verify that promotion and coverage evidence before removing old
versioned lighting rows or archived tables.

## Credential-gated validation

Ordinary CI cannot prove workspace permissions, networking, Unity Catalog
behavior, or service-principal execution. Configure repository secrets
`DATABRICKS_HOST` and `DATABRICKS_TOKEN` (or an approved OIDC equivalent) to
enable the authenticated `databricks bundle validate -t dev` step. Production
deployment should use the organization's approved noninteractive
authentication mechanism.
