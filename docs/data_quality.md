# Data quality

## Quality-result contract

Checks are persisted to
`<catalog>.<data_quality_schema>.quality_results` with:

- pipeline run ID;
- check name;
- severity;
- passed/failed status;
- observed value;
- expected threshold;
- table name;
- check timestamp;
- optional source system.

The merge key is run, table, check, and source. Retrying one task updates that
run's evidence instead of appending uncontrolled duplicates. A failed blocking
check raises an exception after evidence is persisted where the platform
allows it.

## Silver crime checks

Silver's staged candidate is checked before the final table is replaced. The
separate `crime_quality_checks` task repeats the committed-table checks and
provides an explicit orchestration gate for downstream enrichment.

Blocking checks cover:

- exact candidate/input `business_identity` set equality;
- canonical schema and expected H3 field;
- configured minimum and maximum row count;
- duplicate business identities;
- missing incident, offense, business, or source-row identifiers;
- null or implausible occurrence timestamps;
- incomplete, NaN, or out-of-range coordinates;
- unsupported sources;
- per-column critical null rates;
- coordinate null rate;
- at least one valid row for every supported city;
- configured Silver-to-distinct-Bronze ratio;
- configured quarantine rate.

Quarantine reasons are persisted as informational per-reason checks.

## Gold checks

Gold validation executes against the staging table and persists pass/fail
evidence before promotion. It checks:

- exact source/Gold offense key equality in both directions;
- unique `crime_offense_id` values;
- unchanged cardinality;
- required schema, data types, and non-null fields;
- coordinate and timestamp domains;
- uniqueness of lookup keys;
- one-to-one join cardinality and absence of unexplained expansion;
- spatial ambiguity;
- weather, lighting, ACS, and tract coverage;
- source-level counts and coverage;
- weather temperature, solar elevation, rate, and other feature ranges;
- the requested current lighting definition version.

Weather, lighting, and ACS coverage count only rows whose join marker is true
and whose required feature values are non-null and within their declared
domains. Lookup-row presence with unusable values cannot satisfy a threshold.

Coverage thresholds are bundle variables. Development defaults are deliberately
permissive for small fixtures. Production currently configures:

| Threshold | Production value |
|---|---:|
| minimum Silver/Bronze ratio | 0.50 |
| maximum Silver/Bronze ratio | 1.01 |
| maximum coordinate null rate | 0.50 |
| maximum quarantine rate | 0.05 |
| minimum weather coverage | 0.90 |
| minimum lighting coverage | 0.95 |
| minimum ACS coverage | 0.80 |
| minimum tract coverage | 0.90 |
| maximum invalid location rate | 0.50 |
| maximum unmatched location rate | 0.10 |
| minimum ACS tracts per vintage | 5000 |
| minimum TIGER/Line tracts per vintage | 5000 |
| maximum boundary quarantine records | 0 |
| maximum ambiguous spatial matches | 0 |

These are explicit defaults, not universal recommendations. Operators should
calibrate them from source history and change control.

## Quarantine model

Externally sourced invalid rows are split before business transformation.
Crime, weather, and ACS use shared quarantine helpers; boundaries and spatial
mapping use the same stable-entity/per-run-observation principle.

Entity identity is derived from stable source identity and reason code. The
entity table records first-seen evidence. A companion observations table uses
the reject identity plus pipeline run ID. Consequently:

- a retry in the same run produces no second observation;
- the same bad record in a later run is observable again;
- current-run quarantine rates remain correct;
- the raw or minimally transformed payload and validation fields remain
  available for diagnosis.

Typical crime reasons include corrupt source, missing or unsupported source,
missing identifiers, missing/implausible timestamps, and invalid/incomplete
coordinates. Weather adds request, model, hourly-array, timestamp, and query
coordinate reasons. ACS adds rescued schema, tract key/GEOID, vintage, and
numeric-domain reasons.

## Failure semantics

A validation error stops dependent bundle tasks. For rebuilds, the final table
is unchanged because promotion has not occurred. Quarantine and quality tables
may already contain evidence; that is intentional and is independently
idempotent.

No cross-table rollback is attempted. If a later task fails after an upstream
promotion, rerun the failed run. Stable keys, merge sinks, versioned derived
tables, and persistent caches are designed to converge.

## Adding or changing a check

1. Add a pure check or candidate validator.
2. Decide whether it is blocking or informational.
3. Add an environment-supplied threshold when the acceptable value differs by
   deployment.
4. Persist the result using the shared quality-result schema.
5. Test pass, boundary, and failure cases—including preservation of the final
   table for pre-promotion checks.
6. Update this document and the README claims table.

Do not lower a threshold or remove an invariant solely to make a fixture pass.
