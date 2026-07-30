# Data Quality

CrimeNet exposes reusable validators through `crimenet.quality`. Each validator
returns a `QualityReport` containing named `QualityCheckResult` records and
raises `QualityValidationError` when a blocking check fails. Failure reports
include counts and a bounded set of representative rows; validators never
collect an unbounded sample.

## Runtime validators

- Silver crime: canonical schema, non-null and unique `crime_offense_id`,
  recognized source city, source-row hash, coordinate domains, and optional
  occurrence-timestamp coverage.
- Weather: non-null and unique hourly keys, hour alignment, provider/model/H3
  domains, and defensible temperature bounds.
- Socioeconomic: non-null and unique tract/vintage keys, 11-digit GEOIDs,
  vintage range, and derived rates in `[0, 1]`.
- Lighting: versioned key uniqueness, active definition, coordinate and solar
  domains, pvlib lineage, and classification/daylight consistency.
- Gold: source cardinality, unique crime identity, match metrics, ACS
  release-date leakage, enrichment lineage, and configurable match coverage.

Coverage thresholds are explicit inputs. A zero threshold means coverage is
measured and reported but is not blocking; production callers should pass
their approved thresholds rather than embedding them in transformations.

## Test and validation commands

Ordinary tests do not require Databricks credentials:

```bash
uv run pytest tests/unit tests/integration -m "not databricks"
```

Databricks-only tests are separately selected:

```bash
uv run pytest tests/databricks -m databricks
```

Bundle validation is not credential-free. The CLI resolves the workspace
identity during `databricks bundle validate`, including when a target uses a
literal shared root path. Run it with a configured Databricks profile:

```bash
databricks bundle validate --target dev
```

GitHub Actions always runs lint plus the ordinary local suite. It additionally
runs bundle validation when the repository provides `DATABRICKS_HOST` and
`DATABRICKS_TOKEN`; otherwise the workflow emits an explicit explanation.
