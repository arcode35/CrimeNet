# crimenet_data

CrimeNet's Dagster code owns source-level crime ingestion and normalization:

```text
source landing artifacts -> source-preserving bronze -> canonical silver
```

Reusable national OSM/ACS/TIGER/H3 features, national weather, lighting
derivation, and model-table joins are built outside this Dagster graph. The
standalone national builders under `src/crimenet_data/scripts/` are not Dagster
assets.

## Crime lake

The durable lake is Backblaze B2 through its S3-compatible API, rooted at
`s3://crimenet-data`. Crime paths retain their established contracts:

```text
raw_files/landing/<source_key>/...
bronze/crime/<source_key>/...
silver/crime/<source_key>/...
gold/crime/<source_key>/...
```

Set `B2_ENDPOINT_URL`, `B2_KEY_ID`, `B2_APPLICATION_KEY`, and optionally
`B2_REGION` (default `us-east-005`). Credentials are resolved only by
`CrimeLakeResources` and are not part of Dagster metadata.

All 19 active sources begin below `raw_files/landing/<source_key>/`. Each source
has one module owning its immutable configuration, Bronze preparation,
occurrence-year derivation, and Silver adapter. A single registry drives generic
CSV, Parquet, and GeoJSON ingestion; it does not classify sources by age or
acquisition priority. Chandler uses lossless Windows-1252 decoding, Denver
FeatureCollections retain flattened properties and geometry, and Montgomery
uses deterministic malformed-row recovery.

Bronze retains source-native fields—including event-end fields—while adding one
standard provenance representation and the technical `occurrence_year`
partition. Canonical taxonomy mapping remains in Silver. The compatibility
column remains named `source_city`, even when the source represents an agency or
county jurisdiction.

Delta tables retain ZSTD compression and their existing year partitioning.
Backblaze requires delta-rs unsafe rename support, which is centralized in the
resource under the operating rule that only one writer may commit to a given
Delta table at a time. Each writing asset has a table-specific Dagster pool so
deployments can enforce a limit of one while allowing different source tables to
materialize in parallel.

## Getting started

### Installing dependencies

**Option 1: uv**

Ensure [`uv`](https://docs.astral.sh/uv/) is installed following their [official documentation](https://docs.astral.sh/uv/getting-started/installation/).

Create a virtual environment, and install the required dependencies using _sync_:

```bash
uv sync
```

Then, activate the virtual environment:

| OS | Command |
| --- | --- |
| MacOS | ```source .venv/bin/activate``` |
| Windows | ```.venv\Scripts\activate``` |

**Option 2: pip**

Install the python dependencies with [pip](https://pypi.org/project/pip/):

```bash
python3 -m venv .venv
```

Then activate the virtual environment:

| OS | Command |
| --- | --- |
| MacOS | ```source .venv/bin/activate``` |
| Windows | ```.venv\Scripts\activate``` |

Install the required dependencies:

```bash
pip install -e ".[dev]"
```

### Running Dagster

Start the Dagster UI web server:

```bash
dg dev
```

Open http://localhost:3000 in your browser to see the project.

## Learn more

To learn more about this template and Dagster in general:

- [Dagster Documentation](https://docs.dagster.io/)
- [Dagster University](https://courses.dagster.io/)
- [Dagster Slack Community](https://dagster.io/slack)
