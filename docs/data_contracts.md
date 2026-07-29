# Data contracts

## Contract policy

Production readers do not infer municipal schemas. Every supported source has a
versioned `StructType` in `src/crimenet/contracts/bronze.py`. Records carry
`source_contract_version`; canonical records also carry
`transformation_version`.

Contract changes are deliberate:

- an additive, renamed, removed, reordered, or type-changed municipal CSV
  header is a blocking contract event, not silently inferred schema;
- malformed rows that Spark can parse permissively retain a corrupt-record
  payload and are quarantined downstream;
- Auto Loader weather and ACS readers use explicit schemas plus
  `_rescued_data`; rescued records are quarantined;
- a supported upstream change requires a new contract version, an updated
  transform, fixtures, and migration notes.

## Bronze metadata

Every Bronze record adds:

| Field | Meaning |
|---|---|
| `source_system` | stable source name |
| `source_file` | physical lineage only |
| `source_row_hash` | SHA-256 of normalized logical source fields |
| `source_contract_version` | parser/source contract used |
| `ingested_at` | operational first-processing timestamp |
| `corrupt_record` or `rescued_data` | source parser evidence |

The row hash excludes `_source_file`, `source_file`, `source_row_hash`,
contract/system fields, and retrieval/ingestion timestamps. Parser corrupt or
rescue payloads remain part of the logical row, so different malformed inputs
remain distinguishable. A copied or renamed file has the same Bronze identity.

## Municipal source contracts

### Dallas

Format: headered CSV, multiline and quoted fields supported.

Required normalized fields:

- `service_number_id`
- `incident_number_w_year`
- `date1_of_occurrence`
- `time1_of_occurrence`

The canonical incident ID is `incident_number_w_year`; the offense ID is
`service_number_id`. Local timestamps are interpreted in
`America/Chicago` and converted to UTC. Location coordinates are parsed from
the parenthesized `Location1` value.

### Houston

Format: headered CSV.

Required normalized fields:

- `incident`
- `nibrsclass`
- `rmsoccurrencedate`
- `rmsoccurrencehour`

The normalized incident ID is supplied by the source. The offense ID is a
deterministic hash of normalized incident ID plus NIBRS class, the published
row grain represented by `OffenseCount`. Mutable occurrence and location
fields and the file path are deliberately excluded so a source correction
retains the same logical identity. A blank incident or NIBRS class produces no
offense ID and is quarantined. Local timestamps are interpreted in
`America/Chicago` and converted to UTC.

### Fort Worth

Format: JSON Lines.

Required normalized fields:

- `case_no_offense`
- `case_no`
- `from_date`

The incident ID is `case_no`; the offense ID prefers `case_no_offense` and
falls back to `objectid`. Epoch-millisecond timestamps are treated as UTC.
Numeric values use ANSI-safe casts so malformed values become validation
failures instead of aborting before quarantine.

## Canonical Silver crime contract

`src/crimenet/contracts/silver.py` is the ordered schema contract. Its business
key is `business_identity`. Required quality fields are:

- `source_system`
- `source_incident_id`
- `source_offense_id`
- `business_identity`
- `source_row_hash`
- `source_contract_version`
- `transformation_version`
- `occurred_at`

The production quality stage permits only `dallas`, `houston`, and
`fort_worth`. Coordinates may be both null when the environment's configured
null-rate threshold allows it; exactly one null coordinate or an out-of-range
value is invalid.

## Weather contract

Open-Meteo cache files include the request identity and response metadata plus
hourly arrays. The explicit Bronze schema accepts requested timestamps and
`temperature_2m`. Validation requires:

- deterministic request ID, H3 cell, provider, and supported model;
- finite query coordinates;
- non-empty hourly timestamps;
- equal timestamp and value-array lengths;
- parseable hourly timestamps;
- UTC response timezone.

Silver weather grain is provider, model, weather H3 query cell, and UTC
`weather_timestamp`. It carries `weather_definition_version`.

## Lighting contract

The physical key is:

```text
lighting_query_cell_id
+ solar_timestamp_hour
+ lighting_definition_version
```

`solar_timestamp_hour` is a UTC hour. Conditions are restricted to daylight,
civil twilight, nautical twilight, astronomical twilight, or night. Solar
elevation and feature ranges are validated before promotion.

## ACS contract

Landed ACS 5-year records contain Census geography fields, selected estimate
and margin-of-error variables, `acs_vintage`, coverage-period years, dataset,
and retrieval metadata. Cache validation requires unique 11-digit GEOIDs in the
requested state and a target-configured minimum record count; production uses
5,000 tracts per vintage. Auto Loader uses explicit string/numeric fields.

Sentinel values are converted to null. Malformed non-sentinel numerics, invalid
11-digit GEOIDs, missing tract/vintage keys, invalid vintages, and rescued
schema data are quarantined. Silver grain is `geoid + acs_vintage`, and rows
carry `socioeconomic_definition_version`.

Rates use guarded denominators:

- poverty;
- unemployment;
- housing vacancy;
- renter occupancy;
- households without a vehicle.

## Boundary and mapping contracts

TIGER/Line tract rows must have a valid 11-digit GEOID, requested state and
vintage, non-empty WGS84 geometry, and a unique boundary key. Download archive
hashes and boundary-definition versions are retained.

The location mapping records its mapping and boundary versions, spatial status,
match count, GEOID when uniquely matched, and source archive checksum.
Statuses distinguish `matched_contains`, `matched_covers`, `unmatched`, and
`ambiguous`.

## Gold contract

Gold has one row per stable `crime_offense_id`, derived from Silver
`business_identity`. It carries the source offense fields, weather and
lighting keys, tract mapping evidence, ACS vintage/features, definition
versions, and enrichment coverage evidence used during validation.

Gold accepts only unique lookup keys and current requested definition versions.
The staged candidate must have exactly the same source offense key set as
Silver—row-count equality alone is not accepted.
