# Integration temporal-coverage audit

The integration sampler must not estimate source exposure from observed crime
timestamps. In particular, these local audit artifacts are diagnostics only and
are not accepted as coverage inputs:

- `scripts/observation_ranges_candidates.json`
- `scripts/observation_yearly_activity.csv`
- `scripts/observation_suspicious_gaps.csv`
- `scripts/integration_window_diagnostics.csv`

Those files are derived from outcomes. They are useful for detecting a mismatch
between a declared acquisition contract and the data, but cannot create or
extend the acquisition contract.

## Upstream findings

The source acquisition metadata currently establishes several non-calendar
contracts:

- [Montgomery County's publisher](https://data.montgomerycountymd.gov/api/views/icn6-v9z3)
  says its founded-crime feed begins on July 1, 2016.
- [Dallas's publisher](https://www.dallasopendata.com/Public-Safety/test/8tp2-hns7)
  says its current RMS incidents begin on June 1, 2014.
- [Denver's publisher](https://www.arcgis.com/sharing/rest/content/items/16d9c82bb36c4475bf87189cfaed653c?f=json)
  describes a rolling feed containing the previous five calendar years plus
  the current year to date. Coverage therefore has to be frozen with the
  acquired snapshot, not treated as a permanent all-history feed.
- The local LASD landing manifest inventories separate official annual
  resources, including each training year from 2014 through 2023.
- Atlanta's publisher describes the mirror as APD incidents since 2009; Marin's
  publisher states January 1, 2013; Seattle's publisher labels its feed
  2008-present; San Francisco labels its current incident feed 2018-present.

The repository does not currently retain equally explicit, immutable
acquisition-coverage metadata for every Silver-enabled source. Dataset creation
dates and crime timestamp minima/maxima are not substitutes. Consequently, the
job now fails closed when a selected source lacks a catalog row; it does not
ship guessed intervals or fall back to 2014-2023.

## Required catalog contract

`CRIMENET_TEMPORAL_COVERAGE_URI` must identify a CSV tied to the acquired source
snapshot. It has one or more rows per selected source and these columns:

```text
source_city,source_timezone,coverage_start_utc,coverage_end_utc,coverage_basis,coverage_reference
```

- Intervals are half-open: `[coverage_start_utc, coverage_end_utc)`.
- Timestamps must be ISO-8601 values with offsets. They are normalized to UTC.
- `source_timezone` is the IANA timezone used for local training-year clipping.
- Multiple non-overlapping rows represent gaps or source-era changes.
- `coverage_basis` names the outcome-independent evidence type.
- `coverage_reference` identifies the publisher document, annual-resource
  manifest, or immutable acquisition manifest that supports the row.

The effective clipped intervals and their provenance are copied into the
published integration manifest, so later catalog changes cannot obscure the
support used for a completed run.
