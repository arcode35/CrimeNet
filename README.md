# CrimeNet

[![CI](https://github.com/arcode35/crimenet/actions/workflows/ci.yml/badge.svg)](https://github.com/arcode35/crimenet/actions/workflows/ci.yml)

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Dagster](https://img.shields.io/badge/Orchestration-Dagster-4B32C3)
![Polars](https://img.shields.io/badge/Compute-Polars-CD792C)
![Delta Lake](https://img.shields.io/badge/Storage-Delta%20Lake-00ADD8)
![PyTorch](https://img.shields.io/badge/ML-PyTorch-EE4C2C)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-006600)
![H3](https://img.shields.io/badge/Geospatial-H3-1E88E5)
![FastAPI](https://img.shields.io/badge/Serving-FastAPI-009688)
![Next.js](https://img.shields.io/badge/Product-Next.js-black)
![MapLibre](https://img.shields.io/badge/Maps-MapLibre-396CB2)

> **Live product:** https://crimesense.ai

**CrimeSense** is the interactive forecasting product.

**CrimeNet** is the data, feature, machine-learning, and inference system that powers it.

CrimeNet turns more than a decade of heterogeneous public-safety data into a rolling **24-hour geospatial risk surface**. The system standardizes incompatible municipal crime records, joins them against national socioeconomic and OpenStreetMap feature stores, reconstructs environmental state for future timestamps, runs production inference, aggregates predictions across multiple H3 resolutions, and serves the resulting surface through CrimeSense.

The project is deliberately **data-first and modeling-second**: the central engineering problem is building a temporally correct, spatially consistent feature and inference system that can support machine learning reliably from historical training through live serving.

---

## At a Glance

| Metric                                                   |                               Current scale |
| -------------------------------------------------------- | ------------------------------------------: |
| Production crime sources                                 |                                      **15** |
| Historical coverage                                      |                                **12 years** |
| Audited crime events                                     |                                   **16.7M** |
| Modeled crime subtypes                                   |                                      **87** |
| Final model examples                                     |                                   **180M+** |
| National H3 resolution-9 feature cells                   |                                  **25.56M** |
| U.S. state/DC partitions                                 |                                      **51** |
| Census socioeconomic match rate                          |                                  **99.83%** |
| Production features per model                            |                                      **38** |
| Forecast horizon                                         |                                **24 hours** |
| Forecast execution                                       | **24 independently inferred hourly states** |
| Train horizon                                            |                               **2014–2023** |
| Validation horizon                                       |                                    **2024** |
| Test horizon                                             | **2025 onward, bounded by source coverage** |
| Final intensity validation events                        |                                   **1.54M** |
| Final intensity validation rows                          |                                  **15.99M** |
| Validation NLL reduction vs. constant-intensity baseline |                                   **12.8%** |
| Validation information gain                              |                         **3.02 bits/event** |
| Test set evaluated                                       |                                      **No** |
| Large feature-assembly workloads                         |           **billions of intermediate rows** |

### Current production geographies

* Atlanta
* Baltimore
* Chandler, AZ
* Chicago
* Dallas
* Denver
* Fort Worth
* Los Angeles County Sheriff
* Marin County Sheriff, CA
* Montgomery County, MD
* New York City
* San Francisco
* Seattle
* Sonoma County Sheriff, CA
* Washington, DC

Additional source adapters exist for other jurisdictions, but the current audited production forecasting footprint is the 15-source set above.

---

# System Architecture

```mermaid
flowchart TD
    subgraph Sources["Public + External Data"]
        Crime["Municipal / County Crime Data"]
        ACS["Census ACS 5-Year"]
        TIGER["Census TIGER/Line"]
        OSM["OpenStreetMap / Geofabrik"]
        Weather["Open-Meteo / Historical Weather"]
        Solar["Solar Geometry / pvlib"]
        Imagery["NAIP + Sentinel-2"]
    end

    subgraph Data["Canonical Data Platform"]
        Landing["Immutable Landing"]
        Bronze["Bronze\nSource-Aligned"]
        Silver["Silver\nCanonical Crime Model"]
    end

    subgraph Features["Feature Infrastructure"]
        Socio["National Socioeconomic Feature Store"]
        Built["National OSM Feature Store"]
        Environmental["Weather + Solar + Lighting"]
        History["Temporal Crime History"]
        EventSpine["Leakage-Safe Event Spine"]
        Integration["Point-Process Integration Samples"]
        FinalTable["Final Model Table\n180M+ examples"]
    end

    subgraph Models["Modeling"]
        Intensity["Intensity Model\nλ(x,t)"]
        Mark["Mark Model\nP(type | x,t)"]
        Omega["CrimeNet Ω\nResearch Point Process"]
    end

    subgraph Inference["Production Inference"]
        Future["Future-State Feature Reconstruction"]
        Hourly["24 Independent Forecast Hours"]
        Snapshot["Inference Snapshot Materialization"]
        LOD["Multi-Resolution H3 Aggregation"]
    end

    subgraph Product["CrimeSense"]
        API["FastAPI Serving Layer"]
        UI["Next.js + MapLibre\nInteractive Risk Explorer"]
    end

    Crime --> Landing --> Bronze --> Silver
    ACS --> Socio
    TIGER --> Socio
    TIGER --> Built
    OSM --> Built
    Weather --> Environmental
    Solar --> Environmental

    Silver --> EventSpine
    Silver --> History

    Socio --> FinalTable
    Built --> FinalTable
    Environmental --> FinalTable
    History --> FinalTable
    EventSpine --> FinalTable
    Integration --> FinalTable

    FinalTable --> Intensity
    FinalTable --> Mark
    FinalTable --> Omega

    Socio --> Future
    Built --> Future
    Weather --> Future
    Solar --> Future
    History --> Future

    Intensity --> Hourly
    Mark --> Hourly
    Future --> Hourly

    Hourly --> Snapshot --> LOD --> API --> UI
```

The same canonical feature definitions are used from historical training through forecast-time inference.

**Training-serving consistency is enforced by the data system rather than recreated ad hoc inside the API.**

---

# CrimeSense

[**CrimeSense**](https://crimesense.ai) is the interactive product layer built on CrimeNet.

The current interface exposes the output of the end-to-end forecasting pipeline rather than querying raw model objects directly.

It supports:

* navigation across the multi-city geospatial risk surface
* movement through the next **24 forecast hours**
* multiple spatial resolutions for overview and local inspection
* per-cell predicted crime intensity
* conditional distributions across **87 crime subtypes**
* production inference snapshots generated from the same feature contract used in training

The UI is intentionally the final layer of the architecture. The larger project is the infrastructure required to generate the surface correctly and reproducibly.

---

# Data Platform

## Immutable landing

Raw municipal files and external source artifacts are preserved before transformation to support:

* deterministic replay
* source-level auditing
* historical backfills
* schema debugging
* recovery from failed transformations
* reprocessing without repeatedly downloading upstream data

Large datasets and generated model artifacts live in object storage rather than Git.

## Bronze: source-aligned ingestion

Each source enters through an explicit adapter. Source-specific parsing is isolated at the ingestion boundary so incompatible municipal schemas do not leak into downstream modeling code.

The ingestion layer handles:

* CSV
* Parquet
* GeoJSON
* multi-file exports
* malformed or ragged CSVs
* historical schema changes
* source-specific timestamp and identifier conventions

## Silver: canonical crime model

Silver standardizes heterogeneous source records into one common offense representation.

The current audited production footprint contains:

* **16.7M audited crime events**
* **15 production sources**
* **87 canonical crime subtypes**

Normalization handles differences in:

* identifiers
* timestamps
* coordinates
* addresses
* offense descriptions and codes
* null encodings
* duplicate behavior
* historical source-schema changes

The result is a single crime-event contract consumed by downstream spatial indexing, feature generation, training, evaluation, and inference.

---

# National Feature Infrastructure

CrimeNet uses national-scale contextual feature stores so each modeled location is represented through the same feature contract regardless of city.

## National socioeconomic feature store

The socioeconomic store is derived from U.S. Census / ACS data and materialized at H3 resolution.

Current national coverage:

* **25,563,443 unique H3 resolution-9 cells**
* **all 50 states plus Washington, DC**
* **99.83% Census socioeconomic match coverage**

Features include:

* population
* median age
* median household income
* poverty rate
* unemployment rate
* vacancy rate
* renter occupancy
* household vehicle availability

Historical observations use Census vintages that were actually available at the prediction timestamp.

## National OpenStreetMap feature store

OpenStreetMap / Geofabrik data is transformed into reusable H3-aggregated representations of the built environment.

Features include:

* total road-length density
* major-road density
* residential-road density
* service-road density
* intersection density
* dead-end density
* building density
* POI density
* nightlife density
* food density
* retail density
* transit density
* road-class ratios
* one-way-road ratio
* POI-category entropy
* land-use entropy
* commercial/residential mix

OSM snapshots are version-aware so historical examples can be joined against spatial context consistent with information availability.

## Census geography

TIGER/Line boundaries connect locations and H3 cells to historical Census geography.

Boundary vintage and release timing are part of the feature contract rather than silently applying current geography to historical rows.

---

# Point-in-Time Correctness

Leakage prevention is enforced in the data platform, not left to individual model-training scripts.

CrimeNet tracks feature metadata such as:

```text
feature_available_at
feature_version_id
osm_snapshot_date
osm_available_at
acs_vintage
acs_release_date
tiger_line_year
tiger_release_date
```

A feature is eligible only if it was available at or before the relevant prediction timestamp.

Conceptually, every historical join has to answer:

> **What could the system legitimately have known about this place at this time?**

The same rule is applied to:

* Census features
* OSM features
* environmental features
* temporal crime-history features
* train/validation/test assignment
* point-process integration support
* model evaluation

---

# Environmental and Temporal Features

## Weather

Historical weather features are derived from Open-Meteo-compatible sources and spatially deduplicated through H3.

Features include atmospheric context such as temperature and humidity. Missing weather coverage is represented as nullable feature state rather than silently dropping otherwise valid crime rows.

## Solar geometry and lighting

CrimeNet uses `pvlib`-derived solar calculations to represent the physical lighting state of a location and timestamp.

Features include:

* solar elevation
* solar azimuth
* daylight state
* lighting condition

This gives the model a physically grounded representation of lighting rather than relying on clock time alone.

## Temporal crime history

The model table also supports local historical crime context over multiple lookback windows.

History features are constructed using only events observable before the prediction timestamp.

---

# Event Spine, Integration Sampling, and Final Model Table

CrimeNet represents crime as a spatiotemporal event-intensity problem rather than only as ordinary row-wise classification.

## Event spine

Observed offenses are converted into a leakage-safe event spine containing:

* canonical spatial keys
* timestamps
* source identity
* crime labels
* split assignment
* point-in-time feature eligibility

Historical events remain part of the event spine even if optional enrichment fields are unavailable.

## Monte Carlo integration samples

Point-process likelihoods require exposure over non-event space-time.

CrimeNet therefore constructs source-aware Monte Carlo integration samples over authoritative reporting domains using:

* **H3 cell × continuous time** as the integration measure
* source-specific temporal coverage
* outcome-independent sampling support
* deterministic source-level random seeds
* integration weights
* strict source-domain contracts

Sampling never extends a source outside its declared temporal support.

## Temporal splits

The split contract is centralized:

```text
Train       2014–2023
Validation  2024
Test        2025 onward within declared source coverage
```

**All performance results published in this README are validation results from the 2024 temporal validation horizon.**

**The 2025+ test split has not been evaluated.** It remained sealed throughout model development and is intentionally not used for any metric reported below.

This distinction is important: the published numbers should not be interpreted as final test performance.

## Final model table

Observed event rows and integration rows are combined with:

* national socioeconomic features
* national OSM features
* weather
* solar and lighting state
* calendar context
* temporal crime-history features
* event/integration weights
* canonical crime labels
* source-aware split assignment

The current full-scale pipeline produces **180M+ leakage-safe spatiotemporal examples**.

Publication checks include:

* structural null validation
* split correctness
* future-feature leakage
* integration-weight validity
* feature coverage
* weather coverage
* source/split/row-type counts
* immutable snapshot identity

---

# Machine Learning

CrimeNet separates forecasting into two production modeling problems.

## 1. Intensity model — λ(x,t)

The intensity model estimates expected event activity as a function of location and time.

The production modeling workflow includes:

* point-process / Poisson-style intensity modeling
* GPU-accelerated XGBoost training
* Optuna hyperparameter optimization
* chronological temporal validation
* calibration analysis
* feature-importance analysis
* reproducible model artifacts and training metadata

The current published validation results below correspond to:

```text
Model:   xgb_intensity_mvp_slow_deep_d20
Run ID:  ff94186b570641e68eb8c9d5cf64c567
Split:   2024 temporal validation
Test:    NOT EVALUATED
```

## 2. Mark model — P(type | x,t)

The mark model estimates the conditional probability distribution over **87 crime subtypes** for a location and timestamp.

Separating intensity from event type allows the system to model:

1. **how much activity is expected**, and
2. **what kind of activity is expected**

as distinct statistical problems.

---

# Validation Results

## Evaluation protocol

The results in this section are **validation results, not test results**.

The final intensity model was evaluated on the chronological **2024 validation horizon** after training on **2014–2023** data.

The **2025+ test set has not been evaluated**.

The intensity model is evaluated using point-process negative log-likelihood over both observed event rows and sampled non-event exposure.

For NLL:

**lower is better.**

The primary baseline reported here is a **constant-intensity point-process baseline** evaluated under the same validation framework.

The percentage improvement is:

```text
NLL reduction =
    (baseline NLL/event - model NLL/event)
    / baseline NLL/event
```

## Aggregate temporal validation

The final validation covered:

* **15,988,997 evaluation rows**
* **1,537,647 observed crime events**
* all **15 production sources**

| Metric               | Constant-intensity baseline | CrimeNet intensity model |
| -------------------- | --------------------------: | -----------------------: |
| NLL / observed event |                 **16.3492** |              **14.2559** |
| NLL reduction        |                           — |               **12.80%** |
| NLL gain / event     |                           — |          **2.0933 nats** |
| Information gain     |                           — |     **3.020 bits/event** |

Across the full 2024 temporal validation set, CrimeNet therefore achieved a:

> **12.8% reduction in point-process negative log-likelihood relative to the constant-intensity baseline.**

Equivalently, the model gained approximately:

> **3.02 bits of predictive information per observed event over the baseline.**

These values are validation metrics only. No claim about final test performance is made.

## City-level validation

Performance varies by jurisdiction.

| Geography                  |  Events | Baseline NLL/event | Model NLL/event | NLL reduction | Bits/event | Calibration error |
| -------------------------- | ------: | -----------------: | --------------: | ------------: | ---------: | ----------------: |
| Atlanta                    |  45,042 |            15.9880 |     **14.9028** |      **6.8%** |       1.57 |             -7.7% |
| Baltimore                  |  58,013 |            15.7208 |     **14.3754** |      **8.6%** |       1.94 |             +6.2% |
| Chandler, AZ               |  14,998 |            16.0710 |     **15.0726** |      **6.2%** |       1.44 |             -5.5% |
| Chicago                    | 256,804 |            15.6143 |     **13.9100** |     **10.9%** |       2.46 |            -10.8% |
| Dallas                     | 118,027 |            15.9285 |     **14.7194** |      **7.6%** |       1.74 |            -10.7% |
| Denver                     |  65,056 |            15.8204 |     **14.4403** |      **8.7%** |       1.99 |            -11.3% |
| Fort Worth                 |  59,776 |            16.3138 |     **15.2963** |      **6.2%** |       1.47 |             -2.3% |
| Los Angeles County Sheriff | 132,355 |            19.8356 |     **18.6599** |      **5.9%** |       1.70 |           +329.3% |
| Marin County Sheriff, CA   |   1,708 |            79.5323 |     **20.3180** |     **74.5%** |      85.43 |           +398.2% |
| Montgomery County, MD      |  33,103 |            17.8028 |     **15.4106** |     **13.4%** |       3.45 |            +19.4% |
| New York City              | 571,310 |            15.5971 |     **13.0790** |     **16.1%** |       3.63 |            -12.4% |
| San Francisco              |  77,419 |            15.9030 |     **13.2216** |     **16.9%** |       3.87 |             +7.0% |
| Seattle                    |  70,628 |            15.8456 |     **14.0876** |     **11.1%** |       2.54 |             +4.7% |
| Sonoma County Sheriff, CA  |   4,211 |            70.7251 |     **21.1529** |     **70.1%** |      71.52 |           +455.1% |
| Washington, DC             |  29,197 |            15.8414 |     **14.7856** |      **6.7%** |       1.52 |            +57.0% |

Among the larger municipal datasets, notable validation results include:

* **San Francisco:** 16.9% lower NLL
* **New York City:** 16.1% lower NLL across 571,310 observed validation events
* **Chicago:** 10.9% lower NLL across 256,804 events
* **Seattle:** 11.1% lower NLL with a +4.7% aggregate event-count calibration error
* **Montgomery County, MD:** 13.4% lower NLL

## Calibration

Likelihood improvement and absolute rate calibration are related but distinct properties.

Across the full validation set:

```text
Observed events:  1,537,647
Expected events:  1,910,478
Expected/observed: 1.2425
Calibration error: +24.25%
```

The model therefore improves substantially over the constant-intensity baseline in point-process likelihood while still showing meaningful aggregate overprediction.

Calibration also varies significantly by source.

In particular, the Los Angeles County Sheriff, Marin County Sheriff, Sonoma County Sheriff, and Washington, DC validation domains show large positive count-calibration errors.

The unusually large relative NLL reductions for Marin and Sonoma should therefore **not** be interpreted as evidence of exceptionally strong calibrated forecasts. Their constant-intensity baselines are unusually weak under the source-domain exposure configuration, while the fitted model still substantially overpredicts total event counts.

These jurisdictions are reported for completeness rather than used as headline results.

## Interpretation

The validation results support a narrower claim than "CrimeNet predicts crime 12.8% better."

The defensible statement is:

> **On the 2024 temporal validation set, CrimeNet reduced point-process negative log-likelihood by 12.8% relative to a constant-intensity baseline across 1.54M observed events.**

This measures improvement in the likelihood assigned to the observed spatiotemporal event process.

It is **not**:

* a 12.8% increase in classification accuracy
* a 12.8% reduction in crime-count error
* a test-set result
* a comparison against the best published crime-forecasting system

A stronger scientific comparison would require additional historical, spatial, temporal, and point-process baselines evaluated under the same domain and split contract.

## Test-set status

The test horizon is intentionally still sealed.

```text
Train:       2014–2023   USED
Validation:  2024        USED
Test:        2025+       NOT EVALUATED
```

No test-set metric is currently published.

The test split will only provide a meaningful final estimate if model architecture, features, preprocessing, calibration decisions, and evaluation procedures are frozen before it is opened.

---

# CrimeNet Ω

**CrimeNet Omega** is the research point-process model built on the same data and feature infrastructure.

The implemented research architecture supports:

* city embeddings
* offense-family / subtype embeddings
* lighting-state embeddings
* continuous numerical covariates
* event exposure
* marked-event likelihoods
* temporal intensity modeling

Research directions include:

* covariate-conditioned neural Hawkes processes
* hierarchical crime-taxonomy structure
* continuous-time spatial context
* graph-based neighborhood interaction
* explicit observation / reporting models

Omega is evaluated against production tree-based and historical baselines rather than assumed to be superior.

---

# Production Inference

The serving system does not simply send a static training row through a model.

For every forecast timestamp, CrimeNet reconstructs the feature state needed to answer:

> **What will this location look like at this future hour?**

The inference pipeline resolves:

* static socioeconomic context
* built-environment features
* calendar state
* future-hour weather context
* solar geometry
* lighting state
* recent crime-history features

It then runs **24 independently inferred forecast hours**.

Each hour is materialized into an inference snapshot and passed through spatial aggregation layers before being exposed through the API.

This produces a rolling H3 risk surface that can be queried efficiently at multiple map zoom levels.

---

# Multi-Resolution Geospatial Serving

Rendering a large H3 surface directly at full resolution is unnecessarily expensive.

CrimeNet therefore materializes multiple levels of spatial detail for serving.

The serving layer is responsible for:

* H3-based forecast storage
* viewport-aware retrieval
* aggregation across spatial resolutions
* efficient map payloads
* individual-cell inspection
* forecast-hour selection

The result is a map that can move between regional overview and local inspection without requiring the browser to load the entire high-resolution surface.

---

# Distributed Compute

CrimeNet separates durable data from disposable compute.

## CPU / memory-heavy workloads

Used for:

* source normalization
* large joins
* H3 feature construction
* national feature-store generation
* historical backfills
* Parquet materialization
* spatial preprocessing
* final-model-table assembly

## GPU workloads

Used for:

* XGBoost training and HPO
* neural point-process training
* large inference workloads
* aerial / satellite representation learning

Large experiments have run on cloud machines with **up to 8× RTX 5090-class GPUs** and hundreds of gigabytes of host memory.

Training and inference workers are disposable. Dataset snapshots, manifests, feature versions, and model artifacts are durable.

---

# Imagery Pipeline

CrimeNet also contains an imagery feature pipeline using:

* **NAIP** high-resolution aerial imagery
* **Sentinel-2 L2A** multispectral satellite imagery

The pipeline supports:

* scene discovery
* spatial deduplication
* retrieval
* preprocessing
* embedding generation
* versioned feature storage

Image-derived representations are currently an experimental enrichment path rather than a required dependency of the production CrimeSense forecast.

---

# Orchestration

The data platform is orchestrated with **Dagster**.

Assets cover:

* source landing
* Bronze ingestion
* Silver normalization
* event-spine construction
* national socioeconomic features
* national OSM features
* environmental features
* temporal-history features
* integration sampling
* final-model-table construction
* model training
* evaluation

This provides:

* dependency-aware execution
* structured logging
* materialization metadata
* failure isolation
* backfills
* reproducible re-runs

Production serving is kept operationally separate from offline training orchestration so forecast generation and the web product are not coupled to a Dagster development process.

---

# Data Quality and Reproducibility

CrimeNet fails closed when core contracts are violated.

Checks include:

* required-field validation
* timestamp parsing
* coordinate bounds
* source-key deduplication
* canonical-taxonomy coverage
* spatial-match validation
* feature-key uniqueness
* point-in-time eligibility
* join-cardinality checks
* temporal-support validation
* train/validation/test leakage controls
* integration-weight validation
* feature-coverage monitoring
* immutable snapshot identity and lineage

Model lineage can be traced conceptually through:

```text
model artifact
    ↓
final-model-table snapshot
    ↓
event + integration snapshots
    ↓
feature versions
    ↓
Gold / Silver records
    ↓
Bronze source data
    ↓
original landing object
```

The published intensity validation run also stores its model configuration, runtime versions, snapshot identifiers, feature-contract hash, training history, and feature importance alongside the serialized model artifact.

---

# Technology Stack

## Data engineering

* Python
* Dagster
* Polars
* PyArrow
* Apache Spark / PySpark
* Delta Lake / delta-rs
* H3
* DuckDB
* object storage

## Geospatial and external data

* OpenStreetMap / Geofabrik
* U.S. Census ACS
* Census TIGER/Line
* Open-Meteo-compatible weather data
* `pvlib`
* NAIP
* Sentinel-2 L2A

## Machine learning

* XGBoost
* PyTorch
* Optuna
* CUDA / GPU training
* temporal validation
* point-process likelihood evaluation
* calibration analysis
* marked temporal point-process research

## Serving and product

* FastAPI
* Next.js / React
* MapLibre
* H3 multi-resolution spatial serving
* production inference snapshots

## Engineering

* `uv`
* `pytest`
* GitHub Actions
* Python packaging
* configuration-driven pipelines
* immutable dataset / model artifacts

---

# Repository Layout

```text
crimenet/
├── src/
│   ├── crimenet_data/
│   │   ├── assets/
│   │   │   ├── crime/               # Source ingestion + canonicalization
│   │   │   ├── event_spine/         # Leakage-safe event construction
│   │   │   ├── integration/         # Point-process integration sampling
│   │   │   ├── environmental/       # Weather + solar enrichment
│   │   │   └── final_model_table/   # Canonical model dataset
│   │   ├── national_h3_audit/       # National feature-store audits
│   │   ├── resources/               # Storage + path contracts
│   │   └── definitions.py           # Dagster definitions
│   │
│   └── machine_learning/
│       ├── data/                     # Model-table + evaluation utilities
│       ├── experiments/              # HPO + evaluation orchestration
│       └── models/
│           ├── xgboost/
│           └── crimenet_omega/
│
├── tests/
├── artifacts/
├── pyproject.toml
└── README.md
```

Large datasets, model artifacts, source downloads, inference snapshots, and experiment outputs are intentionally excluded from Git.

---

# Local Development

## Install

```bash
uv sync
```

## Start Dagster

```bash
uv run dagster dev -m crimenet_data.definitions
```

## Run tests

```bash
uv run pytest
```

Environment-specific storage and service credentials should be provided through environment variables and must not be committed.

---

# Current Status

## Completed

* [x] 15-source audited production crime footprint
* [x] Canonical 87-subtype crime taxonomy
* [x] 12-year historical modeling horizon
* [x] National H3-r9 socioeconomic feature store
* [x] National OpenStreetMap feature store
* [x] National feature-store structural audits
* [x] TIGER/Line geographic mapping
* [x] Historical weather enrichment
* [x] Solar and lighting enrichment
* [x] Leakage-safe feature-availability logic
* [x] Event-spine construction
* [x] Source-specific temporal-support contracts
* [x] Monte Carlo integration sampling
* [x] 180M+ example final model table
* [x] Chronological train/validation/test split contract
* [x] Production intensity modeling pipeline
* [x] Conditional 87-subtype mark modeling pipeline
* [x] Optuna hyperparameter-optimization infrastructure
* [x] Temporal evaluation infrastructure
* [x] 2024 full-scale temporal validation
* [x] Training-serving feature contract
* [x] Future-state feature reconstruction
* [x] Rolling 24-hour inference
* [x] Production inference snapshot materialization
* [x] Multi-resolution H3 serving layer
* [x] FastAPI inference service
* [x] Interactive CrimeSense geospatial explorer
* [x] Public CrimeSense deployment at [crimesense.ai](https://crimesense.ai)
* [x] CrimeNet Omega initial implementation
* [x] NAIP and Sentinel-2 imagery pipelines
* [x] Dagster orchestration
* [x] GitHub Actions CI

## Intentionally not completed

* [ ] **Final evaluation on the sealed 2025+ test set**

The test set has deliberately not been opened during model development.

## In progress

* [ ] Full-scale CrimeNet Omega experimentation
* [ ] Stronger historical and spatiotemporal baseline comparisons
* [ ] Calibration refinement
* [ ] Hierarchical marked-event modeling
* [ ] Explicit reporting / observation model
* [ ] Larger-scale imagery integration into the production feature contract
* [ ] Additional source promotion beyond the current 15-source production footprint
* [ ] Production monitoring and forecast-quality observability

## Planned

* [ ] Frozen-model evaluation on the untouched test horizon
* [ ] Streaming source ingestion
* [ ] Kafka / Flink event pipeline
* [ ] Automated drift detection
* [ ] Scheduled retraining
* [ ] Low-latency online feature retrieval where justified
* [ ] Expansion of the production forecast footprint beyond the current 15 geographies

---

# Responsible Use

CrimeNet models **observed crime and reporting patterns across geographic areas and time windows**.

It does **not** model an individual's propensity to commit a crime and is not designed to:

* identify likely offenders
* predict person-level criminal behavior
* support person-level surveillance
* make automated policing decisions
* determine guilt, dangerousness, or intent
* replace public-policy or domain-expert review

Observed crime records can reflect:

* reporting behavior
* police deployment
* enforcement intensity
* data-collection practices
* municipal policy
* historical bias

A technically calibrated prediction of recorded crime is therefore **not necessarily an unbiased estimate of all crime that occurred or will occur**.

CrimeSense should be interpreted as a geospatial forecasting and systems-engineering project, not as a person-level decision system.

---

# Background

CrimeNet grew out of **AcciNet**, an earlier statewide geospatial crash-risk platform.

CrimeNet extends that engineering approach into a substantially larger system with:

* heterogeneous public-data ingestion
* national feature infrastructure
* point-in-time feature correctness
* source-specific temporal support
* hundreds of millions of model examples
* distributed GPU experimentation
* production future-state inference
* multi-resolution geospatial serving
* continuous-time event-modeling research
* a live interactive product

The project is intentionally built as an engineering and ML system rather than a one-off modeling notebook.

---

# Live Demo

**CrimeSense:** https://crimesense.ai

---

# License

Copyright 2026 Aldrin Roshan.

Licensed under the [Apache License 2.0](LICENSE).
