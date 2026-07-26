# CrimeNet

[![CI](https://github.com/arcode35/crimenet/actions/workflows/ci.yml/badge.svg)](https://github.com/arcode35/crimenet/actions/workflows/ci.yml)

CrimeNet is an end-to-end geospatial machine learning lakehouse for crime-risk prediction. Built on Databricks, Delta Lake, Unity Catalog, and PySpark, it transforms multi-city crime, weather, spatial, and socioeconomic data into governed ML features for automated XGBoost training, evaluation, deployment, monitoring, and retraining.

## Overview

CrimeNet is designed as a production-style data and machine learning platform rather than a standalone predictive model.

The system:

- Ingests and standardizes crime data from multiple municipal sources
- Enriches incidents with H3 spatial indexes and historical weather data
- Enforces data contracts, validation rules, and quarantine workflows
- Produces reproducible, leakage-aware machine learning features
- Trains and evaluates geospatial XGBoost models
- Tracks model versions, metrics, parameters, and feature lineage
- Detects performance or data drift
- Automatically retrains and promotes qualifying models
- Serves crime-risk predictions for downstream analytics and applications

## Data Sources

Initial crime-data coverage includes:

- Dallas
- Houston
- Fort Worth

Planned enrichment sources include:

- ERA5-Land historical weather through Open-Meteo
- Calendar and temporal features
- Census and socioeconomic indicators
- Spatial and neighborhood-level reference data

## Architecture

```text
Municipal Crime Data          Weather and Spatial Data
          │                              │
          └──────────────┬───────────────┘
                         ▼
                 Raw Landing Volume
                         ▼
                 Bronze Delta Tables
                         ▼
        Standardization and Data Quality
                         ▼
                  Silver Delta Tables
                         ▼
          Spatial and Temporal Enrichment
                         ▼
              ML Feature Engineering
                         ▼
                   Gold Feature Tables
                         ▼
              XGBoost Training Pipeline
                         ▼
       Evaluation, Registry, and Promotion
                         ▼
       Monitoring and Automated Retraining
```

## Technology Stack

### Data platform

- Databricks
- Apache Spark and PySpark
- Delta Lake
- Unity Catalog
- Databricks Asset Bundles
- Azure Data Lake Storage Gen2

### Machine learning

- XGBoost
- MLflow experiment tracking
- MLflow Model Registry
- Leakage-aware temporal validation
- Automated model evaluation and retraining
- Feature and prediction monitoring

### Geospatial processing

- H3 spatial indexing
- Databricks spatial SQL functions
- Multi-city coordinate normalization
- Spatial and weather feature enrichment

### Engineering

- Python
- SQL
- `uv`
- `pytest`
- GitHub Actions
- Structured logging
- Environment-specific deployment configuration

## Repository Structure

```text
crimenet/
├── docs/                   # Architecture, contracts, quality, and operations
├── notebooks/              # Exploration and interactive development
├── resources/              # Databricks job and pipeline definitions
├── scripts/                # Validation and deployment commands
├── src/crimenet/
│   ├── config/             # Environment and resource configuration
│   ├── contracts/          # Bronze, Silver, and Gold schemas
│   ├── ingestion/          # Source readers and ingestion utilities
│   ├── jobs/               # Databricks job entry points
│   ├── quality/            # Validation and quarantine rules
│   ├── spatial/            # H3 and geospatial processing
│   ├── transforms/         # Source-specific standardization
│   ├── utils/              # Spark and logging utilities
│   └── weather/            # Weather planning, retrieval, and caching
├── targets/                # Development and production configuration
└── tests/                  # Unit, integration, and fixture data
```

## Setup

### Prerequisites

- Python 3.12
- `uv`
- Databricks CLI
- Access to a Databricks workspace with Unity Catalog enabled

Install the project dependencies:

```bash
uv sync
```

Authenticate with Databricks:

```bash
databricks auth login
```

Validate the development bundle:

```bash
databricks bundle validate --target dev
```

Deploy the development environment:

```bash
databricks bundle deploy --target dev
```

Run the crime-data pipeline:

```bash
databricks bundle run --target dev crime_pipeline
```

## Development

Run the test suite:

```bash
uv run pytest
```

Run the complete local validation script:

```bash
./scripts/check.sh
```

Build the Python package:

```bash
uv build
```

## Pipeline Stages

1. **Raw landing** — Preserves source files and external API responses for replay and recovery.
2. **Bronze** — Ingests source-aligned records with operational metadata.
3. **Silver** — Standardizes schemas, coordinates, timestamps, and crime categories.
4. **Data quality** — Applies validation rules and quarantines invalid records.
5. **Enrichment** — Adds H3 cells, weather conditions, and temporal features.
6. **Gold** — Produces model-ready features and analytical aggregates.
7. **Training** — Fits and tunes XGBoost crime-risk models.
8. **Evaluation** — Measures predictive quality, calibration, and geographic stability.
9. **Promotion** — Registers and promotes models that satisfy configured thresholds.
10. **Monitoring** — Detects drift and initiates automated retraining when required.

## Machine Learning Design

CrimeNet is intended to predict relative crime risk across geographic cells and time windows.

The ML workflow is designed around:

- Temporal train, validation, and test splits
- Prevention of future-information leakage
- Spatially consistent feature generation
- Class imbalance handling
- Probability calibration
- Explainability through feature attribution
- Reproducible experiment tracking
- Model comparison against historical baselines
- Automated retraining with controlled promotion criteria

The project predicts aggregate geographic risk. It is not intended to infer individual criminal behavior or support person-level profiling.

## Project Status

CrimeNet is under active development.

Current work includes:

- Multi-city Bronze and Silver pipelines
- H3-based spatial standardization
- Historical ERA5-Land temperature ingestion
- Raw API-response caching and recovery
- Data contracts and quality enforcement
- Databricks Asset Bundle orchestration

Planned work includes:

- Gold feature tables
- XGBoost training and hyperparameter tuning
- MLflow model registration
- Drift and performance monitoring
- Automated retraining and model promotion
- Prediction-serving and visualization layers

## Documentation

Additional documentation is available under [`docs/`](docs/):

- [`architecture.md`](docs/architecture.md)
- [`data_contracts.md`](docs/data_contracts.md)
- [`data_quality.md`](docs/data_quality.md)
- [`operations_runbook.md`](docs/operations_runbook.md)

## License

License information will be added before the first public release.
