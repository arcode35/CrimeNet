# CrimeNet

CrimeNet is a multi-city crime-data lakehouse implemented with Databricks,
Delta Lake, Unity Catalog, PySpark, and Declarative Automation Bundles.

## Sources

- Dallas
- Houston
- Fort Worth

## Setup

```bash
uv sync
databricks auth login
databricks bundle validate --target dev
databricks bundle deploy --target dev
databricks bundle run --target dev crime_pipeline
