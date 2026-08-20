# src/crimenet_data/resources/__init__.py

from crimenet_data.resources.crime_lake import CrimeLakeResources
from crimenet_data.resources.duckdb import DuckDBResource

__all__ = [
    "CrimeLakeResources",
    "DuckDBResource",
]