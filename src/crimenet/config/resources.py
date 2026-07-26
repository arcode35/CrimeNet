"""Central construction of Unity Catalog object names."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CrimeNetTables:
    catalog: str
    bronze_schema: str = "bronze"
    silver_schema: str = "silver"

    @property
    def dallas_bronze(self) -> str:
        return f"{self.catalog}.{self.bronze_schema}.dallas_crime"

    @property
    def houston_bronze(self) -> str:
        return f"{self.catalog}.{self.bronze_schema}.houston_crime"

    @property
    def fort_worth_bronze(self) -> str:
        return f"{self.catalog}.{self.bronze_schema}.fort_worth_crime"

    @property
    def crime_offenses_silver(self) -> str:
        return f"{self.catalog}.{self.silver_schema}.crime_offenses"

    def bronze_for_city(self, city: str) -> str:
        table_by_city = {
            "dallas": self.dallas_bronze,
            "houston": self.houston_bronze,
            "fort_worth": self.fort_worth_bronze,
        }

        try:
            return table_by_city[city]
        except KeyError as exc:
            raise ValueError(f"Unsupported city: {city!r}") from exc
