from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CrimeNetTables:
    catalog: str
    bronze_schema: str = "bronze"
    silver_schema: str = "silver"

    @property
    def dallas_bronze(self) -> str:
        return (
            f"{self.catalog}."
            f"{self.bronze_schema}."
            "dallas_crime"
        )

    @property
    def houston_bronze(self) -> str:
        return (
            f"{self.catalog}."
            f"{self.bronze_schema}."
            "houston_crime"
        )

    @property
    def fort_worth_bronze(self) -> str:
        return (
            f"{self.catalog}."
            f"{self.bronze_schema}."
            "fort_worth_crime"
        )

    @property
    def open_meteo_weather_bronze(
        self,
    ) -> str:
        return (
            f"{self.catalog}."
            f"{self.bronze_schema}."
            "open_meteo_weather"
        )

    @property
    def weather_hourly_silver(self) -> str:
        return (
            f"{self.catalog}."
            f"{self.silver_schema}."
            "weather_hourly"
        )

    @property
    def crime_offenses_silver(self) -> str:
        return (
            f"{self.catalog}."
            f"{self.silver_schema}."
            "crime_offenses"
        )

    def bronze_for_source(
        self,
        source: str,
    ) -> str:
        table_by_source = {
            "dallas": self.dallas_bronze,
            "houston": self.houston_bronze,
            "fort_worth": self.fort_worth_bronze,
            "open_meteo_weather": (
                self.open_meteo_weather_bronze
            ),
            "acs5_tract": (
                self.acs5_tract_bronze
            ),
        }

        try:
            return table_by_source[source]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported source: {source!r}"
            ) from exc

    @property
    def acs5_tract_bronze(self) -> str:
        return (
            f"{self.catalog}."
            f"{self.bronze_schema}."
            "acs5_tract_socioeconomic"
        )