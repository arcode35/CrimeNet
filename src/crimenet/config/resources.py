from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class CrimeNetTables:
    catalog: str
    bronze_schema: str = "bronze"
    silver_schema: str = "silver"

    BRONZE_TABLE_BY_SOURCE: ClassVar[
        dict[str, str]
    ] = {
        "dallas": "dallas_crime",
        "fort_worth": "fort_worth_crime",
        "houston": "houston_crime",
        "seattle": "seattle_crime",
        "chicago": "chicago_crime",
        "new_york": "new_york_crime",
        "san_francisco": "san_francisco_crime",
        "baltimore": "baltimore_crime",
        "washington_dc": "washington_dc_crime",
        "open_meteo_weather": "open_meteo_weather",
        "acs5_tract": "acs5_tract_socioeconomic",
    }

    def _bronze_table(
        self,
        table_name: str,
    ) -> str:
        return (
            f"{self.catalog}."
            f"{self.bronze_schema}."
            f"{table_name}"
        )

    def _silver_table(
        self,
        table_name: str,
    ) -> str:
        return (
            f"{self.catalog}."
            f"{self.silver_schema}."
            f"{table_name}"
        )

    def bronze_for_source(
        self,
        source: str,
    ) -> str:
        try:
            table_name = (
                self.BRONZE_TABLE_BY_SOURCE[
                    source
                ]
            )
        except KeyError as exc:
            supported_sources = ", ".join(
                sorted(
                    self.BRONZE_TABLE_BY_SOURCE
                )
            )

            raise ValueError(
                f"Unsupported source: {source!r}. "
                f"Supported sources: "
                f"{supported_sources}"
            ) from exc

        return self._bronze_table(
            table_name
        )

    @property
    def dallas_bronze(self) -> str:
        return self.bronze_for_source(
            "dallas"
        )

    @property
    def houston_bronze(self) -> str:
        return self.bronze_for_source(
            "houston"
        )

    @property
    def fort_worth_bronze(self) -> str:
        return self.bronze_for_source(
            "fort_worth"
        )

    @property
    def seattle_bronze(self) -> str:
        return self.bronze_for_source(
            "seattle"
        )

    @property
    def chicago_bronze(self) -> str:
        return self.bronze_for_source(
            "chicago"
        )

    @property
    def new_york_bronze(self) -> str:
        return self.bronze_for_source(
            "new_york"
        )

    @property
    def san_francisco_bronze(
        self,
    ) -> str:
        return self.bronze_for_source(
            "san_francisco"
        )

    @property
    def baltimore_bronze(self) -> str:
        return self.bronze_for_source(
            "baltimore"
        )

    @property
    def washington_dc_bronze(
        self,
    ) -> str:
        return self.bronze_for_source(
            "washington_dc"
        )

    @property
    def open_meteo_weather_bronze(
        self,
    ) -> str:
        return self.bronze_for_source(
            "open_meteo_weather"
        )

    @property
    def acs5_tract_bronze(
        self,
    ) -> str:
        return self.bronze_for_source(
            "acs5_tract"
        )

    @property
    def weather_hourly_silver(
        self,
    ) -> str:
        return self._silver_table(
            "weather_hourly"
        )

    @property
    def crime_offenses_silver(
        self,
    ) -> str:
        return self._silver_table(
            "crime_offenses"
        )

    @property
    def tract_socioeconomic_silver(
        self,
    ) -> str:
        return self._silver_table(
            "tract_socioeconomic"
        )