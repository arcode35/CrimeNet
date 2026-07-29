from __future__ import annotations

from dataclasses import dataclass

from crimenet.config.validation import validate_identifier


@dataclass(frozen=True)
class CrimeNetTables:
    catalog: str
    bronze_schema: str = "bronze"
    silver_schema: str = "silver"
    gold_schema: str = "gold"
    operations_schema: str = "ops"
    data_quality_schema: str = "data_quality"

    def __post_init__(self) -> None:
        for component_name in (
            "catalog",
            "bronze_schema",
            "silver_schema",
            "gold_schema",
            "operations_schema",
            "data_quality_schema",
        ):
            validate_identifier(
                getattr(self, component_name),
                label=component_name,
            )

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

    @property
    def crime_quarantine(self) -> str:
        return (
            f"{self.catalog}."
            f"{self.data_quality_schema}."
            "crime_quarantine"
        )

    @property
    def quality_results(self) -> str:
        return (
            f"{self.catalog}."
            f"{self.data_quality_schema}."
            "quality_results"
        )

    @property
    def pipeline_failures(self) -> str:
        return (
            f"{self.catalog}."
            f"{self.operations_schema}."
            "pipeline_failures"
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

    @property
    def tract_socioeconomic_silver(self) -> str:
        return (
            f"{self.catalog}."
            f"{self.silver_schema}."
            "tract_socioeconomic"
        )
