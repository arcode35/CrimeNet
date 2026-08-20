import polars as pl
import dagster as dg
from typing import List
from pathlib import Path
import deltalake
CITIES = ["dallas", "new_york", "chicago", "baltimore", "seattle", "san_francisco", "washington_dc", "fort_worth"]
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent

class CrimeLakeResources(dg.ConfigurableResource):
    """ 
        A centralized cloud configuration store.
    """

    bucket: str = "gs://crimenet"
    @property
    def landing_root(self):
        return f"{self.bucket}/raw_files/landing"
    @property
    def bronze_root(self):
        return f"{self.bucket}/bronze"
    @property
    def silver_root(self):
        return f"{self.bucket}/silver"
    @property
    def gold_root(self):
        return f"{self.bucket}/gold"
    @property
    def quality_root(self):
        return f"{self.bucket}/quality"
    def source_uri(self, city: str) -> str:
        return f"{self.landing_root}/{city}/**/*.parquet"

    def resolve_weather_path(self, mode: str) -> str:
        if mode == "coastal" or mode == "land":
            return f"{self.landing_root}/weather/open_meteo/era5_{mode}/**/*.json"
        else:
            raise KeyError(f"{mode} is not a valid weather mode. Valid types: coastal or land.")

    def resolve_socioeconomic_path(self) -> str:
        return f"{self.landing_root}/socioeconomic/acs5/tract_semantic_v2/*.parquet"
    def resolve_crosswalk(self) -> str:
        crosswalk_path = f"{self.landing_root}/reference/canonical_crime_crosswalk_v1_3.csv"
        return pl.scan_csv(crosswalk_path, has_header=True)

    def resolve_city_path(self, city: str, schema: str) -> str:
        schema_dictionary = {
            "bronze": f"{self.bronze_root}/crime/{city}",
            "silver": f"{self.silver_root}/crime/{city}",
            "gold": f"{self.gold_root}/crime/{city}"
        }
        if schema in schema_dictionary:
            return schema_dictionary[schema]
        else: 
            raise KeyError(f"{schema} is not a valid schema. Valid schemas: f{sorted(schema_dictionary.keys())}")
    
    def get_city_fixture(self, city: str) -> pl.LazyFrame:
        fixtures_path = PROJECT_ROOT / "tests" / "fixtures" / "data" / f"{city}.parquet"
        return pl.scan_parquet(fixtures_path, use_statistics=True)

    def write_crimenet_table(
        self,
        lf: pl.LazyFrame,
        target_uri: str,
        partitioning_columns: List[str],
    ) -> None:
        lf.sink_delta(
            target_uri,
            mode="overwrite",
            delta_write_options={
                "schema_mode": "overwrite",
                "partition_by": partitioning_columns,
                "writer_properties": deltalake.WriterProperties(
                    compression="zstd",
                    compression_level=3,
                ),
            },
            credential_provider=pl.CredentialProviderGCP(),
        )
