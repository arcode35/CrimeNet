import polars as pl
import dagster as dg
from typing import List
CITIES = ["dallas", "new_york", "chicago", "baltimore", "seattle", "san_francisco", "washington_dc", "fort_worth"]

class CrimeLakeResources(dg.ConfigurableResource):
    """ 
        A centralized cloud configuration store.
    """
    bucket: str = "gs://crimenet"

    @property
    def landing_root():
        return f"{self.bucket}/raw_files/landing"
    @property
    def bronze_root():
        return f"{self.bucket}/bronze"
    @property
    def silver_root():
        return f"{self.bucket}/silver"
    @property
    def gold_root():
        return f"{self.bucket}/gold"

    def source_uri(city: str):
        return f"{self.landing_root}/{city}/**/*.parquet"

    def resolve_city_path(self, city: str, schema: str) -> str:
        schema_dictionary = {
            "bronze": f"{self.bronze_root}/{city}",
            "silver": f"{self.silver_root}/{city}",
            "gold": f"{self.gold_root}/{city}"
        }
        if schema in schema_dictionary:
            return schema_dictionary[schema]
        else: 
            raise KeyError(f"{schema} is not a valid schema. Valid schemas: f{sorted(schema_dictionary.keys())}")
    