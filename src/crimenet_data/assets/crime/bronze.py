import polars as pl
import dagster as dg
from crimenet_data.resources import CrimeLakeResources, CITIES


def build_bronze_city_assets(city_name: str) -> dg.AssetsDefinition:
    @dg.asset(
        name=f"bronze_{city_name}",
        group_name="bronze_crime"
    )
    def _bronze_asset(crime_lake: CrimeLakeResources) -> pl.LazyFrame:
        return pl.scan_parquet(
            crime_lake.source_uri(city_name),
            hive_partitioning=True,
            credential_provider=pl.CredentialProviderGCP(),
        )
    return _bronze_asset

bronze_assets = [build_bronze_city_assets(city) for city in CITIES]



