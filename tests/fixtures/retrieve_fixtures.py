import polars as pl
FIXTURES_ROOT = "gs://crimenet/fixtures"

def scan_fixture(city_name: str) -> tuple[pl.LazyFrame, pl.LazyFrame]:
   
    data_path = f"{FIXTURES_ROOT}/{city_name}_crime/data/*.parquet"
    diagnostics_path = f"{FIXTURES_ROOT}/{city_name}_crime/diagnostics/*.parquet"

    fixture_data = pl.scan_parquet(
        data_path,
        hive_partitioning=True,
        credential_provider=pl.CredentialProviderGCP(),
    )
    fixture_diagnostics = pl.scan_parquet(
        diagnostics_path,
        hive_partitioning=True,
        credential_provider=pl.CredentialProviderGCP(),
    )
    return fixture_data, fixture_diagnostics


cities = ["dallas", "new_york", "chicago", "baltimore", "seattle", "san_francisco", "washington_dc", "fort_worth"]
for city in cities:
    fixture_data, fixture_diagnostics = scan_fixture(city)
    fixture_data.collect().write_parquet(f"data/{city}.parquet")
    fixture_diagnostics.collect().write_parquet(f"diagnostics/{city}.parquet")

    print(f"exported {city}")