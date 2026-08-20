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


def scan_reference_fixture(city_name: reference) -> tuple[pl.LazyFrame, pl.LazyFrame]:
   
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


fort_worth_df = pl.scan_parquet("/Users/xor/crimenet/crimenet_data/tests/fixtures/data/fort_worth.parquet")

fort_worth_df = (
    fort_worth_df
    .with_columns(
        pl.from_epoch(
            pl.col("from_date").cast(
                pl.Int64,
                strict=False,
            ),
            time_unit="ms",
        )
        .cast(pl.Datetime("ns"))
        .alias("occurrence_timestamp")
    )
    .with_columns(
        pl.col("occurrence_timestamp")
        .dt.year()
        .cast(pl.Int64)
        .alias("occurrence_year")
    )
)
print(fort_worth_df.collect().sum())

print(fort_worth_df.filter(
    pl.col("occurrence_year").is_not_null()
).collect().sum())