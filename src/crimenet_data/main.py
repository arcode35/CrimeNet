import polars as pl

GCP = pl.CredentialProviderGCP()

df = pl.scan_parquet(
    "gs://crimenet/raw_files/landing/tract_resources/crime_location_tract_mapping/*.parquet",
    credential_provider=GCP,
)

print(df.collect_schema())