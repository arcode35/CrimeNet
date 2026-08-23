import polars as pl

GCP = pl.CredentialProviderGCP()

df = pl.scan_delta(
    "gs://crimenet/gold_staging_/model_table_nyc_timestamp_fix",
    credential_provider=GCP,
)

print(df.collect_schema())