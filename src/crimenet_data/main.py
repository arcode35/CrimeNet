import polars as pl

GCP = pl.CredentialProviderGCP()

pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
df = pl.scan_delta(
    "gs://crimenet/gold/event_spine",
    credential_provider=GCP,
)
print(df.collect_schema())