import polars as pl 


df = pl.scan_parquet("gs://crimenet/gold/imagery/embeddings/foundation_v1/naip/*.parquet", credential_provider=pl.CredentialProviderGCP())


print(df.collect_schema())