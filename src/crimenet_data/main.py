import polars as pl

GCP = pl.CredentialProviderGCP()

import polars as pl 
df = pl.read_parquet("gs://crimenet/silver/imagery/h3_temporal_index/part-00000.parquet", credential_provider=GCP)


print(df.estimated_size("gb"))