import polars as pl

GCP = pl.CredentialProviderGCP()

df = pl.read_parquet(
    "gs://crimenet/gold/imagery/embeddings/foundation_v1/"
    "sentinel2/part-00000.parquet",
    credential_provider=GCP,
)

print(
    df.group_by(
        "sentinel_input_mode",
        "temporal_sequence_length",
    )
    .len()
    .sort(
        ["sentinel_input_mode", "temporal_sequence_length"]
    )
)