import polars as pl

GCP = pl.CredentialProviderGCP()
df = pl.scan_delta("gs://crimenet/gold_staging_/model_table_nyc_timestamp_fix", credential_provider=GCP)
(
    df
    .select(
        "canonical_subtype_code",
        "canonical_family_code",
    )
    .unique()
    .group_by(
        "canonical_subtype_code"
    )
    .agg(
        pl.col("canonical_family_code")
        .n_unique()
        .alias("family_count")
    )
    .filter(
        pl.col("family_count") != 1
    )
)

print(
    df.filter(
        pl.col("is_observed_event")
    )
    .select(
        [
            "model_row_id",
            "source_city",
            "row_timestamp_utc",
            "canonical_family_code",
            "canonical_offense_family",
            "canonical_subtype_code",
            "canonical_offense_subtype",
        ]
    )
    .head(20)
    .collect()
)
check = (
    df.filter(
        pl.col("is_observed_event")
    )
    .select(
        [
            "canonical_subtype_code",
            "canonical_family_code",
        ]
    )
    .unique()
    .group_by(
        "canonical_subtype_code"
    )
    .agg(
        pl.col("canonical_family_code")
        .n_unique()
        .alias("family_count")
    )
    .filter(
        pl.col("family_count") != 1
    )
)

