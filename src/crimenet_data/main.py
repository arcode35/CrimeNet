import polars as pl

GCP = pl.CredentialProviderGCP()

pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
df = pl.scan_delta(
    "gs://crimenet/gold_staging_/model_table_nyc_timestamp_fix",
    credential_provider=GCP,
)

marks = (
    df
    .filter(pl.col("is_observed_event"))
    .select(
        [
            "canonical_subtype_code",
            "canonical_offense_subtype",
            "canonical_family_code",
            "canonical_offense_family",
        ]
    )
    .unique()
    .sort(
        [
            "canonical_family_code",
            "canonical_subtype_code",
        ]
    )
    .collect()
)

print(marks)
print(f"\nDistinct marks: {marks['canonical_subtype_code'].n_unique()}")