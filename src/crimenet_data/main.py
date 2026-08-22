import polars as pl

GCP = pl.CredentialProviderGCP()

MANIFEST = (
    "gs://crimenet/bronze/imagery/optimized/"
    "manifests/imagery_items.parquet"
)

sentinel_id = (
    "S2A_MSIL2A_20150802T155136_R011_"
    "T18TWL_20210411T175647"
)

df = pl.read_parquet(
    MANIFEST,
    credential_provider=GCP,
)

# One Sentinel scene has multiple asset rows, so take one.
s = (
    df.filter(
        (pl.col("collection") == "sentinel-2-l2a")
        & (pl.col("item_id") == sentinel_id)
    )
    .unique(subset=["item_id"])
    .row(0, named=True)
)

print("Sentinel capture:", s["capture_timestamp_utc"])
print(
    "Sentinel bbox:",
    s["bbox_min_lon"],
    s["bbox_min_lat"],
    s["bbox_max_lon"],
    s["bbox_max_lat"],
)

# Bounding-box intersection.
naip = (
    df.filter(pl.col("collection") == "naip")
    .unique(subset=["item_id"])
    .filter(
        # NAIP right edge is right of Sentinel left edge
        (pl.col("bbox_max_lon") >= s["bbox_min_lon"])
        &
        # NAIP left edge is left of Sentinel right edge
        (pl.col("bbox_min_lon") <= s["bbox_max_lon"])
        &
        # NAIP top is above Sentinel bottom
        (pl.col("bbox_max_lat") >= s["bbox_min_lat"])
        &
        # NAIP bottom is below Sentinel top
        (pl.col("bbox_min_lat") <= s["bbox_max_lat"])
    )
    .with_columns(
        (
            pl.col("capture_timestamp_utc")
            - pl.lit(s["capture_timestamp_utc"])
        )
        .dt.total_days()
        .abs()
        .alias("days_from_sentinel")
    )
    .sort("days_from_sentinel")
)

row = naip.select("gcs_uri").row(0, named=True)
print(row["gcs_uri"])
q