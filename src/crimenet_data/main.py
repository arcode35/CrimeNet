import os
import polars as pl

B2_REGION = "us-east-005"  # replace if yours differs
B2_ENDPOINT = f"https://s3.{B2_REGION}.backblazeb2.com"

df = (
    pl.scan_parquet(
        "s3://crimenet-data/bronze/crime/sonoma_county_sheriff_ca/"
        "snapshot_id=4f02cc45-5b78-476f-b177-11f238073e01/**/*.parquet",
        storage_options={
            "endpoint_url": B2_ENDPOINT,
            "region": B2_REGION,
            "aws_access_key_id": os.environ["B2_KEY_ID"],
            "aws_secret_access_key": os.environ["B2_APPLICATION_KEY"],
        },
    )
    .select("location")
    .filter(pl.col("location").is_not_null())
    .head(20)
    .collect()
)

print(df)