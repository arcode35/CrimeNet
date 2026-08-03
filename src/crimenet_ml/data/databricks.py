"""Unity Catalog adapter.

Spark performs table access, filtering, and projection. The resulting frames are
collected to the driver for the unchanged single-node XGBRegressor; this module
does not claim or substitute distributed XGBoost training.
"""

from __future__ import annotations

import warnings

from crimenet_ml.data.interfaces import DatasetIdentity, LoadedSplit


class DatabricksTableLoader:
    def __init__(
        self,
        table: str,
        split_column: str,
        feature_set: str,
        row_limit: int = 5_000_000,
        allow_large_collection: bool = False,
        seed: int = 42,
    ) -> None:
        self.table = table
        self.split_column = split_column
        self.feature_set = feature_set
        self.row_limit = row_limit
        self.allow_large_collection = allow_large_collection
        self.seed = seed
        self._counts: dict[str, int] = {}
        self._selected_columns: list[str] = []
        self._delta_version: int | None = None

    @staticmethod
    def _spark():
        try:
            from pyspark.sql import SparkSession
        except ImportError as exc:
            raise RuntimeError("PySpark is required for the Databricks data backend") from exc
        session = SparkSession.getActiveSession()
        if session is None:
            raise RuntimeError("No active Spark session")
        return session

    def load_split(self, split: str, columns: list[str], limit: int | None = None) -> LoadedSplit:
        spark = self._spark()
        frame = spark.table(self.table)
        missing = sorted(set(columns + [self.split_column]) - set(frame.columns))
        if missing:
            raise ValueError(f"Missing Unity Catalog columns: {missing}")
        projected = frame.where(frame[self.split_column] == split).select(*dict.fromkeys(columns))
        count = projected.count()
        if count == 0:
            raise ValueError(f"Empty {split} split")
        if limit is None and count > self.row_limit and not self.allow_large_collection:
            raise RuntimeError(
                f"Refusing to collect {count:,} rows to the driver; raise "
                "data.collection_row_limit "
                "or set data.allow_large_collection=true"
            )
        if count > self.row_limit:
            warnings.warn(f"Collecting {count:,} rows to the driver", stacklevel=2)
        if limit is not None and count > limit:
            from pyspark.sql import functions as functions

            if "source_city" in columns:
                from pyspark.sql import Window

                window = Window.partitionBy("source_city").orderBy(functions.rand(self.seed))
                city_count = projected.select("source_city").distinct().count()
                per_city = max(1, limit // city_count)
                projected = (
                    projected.withColumn("_rank", functions.row_number().over(window))
                    .where(functions.col("_rank") <= per_city)
                    .drop("_rank")
                    .limit(limit)
                )
            else:
                projected = projected.orderBy(functions.rand(self.seed)).limit(limit)
        pandas_frame = projected.toPandas()
        self._counts[split] = len(pandas_frame)
        self._selected_columns = list(dict.fromkeys(columns))
        try:
            history = spark.sql(f"DESCRIBE HISTORY {self.table} LIMIT 1").first()
            self._delta_version = int(history["version"]) if history else None
        except Exception as exc:  # Delta history can be permission- or format-dependent.
            warnings.warn(f"Could not discover Delta version: {exc}", stacklevel=2)
        return LoadedSplit(split, pandas_frame)

    def identity(self) -> DatasetIdentity:
        parts = self.table.split(".")
        catalog, schema_name, table_name = (
            (parts[0], parts[1], parts[2]) if len(parts) == 3 else (None, None, self.table)
        )
        return DatasetIdentity(
            backend="databricks",
            source_location=self.table,
            table_or_path=self.table,
            fingerprint_or_delta_version=self._delta_version,
            row_count=sum(self._counts.values()),
            split_counts=dict(self._counts),
            selected_feature_set=self.feature_set,
            selected_columns=self._selected_columns,
            catalog=catalog,
            schema_name=schema_name,
            table_name=table_name,
        )
