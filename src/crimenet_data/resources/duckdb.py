from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import dagster as dg
import duckdb


class DuckDBResource(dg.ConfigurableResource):
    database: str = ":memory:"
    enable_spatial: bool = True

    @contextmanager
    def get_connection(
        self,
    ) -> Iterator[duckdb.DuckDBPyConnection]:
        if self.database != ":memory:":
            Path(self.database).parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        connection = duckdb.connect(
            database=self.database,
        )

        try:
            if self.enable_spatial:
                connection.install_extension("spatial")
                connection.load_extension("spatial")

            yield connection
        finally:
            connection.close()