from __future__ import annotations

from pathlib import Path

import pandas as pd

from crimenet_ml.data.local import LocalParquetLoader


def _write_partition(root: Path, split: str, city: str, values: list[int]) -> None:
    directory = root / f"dataset_split={split}" / f"source_city={city}"
    directory.mkdir(parents=True)
    pd.DataFrame(
        {
            "x": values,
            "event_multiplicity": [1] * len(values),
            "importance_weight": [1.0] * len(values),
        }
    ).to_parquet(directory / "part.parquet", index=False)


def test_local_loader_filters_split_and_balances_city(tmp_path: Path) -> None:
    _write_partition(tmp_path, "train", "a", list(range(8)))
    _write_partition(tmp_path, "train", "b", list(range(100, 108)))
    _write_partition(tmp_path, "validation", "a", [999])
    loader = LocalParquetLoader(tmp_path, "dataset_split", "history_v1", seed=5)
    loaded = loader.load_split("train", ["x", "source_city"], limit=6)
    assert len(loaded.frame) == 6
    assert set(loaded.frame["source_city"]) == {"a", "b"}
    assert 999 not in set(loaded.frame["x"])
    assert loaded.frame.groupby("source_city").size().to_dict() == {"a": 3, "b": 3}
