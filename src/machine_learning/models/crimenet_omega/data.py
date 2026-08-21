from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset


_INTERNAL_COLUMNS = {
    "row_id": "__row_id",
    "observed": "__observed",
    "event_count": "__event_count",
    "integration_weight": "__integration_weight_seconds",
    "city": "__city",
    "subtype": "__subtype",
    "lighting_condition": "__lighting_condition",
}


@dataclass(frozen=True)
class Omega0Vocabulary:
    cities: list[str]
    lighting_conditions: list[str]
    subtypes: list[str]


@dataclass
class Omega0Preprocessor:
    numeric_features: list[str]
    means: np.ndarray
    stds: np.ndarray
    cities: list[str]
    lighting_conditions: list[str]
    subtypes: list[str]

    @property
    def city_to_id(self) -> dict[str, int]:
        return {value: i for i, value in enumerate(self.cities)}

    @property
    def lighting_to_id(self) -> dict[str, int]:
        return {
            value: i
            for i, value in enumerate(self.lighting_conditions)
        }

    @property
    def subtype_to_id(self) -> dict[str, int]:
        return {value: i for i, value in enumerate(self.subtypes)}

    def to_dict(self) -> dict:
        return {
            "numeric_features": list(self.numeric_features),
            "means": self.means.tolist(),
            "stds": self.stds.tolist(),
            "cities": list(self.cities),
            "lighting_conditions": list(self.lighting_conditions),
            "subtypes": list(self.subtypes),
        }


def _scan_delta(path: str) -> pl.LazyFrame:
    if path.startswith("gs://"):
        return pl.scan_delta(
            path,
            credential_provider=pl.CredentialProviderGCP(),
        )
    return pl.scan_delta(path)


def _require_columns_config(columns: Mapping[str, str]) -> None:
    required = {
        "row_id",
        "observed",
        "event_count",
        "integration_weight",
        "city",
        "subtype",
        "lighting_condition",
    }
    missing = required - set(columns)
    if missing:
        raise KeyError(
            f"Missing data.columns entries: {sorted(missing)}"
        )


def _select_expressions(
    *,
    columns: Mapping[str, str],
    numeric_features: list[str],
) -> list[pl.Expr]:
    _require_columns_config(columns)

    expressions: list[pl.Expr] = [
        pl.col(columns["row_id"]).alias(_INTERNAL_COLUMNS["row_id"]),
        pl.col(columns["observed"]).alias(_INTERNAL_COLUMNS["observed"]),
        pl.col(columns["event_count"]).alias(
            _INTERNAL_COLUMNS["event_count"]
        ),
        pl.col(columns["integration_weight"]).alias(
            _INTERNAL_COLUMNS["integration_weight"]
        ),
        pl.col(columns["city"]).alias(_INTERNAL_COLUMNS["city"]),
        pl.col(columns["subtype"]).alias(_INTERNAL_COLUMNS["subtype"]),
        pl.col(columns["lighting_condition"]).alias(
            _INTERNAL_COLUMNS["lighting_condition"]
        ),
    ]

    internal_source_names = {
        columns["row_id"],
        columns["observed"],
        columns["event_count"],
        columns["integration_weight"],
        columns["city"],
        columns["subtype"],
        columns["lighting_condition"],
    }

    for feature in numeric_features:
        if feature not in internal_source_names:
            expressions.append(pl.col(feature))

    return expressions


def load_training_vocabulary(
    table_root: str,
    train_split: str,
    *,
    columns: Mapping[str, str],
) -> Omega0Vocabulary:
    """
    Build categorical vocabularies from the full training partition.

    This is deliberately independent of development sampling so changing
    train_fraction cannot change the neural-network output dimensionality.
    """
    _require_columns_config(columns)

    lf = (
        _scan_delta(table_root)
        .filter(pl.col("split") == train_split)
        .select(
            [
                pl.col(columns["observed"]).alias(
                    _INTERNAL_COLUMNS["observed"]
                ),
                pl.col(columns["city"])
                .cast(pl.String)
                .alias(_INTERNAL_COLUMNS["city"]),
                pl.col(columns["lighting_condition"])
                .fill_null("UNKNOWN")
                .cast(pl.String)
                .alias(_INTERNAL_COLUMNS["lighting_condition"]),
                pl.col(columns["subtype"])
                .cast(pl.String)
                .alias(_INTERNAL_COLUMNS["subtype"]),
            ]
        )
    )

    frame = lf.collect(engine="streaming")

    if frame.height == 0:
        raise RuntimeError(
            f"No rows found in full training split {train_split!r}."
        )

    observed = frame.filter(pl.col(_INTERNAL_COLUMNS["observed"]))

    cities = sorted(
        frame[_INTERNAL_COLUMNS["city"]]
        .drop_nulls()
        .unique()
        .to_list()
    )
    lighting_conditions = sorted(
        frame[_INTERNAL_COLUMNS["lighting_condition"]]
        .fill_null("UNKNOWN")
        .unique()
        .to_list()
    )
    subtypes = sorted(
        observed[_INTERNAL_COLUMNS["subtype"]]
        .drop_nulls()
        .unique()
        .to_list()
    )

    if not cities:
        raise RuntimeError("Training vocabulary contains no cities.")
    if not lighting_conditions:
        raise RuntimeError(
            "Training vocabulary contains no lighting conditions."
        )
    if not subtypes:
        raise RuntimeError("Training vocabulary contains no subtypes.")

    return Omega0Vocabulary(
        cities=cities,
        lighting_conditions=lighting_conditions,
        subtypes=subtypes,
    )


def load_split(
    table_root: str,
    split: str,
    *,
    numeric_features: list[str],
    columns: Mapping[str, str],
    fraction: float = 1.0,
    seed: int = 42,
) -> pl.DataFrame:
    if not numeric_features:
        raise ValueError("numeric_features must not be empty.")
    if len(set(numeric_features)) != len(numeric_features):
        raise ValueError("numeric_features contains duplicates.")
    if not (0.0 < fraction <= 1.0):
        raise ValueError("fraction must be in (0, 1].")

    select_exprs = _select_expressions(
        columns=columns,
        numeric_features=numeric_features,
    )

    lf = (
        _scan_delta(table_root)
        .filter(pl.col("split") == split)
        .select(select_exprs)
        .with_columns(
            pl.col(_INTERNAL_COLUMNS["lighting_condition"])
            .fill_null("UNKNOWN")
            .cast(pl.String),
            pl.col(_INTERNAL_COLUMNS["city"]).cast(pl.String),
            pl.col(_INTERNAL_COLUMNS["subtype"]).cast(pl.String),
            pl.col(_INTERNAL_COLUMNS["observed"]).cast(pl.Boolean),
            pl.col(_INTERNAL_COLUMNS["event_count"]).cast(pl.Float32),
            pl.col(_INTERNAL_COLUMNS["integration_weight"]).cast(
                pl.Float64
            ),
        )
    )

    # Deterministic row-level development sampling. Events and integration
    # rows are sampled with the same inclusion probability, preserving their
    # relative weighting in expectation.
    if fraction < 1.0:
        denominator = 1_000_000
        cutoff = int(fraction * denominator)

        lf = lf.filter(
            (
                pl.col(_INTERNAL_COLUMNS["row_id"]).hash(seed=seed)
                % denominator
            )
            < cutoff
        )

    # Omega-0 is explicitly a complete-case model over these query-time
    # covariates. We avoid a second remote scan solely for dropped-row stats.
    lf = lf.drop_nulls(
        numeric_features + [_INTERNAL_COLUMNS["city"]]
    )

    frame = lf.collect(engine="streaming")

    if frame.height == 0:
        raise RuntimeError(
            f"No rows loaded for split={split!r}."
        )

    _validate_point_process_rows(frame, split=split)

    return frame


def _validate_point_process_rows(
    frame: pl.DataFrame,
    *,
    split: str,
) -> None:
    observed_col = _INTERNAL_COLUMNS["observed"]
    count_col = _INTERNAL_COLUMNS["event_count"]
    weight_col = _INTERNAL_COLUMNS["integration_weight"]
    subtype_col = _INTERNAL_COLUMNS["subtype"]

    if frame[observed_col].null_count() > 0:
        raise RuntimeError(
            f"[{split}] is_observed_event contains null values."
        )
    if frame[count_col].null_count() > 0:
        raise RuntimeError(
            f"[{split}] event_count contains null values."
        )
    if frame[weight_col].null_count() > 0:
        raise RuntimeError(
            f"[{split}] integration weights contain null values."
        )
    if (frame[weight_col] < 0).any():
        raise RuntimeError(
            f"[{split}] negative integration weights found."
        )

    observed = frame.filter(pl.col(observed_col))
    integration = frame.filter(~pl.col(observed_col))

    if observed.height == 0:
        raise RuntimeError(
            f"[{split}] contains no observed-event rows."
        )
    if integration.height == 0:
        raise RuntimeError(
            f"[{split}] contains no integration rows."
        )

    if observed[subtype_col].null_count() > 0:
        raise RuntimeError(
            f"[{split}] observed rows contain null subtype values."
        )

    invalid_observed = observed.filter(pl.col(count_col) <= 0)
    if invalid_observed.height > 0:
        raise RuntimeError(
            f"[{split}] observed rows must have event_count > 0; "
            f"found {invalid_observed.height:,} invalid rows."
        )

    invalid_integration_count = integration.filter(
        pl.col(count_col) != 0
    )
    if invalid_integration_count.height > 0:
        raise RuntimeError(
            f"[{split}] integration rows must have event_count == 0; "
            f"found {invalid_integration_count.height:,} invalid rows."
        )

    invalid_integration_weight = integration.filter(
        pl.col(weight_col) <= 0
    )
    if invalid_integration_weight.height > 0:
        raise RuntimeError(
            f"[{split}] integration rows must have positive exposure; "
            f"found {invalid_integration_weight.height:,} invalid rows."
        )


def fit_preprocessor(
    frame: pl.DataFrame,
    *,
    numeric_features: list[str],
    vocabulary: Omega0Vocabulary,
) -> Omega0Preprocessor:
    x = (
        frame.select(numeric_features)
        .to_numpy()
        .astype(np.float32)
    )

    if not np.isfinite(x).all():
        raise RuntimeError(
            "Non-finite numeric values remain after complete-case filtering."
        )

    means = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    stds = x.std(axis=0, dtype=np.float64).astype(np.float32)

    if not np.isfinite(means).all() or not np.isfinite(stds).all():
        raise RuntimeError(
            "Non-finite preprocessing statistics encountered."
        )

    stds = np.where(stds < 1e-6, 1.0, stds).astype(np.float32)

    return Omega0Preprocessor(
        numeric_features=list(numeric_features),
        means=means,
        stds=stds,
        cities=list(vocabulary.cities),
        lighting_conditions=list(vocabulary.lighting_conditions),
        subtypes=list(vocabulary.subtypes),
    )


class Omega0Dataset(Dataset):
    def __init__(
        self,
        frame: pl.DataFrame,
        preprocessor: Omega0Preprocessor,
        *,
        numeric_features: list[str],
        exposure_unit_seconds: float,
    ) -> None:
        if exposure_unit_seconds <= 0:
            raise ValueError(
                "exposure_unit_seconds must be > 0."
            )
        if list(numeric_features) != preprocessor.numeric_features:
            raise ValueError(
                "Dataset numeric feature order does not match "
                "the fitted preprocessor."
            )

        self.preprocessor = preprocessor
        self.exposure_unit_seconds = float(exposure_unit_seconds)

        x = (
            frame.select(numeric_features)
            .to_numpy()
            .astype(np.float32)
        )
        x = (x - preprocessor.means) / preprocessor.stds

        if not np.isfinite(x).all():
            raise RuntimeError(
                "Non-finite normalized numeric features encountered."
            )

        self.numeric = torch.from_numpy(x)

        city_map = preprocessor.city_to_id
        lighting_map = preprocessor.lighting_to_id
        subtype_map = preprocessor.subtype_to_id

        cities = frame[_INTERNAL_COLUMNS["city"]].to_list()
        unknown_cities = sorted(set(cities) - set(city_map))
        if unknown_cities:
            raise ValueError(
                f"Unseen cities outside training vocabulary: "
                f"{unknown_cities}"
            )

        self.city = torch.tensor(
            [city_map[value] for value in cities],
            dtype=torch.long,
        )

        lighting = (
            frame[_INTERNAL_COLUMNS["lighting_condition"]]
            .fill_null("UNKNOWN")
            .to_list()
        )
        unknown_lighting = sorted(set(lighting) - set(lighting_map))
        if unknown_lighting:
            raise ValueError(
                "Unseen lighting categories outside training vocabulary: "
                f"{unknown_lighting}"
            )

        self.lighting = torch.tensor(
            [lighting_map[value] for value in lighting],
            dtype=torch.long,
        )

        observed = np.asarray(
            frame[_INTERNAL_COLUMNS["observed"]].to_numpy(),
            dtype=np.bool_,
        )
        self.is_observed = torch.from_numpy(observed)

        self.event_count = torch.from_numpy(
            frame[_INTERNAL_COLUMNS["event_count"]]
            .to_numpy()
            .astype(np.float32)
        )

        # Converts cell-seconds to the exposure unit configured in YAML.
        # With exposure_unit_seconds=3600, this is cell-hours.
        integration_weight = (
            frame[_INTERNAL_COLUMNS["integration_weight"]]
            .to_numpy()
            .astype(np.float64)
            / self.exposure_unit_seconds
        ).astype(np.float32)

        if not np.isfinite(integration_weight).all():
            raise RuntimeError(
                "Non-finite converted integration weights encountered."
            )

        self.integration_weight = torch.from_numpy(integration_weight)

        subtype_ids = np.full(
            frame.height,
            -100,
            dtype=np.int64,
        )

        subtype_values = frame[_INTERNAL_COLUMNS["subtype"]].to_list()

        for i, (is_observed, value) in enumerate(
            zip(observed, subtype_values)
        ):
            if not is_observed:
                continue
            if value not in subtype_map:
                raise ValueError(
                    f"Observed subtype {value!r} is outside the full "
                    "training vocabulary."
                )
            subtype_ids[i] = subtype_map[value]

        self.subtype = torch.from_numpy(subtype_ids)

    def __len__(self) -> int:
        return self.numeric.shape[0]

    def __getitem__(
        self,
        idx: int,
    ) -> dict[str, torch.Tensor]:
        return {
            "numeric": self.numeric[idx],
            "city": self.city[idx],
            "lighting": self.lighting[idx],
            "is_observed": self.is_observed[idx],
            "event_count": self.event_count[idx],
            "integration_weight": self.integration_weight[idx],
            "subtype": self.subtype[idx],
        }