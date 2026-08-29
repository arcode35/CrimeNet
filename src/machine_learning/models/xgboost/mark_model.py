from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb

from machine_learning.data.features import resolve_feature_contract
from machine_learning.data.model_table import resolve_model_table


DEFAULT_TARGET_COLUMN = "canonical_subtype_code"
AUXILIARY_COLUMNS = ("source_city", "is_observed_event", "event_count", "model_row_id")

# Zero-shot mark prediction must not get a hidden shortcut through local/city crime history.
ZERO_SHOT_FORBIDDEN_PREFIXES = (
    "cell_crime_",
    "cell_violent_",
    "cell_property_",
    "city_crime_",
    "k1_crime_",
    "same_mark_",
)
ZERO_SHOT_FORBIDDEN_EXACT = frozenset(
    {
        "source_city",
        "has_crime_cell_28d",
        "has_crime_city_28d",
        "hours_since_last_crime_cell_capped_28d",
        "hours_since_last_crime_city_capped_28d",
        "cell_crime_24h_vs_28d_ratio",
        "cell_share_of_k1_crime_24h",
    }
)


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))


def _resolve_feature_columns(
    config: dict[str, Any],
    *,
    available_columns: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Resolve the canonical feature contract used by HPO and final training.

    xgb_hpo.py calls this helper directly while building the immutable Arrow cache,
    so its signature intentionally matches the current HPO contract.
    """

    feature_cfg = config["features"]

    resolved_numeric = feature_cfg.get("resolved_numeric")
    resolved_categorical = feature_cfg.get("resolved_categorical")

    if resolved_numeric is not None and resolved_categorical is not None:
        numeric = [str(value) for value in resolved_numeric]
        categorical = [str(value) for value in resolved_categorical]
    else:
        if available_columns is None:
            raise ValueError(
                "available_columns is required when the feature contract has not "
                "already been resolved by the coordinator."
            )
        contract = resolve_feature_contract(
            feature_cfg,
            available_columns=available_columns,
        )
        numeric = list(contract.numeric)
        categorical = list(contract.categorical)

    feature_columns = _unique([*numeric, *categorical])
    categorical_columns = _unique(categorical)

    missing_categorical = sorted(set(categorical_columns) - set(feature_columns))
    if missing_categorical:
        raise RuntimeError(
            f"Categorical columns are not present in the resolved feature set: {missing_categorical}"
        )

    if available_columns is not None:
        missing = sorted(set(feature_columns) - set(available_columns))
        if missing:
            raise RuntimeError(f"Resolved mark features are absent from the model table: {missing}")

    if bool(feature_cfg.get("zero_shot_geography", False)):
        forbidden = sorted(
            column
            for column in feature_columns
            if column in ZERO_SHOT_FORBIDDEN_EXACT
            or any(column.startswith(prefix) for prefix in ZERO_SHOT_FORBIDDEN_PREFIXES)
        )
        if forbidden:
            raise RuntimeError(
                "Zero-shot mark feature contract contains geography/history leakage shortcuts: "
                f"{forbidden}"
            )
        if bool(feature_cfg.get("include_local_history", False)):
            raise RuntimeError("zero_shot_geography=true requires include_local_history=false")
        if bool(feature_cfg.get("include_city_history", False)):
            raise RuntimeError("zero_shot_geography=true requires include_city_history=false")

    return feature_columns, categorical_columns


def _deterministic_observed_sample(
    *,
    table: pl.LazyFrame,
    split: str,
    fraction: float,
    seed: int,
    feature_columns: list[str],
    target_column: str,
) -> pl.DataFrame:
    """Load only observed events for the conditional mark model.

    `table` is already split-pruned by ModelTableRef.scan_split(split). Keeping
    split as an argument preserves the HPO cache interface and improves errors.
    `source_city` is always retained as an auxiliary CV/evaluation column even
    though it is deliberately excluded from the zero-shot feature matrix.
    """

    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError(f"{split}: sampling fraction must be in (0, 1], got {fraction}")

    available = set(table.collect_schema().names())
    required = {
        "source_city",
        "is_observed_event",
        "event_count",
        "model_row_id",
        target_column,
        *feature_columns,
    }
    missing = sorted(required - available)
    if missing:
        raise RuntimeError(f"{split}: mark sample is missing required columns: {missing}")

    columns = _unique(["source_city", *feature_columns, target_column])
    frame = table.filter(
        pl.col("is_observed_event").cast(pl.Boolean, strict=False).fill_null(False)
        & (pl.col("event_count").cast(pl.Int64, strict=False) == 1)
        & pl.col(target_column).is_not_null()
    )

    if float(fraction) < 1.0:
        buckets = 1_000_000
        threshold = max(1, int(float(fraction) * buckets))
        frame = frame.filter(
            (pl.col("model_row_id").hash(seed=int(seed)) % buckets) < threshold
        )

    result = frame.select(columns).collect(engine="streaming")
    if result.is_empty():
        raise RuntimeError(f"{split}: deterministic observed-event sample is empty")
    return result


def _apply_geographic_role(
    frame: pl.DataFrame,
    *,
    config: dict[str, Any],
    role: str,
) -> pl.DataFrame:
    """Apply the HPO fold split when the cache layer has not already done it."""

    held_out = [
        str(value)
        for value in config.get("validation", {}).get("geographic_holdout_cities", [])
    ]
    use_all_cities = bool(config.get("final_training", {}).get("use_all_cities", False))

    if use_all_cities or not held_out:
        return frame

    if role == "train":
        filtered = frame.filter(~pl.col("source_city").is_in(held_out))
    elif role == "validation":
        filtered = frame.filter(pl.col("source_city").is_in(held_out))
    else:
        raise ValueError(f"Unknown geographic role: {role!r}")

    if filtered.is_empty():
        raise RuntimeError(f"Geographic {role} frame is empty for holdout cities={held_out}")
    return filtered


def _sample_summary(
    name: str,
    frame: pl.DataFrame,
    *,
    target_column: str,
) -> dict[str, Any]:
    if frame.is_empty():
        raise ValueError(f"{name}: no observed-event rows")

    counts = frame.group_by(target_column).len().sort("len", descending=True)
    class_counts = {
        str(label): int(count)
        for label, count in zip(
            counts.get_column(target_column).to_list(),
            counts.get_column("len").to_list(),
            strict=True,
        )
    }
    city_counts = {
        str(city): int(count)
        for city, count in zip(
            *(
                frame.group_by("source_city")
                .len()
                .sort("source_city")
                .select("source_city", "len")
                .to_dict(as_series=False)
                .values()
            ),
            strict=True,
        )
    }

    print(f"\n{name.upper()}")
    print(f"Observed rows: {frame.height:,}")
    print(f"Classes:       {len(class_counts):,}")
    print(f"Cities:        {len(city_counts):,}")

    return {
        "rows": float(frame.height),
        "class_count": float(len(class_counts)),
        "city_count": float(len(city_counts)),
        "class_counts": class_counts,
        "city_counts": city_counts,
    }


def _build_label_mapping(
    train_frame: pl.DataFrame,
    *,
    target_column: str,
    configured_classes: list[str] | None = None,
) -> tuple[list[str], dict[str, int]]:
    if configured_classes:
        classes = _unique(sorted(str(value) for value in configured_classes))
    else:
        classes = sorted(
            str(value)
            for value in train_frame.get_column(target_column).drop_nulls().unique().to_list()
        )

    if len(classes) < 2:
        raise ValueError("Multiclass mark classifier requires at least two classes")

    class_to_index = {label: index for index, label in enumerate(classes)}
    return classes, class_to_index


def _prepare_xy(
    frame: pl.DataFrame,
    *,
    feature_columns: list[str],
    categorical_columns: list[str],
    target_column: str,
    class_to_index: dict[str, int],
    category_levels: dict[str, list[str]] | None = None,
    unseen_label_policy: str = "fail",
) -> tuple[pd.DataFrame, np.ndarray, dict[str, list[str]]]:
    raw_target = frame.get_column(target_column).cast(pl.String).to_list()
    unknown_labels = sorted({label for label in raw_target if label not in class_to_index})

    if unknown_labels:
        if unseen_label_policy != "fail":
            raise ValueError(
                "Only unseen_validation_label_policy='fail' is supported for the "
                "production geographic-CV mark model."
            )
        raise ValueError(
            "Validation contains target classes absent from the training/configured "
            f"mark vocabulary: {unknown_labels}"
        )

    y = np.asarray([class_to_index[label] for label in raw_target], dtype=np.int32)
    X = frame.select(feature_columns).to_pandas()

    learned_categories: dict[str, list[str]] = {}
    for column in categorical_columns:
        if category_levels is None:
            categories = sorted(X[column].dropna().astype(str).unique().tolist())
        else:
            categories = list(category_levels[column])
        learned_categories[column] = categories
        X[column] = pd.Categorical(X[column], categories=categories)

    for column in feature_columns:
        if column in categorical_columns:
            continue
        X[column] = pd.to_numeric(X[column], errors="coerce").astype(np.float32)

    return X, y, learned_categories


def _validate_labels(*, name: str, y: np.ndarray, num_classes: int) -> None:
    if y.ndim != 1 or y.size == 0:
        raise ValueError(f"{name}: expected a non-empty 1-D label array")
    if not np.isfinite(y).all():
        raise ValueError(f"{name}: labels contain NaN or infinity")
    if int(y.min()) < 0 or int(y.max()) >= int(num_classes):
        raise ValueError(f"{name}: labels are outside [0, {num_classes - 1}]")


def _class_priors(y: np.ndarray, *, num_classes: int) -> np.ndarray:
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    total = float(counts.sum())
    if total <= 0:
        raise ValueError("Cannot compute mark priors from an empty target")
    return counts / total


def _prior_log_loss(y: np.ndarray, *, priors: np.ndarray) -> float:
    probability = np.clip(priors[y], 1e-15, 1.0)
    return float(-np.mean(np.log(probability)))


def _as_probability_matrix(predt: np.ndarray, *, num_classes: int) -> np.ndarray:
    probability = np.asarray(predt, dtype=np.float64)
    if probability.ndim == 1:
        if probability.size % num_classes != 0:
            raise ValueError(
                f"Cannot reshape prediction vector of size={probability.size} "
                f"into num_classes={num_classes}"
            )
        probability = probability.reshape(-1, num_classes)
    if probability.ndim != 2 or probability.shape[1] != num_classes:
        raise ValueError(
            f"Expected probability matrix [N,{num_classes}], got {probability.shape}"
        )
    probability = np.clip(probability, 1e-15, 1.0)
    row_sum = probability.sum(axis=1, keepdims=True)
    if not np.isfinite(row_sum).all() or np.any(row_sum <= 0):
        raise ValueError("Multiclass predictions contain invalid probability mass")
    return probability / row_sum


def _log_loss(y: np.ndarray, probability: np.ndarray) -> float:
    return float(-np.mean(np.log(probability[np.arange(y.size), y])))


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, *, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)
    return matrix


def _f1_from_confusion(matrix: np.ndarray) -> tuple[float, float, np.ndarray]:
    tp = np.diag(matrix).astype(np.float64)
    actual = matrix.sum(axis=1).astype(np.float64)
    predicted = matrix.sum(axis=0).astype(np.float64)
    denominator = (2.0 * tp) + (predicted - tp) + (actual - tp)
    f1 = np.divide(2.0 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)

    supported = actual > 0
    macro = float(f1[supported].mean()) if supported.any() else 0.0
    weighted = float(np.sum(f1 * actual) / max(float(actual.sum()), 1.0))
    return macro, weighted, f1


def _expected_calibration_error(
    y: np.ndarray,
    probability: np.ndarray,
    *,
    bins: int = 15,
) -> float:
    predicted = probability.argmax(axis=1)
    confidence = probability.max(axis=1)
    correct = (predicted == y).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = max(y.size, 1)
    ece = 0.0
    for index in range(bins):
        lower = edges[index]
        upper = edges[index + 1]
        if index == bins - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        if not mask.any():
            continue
        weight = float(mask.sum()) / total
        ece += weight * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(ece)


def _metrics_from_probability(
    y: np.ndarray,
    probability: np.ndarray,
    *,
    num_classes: int,
) -> tuple[dict[str, float], np.ndarray]:
    predicted = probability.argmax(axis=1).astype(np.int64, copy=False)
    confusion = _confusion_matrix(y, predicted, num_classes=num_classes)
    macro_f1, weighted_f1, _ = _f1_from_confusion(confusion)

    k3 = min(3, num_classes)
    k5 = min(5, num_classes)
    order = np.argpartition(probability, kth=max(num_classes - k5, 0), axis=1)
    top5 = order[:, -k5:]
    if k3 == k5:
        top3 = top5
    else:
        top3 = top5[
            np.arange(y.size)[:, None],
            np.argpartition(
                probability[np.arange(y.size)[:, None], top5],
                kth=max(k5 - k3, 0),
                axis=1,
            )[:, -k3:],
        ]

    p_true = probability[np.arange(y.size), y]
    multiclass_brier = float(np.mean(np.sum(probability * probability, axis=1) - 2.0 * p_true + 1.0))

    metrics = {
        "rows": float(y.size),
        "log_loss": _log_loss(y, probability),
        "accuracy": float(np.mean(predicted == y)),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "top3_accuracy": float(np.mean(np.any(top3 == y[:, None], axis=1))),
        "top5_accuracy": float(np.mean(np.any(top5 == y[:, None], axis=1))),
        "mean_confidence": float(probability.max(axis=1).mean()),
        "multiclass_brier": multiclass_brier,
        "ece_15": _expected_calibration_error(y, probability, bins=15),
    }
    return metrics, confusion


def _geographic_report(
    *,
    y: np.ndarray,
    probability: np.ndarray,
    source_city: np.ndarray,
    classes: list[str],
) -> tuple[dict[str, Any], np.ndarray]:
    num_classes = len(classes)
    global_metrics, confusion = _metrics_from_probability(
        y,
        probability,
        num_classes=num_classes,
    )

    city_rows: list[dict[str, Any]] = []
    for city in sorted({str(value) for value in source_city.tolist()}):
        mask = source_city == city
        city_metrics, _ = _metrics_from_probability(
            y[mask],
            probability[mask],
            num_classes=num_classes,
        )
        city_rows.append({"source_city": city, **city_metrics})

    if not city_rows:
        raise RuntimeError("No cities are present in the mark validation frame")

    numeric_metric_names = [
        "log_loss",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "top3_accuracy",
        "top5_accuracy",
        "mean_confidence",
        "multiclass_brier",
        "ece_15",
    ]
    macro_city = {
        key: float(np.mean([float(row[key]) for row in city_rows]))
        for key in numeric_metric_names
    }
    macro_city["city_count"] = float(len(city_rows))

    return {
        "global": global_metrics,
        "macro_city": macro_city,
        "per_city": city_rows,
    }, confusion


def _macro_city_log_loss(
    *,
    y: np.ndarray,
    probability: np.ndarray,
    source_city: np.ndarray,
) -> float:
    losses: list[float] = []
    for city in sorted({str(value) for value in source_city.tolist()}):
        mask = source_city == city
        if not mask.any():
            continue
        losses.append(_log_loss(y[mask], probability[mask]))
    if not losses:
        raise RuntimeError("Cannot compute macro-city mark log loss without validation cities")
    return float(np.mean(losses))


def _assert_snapshot_identity(table_ref: Any, data_cfg: dict[str, Any]) -> None:
    expected_id = data_cfg.get("final_model_snapshot_id")
    expected_uri = data_cfg.get("final_model_snapshot_uri")
    actual_id = getattr(table_ref, "snapshot_id", None)
    actual_uri = getattr(table_ref, "snapshot_uri", None)

    if expected_id is not None and actual_id is not None and str(actual_id) != str(expected_id):
        raise RuntimeError(
            f"Model-table snapshot changed: expected={expected_id!r}, actual={actual_id!r}"
        )
    if expected_uri is not None and actual_uri is not None:
        if str(actual_uri).rstrip("/") != str(expected_uri).rstrip("/"):
            raise RuntimeError("Canonical model-table URI changed between selection and training")


def _resolve_table_for_training(data_cfg: dict[str, Any]):
    """Resolve through the symbol xgb_hpo.py monkey-patches inside cached workers."""

    table_ref = resolve_model_table(
        snapshot_override_uri=(
            data_cfg.get("final_model_snapshot_uri")
            or data_cfg.get("snapshot_override_uri")
        ),
        local_root=data_cfg.get("local_snapshot_root"),
    )
    _assert_snapshot_identity(table_ref, data_cfg)
    return table_ref


def _class_metrics_json(classes: list[str], confusion: np.ndarray) -> dict[str, Any]:
    tp = np.diag(confusion).astype(np.float64)
    actual = confusion.sum(axis=1).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, actual, out=np.zeros_like(tp), where=actual > 0)
    denom = precision + recall
    f1 = np.divide(2.0 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)

    return {
        classes[index]: {
            "support": int(actual[index]),
            "predicted": int(predicted[index]),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
        }
        for index in range(len(classes))
    }



def _build_quantile_matrices(
    *,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_validation: pd.DataFrame,
    y_validation: np.ndarray,
    max_bin: int,
) -> tuple[xgb.QuantileDMatrix, xgb.QuantileDMatrix]:
    """Build quantized matrices; HPO may cache these per geographic fold/max_bin."""

    dtrain = xgb.QuantileDMatrix(
        X_train,
        label=y_train,
        enable_categorical=True,
        max_bin=int(max_bin),
        nthread=-1,
    )
    dvalidation = xgb.QuantileDMatrix(
        X_validation,
        label=y_validation,
        enable_categorical=True,
        max_bin=int(max_bin),
        ref=dtrain,
        nthread=-1,
    )
    return dtrain, dvalidation


def train(
    config: dict[str, Any],
    *,
    run_id: str,
    config_hash: str,
) -> dict[str, Any]:
    """Train p(mark | event, t, s, X) on observed events only.

    During HPO, five geographic folds train on 12 cities and evaluate three held-out
    cities. Final production fitting uses all cities and a fixed round count chosen
    by HPO. The test split is never resolved or scanned by this module.
    """

    model_cfg = config["model"]
    data_cfg = config["data"]
    target_cfg = config.get("target", {})
    arch_cfg = config["architecture"]
    opt_cfg = config["optimization"]
    training_cfg = config["training"]
    artifact_cfg = config["artifacts"]
    validation_cfg = config.get("validation", {})

    seed = int(data_cfg["seed"])
    train_fraction = float(data_cfg["train_fraction"])
    validation_fraction = float(data_cfg["validation_fraction"])
    train_split = str(data_cfg.get("train_split", "train"))
    validation_split = str(data_cfg.get("validation_split", "validation"))
    target_column = str(target_cfg.get("column", DEFAULT_TARGET_COLUMN))
    unseen_policy = str(target_cfg.get("unseen_validation_label_policy", "fail"))
    configured_classes = target_cfg.get("classes")
    if configured_classes is not None:
        configured_classes = [str(value) for value in configured_classes]

    hpo_mode = bool(config.get("hpo_runtime", {}).get("enabled", False))
    fixed_rounds = bool(training_cfg.get("fixed_num_boost_round", False))

    table_ref = _resolve_table_for_training(data_cfg)
    train_lazy = table_ref.scan_split(train_split)
    validation_lazy = table_ref.scan_split(validation_split)

    available_columns = train_lazy.collect_schema().names()
    feature_columns, categorical_columns = _resolve_feature_columns(
        config,
        available_columns=available_columns,
    )
    if target_column in feature_columns:
        raise RuntimeError(f"Mark target {target_column!r} must not be used as a feature")

    print("Loading deterministic observed-event training sample...")
    train_frame = _deterministic_observed_sample(
        table=train_lazy,
        split=train_split,
        fraction=train_fraction,
        seed=seed,
        feature_columns=feature_columns,
        target_column=target_column,
    )
    train_frame = _apply_geographic_role(train_frame, config=config, role="train")

    print("Loading deterministic observed-event validation sample...")
    validation_frame = _deterministic_observed_sample(
        table=validation_lazy,
        split=validation_split,
        fraction=validation_fraction,
        seed=seed,
        feature_columns=feature_columns,
        target_column=target_column,
    )
    validation_frame = _apply_geographic_role(
        validation_frame,
        config=config,
        role="validation",
    )

    train_summary = _sample_summary("train", train_frame, target_column=target_column)
    validation_summary = _sample_summary(
        "validation",
        validation_frame,
        target_column=target_column,
    )

    classes, class_to_index = _build_label_mapping(
        train_frame,
        target_column=target_column,
        configured_classes=configured_classes,
    )
    num_classes = len(classes)

    validation_cities = (
        validation_frame.get_column("source_city").cast(pl.String).to_numpy()
    )
    train_cities = train_frame.get_column("source_city").cast(pl.String).to_numpy()

    X_train, y_train, category_levels = _prepare_xy(
        train_frame,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        target_column=target_column,
        class_to_index=class_to_index,
        unseen_label_policy=unseen_policy,
    )
    X_validation, y_validation, _ = _prepare_xy(
        validation_frame,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        target_column=target_column,
        class_to_index=class_to_index,
        category_levels=category_levels,
        unseen_label_policy=unseen_policy,
    )

    _validate_labels(name="train", y=y_train, num_classes=num_classes)
    _validate_labels(name="validation", y=y_validation, num_classes=num_classes)

    train_priors = _class_priors(y_train, num_classes=num_classes)
    baseline_train_log_loss = _prior_log_loss(y_train, priors=train_priors)
    baseline_validation_log_loss = _prior_log_loss(y_validation, priors=train_priors)

    print("\nEmpirical-prior baseline")
    print(f"Train log loss:      {baseline_train_log_loss:.6f}")
    print(f"Validation log loss: {baseline_validation_log_loss:.6f}")
    print(f"Mark classes:        {num_classes}")
    print(f"Features:            {len(feature_columns)}")

    max_bin = int(arch_cfg["max_bin"])
    dtrain, dvalidation = _build_quantile_matrices(
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        max_bin=max_bin,
    )

    del X_train, X_validation, train_frame, validation_frame
    gc.collect()

    city_lookup = {
        id(dtrain): train_cities,
        id(dvalidation): validation_cities,
    }

    def mark_metrics(predt: np.ndarray, dmatrix: xgb.DMatrix):
        y = dmatrix.get_label().astype(np.int64, copy=False)
        probability = _as_probability_matrix(predt, num_classes=num_classes)
        pooled = _log_loss(y, probability)
        cities = city_lookup[id(dmatrix)]
        macro = _macro_city_log_loss(y=y, probability=probability, source_city=cities)
        return [
            ("mark_log_loss", pooled),
            ("mark_macro_city_log_loss", macro),
        ]

    params = {
        "objective": "multi:softprob",
        "num_class": num_classes,
        "tree_method": arch_cfg["tree_method"],
        "device": arch_cfg["device"],
        "max_bin": max_bin,
        "max_depth": int(arch_cfg["max_depth"]),
        "max_cat_to_onehot": int(arch_cfg["max_cat_to_onehot"]),
        "eta": float(opt_cfg["learning_rate"]),
        "subsample": float(opt_cfg["subsample"]),
        "colsample_bytree": float(opt_cfg["colsample_bytree"]),
        "min_child_weight": float(opt_cfg["min_child_weight"]),
        "reg_lambda": float(opt_cfg["reg_lambda"]),
        "reg_alpha": float(opt_cfg.get("reg_alpha", 0.0)),
        "gamma": float(opt_cfg.get("gamma", 0.0)),
        "seed": seed,
        "nthread": -1,
        "disable_default_eval_metric": True,
    }

    callbacks: list[Any] = []
    evals: list[tuple[xgb.DMatrix, str]] = []
    if not fixed_rounds:
        evals = [(dvalidation, "validation")]
        callbacks.append(
            xgb.callback.EarlyStopping(
                rounds=int(training_cfg["early_stopping_rounds"]),
                metric_name="mark_macro_city_log_loss",
                data_name="validation",
                maximize=False,
                save_best=True,
            )
        )

    print("\nTraining XGBoost conditional mark classifier...\n")
    evals_result: dict[str, dict[str, list[float]]] = {}
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=int(training_cfg["num_boost_round"]),
        evals=evals,
        custom_metric=mark_metrics if evals else None,
        callbacks=callbacks,
        evals_result=evals_result,
        verbose_eval=(training_cfg.get("verbose_eval", False) if evals else False),
    )

    if fixed_rounds:
        best_iteration = int(training_cfg["num_boost_round"]) - 1
    else:
        raw_best = getattr(booster, "best_iteration", None)
        best_iteration = int(raw_best) if raw_best is not None else booster.num_boosted_rounds() - 1

    validation_probability = _as_probability_matrix(
        booster.predict(dvalidation),
        num_classes=num_classes,
    )
    geographic_validation, validation_confusion = _geographic_report(
        y=y_validation,
        probability=validation_probability,
        source_city=validation_cities,
        classes=classes,
    )
    validation_metrics = geographic_validation["global"]
    macro_mark_log_loss = float(geographic_validation["macro_city"]["log_loss"])

    validation_log_loss_gain = baseline_validation_log_loss - float(
        validation_metrics["log_loss"]
    )
    validation_bits_gain = validation_log_loss_gain / np.log(2.0)

    print("\nMARK VALIDATION")
    print(f"Global log loss:     {validation_metrics['log_loss']:.6f}")
    print(f"Macro-city log loss: {macro_mark_log_loss:.6f}")
    print(f"Top-3 accuracy:      {validation_metrics['top3_accuracy']:.4f}")
    print(f"Top-5 accuracy:      {validation_metrics['top5_accuracy']:.4f}")
    print(f"Best iteration:      {best_iteration}")

    metrics: dict[str, float] = {
        "sample_train_rows": float(train_summary["rows"]),
        "sample_validation_rows": float(validation_summary["rows"]),
        "num_classes": float(num_classes),
        "baseline_train_log_loss": float(baseline_train_log_loss),
        "baseline_validation_log_loss": float(baseline_validation_log_loss),
        "sample_validation_log_loss": float(validation_metrics["log_loss"]),
        "sample_validation_accuracy": float(validation_metrics["accuracy"]),
        "sample_validation_macro_f1": float(validation_metrics["macro_f1"]),
        "sample_validation_weighted_f1": float(validation_metrics["weighted_f1"]),
        "sample_validation_top3_accuracy": float(validation_metrics["top3_accuracy"]),
        "sample_validation_top5_accuracy": float(validation_metrics["top5_accuracy"]),
        "sample_validation_mean_confidence": float(validation_metrics["mean_confidence"]),
        "sample_validation_multiclass_brier": float(validation_metrics["multiclass_brier"]),
        "sample_validation_ece_15": float(validation_metrics["ece_15"]),
        "sample_validation_log_loss_gain": float(validation_log_loss_gain),
        "sample_validation_bits_gain": float(validation_bits_gain),
        "geocv_macro_mark_log_loss": macro_mark_log_loss,
        "best_iteration": float(best_iteration),
    }

    # HPO only needs the lightweight metrics/report. Avoid model serialization,
    # train-set prediction, class diagnostics, and confusion CSV for every trial.
    if hpo_mode:
        print("TEST SPLIT HAS NOT BEEN ACCESSED.")
        return {
            "metrics": metrics,
            "history": evals_result,
            "artifacts": [],
            "summary": {
                "best_iteration": best_iteration,
                "feature_count": len(feature_columns),
                "num_classes": num_classes,
                "target_column": target_column,
                "test_split_used": False,
            },
            "geographic_validation": geographic_validation,
        }

    artifact_dir = (
        Path(artifact_cfg["output_root"]) / str(model_cfg["name"]) / str(run_id)
    )
    artifact_dir.mkdir(parents=True, exist_ok=False)
    model_path = artifact_dir / "model.json"
    metadata_path = artifact_dir / "metadata.json"
    importance_path = artifact_dir / "feature_importance.json"
    training_history_path = artifact_dir / "training_history.json"
    class_metrics_path = artifact_dir / "class_metrics.json"
    confusion_path = artifact_dir / "validation_confusion_matrix.csv"
    geographic_path = artifact_dir / "geographic_validation.json"

    booster.save_model(model_path)
    importance = {
        name: booster.get_score(importance_type=name)
        for name in ("gain", "weight", "cover")
    }

    metadata = {
        "model_name": model_cfg["name"],
        "model_type": "xgboost_multiclass_mark_classifier",
        "factorization": "lambda_m(t,s) = lambda_total(t,s) * p(m|event,t,s)",
        "run_id": run_id,
        "config_hash": config_hash,
        "snapshot_id": getattr(table_ref, "snapshot_id", data_cfg.get("final_model_snapshot_id")),
        "snapshot_uri": getattr(table_ref, "snapshot_uri", data_cfg.get("final_model_snapshot_uri")),
        "schema_version": getattr(table_ref, "schema_version", data_cfg.get("final_model_schema_version")),
        "feature_contract_hash": config.get("features", {}).get("feature_contract_hash"),
        "feature_set": config.get("features", {}).get("feature_set"),
        "zero_shot_geography": bool(config.get("features", {}).get("zero_shot_geography", False)),
        "features": feature_columns,
        "categorical_columns": categorical_columns,
        "category_levels": category_levels,
        "target_column": target_column,
        "classes": classes,
        "class_to_index": class_to_index,
        "num_classes": num_classes,
        "training_class_priors": {
            classes[index]: float(train_priors[index]) for index in range(num_classes)
        },
        "train_split": train_split,
        "validation_split": validation_split,
        "train_fraction": train_fraction,
        "validation_fraction": validation_fraction,
        "seed": seed,
        "best_iteration": best_iteration,
        "num_boosted_rounds": booster.num_boosted_rounds(),
        "fixed_num_boost_round": fixed_rounds,
        "test_split_used": False,
    }

    class_metrics = _class_metrics_json(classes, validation_confusion)
    pd.DataFrame(
        validation_confusion,
        index=classes,
        columns=classes,
    ).rename_axis("actual").to_csv(confusion_path)

    for path, value in (
        (metadata_path, metadata),
        (importance_path, importance),
        (training_history_path, evals_result),
        (class_metrics_path, class_metrics),
        (geographic_path, geographic_validation),
    ):
        path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")

    print(f"\nSaved model: {model_path}")
    print("TEST SPLIT HAS NOT BEEN ACCESSED.")

    artifacts = [
        str(model_path),
        str(metadata_path),
        str(importance_path),
        str(training_history_path),
        str(class_metrics_path),
        str(confusion_path),
        str(geographic_path),
    ]
    return {
        "metrics": metrics,
        "history": evals_result,
        "artifacts": artifacts,
        "summary": {
            "best_iteration": best_iteration,
            "feature_count": len(feature_columns),
            "num_classes": num_classes,
            "target_column": target_column,
            "artifact_dir": str(artifact_dir),
            "test_split_used": False,
        },
        "geographic_validation": geographic_validation,
    }
