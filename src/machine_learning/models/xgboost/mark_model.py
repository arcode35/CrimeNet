from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb

from crimenet_data.assets.model_table.transformations import (
    CONTEXT_FEATURE_COLUMNS,
    HISTORY_FEATURE_COLUMNS,
    LIGHTING_FEATURE_COLUMNS,
)


MACHINE_LEARNING_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


CALENDAR_FEATURE_COLUMNS = [
    "local_hour",
    "local_day_of_week",
    "local_hour_sin",
    "local_hour_cos",
    "local_day_of_week_sin",
    "local_day_of_week_cos",
]


DEFAULT_TARGET_COLUMN = (
    "canonical_subtype_code"
)


def _resolve_feature_columns(
    config: dict[str, Any],
) -> tuple[list[str], list[str]]:
    feature_config = config["features"]

    feature_columns: list[str] = []

    if feature_config.get(
        "include_source_city",
        True,
    ):
        feature_columns.append(
            "source_city"
        )

    if feature_config.get(
        "include_calendar",
        True,
    ):
        feature_columns.extend(
            CALENDAR_FEATURE_COLUMNS
        )

    if feature_config.get(
        "include_context",
        True,
    ):
        feature_columns.extend(
            CONTEXT_FEATURE_COLUMNS
        )

    if feature_config.get(
        "include_lighting",
        True,
    ):
        feature_columns.extend(
            LIGHTING_FEATURE_COLUMNS
        )

    if feature_config.get(
        "include_history",
        True,
    ):
        feature_columns.extend(
            HISTORY_FEATURE_COLUMNS
        )

    feature_columns = list(
        dict.fromkeys(
            feature_columns
        )
    )

    categorical_columns = [
        column
        for column in (
            "source_city",
            "lighting_condition",
        )
        if column in feature_columns
    ]

    return (
        feature_columns,
        categorical_columns,
    )


def _deterministic_observed_sample(
    *,
    table: pl.LazyFrame,
    split: str,
    fraction: float,
    seed: int,
    feature_columns: list[str],
    target_column: str,
) -> pl.DataFrame:
    if not (
        0.0
        < fraction
        <= 1.0
    ):
        raise ValueError(
            "Sampling fraction must be in "
            f"(0, 1], got {fraction}."
        )

    buckets = 1_000_000

    threshold = int(
        fraction
        * buckets
    )

    return (
        table

        .filter(
            pl.col("split")
            == split
        )

        # Mark probabilities are conditioned on
        # an event actually occurring.
        .filter(
            pl.col(
                "is_observed_event"
            )
        )

        .filter(
            pl.col("event_count")
            == 1
        )

        .filter(
            pl.col(
                target_column
            )
            .is_not_null()
        )

        .filter(
            (
                pl.col(
                    "model_row_id"
                )
                .hash(
                    seed=seed
                )
                % buckets
            )
            < threshold
        )

        .select(
            [
                *feature_columns,
                target_column,
            ]
        )

        .collect()
    )


def _sample_summary(
    name: str,
    frame: pl.DataFrame,
    *,
    target_column: str,
) -> dict[str, Any]:
    rows = int(
        frame.height
    )

    if rows == 0:
        raise ValueError(
            f"{name}: no observed "
            "event rows were loaded."
        )

    counts_frame = (
        frame
        .group_by(
            target_column
        )
        .len()
        .sort(
            "len",
            descending=True,
        )
    )

    class_counts = {
        str(mark): int(count)
        for mark, count
        in zip(
            counts_frame[
                target_column
            ].to_list(),
            counts_frame[
                "len"
            ].to_list(),
            strict=True,
        )
    }

    print(
        f"\n{name.upper()}"
    )

    print(
        f"Observed rows: "
        f"{rows:,}"
    )

    print(
        f"Classes:       "
        f"{len(class_counts):,}"
    )

    print(
        "\nClass distribution:"
    )

    for (
        label,
        count,
    ) in class_counts.items():
        share = (
            count
            / rows
        )

        print(
            f"  {label:<30} "
            f"{count:>12,} "
            f"({share:>7.3%})"
        )

    return {
        "rows":
            float(rows),

        "class_count":
            float(
                len(class_counts)
            ),

        "class_counts":
            class_counts,
    }


def _build_label_mapping(
    train_frame: pl.DataFrame,
    *,
    target_column: str,
) -> tuple[
    list[str],
    dict[str, int],
]:
    classes = sorted(
        str(value)
        for value
        in train_frame[
            target_column
        ]
        .drop_nulls()
        .unique()
        .to_list()
    )

    if len(classes) < 2:
        raise ValueError(
            "Multiclass classifier "
            "requires at least two "
            "training classes."
        )

    class_to_index = {
        label: index
        for index, label
        in enumerate(
            classes
        )
    }

    return (
        classes,
        class_to_index,
    )


def _prepare_xy(
    frame: pl.DataFrame,
    *,
    feature_columns: list[str],
    categorical_columns: list[str],
    target_column: str,
    class_to_index: dict[str, int],
    category_levels:
        dict[str, list[str]]
        | None = None,
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    dict[str, list[str]],
]:
    raw_target = (
        frame
        .get_column(
            target_column
        )
        .cast(
            pl.String
        )
        .to_list()
    )

    unknown_labels = sorted(
        {
            label
            for label
            in raw_target
            if label
            not in class_to_index
        }
    )

    if unknown_labels:
        raise ValueError(
            "Found target classes outside "
            "the training label vocabulary: "
            f"{unknown_labels}"
        )

    y = np.asarray(
        [
            class_to_index[
                label
            ]
            for label
            in raw_target
        ],
        dtype=np.int32,
    )

    X = (
        frame
        .select(
            feature_columns
        )
        .to_pandas()
    )

    learned_categories: dict[
        str,
        list[str],
    ] = {}

    for column in categorical_columns:
        if category_levels is None:
            categories = sorted(
                X[column]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

        else:
            categories = (
                category_levels[
                    column
                ]
            )

        learned_categories[
            column
        ] = categories

        X[column] = pd.Categorical(
            X[column],
            categories=categories,
        )

    for column in feature_columns:
        if column in categorical_columns:
            continue

        X[column] = (
            pd.to_numeric(
                X[column],
                errors="coerce",
            )
            .astype(
                np.float32
            )
        )

    return (
        X,
        y,
        learned_categories,
    )


def _validate_labels(
    *,
    name: str,
    y: np.ndarray,
    num_classes: int,
) -> None:
    if y.ndim != 1:
        raise ValueError(
            f"{name}: expected 1-D "
            "label array."
        )

    if y.size == 0:
        raise ValueError(
            f"{name}: no labels."
        )

    if (
        y.min() < 0
        or y.max()
        >= num_classes
    ):
        raise ValueError(
            f"{name}: labels outside "
            f"[0, {num_classes - 1}]."
        )


def _class_priors(
    y: np.ndarray,
    *,
    num_classes: int,
) -> np.ndarray:
    counts = np.bincount(
        y,
        minlength=num_classes,
    ).astype(
        np.float64
    )

    total = float(
        counts.sum()
    )

    if total <= 0:
        raise ValueError(
            "Cannot compute class priors "
            "from empty target."
        )

    # These are empirical training priors.
    # No balancing weights are used because
    # this model is estimating the actual
    # conditional mark probability.
    priors = (
        counts
        / total
    )

    return priors


def _prior_log_loss(
    y: np.ndarray,
    *,
    priors: np.ndarray,
) -> float:
    eps = 1e-15

    probability = (
        priors[
            y
        ]
    )

    probability = np.clip(
        probability,
        eps,
        1.0,
    )

    return float(
        -np.mean(
            np.log(
                probability
            )
        )
    )


def _confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    num_classes: int,
) -> np.ndarray:
    matrix = np.zeros(
        (
            num_classes,
            num_classes,
        ),
        dtype=np.int64,
    )

    np.add.at(
        matrix,
        (
            y_true,
            y_pred,
        ),
        1,
    )

    return matrix


def _evaluate_split(
    *,
    name: str,
    booster: xgb.Booster,
    dmatrix: xgb.DMatrix,
    classes: list[str],
) -> tuple[
    dict[str, float],
    dict[str, dict[str, float]],
    np.ndarray,
]:
    y = (
        dmatrix
        .get_label()
        .astype(
            np.int64,
            copy=False,
        )
    )

    probabilities = (
        booster.predict(
            dmatrix
        )
    )

    num_classes = len(
        classes
    )

    if probabilities.ndim == 1:
        probabilities = (
            probabilities.reshape(
                -1,
                num_classes,
            )
        )

    if probabilities.shape != (
        y.shape[0],
        num_classes,
    ):
        raise ValueError(
            f"{name}: unexpected "
            "probability shape "
            f"{probabilities.shape}; "
            "expected "
            f"({y.shape[0]}, "
            f"{num_classes})."
        )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    eps = 1e-15

    row_sums = probabilities.sum(
        axis=1,
        keepdims=True,
    )

    if not np.isfinite(
        probabilities
    ).all():
        raise ValueError(
            f"{name}: predictions "
            "contain NaN or infinity."
        )

    if (
        row_sums <= 0
    ).any():
        raise ValueError(
            f"{name}: invalid "
            "probability rows."
        )

    # Guard against tiny floating-point
    # departures from exactly summing to 1.
    probabilities = (
        probabilities
        / row_sums
    )

    true_probability = (
        probabilities[
            np.arange(
                y.shape[0]
            ),
            y,
        ]
    )

    log_loss = float(
        -np.mean(
            np.log(
                np.clip(
                    true_probability,
                    eps,
                    1.0,
                )
            )
        )
    )

    predicted_class = (
        probabilities.argmax(
            axis=1
        )
    )

    accuracy = float(
        np.mean(
            predicted_class
            == y
        )
    )

    matrix = _confusion_matrix(
        y,
        predicted_class,
        num_classes=num_classes,
    )

    per_class: dict[
        str,
        dict[str, float],
    ] = {}

    f1_values: list[float] = []

    for class_index, class_name in enumerate(
        classes
    ):
        tp = float(
            matrix[
                class_index,
                class_index,
            ]
        )

        fp = float(
            matrix[
                :,
                class_index,
            ].sum()
            - tp
        )

        fn = float(
            matrix[
                class_index,
                :,
            ].sum()
            - tp
        )

        support = float(
            matrix[
                class_index,
                :
            ].sum()
        )

        precision = (
            tp
            / max(
                tp + fp,
                1.0,
            )
        )

        recall = (
            tp
            / max(
                tp + fn,
                1.0,
            )
        )

        if (
            precision
            + recall
            > 0
        ):
            f1 = (
                2.0
                * precision
                * recall
                / (
                    precision
                    + recall
                )
            )
        else:
            f1 = 0.0

        f1_values.append(
            f1
        )

        per_class[
            class_name
        ] = {
            "precision":
                float(
                    precision
                ),

            "recall":
                float(
                    recall
                ),

            "f1":
                float(
                    f1
                ),

            "support":
                support,
        }

    macro_f1 = float(
        np.mean(
            f1_values
        )
    )

    mean_confidence = float(
        probabilities.max(
            axis=1
        ).mean()
    )

    metrics = {
        "rows":
            float(
                y.shape[0]
            ),

        "log_loss":
            log_loss,

        "accuracy":
            accuracy,

        "macro_f1":
            macro_f1,

        "mean_confidence":
            mean_confidence,
    }

    print(
        f"\n{name.upper()}"
    )

    print(
        f"Rows:            "
        f"{y.shape[0]:,}"
    )

    print(
        f"Log loss:        "
        f"{log_loss:.6f}"
    )

    print(
        f"Accuracy:        "
        f"{accuracy:.6f}"
    )

    print(
        f"Macro F1:        "
        f"{macro_f1:.6f}"
    )

    print(
        f"Mean confidence: "
        f"{mean_confidence:.6f}"
    )

    return (
        metrics,
        per_class,
        matrix,
    )


def train(
    config: dict[str, Any],
    *,
    run_id: str,
    config_hash: str,
) -> dict[str, Any]:
    """
    Train the conditional XGBoost CrimeNet
    offense-mark classifier.

    This estimates

        p(m | event occurs, t, s, H_t, X_t)

    using observed-event rows only.

    Combined with the separately trained
    point-process intensity model,

        lambda_m(t, s)
            = lambda(t, s) * p(m | event, t, s)

    defines the factorized marked
    point-process baseline.

    This function deliberately does not
    create or manipulate MLflow runs.
    """

    model_config = config[
        "model"
    ]

    data_config = config[
        "data"
    ]

    target_config = config.get(
        "target",
        {},
    )

    architecture_config = config[
        "architecture"
    ]

    optimization_config = config[
        "optimization"
    ]

    training_config = config[
        "training"
    ]

    artifact_config = config[
        "artifacts"
    ]

    seed = int(
        data_config[
            "seed"
        ]
    )

    train_fraction = float(
        data_config[
            "train_fraction"
        ]
    )

    validation_fraction = float(
        data_config[
            "validation_fraction"
        ]
    )

    target_column = str(
        target_config.get(
            "column",
            DEFAULT_TARGET_COLUMN,
        )
    )

    max_bin = int(
        architecture_config[
            "max_bin"
        ]
    )

    (
        feature_columns,
        categorical_columns,
    ) = _resolve_feature_columns(
        config
    )

    if not feature_columns:
        raise ValueError(
            "Resolved feature set "
            "is empty."
        )

    if target_column in feature_columns:
        raise ValueError(
            "Target column must not "
            "be included as a feature: "
            f"{target_column}"
        )

    artifact_dir = (
        MACHINE_LEARNING_ROOT
        / artifact_config[
            "output_root"
        ]
        / model_config[
            "name"
        ]
        / run_id
    )

    artifact_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    model_path = (
        artifact_dir
        / "model.json"
    )

    metadata_path = (
        artifact_dir
        / "metadata.json"
    )

    importance_path = (
        artifact_dir
        / "feature_importance.json"
    )

    training_history_path = (
        artifact_dir
        / "training_history.json"
    )

    class_metrics_path = (
        artifact_dir
        / "class_metrics.json"
    )

    confusion_matrix_path = (
        artifact_dir
        / "validation_confusion_matrix.csv"
    )

    print(
        "Experiment artifacts: "
        f"{artifact_dir}"
    )

    credentials = (
        pl.CredentialProviderGCP()
    )

    table = pl.scan_delta(
        data_config[
            "model_table_root"
        ],
        credential_provider=
            credentials,
    )

    print(
        "Loading deterministic "
        "observed-event training sample..."
    )

    train_frame = (
        _deterministic_observed_sample(
            table=table,

            split=data_config[
                "train_split"
            ],

            fraction=train_fraction,

            seed=seed,

            feature_columns=
                feature_columns,

            target_column=
                target_column,
        )
    )

    print(
        "Loading deterministic "
        "observed-event validation sample..."
    )

    validation_frame = (
        _deterministic_observed_sample(
            table=table,

            split=data_config[
                "validation_split"
            ],

            fraction=
                validation_fraction,

            seed=seed,

            feature_columns=
                feature_columns,

            target_column=
                target_column,
        )
    )

    train_summary = (
        _sample_summary(
            "train",
            train_frame,
            target_column=
                target_column,
        )
    )

    validation_summary = (
        _sample_summary(
            "validation",
            validation_frame,
            target_column=
                target_column,
        )
    )

    (
        classes,
        class_to_index,
    ) = _build_label_mapping(
        train_frame,
        target_column=
            target_column,
    )

    num_classes = len(
        classes
    )

    print(
        "\nMark vocabulary:"
    )

    for (
        index,
        label,
    ) in enumerate(
        classes
    ):
        print(
            f"  {index:>3}: "
            f"{label}"
        )

    (
        X_train,
        y_train,
        categories,
    ) = _prepare_xy(
        train_frame,

        feature_columns=
            feature_columns,

        categorical_columns=
            categorical_columns,

        target_column=
            target_column,

        class_to_index=
            class_to_index,
    )

    (
        X_validation,
        y_validation,
        _,
    ) = _prepare_xy(
        validation_frame,

        feature_columns=
            feature_columns,

        categorical_columns=
            categorical_columns,

        target_column=
            target_column,

        class_to_index=
            class_to_index,

        category_levels=
            categories,
    )

    del train_frame
    del validation_frame

    gc.collect()

    _validate_labels(
        name="train",
        y=y_train,
        num_classes=
            num_classes,
    )

    _validate_labels(
        name="validation",
        y=y_validation,
        num_classes=
            num_classes,
    )

    train_priors = _class_priors(
        y_train,
        num_classes=
            num_classes,
    )

    baseline_train_log_loss = (
        _prior_log_loss(
            y_train,
            priors=train_priors,
        )
    )

    baseline_validation_log_loss = (
        _prior_log_loss(
            y_validation,
            priors=train_priors,
        )
    )

    print(
        "\nEmpirical-prior baseline"
    )

    print(
        "Train log loss:      "
        f"{baseline_train_log_loss:.6f}"
    )

    print(
        "Validation log loss: "
        f"{baseline_validation_log_loss:.6f}"
    )

    dtrain = xgb.QuantileDMatrix(
        X_train,
        label=y_train,
        enable_categorical=True,
        max_bin=max_bin,
        nthread=-1,
    )

    dvalidation = (
        xgb.QuantileDMatrix(
            X_validation,
            label=y_validation,
            enable_categorical=True,
            max_bin=max_bin,
            ref=dtrain,
            nthread=-1,
        )
    )

    del X_train
    del X_validation

    gc.collect()

    params = {
        "objective":
            "multi:softprob",

        "eval_metric":
            "mlogloss",

        "num_class":
            num_classes,

        "tree_method":
            architecture_config[
                "tree_method"
            ],

        "device":
            architecture_config[
                "device"
            ],

        "max_bin":
            max_bin,

        "max_depth":
            int(
                architecture_config[
                    "max_depth"
                ]
            ),

        "eta":
            float(
                optimization_config[
                    "learning_rate"
                ]
            ),

        "subsample":
            float(
                optimization_config[
                    "subsample"
                ]
            ),

        "colsample_bytree":
            float(
                optimization_config[
                    "colsample_bytree"
                ]
            ),

        "min_child_weight":
            float(
                optimization_config[
                    "min_child_weight"
                ]
            ),

        "reg_lambda":
            float(
                optimization_config[
                    "reg_lambda"
                ]
            ),

        "reg_alpha":
            float(
                optimization_config[
                    "reg_alpha"
                ]
            ),

        "gamma":
            float(
                optimization_config.get(
                    "gamma",
                    0.0,
                )
            ),

        "max_cat_to_onehot":
            int(
                architecture_config[
                    "max_cat_to_onehot"
                ]
            ),

        "seed":
            seed,

        "nthread":
            -1,
    }

    print(
        "\nTraining XGBoost "
        "conditional mark classifier...\n"
    )

    evals_result: dict[
        str,
        dict[
            str,
            list[float],
        ],
    ] = {}

    booster = xgb.train(
        params=params,

        dtrain=dtrain,

        num_boost_round=int(
            training_config[
                "num_boost_round"
            ]
        ),

        evals=[
            (
                dtrain,
                "train",
            ),
            (
                dvalidation,
                "validation",
            ),
        ],

        early_stopping_rounds=int(
            training_config[
                "early_stopping_rounds"
            ]
        ),

        evals_result=
            evals_result,

        verbose_eval=
            training_config[
                "verbose_eval"
            ],
    )

    best_iteration = getattr(
        booster,
        "best_iteration",
        None,
    )

    if best_iteration is not None:
        booster = booster[
            : best_iteration + 1
        ]

    (
        train_metrics,
        train_class_metrics,
        train_confusion,
    ) = _evaluate_split(
        name="train",
        booster=booster,
        dmatrix=dtrain,
        classes=classes,
    )

    (
        validation_metrics,
        validation_class_metrics,
        validation_confusion,
    ) = _evaluate_split(
        name="validation",
        booster=booster,
        dmatrix=dvalidation,
        classes=classes,
    )

    train_log_loss_gain = (
        baseline_train_log_loss
        - train_metrics[
            "log_loss"
        ]
    )

    validation_log_loss_gain = (
        baseline_validation_log_loss
        - validation_metrics[
            "log_loss"
        ]
    )

    train_bits_gain = (
        train_log_loss_gain
        / np.log(2.0)
    )

    validation_bits_gain = (
        validation_log_loss_gain
        / np.log(2.0)
    )

    print(
        "\nLOG-LOSS IMPROVEMENT "
        "OVER EMPIRICAL PRIOR"
    )

    print(
        "Train gain:      "
        f"{train_log_loss_gain:.6f} "
        "nats/event "
        f"({train_bits_gain:.6f} bits)"
    )

    print(
        "Validation gain: "
        f"{validation_log_loss_gain:.6f} "
        "nats/event "
        f"({validation_bits_gain:.6f} bits)"
    )

    importance = {
        "gain":
            booster.get_score(
                importance_type="gain"
            ),

        "weight":
            booster.get_score(
                importance_type="weight"
            ),

        "cover":
            booster.get_score(
                importance_type="cover"
            ),
    }

    print(
        "\nTOP 30 FEATURES BY GAIN"
    )

    sorted_gain = sorted(
        importance[
            "gain"
        ].items(),
        key=lambda item:
            item[1],
        reverse=True,
    )

    for (
        feature,
        gain,
    ) in sorted_gain[:30]:
        print(
            f"{feature:<50} "
            f"{gain:,.6f}"
        )

    booster.save_model(
        model_path
    )

    metadata = {
        "model_name":
            model_config[
                "name"
            ],

        "model_type":
            "xgboost_multiclass_mark_classifier",

        "factorization":
            (
                "lambda_m(t,s) = "
                "lambda_total(t,s) * "
                "p(m|event,t,s)"
            ),

        "run_id":
            run_id,

        "config_hash":
            config_hash,

        "model_table_root":
            data_config[
                "model_table_root"
            ],

        "train_split":
            data_config[
                "train_split"
            ],

        "validation_split":
            data_config[
                "validation_split"
            ],

        "train_fraction":
            train_fraction,

        "validation_fraction":
            validation_fraction,

        "seed":
            seed,

        "target_column":
            target_column,

        "classes":
            classes,

        "class_to_index":
            class_to_index,

        "num_classes":
            num_classes,

        "training_class_priors": {
            classes[index]:
                float(
                    train_priors[
                        index
                    ]
                )
            for index
            in range(
                num_classes
            )
        },

        "features":
            feature_columns,

        "feature_set":
            config[
                "features"
            ].get(
                "feature_set"
            ),

        "categorical_columns":
            categorical_columns,

        "category_levels":
            categories,

        "best_iteration":
            best_iteration,

        "test_split_used":
            False,
    }

    with metadata_path.open(
        "w"
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    with importance_path.open(
        "w"
    ) as file:
        json.dump(
            importance,
            file,
            indent=2,
        )

    with training_history_path.open(
        "w"
    ) as file:
        json.dump(
            evals_result,
            file,
            indent=2,
        )

    class_metrics = {
        "train":
            train_class_metrics,

        "validation":
            validation_class_metrics,
    }

    with class_metrics_path.open(
        "w"
    ) as file:
        json.dump(
            class_metrics,
            file,
            indent=2,
        )

    confusion_frame = pd.DataFrame(
        validation_confusion,
        index=classes,
        columns=classes,
    )

    confusion_frame.index.name = (
        "actual"
    )

    confusion_frame.to_csv(
        confusion_matrix_path
    )

    print(
        f"\nSaved model: "
        f"{model_path}"
    )

    print(
        "TEST SPLIT HAS NOT "
        "BEEN ACCESSED."
    )

    metrics = {
        "sample_train_rows":
            train_summary[
                "rows"
            ],

        "sample_validation_rows":
            validation_summary[
                "rows"
            ],

        "num_classes":
            float(
                num_classes
            ),

        "baseline_train_log_loss":
            baseline_train_log_loss,

        "baseline_validation_log_loss":
            baseline_validation_log_loss,

        "sample_train_log_loss":
            train_metrics[
                "log_loss"
            ],

        "sample_validation_log_loss":
            validation_metrics[
                "log_loss"
            ],

        "sample_train_accuracy":
            train_metrics[
                "accuracy"
            ],

        "sample_validation_accuracy":
            validation_metrics[
                "accuracy"
            ],

        "sample_train_macro_f1":
            train_metrics[
                "macro_f1"
            ],

        "sample_validation_macro_f1":
            validation_metrics[
                "macro_f1"
            ],

        "sample_train_mean_confidence":
            train_metrics[
                "mean_confidence"
            ],

        "sample_validation_mean_confidence":
            validation_metrics[
                "mean_confidence"
            ],

        "sample_train_log_loss_gain":
            train_log_loss_gain,

        "sample_validation_log_loss_gain":
            validation_log_loss_gain,

        "sample_train_bits_gain":
            train_bits_gain,

        "sample_validation_bits_gain":
            validation_bits_gain,

        "best_iteration":
            float(
                best_iteration
                if best_iteration
                is not None
                else -1
            ),
    }

    # Prevent accidental retention of large
    # matrices longer than necessary.
    del train_confusion

    gc.collect()

    return {
        "metrics":
            metrics,

        "history":
            evals_result,

        "artifacts": [
            str(
                model_path
            ),
            str(
                metadata_path
            ),
            str(
                importance_path
            ),
            str(
                training_history_path
            ),
            str(
                class_metrics_path
            ),
            str(
                confusion_matrix_path
            ),
        ],

        "summary": {
            "best_iteration":
                best_iteration,

            "feature_count":
                len(
                    feature_columns
                ),

            "num_classes":
                num_classes,

            "target_column":
                target_column,

            "artifact_dir":
                str(
                    artifact_dir
                ),

            "test_split_used":
                False,
        },
    }