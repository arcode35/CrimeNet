from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb


MACHINE_LEARNING_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

EPS = 1e-15


def _validate_categories(
    *,
    city: str,
    frame: pl.DataFrame,
    categorical_columns: list[str],
    category_levels: dict[str, list[str]],
) -> None:
    for column in categorical_columns:
        known = {
            str(value)
            for value
            in category_levels[column]
        }

        observed = (
            frame
            .get_column(column)
            .drop_nulls()
            .cast(pl.String)
            .unique()
            .to_list()
        )

        unknown = sorted(
            value
            for value in observed
            if value not in known
        )

        if unknown:
            print(
                f"\nWARNING: {city} / {column} "
                "contains categories unseen "
                "during training:"
            )
            print(unknown)


def _prepare_city(
    *,
    city: str,
    frame: pl.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    category_levels: dict[str, list[str]],
    target_column: str,
    class_to_index: dict[str, int],
) -> tuple[
    pd.DataFrame,
    np.ndarray,
]:
    _validate_categories(
        city=city,
        frame=frame,
        categorical_columns=categorical_columns,
        category_levels=category_levels,
    )

    raw_target = (
        frame
        .get_column(target_column)
        .cast(pl.String)
        .to_list()
    )

    unknown_labels = sorted(
        {
            label
            for label in raw_target
            if label not in class_to_index
        }
    )

    if unknown_labels:
        raise ValueError(
            f"{city}: validation contains target "
            "classes outside the training vocabulary: "
            f"{unknown_labels}"
        )

    y = np.asarray(
        [
            class_to_index[label]
            for label in raw_target
        ],
        dtype=np.int32,
    )

    X = (
        frame
        .select(feature_columns)
        .to_pandas()
    )

    for column in categorical_columns:
        X[column] = pd.Categorical(
            X[column],
            categories=category_levels[column],
        )

    for column in feature_columns:
        if column in categorical_columns:
            continue

        X[column] = (
            pd.to_numeric(
                X[column],
                errors="coerce",
            )
            .astype(np.float32)
        )

    return X, y


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


def _class_metrics(
    matrix: np.ndarray,
    *,
    classes: list[str],
) -> tuple[
    dict[str, dict[str, float]],
    float,
    float,
]:
    per_class: dict[
        str,
        dict[str, float],
    ] = {}

    f1_values: list[float] = []
    weighted_f1_numerator = 0.0
    total_support = 0.0

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

        predicted_count = float(
            matrix[
                :,
                class_index
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

        if precision + recall > 0:
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

        f1_values.append(f1)

        weighted_f1_numerator += (
            support
            * f1
        )

        total_support += support

        per_class[class_name] = {
            "precision":
                float(precision),
            "recall":
                float(recall),
            "f1":
                float(f1),
            "support":
                support,
            "predicted_count":
                predicted_count,
        }

    macro_f1 = float(
        np.mean(f1_values)
    )

    weighted_f1 = (
        weighted_f1_numerator
        / max(
            total_support,
            1.0,
        )
    )

    return (
        per_class,
        macro_f1,
        float(weighted_f1),
    )


def _top_k_correct(
    probabilities: np.ndarray,
    y: np.ndarray,
    *,
    k: int,
) -> int:
    k = min(
        k,
        probabilities.shape[1],
    )

    top_k = np.argpartition(
        probabilities,
        -k,
        axis=1,
    )[
        :,
        -k:
    ]

    return int(
        np.any(
            top_k
            == y[:, None],
            axis=1,
        ).sum()
    )


def _evaluate_city(
    *,
    city: str,
    booster: xgb.Booster,
    frame: pl.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    category_levels: dict[str, list[str]],
    target_column: str,
    classes: list[str],
    class_to_index: dict[str, int],
    training_priors: np.ndarray,
) -> tuple[
    dict[str, float | str],
    np.ndarray,
    dict[str, dict[str, float]],
]:
    X, y = _prepare_city(
        city=city,
        frame=frame,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        category_levels=category_levels,
        target_column=target_column,
        class_to_index=class_to_index,
    )

    rows = int(
        y.shape[0]
    )

    if rows == 0:
        raise RuntimeError(
            f"{city}: no observed validation events."
        )

    dmatrix = xgb.DMatrix(
        X,
        label=y,
        enable_categorical=True,
        nthread=-1,
    )

    probabilities = booster.predict(
        dmatrix
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

    expected_shape = (
        rows,
        num_classes,
    )

    if probabilities.shape != expected_shape:
        raise ValueError(
            f"{city}: unexpected probability "
            f"shape {probabilities.shape}; "
            f"expected {expected_shape}."
        )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    if not np.isfinite(
        probabilities
    ).all():
        raise ValueError(
            f"{city}: prediction matrix "
            "contains NaN or infinity."
        )

    if (
        probabilities < 0
    ).any():
        raise ValueError(
            f"{city}: prediction matrix "
            "contains negative probabilities."
        )

    row_sums = probabilities.sum(
        axis=1,
        keepdims=True,
    )

    if (
        row_sums <= 0
    ).any():
        raise ValueError(
            f"{city}: invalid probability rows."
        )

    # Match the training evaluator:
    # normalize away tiny floating point
    # departures from exactly one.
    probabilities = (
        probabilities
        / row_sums
    )

    true_probability = (
        probabilities[
            np.arange(rows),
            y,
        ]
    )

    log_loss_sum = float(
        -np.log(
            np.clip(
                true_probability,
                EPS,
                1.0,
            )
        ).sum()
    )

    log_loss = (
        log_loss_sum
        / rows
    )

    prior_probability = (
        training_priors[
            y
        ]
    )

    prior_log_loss_sum = float(
        -np.log(
            np.clip(
                prior_probability,
                EPS,
                1.0,
            )
        ).sum()
    )

    prior_log_loss = (
        prior_log_loss_sum
        / rows
    )

    predicted_class = (
        probabilities.argmax(
            axis=1
        )
    )

    correct = int(
        np.sum(
            predicted_class
            == y
        )
    )

    accuracy = (
        correct
        / rows
    )

    matrix = _confusion_matrix(
        y,
        predicted_class,
        num_classes=num_classes,
    )

    (
        per_class,
        macro_f1,
        weighted_f1,
    ) = _class_metrics(
        matrix,
        classes=classes,
    )

    confidence_sum = float(
        probabilities.max(
            axis=1
        ).sum()
    )

    mean_confidence = (
        confidence_sum
        / rows
    )

    top3_correct = _top_k_correct(
        probabilities,
        y,
        k=3,
    )

    top5_correct = _top_k_correct(
        probabilities,
        y,
        k=5,
    )

    log_loss_gain = (
        prior_log_loss
        - log_loss
    )

    bits_gain = (
        log_loss_gain
        / np.log(2.0)
    )

    result: dict[
        str,
        float | str,
    ] = {
        "source_city":
            city,

        "rows":
            float(rows),

        "log_loss":
            float(log_loss),

        "prior_log_loss":
            float(prior_log_loss),

        "log_loss_gain":
            float(log_loss_gain),

        "bits_gain":
            float(bits_gain),

        "accuracy":
            float(accuracy),

        "macro_f1":
            float(macro_f1),

        "weighted_f1":
            float(weighted_f1),

        "mean_confidence":
            float(mean_confidence),

        "top3_accuracy":
            float(
                top3_correct
                / rows
            ),

        "top5_accuracy":
            float(
                top5_correct
                / rows
            ),

        # Aggregation fields.
        "_log_loss_sum":
            log_loss_sum,

        "_prior_log_loss_sum":
            prior_log_loss_sum,

        "_correct":
            float(correct),

        "_confidence_sum":
            confidence_sum,

        "_top3_correct":
            float(top3_correct),

        "_top5_correct":
            float(top5_correct),
    }

    print(
        f"\n{city}"
    )

    print(
        f"  Observed events:   {rows:,}"
    )

    print(
        f"  Model log loss:    {log_loss:.6f}"
    )

    print(
        f"  Prior log loss:    {prior_log_loss:.6f}"
    )

    print(
        f"  Gain:              {log_loss_gain:.6f} nats/event"
    )

    print(
        f"  Bits gained:       {bits_gain:.6f}"
    )

    print(
        f"  Accuracy:          {accuracy:.6f}"
    )

    print(
        f"  Macro F1:          {macro_f1:.6f}"
    )

    print(
        f"  Weighted F1:       {weighted_f1:.6f}"
    )

    print(
        f"  Top-3 accuracy:    {top3_correct / rows:.6f}"
    )

    print(
        f"  Top-5 accuracy:    {top5_correct / rows:.6f}"
    )

    print(
        f"  Mean confidence:   {mean_confidence:.6f}"
    )

    del X
    del y
    del dmatrix
    del probabilities
    del predicted_class

    gc.collect()

    return (
        result,
        matrix,
        per_class,
    )


def validate(
    config: dict[str, Any],
    *,
    run_id: str,
) -> dict[str, Any]:
    model_config = config[
        "model"
    ]

    data_config = config[
        "data"
    ]

    validation_config = config.get(
        "validation",
        {},
    )

    validation_year = int(
        validation_config.get(
            "year",
            2024,
        )
    )

    # CPU by default so full validation can run
    # while another XGBoost training job owns
    # the GPU.
    validation_device = str(
        validation_config.get(
            "device",
            "cpu",
        )
    )

    artifact_root = (
        MACHINE_LEARNING_ROOT
        / config[
            "artifacts"
        ][
            "output_root"
        ]
    )

    artifact_dir = (
        artifact_root
        / model_config[
            "name"
        ]
        / run_id
    )

    model_path = (
        artifact_dir
        / "model.json"
    )

    metadata_path = (
        artifact_dir
        / "metadata.json"
    )

    city_output_path = (
        artifact_dir
        / (
            f"full_validation_"
            f"{validation_year}_by_city.csv"
        )
    )

    class_output_path = (
        artifact_dir
        / (
            f"full_validation_"
            f"{validation_year}_class_metrics.json"
        )
    )

    confusion_output_path = (
        artifact_dir
        / (
            f"full_validation_"
            f"{validation_year}_confusion_matrix.csv"
        )
    )

    summary_output_path = (
        artifact_dir
        / (
            f"full_validation_"
            f"{validation_year}_summary.json"
        )
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing model artifact: "
            f"{model_path}"
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing model metadata: "
            f"{metadata_path}"
        )

    with metadata_path.open() as file:
        metadata = json.load(file)

    metadata_run_id = (
        metadata.get(
            "run_id"
        )
    )

    if (
        metadata_run_id is not None
        and metadata_run_id != run_id
    ):
        raise RuntimeError(
            "Requested run ID does not match "
            "model metadata. "
            f"Requested={run_id}, "
            f"metadata={metadata_run_id}"
        )

    configured_table_root = (
        data_config[
            "model_table_root"
        ]
    )

    metadata_table_root = (
        metadata.get(
            "model_table_root"
        )
    )

    if (
        metadata_table_root is not None
        and metadata_table_root
        != configured_table_root
    ):
        raise RuntimeError(
            "Model-table path in training "
            "metadata differs from the "
            "supplied experiment config."
        )

    feature_columns = list(
        metadata[
            "features"
        ]
    )

    categorical_columns = list(
        metadata[
            "categorical_columns"
        ]
    )

    category_levels = {
        str(key):
            [
                str(value)
                for value in values
            ]
        for key, values
        in metadata[
            "category_levels"
        ].items()
    }

    target_column = str(
        metadata[
            "target_column"
        ]
    )

    classes = [
        str(value)
        for value
        in metadata[
            "classes"
        ]
    ]

    class_to_index = {
        str(key):
            int(value)
        for key, value
        in metadata[
            "class_to_index"
        ].items()
    }

    num_classes = int(
        metadata[
            "num_classes"
        ]
    )

    if num_classes != len(
        classes
    ):
        raise RuntimeError(
            "Metadata class count does not "
            "match class vocabulary."
        )

    expected_mapping = {
        class_name:
            index
        for index, class_name
        in enumerate(classes)
    }

    if (
        class_to_index
        != expected_mapping
    ):
        raise RuntimeError(
            "Metadata class_to_index does "
            "not agree with class ordering."
        )

    configured_target = (
        config
        .get(
            "target",
            {},
        )
        .get(
            "column"
        )
    )

    if (
        configured_target is not None
        and str(configured_target)
        != target_column
    ):
        raise RuntimeError(
            "Configured target does not "
            "match training metadata. "
            f"Config={configured_target}, "
            f"metadata={target_column}"
        )

    prior_map = (
        metadata[
            "training_class_priors"
        ]
    )

    training_priors = np.asarray(
        [
            float(
                prior_map[
                    class_name
                ]
            )
            for class_name
            in classes
        ],
        dtype=np.float64,
    )

    if not np.isfinite(
        training_priors
    ).all():
        raise ValueError(
            "Training class priors contain "
            "NaN or infinity."
        )

    if (
        training_priors < 0
    ).any():
        raise ValueError(
            "Training class priors contain "
            "negative values."
        )

    prior_sum = float(
        training_priors.sum()
    )

    if prior_sum <= 0:
        raise ValueError(
            "Training class priors sum to zero."
        )

    training_priors = (
        training_priors
        / prior_sum
    )

    best_iteration = (
        metadata.get(
            "best_iteration"
        )
    )

    print(
        f"Run ID:             {run_id}"
    )

    print(
        f"Validation year:    {validation_year}"
    )

    print(
        f"Validation device:  {validation_device}"
    )

    print(
        f"Target:             {target_column}"
    )

    print(
        f"Classes:            {num_classes}"
    )

    print(
        f"Features:           {len(feature_columns)}"
    )

    print(
        f"Best iteration:     {best_iteration}"
    )

    booster = xgb.Booster()

    booster.load_model(
        model_path
    )

    booster.set_param(
        {
            "device":
                validation_device,
        }
    )

    credentials = (
        pl.CredentialProviderGCP()
    )

    table = pl.scan_delta(
        configured_table_root,
        credential_provider=
            credentials,
    )

    validation_split = (
        data_config.get(
            "validation_split",
            "validation",
        )
    )

    validation_rows = (
        table

        .filter(
            pl.col("split")
            == validation_split
        )

        .filter(
            pl.col("row_year")
            == validation_year
        )

        # Mark probabilities are conditioned
        # on an event actually occurring.
        .filter(
            pl.col(
                "is_observed_event"
            )
        )

        .filter(
            pl.col(
                "event_count"
            )
            == 1
        )

        .filter(
            pl.col(
                target_column
            )
            .is_not_null()
        )
    )

    cities = (
        validation_rows

        .select(
            "source_city"
        )

        .unique()

        .sort(
            "source_city"
        )

        .collect()

        .get_column(
            "source_city"
        )

        .to_list()
    )

    print(
        f"\nValidation cities: "
        f"{cities}"
    )

    if not cities:
        raise RuntimeError(
            f"No {validation_year} "
            "validation cities found."
        )

    results: list[
        dict[str, float | str]
    ] = []

    city_class_metrics: dict[
        str,
        dict[
            str,
            dict[str, float],
        ],
    ] = {}

    global_confusion = np.zeros(
        (
            num_classes,
            num_classes,
        ),
        dtype=np.int64,
    )

    for city in cities:
        print(
            "\n"
            "========================================"
        )

        print(
            f"Loading {city}..."
        )

        print(
            "========================================"
        )

        frame = (
            validation_rows

            .filter(
                pl.col(
                    "source_city"
                )
                == city
            )

            .select(
                [
                    *feature_columns,
                    target_column,
                ]
            )

            .collect()
        )

        (
            result,
            city_confusion,
            per_class,
        ) = _evaluate_city(
            city=str(city),
            booster=booster,
            frame=frame,
            feature_columns=
                feature_columns,
            categorical_columns=
                categorical_columns,
            category_levels=
                category_levels,
            target_column=
                target_column,
            classes=
                classes,
            class_to_index=
                class_to_index,
            training_priors=
                training_priors,
        )

        results.append(
            result
        )

        city_class_metrics[
            str(city)
        ] = per_class

        global_confusion += (
            city_confusion
        )

        del frame
        del city_confusion

        gc.collect()

    total_rows = int(
        sum(
            float(
                result[
                    "rows"
                ]
            )
            for result
            in results
        )
    )

    total_log_loss = float(
        sum(
            float(
                result[
                    "_log_loss_sum"
                ]
            )
            for result
            in results
        )
    )

    total_prior_log_loss = float(
        sum(
            float(
                result[
                    "_prior_log_loss_sum"
                ]
            )
            for result
            in results
        )
    )

    total_correct = int(
        sum(
            float(
                result[
                    "_correct"
                ]
            )
            for result
            in results
        )
    )

    total_confidence = float(
        sum(
            float(
                result[
                    "_confidence_sum"
                ]
            )
            for result
            in results
        )
    )

    total_top3 = int(
        sum(
            float(
                result[
                    "_top3_correct"
                ]
            )
            for result
            in results
        )
    )

    total_top5 = int(
        sum(
            float(
                result[
                    "_top5_correct"
                ]
            )
            for result
            in results
        )
    )

    if total_rows <= 0:
        raise RuntimeError(
            "Global validation contains "
            "zero observed events."
        )

    global_log_loss = (
        total_log_loss
        / total_rows
    )

    global_prior_log_loss = (
        total_prior_log_loss
        / total_rows
    )

    global_gain = (
        global_prior_log_loss
        - global_log_loss
    )

    global_bits = (
        global_gain
        / np.log(2.0)
    )

    global_accuracy = (
        total_correct
        / total_rows
    )

    global_mean_confidence = (
        total_confidence
        / total_rows
    )

    global_top3 = (
        total_top3
        / total_rows
    )

    global_top5 = (
        total_top5
        / total_rows
    )

    (
        overall_class_metrics,
        global_macro_f1,
        global_weighted_f1,
    ) = _class_metrics(
        global_confusion,
        classes=classes,
    )

    print(
        "\n"
        "============================================================"
    )

    print(
        f"FULL {validation_year} "
        "MARK VALIDATION — OVERALL"
    )

    print(
        "============================================================"
    )

    print(
        f"Observed events:          "
        f"{total_rows:,}"
    )

    print(
        f"Model log loss:           "
        f"{global_log_loss:.6f}"
    )

    print(
        f"Training-prior log loss:  "
        f"{global_prior_log_loss:.6f}"
    )

    print(
        f"Log-loss gain / event:    "
        f"{global_gain:.6f}"
    )

    print(
        f"Bits gained / event:      "
        f"{global_bits:.6f}"
    )

    print(
        f"Accuracy:                 "
        f"{global_accuracy:.6f}"
    )

    print(
        f"Macro F1:                 "
        f"{global_macro_f1:.6f}"
    )

    print(
        f"Weighted F1:              "
        f"{global_weighted_f1:.6f}"
    )

    print(
        f"Top-3 accuracy:           "
        f"{global_top3:.6f}"
    )

    print(
        f"Top-5 accuracy:           "
        f"{global_top5:.6f}"
    )

    print(
        f"Mean confidence:          "
        f"{global_mean_confidence:.6f}"
    )

    # Strip private aggregation fields
    # before writing user-facing city output.
    public_results = []

    for result in results:
        public_results.append(
            {
                key:
                    value
                for key, value
                in result.items()
                if not key.startswith(
                    "_"
                )
            }
        )

    results_df = pl.DataFrame(
        public_results
    ).sort(
        "source_city"
    )

    print(
        "\n"
        "============================================================"
    )

    print(
        f"FULL {validation_year} "
        "MARK VALIDATION — BY CITY"
    )

    print(
        "============================================================"
    )

    print(
        results_df
    )

    results_df.write_csv(
        city_output_path
    )

    confusion_frame = (
        pd.DataFrame(
            global_confusion,
            index=classes,
            columns=classes,
        )
    )

    confusion_frame.index.name = (
        "actual"
    )

    confusion_frame.to_csv(
        confusion_output_path
    )

    class_payload = {
        "overall":
            overall_class_metrics,

        "by_city":
            city_class_metrics,
    }

    with class_output_path.open(
        "w"
    ) as file:
        json.dump(
            class_payload,
            file,
            indent=2,
        )

    overall_metrics = {
        "rows":
            float(total_rows),

        "log_loss":
            float(
                global_log_loss
            ),

        "prior_log_loss":
            float(
                global_prior_log_loss
            ),

        "log_loss_gain":
            float(
                global_gain
            ),

        "bits_gain":
            float(
                global_bits
            ),

        "accuracy":
            float(
                global_accuracy
            ),

        "macro_f1":
            float(
                global_macro_f1
            ),

        "weighted_f1":
            float(
                global_weighted_f1
            ),

        "top3_accuracy":
            float(
                global_top3
            ),

        "top5_accuracy":
            float(
                global_top5
            ),

        "mean_confidence":
            float(
                global_mean_confidence
            ),
    }

    summary_payload = {
        "run_id":
            run_id,

        "validation_year":
            validation_year,

        "validation_device":
            validation_device,

        "target_column":
            target_column,

        "num_classes":
            num_classes,

        "city_count":
            len(cities),

        "metrics":
            overall_metrics,

        "test_split_used":
            False,
    }

    with summary_output_path.open(
        "w"
    ) as file:
        json.dump(
            summary_payload,
            file,
            indent=2,
        )

    print(
        f"\nSaved city diagnostics: "
        f"{city_output_path}"
    )

    print(
        f"Saved class metrics: "
        f"{class_output_path}"
    )

    print(
        f"Saved confusion matrix: "
        f"{confusion_output_path}"
    )

    print(
        f"Saved summary: "
        f"{summary_output_path}"
    )

    print(
        "\nTEST SPLIT HAS NOT BEEN ACCESSED."
    )

    city_metrics = {
        str(
            result[
                "source_city"
            ]
        ): {
            key:
                float(value)
            for key, value
            in result.items()
            if (
                key
                != "source_city"
                and not key.startswith(
                    "_"
                )
            )
        }
        for result
        in results
    }

    return {
        "metrics":
            overall_metrics,

        "city_metrics":
            city_metrics,

        "artifacts": [
            str(
                city_output_path
            ),
            str(
                class_output_path
            ),
            str(
                confusion_output_path
            ),
            str(
                summary_output_path
            ),
        ],

        "summary": {
            "validation_year":
                validation_year,

            "city_count":
                len(cities),

            "target_column":
                target_column,

            "num_classes":
                num_classes,

            "test_split_used":
                False,
        },
    }