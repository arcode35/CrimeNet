from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from machine_learning.data.model_table import resolve_model_table

from machine_learning.models.crimenet_omega.data import (
    Omega0Dataset,
    fit_preprocessor,
    load_split,
    load_training_vocabulary,
)
from machine_learning.models.crimenet_omega.loss import (
    MarkedPointProcessNLL,
)
from machine_learning.models.crimenet_omega.model import (
    CrimeNetOmega0,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise TypeError(
            f"Expected YAML mapping at root: {path}"
        )
    return config


def move_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
    }


def _resolve_autocast_dtype(
    name: str,
) -> torch.dtype | None:
    normalized = name.lower()
    if normalized in {"float32", "fp32"}:
        return None
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError(
        "training.dtype must be one of: "
        "float32, fp32, bfloat16, bf16."
    )


def _expected_exposure_seconds_for_unit(
    intensity_unit: str,
) -> float:
    mapping = {
        "events_per_cell_second": 1.0,
        "events_per_cell_hour": 3600.0,
        "events_per_cell_day": 86400.0,
    }
    if intensity_unit not in mapping:
        raise ValueError(
            f"Unsupported intensity unit {intensity_unit!r}."
        )
    return mapping[intensity_unit]


def _exposure_label(seconds: float) -> str:
    if seconds == 1.0:
        return "cell-second"
    if seconds == 3600.0:
        return "cell-hour"
    if seconds == 86400.0:
        return "cell-day"
    return f"cell-{seconds:g}-seconds"


def _validate_cross_config(
    cfg: dict[str, Any],
) -> None:
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    likelihood_cfg = cfg["likelihood"]
    training_cfg = cfg["training"]

    splits = data_cfg["splits"]
    if "test" in splits:
        raise ValueError("Omega training config must not expose the sealed test split")
    split_values = {str(splits["train"]), str(splits["validation"])}
    if len(split_values) != 2:
        raise ValueError(
            "train and validation split names must be distinct."
        )

    mark_target = str(model_cfg["mark"]["target"])
    configured_subtype = str(data_cfg["columns"]["subtype"])
    if mark_target != configured_subtype:
        raise ValueError(
            "model.mark.target must match data.columns.subtype; "
            f"got {mark_target!r} vs {configured_subtype!r}."
        )

    exposure_seconds = float(
        likelihood_cfg["exposure_unit_seconds"]
    )
    intensity_unit = str(model_cfg["intensity"]["unit"])
    expected_seconds = _expected_exposure_seconds_for_unit(
        intensity_unit
    )
    if exposure_seconds != expected_seconds:
        raise ValueError(
            "Intensity unit and exposure unit are inconsistent: "
            f"{intensity_unit!r} requires "
            f"{expected_seconds:g} seconds, but YAML specifies "
            f"{exposure_seconds:g}."
        )

    early_cfg = training_cfg.get("early_stopping", {})
    if bool(early_cfg.get("enabled", False)):
        if str(
            early_cfg.get(
                "metric",
                "validation_nll_per_event",
            )
        ) != "validation_nll_per_event":
            raise ValueError(
                "Omega-0 currently early-stops only on "
                "validation_nll_per_event."
            )
        if str(early_cfg.get("mode", "min")).lower() != "min":
            raise ValueError(
                "Omega-0 validation NLL must use early-stopping mode=min."
            )

@torch.no_grad()
def evaluate(
    model: CrimeNetOmega0,
    objective: MarkedPointProcessNLL,
    loader: DataLoader,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
) -> dict[str, float]:
    model.eval()

    total_nll = 0.0
    total_intensity_event_nll = 0.0
    total_mark_nll = 0.0
    total_integral = 0.0
    total_events = 0.0

    # For diagnostic mean intensities.
    event_intensity_sum = 0.0
    event_intensity_weight = 0.0

    integration_intensity_sum = 0.0
    integration_row_count = 0

    for batch in loader:
        batch = move_batch(batch, device)

        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype or torch.bfloat16,
            enabled=(
                device.type == "cuda"
                and autocast_dtype is not None
            ),
        ):
            output = model(
                numeric=batch["numeric"],
                city=batch["city"],
                lighting=batch["lighting"],
            )

        # Raw batch NLL. This allows batches containing zero events.
        _, metrics = objective(
            output,
            batch,
            normalize=False,
        )

        # ---------------------------------------------------------
        # Likelihood totals
        # ---------------------------------------------------------

        total_nll += metrics["total_nll"].item()

        total_intensity_event_nll += (
            metrics["intensity_event_nll"].item()
        )

        total_mark_nll += (
            metrics["mark_nll"].item()
        )

        total_integral += metrics["integral"].item()

        total_events += metrics["num_events"].item()

        # ---------------------------------------------------------
        # Intensity diagnostics
        # ---------------------------------------------------------

        intensity = output["intensity"].float()

        observed = batch["is_observed"].bool()
        event_count = batch["event_count"].float()

        event_mask = observed & (event_count > 0)
        integration_mask = ~observed

        # Event-count-weighted mean intensity.
        #
        # If one row represents multiple observed events, it should
        # contribute proportionally to the event-level diagnostic.
        if event_mask.any():
            counts = event_count[event_mask]

            event_intensity_sum += (
                intensity[event_mask] * counts
            ).sum().item()

            event_intensity_weight += counts.sum().item()

        # Ordinary row-wise mean intensity over integration points.
        if integration_mask.any():
            integration_intensity_sum += (
                intensity[integration_mask].sum().item()
            )

            integration_row_count += int(
                integration_mask.sum().item()
            )

    if total_events <= 0:
        raise RuntimeError(
            "Validation loader produced no observed events."
        )

    if integration_row_count <= 0:
        raise RuntimeError(
            "Validation loader produced no integration rows."
        )

    # -------------------------------------------------------------
    # Split-level diagnostics
    # -------------------------------------------------------------

    nll_per_event = total_nll / total_events

    intensity_event_nll_per_event = (
        total_intensity_event_nll
        / total_events
    )

    mark_nll_per_event = (
        total_mark_nll
        / total_events
    )

    integral_per_event = (
        total_integral
        / total_events
    )

    mean_event_intensity = (
        event_intensity_sum
        / event_intensity_weight
    )

    mean_integration_intensity = (
        integration_intensity_sum
        / integration_row_count
    )

    # In a point-process likelihood, the compensator is the
    # expected integrated event count over the domain.
    predicted_events = total_integral
    observed_events = total_events

    calibration_ratio = (
        predicted_events
        / observed_events
    )

    return {
        "nll_per_event":
            nll_per_event,

        "intensity_event_nll_per_event":
            intensity_event_nll_per_event,

        "mark_nll_per_event":
            mark_nll_per_event,

        "integral_per_event":
            integral_per_event,

        "mean_event_intensity":
            mean_event_intensity,

        "mean_integration_intensity":
            mean_integration_intensity,

        "observed_events":
            observed_events,

        "predicted_events":
            predicted_events,

        "calibration_ratio":
            calibration_ratio,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    _validate_cross_config(cfg)

    experiment_cfg = cfg["experiment"]
    data_cfg = cfg["data"]
    features_cfg = cfg["features"]
    model_cfg = cfg["model"]
    training_cfg = cfg["training"]
    likelihood_cfg = cfg["likelihood"]
    optimizer_cfg = training_cfg["optimizer"]
    dataloader_cfg = training_cfg["dataloader"]
    output_cfg = cfg["output"]

    seed = int(experiment_cfg["seed"])
    torch.manual_seed(seed)

    numeric_features = list(features_cfg["numeric"])
    columns_cfg = dict(data_cfg["columns"])

    snapshot_source = str(data_cfg["snapshot_source"])
    if snapshot_source == "current_final_model_snapshot":
        # Freeze the moving pointer once; downstream loaders receive only the
        # exact immutable URI for the remainder of this run.
        snapshot_source = resolve_model_table().snapshot_uri
    train_split = str(data_cfg["splits"]["train"])
    validation_split = str(data_cfg["splits"]["validation"])

    train_fraction = float(data_cfg["train_fraction"])
    validation_fraction = float(
        data_cfg["validation_fraction"]
    )

    batch_size = int(training_cfg["batch_size"])
    validation_batch_size = int(
        training_cfg["validation_batch_size"]
    )
    epochs = int(training_cfg["epochs"])
    gradient_clip_norm = float(
        training_cfg["gradient_clip_norm"]
    )

    learning_rate = float(
        optimizer_cfg["learning_rate"]
    )
    weight_decay = float(
        optimizer_cfg["weight_decay"]
    )

    num_workers = int(
        dataloader_cfg["num_workers"]
    )

    exposure_unit_seconds = float(
        likelihood_cfg["exposure_unit_seconds"]
    )
    exposure_label = _exposure_label(
        exposure_unit_seconds
    )

    output_root = Path(output_cfg["root"])
    checkpoint_path = (
        output_root / output_cfg["checkpoint_name"]
    )

    configured_device = str(
        training_cfg.get("device", "cuda")
    ).lower()

    if (
        configured_device == "cuda"
        and not torch.cuda.is_available()
    ):
        print(
            "CUDA requested but unavailable; falling back to CPU."
        )
        configured_device = "cpu"

    device = torch.device(configured_device)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    autocast_dtype = _resolve_autocast_dtype(
        str(training_cfg.get("dtype", "bfloat16"))
    )

    print(f"Config: {args.config}")
    print(f"Device: {device}")
    print(
        f"Autocast dtype: "
        f"{autocast_dtype if autocast_dtype else 'float32'}"
    )
    print(f"Model table: {snapshot_source}")
    print(f"Train split: {train_split}")
    print(f"Validation split: {validation_split}")
    print(f"Train fraction: {train_fraction:g}")
    print(
        f"Validation fraction: {validation_fraction:g}"
    )
    print(
        f"Numeric features: {len(numeric_features)}"
    )

    print("\nLoading full-training categorical vocabulary...")
    vocabulary = load_training_vocabulary(
        snapshot_source,
        train_split,
        columns=columns_cfg,
    )
    print(
        f"Vocabulary: cities={len(vocabulary.cities)}, "
        f"lighting={len(vocabulary.lighting_conditions)}, "
        f"subtypes={len(vocabulary.subtypes)}"
    )

    print("\nLoading training sample...")
    train_frame = load_split(
        snapshot_source,
        train_split,
        numeric_features=numeric_features,
        columns=columns_cfg,
        fraction=train_fraction,
        seed=seed,
    )
    print(f"Train rows: {train_frame.height:,}")

    preprocessor = fit_preprocessor(
        train_frame,
        numeric_features=numeric_features,
        vocabulary=vocabulary,
    )

    train_dataset = Omega0Dataset(
        train_frame,
        preprocessor,
        numeric_features=numeric_features,
        exposure_unit_seconds=exposure_unit_seconds,
    )
    w = train_dataset.integration_weight[
        ~train_dataset.is_observed
    ].numpy()

    import numpy as np

    for q in [0, .5, .9, .95, .99, .999, 1.0]:
        print(q, np.quantile(w, q))
if __name__ == "__main__":
    main()
