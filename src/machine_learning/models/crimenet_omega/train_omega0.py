from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

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
    split_values = {
        str(splits["train"]),
        str(splits["validation"]),
        str(splits["test"]),
    }
    if len(split_values) != 3:
        raise ValueError(
            "train, validation, and test split names must be distinct."
        )

    if bool(data_cfg.get("allow_test_access", False)):
        raise ValueError(
            "Omega-0 training requires data.allow_test_access=false."
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
) -> float:
    model.eval()

    total_nll = 0.0
    total_events = 0.0

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

        loss, metrics = objective(output, batch)

        n = float(metrics["num_events"])
        total_nll += float(loss) * n
        total_events += n

    if total_events <= 0:
        raise RuntimeError(
            "Validation loader produced no observed events."
        )

    return total_nll / total_events


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

    table_root = str(data_cfg["model_table_root"])
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
    print(f"Model table: {table_root}")
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
        table_root,
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
        table_root,
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

    print("\nLoading validation sample...")
    val_frame = load_split(
        table_root,
        validation_split,
        numeric_features=numeric_features,
        columns=columns_cfg,
        fraction=validation_fraction,
        seed=seed + 1,
    )
    print(f"Validation rows: {val_frame.height:,}")

    val_dataset = Omega0Dataset(
        val_frame,
        preprocessor,
        numeric_features=numeric_features,
        exposure_unit_seconds=exposure_unit_seconds,
    )

    pin_memory = (
        bool(dataloader_cfg.get("pin_memory", True))
        and device.type == "cuda"
    )
    persistent_workers = (
        bool(
            dataloader_cfg.get(
                "persistent_workers",
                True,
            )
        )
        and num_workers > 0
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=validation_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    model = CrimeNetOmega0(
        model_cfg=model_cfg,
        num_numeric=len(numeric_features),
        num_cities=len(preprocessor.cities),
        num_lighting_conditions=len(
            preprocessor.lighting_conditions
        ),
        num_subtypes=len(preprocessor.subtypes),
    )

    objective = MarkedPointProcessNLL(
        likelihood_cfg
    )

    total_events = float(
        train_dataset.event_count[
            train_dataset.is_observed
        ].sum()
    )
    integration_mask = ~train_dataset.is_observed
    total_exposure = float(
        train_dataset.integration_weight[
            integration_mask
        ].sum()
    )

    if total_events <= 0:
        raise RuntimeError(
            "Training sample contains no observed events."
        )
    if total_exposure <= 0:
        raise RuntimeError(
            "Training exposure is zero or negative."
        )

    base_rate = total_events / total_exposure

    print(
        f"\nEmpirical base rate: {base_rate:.8f} "
        f"events/{exposure_label}"
    )
    print(f"Observed events: {total_events:,.0f}")
    print(
        f"Exposure: {total_exposure:,.2f} {exposure_label}s"
    )

    model.initialize_base_rate(base_rate)
    model.to(device)
    objective.to(device)

    optimizer_name = str(
        optimizer_cfg["name"]
    ).lower()
    if optimizer_name != "adamw":
        raise ValueError(
            "Omega-0 currently supports only AdamW; "
            f"got {optimizer_name!r}."
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    early_cfg = training_cfg.get("early_stopping", {})
    early_stopping_enabled = bool(
        early_cfg.get("enabled", False)
    )
    patience = int(early_cfg.get("patience", 3))
    if patience <= 0:
        raise ValueError(
            "early_stopping.patience must be > 0."
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    if bool(output_cfg.get("save_config", True)):
        with (
            output_root / "resolved_config.yaml"
        ).open("w", encoding="utf-8") as f:
            yaml.safe_dump(
                cfg,
                f,
                sort_keys=False,
            )

    if bool(
        output_cfg.get(
            "save_preprocessor",
            True,
        )
    ):
        with (
            output_root / "preprocessor.json"
        ).open("w", encoding="utf-8") as f:
            json.dump(
                preprocessor.to_dict(),
                f,
                indent=2,
            )

    best_val = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()

        running_nll = 0.0
        running_events = 0.0

        for step, batch in enumerate(
            train_loader,
            start=1,
        ):
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)

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

            # Likelihood executes in explicit FP32 outside autocast.
            loss, metrics = objective(output, batch)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at epoch={epoch}, "
                    f"step={step}: {float(loss)}"
                )

            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=gradient_clip_norm,
            )

            if not torch.isfinite(grad_norm):
                raise RuntimeError(
                    f"Non-finite gradient norm at epoch={epoch}, "
                    f"step={step}."
                )

            optimizer.step()

            n = metrics["num_events"].item()
            running_nll += loss.detach().item() * n
            running_events += n

            if step % 100 == 0:
                print(
                    f"epoch={epoch} "
                    f"step={step} "
                    f"nll/event="
                    f"{running_nll / running_events:.6f} "
                    f"intensity="
                    f"{metrics['mean_intensity'].item():.8f}"
                    f"mark_nll="
                    f"{float(metrics['mark_nll_per_event']):.6f} "
                    f"grad_norm={float(grad_norm):.4f}"
                )

        if running_events <= 0:
            raise RuntimeError(
                "Training epoch produced no observed events."
            )

        train_nll = running_nll / running_events

        val_nll = evaluate(
            model,
            objective,
            val_loader,
            device,
            autocast_dtype,
        )

        print(
            f"\nEpoch {epoch}: "
            f"train NLL/event={train_nll:.6f} "
            f"validation NLL/event={val_nll:.6f}\n"
        )

        if val_nll < best_val:
            best_val = val_nll
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "preprocessor": preprocessor.to_dict(),
                    "config": cfg,
                    "validation_nll_per_event": best_val,
                    "epoch": epoch,
                    "model": "CrimeNetOmega0",
                },
                checkpoint_path,
            )

            print(
                f"Saved best checkpoint: {checkpoint_path}"
            )
        else:
            epochs_without_improvement += 1

            if (
                early_stopping_enabled
                and epochs_without_improvement >= patience
            ):
                print(
                    "Early stopping: no validation improvement "
                    f"for {patience} epochs."
                )
                break

    print(
        "\nTraining complete. "
        f"Best validation NLL/event: {best_val:.6f}"
    )


if __name__ == "__main__":
    main()