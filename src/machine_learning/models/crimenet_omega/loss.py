from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class MarkedPointProcessNLL(nn.Module):
    """
    Omega-0 negative log likelihood:

        - sum_events [
              log lambda(q_i)
              + log p(mark_i | q_i)
          ]
        + sum_integration [
              weight_q * lambda(q)
          ]

    The model intensity and integration_weight must use reciprocal units.

    Training uses ``normalize=True`` and minimizes NLL per observed event.
    Validation uses ``normalize=False`` so integration-only batches can
    contribute their compensator term without requiring an observed event
    in every individual batch.
    """

    SUPPORTED_TYPE = "marked_spatiotemporal_point_process"

    def __init__(
        self,
        likelihood_cfg: Mapping[str, Any],
    ) -> None:
        super().__init__()

        required = {
            "type",
            "exposure_unit_seconds",
            "intensity_event_term",
            "compensator_term",
            "mark_term",
        }
        missing = required - set(likelihood_cfg)
        if missing:
            raise KeyError(
                f"Missing likelihood config keys: {sorted(missing)}"
            )

        likelihood_type = str(likelihood_cfg["type"])
        if likelihood_type != self.SUPPORTED_TYPE:
            raise ValueError(
                "Omega-0 requires likelihood.type="
                f"{self.SUPPORTED_TYPE!r}; got {likelihood_type!r}."
            )

        exposure_unit_seconds = float(
            likelihood_cfg["exposure_unit_seconds"]
        )
        if exposure_unit_seconds <= 0:
            raise ValueError(
                "likelihood.exposure_unit_seconds must be > 0."
            )
        self.exposure_unit_seconds = exposure_unit_seconds

        for key in (
            "intensity_event_term",
            "compensator_term",
            "mark_term",
        ):
            if not bool(likelihood_cfg[key]):
                raise ValueError(
                    f"Omega-0 requires likelihood.{key}=true."
                )

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        *,
        normalize: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # Keep likelihood arithmetic in FP32 even when the neural forward
        # pass uses BF16 autocast.
        intensity = outputs["intensity"].float()
        mark_logits = outputs["mark_logits"].float()

        observed = batch["is_observed"].bool()
        count = batch["event_count"].float()
        integration_weight = batch["integration_weight"].float()
        subtype = batch["subtype"].long()

        if not torch.isfinite(intensity).all():
            raise RuntimeError(
                "Non-finite intensity encountered."
            )
        if not torch.isfinite(mark_logits).all():
            raise RuntimeError(
                "Non-finite mark logits encountered."
            )
        if not torch.isfinite(integration_weight).all():
            raise RuntimeError(
                "Non-finite integration weight encountered."
            )
        if (intensity <= 0).any():
            raise RuntimeError(
                "Intensity must be strictly positive."
            )
        if (count < 0).any():
            raise RuntimeError(
                "event_count must be non-negative."
            )
        if (integration_weight < 0).any():
            raise RuntimeError(
                "integration_weight must be non-negative."
            )

        event_mask = observed & (count > 0)
        integration_mask = ~observed
        num_events = count[event_mask].sum()

        # Event and mark terms are zero for a legitimate integration-only
        # validation batch.
        if event_mask.any():
            intensity_event_nll = -(
                count[event_mask]
                * torch.log(intensity[event_mask])
            ).sum()

            event_subtype = subtype[event_mask]
            if (event_subtype < 0).any():
                raise RuntimeError(
                    "Observed event rows contain invalid subtype ids."
                )
            if (
                event_subtype >= mark_logits.shape[-1]
            ).any():
                raise RuntimeError(
                    "Observed event rows contain subtype ids outside "
                    "the mark-head vocabulary."
                )

            log_mark_probs = F.log_softmax(
                mark_logits[event_mask],
                dim=-1,
            )
            selected_log_mark_prob = (
                log_mark_probs.gather(
                    dim=1,
                    index=event_subtype.unsqueeze(1),
                )
                .squeeze(1)
            )
            mark_nll = -(
                count[event_mask]
                * selected_log_mark_prob
            ).sum()
        else:
            # Zero-valued tensors on the correct device/dtype.
            intensity_event_nll = intensity.sum() * 0.0
            mark_nll = mark_logits.sum() * 0.0

        integral = (
            integration_weight[integration_mask]
            * intensity[integration_mask]
        ).sum()

        total_nll = (
            intensity_event_nll
            + mark_nll
            + integral
        )

        if normalize:
            if num_events <= 0:
                raise RuntimeError(
                    "Training batch contains no observed events."
                )
            loss = total_nll / num_events
        else:
            # Validation aggregates raw NLL across all batches and divides
            # by the total number of observed events only after the full
            # validation loader has been consumed.
            loss = total_nll

        denominator = num_events.clamp_min(1.0)

        metrics = {
            "total_nll": total_nll.detach(),
            "num_events": num_events.detach(),
            "intensity_event_nll":
                intensity_event_nll.detach(),
            "mark_nll": mark_nll.detach(),
            "integral": integral.detach(),
            "nll_per_event":
                (total_nll / denominator).detach(),
            "intensity_event_nll_per_event": (
                intensity_event_nll / denominator
            ).detach(),
            "mark_nll_per_event": (
                mark_nll / denominator
            ).detach(),
            "integral_per_event": (
                integral / denominator
            ).detach(),
            "mean_intensity": intensity.mean().detach(),
        }

        return loss, metrics