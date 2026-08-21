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
                f"Omega-0 requires likelihood.type="
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

        # All three terms are mathematically required for this Omega-0 model.
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
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # Explicit FP32 likelihood arithmetic even when the neural forward
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

        event_mask = observed & (count > 0)
        integration_mask = ~observed

        num_events = count[event_mask].sum()
        if num_events <= 0:
            raise RuntimeError(
                "Batch contains no observed events. "
                "Use a sufficiently large batch or a stratified sampler."
            )

        intensity_event_nll = -(
            count[event_mask]
            * torch.log(intensity[event_mask])
        ).sum()

        log_mark_probs = F.log_softmax(
            mark_logits[event_mask],
            dim=-1,
        )
        selected_log_mark_prob = (
            log_mark_probs.gather(
                dim=1,
                index=subtype[event_mask].unsqueeze(1),
            )
            .squeeze(1)
        )
        mark_nll = -(
            count[event_mask]
            * selected_log_mark_prob
        ).sum()

        integral = (
            integration_weight[integration_mask]
            * intensity[integration_mask]
        ).sum()

        total_nll = (
            intensity_event_nll
            + mark_nll
            + integral
        )
        nll_per_event = total_nll / num_events

        metrics = {
            "nll_per_event": nll_per_event.detach(),
            "intensity_event_nll_per_event": (
                intensity_event_nll / num_events
            ).detach(),
            "mark_nll_per_event": (
                mark_nll / num_events
            ).detach(),
            "integral_per_event": (
                integral / num_events
            ).detach(),
            "mean_intensity": intensity.mean().detach(),
            "num_events": num_events.detach(),
        }

        return nll_per_event, metrics