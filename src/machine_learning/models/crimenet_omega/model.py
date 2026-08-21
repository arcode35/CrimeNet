from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        expansion: int,
        dropout: float,
    ) -> None:
        super().__init__()

        if dim <= 0:
            raise ValueError("dim must be > 0.")
        if expansion <= 0:
            raise ValueError("expansion must be > 0.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        hidden_dim = dim * expansion

        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class CrimeNetOmega0(nn.Module):
    """
    Context-only marked neural point process.

    lambda_m(s, t) = lambda(s, t) * p(m | s, t)

    Omega-0 deliberately contains no event-history, Hawkes, graph,
    latent-state, or higher-order history components.
    """

    SUPPORTED_ARCHITECTURE = "context_only_marked_point_process"
    SUPPORTED_INTENSITY_ACTIVATIONS = {"softplus"}
    SUPPORTED_INTENSITY_UNITS = {
        "events_per_cell_second",
        "events_per_cell_hour",
        "events_per_cell_day",
    }
    OMEGA0_DISABLED_COMPONENTS = (
        "aggregate_history",
        "raw_event_history",
        "hawkes",
        "multiscale_memory",
        "spatial_graph",
        "dynamic_graph",
        "latent_state",
        "higher_order_residual",
    )

    def __init__(
        self,
        *,
        model_cfg: Mapping[str, Any],
        num_numeric: int,
        num_cities: int,
        num_lighting_conditions: int,
        num_subtypes: int,
    ) -> None:
        super().__init__()

        self._validate_dimensions(
            num_numeric=num_numeric,
            num_cities=num_cities,
            num_lighting_conditions=num_lighting_conditions,
            num_subtypes=num_subtypes,
        )
        self._validate_config(model_cfg)

        self.model_name = str(model_cfg["name"])
        self.architecture = str(model_cfg["architecture"])
        self.hidden_dim = int(model_cfg["hidden_dim"])
        self.residual_blocks = int(model_cfg["residual_blocks"])
        self.residual_expansion = int(
            model_cfg["residual_expansion"]
        )
        self.dropout = float(model_cfg["dropout"])

        embeddings_cfg = model_cfg["embeddings"]
        self.city_embedding_dim = int(embeddings_cfg["city_dim"])
        self.lighting_embedding_dim = int(
            embeddings_cfg["lighting_dim"]
        )

        intensity_cfg = model_cfg["intensity"]
        self.intensity_activation = str(
            intensity_cfg["activation"]
        ).lower()
        self.intensity_epsilon = float(intensity_cfg["epsilon"])
        self.intensity_unit = str(intensity_cfg["unit"])

        mark_cfg = model_cfg["mark"]
        self.mark_target = str(mark_cfg["target"])
        self.mark_hierarchical = bool(mark_cfg["hierarchical"])

        self.city_embedding = nn.Embedding(
            num_embeddings=num_cities,
            embedding_dim=self.city_embedding_dim,
        )
        self.lighting_embedding = nn.Embedding(
            num_embeddings=num_lighting_conditions,
            embedding_dim=self.lighting_embedding_dim,
        )

        self.numeric_encoder = nn.Sequential(
            nn.Linear(num_numeric, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )

        fused_dim = (
            self.hidden_dim
            + self.city_embedding_dim
            + self.lighting_embedding_dim
        )

        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, self.hidden_dim),
            nn.GELU(),
        )

        self.blocks = nn.Sequential(
            *[
                ResidualBlock(
                    self.hidden_dim,
                    expansion=self.residual_expansion,
                    dropout=self.dropout,
                )
                for _ in range(self.residual_blocks)
            ]
        )

        self.intensity_head = nn.Linear(self.hidden_dim, 1)
        self.mark_head = nn.Linear(self.hidden_dim, num_subtypes)

    @staticmethod
    def _validate_dimensions(
        *,
        num_numeric: int,
        num_cities: int,
        num_lighting_conditions: int,
        num_subtypes: int,
    ) -> None:
        dimensions = {
            "num_numeric": num_numeric,
            "num_cities": num_cities,
            "num_lighting_conditions": num_lighting_conditions,
            "num_subtypes": num_subtypes,
        }
        for name, value in dimensions.items():
            if value <= 0:
                raise ValueError(
                    f"{name} must be > 0; got {value}."
                )

    @classmethod
    def _validate_config(
        cls,
        model_cfg: Mapping[str, Any],
    ) -> None:
        required_top_level = {
            "name",
            "architecture",
            "hidden_dim",
            "residual_blocks",
            "residual_expansion",
            "dropout",
            "embeddings",
            "intensity",
            "mark",
            "components",
        }
        missing = required_top_level - set(model_cfg)
        if missing:
            raise KeyError(
                f"Missing model config keys: {sorted(missing)}"
            )

        architecture = str(model_cfg["architecture"])
        if architecture != cls.SUPPORTED_ARCHITECTURE:
            raise ValueError(
                "CrimeNetOmega0 requires "
                f"architecture={cls.SUPPORTED_ARCHITECTURE!r}; "
                f"got {architecture!r}."
            )

        hidden_dim = int(model_cfg["hidden_dim"])
        residual_blocks = int(model_cfg["residual_blocks"])
        residual_expansion = int(model_cfg["residual_expansion"])
        dropout = float(model_cfg["dropout"])

        if hidden_dim <= 0:
            raise ValueError("model.hidden_dim must be > 0.")
        if residual_blocks < 0:
            raise ValueError(
                "model.residual_blocks must be >= 0."
            )
        if residual_expansion <= 0:
            raise ValueError(
                "model.residual_expansion must be > 0."
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "model.dropout must be in [0, 1)."
            )

        embeddings_cfg = model_cfg["embeddings"]
        required_embeddings = {"city_dim", "lighting_dim"}
        missing_embeddings = required_embeddings - set(embeddings_cfg)
        if missing_embeddings:
            raise KeyError(
                "Missing model.embeddings keys: "
                f"{sorted(missing_embeddings)}"
            )
        if int(embeddings_cfg["city_dim"]) <= 0:
            raise ValueError(
                "model.embeddings.city_dim must be > 0."
            )
        if int(embeddings_cfg["lighting_dim"]) <= 0:
            raise ValueError(
                "model.embeddings.lighting_dim must be > 0."
            )

        intensity_cfg = model_cfg["intensity"]
        required_intensity = {"activation", "epsilon", "unit"}
        missing_intensity = required_intensity - set(intensity_cfg)
        if missing_intensity:
            raise KeyError(
                "Missing model.intensity keys: "
                f"{sorted(missing_intensity)}"
            )

        activation = str(intensity_cfg["activation"]).lower()
        if activation not in cls.SUPPORTED_INTENSITY_ACTIVATIONS:
            raise ValueError(
                f"Unsupported intensity activation {activation!r}; "
                f"supported={sorted(cls.SUPPORTED_INTENSITY_ACTIVATIONS)}."
            )
        if float(intensity_cfg["epsilon"]) <= 0:
            raise ValueError(
                "model.intensity.epsilon must be > 0."
            )

        unit = str(intensity_cfg["unit"])
        if unit not in cls.SUPPORTED_INTENSITY_UNITS:
            raise ValueError(
                f"Unsupported intensity unit {unit!r}; "
                f"supported={sorted(cls.SUPPORTED_INTENSITY_UNITS)}."
            )

        mark_cfg = model_cfg["mark"]
        required_mark = {"target", "hierarchical"}
        missing_mark = required_mark - set(mark_cfg)
        if missing_mark:
            raise KeyError(
                f"Missing model.mark keys: {sorted(missing_mark)}"
            )
        if bool(mark_cfg["hierarchical"]):
            raise ValueError(
                "Omega-0 implements a flat subtype mark head. "
                "Set model.mark.hierarchical=false."
            )

        components_cfg = model_cfg["components"]
        for component in cls.OMEGA0_DISABLED_COMPONENTS:
            if component not in components_cfg:
                raise KeyError(
                    f"Missing model.components.{component}"
                )
            if bool(components_cfg[component]):
                raise ValueError(
                    f"Omega-0 does not implement component "
                    f"{component!r}; it must remain false."
                )

    def initialize_base_rate(self, base_rate: float) -> None:
        """
        Initialize the intensity head to an approximately constant empirical
        base rate in the configured intensity unit.
        """
        rate = max(float(base_rate), self.intensity_epsilon)

        if self.intensity_activation == "softplus":
            raw_bias = (
                rate
                if rate > 20.0
                else math.log(math.expm1(rate))
            )
        else:
            raise RuntimeError(
                "Unsupported validated intensity activation."
            )

        nn.init.zeros_(self.intensity_head.weight)
        with torch.no_grad():
            self.intensity_head.bias.fill_(raw_bias)

    def _activate_intensity(
        self,
        raw_intensity: torch.Tensor,
    ) -> torch.Tensor:
        if self.intensity_activation == "softplus":
            return (
                F.softplus(raw_intensity)
                + self.intensity_epsilon
            )
        raise RuntimeError(
            "Unsupported validated intensity activation."
        )

    def forward(
        self,
        *,
        numeric: torch.Tensor,
        city: torch.Tensor,
        lighting: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        x_numeric = self.numeric_encoder(numeric)
        x_city = self.city_embedding(city)
        x_lighting = self.lighting_embedding(lighting)

        h = torch.cat(
            [x_numeric, x_city, x_lighting],
            dim=-1,
        )
        h = self.fusion(h)
        h = self.blocks(h)

        raw_intensity = self.intensity_head(h).squeeze(-1)
        intensity = self._activate_intensity(raw_intensity)
        mark_logits = self.mark_head(h)

        return {
            "intensity": intensity,
            "mark_logits": mark_logits,
            "context_state": h,
        }