"""Per-hazard output heads producing multi-lead logit maps."""

from __future__ import annotations

import torch
from torch import nn

from nowcast import schema

# Number of continuous terrain planes the flash-flood head consumes, produced by
# :func:`nowcast.data.transforms.transform_terrain` (standardised log1p flow
# accumulation + standardised HAND). This is deliberately *not*
# ``len(schema.FLASH_FLOOD_EXTRA_CHANNELS)``: the categorical D8 ``flow_direction``
# pointer in that schema tuple is a datacube contract entry, not a model input.
N_TERRAIN_PLANES: int = 2


class HazardHead(nn.Module):
    """Convolutional head: features ``(B, F, H, W)`` -> logits ``(B, n_leads, H, W)``."""

    def __init__(
        self, in_features: int, n_leads: int = schema.N_LEADS, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.n_leads = n_leads
        self.net = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, padding=1),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(in_features, n_leads, 1),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """Return per-lead logits ``(B, n_leads, H, W)``."""
        return self.net(feats)


class FlashFloodHead(nn.Module):
    """Flash-flood head: consumes shared features *and* routed-terrain context.

    ``terrain`` is a ``(B, n_terrain, H, W)`` tensor of transformed routed-terrain
    context (see :func:`nowcast.data.transforms.transform_terrain` --
    standardised ``log1p`` flow accumulation + standardised HAND, ``n_terrain ==
    N_TERRAIN_PLANES``). It is encoded and concatenated with the shared features
    before the final projection to per-lead logits.
    """

    def __init__(
        self,
        in_features: int,
        n_terrain: int | None = None,
        n_leads: int = schema.N_LEADS,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_terrain = n_terrain if n_terrain is not None else N_TERRAIN_PLANES
        self.n_leads = n_leads
        self.terrain_encoder = nn.Sequential(
            nn.Conv2d(self.n_terrain, in_features, 3, padding=1),
            nn.GELU(),
        )
        self.net = nn.Sequential(
            nn.Conv2d(2 * in_features, in_features, 3, padding=1),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(in_features, n_leads, 1),
        )

    def forward(self, feats: torch.Tensor, terrain: torch.Tensor) -> torch.Tensor:
        """Return per-lead logits ``(B, n_leads, H, W)``."""
        encoded = self.terrain_encoder(terrain)
        return self.net(torch.cat([feats, encoded], dim=1))
