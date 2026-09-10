"""Per-hazard output heads producing multi-lead logit maps."""

from __future__ import annotations

import torch
from torch import nn

from nowcast import schema


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

    ``terrain`` is a ``(B, n_terrain, H, W)`` tensor carrying the
    :data:`nowcast.schema.FLASH_FLOOD_EXTRA_CHANNELS` (flow accumulation, flow
    direction, HAND). It is encoded and concatenated with the shared features
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
        self.n_terrain = (
            n_terrain if n_terrain is not None else len(schema.FLASH_FLOOD_EXTRA_CHANNELS)
        )
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
