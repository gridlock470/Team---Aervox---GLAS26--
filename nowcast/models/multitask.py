"""Multi-task nowcasting network: one shared backbone, one head per hazard."""

from __future__ import annotations

import torch
from torch import nn

from nowcast import config
from nowcast.models.backbone import SpatiotemporalBackbone
from nowcast.models.heads import FlashFloodHead, HazardHead


class MultiTaskNowcastNet(nn.Module):
    """Shared :class:`SpatiotemporalBackbone` + a ``ModuleDict`` of hazard heads.

    ``forward`` returns a dict keyed by :data:`nowcast.config.HAZARDS`; each value
    is a ``(B, N_LEADS, H, W)`` logit map. :meth:`stack_logits` and
    :meth:`predict_proba` stack those into ``(B, N_HAZARDS, N_LEADS, H, W)`` to
    match :data:`nowcast.schema.TARGET_SHAPE` with a batch dim. The
    ``flash_flood`` hazard uses a :class:`FlashFloodHead` that additionally
    consumes a ``(B, n_terrain, H, W)`` terrain tensor.
    """

    def __init__(self, backbone_kwargs: dict | None = None) -> None:
        super().__init__()
        self.backbone = SpatiotemporalBackbone(**(backbone_kwargs or {}))
        features = self.backbone.out_features
        heads: dict[str, nn.Module] = {}
        for hazard in config.HAZARDS:
            if hazard == "flash_flood":
                heads[hazard] = FlashFloodHead(features)
            else:
                heads[hazard] = HazardHead(features)
        self.heads = nn.ModuleDict(heads)

    def forward(
        self, x: torch.Tensor, terrain: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Run the backbone once and every head; see the class docstring."""
        feats = self.backbone(x)
        batch, _, height, width = feats.shape
        out: dict[str, torch.Tensor] = {}
        for hazard, head in self.heads.items():
            if isinstance(head, FlashFloodHead):
                terrain_in = (
                    terrain
                    if terrain is not None
                    else feats.new_zeros(batch, head.n_terrain, height, width)
                )
                out[hazard] = head(feats, terrain_in)
            else:
                out[hazard] = head(feats)
        return out

    @staticmethod
    def stack_logits(out_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """Stack a head-output dict into ``(B, N_HAZARDS, N_LEADS, H, W)``."""
        return torch.stack([out_dict[hazard] for hazard in config.HAZARDS], dim=1)

    def predict_proba(
        self, x: torch.Tensor, terrain: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Sigmoid probabilities, shape ``(B, N_HAZARDS, N_LEADS, H, W)``."""
        return torch.sigmoid(self.stack_logits(self.forward(x, terrain)))
