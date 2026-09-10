"""Experimental transformer backbone -- an upgrade path, not the default.

:class:`~nowcast.models.multitask.MultiTaskNowcastNet` uses the ConvLSTM
:class:`~nowcast.models.backbone.SpatiotemporalBackbone`. This module is a
skeleton kept importable and forward-runnable at tiny size so a patch-embed +
temporal-attention backbone can be benchmarked against it later.
"""

from __future__ import annotations

import torch
from torch import nn

from nowcast import schema


class SpatiotemporalTransformerBackbone(nn.Module):
    """Patch-embed each frame, then self-attend over the time axis.

    Maps ``(B, C, T, H, W)`` -> ``(B, embed_dim, H, W)`` using the final temporal
    token per spatial location. NOT wired into the production model -- see the
    module docstring.
    """

    def __init__(
        self,
        in_channels: int = schema.N_CHANNELS,
        embed_dim: int = 16,
        patch_size: int = 1,
        n_heads: int = 2,
        depth: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.out_features = embed_dim
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, patch_size, stride=patch_size)
        encoder_layer = nn.TransformerEncoderLayer(
            embed_dim,
            n_heads,
            dim_feedforward=2 * embed_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.unembed = nn.ConvTranspose2d(
            embed_dim, embed_dim, patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, C, T, H, W)`` -> ``(B, embed_dim, H, W)``."""
        batch, channels, steps, height, width = x.shape
        frames = x.permute(0, 2, 1, 3, 4).reshape(batch * steps, channels, height, width)
        tokens = self.patch_embed(frames)
        patch_h, patch_w = tokens.shape[-2:]
        tokens = tokens.reshape(batch, steps, self.embed_dim, patch_h * patch_w)
        tokens = tokens.permute(0, 3, 1, 2).reshape(
            batch * patch_h * patch_w, steps, self.embed_dim
        )
        tokens = self.temporal_encoder(tokens)
        last = tokens[:, -1]
        grid = last.reshape(batch, patch_h * patch_w, self.embed_dim).permute(0, 2, 1)
        grid = grid.reshape(batch, self.embed_dim, patch_h, patch_w)
        out = self.unembed(grid)
        return out[..., :height, :width]

    def cross_attention(
        self, insat_tokens: torch.Tensor, imdaa_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Planned INSAT<->IMDAA cross-attention fusion (NOT IMPLEMENTED).

        Design intent: fast-updating INSAT satellite tokens act as queries
        against the slower IMDAA reanalysis tokens (keys/values) so the
        cloud-top signal can attend to the dynamical state before decoding.
        Wire this in when the transformer backbone becomes the default; until
        then it raises :class:`NotImplementedError`.
        """
        raise NotImplementedError("cross_attention is a documented upgrade-path stub")
