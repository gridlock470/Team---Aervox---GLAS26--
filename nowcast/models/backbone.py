"""Shared spatiotemporal backbone (ConvLSTM encoder + conv head)."""

from __future__ import annotations

import torch
from torch import nn

from nowcast import schema
from nowcast.models.blocks import ConvLSTMBlock, num_groups


class SpatiotemporalBackbone(nn.Module):
    """Encode ``(B, C, T, H, W)`` into shared features ``(B, F, H, W)``.

    A stack of :class:`~nowcast.models.blocks.ConvLSTMBlock` layers consumes the
    history axis; the final hidden state is refined by a small conv head. The
    spatial resolution ``H, W`` is preserved end to end. ``out_features`` reports
    the channel count ``F`` of the returned feature map.
    """

    def __init__(
        self,
        in_channels: int = schema.N_CHANNELS,
        hidden: int = 16,
        depth: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden = hidden
        self.out_features = hidden
        blocks: list[ConvLSTMBlock] = []
        prev = in_channels
        for layer in range(depth):
            is_last = layer == depth - 1
            blocks.append(
                ConvLSTMBlock(prev, hidden, kernel_size, return_sequence=not is_last)
            )
            prev = hidden
        self.encoder = nn.ModuleList(blocks)
        self.head = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(num_groups(hidden), hidden),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, C, T, H, W)`` -> ``(B, out_features, H, W)``."""
        z = x.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W) for the ConvLSTM stack
        for block in self.encoder:
            z = block(z)
        return self.head(z)
