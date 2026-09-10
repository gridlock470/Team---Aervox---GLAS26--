"""Reusable spatiotemporal building blocks: ConvLSTM cells and Conv3d blocks."""

from __future__ import annotations

import torch
from torch import nn


def num_groups(channels: int, maximum: int = 8) -> int:
    """Return the largest divisor of ``channels`` not exceeding ``maximum`` (>= 1)."""
    for group in range(min(maximum, channels), 0, -1):
        if channels % group == 0:
            return group
    return 1


class ConvLSTMCell(nn.Module):
    """A single convolutional-LSTM cell (Shi et al., 2015)."""

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=kernel_size // 2,
        )

    def forward(
        self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance the cell by one timestep.

        ``x`` is ``(B, in_channels, H, W)``; ``state`` is ``(h, c)`` each
        ``(B, hidden_channels, H, W)``. Returns the updated ``(h, c)``.
        """
        h_prev, c_prev = state
        gates = self.conv(torch.cat([x, h_prev], dim=1))
        input_gate, forget_gate, cell_gate, output_gate = torch.chunk(gates, 4, dim=1)
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        cell_gate = torch.tanh(cell_gate)
        output_gate = torch.sigmoid(output_gate)
        c = forget_gate * c_prev + input_gate * cell_gate
        h = output_gate * torch.tanh(c)
        return h, c

    def init_state(
        self, batch: int, size: tuple[int, int], device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return zero-initialised ``(h, c)`` for a fresh sequence."""
        h = torch.zeros(batch, self.hidden_channels, *size, device=device, dtype=dtype)
        return h, torch.zeros_like(h)


class ConvLSTMBlock(nn.Module):
    """Run a :class:`ConvLSTMCell` over the time axis of a ``(B, T, C, H, W)`` tensor.

    Returns either the last hidden state ``(B, hidden, H, W)`` or, when
    ``return_sequence`` is set, the full sequence ``(B, T, hidden, H, W)``.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        *,
        return_sequence: bool = False,
    ) -> None:
        super().__init__()
        self.cell = ConvLSTMCell(in_channels, hidden_channels, kernel_size)
        self.hidden_channels = hidden_channels
        self.return_sequence = return_sequence

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """See the class docstring for shape semantics."""
        batch, steps, _, height, width = x.shape
        state = self.cell.init_state(batch, (height, width), x.device, x.dtype)
        outputs: list[torch.Tensor] = []
        hidden = state[0]
        for step in range(steps):
            hidden, cell = self.cell(x[:, step], state)
            state = (hidden, cell)
            if self.return_sequence:
                outputs.append(hidden)
        if self.return_sequence:
            return torch.stack(outputs, dim=1)
        return hidden


class Conv3dBlock(nn.Module):
    """``Conv3d`` -> ``GroupNorm`` -> ``GELU`` on ``(B, C, T, H, W)`` tensors."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(num_groups(out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block; temporal and spatial dims are preserved."""
        return self.block(x)
