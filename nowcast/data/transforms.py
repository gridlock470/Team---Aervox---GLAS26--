"""Per-channel normalisation statistics and the ``Normalizer`` transform."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nowcast import config, schema


def compute_norm_stats(features_ds: Any) -> dict[str, dict[str, float]]:
    """Return ``{channel: {"mean": .., "std": ..}}`` for every ``FEATURE_CHANNELS``.

    ``features_ds`` is an xarray ``Dataset`` (or any mapping exposing the channel
    names as arrays). A near-zero standard deviation is clamped to ``1.0`` so the
    :class:`Normalizer` never divides by zero.
    """
    stats: dict[str, dict[str, float]] = {}
    for channel in schema.FEATURE_CHANNELS:
        if channel not in features_ds:
            raise KeyError(f"feature channel '{channel}' missing from dataset")
        values = np.asarray(features_ds[channel].values, dtype="float64")
        std = float(np.nanstd(values))
        stats[channel] = {
            "mean": float(np.nanmean(values)),
            "std": std if std > 1e-6 else 1.0,
        }
    return stats


def save_norm_stats(
    stats: dict[str, dict[str, float]], path: str | Path = config.NORM_STATS_PATH
) -> Path:
    """Write ``stats`` to ``path`` as JSON and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2, sort_keys=True))
    return path


def load_norm_stats(path: str | Path = config.NORM_STATS_PATH) -> dict[str, dict[str, float]]:
    """Load normalisation stats previously written by :func:`save_norm_stats`."""
    return json.loads(Path(path).read_text())


class Normalizer:
    """Standardise a feature tensor channel-wise: ``(x - mean) / std``.

    Accepts ``(C, T, H, W)`` or ``(C, H, W)`` NumPy arrays or torch tensors; the
    channel axis is first and must match :data:`nowcast.schema.FEATURE_CHANNELS`.
    :meth:`inverse` undoes the transform.
    """

    def __init__(self, stats: dict[str, dict[str, float]]) -> None:
        self.stats = stats
        self.mean = np.asarray(
            [stats[c]["mean"] for c in schema.FEATURE_CHANNELS], dtype="float32"
        )
        self.std = np.asarray(
            [stats[c]["std"] for c in schema.FEATURE_CHANNELS], dtype="float32"
        )

    @staticmethod
    def _channel_shape(x: np.ndarray | torch.Tensor) -> tuple[int, ...]:
        return (-1,) + (1,) * (x.ndim - 1)

    def _params(
        self, x: np.ndarray | torch.Tensor
    ) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
        shape = self._channel_shape(x)
        if isinstance(x, torch.Tensor):
            mean = torch.as_tensor(self.mean, device=x.device, dtype=x.dtype)
            std = torch.as_tensor(self.std, device=x.device, dtype=x.dtype)
            return mean.reshape(shape), std.reshape(shape)
        return self.mean.reshape(shape), self.std.reshape(shape)

    def __call__(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Normalise ``x``."""
        mean, std = self._params(x)
        if isinstance(x, torch.Tensor):
            return (x - mean) / std
        return (np.asarray(x, dtype="float32") - mean) / std

    def inverse(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Undo :meth:`__call__`."""
        mean, std = self._params(x)
        if isinstance(x, torch.Tensor):
            return x * std + mean
        return np.asarray(x, dtype="float32") * std + mean
