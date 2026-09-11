"""Per-channel normalisation statistics and the ``Normalizer`` transform."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nowcast import config, schema

# Continuous terrain planes fed to the flash-flood head, produced by
# :func:`transform_terrain`: standardised ``log1p`` flow accumulation and
# standardised HAND. The categorical ESRI D8 ``flow_direction`` pointer that is
# part of :data:`nowcast.schema.FLASH_FLOOD_EXTRA_CHANNELS` is deliberately
# excluded -- it is nominal and meaningless as a convolution input.
TERRAIN_PLANES: int = 2


def compute_norm_stats(
    features_ds: Any, *, date_range: tuple[str, str] | None = None
) -> dict[str, dict[str, float]]:
    """Return ``{channel: {"mean": .., "std": ..}}`` for every ``FEATURE_CHANNELS``.

    ``features_ds`` is an xarray ``Dataset`` (or any mapping exposing the channel
    names as arrays). When ``date_range`` (inclusive ISO ``(start, end)``) is
    given and the dataset has a ``time`` coordinate, statistics are computed over
    that window only (train-range stats -- no val/test leakage). A near-zero
    standard deviation is clamped to ``1.0`` so :class:`Normalizer` never divides
    by zero.
    """
    features_ds = _slice_time(features_ds, date_range)
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


def _slice_time(ds: Any, date_range: tuple[str, str] | None) -> Any:
    """Return ``ds`` restricted to an inclusive ``(start, end)`` time window."""
    if date_range is None:
        return ds
    coords = getattr(ds, "coords", {})
    if "time" not in coords:
        return ds
    return ds.sel(time=slice(date_range[0], date_range[1]))


def _standardise(arr: np.ndarray) -> np.ndarray:
    std = float(np.nanstd(arr))
    return (arr - float(np.nanmean(arr))) / (std if std > 1e-6 else 1.0)


def transform_terrain(
    flow_accumulation: np.ndarray, hand: np.ndarray
) -> np.ndarray:
    """Return the continuous terrain tensor ``(TERRAIN_PLANES, H, W)``.

    Plane 0 is ``log1p(flow_accumulation)`` then standardised (accumulation is
    heavy-tailed, ranging into the thousands). Plane 1 is standardised HAND. See
    :data:`TERRAIN_PLANES` for why ``flow_direction`` is not included.
    """
    acc = np.log1p(np.clip(np.asarray(flow_accumulation, dtype="float64"), 0.0, None))
    hand_arr = np.asarray(hand, dtype="float64")
    return np.stack([_standardise(acc), _standardise(hand_arr)]).astype("float32")


def resolve_norm_stats(
    features_ds: Any,
    *,
    date_range: tuple[str, str] | None = None,
    path: str | Path = config.NORM_STATS_PATH,
) -> dict[str, dict[str, float]]:
    """Load stats from ``path`` if present, else compute (over ``date_range``) and persist.

    This is the train-time hook: the first run computes per-channel statistics
    from the training window and writes ``path``; later runs just load it.
    """
    path = Path(path)
    if path.exists():
        return load_norm_stats(path)
    stats = compute_norm_stats(features_ds, date_range=date_range)
    save_norm_stats(stats, path)
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
