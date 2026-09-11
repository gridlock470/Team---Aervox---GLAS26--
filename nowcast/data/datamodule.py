"""PyTorch/Lightning data plumbing over the feature datacube and label maps."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset

from nowcast import config, schema
from nowcast.data.transforms import (
    Normalizer,
    compute_norm_stats,
    resolve_norm_stats,
    transform_terrain,
)
from nowcast.features.labels import valid_label_times

_TERRAIN_VARS: tuple[str, str] = ("flow_accumulation", "hand")


def _as_dataset(obj: Any) -> xr.Dataset:
    if isinstance(obj, (str, Path)):
        return xr.open_zarr(obj)
    return obj


def _as_dataarray(obj: Any) -> xr.DataArray:
    if isinstance(obj, (str, Path)):
        opened = xr.open_zarr(obj)
        return opened[next(iter(opened.data_vars))]
    if isinstance(obj, xr.Dataset):
        return obj[next(iter(obj.data_vars))]
    return obj


def _resolve_terrain(terrain: Any, features: xr.Dataset) -> np.ndarray:
    """Return the transformed terrain tensor ``(TERRAIN_PLANES, H, W)``.

    ``terrain`` may be:

    * an already-transformed ``np.ndarray`` ``(P, H, W)`` -- used verbatim;
    * an xarray ``Dataset`` or a Zarr path exposing ``flow_accumulation`` and
      ``hand`` (e.g. the DEM routing store at ``config.DEM_ROUTING_PATH`` or the
      datacube static vars) -- transformed here;
    * ``None`` -- the two continuous terrain vars are read from ``features``
      (they are part of ``schema.FEATURE_CHANNELS``); this keeps the synthetic
      path and any datacube that carries them working without a separate store.
    """
    if isinstance(terrain, np.ndarray):
        return terrain.astype("float32")
    source = _as_dataset(terrain) if terrain is not None else features
    missing = [v for v in _TERRAIN_VARS if v not in source]
    if missing:
        raise KeyError(
            f"terrain source is missing {missing}; pass an explicit `terrain=` "
            "Dataset/path (e.g. config.DEM_ROUTING_PATH)"
        )
    return transform_terrain(
        source["flow_accumulation"].values, source["hand"].values
    )


class NowcastDataset(Dataset):
    """Sliding-window samples over a feature datacube and its label maps.

    Item ``idx`` is ``(x, y, terrain)`` where

    * ``x`` -- ``(N_CHANNELS, INPUT_SEQ_LEN, H, W)`` history of
      :data:`nowcast.schema.FEATURE_CHANNELS` (static channels broadcast over
      time), normalised when a :class:`Normalizer` is supplied;
    * ``y`` -- ``(N_HAZARDS, N_LEADS, H, W)`` occurrence probabilities at the
      analysis time (the last input step), one channel per
      :data:`nowcast.config.LEAD_TIMES_H`;
    * ``terrain`` -- ``(TERRAIN_PLANES, H, W)`` transformed routed-terrain context
      (standardised ``log1p`` flow accumulation + standardised HAND); see
      :func:`nowcast.data.transforms.transform_terrain`.

    ``features``/``labels`` may be in-memory xarray objects or Zarr paths;
    ``terrain`` is an independent source (see :func:`_resolve_terrain`).

    A window is only served when its analysis time (the last input step) has a
    *real* label -- :func:`nowcast.features.labels.valid_label_times` is
    recomputed on this dataset's own (already-sliced) ``time`` coordinate, so a
    window whose ``max(LEAD_TIMES_H)``-hour horizon would run past this split
    (or across a time gap) is excluded. This must happen *after* slicing to a
    split: reusing a validity mask computed on the pre-split series would
    under-restrict the split's tail and either serve a fabricated zero-padded
    label or, worse, a real future occurrence that actually belongs to the next
    split -- leaking information across the chronological boundary (F19).
    """

    def __init__(
        self,
        features: Any,
        labels: Any,
        norm: Normalizer | None = None,
        terrain: Any | None = None,
    ) -> None:
        self.features = _as_dataset(features)
        self.labels = _as_dataarray(labels)
        self.norm = norm
        self.seq_len = config.INPUT_SEQ_LEN
        self.n_times = int(self.features.sizes["time"])
        window_count = self.n_times - self.seq_len + 1
        if window_count <= 0:
            raise ValueError(
                f"need at least {self.seq_len} time steps, got {self.n_times}"
            )
        valid = np.asarray(
            valid_label_times(self.features["time"]).values, dtype=bool
        )
        analysis_positions = np.arange(window_count) + self.seq_len - 1
        self._window_starts = np.flatnonzero(valid[analysis_positions])
        self.terrain = _resolve_terrain(terrain, self.features)

    def __len__(self) -> int:
        return int(self._window_starts.size)

    def time_values(self) -> np.ndarray:
        """The ``time`` coordinate of this split (used to assert split disjointness)."""
        return np.asarray(self.features["time"].values)

    def _series(self, channel: str, start: int, stop: int) -> np.ndarray:
        array = self.features[channel]
        if "time" in array.dims:
            values = array.isel(time=slice(start, stop)).values
        else:
            values = np.broadcast_to(array.values, (stop - start, *array.shape))
        return np.asarray(values, dtype="float32")

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = int(self._window_starts[i])
        start, stop = idx, idx + self.seq_len
        x = np.stack([self._series(c, start, stop) for c in schema.FEATURE_CHANNELS])
        y = np.asarray(self.labels.isel(time=stop - 1).values, dtype="float32")
        x_t = torch.from_numpy(np.ascontiguousarray(x))
        if self.norm is not None:
            x_t = self.norm(x_t)
        y_t = torch.from_numpy(np.ascontiguousarray(y))
        terrain_t = torch.from_numpy(np.ascontiguousarray(self.terrain))
        return x_t, y_t, terrain_t


class NowcastDataModule(L.LightningDataModule):
    """Split the datacube on the ``time`` coord by date range and serve loaders.

    Pass ``features``/``labels`` (xarray objects or Zarr paths) plus a ``terrain``
    source to split by :data:`nowcast.config.TRAIN_DATE_RANGE` /
    ``VAL_DATE_RANGE`` / ``TEST_DATE_RANGE`` in :meth:`setup`, or inject
    ready-made datasets (see :meth:`from_synthetic`) so tests need no Zarr.
    ``setup`` resolves train-range normalisation stats (compute-if-missing) into
    ``config.NORM_STATS_PATH`` unless a ``norm`` is supplied.
    """

    def __init__(
        self,
        features: Any | None = None,
        labels: Any | None = None,
        norm: Normalizer | None = None,
        terrain: Any | None = None,
        batch_size: int = 4,
        num_workers: int = 0,
        *,
        norm_stats_path: str | Path = config.NORM_STATS_PATH,
        train_dataset: Dataset | None = None,
        val_dataset: Dataset | None = None,
        test_dataset: Dataset | None = None,
    ) -> None:
        super().__init__()
        self._features = features
        self._labels = labels
        self.norm = norm
        self.terrain = terrain
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._norm_stats_path = norm_stats_path
        self._train = train_dataset
        self._val = val_dataset
        self._test = test_dataset

    def _subset(
        self,
        features: xr.Dataset,
        labels: xr.DataArray,
        date_range: tuple[str, str],
    ) -> NowcastDataset | None:
        start = np.datetime64(date_range[0], "ns")
        end_exclusive = np.datetime64(date_range[1], "ns") + np.timedelta64(1, "D")
        times = np.asarray(features["time"].values, dtype="datetime64[ns]")
        mask = (times >= start) & (times < end_exclusive)
        if not mask.any():
            return None
        f_sub = features.isel(time=mask)
        # A window also needs max(LEAD_TIMES_H) hours past its analysis time to
        # survive the F19 horizon filter in NowcastDataset -- see there.
        min_span = config.INPUT_SEQ_LEN + max(config.LEAD_TIMES_H)
        if int(f_sub.sizes["time"]) < min_span:
            return None
        return NowcastDataset(
            f_sub, labels.isel(time=mask), norm=self.norm, terrain=self.terrain
        )

    def setup(self, stage: str | None = None) -> None:
        """Build train/val/test datasets (no-op when datasets were injected)."""
        if self._train is not None or self._features is None:
            return
        features = _as_dataset(self._features)
        labels = _as_dataarray(self._labels)
        if self.norm is None:
            self.norm = Normalizer(
                resolve_norm_stats(
                    features,
                    date_range=config.TRAIN_DATE_RANGE,
                    path=self._norm_stats_path,
                )
            )
        self._train = self._subset(features, labels, config.TRAIN_DATE_RANGE)
        self._val = self._subset(features, labels, config.VAL_DATE_RANGE)
        self._test = self._subset(features, labels, config.TEST_DATE_RANGE)
        self._assert_disjoint()

    def _assert_disjoint(self) -> None:
        if self._val is None or self._test is None:
            return
        overlap = set(self._val.time_values().tolist()) & set(
            self._test.time_values().tolist()
        )
        if overlap:
            raise ValueError("VAL_DATE_RANGE and TEST_DATE_RANGE overlap in time")

    @classmethod
    def from_synthetic(
        cls,
        n_hours: int = 90,
        *,
        seed: int = 0,
        batch_size: int = 2,
        norm: Normalizer | None = None,
    ) -> NowcastDataModule:
        """Build a datamodule from synthetic xarray data using contiguous splits."""
        from nowcast.testing import synthetic

        features = synthetic.make_feature_datacube(n_hours=n_hours, seed=seed)
        labels = synthetic.make_targets(n_hours=n_hours, seed=seed)
        # +max(LEAD_TIMES_H) so each split has room for >= 1 sample surviving
        # the F19 label-horizon filter (the last max_lead window starts of
        # every split are always invalid -- see NowcastDataset).
        window = config.INPUT_SEQ_LEN + max(config.LEAD_TIMES_H) + 2
        if n_hours < 3 * window:
            raise ValueError(f"n_hours must be >= {3 * window} for three splits")
        bounds = (
            slice(0, n_hours - 2 * window),
            slice(n_hours - 2 * window, n_hours - window),
            slice(n_hours - window, n_hours),
        )
        if norm is None:
            # Train-slice statistics only -- no val/test leakage (F16).
            norm = Normalizer(compute_norm_stats(features.isel(time=bounds[0])))
        datasets = [
            NowcastDataset(features.isel(time=s), labels.isel(time=s), norm=norm)
            for s in bounds
        ]
        return cls(
            batch_size=batch_size,
            train_dataset=datasets[0],
            val_dataset=datasets[1],
            test_dataset=datasets[2],
        )

    def _loader(self, dataset: Dataset | None, *, shuffle: bool) -> DataLoader:
        if dataset is None:
            raise RuntimeError("call setup() before requesting a dataloader")
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self._train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self._val, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self._test, shuffle=False)
