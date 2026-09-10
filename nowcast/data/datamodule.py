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
from nowcast.data.transforms import Normalizer, compute_norm_stats


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


class NowcastDataset(Dataset):
    """Sliding-window samples over a feature datacube and its label maps.

    Item ``idx`` is ``(x, y, terrain)`` where

    * ``x`` -- ``(N_CHANNELS, INPUT_SEQ_LEN, H, W)`` history of
      :data:`nowcast.schema.FEATURE_CHANNELS` (static channels broadcast over
      time);
    * ``y`` -- ``(N_HAZARDS, N_LEADS, H, W)`` occurrence probabilities at the
      analysis time (the last input step), one channel per
      :data:`nowcast.config.LEAD_TIMES_H`;
    * ``terrain`` -- ``(n_terrain, H, W)`` routed-terrain context
      (:data:`nowcast.schema.FLASH_FLOOD_EXTRA_CHANNELS`).

    ``features``/``labels`` may be in-memory xarray objects or Zarr paths.
    """

    def __init__(
        self,
        features: Any,
        labels: Any,
        norm: Normalizer | None = None,
        terrain: np.ndarray | None = None,
    ) -> None:
        self.features = _as_dataset(features)
        self.labels = _as_dataarray(labels)
        self.norm = norm
        self.seq_len = config.INPUT_SEQ_LEN
        self.n_times = int(self.features.sizes["time"])
        self._length = self.n_times - self.seq_len + 1
        if self._length <= 0:
            raise ValueError(
                f"need at least {self.seq_len} time steps, got {self.n_times}"
            )
        if terrain is not None:
            self.terrain = np.asarray(terrain, dtype="float32")
        else:
            self.terrain = np.stack(
                [
                    np.asarray(self.features[channel].values, dtype="float32")
                    for channel in schema.FLASH_FLOOD_EXTRA_CHANNELS
                ]
            )

    def __len__(self) -> int:
        return self._length

    def _series(self, channel: str, start: int, stop: int) -> np.ndarray:
        array = self.features[channel]
        if "time" in array.dims:
            values = array.isel(time=slice(start, stop)).values
        else:
            values = np.broadcast_to(array.values, (stop - start, *array.shape))
        return np.asarray(values, dtype="float32")

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    """Split the datacube by calendar year and serve ``DataLoader``s.

    Pass ``features``/``labels`` (xarray objects or Zarr paths) to split by
    :data:`nowcast.config.TRAIN_YEARS` / ``VAL_YEARS`` / ``TEST_YEARS`` in
    :meth:`setup`, or inject ready-made datasets (see :meth:`from_synthetic`) so
    tests need no Zarr.
    """

    def __init__(
        self,
        features: Any | None = None,
        labels: Any | None = None,
        norm: Normalizer | None = None,
        terrain: np.ndarray | None = None,
        batch_size: int = 4,
        num_workers: int = 0,
        *,
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
        self._train = train_dataset
        self._val = val_dataset
        self._test = test_dataset

    def _subset(
        self, features: xr.Dataset, labels: xr.DataArray, years: tuple[int, ...]
    ) -> NowcastDataset | None:
        mask = np.isin(features["time"].dt.year.values, np.asarray(years))
        if not mask.any():
            return None
        f_sub = features.isel(time=mask)
        if int(f_sub.sizes["time"]) < config.INPUT_SEQ_LEN + 1:
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
        self._train = self._subset(features, labels, config.TRAIN_YEARS)
        self._val = self._subset(features, labels, config.VAL_YEARS)
        self._test = self._subset(features, labels, config.TEST_YEARS)

    @classmethod
    def from_synthetic(
        cls,
        n_hours: int = 56,
        *,
        seed: int = 0,
        batch_size: int = 2,
        norm: Normalizer | None = None,
    ) -> NowcastDataModule:
        """Build a datamodule from synthetic xarray data using contiguous splits."""
        from nowcast.testing import synthetic

        features = synthetic.make_feature_datacube(n_hours=n_hours, seed=seed)
        labels = synthetic.make_targets(n_hours=n_hours, seed=seed)
        if norm is None:
            norm = Normalizer(compute_norm_stats(features))
        window = config.INPUT_SEQ_LEN + 2
        if n_hours < 3 * window:
            raise ValueError(f"n_hours must be >= {3 * window} for three splits")
        bounds = (
            slice(0, n_hours - 2 * window),
            slice(n_hours - 2 * window, n_hours - window),
            slice(n_hours - window, n_hours),
        )
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
