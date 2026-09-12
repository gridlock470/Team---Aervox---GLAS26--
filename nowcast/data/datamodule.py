"""PyTorch/Lightning data plumbing over the feature datacube and label maps."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler

from nowcast import config, schema
from nowcast.data.transforms import (
    Normalizer,
    compute_norm_stats,
    resolve_norm_stats,
    transform_terrain,
)
from nowcast.features.labels import valid_label_times

_TERRAIN_VARS: tuple[str, str] = ("flow_accumulation", "hand")

# Analysis times scanned per pass when building the storm catalogue. A full
# train split is ~17k hours and a label field is (hazard, lead, lat, lon), so
# materialising every label at once would cost gigabytes for one boolean each.
_CATALOGUE_CHUNK: int = 256


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

    ``__init__`` also builds a per-window *storm catalogue*
    (:attr:`event_flags`) once, which :meth:`sample_weights` turns into the
    event-balanced train sampler; see :meth:`NowcastDataModule.train_sampler`.
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
        self._analysis_times = self._window_starts + self.seq_len - 1
        self.event_flags = self._build_event_flags()
        self.terrain = _resolve_terrain(terrain, self.features)

    def __len__(self) -> int:
        return int(self._window_starts.size)

    def _build_event_flags(self) -> np.ndarray:
        """Storm catalogue: ``(n_windows, n_hazards)`` bool, computed once.

        Entry ``[i, h]`` is True when this window's label holds at least one
        cell at or above :data:`nowcast.config.LABEL_OCCURRENCE_THRESHOLD` for
        hazard ``h``, reduced over ``lead``/``lat``/``lon``.

        Row ``i`` describes window ``i`` in ``_window_starts`` space -- the same
        index :meth:`__getitem__` takes -- *not* a raw time index. Anything
        built on this therefore inherits the F19 horizon filter instead of
        silently re-admitting the windows that filter dropped.
        """
        reduce_dims = [d for d in self.labels.dims if d not in ("time", "hazard")]
        n_hazards = int(self.labels.sizes["hazard"])
        n_windows = int(self._analysis_times.size)
        flags = np.zeros((n_windows, n_hazards), dtype=bool)
        for lo in range(0, n_windows, _CATALOGUE_CHUNK):
            times = self._analysis_times[lo : lo + _CATALOGUE_CHUNK]
            block = self.labels.isel(time=times) >= config.LABEL_OCCURRENCE_THRESHOLD
            hit = block.any(dim=reduce_dims).transpose("time", "hazard")
            flags[lo : lo + _CATALOGUE_CHUNK] = np.asarray(hit.values, dtype=bool)
        return flags

    def has_event(self) -> np.ndarray:
        """``(n_windows,)`` bool -- window holds an occurrence of *any* hazard."""
        return self.event_flags.any(axis=1)

    def sample_weights(
        self,
        *,
        target_fraction: float = config.EVENT_SAMPLER_TARGET_FRACTION,
        max_replication: float = config.EVENT_SAMPLER_MAX_REPLICATION,
    ) -> np.ndarray:
        """Per-window sampling weights, indexed in ``_window_starts`` space.

        An event window is weighted by its *rarest* hazard rather than by a
        plain "contains an event" flag: thunderstorms outnumber cloudbursts by
        orders of magnitude, so an any-event catalogue is almost entirely
        thunderstorm days and leaves cloudburst exactly as invisible as before.
        Quiet windows keep a floor weight of ``1.0`` -- a model that never sees
        calm air cries wolf.

        Event weights are scaled so the sampler puts ``target_fraction`` of its
        probability mass on event windows, then clipped to ``max_replication``
        times the quiet floor so a handful of storms cannot be memorised.

        Returns all-ones when the split holds no occurrence at all, or holds
        nothing *but* occurrences: both are degenerate, and neither may divide
        by zero or leave the sampler with an all-zero weight vector.
        """
        if not 0.0 < target_fraction < 1.0:
            raise ValueError("target_fraction must lie strictly between 0 and 1")
        n_windows = int(self.event_flags.shape[0])
        weights = np.ones(n_windows, dtype="float64")
        is_event = self.has_event()
        n_events = int(is_event.sum())
        n_quiet = n_windows - n_events
        if n_events == 0 or n_quiet == 0:
            return weights

        # Rarity relative to the most common hazard actually present in this
        # split: 1.0 for that hazard, > 1.0 for rarer ones. Hazards with zero
        # prevalence score 0 so the per-window max ignores them.
        prevalence = self.event_flags.mean(axis=0)
        present = prevalence > 0.0
        rarity = np.zeros(prevalence.shape, dtype="float64")
        rarity[present] = prevalence[present].max() / prevalence[present]
        per_window = (self.event_flags * rarity).max(axis=1)

        # Solve  s * R / (s * R + n_quiet) == target_fraction  for the common
        # scale s, where R is this split's total unscaled event rarity mass.
        total_rarity = float(per_window[is_event].sum())
        scale = target_fraction * n_quiet / ((1.0 - target_fraction) * total_rarity)
        weights[is_event] = np.minimum(scale * per_window[is_event], max_replication)
        return weights

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

    ``event_balanced_sampling`` (default ``True``) puts a
    :class:`~torch.utils.data.WeightedRandomSampler` on the *train* loader only,
    so rare-hazard windows actually appear in a batch; set it ``False`` for the
    ablation. Val and test stay unweighted and unshuffled either way.
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
        event_balanced_sampling: bool = True,
        train_dataset: Dataset | None = None,
        val_dataset: Dataset | None = None,
        test_dataset: Dataset | None = None,
    ) -> None:
        super().__init__()
        self.event_balanced_sampling = event_balanced_sampling
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
        event_balanced_sampling: bool = True,
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
            event_balanced_sampling=event_balanced_sampling,
            train_dataset=datasets[0],
            val_dataset=datasets[1],
            test_dataset=datasets[2],
        )

    def train_sampler(self) -> WeightedRandomSampler | None:
        """The train split's event-balanced sampler, or ``None`` when unusable.

        ``None`` -- meaning "fall back to plain uniform shuffling" -- when the
        ``event_balanced_sampling`` ablation flag is off, when the injected
        train dataset publishes no storm catalogue, or when the split is empty.
        """
        if not self.event_balanced_sampling:
            return None
        weights_fn = getattr(self._train, "sample_weights", None)
        if weights_fn is None:
            return None
        weights = np.asarray(weights_fn(), dtype="float64")
        if weights.size == 0 or float(weights.sum()) <= 0.0:
            return None
        return WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=int(weights.size),
            replacement=True,
        )

    def _loader(
        self,
        dataset: Dataset | None,
        *,
        shuffle: bool,
        sampler: Sampler | None = None,
    ) -> DataLoader:
        if dataset is None:
            raise RuntimeError("call setup() before requesting a dataloader")
        extra: dict = {}
        if self.num_workers > 0:
            # Windows spawns workers rather than forking, so each one re-imports
            # this process and re-opens the Zarr store. Without persistent
            # workers that cost is paid again every epoch and can outweigh the
            # parallel read it buys.
            extra = {"persistent_workers": True, "prefetch_factor": 2}
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle and sampler is None,
            sampler=sampler,
            num_workers=self.num_workers,
            **extra,
        )

    def train_dataloader(self) -> DataLoader:
        """Train loader, event-balanced via :meth:`train_sampler` by default.

        The sampler draws *with replacement*, so an epoch keeps its length but
        re-visits rare-hazard windows. Raw probabilities therefore come out
        biased high (the effective class prior is no longer climatology) --
        that is the post-hoc calibration stage's job, not this one's.
        """
        sampler = self.train_sampler()
        return self._loader(self._train, shuffle=sampler is None, sampler=sampler)

    def val_dataloader(self) -> DataLoader:
        """Val loader -- deliberately unweighted and unshuffled.

        Balancing this would make every reported CSI/POD/FAR describe a climate
        that does not exist.
        """
        return self._loader(self._val, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Test loader -- unweighted and unshuffled, for the same reason as val."""
        return self._loader(self._test, shuffle=False)
