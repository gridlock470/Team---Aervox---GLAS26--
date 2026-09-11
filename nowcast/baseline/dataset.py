"""Turn the gridded feature cube + label cube into a flat per-pixel table.

The LightGBM baseline treats every ``(time, lat, lon)`` pixel as an independent
sample: features are the :data:`nowcast.schema.FEATURE_CHANNELS` values at that
pixel (static terrain channels broadcast), targets are the
``n_hazards * n_leads`` probability values at that pixel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from nowcast import config, schema
from nowcast.features.labels import valid_label_times


@dataclass
class PixelDataset:
    """A flat supervised table.

    Attributes
    ----------
    X:
        ``(n_rows, n_features)`` float32 design matrix.
    y:
        ``(n_rows, n_hazards * n_leads)`` float32 target matrix (hazard-major,
        lead-minor - matching :attr:`target_names`).
    feature_names:
        Column names of ``X`` (equal to ``schema.FEATURE_CHANNELS``).
    target_names:
        Column names of ``y`` as ``"<hazard>__lead<h>"``.
    times:
        ``datetime64`` array, one entry per row (the source timestep).
    """

    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    target_names: list[str]
    times: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        """Return targets as ``{target_name: column_vector}``."""
        return {name: self.y[:, i] for i, name in enumerate(self.target_names)}


def target_names() -> list[str]:
    """The ordered target column names for the pixel dataset."""
    return [
        f"{hazard}__lead{lead}"
        for hazard in config.HAZARDS
        for lead in config.LEAD_TIMES_H
    ]


def make_pixel_dataset(
    features_ds: xr.Dataset,
    labels_da: xr.DataArray,
    n_per_time: int | None = None,
    seed: int = config.RANDOM_SEED,
    *,
    drop_label_horizon: bool = True,
) -> PixelDataset:
    """Sample pixels from every timestep into a flat table.

    ``n_per_time`` pixels are drawn (without replacement) per timestep; ``None``
    keeps every pixel. Rows with any non-finite feature or target are dropped.

    When ``drop_label_horizon`` is true (the default) timesteps whose label
    horizon runs past the end of *this* dataset or across a time gap are
    excluded - their labels are zero-padded, not observed. Because the check
    runs on the ``time`` coord passed in, a chronological split that slices
    first (see :func:`split_by_date_range`) automatically drops the last
    ``max(LEAD_TIMES_H)`` hours of each split, removing cross-boundary leakage.
    """
    feature_names = list(schema.FEATURE_CHANNELS)
    rng = np.random.default_rng(seed)

    times = np.asarray(features_ds["time"].values)
    labels_da = labels_da.transpose("time", "hazard", "lead", "lat", "lon")

    if drop_label_horizon and times.size:
        valid = np.asarray(valid_label_times(features_ds["time"]).values, dtype=bool)
    else:
        valid = np.ones(times.size, dtype=bool)
    n_lat = features_ds.sizes["lat"]
    n_lon = features_ds.sizes["lon"]
    n_pix = n_lat * n_lon

    static: dict[str, np.ndarray] = {}
    dynamic: dict[str, np.ndarray] = {}
    for name in feature_names:
        da = features_ds[name]
        if "time" in da.dims:
            dynamic[name] = np.asarray(
                da.transpose("time", "lat", "lon").values, dtype="float64"
            )
        else:
            static[name] = np.asarray(
                da.transpose("lat", "lon").values, dtype="float64"
            ).ravel()

    n_targets = schema.N_HAZARDS * schema.N_LEADS
    n_feat = len(feature_names)
    keep_ti = np.flatnonzero(valid)
    if keep_ti.size == 0:
        return PixelDataset(
            np.empty((0, n_feat), dtype="float32"),
            np.empty((0, n_targets), dtype="float32"),
            feature_names,
            target_names(),
            np.asarray(times[:0]),
        )
    y_all = np.asarray(labels_da.values, dtype="float64").reshape(
        times.size, n_targets, n_pix
    )

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    t_parts: list[np.ndarray] = []
    for ti in keep_ti:
        if n_per_time is None or n_per_time >= n_pix:
            idx = np.arange(n_pix)
        else:
            idx = rng.choice(n_pix, size=n_per_time, replace=False)
        cols = [
            static[name][idx] if name in static else dynamic[name][ti].ravel()[idx]
            for name in feature_names
        ]
        x_parts.append(np.stack(cols, axis=1))
        y_parts.append(y_all[ti][:, idx].T)
        t_parts.append(np.full(idx.size, times[ti]))

    X = np.concatenate(x_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    row_times = np.concatenate(t_parts, axis=0)

    finite = np.isfinite(X).all(axis=1) & np.isfinite(y).all(axis=1)
    return PixelDataset(
        X[finite].astype("float32"),
        y[finite].astype("float32"),
        feature_names,
        target_names(),
        row_times[finite],
    )


def _in_date_range(time_coord: xr.DataArray, date_range: tuple[str, str]) -> np.ndarray:
    """Boolean mask for timestamps inside ``date_range`` (both ends inclusive)."""
    t = np.asarray(time_coord.values, dtype="datetime64[ns]")
    start = np.datetime64(pd.Timestamp(date_range[0]), "ns")
    # inclusive end-of-day: exclusive upper bound is the day after the end date
    end_excl = np.datetime64(pd.Timestamp(date_range[1]).normalize(), "ns") + np.timedelta64(
        1, "D"
    )
    return (t >= start) & (t < end_excl)


def split_by_date_range(
    features_ds: xr.Dataset,
    labels_da: xr.DataArray,
    *,
    train_range: tuple[str, str] = config.TRAIN_DATE_RANGE,
    val_range: tuple[str, str] = config.VAL_DATE_RANGE,
    n_per_time: int | None = None,
    seed: int = config.RANDOM_SEED,
) -> tuple[PixelDataset, PixelDataset]:
    """Build disjoint ``(train, val)`` pixel datasets by chronological date range.

    Selection is on the ``time`` coordinate (both ends inclusive). ``config``
    guarantees ``TRAIN_DATE_RANGE`` and ``VAL_DATE_RANGE`` are disjoint; each
    split is sliced before :func:`make_pixel_dataset` so the label-horizon drop
    also prevents occurrence leaking across the split boundary.
    """
    train_mask = _in_date_range(features_ds["time"], train_range)
    val_mask = _in_date_range(features_ds["time"], val_range)
    if bool(np.any(train_mask & val_mask)):
        raise ValueError("train_range and val_range overlap on the time coord")

    train = make_pixel_dataset(
        features_ds.isel(time=train_mask),
        labels_da.isel(time=train_mask),
        n_per_time=n_per_time,
        seed=seed,
    )
    val = make_pixel_dataset(
        features_ds.isel(time=val_mask),
        labels_da.isel(time=val_mask),
        n_per_time=n_per_time,
        seed=seed + 1,
    )
    return train, val
