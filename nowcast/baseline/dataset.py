"""Turn the gridded feature cube + label cube into a flat per-pixel table.

The LightGBM baseline treats every ``(time, lat, lon)`` pixel as an independent
sample: features are the :data:`nowcast.schema.FEATURE_CHANNELS` values at that
pixel (static terrain channels broadcast), targets are the
``n_hazards * n_leads`` probability values at that pixel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from nowcast import config, schema


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
) -> PixelDataset:
    """Sample pixels from every timestep into a flat table.

    ``n_per_time`` pixels are drawn (without replacement) per timestep; ``None``
    keeps every pixel. Rows with any non-finite feature or target are dropped.
    """
    feature_names = list(schema.FEATURE_CHANNELS)
    rng = np.random.default_rng(seed)

    times = np.asarray(features_ds["time"].values)
    labels_da = labels_da.transpose("time", "hazard", "lead", "lat", "lon")
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
    if times.size == 0:
        return PixelDataset(
            np.empty((0, n_feat), dtype="float32"),
            np.empty((0, n_targets), dtype="float32"),
            feature_names,
            target_names(),
            np.asarray(times),
        )
    y_all = np.asarray(labels_da.values, dtype="float64").reshape(
        times.size, n_targets, n_pix
    )

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    t_parts: list[np.ndarray] = []
    for ti in range(times.size):
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


def split_by_year(
    features_ds: xr.Dataset,
    labels_da: xr.DataArray,
    *,
    train_years: tuple[int, ...] = config.TRAIN_YEARS,
    val_years: tuple[int, ...] = config.VAL_YEARS,
    n_per_time: int | None = None,
    seed: int = config.RANDOM_SEED,
) -> tuple[PixelDataset, PixelDataset]:
    """Build ``(train, val)`` pixel datasets by calendar-year membership."""
    years = np.asarray(features_ds["time"].dt.year.values)
    train_mask = np.isin(years, np.asarray(train_years))
    val_mask = np.isin(years, np.asarray(val_years))

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
