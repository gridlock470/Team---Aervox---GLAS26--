"""Synthetic, schema-valid data generators.

Every downstream slice (ingestion, features, model) can be exercised *now* —
before any real IMDAA / MERA / INSAT data exists — by building fake datasets
that satisfy :mod:`nowcast.schema`. All randomness is seeded so tests are
deterministic.

Nothing in here touches the real ``data/`` directory unless the caller passes
an explicit path to :func:`write_synthetic_datacube`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from nowcast import config, schema

_H, _W = config.GRID_SHAPE
_LEVELS = np.asarray(config.PRESSURE_LEVELS_HPA, dtype="float32")


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _smooth(field: np.ndarray, passes: int = 2) -> np.ndarray:
    """Cheap box blur so synthetic fields look spatially correlated."""
    out = field.astype("float64")
    for _ in range(passes):
        out = (
            out
            + np.roll(out, 1, axis=-1)
            + np.roll(out, -1, axis=-1)
            + np.roll(out, 1, axis=-2)
            + np.roll(out, -1, axis=-2)
        ) / 5.0
    return out


def _times(n_hours: int, start: str = "2018-05-02T00:00:00") -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=n_hours, freq=config.TIMESTEP)


def make_dem(seed: int = 0) -> xr.DataArray:
    """Synthetic elevation field (metres) with a diagonal valley for routing."""
    rng = _rng(seed)
    yy, xx = np.meshgrid(
        np.linspace(0, 1, _H), np.linspace(0, 1, _W), indexing="ij"
    )
    # A ridge to the north, a valley running SE, plus noise.
    base = 3200.0 * yy + 400.0
    valley = -1500.0 * np.exp(-(((xx - yy) / 0.15) ** 2))
    noise = _smooth(rng.normal(0, 120, size=(_H, _W)), passes=3)
    elev = np.clip(base + valley + noise, 150.0, 6500.0).astype("float32")
    return xr.DataArray(
        elev,
        dims=schema.STATIC_DIMS,
        coords={"lat": config.GRID_LAT, "lon": config.GRID_LON},
        name="elevation",
        attrs={"units": "m"},
    )


def _static_vars(seed: int) -> dict[str, xr.DataArray]:
    dem = make_dem(seed)
    elev = dem.values
    gy, gx = np.gradient(elev.astype("float64"))
    slope = np.degrees(
        np.arctan(np.hypot(gy, gx) / (config.GRID_RESOLUTION_DEG * 111_000))
    )
    # Fake but monotone flow accumulation: lower cells accumulate more.
    inv = elev.max() - elev
    flow_acc = _smooth(inv, passes=4)
    flow_acc = (flow_acc / flow_acc.max() * 5000.0).astype("float32")
    flow_dir = (_rng(seed + 1).integers(1, 129, size=(_H, _W))).astype("float32")
    hand = np.clip(
        elev - elev.min() - _smooth(inv, passes=6) / inv.max() * 50.0, 0, None
    )

    def _da(arr, units):
        return xr.DataArray(
            np.asarray(arr, dtype="float32"),
            dims=schema.STATIC_DIMS,
            coords={"lat": config.GRID_LAT, "lon": config.GRID_LON},
            attrs={"units": units},
        )

    return {
        "elevation": _da(elev, "m"),
        "slope": _da(slope, "deg"),
        "flow_accumulation": _da(flow_acc, "cells"),
        "flow_direction": _da(flow_dir, "1"),
        "hand": _da(hand, "m"),
    }


def make_datacube(
    n_hours: int = 48,
    *,
    seed: int = 0,
    with_satellite: bool = True,
    with_static: bool = True,
) -> xr.Dataset:
    """A schema-valid raw datacube filled with plausible random fields."""
    rng = _rng(seed)
    times = _times(n_hours)
    coords = {
        "time": times,
        "level": _LEVELS,
        "lat": config.GRID_LAT,
        "lon": config.GRID_LON,
    }
    diurnal = np.cos(2 * np.pi * (times.hour.values / 24.0))[:, None, None]

    def _surf(mean, amp, units, noise=1.0):
        arr = mean + amp * diurnal + _smooth(
            rng.normal(0, noise, size=(n_hours, _H, _W)), passes=2
        )
        return xr.DataArray(
            arr.astype("float32"), dims=schema.SURFACE_DIMS, attrs={"units": units}
        )

    def _lvl(mean, units, noise=1.0):
        arr = mean + _smooth(
            rng.normal(0, noise, size=(n_hours, len(_LEVELS), _H, _W)), passes=1
        )
        return xr.DataArray(
            arr.astype("float32"), dims=schema.LEVEL_DIMS, attrs={"units": units}
        )

    t_profile = np.linspace(288, 233, len(_LEVELS))[None, :, None, None]
    z_profile = np.linspace(1000, 10500, len(_LEVELS))[None, :, None, None] * 9.81

    data = {
        "t2m": _surf(298.0, 6.0, "K"),
        "u10": _surf(1.5, 2.0, "m s-1", noise=2.0),
        "v10": _surf(-0.5, 2.0, "m s-1", noise=2.0),
        "prmsl": _surf(100_800.0, 60.0, "Pa", noise=40.0),
        "tcwv": abs(_surf(45.0, 8.0, "kg m-2", noise=4.0)),
        "cape": abs(_surf(800.0, 600.0, "J kg-1", noise=200.0)),
        "cin": -abs(_surf(40.0, 30.0, "J kg-1", noise=15.0)),
        "precip": xr.DataArray(
            np.clip(
                rng.gamma(0.4, 6.0, size=(n_hours, _H, _W)).astype("float32"),
                0.0,
                None,
            ),
            dims=schema.SURFACE_DIMS,
            attrs={"units": "mm h-1"},
        ),
        "t": _lvl(t_profile, "K"),
        "rh": xr.DataArray(
            np.clip(_lvl(60.0, "%", noise=12.0).values, 0.0, 100.0).astype("float32"),
            dims=schema.LEVEL_DIMS,
            attrs={"units": "%"},
        ),
        "z": _lvl(z_profile, "m2 s-2", noise=50.0),
        "u": _lvl(8.0, "m s-1", noise=6.0),
        "v": _lvl(2.0, "m s-1", noise=6.0),
    }

    if with_satellite:
        data["ctt"] = xr.DataArray(
            np.clip(_surf(265.0, 5.0, "K", noise=12.0).values, 190.0, 300.0).astype(
                "float32"
            ),
            dims=schema.SURFACE_DIMS,
            attrs={"units": "K"},
        )
        data["wv_bt"] = _surf(240.0, 3.0, "K", noise=6.0)

    ds = xr.Dataset(data, coords=coords)

    if with_static:
        ds = ds.assign(_static_vars(seed))

    ds.attrs["title"] = "synthetic nowcast datacube"
    ds.attrs["synthetic"] = 1
    return ds


def make_feature_datacube(n_hours: int = 48, *, seed: int = 0) -> xr.Dataset:
    """A datacube that additionally carries every name in
    :data:`nowcast.schema.FEATURE_CHANNELS` (post feature-engineering shape)."""
    rng = _rng(seed)
    ds = make_datacube(n_hours, seed=seed)
    times = ds["time"]
    for ch in schema.FEATURE_CHANNELS:
        if ch in ds:
            continue
        if ch in schema.STATIC_CHANNELS:
            arr = _smooth(rng.normal(0, 1, size=(_H, _W)))
            ds[ch] = xr.DataArray(arr.astype("float32"), dims=schema.STATIC_DIMS)
        else:
            arr = _smooth(rng.normal(0, 1, size=(len(times), _H, _W)), passes=1)
            ds[ch] = xr.DataArray(arr.astype("float32"), dims=schema.SURFACE_DIMS)
    return ds


def make_targets(n_hours: int = 48, *, seed: int = 0) -> xr.DataArray:
    """Schema-valid target probability maps, dims
    ``(time, hazard, lead, lat, lon)`` with values in ``[0, 1]``."""
    rng = _rng(seed)
    shape = (n_hours, schema.N_HAZARDS, schema.N_LEADS, _H, _W)
    raw = np.zeros(shape, dtype="float64")
    for h in range(schema.N_HAZARDS):
        for lead in range(schema.N_LEADS):
            raw[:, h, lead] = _smooth(
                rng.random(size=(n_hours, _H, _W)) ** (2 + lead), passes=2
            )
    raw = np.clip(raw, 0.0, 1.0).astype("float32")
    return xr.DataArray(
        raw,
        dims=schema.TARGET_DIMS,
        coords={
            "time": _times(n_hours),
            "hazard": list(config.HAZARDS),
            "lead": list(config.LEAD_TIMES_H),
            "lat": config.GRID_LAT,
            "lon": config.GRID_LON,
        },
        name="hazard_probability",
    )


def make_sample(*, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """One ``(x, y)`` pair matching :data:`nowcast.schema.INPUT_SHAPE` /
    :data:`nowcast.schema.TARGET_SHAPE`."""
    rng = _rng(seed)
    x = rng.standard_normal(schema.INPUT_SHAPE).astype("float32")
    y = rng.random(schema.TARGET_SHAPE).astype("float32")
    return x, y


def make_batch(batch_size: int = 2, *, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A stacked batch: ``(B, C, T, H, W)`` and ``(B, n_haz, n_lead, H, W)``."""
    xs, ys = zip(
        *(make_sample(seed=seed + i) for i in range(batch_size)), strict=True
    )
    return np.stack(xs), np.stack(ys)


def write_synthetic_datacube(path: str | Path, n_hours: int = 48, *, seed: int = 0) -> Path:
    """Write a synthetic datacube to ``path`` as Zarr and return the path."""
    path = Path(path)
    ds = make_datacube(n_hours, seed=seed)
    ds.to_zarr(path, mode="w")
    return path
