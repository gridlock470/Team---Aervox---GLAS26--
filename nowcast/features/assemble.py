"""Assemble the ordered, float32 feature cube consumed by the models.

The output Dataset holds *exactly* the names in :data:`nowcast.schema.FEATURE_CHANNELS`,
in that order. Static terrain channels keep dims ``(lat, lon)``; every other
channel is ``(time, lat, lon)``. All values are finite.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from nowcast import schema
from nowcast.features.kinematics import bulk_shear, convergence_850
from nowcast.features.moisture import integrated_water_vapour, iwv_tendency
from nowcast.features.satellite import ctt_drop_rate
from nowcast.features.thermo import ensure_cape_cin, lifted_index

_FILL_DEFAULT = 0.0
_SURFACE_DIMS = ("time", "lat", "lon")


def _fill(da: xr.DataArray) -> xr.DataArray:
    """Cast to float32 and replace non-finite values with the field median."""
    arr = da.astype("float32")
    vals = arr.values
    if np.isfinite(vals).all():
        return arr
    median = float(np.nanmedian(vals)) if np.isfinite(vals).any() else _FILL_DEFAULT
    if not np.isfinite(median):
        median = _FILL_DEFAULT
    return arr.where(np.isfinite(arr), median)


def _surface_constant(ds: xr.Dataset, name: str, value: float) -> xr.DataArray:
    shape = tuple(ds.sizes[d] for d in _SURFACE_DIMS)
    return xr.DataArray(
        np.full(shape, value, dtype="float32"),
        dims=_SURFACE_DIMS,
        coords={d: ds[d] for d in _SURFACE_DIMS},
        name=name,
    )


def _satellite(ds: xr.Dataset, name: str, default: float) -> xr.DataArray:
    if name in ds.data_vars:
        return ds[name]
    return _surface_constant(ds, name, default)


def assemble_features(ds: xr.Dataset) -> xr.Dataset:
    """Return a Dataset whose ``data_vars`` equal ``schema.FEATURE_CHANNELS``."""
    cape, cin = ensure_cape_cin(ds)
    iwv = integrated_water_vapour(ds)
    has_ctt = "ctt" in ds.data_vars

    channels: dict[str, xr.DataArray] = {
        "t2m": ds["t2m"],
        "prmsl": ds["prmsl"],
        "tcwv": ds["tcwv"],
        "cape": cape,
        "cin": cin,
        "iwv": iwv,
        "iwv_tendency_1h": iwv_tendency(iwv, hours=1),
        "lifted_index": lifted_index(ds),
        "conv_850": convergence_850(ds),
        "shear_0_6km": bulk_shear(ds, 0.0, 6000.0),
        "shear_0_1km": bulk_shear(ds, 0.0, 1000.0),
        "u10": ds["u10"],
        "v10": ds["v10"],
        "ctt": _satellite(ds, "ctt", 260.0),
        "ctt_drop_rate_1h": (
            ctt_drop_rate(ds, hours=1)
            if has_ctt
            else _surface_constant(ds, "ctt_drop_rate_1h", 0.0)
        ),
        "wv_bt": _satellite(ds, "wv_bt", 240.0),
        "elevation": ds["elevation"],
        "slope": ds["slope"],
        "flow_accumulation": ds["flow_accumulation"],
        "hand": ds["hand"],
    }

    missing = set(schema.FEATURE_CHANNELS) - set(channels)
    if missing:
        raise KeyError(f"cannot assemble features, missing inputs for: {sorted(missing)}")

    out: dict[str, xr.DataArray] = {}
    for name in schema.FEATURE_CHANNELS:
        da = channels[name].rename(name)
        drop = [c for c in ("level",) if c in da.coords]
        if drop:
            da = da.drop_vars(drop)
        out[name] = _fill(da)

    result = xr.Dataset(out)
    assert set(result.data_vars) == set(schema.FEATURE_CHANNELS), (
        "assemble_features produced the wrong channel set: "
        f"{sorted(set(result.data_vars) ^ set(schema.FEATURE_CHANNELS))}"
    )
    result = result[list(schema.FEATURE_CHANNELS)]
    result.attrs["feature_channels"] = list(schema.FEATURE_CHANNELS)
    return result


def write_features(ds: xr.Dataset, path: str | Path) -> Path:
    """Write the feature cube to ``path`` as a Zarr store and return the path."""
    path = Path(path)
    ds.to_zarr(path, mode="w")
    return path


def open_features(path: str | Path) -> xr.Dataset:
    """Open (and eagerly load) a feature cube written by :func:`write_features`."""
    return xr.open_zarr(Path(path)).load()
