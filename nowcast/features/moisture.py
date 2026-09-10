"""Moisture predictors: column-integrated water vapour (IWV) and its tendency.

``integrated_water_vapour`` prefers a proper vertical integral of pressure-level
specific humidity (derived from relative humidity and temperature via MetPy) and
falls back to the surface ``tcwv`` field when pressure levels are unavailable.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

G: float = 9.80665  # standard gravity, m s-2
_TIME_DIM = "time"
_LEVEL_DIM = "level"
_ORDER = ("time", "level", "lat", "lon")


def _first_step_backfill(da: xr.DataArray, dim: str = _TIME_DIM) -> xr.DataArray:
    """Replace the (NaN) first sample along ``dim`` with the second sample."""
    if da.sizes.get(dim, 0) < 2:
        return da.fillna(0.0)
    data = np.array(da.values, copy=True)
    data[0] = data[1]
    return da.copy(data=data)


def integrated_water_vapour(ds: xr.Dataset) -> xr.DataArray:
    """Column-integrated water vapour ``iwv`` in kg m-2.

    When pressure-level ``rh`` and ``t`` are present the field is
    ``(1/g) * |integral of q dp|`` with ``q`` the specific humidity; otherwise
    the surface ``tcwv`` field is returned unchanged (renamed to ``iwv``).
    """
    has_levels = _LEVEL_DIM in ds.dims and {"rh", "t"} <= set(ds.data_vars)
    if not has_levels:
        fallback = ds["tcwv"].astype("float32").rename("iwv")
        return fallback.assign_attrs(
            units="kg m-2", long_name="integrated water vapour (tcwv fallback)"
        )

    from metpy.calc import mixing_ratio_from_relative_humidity
    from metpy.units import units

    dims = [d for d in _ORDER if d in ds["t"].dims]
    t = ds["t"].transpose(*dims)
    rh = ds["rh"].transpose(*dims)
    level_axis = dims.index(_LEVEL_DIM)

    p_shape = [1] * t.ndim
    p_shape[level_axis] = t.sizes[_LEVEL_DIM]
    p_hpa = ds[_LEVEL_DIM].values.astype("float64").reshape(p_shape)

    rh_vals = np.asarray(rh.values, dtype="float64")
    if np.isfinite(rh_vals).any() and np.nanmax(rh_vals) > 1.5:
        rh_vals = rh_vals / 100.0
    rh_vals = np.clip(rh_vals, 1e-6, 1.0)

    w = (
        mixing_ratio_from_relative_humidity(
            p_hpa * units.hPa,
            np.asarray(t.values, dtype="float64") * units.K,
            rh_vals * units.dimensionless,
        )
        .to("kg/kg")
        .magnitude
    )
    q = w / (1.0 + w)  # specific humidity, kg kg-1

    p_pa = p_hpa.reshape(-1) * 100.0  # 1-D along level, Pa (descending)
    iwv = np.abs(np.trapezoid(q, x=p_pa, axis=level_axis)) / G

    out_dims = [d for d in dims if d != _LEVEL_DIM]
    return xr.DataArray(
        iwv.astype("float32"),
        dims=out_dims,
        coords={d: ds[d] for d in out_dims},
        name="iwv",
        attrs={"units": "kg m-2", "long_name": "integrated water vapour"},
    )


def iwv_tendency(iwv_da: xr.DataArray, hours: int = 1) -> xr.DataArray:
    """Forward-difference tendency of ``iwv_da`` along ``time``.

    Units are kg m-2 per ``hours``. The first step (undefined for a forward
    difference) is back-filled from the second step.
    """
    if _TIME_DIM not in iwv_da.dims or iwv_da.sizes[_TIME_DIM] < 2:
        return xr.zeros_like(iwv_da, dtype="float32").rename("iwv_tendency_1h")
    diff = (iwv_da - iwv_da.shift({_TIME_DIM: 1})) / float(hours)
    diff = _first_step_backfill(diff)
    return diff.astype("float32").rename("iwv_tendency_1h").assign_attrs(
        units="kg m-2 h-1", long_name="integrated water vapour tendency"
    )
