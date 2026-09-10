"""Thermodynamic instability predictors: lifted index and CAPE/CIN.

Both quantities are computed column-by-column with MetPy. The loops are
acceptable here because feature engineering runs once, offline, on the datacube.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

_ORDER = ("time", "level", "lat", "lon")
_LEVEL_DIM = "level"


def _prep_columns(
    ds: xr.Dataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[int]]:
    """Return ``(pressure, temp_cols, rh_cols, out_dims, out_shape)``.

    ``pressure`` is 1-D descending (surface first). ``temp_cols`` / ``rh_cols``
    have shape ``(n_level, n_column)`` with columns flattened in C order over
    ``out_dims``. Relative humidity is normalised to a 0-1 fraction.
    """
    dims = [d for d in _ORDER if d in ds["t"].dims]
    t = ds["t"].transpose(*dims)
    rh = ds["rh"].transpose(*dims)
    p = ds[_LEVEL_DIM].values.astype("float64")
    order = np.argsort(-p)
    p_sorted = p[order]
    level_axis = dims.index(_LEVEL_DIM)

    tv = np.moveaxis(np.asarray(t.values, dtype="float64"), level_axis, 0)[order]
    rv = np.moveaxis(np.asarray(rh.values, dtype="float64"), level_axis, 0)[order]
    if np.isfinite(rv).any() and np.nanmax(rv) > 1.5:
        rv = rv / 100.0
    rv = np.clip(rv, 1e-4, 1.0)

    n_level = tv.shape[0]
    out_dims = [d for d in dims if d != _LEVEL_DIM]
    out_shape = [ds.sizes[d] for d in out_dims]
    return p_sorted, tv.reshape(n_level, -1), rv.reshape(n_level, -1), out_dims, out_shape


def _column_da(
    values: np.ndarray, shape: list[int], out_dims: list[str], ds: xr.Dataset, name: str, ln: str
) -> xr.DataArray:
    return xr.DataArray(
        values.reshape(shape).astype("float32"),
        dims=out_dims,
        coords={d: ds[d] for d in out_dims},
        name=name,
        attrs={"long_name": ln},
    )


def lifted_index(ds: xr.Dataset) -> xr.DataArray:
    """500 hPa lifted index in K (positive = stable, negative = unstable)."""
    from metpy.calc import dewpoint_from_relative_humidity, parcel_profile
    from metpy.calc import lifted_index as _mp_lifted_index
    from metpy.units import units

    p, temp_cols, rh_cols, out_dims, shape = _prep_columns(ds)
    p_q = p * units.hPa
    n_col = temp_cols.shape[1]
    out = np.full(n_col, np.nan, dtype="float64")

    for c in range(n_col):
        tc = temp_cols[:, c]
        rc = rh_cols[:, c]
        if not (np.isfinite(tc).all() and np.isfinite(rc).all()):
            continue
        t_q = tc * units.K
        td_q = dewpoint_from_relative_humidity(t_q, rc * units.dimensionless)
        try:
            profile = parcel_profile(p_q, t_q[0], td_q[0])
            li = _mp_lifted_index(p_q, t_q, profile)
            out[c] = float(np.atleast_1d(li.to("K").magnitude)[0])
        except (ValueError, IndexError):
            continue

    da = _column_da(out, shape, out_dims, ds, "lifted_index", "500 hPa lifted index")
    return da.assign_attrs(units="K")


def ensure_cape_cin(ds: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray]:
    """Return ``(cape, cin)`` in J kg-1.

    Existing ``cape``/``cin`` fields are passed through; otherwise surface-based
    values are computed column-wise with MetPy. ``cin`` is non-positive.
    """
    if {"cape", "cin"} <= set(ds.data_vars):
        return (
            ds["cape"].astype("float32").rename("cape"),
            ds["cin"].astype("float32").rename("cin"),
        )

    from metpy.calc import dewpoint_from_relative_humidity, surface_based_cape_cin
    from metpy.units import units

    p, temp_cols, rh_cols, out_dims, shape = _prep_columns(ds)
    p_q = p * units.hPa
    n_col = temp_cols.shape[1]
    cape = np.full(n_col, np.nan, dtype="float64")
    cin = np.full(n_col, np.nan, dtype="float64")

    for c in range(n_col):
        tc = temp_cols[:, c]
        rc = rh_cols[:, c]
        if not (np.isfinite(tc).all() and np.isfinite(rc).all()):
            continue
        t_q = tc * units.K
        td_q = dewpoint_from_relative_humidity(t_q, rc * units.dimensionless)
        try:
            cc, ci = surface_based_cape_cin(p_q, t_q, td_q)
            cape[c] = float(cc.to("J/kg").magnitude)
            cin[c] = float(ci.to("J/kg").magnitude)
        except (ValueError, IndexError):
            continue

    cape_da = _column_da(cape, shape, out_dims, ds, "cape", "surface-based CAPE")
    cin_da = _column_da(cin, shape, out_dims, ds, "cin", "surface-based CIN")
    return cape_da.assign_attrs(units="J kg-1"), cin_da.assign_attrs(units="J kg-1")
