"""Kinematic predictors: 850 hPa convergence and bulk wind shear."""

from __future__ import annotations

import numpy as np
import xarray as xr

_EARTH_RADIUS_M: float = 6_371_000.0
G: float = 9.80665
_LEVEL_DIM = "level"


def _grid_spacing_m(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, float]:
    """Return ``(dx, dy)`` grid spacing in metres.

    ``dx`` has shape ``(n_lat,)`` because zonal spacing shrinks with latitude;
    ``dy`` is a scalar.
    """
    # REVIEW: candidate for nowcast/common
    lat = np.asarray(lat, dtype="float64")
    lon = np.asarray(lon, dtype="float64")
    dlon_rad = float(np.mean(np.deg2rad(np.gradient(lon))))
    dlat_rad = float(np.mean(np.deg2rad(np.gradient(lat))))
    dx = _EARTH_RADIUS_M * np.cos(np.deg2rad(lat)) * dlon_rad
    dy = _EARTH_RADIUS_M * dlat_rad
    return dx, dy


def convergence_850(ds: xr.Dataset) -> xr.DataArray:
    """Horizontal mass convergence at 850 hPa: ``-(du/dx + dv/dy)`` in s-1.

    Positive values indicate low-level convergence (ascent forcing).
    """
    u = ds["u"].sel({_LEVEL_DIM: 850}, method="nearest")
    v = ds["v"].sel({_LEVEL_DIM: 850}, method="nearest")
    dims = [d for d in ("time", "lat", "lon") if d in u.dims]
    u = u.transpose(*dims)
    v = v.transpose(*dims)
    lat_axis = dims.index("lat")
    lon_axis = dims.index("lon")

    dx, dy = _grid_spacing_m(ds["lat"].values, ds["lon"].values)
    u_vals = np.asarray(u.values, dtype="float64")
    v_vals = np.asarray(v.values, dtype="float64")
    du_dx = np.gradient(u_vals, axis=lon_axis)
    dv_dy = np.gradient(v_vals, axis=lat_axis)
    dx_b = dx.reshape([-1 if i == lat_axis else 1 for i in range(u_vals.ndim)])
    conv = -(du_dx / dx_b + dv_dy / dy)

    return xr.DataArray(
        conv.astype("float32"),
        dims=dims,
        coords={d: ds[d] for d in dims},
        name="conv_850",
        attrs={"units": "s-1", "long_name": "850 hPa horizontal convergence"},
    )


def _surface_level_index(ds: xr.Dataset) -> int:
    return int(np.argmax(np.asarray(ds[_LEVEL_DIM].values)))


def _interp_to_height(var: xr.DataArray, height_agl: xr.DataArray, target_m: float) -> xr.DataArray:
    def _col(v_col: np.ndarray, h_col: np.ndarray) -> float:
        order = np.argsort(h_col)
        return float(np.interp(target_m, h_col[order], v_col[order]))

    return xr.apply_ufunc(
        _col,
        var,
        height_agl,
        input_core_dims=[[_LEVEL_DIM], [_LEVEL_DIM]],
        vectorize=True,
        output_dtypes=[np.float64],
    )


def bulk_shear(ds: xr.Dataset, bottom_m: float, top_m: float) -> xr.DataArray:
    """Magnitude (m s-1) of the vector wind difference between two heights AGL.

    Heights are ``z / g`` referenced to the lowest model level. The channel is
    named ``shear_<bottom_km>_<top_km>km`` (e.g. ``shear_0_6km``).
    """
    dims = [d for d in ("time", _LEVEL_DIM, "lat", "lon") if d in ds["u"].dims]
    u = ds["u"].transpose(*dims)
    v = ds["v"].transpose(*dims)
    z = ds["z"].transpose(*dims)

    height = z / G
    sfc = _surface_level_index(ds)
    height_agl = height - height.isel({_LEVEL_DIM: sfc})

    u_bot = _interp_to_height(u, height_agl, bottom_m)
    u_top = _interp_to_height(u, height_agl, top_m)
    v_bot = _interp_to_height(v, height_agl, bottom_m)
    v_top = _interp_to_height(v, height_agl, top_m)
    mag = np.hypot(u_top - u_bot, v_top - v_bot)

    name = f"shear_{int(bottom_m) // 1000}_{int(top_m) // 1000}km"
    return mag.astype("float32").rename(name).assign_attrs(
        units="m s-1", long_name=f"bulk wind shear {int(bottom_m)}-{int(top_m)} m AGL"
    )
