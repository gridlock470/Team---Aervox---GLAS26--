"""Target-grid construction and regridding onto the common analysis grid.

Every gridded source (IMDAA, MERA, IMERG, INSAT, DEM) is cropped to the pilot
bounding box and interpolated onto the exact ``config.GRID_LAT`` /
``config.GRID_LON`` cell centres defined in :mod:`nowcast.config`. No external
regridding engine (xesmf/esmpy) is used -- plain :meth:`xarray.DataArray.interp`
bilinear interpolation is enough on the 0.1 deg grid.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from nowcast import config

__all__ = ["target_grid", "crop_bbox", "regrid_to_target"]

_LAT = np.asarray(config.GRID_LAT, dtype="float64")
_LON = np.asarray(config.GRID_LON, dtype="float64")


def target_grid() -> dict[str, xr.DataArray]:
    """Return the target grid as ``{"lat": DataArray, "lon": DataArray}``.

    The coordinates are strictly ascending and match ``config.GRID_LAT`` /
    ``config.GRID_LON`` exactly (shape ``config.GRID_SHAPE``).
    """
    lat = xr.DataArray(_LAT.copy(), dims="lat", coords={"lat": _LAT.copy()}, name="lat")
    lon = xr.DataArray(_LON.copy(), dims="lon", coords={"lon": _LON.copy()}, name="lon")
    lat.attrs.update(units="degrees_north", standard_name="latitude")
    lon.attrs.update(units="degrees_east", standard_name="longitude")
    return {"lat": lat, "lon": lon}


def crop_bbox(obj: xr.DataArray | xr.Dataset, *, pad: float = 0.5) -> xr.DataArray | xr.Dataset:
    """Crop ``obj`` to the pilot bounding box (plus ``pad`` degrees of margin).

    Parameters
    ----------
    obj:
        Object carrying ascending ``lat`` and ``lon`` coordinates.
    pad:
        Extra margin in degrees kept around the box so a subsequent
        interpolation to the exact grid does not hit the data edge.
    """
    if "lat" not in obj.coords or "lon" not in obj.coords:
        raise KeyError("crop_bbox requires 'lat'/'lon' coords; run standardize_coords first")
    out = obj
    if out["lat"].size > 1 and bool(out["lat"][0] > out["lat"][-1]):
        out = out.sortby("lat")
    if out["lon"].size > 1 and bool(out["lon"][0] > out["lon"][-1]):
        out = out.sortby("lon")
    lat_slice = slice(config.BBOX_SOUTH - pad, config.BBOX_NORTH + pad)
    lon_slice = slice(config.BBOX_WEST - pad, config.BBOX_EAST + pad)
    return out.sel(lat=lat_slice, lon=lon_slice)


def regrid_to_target(
    obj: xr.DataArray | xr.Dataset, method: str = "linear"
) -> xr.DataArray | xr.Dataset:
    """Interpolate ``obj`` onto the exact target grid.

    Parameters
    ----------
    obj:
        Object with ascending ``lat`` / ``lon`` coordinates (call
        :func:`nowcast.common.io.standardize_coords` first).
    method:
        Interpolation method passed to :meth:`xarray.DataArray.interp`
        (``"linear"`` or ``"nearest"``).

    Returns
    -------
    Same type as ``obj`` with ``lat`` / ``lon`` replaced by the target grid.
    The output coordinates are set exactly to ``config.GRID_LAT`` /
    ``config.GRID_LON`` so downstream ``schema.validate_datacube`` passes.
    """
    if "lat" not in obj.coords or "lon" not in obj.coords:
        raise KeyError("regrid_to_target requires 'lat' and 'lon' coordinates")

    src = obj
    # ``interp`` needs monotonic coordinates; enforce ascending.
    if src["lat"].size > 1 and bool(src["lat"][0] > src["lat"][-1]):
        src = src.sortby("lat")
    if src["lon"].size > 1 and bool(src["lon"][0] > src["lon"][-1]):
        src = src.sortby("lon")

    interp_kwargs = {"bounds_error": False, "fill_value": None}
    if method == "nearest":
        interp_kwargs = {"bounds_error": False}
    out = src.interp(lat=_LAT, lon=_LON, method=method, kwargs=interp_kwargs)
    out = out.assign_coords(lat=_LAT.copy(), lon=_LON.copy())
    out["lat"].attrs.update(units="degrees_north", standard_name="latitude")
    out["lon"].attrs.update(units="degrees_east", standard_name="longitude")
    return out
