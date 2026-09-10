"""Assemble source fields into the validated datacube and read/write it.

:func:`build_datacube` merges the standardised surface / level / satellite /
static fields onto one :class:`xarray.Dataset` with the schema dimension order,
runs :func:`nowcast.schema.validate_datacube`, and returns it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from nowcast import config, schema
from nowcast.common import io as _io

__all__ = ["build_datacube", "write_datacube", "open_datacube"]

_LEVELS = np.asarray(config.PRESSURE_LEVELS_HPA, dtype="float64")


def _ensure_grid_coords(ds: xr.Dataset) -> xr.Dataset:
    """Snap lat/lon (and level) coords to the exact config values."""
    if "lat" in ds.coords:
        ds = ds.assign_coords(lat=np.asarray(config.GRID_LAT, dtype="float64"))
    if "lon" in ds.coords:
        ds = ds.assign_coords(lon=np.asarray(config.GRID_LON, dtype="float64"))
    if "level" in ds.coords and ds["level"].size == _LEVELS.size:
        ds = ds.assign_coords(level=_LEVELS.copy())
    return ds


def build_datacube(
    surface: xr.Dataset,
    level: xr.Dataset,
    satellite: xr.Dataset | None = None,
    static: xr.Dataset | xr.DataArray | None = None,
    *,
    validate: bool = True,
) -> xr.Dataset:
    """Merge source datasets into one schema-conforming datacube.

    Parameters
    ----------
    surface:
        Surface fields with dims ``(time, lat, lon)`` (from
        :func:`nowcast.ingest.imdaa.load_imdaa_single_level` plus merged precip).
    level:
        Pressure-level fields with dims ``(time, level, lat, lon)``.
    satellite:
        Optional INSAT fields (``ctt``, ``wv_bt``).
    static:
        Optional terrain fields (``elevation``, ``slope``,
        ``flow_accumulation``, ``flow_direction``, ``hand``). A bare
        DataArray is treated as ``elevation``.
    validate:
        Run :func:`nowcast.schema.validate_datacube` before returning.

    Returns
    -------
    xarray.Dataset
        The datacube. Dimension order follows ``schema.SURFACE_DIMS`` /
        ``schema.LEVEL_DIMS`` / ``schema.STATIC_DIMS``.
    """
    parts: list[xr.Dataset] = [surface, level]
    if satellite is not None:
        parts.append(satellite)
    if static is not None:
        if isinstance(static, xr.DataArray):
            static = static.to_dataset(name=static.name or "elevation")
        parts.append(static)

    h, w = config.GRID_SHAPE
    for part in parts:
        if "lat" in part.dims and part.sizes["lat"] != h:
            raise schema.SchemaError(
                f"input has {part.sizes['lat']} latitudes, expected {h} (regrid first)"
            )
        if "lon" in part.dims and part.sizes["lon"] != w:
            raise schema.SchemaError(
                f"input has {part.sizes['lon']} longitudes, expected {w} (regrid first)"
            )

    parts = [_ensure_grid_coords(p) for p in parts]
    ds = xr.merge(parts, compat="override", join="exact")
    ds = _ensure_grid_coords(ds)

    for spec in schema.SURFACE_VARS + schema.SATELLITE_VARS:
        if spec.name in ds:
            ds[spec.name] = ds[spec.name].transpose(*schema.SURFACE_DIMS)
    for spec in schema.LEVEL_VARS:
        if spec.name in ds:
            ds[spec.name] = ds[spec.name].transpose(*schema.LEVEL_DIMS)
    for spec in schema.STATIC_VARS:
        if spec.name in ds:
            ds[spec.name] = ds[spec.name].transpose(*schema.STATIC_DIMS)

    ds.attrs.setdefault("title", "SIH nowcast datacube")
    ds.attrs["grid_resolution_deg"] = config.GRID_RESOLUTION_DEG

    if validate:
        has_full_static = all(s.name in ds.data_vars for s in schema.STATIC_VARS)
        has_full_sat = all(s.name in ds.data_vars for s in schema.SATELLITE_VARS)
        schema.validate_datacube(
            ds,
            require_static=has_full_static,
            require_satellite=has_full_sat,
        )
    return ds


def write_datacube(
    ds: xr.Dataset, path: str | Path = config.DATACUBE_PATH, mode: str = "w"
) -> Path:
    """Write the datacube to Zarr at ``path`` (default ``config.DATACUBE_PATH``)."""
    return _io.write_zarr(ds, path, mode=mode)


def open_datacube(path: str | Path = config.DATACUBE_PATH) -> xr.Dataset:
    """Open the datacube Zarr store (default ``config.DATACUBE_PATH``)."""
    return _io.open_zarr(path)
