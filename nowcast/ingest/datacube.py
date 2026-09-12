"""Assemble source fields into the validated datacube and read/write it.

:func:`build_datacube` merges the standardised surface / level / satellite /
static fields onto one :class:`xarray.Dataset` with the schema dimension order,
runs :func:`nowcast.schema.validate_datacube`, and returns it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from nowcast import config, schema
from nowcast.common import io as _io

__all__ = ["build_datacube", "write_datacube", "open_datacube"]

_LEVELS = np.asarray(config.PRESSURE_LEVELS_HPA, dtype="float64")


def _align_time(
    ds: xr.Dataset,
    ref_time: xr.DataArray | None,
    *,
    tolerance: str | None = None,
    method: str | None = None,
) -> xr.Dataset:
    """Reindex ``ds`` onto ``ref_time`` (the surface/IMDAA clock).

    ``tolerance`` (a pandas offset string) bounds a nearest-match reindex so a
    stale sample is never carried across a gap; without it an exact reindex is
    used. ``method="interp"`` linearly interpolates along time instead, which is
    what a coarser-cadence source needs (see :func:`_interp_time`). A missing /
    non-datetime time axis passes through untouched.
    """
    if ref_time is None or "time" not in ds.coords:
        return ds
    if not np.issubdtype(np.asarray(ds["time"].values).dtype, np.datetime64):
        return ds
    if method == "interp":
        return _interp_time(ds, ref_time)
    if tolerance is None:
        return ds.reindex(time=ref_time)
    return ds.reindex(
        time=ref_time, method="nearest", tolerance=pd.Timedelta(tolerance)
    )


def _interp_time(ds: xr.Dataset, ref_time: xr.DataArray) -> xr.Dataset:
    """Linearly interpolate ``ds`` onto ``ref_time``, one variable at a time.

    Real ERA5/IMDAA pressure-level archives are 3-hourly while the single-level
    fields are hourly, so an exact reindex leaves every level variable NaN at
    two hours in three -- and ``schema.validate_sample`` rejects any input
    tensor containing NaN, so the cube would be untrainable. Interpolating in
    time is the standard treatment for the smooth, slowly-varying pressure-level
    state fields; steps outside the source's own range stay NaN rather than
    being extrapolated.

    Variables are interpolated and cast back to their input dtype one at a time
    because ``interp`` materialises float64: doing the whole block at once needs
    roughly twice the peak memory of the finished cube.
    """
    if ds["time"].equals(ref_time):
        return ds
    timed = [n for n, v in ds.data_vars.items() if "time" in v.dims]
    rebuilt: dict[str, xr.DataArray] = {
        n: v for n, v in ds.data_vars.items() if "time" not in v.dims
    }
    for name in timed:
        var = ds[name]
        interpolated = var.interp(time=ref_time, method="linear")
        if np.issubdtype(var.dtype, np.floating):
            interpolated = interpolated.astype(var.dtype, copy=False)
        rebuilt[name] = interpolated
        # release the coarse-cadence source as we go; holding all of them plus
        # all of the hourly output at once is what blows the memory budget.
        ds = ds.drop_vars(name)
    out = xr.Dataset(rebuilt)
    out.attrs = dict(ds.attrs)
    out.attrs["time_alignment"] = "linear interpolation onto the surface clock"
    return out


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
    level_time_method: str | None = "interp",
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
    level_time_method:
        How to put ``level`` on the surface clock. ``"interp"`` (default)
        linearly interpolates along time, which is what real archives need --
        ERA5/IMDAA pressure levels are 3-hourly while the single-level fields
        are hourly, and an exact reindex would leave every level variable NaN at
        two hours in three. ``None`` restores the exact reindex. Either way this
        is a no-op when the two clocks already match.

    Returns
    -------
    xarray.Dataset
        The datacube. Dimension order follows ``schema.SURFACE_DIMS`` /
        ``schema.LEVEL_DIMS`` / ``schema.STATIC_DIMS``.
    """
    ref_time = surface["time"] if "time" in surface.coords else None
    parts: list[xr.Dataset] = [
        surface,
        _align_time(level, ref_time, method=level_time_method),
    ]
    if satellite is not None:
        parts.append(_align_time(satellite, ref_time, tolerance=config.TIMESTEP))
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
    ds = xr.merge(parts, compat="override", join="outer")
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
