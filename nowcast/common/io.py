"""Coordinate standardisation and NetCDF / Zarr I/O helpers.

Raw reanalysis and satellite files use a zoo of coordinate names
(``latitude``, ``lat_0``, ``valid_time`` ...). :func:`standardize_coords`
renames them to the project convention (``lat`` / ``lon`` / ``time``), forces
ascending latitude and a ``[-180, 180]`` longitude convention so that every
downstream module can assume one layout.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

__all__ = ["standardize_coords", "open_netcdf", "write_zarr", "open_zarr"]

# Raw coordinate name -> canonical name.
_COORD_ALIASES: dict[str, str] = {
    "latitude": "lat",
    "lat_0": "lat",
    "g0_lat_0": "lat",
    "grid_latitude": "lat",
    "y": "lat",
    "nav_lat": "lat",
    "longitude": "lon",
    "lon_0": "lon",
    "g0_lon_0": "lon",
    "grid_longitude": "lon",
    "x": "lon",
    "nav_lon": "lon",
    "valid_time": "time",
    "time_counter": "time",
    "forecast_time0": "time",
    "t": "time",
    "Time": "time",
}

# Raw data-variable name -> canonical, applied opportunistically.
_VAR_ALIASES: dict[str, str] = {
    "level": "level",
    "lev": "level",
    "plev": "level",
    "pressure": "level",
    "isobaricInhPa": "level",
}


def standardize_coords(ds: xr.Dataset) -> xr.Dataset:
    """Return ``ds`` with canonical coordinate names, ascending lat, wrapped lon.

    * ``latitude``/``longitude``/``valid_time``/... are renamed to
      ``lat``/``lon``/``time``.
    * ``lat`` is sorted ascending.
    * ``lon`` is mapped into ``[-180, 180]`` and sorted ascending.
    * a vertical coordinate (``lev``/``plev``/``isobaricInhPa`` ...) becomes
      ``level``.
    """
    rename: dict[str, str] = {}
    for name in list(ds.variables):
        if name in _COORD_ALIASES and _COORD_ALIASES[name] not in ds.variables:
            rename[name] = _COORD_ALIASES[name]
        elif name in _VAR_ALIASES:
            target = _VAR_ALIASES[name]
            if target != name and target not in ds.variables:
                rename[name] = target
    if rename:
        ds = ds.rename(rename)

    if "lon" in ds.coords:
        lon = ((ds["lon"] + 180.0) % 360.0) - 180.0
        ds = ds.assign_coords(lon=lon)
        ds = ds.sortby("lon")
    if "lat" in ds.coords and ds["lat"].size > 1 and bool(ds["lat"][0] > ds["lat"][-1]):
        ds = ds.sortby("lat")
    if "level" in ds.coords:
        ds = ds.assign_coords(level=ds["level"].astype("float64"))

    return ds


def open_netcdf(path: str | Path, **kwargs) -> xr.Dataset:
    """Open a NetCDF (or NetCDF4/HDF5) file, trying the available engines.

    Returns the dataset with :func:`standardize_coords` already applied.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NetCDF file not found: {path}")
    last_err: Exception | None = None
    for engine in ("netcdf4", "h5netcdf", "scipy"):
        try:
            ds = xr.open_dataset(path, engine=engine, **kwargs)
            return standardize_coords(ds)
        except (ValueError, OSError, ImportError) as err:  # pragma: no cover - engine probing
            last_err = err
    raise OSError(f"could not open {path} with any NetCDF engine") from last_err


def write_zarr(ds: xr.Dataset, path: str | Path, mode: str = "w") -> Path:
    """Write ``ds`` to a Zarr store at ``path`` and return the path.

    Chunk encoding is dropped so re-writing an in-memory dataset never trips
    Zarr's "conflicting chunk" error.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    to_write = ds.copy()
    for var in to_write.variables:
        to_write[var].encoding.pop("chunks", None)
        to_write[var].encoding.pop("preferred_chunks", None)
    to_write.to_zarr(path, mode=mode, consolidated=True)
    return path


def open_zarr(path: str | Path, **kwargs) -> xr.Dataset:
    """Open a Zarr store written by :func:`write_zarr`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Zarr store not found: {path}")
    return xr.open_zarr(path, consolidated=True, **kwargs)


def _is_ascending(values: np.ndarray) -> bool:
    """True if ``values`` is monotonically non-decreasing (helper for tests)."""
    values = np.asarray(values)
    return bool(np.all(np.diff(values) >= 0))
