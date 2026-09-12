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

__all__ = [
    "standardize_coords",
    "cftime_to_datetime64",
    "open_netcdf",
    "write_zarr",
    "open_zarr",
]

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
    "pressure_level": "level",  # ERA5 CDS NetCDF
}


def _is_cftime(value) -> bool:
    """True when ``value`` looks like a :mod:`cftime` datetime object."""
    return all(
        hasattr(value, attr)
        for attr in ("year", "month", "day", "hour", "minute", "second")
    ) and not isinstance(value, np.datetime64)


def cftime_to_datetime64(values: np.ndarray) -> np.ndarray:
    """Convert a ``cftime`` object axis to ``datetime64[ns]``, field for field.

    Some products label their time axis with a calendar xarray does not treat
    as standard even though the values are ordinary UTC -- GPM IMERG declares
    ``calendar = "julian"`` on ``seconds since 1980-01-06``, so xarray decodes
    it to :class:`cftime.DatetimeJulian` objects. Reading the calendar fields
    verbatim (the same rule :meth:`xarray.CFTimeIndex.to_datetimeindex` uses)
    recovers the intended timestamps, which matters because every downstream
    ``np.datetime64`` guard (resampling, reindex tolerances, gap detection)
    silently no-ops on an object-dtype axis.
    """
    flat = np.asarray(values).ravel()
    stamps = [
        np.datetime64(
            f"{t.year:04d}-{t.month:02d}-{t.day:02d}T"
            f"{t.hour:02d}:{t.minute:02d}:{t.second:02d}."
            f"{getattr(t, 'microsecond', 0):06d}",
            "ns",
        )
        for t in flat
    ]
    return np.asarray(stamps, dtype="datetime64[ns]").reshape(np.asarray(values).shape)


def standardize_coords(ds: xr.Dataset) -> xr.Dataset:
    """Return ``ds`` with canonical coordinate names, ascending lat, wrapped lon.

    * ``latitude``/``longitude``/``valid_time``/... are renamed to
      ``lat``/``lon``/``time``.
    * ``lat`` is sorted ascending.
    * ``lon`` is mapped into ``[-180, 180]`` and sorted ascending.
    * a vertical coordinate (``lev``/``plev``/``isobaricInhPa`` ...) becomes
      ``level``.
    * a ``cftime``-decoded ``time`` axis becomes ``datetime64[ns]``.
    """
    rename: dict[str, str] = {}
    for name in list(ds.variables):
        # Coordinate aliases must only ever fire on actual coordinates/dims.
        # Several aliases collide with legitimate data-variable names -- ECMWF
        # calls temperature ``t``, which must not become ``time``.
        is_coord_like = name in ds.coords or name in ds.dims
        if is_coord_like and name in _COORD_ALIASES and _COORD_ALIASES[name] not in ds.variables:
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
    if (
        "time" in ds.coords
        and ds["time"].dtype == object
        and ds["time"].size
        and _is_cftime(np.asarray(ds["time"].values).ravel()[0])
    ):
        ds = ds.assign_coords(time=cftime_to_datetime64(ds["time"].values))

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


# Hours per Zarr chunk along ``time``. Consumers read short contiguous windows
# (``config.INPUT_SEQ_LEN`` is 12 hours) across the FULL spatial and level grid,
# so time is the only axis worth splitting. Left to itself Zarr auto-chunks a
# multi-month cube into quarters of the record and shards lat/lon as well, which
# turns one 12-hour training window into dozens of chunk reads.
DEFAULT_TIME_CHUNK: int = 96


def write_zarr(
    ds: xr.Dataset,
    path: str | Path,
    mode: str = "w",
    *,
    time_chunk: int | None = DEFAULT_TIME_CHUNK,
) -> Path:
    """Write ``ds`` to a Zarr store at ``path`` and return the path.

    Incoming chunk encoding is dropped so re-writing an in-memory dataset never
    trips Zarr's "conflicting chunk" error. ``time_chunk`` then re-imposes a
    read-friendly layout: every in-memory variable with a ``time`` dimension is
    chunked ``time_chunk`` steps deep and left whole on every other axis. Pass
    ``None`` to keep Zarr's own guess. Dask-backed variables are left alone --
    their existing graph chunks drive the write, and overriding them is what
    raises the conflicting-chunk error.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    to_write = ds.copy()
    for name in to_write.variables:
        var = to_write[name]
        var.encoding.pop("chunks", None)
        var.encoding.pop("preferred_chunks", None)
        if time_chunk is None or "time" not in var.dims:
            continue
        if not isinstance(var.data, np.ndarray):
            continue  # dask-backed: let its own chunks win
        var.encoding["chunks"] = tuple(
            min(int(time_chunk), var.sizes[d]) if d == "time" else var.sizes[d]
            for d in var.dims
        )
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
