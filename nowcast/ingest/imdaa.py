"""IMDAA (NCMRWF regional reanalysis) loaders.

IMDAA is the primary dynamical source for the datacube: 2 m / 10 m fields,
MSLP, column water vapour, CAPE/CIN and precipitation on single levels, plus
temperature, humidity, geopotential and wind on pressure levels.

Both loaders:

1. open the NetCDF file(s) with :func:`nowcast.common.io.open_netcdf`
   (coords already standardised);
2. rename + unit-convert via :mod:`nowcast.ingest.names`;
3. crop to the pilot bbox and interpolate onto the target grid;
4. return an :class:`xarray.Dataset` in schema units.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import xarray as xr

from nowcast import config
from nowcast.common import grid as _grid
from nowcast.common import io as _io
from nowcast.ingest import names as _names
from nowcast.ingest._util import apply_var_map, select_schema_vars

__all__ = ["load_imdaa_single_level", "load_imdaa_pressure_level"]

_SURFACE_WANTED: tuple[str, ...] = (
    "t2m",
    "u10",
    "v10",
    "prmsl",
    "tcwv",
    "precip",
    "cape",
    "cin",
)
_LEVEL_WANTED: tuple[str, ...] = ("t", "rh", "z", "u", "v")


def _as_paths(paths: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        return [Path(paths)]
    return [Path(p) for p in paths]


def _open_many(paths: Sequence[Path]) -> xr.Dataset:
    """Open and merge one or more IMDAA NetCDF files onto a common layout."""
    if not paths:
        raise ValueError("no IMDAA paths supplied")
    datasets = [_io.open_netcdf(p) for p in paths]
    if len(datasets) == 1:
        return datasets[0]
    try:
        return xr.merge(datasets, compat="override", join="outer")
    except (ValueError, xr.MergeError):
        return xr.concat(datasets, dim="time")


def load_imdaa_single_level(
    paths: str | Path | Iterable[str | Path],
    *,
    accum_window_h: float | None = None,
    cumulative_accum: bool | None = None,
) -> xr.Dataset:
    """Load IMDAA single-level fields onto the target grid.

    Parameters
    ----------
    paths:
        One path or an iterable of paths to IMDAA single-level NetCDF files.
    accum_window_h, cumulative_accum:
        Optional overrides for the ``APCP`` accumulation convention. ``None``
        (default) trusts the documented product convention encoded on
        ``names.IMDAA_SINGLE_LEVEL["APCP_sfc"]`` (cumulative, resets each
        cycle, 1 h first step).

    Returns
    -------
    xarray.Dataset
        Variables ``t2m, u10, v10, prmsl, tcwv, precip`` (and ``cape``/``cin``
        when present) with dims ``(time, lat, lon)`` on the target grid.
    """
    raw = _open_many(_as_paths(paths))
    mapped = apply_var_map(
        raw,
        _names.IMDAA_SINGLE_LEVEL,
        accum_window_h=accum_window_h,
        cumulative_accum=cumulative_accum,
    )
    mapped = select_schema_vars(mapped, _SURFACE_WANTED)
    mapped = _grid.crop_bbox(mapped)
    mapped = _grid.regrid_to_target(mapped, method="linear")
    if "precip" in mapped:
        mapped["precip"] = mapped["precip"].clip(min=0.0)
    mapped.attrs["source"] = "IMDAA single-level"
    return mapped


def load_imdaa_pressure_level(
    paths: str | Path | Iterable[str | Path],
) -> xr.Dataset:
    """Load IMDAA pressure-level fields onto the target grid.

    Returns
    -------
    xarray.Dataset
        Variables ``t, rh, z, u, v`` with dims ``(time, level, lat, lon)`` on
        the target grid, ``level`` restricted/ordered to
        ``config.PRESSURE_LEVELS_HPA``.
    """
    raw = _open_many(_as_paths(paths))
    mapped = apply_var_map(raw, _names.IMDAA_PRESSURE_LEVEL)
    mapped = select_schema_vars(mapped, _LEVEL_WANTED)
    if "level" not in mapped.coords:
        raise ValueError("IMDAA pressure-level file has no vertical coordinate")
    wanted_levels = [
        lev for lev in config.PRESSURE_LEVELS_HPA if lev in set(mapped["level"].values.tolist())
    ]
    if wanted_levels:
        mapped = mapped.sel(level=wanted_levels)
    mapped = _grid.crop_bbox(mapped)
    mapped = _grid.regrid_to_target(mapped, method="linear")
    mapped = mapped.transpose("time", "level", "lat", "lon", missing_dims="ignore")
    mapped.attrs["source"] = "IMDAA pressure-level"
    return mapped
