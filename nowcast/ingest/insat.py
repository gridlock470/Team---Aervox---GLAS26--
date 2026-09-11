"""INSAT-3D / 3DR imager loaders: L1C brightness temperatures and QPE.

``load_insat_l1c`` returns cloud-top temperature (``ctt``, TIR-1 brightness
temperature) and the water-vapour channel brightness temperature (``wv_bt``).
``load_insat_qpe`` returns the operational rain rate as ``precip`` (``mm h-1``).

MOSDAC distributes these as HDF5 (``.h5``). They are opened through
:func:`nowcast.common.io.open_netcdf`, which probes the ``netcdf4`` and
``h5netcdf`` engines -- both read the HDF5 container. Real files may need the
per-file radiance/BT lookup tables for the raw ``IMG_*`` count datasets; this
loader prefers the ready-made ``IMG_*_TEMP`` datasets and documents the
limitation.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import xarray as xr

from nowcast import config
from nowcast.common import grid as _grid
from nowcast.common import io as _io
from nowcast.ingest import names as _names
from nowcast.ingest._util import apply_var_map, resample_to_step, select_schema_vars

__all__ = ["load_insat_l1c", "load_insat_qpe"]


def _open_concat(paths: str | Path | Iterable[str | Path]) -> xr.Dataset:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    datasets = [_io.open_netcdf(p) for p in paths]
    if len(datasets) == 1:
        return datasets[0]
    return xr.concat(datasets, dim="time")


def load_insat_l1c(
    paths: str | Path | Iterable[str | Path], *, resample: bool = True
) -> xr.Dataset:
    """Load INSAT L1C imager brightness temperatures onto the target grid.

    Parameters
    ----------
    paths:
        One or more INSAT L1C HDF5 granules.
    resample:
        Snap the native (sub-hourly, irregular) acquisition times onto the
        datacube step ``config.TIMESTEP`` by nearest match within one step
        (brightness temperature is a state field, so nearest -- not mean).

    Returns
    -------
    xarray.Dataset
        ``ctt`` (TIR-1 BT, K) and ``wv_bt`` (water-vapour BT, K) with dims
        ``(time, lat, lon)``.
    """
    raw = _open_concat(paths)
    mapped = apply_var_map(raw, _names.INSAT_L1C_VARS)
    mapped = select_schema_vars(mapped, ("ctt", "wv_bt"))
    if "ctt" not in mapped:
        raise ValueError("no INSAT TIR-1 brightness-temperature dataset recognised")
    if resample:
        mapped = resample_to_step(
            mapped, config.TIMESTEP, how="nearest", tolerance=config.TIMESTEP
        )
    mapped = _grid.crop_bbox(mapped)
    mapped = _grid.regrid_to_target(mapped, method="linear")
    mapped.attrs["source"] = "INSAT-3D/3DR L1C"
    return mapped


def load_insat_qpe(
    paths: str | Path | Iterable[str | Path], *, resample: bool = True
) -> xr.Dataset:
    """Load INSAT QPE rain rate onto the target grid as ``precip`` (``mm h-1``).

    ``resample`` bin-averages the native sub-hourly rain rate onto
    ``config.TIMESTEP`` (mean, correct for a rate).
    """
    raw = _open_concat(paths)
    mapped = apply_var_map(raw, _names.INSAT_QPE_VARS)
    mapped = select_schema_vars(mapped, ("precip",))
    if "precip" not in mapped:
        raise ValueError("no INSAT QPE rain-rate dataset recognised")
    if resample:
        mapped = resample_to_step(mapped, config.TIMESTEP, how="mean")
    mapped = _grid.crop_bbox(mapped)
    mapped = _grid.regrid_to_target(mapped, method="linear")
    mapped["precip"] = mapped["precip"].clip(min=0.0)
    mapped.attrs["source"] = "INSAT-3D/3DR QPE"
    return mapped
