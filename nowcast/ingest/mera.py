"""MERA reanalysis precipitation loader (secondary precip source).

Only the precipitation field is taken from MERA; it is the preferred input to
:func:`nowcast.ingest.merge_precip.merge_precip` where finite.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import xarray as xr

from nowcast.common import grid as _grid
from nowcast.common import io as _io
from nowcast.ingest import names as _names
from nowcast.ingest._util import apply_var_map, select_schema_vars

__all__ = ["load_mera"]


def load_mera(
    paths: str | Path | Iterable[str | Path],
    *,
    accum_window_h: float = 1.0,
    cumulative_accum: bool = True,
) -> xr.Dataset:
    """Load MERA precipitation onto the target grid as ``precip`` (``mm h-1``).

    Parameters
    ----------
    paths:
        One or more MERA NetCDF files.
    accum_window_h:
        Accumulation window in hours for an accumulated raw field.
    cumulative_accum:
        Whether the raw accumulation is a running total (MERA ``tp`` usually
        is) rather than a per-step value.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    datasets = [_io.open_netcdf(p) for p in paths]
    raw = datasets[0] if len(datasets) == 1 else xr.concat(datasets, dim="time")

    mapped = apply_var_map(
        raw,
        _names.MERA_VARS,
        accum_window_h=accum_window_h,
        cumulative_accum=cumulative_accum,
    )
    mapped = select_schema_vars(mapped, ("precip",))
    if "precip" not in mapped:
        raise ValueError("no MERA precipitation variable recognised in the input files")
    mapped = _grid.crop_bbox(mapped)
    mapped = _grid.regrid_to_target(mapped, method="linear")
    mapped["precip"] = mapped["precip"].clip(min=0.0).transpose("time", "lat", "lon")
    mapped.attrs["source"] = "MERA"
    return mapped
