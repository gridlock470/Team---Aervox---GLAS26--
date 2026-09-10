"""Internal helpers shared by the source loaders (not a public API)."""

from __future__ import annotations

import numpy as np
import xarray as xr

from nowcast.ingest import names as _names

__all__ = ["apply_var_map", "accumulated_to_rate", "select_schema_vars"]


def accumulated_to_rate(
    da: xr.DataArray, window_h: float = 1.0, *, cumulative: bool = False
) -> xr.DataArray:
    """Convert an accumulated precip depth (mm) to a rate in ``mm h-1``.

    Parameters
    ----------
    da:
        Accumulated depth. Either a per-step accumulation over ``window_h``
        hours (``cumulative=False``) or a running total since some reference
        time (``cumulative=True``).
    window_h:
        Length of the accumulation window in hours.
    cumulative:
        If ``True``, difference successive time steps before dividing; the
        first step is divided by ``window_h`` as-is.
    """
    if cumulative and "time" in da.dims and da.sizes["time"] > 1:
        orig_dims = da.dims
        diffed = da.diff("time")
        first = da.isel(time=0)
        da = xr.concat([first, diffed], dim="time").clip(min=0.0)
        da = da.transpose(*orig_dims)
    rate = da / float(window_h)
    rate.attrs["units"] = "mm h-1"
    return rate


def apply_var_map(
    ds: xr.Dataset,
    table: dict[str, _names.VarMap],
    *,
    accum_window_h: float = 1.0,
    cumulative_accum: bool = False,
) -> xr.Dataset:
    """Rename + unit-convert every mapped variable in ``ds``.

    Returns a new dataset containing only the variables found in ``table``,
    named and scaled to the schema contract. Accumulated variables are turned
    into ``mm h-1`` rates.
    """
    out: dict[str, xr.DataArray] = {}
    for raw_name in list(ds.data_vars):
        vm = _names.lookup(table, raw_name)
        if vm is None:
            continue
        da = vm.convert(ds[raw_name])
        if vm.accumulated:
            da = accumulated_to_rate(
                da, window_h=accum_window_h, cumulative=cumulative_accum
            )
        da = da.astype("float32")
        da.attrs.setdefault("units", vm.raw_units)
        if vm.schema_name in out:
            # keep the first finite estimate, fill gaps from the later one
            da = xr.where(np.isfinite(out[vm.schema_name]), out[vm.schema_name], da)
        out[vm.schema_name] = da
    return xr.Dataset(out, coords={k: ds.coords[k] for k in ds.coords})


def select_schema_vars(ds: xr.Dataset, wanted: tuple[str, ...]) -> xr.Dataset:
    """Return ``ds`` restricted to the ``wanted`` variables that are present."""
    present = [v for v in wanted if v in ds.data_vars]
    return ds[present]
