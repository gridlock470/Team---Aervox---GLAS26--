"""Internal helpers shared by the source loaders (not a public API)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from nowcast.ingest import names as _names

__all__ = [
    "apply_var_map",
    "accumulated_to_rate",
    "select_schema_vars",
    "step_interval_hours",
    "resample_to_step",
]


def step_interval_hours(da: xr.DataArray, fallback_h: float = 1.0) -> np.ndarray:
    """Per-step spacing (hours) of ``da``'s ``time`` axis.

    Returns an array of length ``time`` where element ``i`` is the number of
    hours between step ``i-1`` and step ``i``; element 0 uses the median
    spacing (or ``fallback_h`` when the axis is not datetime / has < 2 steps).
    """
    n = int(da.sizes.get("time", 1))
    if "time" not in da.coords or n < 2:
        return np.full(max(n, 1), float(fallback_h))
    t = np.asarray(da["time"].values)
    if not np.issubdtype(t.dtype, np.datetime64):
        return np.full(n, float(fallback_h))
    deltas = np.diff(t).astype("timedelta64[s]").astype("float64") / 3600.0
    deltas = np.where(deltas > 0, deltas, float(fallback_h))
    first = float(np.median(deltas)) if deltas.size else float(fallback_h)
    return np.concatenate([[first], deltas])


def accumulated_to_rate(
    da: xr.DataArray,
    window_h: float = 1.0,
    *,
    cumulative: bool = False,
) -> xr.DataArray:
    """Convert an accumulated precip depth (mm) to a rate in ``mm h-1``.

    Parameters
    ----------
    da:
        Accumulated depth in mm.
    window_h:
        Nominal length (hours) of the first step of each accumulation cycle.
    cumulative:
        If ``False`` each value is the depth that fell during the single step
        ending at that timestamp -> just divide by the real step spacing.
        If ``True`` the values are a running total since forecast-cycle init
        that resets (drops) at the start of every cycle: the per-step depth is
        recovered by differencing, restarting the difference at each reset
        (where the running total decreases), and the first step of each cycle
        is taken as its own value. Every per-step depth is then divided by the
        real spacing (hours) of that step.
    """
    if "time" not in da.dims or da.sizes["time"] < 2:
        rate = da / float(window_h)
        rate.attrs["units"] = "mm h-1"
        return rate

    orig_dims = da.dims
    da_t = da.transpose("time", ...)
    vals = np.asarray(da_t.values, dtype="float64")

    if cumulative:
        prev = np.empty_like(vals)
        prev[0] = 0.0
        prev[1:] = vals[:-1]
        incr = vals - prev
        # a decrease along time marks a forecast-cycle reset: the value itself
        # is the accumulation since that reset, not (value - previous).
        reset = incr < 0.0
        incr = np.where(reset, vals, incr)
    else:
        incr = vals

    incr = np.clip(incr, 0.0, None)

    hrs = step_interval_hours(da_t, fallback_h=window_h)
    bcast = (-1,) + (1,) * (incr.ndim - 1)
    rate_vals = (incr / hrs.reshape(bcast)).astype("float32")

    out = xr.DataArray(rate_vals, dims=da_t.dims, coords=da_t.coords, attrs=da.attrs)
    out = out.transpose(*orig_dims)
    out.attrs["units"] = "mm h-1"
    return out


def apply_var_map(
    ds: xr.Dataset,
    table: dict[str, _names.VarMap],
    *,
    accum_window_h: float | None = None,
    cumulative_accum: bool | None = None,
) -> xr.Dataset:
    """Rename + unit-convert every mapped variable in ``ds``.

    Returns a new dataset containing only the variables found in ``table``,
    named and scaled to the schema contract. Accumulated variables are turned
    into ``mm h-1`` rates using the per-variable convention declared on the
    :class:`~nowcast.ingest.names.VarMap` (``cumulative`` / ``accum_window_h``).

    Parameters
    ----------
    accum_window_h, cumulative_accum:
        Optional explicit overrides applied to *every* accumulated variable in
        the table (use only when a specific file is known to differ from the
        documented product convention). ``None`` -> trust the ``VarMap``.
    """
    out: dict[str, xr.DataArray] = {}
    for raw_name in list(ds.data_vars):
        vm = _names.lookup(table, raw_name)
        if vm is None:
            continue
        da = vm.convert(ds[raw_name])
        if vm.accumulated:
            window = vm.accum_window_h if accum_window_h is None else accum_window_h
            cumulative = vm.cumulative if cumulative_accum is None else cumulative_accum
            da = accumulated_to_rate(da, window_h=window, cumulative=cumulative)
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


def resample_to_step(
    obj: xr.Dataset | xr.DataArray,
    step: str,
    *,
    how: str = "mean",
    tolerance: str | pd.Timedelta | None = None,
) -> xr.Dataset | xr.DataArray:
    """Resample ``obj`` onto a regular ``step`` (pandas offset) time axis.

    ``how="mean"`` bin-averages (correct for rates such as precip);
    ``how="nearest"`` snaps each target step to the closest source sample
    within ``tolerance`` (correct for state fields such as brightness
    temperature). A non-datetime or single-step ``time`` axis is returned
    unchanged.
    """
    if "time" not in getattr(obj, "coords", {}) or obj["time"].size < 2:
        return obj
    if not np.issubdtype(np.asarray(obj["time"].values).dtype, np.datetime64):
        return obj

    if how == "mean":
        return obj.resample(time=step).mean()
    if how == "nearest":
        t = pd.DatetimeIndex(obj["time"].values)
        target = pd.date_range(t.min().floor(step), t.max().ceil(step), freq=step)
        tol = pd.Timedelta(tolerance) if tolerance is not None else None
        return obj.reindex(time=target, method="nearest", tolerance=tol)
    raise ValueError(f"unknown resample method {how!r}")
