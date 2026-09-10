"""Satellite lead-signal predictors from INSAT cloud-top temperature (CTT)."""

from __future__ import annotations

import numpy as np
import xarray as xr

_TIME_DIM = "time"


def _first_step_backfill(da: xr.DataArray, dim: str = _TIME_DIM) -> xr.DataArray:
    """Replace the (NaN) first sample along ``dim`` with the second sample."""
    if da.sizes.get(dim, 0) < 2:
        return da.fillna(0.0)
    data = np.array(da.values, copy=True)
    data[0] = data[1]
    return da.copy(data=data)


def ctt_drop_rate(ds: xr.Dataset, hours: int = 1) -> xr.DataArray:
    """Cloud-top cooling rate in K per ``hours``: ``-d(ctt)/dt``.

    A cooling cloud top (deepening convection) yields a positive value. The
    first step is back-filled from the second.
    """
    ctt = ds["ctt"].astype("float64")
    dims = [d for d in ("time", "lat", "lon") if d in ctt.dims]
    ctt = ctt.transpose(*dims)
    rate = -(ctt - ctt.shift({_TIME_DIM: 1})) / float(hours)
    rate = _first_step_backfill(rate)
    return rate.astype("float32").rename("ctt_drop_rate_1h").assign_attrs(
        units="K h-1", long_name="cloud-top temperature cooling rate"
    )
