"""F6 regression: accumulated-precip -> rate conversion, including cycle resets.

`precip` is the only label source, so recovering the true hourly rate from a
cumulative accumulation that resets every forecast cycle must be exact -- also
across the reset boundary, where the old ``diff().clip(min=0)`` silently zeroed
real precipitation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from nowcast.ingest._util import accumulated_to_rate, step_interval_hours


def _cube(values_1d: list[float], *, hourly: bool = True) -> xr.DataArray:
    """A (time, lat, lon) DataArray, spatially constant per step."""
    arr = np.array(values_1d, dtype="float64")[:, None, None] * np.ones((1, 2, 3))
    if hourly:
        time = pd.date_range("2020-06-01", periods=len(values_1d), freq="1h")
    else:
        time = np.arange(len(values_1d))
    return xr.DataArray(
        arr.astype("float32"),
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": [10.0, 10.1], "lon": [70.0, 70.1, 70.2]},
    )


def test_cumulative_sawtooth_with_reset_recovers_hourly_rate():
    # cycle 1 accum: 2, 5, 9   (hourly increments 2, 3, 4)
    # cycle 2 accum: 1, 4, 8   (reset, then hourly increments 1, 3, 4)
    accum = _cube([2, 5, 9, 1, 4, 8])
    rate = accumulated_to_rate(accum, window_h=1.0, cumulative=True)
    expected = [2, 3, 4, 1, 3, 4]  # incl. the step straddling the reset
    np.testing.assert_allclose(rate.isel(lat=0, lon=0).values, expected, atol=1e-5)
    assert rate.attrs["units"] == "mm h-1"
    assert float(rate.min()) >= 0.0
    assert rate.dims == ("time", "lat", "lon")


def test_reset_is_not_zeroed():
    # old behaviour: diff at the reset is negative -> clipped to 0 -> lost rain
    accum = _cube([10, 20, 3, 9])  # reset between step 1 and 2; real hour-2 rain = 3
    rate = accumulated_to_rate(accum, window_h=1.0, cumulative=True)
    assert float(rate.isel(time=2, lat=0, lon=0)) == 3.0
    assert float(rate.isel(time=3, lat=0, lon=0)) == 6.0


def test_non_cumulative_is_just_divided_by_window():
    per_step = _cube([1, 2, 3, 4], hourly=False)
    rate = accumulated_to_rate(per_step, window_h=1.0, cumulative=False)
    np.testing.assert_allclose(rate.isel(lat=0, lon=0).values, [1, 2, 3, 4])


def test_three_hourly_accumulation_uses_real_interval():
    accum = _cube([6, 12, 18])
    accum = accum.assign_coords(time=pd.date_range("2020-06-01", periods=3, freq="3h"))
    rate = accumulated_to_rate(accum, window_h=3.0, cumulative=True)
    # 6 mm over the first 3 h -> 2 mm/h; then 6 mm / 3 h each step
    np.testing.assert_allclose(rate.isel(lat=0, lon=0).values, [2, 2, 2], atol=1e-5)


def test_step_interval_hours_from_time_axis():
    da = _cube([0, 0, 0, 0])
    np.testing.assert_allclose(step_interval_hours(da), [1, 1, 1, 1])
    da3 = da.assign_coords(time=pd.date_range("2020-06-01", periods=4, freq="3h"))
    np.testing.assert_allclose(step_interval_hours(da3), [3, 3, 3, 3])
    di = _cube([0, 0, 0], hourly=False)
    np.testing.assert_allclose(step_interval_hours(di, fallback_h=1.0), [1, 1, 1])


def test_single_step_falls_back_to_window():
    one = _cube([5.0])
    rate = accumulated_to_rate(one, window_h=2.0, cumulative=True)
    assert float(rate.isel(time=0, lat=0, lon=0)) == 2.5
