"""Hazard occurrence labels and smoothed probability targets.

``build_labels`` produces a DataArray with dims :data:`nowcast.schema.TARGET_DIMS`
(``time, hazard, lead, lat, lon``) and values in ``[0, 1]``. For each hazard a
binary occurrence field is derived from the datacube, shifted so that
``label(t, lead=h) == occurrence(t + h)`` for every ``h`` in
``config.LEAD_TIMES_H``, then Gaussian-smoothed per ``(hazard, lead)`` map.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter

from nowcast import config, schema

# ESRI D8 flow-direction codes -> (row offset, col offset).
_D8_OFFSETS: dict[int, tuple[int, int]] = {
    1: (0, 1),
    2: (1, 1),
    4: (1, 0),
    8: (1, -1),
    16: (0, -1),
    32: (-1, -1),
    64: (-1, 0),
    128: (-1, 1),
}
_SPATIAL = ("time", "lat", "lon")


def _thunderstorm_mask(precip: xr.DataArray, ctt: xr.DataArray | None) -> xr.DataArray:
    mask = precip > config.THUNDERSTORM_PRECIP_MM_H
    if ctt is not None:
        mask = mask | ((ctt < config.THUNDERSTORM_CTT_MAX_K) & (precip > 1.0))
    return mask


def _cloudburst_mask(precip: xr.DataArray) -> xr.DataArray:
    accum_2h = precip.rolling(time=2, min_periods=1).sum()
    return (accum_2h > config.CLOUDBURST_ACCUM_MM_2H) | (
        precip > config.CLOUDBURST_PRECIP_MM_H
    )


def _upstream_accumulate(field_2d: np.ndarray, flow_dir_2d: np.ndarray) -> np.ndarray:
    """D8 upstream accumulation of ``field_2d`` following ``flow_dir_2d``.

    Each output cell holds the sum of ``field_2d`` over itself and every cell
    that drains through it (Kahn topological sweep). Cells caught in a
    flow-direction cycle - possible with synthetic/random pointers - simply stop
    propagating once the acyclic frontier is exhausted; a real D8 grid from
    ``nowcast.dem.routing`` is acyclic and fully resolved.
    """
    field_2d = np.asarray(field_2d, dtype="float64")
    flow_dir_2d = np.asarray(flow_dir_2d)
    h, w = field_2d.shape
    n = h * w
    receiver = np.full(n, -1, dtype=np.int64)
    indeg = np.zeros(n, dtype=np.int64)

    for i in range(h):
        for j in range(w):
            code = int(round(float(flow_dir_2d[i, j])))
            off = _D8_OFFSETS.get(code)
            if off is None:
                continue
            ni, nj = i + off[0], j + off[1]
            if 0 <= ni < h and 0 <= nj < w:
                receiver[i * w + j] = ni * w + nj
                indeg[ni * w + nj] += 1

    acc = field_2d.ravel().copy()
    queue = deque(int(k) for k in np.flatnonzero(indeg == 0))
    while queue:
        k = queue.popleft()
        r = int(receiver[k])
        if r < 0:
            continue
        acc[r] += acc[k]
        indeg[r] -= 1
        if indeg[r] == 0:
            queue.append(r)
    return acc.reshape(h, w)


def _lowest_neighbour_spread(mask_t: np.ndarray, elevation: np.ndarray) -> np.ndarray:
    """Flag each masked cell plus its single lowest 8-neighbour (downhill)."""
    out = mask_t.copy()
    h, w = elevation.shape
    rows, cols = np.where(mask_t)
    for i, j in zip(rows, cols, strict=True):
        best = None
        best_elev = elevation[i, j]
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < h and 0 <= nj < w and elevation[ni, nj] < best_elev:
                    best_elev = elevation[ni, nj]
                    best = (ni, nj)
        if best is not None:
            out[best] = True
    return out


def _flash_flood_mask(
    precip: xr.DataArray,
    cloudburst: xr.DataArray,
    ds: xr.Dataset,
    routing: dict | None,
) -> xr.DataArray:
    dims = [d for d in _SPATIAL if d in precip.dims]
    accum_2h = precip.rolling(time=2, min_periods=1).sum().transpose(*dims)
    cb = cloudburst.transpose(*dims)
    rain = accum_2h.where(cb, 0.0)
    rain_vals = np.asarray(rain.values, dtype="float64")
    out = np.zeros(rain_vals.shape, dtype=bool)

    if routing is not None and "flow_direction" in routing:
        fdir = routing["flow_direction"]
        fdir = np.asarray(getattr(fdir, "values", fdir))
        for t in range(rain_vals.shape[0]):
            acc = _upstream_accumulate(rain_vals[t], fdir)
            out[t] = acc > config.FLASH_FLOOD_UPSTREAM_ACCUM_MM
    else:
        # Fallback (documented): no routing network available, so approximate
        # drainage by flagging each cloudburst cell and its lowest neighbour.
        elevation = np.asarray(
            ds["elevation"].transpose("lat", "lon").values, dtype="float64"
        )
        cb_vals = np.asarray(cb.values, dtype=bool)
        for t in range(cb_vals.shape[0]):
            out[t] = _lowest_neighbour_spread(cb_vals[t], elevation)

    return xr.DataArray(out, dims=dims, coords={d: precip[d] for d in dims})


def _shift_occurrence(base: np.ndarray, lead: int) -> np.ndarray:
    """``result[t] = base[t + lead]`` with zero padding past the end."""
    shifted = np.zeros_like(base)
    n = base.shape[0]
    if lead < n:
        shifted[: n - lead] = base[lead:]
    return shifted


def build_labels(ds: xr.Dataset, routing: dict | None = None) -> xr.DataArray:
    """Build smoothed hazard-probability targets, dims ``schema.TARGET_DIMS``.

    ``routing`` (optional) is the drainage network from
    ``nowcast.dem.routing`` - a mapping with keys ``flow_direction`` and
    ``flow_accumulation``. When absent, the flash-flood mask falls back to
    cloudburst cells spread one step downhill.
    """
    precip = ds["precip"]
    dims = [d for d in _SPATIAL if d in precip.dims]
    precip = precip.transpose(*dims)
    ctt = ds["ctt"].transpose(*dims) if "ctt" in ds.data_vars else None

    occurrence = {
        "thunderstorm": _thunderstorm_mask(precip, ctt),
        "cloudburst": _cloudburst_mask(precip),
        "flash_flood": _flash_flood_mask(
            precip, _cloudburst_mask(precip), ds, routing
        ),
    }

    time = precip["time"]
    lat = precip["lat"]
    lon = precip["lon"]
    leads = config.LEAD_TIMES_H
    sigma = float(config.LABEL_SMOOTH_SIGMA)
    n_time = time.size

    out = np.zeros(
        (n_time, schema.N_HAZARDS, schema.N_LEADS, lat.size, lon.size), dtype="float32"
    )
    for hi, hazard in enumerate(config.HAZARDS):
        base = np.asarray(
            occurrence[hazard].transpose(*dims).values, dtype="float32"
        )
        for li, lead in enumerate(leads):
            shifted = _shift_occurrence(base, lead)
            smoothed = gaussian_filter(shifted, sigma=(0.0, sigma, sigma))
            out[:, hi, li] = np.clip(smoothed, 0.0, 1.0)

    return xr.DataArray(
        out,
        dims=schema.TARGET_DIMS,
        coords={
            "time": time,
            "hazard": list(config.HAZARDS),
            "lead": list(leads),
            "lat": lat,
            "lon": lon,
        },
        name="hazard_probability",
        attrs={"long_name": "hazard occurrence probability", "units": "1"},
    )


def write_labels(da: xr.DataArray, path: str | Path) -> Path:
    """Write labels to ``path`` as NetCDF and return the path."""
    path = Path(path)
    da.to_dataset(name="hazard_probability").to_netcdf(path)
    return path


def open_labels(path: str | Path) -> xr.DataArray:
    """Open (and eagerly load) labels written by :func:`write_labels`."""
    return xr.open_dataset(Path(path))["hazard_probability"].load()
