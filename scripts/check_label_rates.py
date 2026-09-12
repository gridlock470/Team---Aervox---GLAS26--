"""Verify hazard label thresholds against the real IMERG rainfall record.

READ-ONLY audit. Nothing under ``DataSet/`` and no module under ``nowcast/`` is
modified; if a configured threshold looks wrong this script *reports* it.

The point of this check is that ``config.LABEL_OCCURRENCE_THRESHOLD`` and the
per-hazard precipitation thresholds have never been compared against observed
rainfall. A threshold that is far too loose makes "cloudburst" an everyday
event; one that is too strict yields zero positives and nothing trains.

What it measures, streaming one daily file at a time (never concatenating the
whole record into memory):

1. the hourly rain-rate distribution (percentiles, max, exact-zero fraction);
2. the positive base rate per hazard using the *real* mask functions imported
   from :mod:`nowcast.features.labels`;
3. sensitivity of each base rate to ``LABEL_OCCURRENCE_THRESHOLD`` and to the
   physical mm/h thresholds;
4. monthly and per-cell spread of the positives (clustering = leakage-shaped);
5. the 2-3 May 2018 north-India thunderstorm/dust-storm outbreak as a demo case.

Two upstream defects were found while writing this and are reproduced in the
output rather than silently worked around -- see ``_load_precip``.

Usage::

    .venv\\Scripts\\python.exe scripts/check_label_rates.py --limit 5
    .venv\\Scripts\\python.exe scripts/check_label_rates.py
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nowcast import config  # noqa: E402
from nowcast.common import grid as _grid  # noqa: E402
from nowcast.common import io as _io  # noqa: E402
from nowcast.features import labels as L  # noqa: E402
from nowcast.ingest import names as _names  # noqa: E402
from nowcast.ingest._util import (  # noqa: E402
    apply_var_map,
    resample_to_step,
    select_schema_vars,
)

# Histogram used to recover percentiles without holding every value in memory.
# 0.01 mm/h resolution up to 400 mm/h is fine enough for the 99.99th percentile.
_HIST_HI = 400.0
_HIST_BINS = 40_000
_EDGES = np.linspace(0.0, _HIST_HI, _HIST_BINS + 1)

# Sweep grids.
_OCC_GRID = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
_TS_PRECIP_GRID = (1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0)
_CB_PRECIP_GRID = (10.0, 20.0, 30.0, 50.0, 75.0, 100.0)

_DEMO_DATES = ("2018-05-02", "2018-05-03")


def _load_precip(path: Path) -> xr.Dataset:
    """Mirror :func:`nowcast.ingest.imerg.load_imerg`, with the time axis fixed.

    Two upstream defects matter here and are deliberately *not* hidden:

    * The granules decode to ``cftime.DatetimeJulian`` (object dtype), and
      ``ingest._util.resample_to_step`` returns early for any non-``datetime64``
      axis. So the production loader silently skips the half-hourly -> hourly
      bin-average and hands the datacube a 48-step half-hourly ``precip`` field
      labelled as hourly. We convert the index so the intended hourly mean is
      what gets measured (and report the half-hourly numbers separately).
    * Because the resample is skipped upstream, ``labels.valid_label_times``
      would see 30-minute steps, find nothing contiguous at ``config.TIMESTEP``
      and mark every timestep invalid.
    """
    raw = _io.open_netcdf(path)
    if not np.issubdtype(np.asarray(raw["time"].values).dtype, np.datetime64):
        raw = raw.assign_coords(time=raw.indexes["time"].to_datetimeindex())
    mapped = apply_var_map(raw, _names.IMERG_VARS)
    mapped = select_schema_vars(mapped, ("precip",))
    if "precip" not in mapped:
        raise ValueError(f"no IMERG precipitation variable in {path.name}")
    half_hourly = mapped["precip"].transpose("time", "lat", "lon").values.astype("float64")
    mapped = resample_to_step(mapped, config.TIMESTEP, how="mean")
    mapped = _grid.crop_bbox(mapped)
    mapped = _grid.regrid_to_target(mapped, method="linear")
    mapped["precip"] = mapped["precip"].clip(min=0.0)
    mapped.attrs["_half_hourly_max"] = float(np.nanmax(half_hourly))
    return mapped


def _smoothed_labels(mask: np.ndarray, norm: float) -> np.ndarray:
    """Peak-normalised Gaussian smoothing, exactly as ``labels.build_labels``.

    ``build_labels`` smooths with ``sigma=(0.0, sigma, sigma)`` -- no mixing
    along time -- so smoothing day-by-day is identical to smoothing the whole
    record at once. Leads are pure time shifts of this same field, so the
    positive *rate* is lead-invariant away from the horizon padding.
    """
    sigma = float(config.LABEL_SMOOTH_SIGMA)
    smoothed = gaussian_filter(mask.astype("float32"), sigma=(0.0, sigma, sigma))
    return np.clip(smoothed / norm, 0.0, 1.0)


def _file_dates(data_dir: Path) -> list[tuple[pd.Timestamp, Path]]:
    out: list[tuple[pd.Timestamp, Path]] = []
    for p in sorted(data_dir.glob("imerg_*.nc")):
        if p.suffix != ".nc" or p.name.endswith(".part"):
            continue
        m = re.search(r"(\d{8})", p.name)
        if m:
            out.append((pd.Timestamp(m.group(1)), p))
    out.sort(key=lambda t: t[0])
    return out


def _load_routing() -> dict | None:
    try:
        r = xr.open_zarr(config.DEM_ROUTING_PATH)
    except (OSError, ValueError, KeyError):
        return None
    if "flow_direction" not in r:
        return None
    return {
        "flow_direction": r["flow_direction"],
        "flow_accumulation": r.get("flow_accumulation"),
        "elevation": r.get("elevation"),
    }


def _percentile_from_hist(hist: np.ndarray, total: int, q: float) -> float:
    if total == 0:
        return float("nan")
    target = q / 100.0 * total
    cum = np.cumsum(hist)
    idx = int(np.searchsorted(cum, target, side="left"))
    idx = min(idx, len(_EDGES) - 2)
    return float(_EDGES[idx])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="only the first N files")
    ap.add_argument("--data-dir", type=Path, default=config.RAW_IMERG_DIR)
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    dated = _file_dates(args.data_dir)
    if args.limit is not None:
        dated = dated[: args.limit]
    if not dated:
        print(f"no imerg_*.nc files under {args.data_dir}")
        return 1

    routing = _load_routing()
    norm = L._peak_normaliser(float(config.LABEL_SMOOTH_SIGMA))
    hazards = tuple(config.HAZARDS)

    hist = np.zeros(_HIST_BINS + 1, dtype="int64")
    n_cells_ts = 0
    n_zero = 0
    gmax = 0.0
    gmax_hh = 0.0

    raw_pos = dict.fromkeys(hazards, 0)
    occ_sweep = {h: dict.fromkeys(_OCC_GRID, 0) for h in hazards}
    ts_sweep = dict.fromkeys(_TS_PRECIP_GRID, 0)
    cb_sweep = dict.fromkeys(_CB_PRECIP_GRID, 0)
    monthly: dict[str, dict[int, int]] = {h: defaultdict(int) for h in hazards}
    spatial = {h: np.zeros(config.GRID_SHAPE, dtype="int64") for h in hazards}
    demo: dict[str, dict] = {}

    prev_date: pd.Timestamp | None = None
    prev_tail: xr.DataArray | None = None

    for i, (date, path) in enumerate(dated, 1):
        ds = _load_precip(path)
        precip = ds["precip"].transpose("time", "lat", "lon")
        gmax_hh = max(gmax_hh, float(ds.attrs["_half_hourly_max"]))

        # The cloudburst / flash-flood masks use a 2-hour rolling sum. Chunking
        # by day would reset that window every midnight, so carry the previous
        # day's final hour in when the days are genuinely consecutive.
        carried = 0
        work = precip
        if prev_tail is not None and prev_date is not None and (date - prev_date).days == 1:
            work = xr.concat([prev_tail, precip], dim="time")
            carried = 1
        prev_tail = precip.isel(time=[-1])
        prev_date = date

        ts_mask = L._thunderstorm_mask(work, None)
        cb_mask = L._cloudburst_mask(work)
        ff_mask = L._flash_flood_mask(work, cb_mask, ds, routing)
        masks = {
            "thunderstorm": np.asarray(ts_mask.values, dtype=bool)[carried:],
            "cloudburst": np.asarray(cb_mask.values, dtype=bool)[carried:],
            "flash_flood": np.asarray(ff_mask.values, dtype=bool)[carried:],
        }

        vals = np.asarray(precip.values, dtype="float64").ravel()
        n_cells_ts += vals.size
        n_zero += int((vals == 0.0).sum())
        gmax = max(gmax, float(vals.max()))
        hist[:-1] += np.histogram(np.clip(vals, 0.0, _HIST_HI), bins=_EDGES)[0].astype(
            "int64"
        )
        hist[-1] += int((vals > _HIST_HI).sum())

        work_vals = np.asarray(work.values, dtype="float64")
        accum2 = np.asarray(
            work.rolling(time=2, min_periods=1).sum().values, dtype="float64"
        )[carried:]
        pv = work_vals[carried:]
        for thr in _TS_PRECIP_GRID:
            ts_sweep[thr] += int((pv > thr).sum())
        for thr in _CB_PRECIP_GRID:
            cb_sweep[thr] += int(((pv > thr) | (accum2 > config.CLOUDBURST_ACCUM_MM_2H)).sum())

        month = int(date.month)
        day_demo: dict[str, int] = {}
        for h in hazards:
            m = masks[h]
            raw_pos[h] += int(m.sum())
            lab = _smoothed_labels(m, norm)
            for thr in _OCC_GRID:
                occ_sweep[h][thr] += int((lab >= thr).sum())
            pos_at_cfg = lab >= float(config.LABEL_OCCURRENCE_THRESHOLD)
            monthly[h][month] += int(pos_at_cfg.sum())
            spatial[h] += pos_at_cfg.sum(axis=0).astype("int64")
            day_demo[h] = int(pos_at_cfg.sum())

        if str(date.date()) in _DEMO_DATES:
            demo[str(date.date())] = {
                "max_hourly": float(vals.max()),
                "mean_hourly": float(vals.mean()),
                "wet_frac": float((vals > 0.1).mean()),
                "raw": {h: int(masks[h].sum()) for h in hazards},
                "labelled": day_demo,
            }

        ds.close()
        if i % 25 == 0 or i == len(dated):
            print(f"  ... {i}/{len(dated)} files", flush=True)

    n_hours = n_cells_ts // (config.GRID_SHAPE[0] * config.GRID_SHAPE[1])

    def rate(n: int) -> str:
        if n_cells_ts == 0 or n == 0:
            return "0.000000%  (no positives)"
        pct = 100.0 * n / n_cells_ts
        return f"{pct:.6f}%  (1 in {n_cells_ts / n:,.0f})"

    print("\n" + "=" * 78)
    print("LABEL THRESHOLD AUDIT  --  observed IMERG vs nowcast/config.py")
    print("=" * 78)
    print(f"files={len(dated)}  hours={n_hours:,}  cell-timesteps={n_cells_ts:,}")
    print(f"grid={config.GRID_SHAPE}  routing={'D8 zarr' if routing else 'FALLBACK downhill'}")
    print(f"date range: {dated[0][0].date()} .. {dated[-1][0].date()}")

    print("\n[1] HOURLY RAIN-RATE DISTRIBUTION (mm/h, all cell-timesteps)")
    for q in (50.0, 90.0, 99.0, 99.9, 99.99):
        print(f"    p{q:<6} = {_percentile_from_hist(hist, n_cells_ts, q):8.3f}")
    print(f"    max     = {gmax:8.3f}     (native half-hourly max {gmax_hh:.3f})")
    print(f"    exactly zero: {100.0 * n_zero / n_cells_ts:.3f}%  ({n_zero:,} cells)")

    print("\n[2] BASE RATE PER HAZARD")
    print(f"    at LABEL_OCCURRENCE_THRESHOLD = {config.LABEL_OCCURRENCE_THRESHOLD}")
    for h in hazards:
        print(f"    {h:<14} raw mask   {rate(raw_pos[h])}")
        print(f"    {'':<14} labelled   {rate(occ_sweep[h][0.5])}")

    print("\n[3] LABEL_OCCURRENCE_THRESHOLD SENSITIVITY (% of cell-timesteps)")
    print("    thr   " + "".join(f"{h[:12]:>14}" for h in hazards))
    for thr in _OCC_GRID:
        row = "".join(f"{100.0 * occ_sweep[h][thr] / n_cells_ts:>14.6f}" for h in hazards)
        mark = "  <== config" if abs(thr - config.LABEL_OCCURRENCE_THRESHOLD) < 1e-9 else ""
        print(f"    {thr:<5.2f}{row}{mark}")

    print("\n[4] PHYSICAL mm/h THRESHOLD SENSITIVITY (raw mask, % of cell-timesteps)")
    print("    thunderstorm (precip > X):")
    for thr in _TS_PRECIP_GRID:
        mark = "  <== config" if thr == config.THUNDERSTORM_PRECIP_MM_H else ""
        print(f"      {thr:>6.1f}  {100.0 * ts_sweep[thr] / n_cells_ts:9.6f}%{mark}")
    print(f"    cloudburst (precip > X or accum2h > {config.CLOUDBURST_ACCUM_MM_2H}):")
    for thr in _CB_PRECIP_GRID:
        mark = "  <== config" if thr == config.CLOUDBURST_PRECIP_MM_H else ""
        print(f"      {thr:>6.1f}  {100.0 * cb_sweep[thr] / n_cells_ts:9.6f}%{mark}")

    print("\n[5] SEASONALITY AND SPATIAL SPREAD (labelled positives at config)")
    months = sorted({m for h in hazards for m in monthly[h]})
    print("    month " + "".join(f"{h[:12]:>14}" for h in hazards))
    for m in months:
        print(f"    {m:<6}" + "".join(f"{monthly[h].get(m, 0):>14,}" for h in hazards))
    n_grid = config.GRID_SHAPE[0] * config.GRID_SHAPE[1]
    for h in hazards:
        sp = spatial[h].ravel()
        tot = int(sp.sum())
        if tot == 0:
            print(f"    {h:<14} no positive cells")
            continue
        order = np.sort(sp)[::-1]
        top10 = 100.0 * order[: max(1, n_grid // 10)].sum() / tot
        print(
            f"    {h:<14} cells hit {int((sp > 0).sum())}/{n_grid}"
            f"  top-10% of cells hold {top10:.1f}% of positives"
        )

    print("\n[6] DEMO CASE 2-3 MAY 2018")
    if not demo:
        print("    demo dates not in the processed file set")
    for d in _DEMO_DATES:
        if d not in demo:
            print(f"    {d}: FILE ABSENT")
            continue
        info = demo[d]
        print(
            f"    {d}: max={info['max_hourly']:.2f} mm/h  mean={info['mean_hourly']:.4f}"
            f"  wet(>0.1)={100 * info['wet_frac']:.1f}%"
        )
        for h in hazards:
            print(
                f"        {h:<14} raw={info['raw'][h]:>6}"
                f"   labelled@{config.LABEL_OCCURRENCE_THRESHOLD}={info['labelled'][h]:>6}"
            )
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
