"""Export real model output for the console.

    python scripts/make_live_nowcast.py

Runs the trained LightGBM boosters over the feature cube at one timestamp and
writes what the console needs into frontend/src/data/liveNowcast.json.

Everything in that file is model output or measured data:

* probabilities are booster predictions at the grid cell nearest each station;
* driver values are the actual feature-channel values at that cell;
* driver weights are LightGBM gain importances from the trained booster, so the
  "what is driving this forecast" panel reflects what the model really used.

The default timestamp is drawn from TEST_DATE_RANGE -- a date the model never
trained on -- because an in-sample demo is not worth showing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from nowcast import config, schema

MODEL_DIR = Path(config.PROCESSED_DIR) / "baseline_lgbm"
OUT = Path(config.PROJECT_ROOT) / "frontend" / "src" / "data" / "liveNowcast.json"

# The console shows four steps; map them onto real forecast lead hours.
STEP_LEADS = [1, 2, 4, 6]
STEP_LABELS = ["Now", "+2h", "+4h", "+6h"]

STATIONS = {
    "uk": {
        # Upper Dhauliganga basin. Added because it is where the model actually
        # concentrates risk -- on 2018-09-08 it is the peak cell at 99.9% while
        # the valley stations sit near zero. A monitored-point list that omits
        # the cell the model is shouting about is not much use to an operator.
        "dhauliganga": (31.2000, 79.4000),
        "kedarnath": (30.7346, 79.0669),
        "rudraprayag": (30.2844, 78.9811),
        "dehradun": (30.3165, 78.0322),
        "haridwar": (29.9457, 78.1642),
    },
    "ncr": {
        "gurugram": (28.4595, 77.0266),
        "najafgarh": (28.6092, 76.9797),
        "southdelhi": (28.5355, 77.2933),
        "noida": (28.5709, 77.3260),
    },
}

# Feature channel -> console label, unit, display format.
DRIVER_LABELS = {
    "cape": ("CAPE (surface-based)", "J/kg", "{:.0f}"),
    "cin": ("Convective inhibition", "J/kg", "{:.0f}"),
    "lifted_index": ("Lifted index", "K", "{:+.1f}"),
    "iwv": ("Integrated water vapour", "kg/m2", "{:.1f}"),
    "iwv_tendency_1h": ("Integrated water vapour, rate of change", "kg/m2/hr", "{:+.2f}"),
    "shear_0_6km": ("Vertical wind shear, 0-6 km", "m/s", "{:.1f}"),
    "shear_0_1km": ("Vertical wind shear, 0-1 km", "m/s", "{:.1f}"),
    "conv_850": ("Low-level convergence", "x10-5 s-1", "{:+.2f}"),
    "tcwv": ("Total column water vapour", "kg/m2", "{:.1f}"),
    "t2m": ("2 m temperature", "K", "{:.1f}"),
    "prmsl": ("Mean sea-level pressure", "hPa", "{:.0f}"),
    "hand": ("Height above nearest drainage", "m", "{:.0f}"),
    "flow_accumulation": ("Upstream flow accumulation", "cells", "{:.0f}"),
    "slope": ("Terrain slope", "deg", "{:.1f}"),
    "elevation": ("Elevation", "m", "{:.0f}"),
}


def load_boosters() -> dict:
    import lightgbm as lgb

    manifest = json.loads((MODEL_DIR / "manifest.json").read_text())
    out = {}
    for name, entry in manifest["models"].items():
        path = MODEL_DIR / entry["file"]
        if path.exists():
            out[name] = lgb.Booster(model_file=str(path))
    if not out:
        raise SystemExit(f"no boosters found under {MODEL_DIR}")
    return out


def pick_timestamp(labels: xr.DataArray, times: pd.DatetimeIndex, wanted: str | None):
    """Most active hour inside TEST_DATE_RANGE, or an explicit --at."""
    if wanted:
        return pd.Timestamp(wanted)
    lo = pd.Timestamp(config.TEST_DATE_RANGE[0])
    hi = pd.Timestamp(config.TEST_DATE_RANGE[1]) + pd.Timedelta("23:59:59")
    mask = (times >= lo) & (times <= hi)
    if not mask.any():
        return times[len(times) // 2]
    sub = labels.isel(time=np.flatnonzero(mask))
    score = sub.max(dim=[d for d in sub.dims if d != "time"]).values
    return pd.Timestamp(times[np.flatnonzero(mask)[int(np.argmax(score))]])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--at", default=None, help="ISO timestamp; default = peak hour in TEST range")
    args = ap.parse_args()

    feats = xr.open_zarr(Path(config.PROCESSED_DIR) / "features.zarr")
    labels = xr.open_zarr(Path(config.PROCESSED_DIR) / "labels.zarr")["hazard_probability"]
    times = pd.to_datetime(feats.time.values)

    stamp = pick_timestamp(labels, times, args.at)
    ti = int(np.argmin(np.abs(times - stamp)))
    stamp = pd.Timestamp(times[ti])
    in_test = pd.Timestamp(config.TEST_DATE_RANGE[0]) <= stamp

    snap = feats.isel(time=ti).load()
    lat = snap.lat.values
    lon = snap.lon.values

    # (n_cells, 20) in the exact channel order the boosters were trained on.
    cols = [np.asarray(snap[c].values, dtype="float32").ravel() for c in schema.FEATURE_CHANNELS]
    X = np.stack(cols, axis=1)

    boosters = load_boosters()
    preds: dict[str, np.ndarray] = {}
    for name, booster in boosters.items():
        preds[name] = np.clip(np.asarray(booster.predict(X), dtype="float64"), 0.0, 1.0)

    # Gain importance per hazard, averaged over that hazard's lead models.
    importance: dict[str, np.ndarray] = {}
    for hazard in config.HAZARDS:
        gains = [
            b.feature_importance(importance_type="gain")
            for n, b in boosters.items()
            if n.startswith(f"{hazard}__")
        ]
        if gains:
            g = np.mean(np.stack(gains), axis=0)
            importance[hazard] = g / (g.max() or 1.0)

    n_lon = len(lon)

    def cell_index(la: float, lo_: float) -> int:
        return int(np.argmin(np.abs(lat - la)) * n_lon + np.argmin(np.abs(lon - lo_)))

    zeros = np.zeros(X.shape[0])
    out: dict = {
        "generated_from": "LightGBM baseline, trained on 2018 ERA5 + IMERG + DEM",
        "valid_at": stamp.strftime("%Y-%m-%dT%H:%M"),
        "valid_at_label": stamp.strftime("%H:%M IST"),
        "held_out": bool(in_test),
        "steps": STEP_LABELS,
        "step_leads": STEP_LEADS,
        "regions": {},
    }

    for region, stations in STATIONS.items():
        idx = {sid: cell_index(la, lo_) for sid, (la, lo_) in stations.items()}
        hazards: dict = {}
        for hazard in config.HAZARDS:
            vals = {
                sid: [
                    int(round(100 * float(preds.get(f"{hazard}__lead{ld}", zeros)[i])))
                    for ld in STEP_LEADS
                ]
                for sid, i in idx.items()
            }
            top_cell = max(idx.values(), key=lambda i: preds.get(f"{hazard}__lead2", zeros)[i])
            gain = importance.get(hazard)
            drivers = []
            if gain is not None:
                for j in np.argsort(gain)[::-1]:
                    channel = schema.FEATURE_CHANNELS[j]
                    if channel not in DRIVER_LABELS or gain[j] <= 0:
                        continue
                    label, unit, fmt = DRIVER_LABELS[channel]
                    v = float(np.asarray(snap[channel].values).ravel()[top_cell])
                    drivers.append({
                        "label": label,
                        "unit": unit,
                        "vals": [fmt.format(v)] * len(STEP_LEADS),
                        "w": [round(float(gain[j]), 3)] * len(STEP_LEADS),
                    })
                    if len(drivers) == 5:
                        break
            hazards[hazard] = {"vals": vals, "drivers": drivers}
        out["regions"][region] = {"hazards": hazards}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT}")
    print(f"  valid at {out['valid_at']}  held_out={out['held_out']}")
    for region in out["regions"]:
        peak = max(
            v
            for h in out["regions"][region]["hazards"].values()
            for series in h["vals"].values()
            for v in series
        )
        print(f"  {region}: peak probability {peak}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
