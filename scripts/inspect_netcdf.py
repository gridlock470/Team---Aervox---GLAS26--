"""Report real, honest statistics for an arbitrary NetCDF file.

    python scripts/inspect_netcdf.py <path-to-file.nc>

Backs the Insights tab's NetCDF upload: prints ONE JSON object to stdout and
always exits 0 (a file that can't be read is reported via the "error" field,
never via a crash or a substitute number standing in for real data).

Deliberately does NOT reuse nowcast.ingest.imdaa/imerg's load_* functions --
both crop to this project's own bounding box and regrid to its own target
grid before returning anything, which is correct for the training pipeline
but would silently distort or discard data outside that region for a feature
whose whole point is "tell me what's actually in this file". Only the
generic primitive underneath them, nowcast.common.io.open_netcdf (a plain
multi-engine xarray.open_dataset wrapper, no cropping/regridding), is reused.
So this script works on any valid NetCDF4 file, not just this project's own
ERA5/IMERG layout.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from nowcast.common.io import open_netcdf  # noqa: E402

HIST_BINS = 10


def _label_for(var) -> str:
    return str(var.attrs.get("long_name") or var.attrs.get("standard_name") or var.name)


def _trend_for(var, ds) -> dict | None:
    if "time" not in var.dims:
        return None
    other_dims = [d for d in var.dims if d != "time"]
    series = var.mean(dim=other_dims, skipna=True) if other_dims else var
    values = np.asarray(series.values, dtype="float64")
    time_vals = ds["time"].values
    labels = [str(np.datetime_as_string(t, unit="m")) for t in time_vals]
    return {
        "labels": labels,
        "values": [None if not np.isfinite(v) else float(v) for v in values],
    }


def inspect(path: str) -> dict:
    try:
        ds = open_netcdf(path)
    except Exception as exc:  # noqa: BLE001 - any open failure is reported, not raised
        return {"error": f"Couldn't open this file as NetCDF: {exc}", "dims": {}, "variables": []}

    dims = {str(name): int(size) for name, size in ds.sizes.items()}
    variables = []

    for name, var in ds.data_vars.items():
        values = np.asarray(var.values, dtype="float64").ravel()
        finite = values[np.isfinite(values)]
        missing = int(values.size - finite.size)

        if finite.size == 0:
            variables.append({
                "name": str(name), "longName": _label_for(var),
                "units": str(var.attrs.get("units") or ""),
                "shape": list(var.shape), "count": 0, "missing": missing,
                "mean": None, "min": None, "max": None, "std": None,
                "histogram": None, "trend": None,
            })
            continue

        counts, edges = np.histogram(finite, bins=HIST_BINS)
        variables.append({
            "name": str(name),
            "longName": _label_for(var),
            "units": str(var.attrs.get("units") or ""),
            "shape": list(var.shape),
            "count": int(finite.size),
            "missing": missing,
            "mean": float(np.mean(finite)),
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
            "std": float(np.std(finite)),
            "histogram": {"edges": [float(e) for e in edges], "counts": [int(c) for c in counts]},
            "trend": _trend_for(var, ds),
        })

    return {"error": None, "dims": dims, "variables": variables}


def main() -> None:
    if len(sys.argv) != 2:
        print(json.dumps({"error": "No file path given.", "dims": {}, "variables": []}))
        return
    result = inspect(sys.argv[1])
    print(json.dumps(result))


if __name__ == "__main__":
    main()
