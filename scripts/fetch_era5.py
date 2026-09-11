r"""Fetch ERA5 reanalysis (single + pressure levels) for the pilot region.

ERA5 stands in for IMDAA: same variables, 0.25 deg instead of 0.12 deg, but
self-service via the Copernicus CDS API with no order queue.

One request per (dataset, year, month) so the job is resumable and no single
request is large. Requires ``%USERPROFILE%\.cdsapirc``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nowcast import config

SINGLE_LEVEL_VARS = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "total_column_water_vapour",
    "convective_available_potential_energy",
    "convective_inhibition",
    "total_precipitation",
]

PRESSURE_LEVEL_VARS = [
    "temperature",
    "relative_humidity",
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
]

PAD = 0.5  # deg, matches nowcast.common.grid.crop_bbox

# CDS wants [North, West, South, East]; widen to the padded box so the later
# regrid onto config.GRID_LAT/GRID_LON never reaches past the data edge.
AREA = [
    config.BBOX_NORTH + PAD,
    config.BBOX_WEST - PAD,
    config.BBOX_SOUTH - PAD,
    config.BBOX_EAST + PAD,
]

ALL_DAYS = [f"{d:02d}" for d in range(1, 32)]
ALL_TIMES = [f"{h:02d}:00" for h in range(24)]


def build_request(dataset: str, year: int, month: int, days, times) -> dict:
    req = {
        "product_type": ["reanalysis"],
        "year": [str(year)],
        "month": [f"{month:02d}"],
        "day": list(days),
        "time": list(times),
        "area": AREA,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    if dataset == "reanalysis-era5-single-levels":
        req["variable"] = SINGLE_LEVEL_VARS
    else:
        req["variable"] = PRESSURE_LEVEL_VARS
        req["pressure_level"] = [str(p) for p in config.PRESSURE_LEVELS_HPA]
    return req


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2018,2019")
    ap.add_argument("--months", default="4,5,6,7,8,9")
    ap.add_argument("--test", action="store_true",
                    help="single day, single hour -- validates licence + format")
    args = ap.parse_args()

    import cdsapi

    client = cdsapi.Client()
    out_dir = Path(config.RAW_DIR) / "era5"
    out_dir.mkdir(parents=True, exist_ok=True)

    years = [int(y) for y in args.years.split(",")]
    months = [int(m) for m in args.months.split(",")]
    days, times = ALL_DAYS, ALL_TIMES
    tag = ""
    if args.test:
        years, months = [2018], [5]
        days, times = ["02"], ["06:00"]
        tag = "_test"

    datasets = {
        "reanalysis-era5-single-levels": "sl",
        "reanalysis-era5-pressure-levels": "pl",
    }

    jobs = [(ds, short, y, m) for ds, short in datasets.items() for y in years for m in months]
    print(f"{len(jobs)} requests, area(N,W,S,E)={AREA}, out={out_dir}", flush=True)

    done = failed = 0
    for i, (ds, short, y, m) in enumerate(jobs, 1):
        dest = out_dir / f"era5_{short}_{y}{m:02d}{tag}.nc"
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[{i}/{len(jobs)}] {dest.name} present, skip", flush=True)
            done += 1
            continue
        tmp = dest.with_suffix(".nc.part")
        try:
            print(f"[{i}/{len(jobs)}] requesting {dest.name} ...", flush=True)
            client.retrieve(ds, build_request(ds, y, m, days, times), str(tmp))
            tmp.replace(dest)
            print(f"[{i}/{len(jobs)}] {dest.name} OK {dest.stat().st_size/1e6:.1f} MB", flush=True)
            done += 1
        except KeyboardInterrupt:
            print("interrupted -- rerun to resume", flush=True)
            return 130
        except Exception as e:  # noqa: BLE001 - report and continue
            tmp.unlink(missing_ok=True)
            print(f"[{i}/{len(jobs)}] {dest.name} FAILED {type(e).__name__}: {e}", flush=True)
            failed += 1

    mb = sum(f.stat().st_size for f in out_dir.glob("*.nc")) / 1e6
    print(f"\ndone: {done} ok, {failed} failed, {mb:.1f} MB")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
