r"""Fetch GPM IMERG Final Run V07 half-hourly rainfall, subset to the pilot bbox.

Downloads one NetCDF per day into ``config.RAW_IMERG_DIR``. Only the bbox
window of ``precipitation`` is written, which is ~12 KB/timestep instead of
the ~8 MB global granule.

Resumable: a day whose output file already exists is skipped, so an
interrupted run continues where it stopped.

Requires Earthdata credentials in ``%USERPROFILE%\_netrc`` -- create them with
``python -c "import earthaccess; earthaccess.login(persist=True)"``.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from datetime import date, timedelta

import numpy as np
import xarray as xr

from nowcast import config

SHORT_NAME = "GPM_3IMERGHH"
VERSION = "07"
EXPECTED_PER_DAY = 48  # half-hourly


def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def subset_granule(fh) -> xr.Dataset:
    """Open one IMERG granule and crop to the pilot bbox."""
    ds = xr.open_dataset(fh, group="Grid", engine="h5netcdf", decode_times=True)
    # IMERG is stored lon-major; select by label works for both orderings.
    sub = ds[["precipitation"]].sel(
        lat=slice(config.BBOX_SOUTH, config.BBOX_NORTH),
        lon=slice(config.BBOX_WEST, config.BBOX_EAST),
    )
    return sub.load()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2020-12-31")
    args = ap.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    import earthaccess

    earthaccess.login(strategy="netrc")

    out_dir = config.RAW_IMERG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    bbox = (config.BBOX_WEST, config.BBOX_SOUTH, config.BBOX_EAST, config.BBOX_NORTH)

    days = list(daterange(start, end))
    print(f"{len(days)} days, bbox={bbox}, out={out_dir}", flush=True)

    done = short = failed = 0
    for i, d in enumerate(days, 1):
        dest = out_dir / f"imerg_{d:%Y%m%d}.nc"
        if dest.exists() and dest.stat().st_size > 0:
            done += 1
            continue
        try:
            results = earthaccess.search_data(
                short_name=SHORT_NAME,
                version=VERSION,
                temporal=(f"{d:%Y-%m-%d}", f"{d:%Y-%m-%d}"),
                bounding_box=bbox,
            )
            if not results:
                print(f"[{i}/{len(days)}] {d} NO GRANULES", flush=True)
                failed += 1
                continue

            files = earthaccess.open(results)
            parts = [subset_granule(f) for f in files]
            day_ds = xr.concat(parts, dim="time").sortby("time")

            n = day_ds.sizes.get("time", 0)
            if n != EXPECTED_PER_DAY:
                # Report rather than silently accepting a gappy day.
                print(f"[{i}/{len(days)}] {d} WARNING {n}/{EXPECTED_PER_DAY} timesteps", flush=True)
                short += 1

            day_ds["precipitation"] = day_ds["precipitation"].astype(np.float32)
            day_ds.attrs.update(
                source=f"{SHORT_NAME} v{VERSION}",
                bbox_west=config.BBOX_WEST, bbox_east=config.BBOX_EAST,
                bbox_south=config.BBOX_SOUTH, bbox_north=config.BBOX_NORTH,
            )
            tmp = dest.with_suffix(".nc.part")
            day_ds.to_netcdf(tmp, engine="h5netcdf")
            tmp.replace(dest)
            day_ds.close()

            if i % 25 == 0 or i == 1:
                mb = sum(f.stat().st_size for f in out_dir.glob("*.nc")) / 1e6
                print(f"[{i}/{len(days)}] {d} OK  ({mb:.1f} MB on disk)", flush=True)
            done += 1
        except KeyboardInterrupt:
            print("interrupted -- rerun to resume", flush=True)
            return 130
        except Exception as e:  # noqa: BLE001 - keep going, report at end
            print(f"[{i}/{len(days)}] {d} FAILED {type(e).__name__}: {e}", flush=True)
            traceback.print_exc(limit=2)
            failed += 1

    mb = sum(f.stat().st_size for f in out_dir.glob("*.nc")) / 1e6
    print(f"\ndone: {done} days, {short} short days, {failed} failed, {mb:.1f} MB")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
