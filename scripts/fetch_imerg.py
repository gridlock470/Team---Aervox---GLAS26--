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
import threading
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, timedelta

import numpy as np
import xarray as xr

from nowcast import config
from nowcast.common import grid

SHORT_NAME = "GPM_3IMERGHH"
VERSION = "07"
EXPECTED_PER_DAY = 48  # half-hourly
PAD = 0.5  # deg of margin -- must match nowcast.common.grid.crop_bbox default
# PROCESSES, not threads: earthaccess reads granules through fsspec, which
# shares ONE asyncio event loop across all threads in a process. Eight threads
# saturate it and every range-read dies with FSTimeoutError. Separate processes
# each get their own loop and their own HTTP session.
DEFAULT_WORKERS = 4

_PRINT_LOCK = threading.Lock()


def _log(msg: str) -> None:
    with _PRINT_LOCK:
        print(msg, flush=True)


def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def subset_granule(fh) -> xr.Dataset:
    """Open one IMERG granule and crop to the pilot bbox plus margin.

    IMERG cell centres are registered on the half-tenth (27.85, 27.95, ...)
    while ``config.GRID_LAT`` sits on the tenth (27.8, 27.9, ...), so the two
    grids are offset by half a cell. ``crop_bbox`` keeps ``PAD`` degrees of
    margin so the later ``regrid_to_target`` interpolation onto the exact
    contract grid never reaches past the edge of the stored data.
    """
    ds = xr.open_dataset(fh, group="Grid", engine="h5netcdf", decode_times=True)
    return grid.crop_bbox(ds[["precipitation"]], pad=PAD).load()


def fetch_day(d: date, out_dir, bbox, total: int, idx: int) -> str:
    """Fetch one day. Returns "done" | "short" | "skip" | "fail".

    Runs in its own process, so it must establish its own earthaccess
    session rather than inheriting one from the parent.
    """
    dest = out_dir / f"imerg_{d:%Y%m%d}.nc"
    if dest.exists() and dest.stat().st_size > 0:
        return "skip"

    tmp = dest.with_suffix(".nc.part")
    try:
        import earthaccess

        # Inside the try on purpose: a transient DNS failure resolving the
        # Earthdata host used to raise out of the worker, propagate through
        # future.result() and kill the entire run over one network blip.
        earthaccess.login(strategy="netrc")

        results = earthaccess.search_data(
            short_name=SHORT_NAME,
            version=VERSION,
            temporal=(f"{d:%Y-%m-%d}", f"{d:%Y-%m-%d}"),
            bounding_box=bbox,
        )
        if not results:
            _log(f"[{idx}/{total}] {d} NO GRANULES")
            return "fail"

        files = earthaccess.open(results)
        parts = [subset_granule(f) for f in files]
        day_ds = xr.concat(parts, dim="time").sortby("time")

        status = "done"
        n = day_ds.sizes.get("time", 0)
        if n != EXPECTED_PER_DAY:
            # Report rather than silently accepting a gappy day.
            _log(f"[{idx}/{total}] {d} WARNING {n}/{EXPECTED_PER_DAY} timesteps")
            status = "short"

        day_ds["precipitation"] = day_ds["precipitation"].astype(np.float32)
        day_ds.attrs.update(
            source=f"{SHORT_NAME} v{VERSION}",
            bbox_west=config.BBOX_WEST, bbox_east=config.BBOX_EAST,
            bbox_south=config.BBOX_SOUTH, bbox_north=config.BBOX_NORTH,
        )
        day_ds.to_netcdf(tmp, engine="h5netcdf")
        day_ds.close()
        tmp.replace(dest)
        return status
    except Exception as e:  # noqa: BLE001 - keep going, report at end
        tmp.unlink(missing_ok=True)
        _log(f"[{idx}/{total}] {d} FAILED {type(e).__name__}: {e}")
        traceback.print_exc(limit=2)
        return "fail"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2020-12-31")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = ap.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    import earthaccess

    earthaccess.login(strategy="netrc")

    out_dir = config.RAW_IMERG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    bbox = (config.BBOX_WEST, config.BBOX_SOUTH, config.BBOX_EAST, config.BBOX_NORTH)

    days = list(daterange(start, end))
    total = len(days)
    _log(f"{total} days, {args.workers} workers, bbox={bbox}, out={out_dir}")

    counts = {"done": 0, "short": 0, "skip": 0, "fail": 0}
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(fetch_day, d, out_dir, bbox, total, i): d
            for i, d in enumerate(days, 1)
        }
        try:
            for fut in as_completed(futures):
                try:
                    counts[fut.result()] += 1
                except Exception as e:  # noqa: BLE001 - one bad day must not end the run
                    _log(f"worker for {futures[fut]} died: {type(e).__name__}: {e}")
                    counts["fail"] += 1
                completed += 1
                if completed % 25 == 0 or completed == total:
                    mb = sum(f.stat().st_size for f in out_dir.glob("*.nc")) / 1e6
                    _log(f"progress {completed}/{total}  ok={counts['done']} "
                         f"short={counts['short']} skip={counts['skip']} "
                         f"fail={counts['fail']}  ({mb:.1f} MB)")
        except KeyboardInterrupt:
            _log("interrupted -- rerun to resume")
            pool.shutdown(cancel_futures=True)
            return 130

    mb = sum(f.stat().st_size for f in out_dir.glob("*.nc")) / 1e6
    _log(f"done: {counts['done']} fetched, {counts['skip']} skipped, "
         f"{counts['short']} short, {counts['fail']} failed, {mb:.1f} MB")
    return 1 if counts["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
