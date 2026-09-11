"""Fetch Copernicus GLO-30 DEM tiles covering the pilot bbox.

Public AWS Open Data bucket — no credentials required. Tiles are 1x1 degree
COGs named by the SOUTH-WEST corner, so the bbox is floored/ceiled to whole
degrees to guarantee full coverage.
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.request
from math import ceil, floor

from nowcast import config

BASE = "https://copernicus-dem-30m.s3.amazonaws.com"


def tile_name(lat: int, lon: int) -> str:
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM"


def main() -> int:
    out_dir = config.RAW_DEM_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    lat0, lat1 = floor(config.BBOX_SOUTH), ceil(config.BBOX_NORTH)
    lon0, lon1 = floor(config.BBOX_WEST), ceil(config.BBOX_EAST)

    tiles = [
        (lat, lon)
        for lat in range(lat0, lat1)
        for lon in range(lon0, lon1)
    ]
    print(f"bbox S{lat0} N{lat1} W{lon0} E{lon1} -> {len(tiles)} tiles", flush=True)

    ok = skipped = failed = 0
    for i, (lat, lon) in enumerate(tiles, 1):
        name = tile_name(lat, lon)
        dest = out_dir / f"{name}.tif"
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[{i}/{len(tiles)}] {name} already present, skip", flush=True)
            skipped += 1
            continue
        url = f"{BASE}/{name}/{name}.tif"
        tmp = dest.with_suffix(".tif.part")
        try:
            with urllib.request.urlopen(url, timeout=120) as r, tmp.open("wb") as fh:
                while chunk := r.read(1 << 20):
                    fh.write(chunk)
            tmp.replace(dest)
            mb = dest.stat().st_size / 1e6
            print(f"[{i}/{len(tiles)}] {name} OK {mb:.1f} MB", flush=True)
            ok += 1
        except urllib.error.HTTPError as e:
            tmp.unlink(missing_ok=True)
            # Ocean-only tiles genuinely do not exist in GLO-30.
            level = "absent (ocean?)" if e.code == 404 else f"HTTP {e.code}"
            print(f"[{i}/{len(tiles)}] {name} {level}", flush=True)
            failed += e.code != 404
        except Exception as e:  # noqa: BLE001 - report and continue
            tmp.unlink(missing_ok=True)
            print(f"[{i}/{len(tiles)}] {name} FAILED {type(e).__name__}: {e}", flush=True)
            failed += 1

    total_mb = sum(f.stat().st_size for f in out_dir.glob("*.tif")) / 1e6
    print(f"\ndone: {ok} fetched, {skipped} skipped, {failed} failed, {total_mb:.1f} MB on disk")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
