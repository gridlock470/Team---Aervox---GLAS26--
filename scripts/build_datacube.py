"""Build the real datacube from the on-disk ERA5 / IMERG / DEM archives.

ERA5 stands in for IMDAA until the NCMRWF order lands: identical variable roles
at 0.25 deg instead of 0.12 deg, so it feeds the ``imdaa_single`` /
``imdaa_level`` slots of :func:`nowcast.pipelines.ingest_flow.run_ingest`.

Layout this script expects (all already standardised by the download step):

* ``DataSet/raw/era5/era5_sl_YYYYMM.nc``  -- hourly single levels
  (``t2m u10 v10 msl tcwv cape cin tp``);
* ``DataSet/raw/era5/era5_pl_YYYYMM.nc``  -- 3-hourly pressure levels
  (``t r z u v`` on the seven ``config.PRESSURE_LEVELS_HPA``);
* ``DataSet/raw/imerg/imerg_YYYYMMDD.nc`` -- 48 half-hourly ``precipitation``
  steps per day, ``mm h-1``;
* ``DataSet/processed/dem_routing.zarr``  -- terrain routing already computed on
  the contract grid (do not recompute it here).

Only the window every source actually covers may be built: IMERG stops well
before ERA5 does, and ``merge_precip`` fills unmatched cells with ``0.0``, so
running past the precip record would write fabricated "no rain" into the only
label source. Hence ``--start`` / ``--end``, and hence the coverage report the
run prints at the end.

Typical use::

    .venv/Scripts/python.exe scripts/build_datacube.py

which builds 2018-04-01 .. 2018-09-30 into ``config.DATACUBE_PATH``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nowcast import config, schema  # noqa: E402
from nowcast.features.labels import valid_label_times  # noqa: E402
from nowcast.ingest.datacube import open_datacube  # noqa: E402
from nowcast.pipelines.ingest_flow import run_ingest  # noqa: E402

_LOG = logging.getLogger("build_datacube")

# IMERG covers 2018-04-01 .. 2018-09-30 completely (183/183 days verified by
# date, not by file count); 2019 was never fetched. ERA5 runs 2018-04..09 and
# 2019-04..09, so 2018-04-01 .. 2018-09-30 is the window every source shares,
# and it spans all three of config's TRAIN / VAL / TEST date ranges.
DEFAULT_START = "2018-04-01"
DEFAULT_END = "2018-09-30"

# Channels that no acquired source can fill. INSAT was never obtained, so
# features/assemble.py constant-fills these -- expected, not a failure.
MISSING_SATELLITE_CHANNELS = ("ctt", "wv_bt", "ctt_drop_rate_1h")


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------
def _month_stamps(start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    """Every ``YYYYMM`` stamp touched by the inclusive ``[start, end]`` window."""
    months = pd.period_range(start.to_period("M"), end.to_period("M"), freq="M")
    return [p.strftime("%Y%m") for p in months]


def _era5_paths(era5_dir: Path, kind: str, start: pd.Timestamp, end: pd.Timestamp) -> list[Path]:
    """ERA5 ``sl`` or ``pl`` monthly files overlapping the window, in time order."""
    found, missing = [], []
    for stamp in _month_stamps(start, end):
        path = era5_dir / f"era5_{kind}_{stamp}.nc"
        (found if path.is_file() else missing).append(path if path.is_file() else stamp)
    if missing:
        raise FileNotFoundError(
            f"missing ERA5 {kind} months {missing} under {era5_dir}; "
            "narrow --start/--end or download them first"
        )
    return found


def _imerg_paths(imerg_dir: Path, start: pd.Timestamp, end: pd.Timestamp) -> list[Path]:
    """IMERG daily files inside the window, in time order.

    The day BEFORE the window is included when it exists: ``load_imerg`` labels
    each hourly bin by the END of its accumulation window, so the first hour of
    the window is built from granules that live in the previous day's file.

    Missing days are reported, not invented: a gap simply leaves those hours
    NaN after the tolerance-bounded reindex in ``_stage_precip``.
    """
    found, missing = [], []
    lead_in = imerg_dir / f"imerg_{(start.normalize() - pd.Timedelta(days=1)):%Y%m%d}.nc"
    if lead_in.is_file():
        found.append(lead_in)
    for day in pd.date_range(start.normalize(), end.normalize(), freq="D"):
        path = imerg_dir / f"imerg_{day:%Y%m%d}.nc"
        if path.is_file():
            found.append(path)
        else:
            missing.append(f"{day:%Y-%m-%d}")
    if not found:
        raise FileNotFoundError(f"no IMERG daily files in {start:%F}..{end:%F} under {imerg_dir}")
    if missing:
        _LOG.warning(
            "%d IMERG day(s) absent, those hours stay NaN: %s%s",
            len(missing),
            ", ".join(missing[:10]),
            " ..." if len(missing) > 10 else "",
        )
    return found


def _load_static(path: Path) -> xr.Dataset:
    """Load the pre-computed terrain routing onto the contract grid.

    ``dem_routing.zarr`` is already on ``config.GRID_LAT`` / ``GRID_LON``; only
    the CRS marker coordinate is dropped, since it is not a schema variable and
    would otherwise ride into the datacube.
    """
    static = xr.open_zarr(path, consolidated=True)
    wanted = [s.name for s in schema.STATIC_VARS if s.name in static.data_vars]
    absent = [s.name for s in schema.STATIC_VARS if s.name not in static.data_vars]
    if absent:
        raise KeyError(f"{path} is missing terrain variables {absent}")
    if "streams" in static.data_vars:
        wanted.append("streams")
    static = static[wanted]
    extra_coords = [c for c in static.coords if c not in ("lat", "lon")]
    if extra_coords:
        static = static.drop_vars(extra_coords)
    return static.load()


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def _dir_size_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _fmt_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024.0 or unit == "GiB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} GiB"


def _report_time(ds: xr.Dataset) -> None:
    times = pd.DatetimeIndex(ds["time"].values)
    print("\n== time coverage ==")
    print(f"  first          : {times[0]}")
    print(f"  last           : {times[-1]}")
    print(f"  steps          : {len(times)}")
    print(f"  first 5 stamps : {[str(t) for t in times[:5]]}")
    if len(times) < 2:
        return
    deltas = pd.Series(np.diff(times.values))
    modal = deltas.mode()
    print(f"  modal delta    : {modal.iloc[0] if not modal.empty else 'n/a'}")
    step = pd.Timedelta(config.TIMESTEP)
    gaps = deltas[deltas != step]
    print(f"  expected step  : {step}")
    if gaps.empty:
        print("  gaps           : none (strictly hourly, no missing steps)")
    else:
        print(f"  gaps           : {len(gaps)} irregular interval(s)")
        for idx, gap in list(gaps.items())[:10]:
            print(f"      {times[idx]} -> {times[idx + 1]}  ({gap})")


def _report_channels(ds: xr.Dataset) -> None:
    print("\n== schema.FEATURE_CHANNELS ==")
    raw_present, derived, filled = [], [], []
    for channel in schema.FEATURE_CHANNELS:
        if channel in ds.data_vars:
            raw_present.append(channel)
        elif channel in MISSING_SATELLITE_CHANNELS:
            filled.append(channel)
        else:
            derived.append(channel)
    print(f"  in the cube now ({len(raw_present)}): {', '.join(raw_present)}")
    print(f"  derived later by features/assemble ({len(derived)}): {', '.join(derived)}")
    print(f"  CONSTANT-FILLED, no INSAT ({len(filled)}): {', '.join(filled)}")

    missing_required = [
        spec.name
        for spec in schema.SURFACE_VARS + schema.LEVEL_VARS + schema.STATIC_VARS
        if spec.name not in ds.data_vars
    ]
    print(
        "  required datacube variables: "
        + ("ALL PRESENT" if not missing_required else f"MISSING {missing_required}")
    )


def _report_nans(ds: xr.Dataset) -> None:
    print("\n== NaN count per variable ==")
    for name in sorted(ds.data_vars):
        values = ds[name].values
        if not np.issubdtype(values.dtype, np.floating):
            print(f"  {name:<18} (non-float, skipped)")
            continue
        n_nan = int(np.isnan(values).sum())
        total = values.size
        pct = 100.0 * n_nan / total if total else 0.0
        print(f"  {name:<18} {n_nan:>12,} / {total:>12,}  ({pct:6.3f} %)")


def _report_physics(ds: xr.Dataset) -> None:
    expected = {
        "t2m": "K, expect ~270-320",
        "tcwv": "kg m-2, expect ~5-70",
        "cape": "J kg-1, expect 0-6000",
        "precip": "mm h-1, expect 0-~100",
        "prmsl": "Pa, expect ~95000-103000",
        "z": "m2 s-2, expect ~1e3-1.1e5",
    }
    print("\n== physical sanity ==")
    for name, note in expected.items():
        if name not in ds.data_vars:
            print(f"  {name:<8} ABSENT")
            continue
        values = ds[name].values
        print(
            f"  {name:<8} min={np.nanmin(values):12.3f}  "
            f"mean={np.nanmean(values):12.3f}  max={np.nanmax(values):12.3f}   ({note})"
        )


def _report_labels(ds: xr.Dataset) -> None:
    """Confirm the clock actually supports a full label horizon."""
    mask = valid_label_times(ds["time"].values)
    n_valid = int(mask.values.sum())
    n_total = int(mask.size)
    print("\n== trainability ==")
    print(
        f"  valid_label_times : {n_valid:,} / {n_total:,} steps carry a full "
        f"{max(config.LEAD_TIMES_H)} h label horizon"
    )
    if n_valid == 0:
        print("  *** ZERO valid steps -- the cube cannot train anything ***")

    times = pd.DatetimeIndex(ds["time"].values)
    for split, (lo, hi) in (
        ("TRAIN", config.TRAIN_DATE_RANGE),
        ("VAL", config.VAL_DATE_RANGE),
        ("TEST", config.TEST_DATE_RANGE),
    ):
        inside = (times >= pd.Timestamp(lo)) & (
            times <= pd.Timestamp(hi) + pd.Timedelta(hours=23, minutes=59)
        )
        n_in = int(inside.sum())
        flag = "" if n_in else "   *** EMPTY ***"
        print(f"  {split:<5} {lo} .. {hi} : {n_in:,} steps in the cube{flag}")


def verify(store: Path) -> xr.Dataset:
    """Open the written store and print the full verification report."""
    ds = open_datacube(store).load()
    print("\n" + "=" * 72)
    print(f"DATACUBE VERIFICATION  {store}")
    print("=" * 72)
    print("\n== dims ==")
    print(f"  {dict(ds.sizes)}")
    print(f"  expected grid (lat, lon) = {config.GRID_SHAPE}")

    lat_ok = np.allclose(ds["lat"].values, config.GRID_LAT, atol=1e-9)
    lon_ok = np.allclose(ds["lon"].values, config.GRID_LON, atol=1e-9)
    print(f"  lat matches config.GRID_LAT exactly : {lat_ok}")
    print(f"  lon matches config.GRID_LON exactly : {lon_ok}")
    if "level" in ds.coords:
        lev_ok = np.array_equal(
            ds["level"].values, np.asarray(config.PRESSURE_LEVELS_HPA, dtype="float64")
        )
        print(f"  level matches config.PRESSURE_LEVELS_HPA : {lev_ok}")

    _report_time(ds)
    _report_channels(ds)
    _report_nans(ds)
    _report_physics(ds)
    _report_labels(ds)

    print("\n== store ==")
    print(f"  path : {store}")
    print(f"  size : {_fmt_bytes(_dir_size_bytes(store))}")
    chunks = {n: ds[n].encoding.get("chunks") for n in ("t2m", "t") if n in ds.data_vars}
    print(f"  chunks : {chunks}")

    schema.validate_datacube(ds, require_static=True, require_satellite=False)
    print("\n  schema.validate_datacube: PASSED (require_static=True)")
    return ds


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--start", default=DEFAULT_START, help=f"ISO date (default {DEFAULT_START})"
    )
    parser.add_argument(
        "--end", default=DEFAULT_END, help=f"ISO date, inclusive (default {DEFAULT_END})"
    )
    parser.add_argument("--era5-dir", type=Path, default=config.RAW_DIR / "era5")
    parser.add_argument("--imerg-dir", type=Path, default=config.RAW_IMERG_DIR)
    parser.add_argument("--static", type=Path, default=config.DEM_ROUTING_PATH)
    parser.add_argument("--out", type=Path, default=config.DATACUBE_PATH)
    parser.add_argument(
        "--dry-run", action="store_true", help="list the inputs that would be used, then stop"
    )
    parser.add_argument(
        "--verify-only", action="store_true", help="skip the build, just report on --out"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    # ``force=True``: importing the pipeline pulls in Prefect, which installs
    # its own root handlers, and a plain basicConfig would then be a no-op.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )
    _LOG.setLevel(logging.INFO)
    args = parse_args(argv)

    if args.verify_only:
        verify(args.out)
        return 0

    start = pd.Timestamp(args.start)
    # The window is inclusive of the whole end DAY.
    end = pd.Timestamp(args.end)
    if end.normalize() == end:
        end = end + pd.Timedelta(hours=23, minutes=59, seconds=59)
    if end < start:
        raise SystemExit(f"--end {args.end} precedes --start {args.start}")

    single = _era5_paths(args.era5_dir, "sl", start, end)
    level = _era5_paths(args.era5_dir, "pl", start, end)
    imerg = _imerg_paths(args.imerg_dir, start, end)

    _LOG.info("window          : %s .. %s", start, end)
    _LOG.info("ERA5 single-lvl : %d file(s) %s", len(single), [p.name for p in single])
    _LOG.info("ERA5 pressure   : %d file(s) %s", len(level), [p.name for p in level])
    _LOG.info("IMERG daily     : %d file(s) %s .. %s", len(imerg), imerg[0].name, imerg[-1].name)
    _LOG.info("static terrain  : %s", args.static)
    _LOG.info("output store    : %s", args.out)

    if args.dry_run:
        _LOG.info("--dry-run: stopping before the build")
        return 0

    static = _load_static(args.static)
    _LOG.info("terrain loaded  : %s", list(static.data_vars))

    written = run_ingest(
        imdaa_single=single,
        imdaa_level=level,
        imerg_paths=imerg,
        static=static,
        out_path=args.out,
        time_range=(str(start), str(end)),
    )
    _LOG.info("datacube written: %s", written)

    verify(Path(written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
