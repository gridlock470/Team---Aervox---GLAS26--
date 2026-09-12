"""Build the model-input feature cube and the hazard label cube from the datacube.

This is the step between ``scripts/build_datacube.py`` and training. It opens
``config.DATACUBE_PATH``, runs ``nowcast.features.assemble.assemble_features``
(which derives the 9 channels the datacube does not carry: ``iwv``,
``iwv_tendency_1h``, ``lifted_index``, ``conv_850``, ``shear_0_6km``,
``shear_0_1km``, and the three constant-filled INSAT stand-ins) and
``nowcast.features.labels.build_labels``, then writes both to Zarr:

* ``DataSet/processed/features.zarr`` -- exactly ``schema.FEATURE_CHANNELS``,
  in that order, float32, all finite;
* ``DataSet/processed/labels.zarr``   -- ``hazard_probability`` with dims
  ``schema.TARGET_DIMS`` plus the ``label_valid`` coordinate on ``time``.

Point ``NowcastDataModule(features=..., labels=..., terrain=config.DEM_ROUTING_PATH)``
at those two stores.

Terrain routing
---------------
``build_labels`` defaults to ``routing=None``, which makes the flash-flood mask
fall back to "cloudburst cell plus its lowest neighbour" instead of routing rain
over the real D8 drainage network. An audit put the two at IoU 0.235 -- a
similar positive count in mostly *different* cells -- so this script threads the
real store at ``config.DEM_ROUTING_PATH`` through by default. ``--no-routing``
reproduces the fallback for comparison; it is not the production path.

Typical use::

    .venv/Scripts/python.exe scripts/build_features.py --limit-hours 48   # smoke test
    .venv/Scripts/python.exe scripts/build_features.py                    # full build
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nowcast import config, schema  # noqa: E402
from nowcast.features.assemble import assemble_features  # noqa: E402
from nowcast.features.labels import build_labels  # noqa: E402

_LOG = logging.getLogger("build_features")

DEFAULT_FEATURES_OUT: Path = config.PROCESSED_DIR / "features.zarr"
DEFAULT_LABELS_OUT: Path = config.PROCESSED_DIR / "labels.zarr"

# Zarr chunking. Time-major, whole grid per chunk: the datacube uses time=96 and
# every consumer reads a contiguous INPUT_SEQ_LEN-hour window over the full map.
_FEATURE_TIME_CHUNK = 96
# Labels are (time, hazard, lead, lat, lon) -- 18x wider per step, so a smaller
# time chunk keeps a chunk near 10 MB instead of a single 0.6 GB blob.
_LABEL_TIME_CHUNK = 64

# Physical sanity ranges quoted in the report (not enforced).
_PHYSICS_NOTES: dict[str, str] = {
    "cape": "J kg-1, expect 0-6000",
    "lifted_index": "K, expect ~-15..+25 (negative = unstable)",
    "iwv": "kg m-2, expect ~5-70",
    "shear_0_6km": "m s-1, expect 0-50",
    "shear_0_1km": "m s-1, expect 0-30",
    "cin": "J kg-1, expect <= 0",
    "conv_850": "s-1, expect ~+/-1e-3",
    "t2m": "K, expect ~270-320",
}


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def _open_datacube(path: Path, limit_hours: int | None) -> xr.Dataset:
    ds = xr.open_zarr(path)
    if limit_hours is not None:
        ds = ds.isel(time=slice(0, int(limit_hours)))
    return ds


def _open_routing(path: Path | None) -> xr.Dataset | None:
    """Open the D8 routing store used by the flash-flood label.

    Returns ``None`` only when routing was explicitly disabled. A *missing*
    store is an error rather than a silent downgrade: the fallback labels
    different cells, and a label set that quietly changed meaning is worse than
    a failed run.
    """
    if path is None:
        _LOG.warning(
            "--no-routing: flash-flood labels use the lowest-neighbour FALLBACK, "
            "which an audit put at IoU 0.235 against the D8 labels"
        )
        return None
    if not Path(path).exists():
        raise FileNotFoundError(
            f"terrain routing store {path} not found; build it first or pass --no-routing"
        )
    routing = xr.open_zarr(path).load()
    if "flow_direction" not in routing.data_vars:
        raise KeyError(f"{path} carries no 'flow_direction'; cannot route flash floods")
    _LOG.info("routing loaded : %s (D8 flow_direction present)", path)
    return routing


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def _write_features(features: xr.Dataset, path: Path) -> Path:
    chunks = {"time": min(_FEATURE_TIME_CHUNK, int(features.sizes["time"]))}
    features.chunk(chunks).to_zarr(path, mode="w", consolidated=True)
    return path


def _write_labels(labels: xr.DataArray, path: Path) -> Path:
    chunks = {"time": min(_LABEL_TIME_CHUNK, int(labels.sizes["time"]))}
    ds = labels.to_dataset(name="hazard_probability")
    # ``hazard`` is a string coordinate; leave it un-chunked and let Zarr store
    # it as a small object array alongside the float32 cube.
    ds.chunk(chunks).to_zarr(path, mode="w", consolidated=True)
    return path


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


def _report_features(store: Path) -> xr.Dataset:
    ds = xr.open_zarr(store).load()
    print("\n" + "=" * 78)
    print(f"FEATURE CUBE  {store}")
    print("=" * 78)
    print(f"  dims   : {dict(ds.sizes)}")
    print(f"  size   : {_fmt_bytes(_dir_size_bytes(store))}")

    present = [c for c in schema.FEATURE_CHANNELS if c in ds.data_vars]
    missing = [c for c in schema.FEATURE_CHANNELS if c not in ds.data_vars]
    extra = [c for c in ds.data_vars if c not in schema.FEATURE_CHANNELS]
    print(f"\n  schema.FEATURE_CHANNELS present : {len(present)} / {len(schema.FEATURE_CHANNELS)}")
    if missing:
        print(f"  *** MISSING: {missing} ***")
    if extra:
        print(f"  *** UNEXPECTED EXTRA CHANNELS: {extra} ***")
    # Zarr does not preserve variable insertion order, and it does not need to:
    # ``NowcastDataset.__getitem__`` stacks channels by *name* in
    # ``schema.FEATURE_CHANNELS`` order. Set equality is the real contract.
    set_ok = set(ds.data_vars) == set(schema.FEATURE_CHANNELS)
    print(f"  channel set matches schema      : {set_ok}")

    print("\n  channel                 dims                      NaN         inf")
    for name in schema.FEATURE_CHANNELS:
        if name not in ds.data_vars:
            continue
        values = ds[name].values
        n_nan = int(np.isnan(values).sum())
        n_inf = int(np.isinf(values).sum())
        dims = "(" + ",".join(ds[name].dims) + ")"
        flag = "" if (n_nan == 0 and n_inf == 0) else "   <-- NON-FINITE"
        print(f"  {name:<22} {dims:<22} {n_nan:>10,} {n_inf:>11,}{flag}")

    print("\n  physical sanity")
    for name, note in _PHYSICS_NOTES.items():
        if name not in ds.data_vars:
            continue
        values = ds[name].values
        print(
            f"  {name:<16} min={np.nanmin(values):13.4f}  mean={np.nanmean(values):13.4f}  "
            f"max={np.nanmax(values):13.4f}   ({note})"
        )
    return ds


def _report_labels(store: Path) -> xr.DataArray:
    ds = xr.open_zarr(store)
    labels = ds["hazard_probability"]
    print("\n" + "=" * 78)
    print(f"LABEL CUBE  {store}")
    print("=" * 78)
    print(f"  dims   : {dict(labels.sizes)}  (schema.TARGET_DIMS = {schema.TARGET_DIMS})")
    print(f"  size   : {_fmt_bytes(_dir_size_bytes(store))}")
    print(f"  routing: {labels.attrs.get('flash_flood_routing', 'unrecorded')}")

    values = labels.values
    print(f"  range  : [{float(np.nanmin(values)):.4f}, {float(np.nanmax(values)):.4f}]")
    print(f"  NaN    : {int(np.isnan(values).sum()):,}   inf: {int(np.isinf(values).sum()):,}")

    threshold = float(config.LABEL_OCCURRENCE_THRESHOLD)
    print(f"\n  positive-cell rate at >= {threshold}")
    for hi, hazard in enumerate(config.HAZARDS):
        block = values[:, hi]
        rate = float((block >= threshold).mean())
        print(f"  {hazard:<14} {rate:.3e}  ({int((block >= threshold).sum()):,} cells)")

    if "label_valid" in labels.coords:
        valid = np.asarray(labels["label_valid"].values, dtype=bool)
        print(f"\n  label_valid : {int(valid.sum()):,} / {valid.size:,} steps")
    return labels


def _check_datamodule(features_store: Path, labels_store: Path) -> None:
    """Prove the blocked step is unblocked: ``setup('fit')`` against the stores."""
    from nowcast.data.datamodule import NowcastDataModule

    print("\n" + "=" * 78)
    print("NowcastDataModule.setup('fit')")
    print("=" * 78)
    dm = NowcastDataModule(
        features=features_store,
        labels=labels_store,
        terrain=config.DEM_ROUTING_PATH,
        batch_size=2,
    )
    dm.setup("fit")
    for split, dataset in (("train", dm._train), ("val", dm._val), ("test", dm._test)):
        n = 0 if dataset is None else len(dataset)
        print(f"  {split:<6} windows : {n:,}")
    batch = next(iter(dm.train_dataloader()))
    x, y, terrain = batch
    print(f"  x       : {tuple(x.shape)}  {x.dtype}  (expect (B, {schema.INPUT_SHAPE}))")
    print(f"  y       : {tuple(y.shape)}  {y.dtype}  (expect (B, {schema.TARGET_SHAPE}))")
    print(f"  terrain : {tuple(terrain.shape)}  {terrain.dtype}")
    print(f"  x finite: {bool(np.isfinite(x.numpy()).all())}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--datacube", type=Path, default=config.DATACUBE_PATH)
    parser.add_argument("--features-out", type=Path, default=DEFAULT_FEATURES_OUT)
    parser.add_argument("--labels-out", type=Path, default=DEFAULT_LABELS_OUT)
    parser.add_argument("--routing", type=Path, default=config.DEM_ROUTING_PATH)
    parser.add_argument(
        "--no-routing",
        action="store_true",
        help="use the lowest-neighbour flash-flood fallback instead of D8 routing",
    )
    parser.add_argument(
        "--limit-hours",
        type=int,
        default=None,
        metavar="N",
        help="build only the first N hours (fast smoke test)",
    )
    parser.add_argument(
        "--verify-only", action="store_true", help="skip the build, just report on the stores"
    )
    parser.add_argument(
        "--no-datamodule-check",
        action="store_true",
        help="skip the closing NowcastDataModule.setup('fit') check",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )
    args = parse_args(argv)

    if not args.verify_only:
        cube = _open_datacube(args.datacube, args.limit_hours)
        times = pd.DatetimeIndex(cube["time"].values)
        _LOG.info("datacube       : %s", args.datacube)
        _LOG.info("window         : %s .. %s (%d steps)", times[0], times[-1], len(times))
        routing = None if args.no_routing else _open_routing(args.routing)

        t0 = time.perf_counter()
        features = assemble_features(cube)
        t_features = time.perf_counter() - t0
        _LOG.info("assembled %d channels in %.1f s", len(features.data_vars), t_features)

        t0 = time.perf_counter()
        _write_features(features, args.features_out)
        t_write_features = time.perf_counter() - t0
        _LOG.info("features written: %s (%.1f s)", args.features_out, t_write_features)

        t0 = time.perf_counter()
        labels = build_labels(cube, routing=routing)
        labels.attrs["flash_flood_routing"] = (
            "lowest-neighbour fallback" if routing is None else f"D8 from {args.routing}"
        )
        t_labels = time.perf_counter() - t0
        _LOG.info("labels built in %.1f s", t_labels)

        t0 = time.perf_counter()
        _write_labels(labels, args.labels_out)
        t_write_labels = time.perf_counter() - t0
        _LOG.info("labels written : %s (%.1f s)", args.labels_out, t_write_labels)

        total = t_features + t_write_features + t_labels + t_write_labels
        print("\n== wall clock ==")
        print(f"  assemble_features : {t_features:8.1f} s")
        print(f"  write features    : {t_write_features:8.1f} s")
        print(f"  build_labels      : {t_labels:8.1f} s")
        print(f"  write labels      : {t_write_labels:8.1f} s")
        print(f"  TOTAL             : {total:8.1f} s  ({total / 60.0:.1f} min)")

    _report_features(args.features_out)
    _report_labels(args.labels_out)
    if not args.no_datamodule_check:
        _check_datamodule(args.features_out, args.labels_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
