"""Prefect flow that builds the datacube from raw source files.

Stages (one ``@task`` each):

1. run any portal wget scripts in the raw directories;
2. load IMDAA single-level + pressure-level fields;
3. load the precip sources (MERA / IMERG / INSAT QPE) and merge them;
4. load INSAT L1C brightness temps and (optionally) DEM-derived static fields;
5. build + validate + write the datacube.

Prefect is an optional heavy dependency. The import is guarded: importing this
module always works. :func:`ingest_datacube` is the Prefect ``@flow``; when
Prefect is missing it raises a clear :class:`RuntimeError`. The plain-Python
driver :func:`run_ingest` runs the identical pipeline with no Prefect.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from nowcast import config
from nowcast.ingest import imdaa, insat, mera, merge_precip
from nowcast.ingest.datacube import build_datacube, write_datacube
from nowcast.ingest.run_wget import run_wget_scripts

__all__ = ["ingest_datacube", "run_ingest", "PREFECT_AVAILABLE"]

_LOG = logging.getLogger(__name__)

try:  # pragma: no cover - depends on env
    from prefect import flow as _prefect_flow
    from prefect import task as _prefect_task

    PREFECT_AVAILABLE = True
except ImportError:  # pragma: no cover - Prefect not yet installed
    PREFECT_AVAILABLE = False
    _prefect_flow = None
    _prefect_task = None


# ---------------------------------------------------------------------------
# Plain-Python stage implementations (the real work)
# ---------------------------------------------------------------------------
def _stage_wget(directories) -> list[str]:
    """Run every download script under each directory; return a size log."""
    logged: list[str] = []
    for directory in directories:
        directory = Path(directory)
        if not directory.is_dir():
            continue
        for res in run_wget_scripts(directory):
            logged.append(f"{res.script.name}: rc={res.returncode}, {res.total_bytes} bytes")
    return logged


def _stage_imdaa(single_paths, level_paths) -> tuple[xr.Dataset, xr.Dataset]:
    """Load IMDAA single-level and pressure-level fields onto the grid."""
    return (
        imdaa.load_imdaa_single_level(single_paths),
        imdaa.load_imdaa_pressure_level(level_paths),
    )


def _stage_precip(surface, mera_paths, imerg_paths, insat_qpe_paths) -> xr.Dataset:
    """Merge external precip products into ``surface['precip']`` (MERA preferred)."""
    mera_ds = mera.load_mera(mera_paths) if mera_paths else None
    imerg_ds = None
    if imerg_paths:
        from nowcast.ingest.imerg import load_imerg

        imerg_ds = load_imerg(imerg_paths)
    qpe_ds = insat.load_insat_qpe(insat_qpe_paths) if insat_qpe_paths else None

    if mera_ds is None and imerg_ds is None and qpe_ds is None:
        return surface
    merged = merge_precip.merge_precip(mera=mera_ds, imerg=imerg_ds, insat_qpe=qpe_ds)
    if np.issubdtype(np.asarray(merged["time"].values).dtype, np.datetime64):
        # Snap to the IMDAA clock, but never carry a value across a gap (cells
        # outside tolerance become NaN). The radius is HALF a step: a tolerance
        # of a whole step lets a sample slide a full timestep and land on a
        # neighbouring hour, which is the very carry-across this guards against.
        merged = merged.reindex(
            time=surface["time"],
            method="nearest",
            tolerance=pd.Timedelta(config.TIMESTEP) / 2,
        )
    else:
        merged = merged.reindex(time=surface["time"], method="nearest")
    return surface.assign(precip=merged)


def _stage_satellite(l1c_paths) -> xr.Dataset | None:
    """Load INSAT L1C brightness temperatures, or ``None`` if not requested."""
    if not l1c_paths:
        return None
    return insat.load_insat_l1c(l1c_paths)


def _stage_build(surface, level, satellite, static, out_path) -> str:
    """Build + validate + write the datacube; return the store path."""
    cube = build_datacube(surface, level, satellite=satellite, static=static)
    return str(write_datacube(cube, out_path))


# Prefect task views over the same callables (used by the flow only).
if PREFECT_AVAILABLE:  # pragma: no cover - depends on env
    _task_wget = _prefect_task(name="run-wget-scripts")(_stage_wget)
    _task_imdaa = _prefect_task(name="load-imdaa")(_stage_imdaa)
    _task_precip = _prefect_task(name="merge-precip")(_stage_precip)
    _task_satellite = _prefect_task(name="load-satellite")(_stage_satellite)
    _task_build = _prefect_task(name="build-datacube")(_stage_build)
else:
    _task_wget = _stage_wget
    _task_imdaa = _stage_imdaa
    _task_precip = _stage_precip
    _task_satellite = _stage_satellite
    _task_build = _stage_build


def _trim_time(ds: xr.Dataset, time_range, *, pad: str | None = None) -> xr.Dataset:
    """Restrict ``ds`` to the inclusive ``(start, end)`` ISO window.

    ``pad`` widens the slice on both sides. A coarser-cadence source needs it:
    the pressure levels are 3-hourly, so trimming them to exactly the window
    leaves the final hours of the cube with no later sample to interpolate
    towards and they come out NaN.
    """
    if not time_range or "time" not in ds.coords:
        return ds
    if not np.issubdtype(np.asarray(ds["time"].values).dtype, np.datetime64):
        return ds
    start, end = time_range
    if pad:
        margin = pd.Timedelta(pad)
        trimmed = ds.sel(
            time=slice(pd.Timestamp(start) - margin, pd.Timestamp(end) + margin)
        )
    else:
        trimmed = ds.sel(time=slice(start, end))
    if trimmed.sizes.get("time", 0) == 0:
        raise ValueError(
            f"time_range {time_range} selects no steps from a source spanning "
            f"{ds['time'].values[0]} .. {ds['time'].values[-1]}"
        )
    return trimmed


def _clip_to_span(ds: xr.Dataset, other: xr.Dataset) -> xr.Dataset:
    """Clip ``ds``'s time axis to the span ``other`` actually covers.

    The surface source defines the datacube clock, but the pressure levels are a
    REQUIRED part of the contract (``schema.LEVEL_VARS``) and their archive can
    stop earlier -- ERA5's 3-hourly levels end at 21:00 on a day whose hourly
    single levels run to 23:00. Those trailing hours cannot be interpolated
    towards anything, so they would enter the cube as all-NaN level fields and
    ``schema.validate_sample`` rejects any input tensor containing NaN. Ending
    the clock at the last fully-populated step is honest; carrying NaN forward
    is not.
    """
    for obj in (ds, other):
        if "time" not in obj.coords or obj.sizes.get("time", 0) == 0:
            return ds
        if not np.issubdtype(np.asarray(obj["time"].values).dtype, np.datetime64):
            return ds
    lo = np.asarray(other["time"].values).min()
    hi = np.asarray(other["time"].values).max()
    clipped = ds.sel(time=slice(lo, hi))
    dropped = ds.sizes["time"] - clipped.sizes.get("time", 0)
    if dropped:
        _LOG.info(
            "clipped %d surface step(s) not covered by the pressure-level source "
            "(clock now %s .. %s)",
            dropped,
            clipped["time"].values[0],
            clipped["time"].values[-1],
        )
    return clipped


def _pipeline(
    *,
    imdaa_single,
    imdaa_level,
    mera_paths,
    imerg_paths,
    insat_qpe_paths,
    insat_l1c_paths,
    static,
    wget_dirs,
    out_path,
    stages,
    time_range=None,
) -> str:
    """Shared body for :func:`run_ingest` and :func:`ingest_datacube`."""
    wget, load_imdaa, load_precip, load_sat, build = stages
    if wget_dirs:
        for line in wget(wget_dirs):
            _LOG.info("wget: %s", line)
    surface, level = load_imdaa(imdaa_single, imdaa_level)
    surface = _trim_time(surface, time_range)
    # The surface clock defines the cube; the levels only need to bracket it.
    level = _trim_time(level, time_range, pad="1D")
    surface = _clip_to_span(surface, level)
    surface = load_precip(surface, mera_paths, imerg_paths, insat_qpe_paths)
    satellite = load_sat(insat_l1c_paths)
    return build(surface, level, satellite, static, out_path)


def run_ingest(
    *,
    imdaa_single,
    imdaa_level,
    mera_paths=None,
    imerg_paths=None,
    insat_qpe_paths=None,
    insat_l1c_paths=None,
    static=None,
    wget_dirs=(),
    out_path: str | Path = config.DATACUBE_PATH,
    time_range: tuple[str, str] | None = None,
) -> str:
    """Plain-Python driver for the ingest pipeline (no Prefect required).

    Parameters
    ----------
    time_range:
        Optional inclusive ``(start, end)`` ISO window. The datacube clock comes
        from the single-level source, which routinely covers more than the
        precip source does; since :func:`nowcast.ingest.merge_precip.merge_precip`
        fills unmatched cells with ``0.0``, building past the precip record
        would write fabricated "no rain" into the only label source. Trim to the
        window every source actually covers.

    Returns the path of the written datacube Zarr store.
    """
    return _pipeline(
        imdaa_single=imdaa_single,
        imdaa_level=imdaa_level,
        mera_paths=mera_paths,
        imerg_paths=imerg_paths,
        insat_qpe_paths=insat_qpe_paths,
        insat_l1c_paths=insat_l1c_paths,
        static=static,
        wget_dirs=wget_dirs,
        out_path=out_path,
        time_range=time_range,
        stages=(_stage_wget, _stage_imdaa, _stage_precip, _stage_satellite, _stage_build),
    )


def _ingest_datacube_impl(
    *,
    imdaa_single,
    imdaa_level,
    mera_paths=None,
    imerg_paths=None,
    insat_qpe_paths=None,
    insat_l1c_paths=None,
    static=None,
    wget_dirs=(),
    out_path: str | Path = config.DATACUBE_PATH,
    time_range: tuple[str, str] | None = None,
) -> str:
    """Flow body: same pipeline, wired through Prefect tasks."""
    if not PREFECT_AVAILABLE:  # pragma: no cover - exercised only without Prefect
        raise RuntimeError(
            "Prefect is not installed. Install 'prefect' to run this flow, or call "
            "nowcast.pipelines.ingest_flow.run_ingest(...) for the plain-Python pipeline."
        )
    return _pipeline(
        imdaa_single=imdaa_single,
        imdaa_level=imdaa_level,
        mera_paths=mera_paths,
        imerg_paths=imerg_paths,
        insat_qpe_paths=insat_qpe_paths,
        insat_l1c_paths=insat_l1c_paths,
        static=static,
        wget_dirs=wget_dirs,
        out_path=out_path,
        time_range=time_range,
        stages=(_task_wget, _task_imdaa, _task_precip, _task_satellite, _task_build),
    )


if PREFECT_AVAILABLE:  # pragma: no cover - depends on env
    ingest_datacube = _prefect_flow(name="ingest-datacube")(_ingest_datacube_impl)
else:
    ingest_datacube = _ingest_datacube_impl
