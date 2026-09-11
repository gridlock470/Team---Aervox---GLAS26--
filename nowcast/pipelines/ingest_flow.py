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
        # snap to the IMDAA clock, but never carry a value across a gap larger
        # than one datacube step (cells outside tolerance become NaN).
        merged = merged.reindex(
            time=surface["time"],
            method="nearest",
            tolerance=pd.Timedelta(config.TIMESTEP),
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
) -> str:
    """Shared body for :func:`run_ingest` and :func:`ingest_datacube`."""
    wget, load_imdaa, load_precip, load_sat, build = stages
    if wget_dirs:
        for line in wget(wget_dirs):
            _LOG.info("wget: %s", line)
    surface, level = load_imdaa(imdaa_single, imdaa_level)
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
) -> str:
    """Plain-Python driver for the ingest pipeline (no Prefect required).

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
        stages=(_task_wget, _task_imdaa, _task_precip, _task_satellite, _task_build),
    )


if PREFECT_AVAILABLE:  # pragma: no cover - depends on env
    ingest_datacube = _prefect_flow(name="ingest-datacube")(_ingest_datacube_impl)
else:
    ingest_datacube = _ingest_datacube_impl
