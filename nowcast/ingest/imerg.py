"""GPM IMERG precipitation: network fetch (guarded) + local loader.

:func:`fetch_imerg` uses ``earthaccess`` and NASA Earthdata credentials. It is
network-guarded: with no credentials / no connectivity it raises a clear
:class:`ImergAuthError` instead of a raw library traceback. Tests only check
that it imports and fails cleanly offline.

:func:`load_imerg` reads already-downloaded IMERG HDF5/NetCDF granules and
returns ``precip`` (``mm h-1``) on the target grid.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from nowcast import config
from nowcast.common import grid as _grid
from nowcast.common import io as _io
from nowcast.ingest import names as _names
from nowcast.ingest._util import apply_var_map, resample_to_step, select_schema_vars

__all__ = ["fetch_imerg", "load_imerg", "ImergAuthError"]

_IMERG_SHORT_NAME = "GPM_3IMERGHH"  # half-hourly


def _label_bin_end(obj: xr.Dataset) -> xr.Dataset:
    """Move an hourly bin's label from the bin's start to its end.

    ``GPM_3IMERGHH`` timestamps each half-hourly window by its START, and
    :func:`nowcast.ingest._util.resample_to_step` bins left-closed/left-labelled,
    so a freshly resampled ``precip(T)`` is the mean rate over ``[T, T+1h)``.
    The datacube convention is the opposite and everything downstream depends on
    it: ``names.IMDAA_SINGLE_LEVEL['tp']`` and ``['APCP_sfc']`` accumulate over
    the hour ENDING at the timestamp, and
    :func:`nowcast.features.labels._cloudburst_mask` sums
    ``precip.rolling(time=2)``, which is xarray's backward-looking window.
    Shifting the label one step right puts the only label source in the project
    in phase with them.
    """
    if "time" not in obj.coords or obj["time"].size == 0:
        return obj
    if not np.issubdtype(np.asarray(obj["time"].values).dtype, np.datetime64):
        return obj
    shifted = obj.assign_coords(time=obj["time"] + pd.Timedelta(config.TIMESTEP))
    shifted["time"].attrs["bin_label"] = "end of the accumulation window"
    return shifted


class ImergAuthError(RuntimeError):
    """Raised when IMERG cannot be fetched (no creds / offline / library missing)."""


def _require_earthaccess():
    try:
        import earthaccess  # noqa: PLC0415
    except ImportError as err:  # pragma: no cover - depends on env
        raise ImergAuthError(
            "the 'earthaccess' package is not installed; run "
            "`pip install earthaccess` and set EARTHDATA_USERNAME / EARTHDATA_PASSWORD"
        ) from err
    return earthaccess


def fetch_imerg(
    start: str | _dt.date | _dt.datetime,
    end: str | _dt.date | _dt.datetime,
    out_dir: str | Path,
) -> list[Path]:
    """Download IMERG granules for ``[start, end]`` into ``out_dir``.

    Requires NASA Earthdata credentials in the environment
    (``EARTHDATA_USERNAME`` / ``EARTHDATA_PASSWORD``) or a ``~/.netrc`` entry.

    Raises
    ------
    ImergAuthError
        If ``earthaccess`` is missing, authentication fails, or the search
        cannot reach the CMR API.
    """
    earthaccess = _require_earthaccess()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        auth = earthaccess.login(strategy="environment")
        if auth is None or not getattr(auth, "authenticated", False):
            raise ImergAuthError("Earthdata login failed: no valid credentials found")
        results = earthaccess.search_data(
            short_name=_IMERG_SHORT_NAME,
            temporal=(str(start), str(end)),
            bounding_box=(
                config.BBOX_WEST,
                config.BBOX_SOUTH,
                config.BBOX_EAST,
                config.BBOX_NORTH,
            ),
        )
        files = earthaccess.download(results, local_path=str(out))
    except ImergAuthError:
        raise
    except Exception as err:  # noqa: BLE001 - normalise any network/library error
        raise ImergAuthError(f"IMERG fetch failed: {err}") from err
    return [Path(f) for f in files]


def load_imerg(
    paths: str | Path | Iterable[str | Path],
    *,
    group: str | None = "Grid",
    resample: bool = True,
) -> xr.Dataset:
    """Load local IMERG granules and return ``precip`` (``mm h-1``) on the grid.

    Parameters
    ----------
    paths:
        One or more IMERG ``.HDF5`` / ``.nc4`` granule paths.
    group:
        NetCDF group that holds the gridded fields (IMERG V07 uses ``"Grid"``);
        pass ``None`` for flat files such as the synthetic test fixtures.
    resample:
        Bin-average the native half-hourly (``GPM_3IMERGHH``) rate onto the
        datacube step ``config.TIMESTEP`` before returning, and label each bin
        by the END of its window (see :func:`_label_bin_end`). Skipped when the
        time axis is non-temporal or single-step.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    datasets = []
    for p in paths:
        try:
            datasets.append(_io.open_netcdf(p, group=group) if group else _io.open_netcdf(p))
        except (OSError, ValueError):
            datasets.append(_io.open_netcdf(p))
    if len(datasets) == 1:
        raw = datasets[0]
    else:
        raw = xr.concat(datasets, dim="time").sortby("time")

    mapped = apply_var_map(raw, _names.IMERG_VARS)
    mapped = select_schema_vars(mapped, ("precip",))
    if "precip" not in mapped:
        raise ValueError("no IMERG precipitation variable recognised in the input files")
    if resample:
        mapped = resample_to_step(mapped, config.TIMESTEP, how="mean")
        mapped = _label_bin_end(mapped)
    mapped = _grid.crop_bbox(mapped)
    mapped = _grid.regrid_to_target(mapped, method="linear")
    mapped["precip"] = mapped["precip"].clip(min=0.0)
    mapped.attrs["source"] = "GPM IMERG"
    return mapped
