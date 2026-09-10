"""Copernicus GLO-30 DEM: network fetch (guarded) + local loader.

:func:`fetch_glo30` downloads the 1x1 degree GLO-30 tiles covering a bounding
box over HTTP (via ``pooch``). It is network-guarded and raises a clear
:class:`DemDownloadError` when offline / when a tile is missing.

:func:`load_dem` opens a local DEM (GeoTIFF or NetCDF), returns an
``elevation`` :class:`xarray.DataArray` regridded to the target grid.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import xarray as xr

from nowcast import config
from nowcast.common import grid as _grid
from nowcast.common import io as _io

__all__ = ["fetch_glo30", "load_dem", "DemDownloadError"]

# Public AWS mirror of the Copernicus GLO-30 DEM.
_GLO30_BASE = (
    "https://copernicus-dem-30m.s3.amazonaws.com/"
    "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM/"
    "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM.tif"
)


class DemDownloadError(RuntimeError):
    """Raised when the GLO-30 DEM cannot be downloaded."""


def _tile_names(bbox: tuple[float, float, float, float]) -> list[tuple[int, int]]:
    west, south, east, north = bbox
    lats = range(math.floor(south), math.ceil(north))
    lons = range(math.floor(west), math.ceil(east))
    return [(la, lo) for la in lats for lo in lons]


def fetch_glo30(
    bbox: tuple[float, float, float, float] | None = None,
    out_dir: str | Path = config.RAW_DEM_DIR,
) -> list[Path]:
    """Download the GLO-30 tiles covering ``bbox`` (default: the pilot bbox).

    Parameters
    ----------
    bbox:
        ``(west, south, east, north)`` in degrees. Defaults to the pilot box.
    out_dir:
        Directory to save the ``.tif`` tiles into.

    Raises
    ------
    DemDownloadError
        If ``pooch`` is missing or any tile cannot be retrieved (offline,
        404, ...).
    """
    if bbox is None:
        bbox = (
            config.BBOX_WEST,
            config.BBOX_SOUTH,
            config.BBOX_EAST,
            config.BBOX_NORTH,
        )
    try:
        import pooch  # noqa: PLC0415
    except ImportError as err:  # pragma: no cover - depends on env
        raise DemDownloadError("the 'pooch' package is required to fetch the GLO-30 DEM") from err

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for lat, lon in _tile_names(bbox):
        ns, ew = ("N" if lat >= 0 else "S"), ("E" if lon >= 0 else "W")
        url = _GLO30_BASE.format(ns=ns, ew=ew, lat=abs(lat), lon=abs(lon))
        try:
            path = pooch.retrieve(url=url, known_hash=None, path=str(out))
        except Exception as err:  # noqa: BLE001 - normalise network/library errors
            raise DemDownloadError(f"could not download GLO-30 tile {url}: {err}") from err
        saved.append(Path(path))
    if not saved:
        raise DemDownloadError(f"no GLO-30 tiles intersect bbox {bbox}")
    return saved


def _open_dem_any(path: Path) -> xr.DataArray:
    """Open a DEM file as a 2-D DataArray with ``lat`` / ``lon`` coords."""
    if path.suffix.lower() in {".tif", ".tiff"}:
        try:
            import rioxarray  # noqa: F401, PLC0415

            da = xr.open_dataarray(path, engine="rasterio")
        except Exception as err:  # noqa: BLE001
            raise OSError(f"could not read GeoTIFF DEM {path}: {err}") from err
        da = da.squeeze(drop=True)
        da = da.rename({d: "lat" for d in da.dims if d in ("y", "latitude")})
        da = da.rename({d: "lon" for d in da.dims if d in ("x", "longitude")})
        return da
    ds = _io.open_netcdf(path)
    for cand in ("elevation", "band_data", "Band1", "z", "dem"):
        if cand in ds:
            return ds[cand]
    return ds[list(ds.data_vars)[0]]


def load_dem(path: str | Path | Iterable[str | Path]) -> xr.DataArray:
    """Load a DEM (one file or a list of tiles) as ``elevation`` on the grid.

    Returns
    -------
    xarray.DataArray
        Name ``elevation``, units ``m``, dims ``(lat, lon)`` exactly on the
        target grid.
    """
    paths = [Path(path)] if isinstance(path, (str, Path)) else [Path(p) for p in path]
    tiles = [_open_dem_any(p) for p in paths]
    if len(tiles) == 1:
        dem = tiles[0]
    else:
        dem = xr.combine_by_coords(
            [t.to_dataset(name="elevation") for t in tiles]
        )["elevation"]

    dem = dem.to_dataset(name="elevation")
    dem = _io.standardize_coords(dem)
    dem = _grid.crop_bbox(dem)
    dem = _grid.regrid_to_target(dem, method="linear")["elevation"]
    dem = dem.astype("float32")
    dem.name = "elevation"
    dem.attrs.update(units="m", long_name="surface elevation (Copernicus GLO-30)")
    dem = dem.where(np.isfinite(dem), 0.0)
    return dem
