"""CLI: build the terrain-routing Zarr store from a DEM.

Usage
-----
``python -m nowcast.dem.run --dem path/to/dem.tif``

Loads the DEM, computes the routing stack and writes it to
``config.DEM_ROUTING_PATH`` (override with ``--out``).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from nowcast import config
from nowcast.common import io as _io
from nowcast.dem.routing import compute_routing

__all__ = ["main", "build_routing_store"]

_LOG = logging.getLogger(__name__)


def build_routing_store(
    dem_path: str | Path, out_path: str | Path = config.DEM_ROUTING_PATH
) -> Path:
    """Load ``dem_path``, compute routing and write it to ``out_path`` as Zarr."""
    from nowcast.ingest.dem import load_dem  # local import: optional rasterio dep

    elevation = load_dem(dem_path)
    routing = compute_routing(elevation)
    routing = routing.assign(elevation=elevation)
    _LOG.info("routing backend: %s", routing.attrs.get("routing_backend"))
    return _io.write_zarr(routing, out_path, mode="w")


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m nowcast.dem.run``."""
    parser = argparse.ArgumentParser(description="Compute DEM routing -> Zarr")
    parser.add_argument("--dem", required=True, help="path to the DEM file (GeoTIFF or NetCDF)")
    parser.add_argument(
        "--out",
        default=str(config.DEM_ROUTING_PATH),
        help="output Zarr path (default: config.DEM_ROUTING_PATH)",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(name)s: %(message)s")
    out = build_routing_store(args.dem, args.out)
    _LOG.info("wrote routing store -> %s", out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
