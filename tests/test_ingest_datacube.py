"""Tests for :mod:`nowcast.ingest.datacube`."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from nowcast import config, schema
from nowcast.ingest.datacube import build_datacube, open_datacube, write_datacube
from nowcast.testing import synthetic


@pytest.fixture
def tmp_path():
    """Isolated temp dir (the shared pytest tmp root is permission-locked here)."""
    d = Path(tempfile.mkdtemp(prefix="sih-dc-"))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _parts(n_hours: int = 6):
    dc = synthetic.make_datacube(n_hours=n_hours, seed=2, with_satellite=True, with_static=True)
    surface = dc[["t2m", "u10", "v10", "prmsl", "tcwv", "cape", "cin", "precip"]]
    level = dc[["t", "rh", "z", "u", "v"]]
    satellite = dc[["ctt", "wv_bt"]]
    static = dc[["elevation", "slope", "flow_accumulation", "flow_direction", "hand"]]
    return surface, level, satellite, static


def test_build_datacube_passes_schema_validation():
    surface, level, satellite, static = _parts()
    cube = build_datacube(surface, level, satellite=satellite, static=static)
    schema.validate_datacube(cube, require_static=True, require_satellite=True)
    assert cube["t2m"].dims == schema.SURFACE_DIMS
    assert cube["t"].dims == schema.LEVEL_DIMS
    assert cube["elevation"].dims == schema.STATIC_DIMS
    np.testing.assert_array_equal(
        cube["level"].values, np.asarray(config.PRESSURE_LEVELS_HPA, dtype="float64")
    )


def test_build_datacube_surface_and_level_only():
    surface, level, _, _ = _parts()
    cube = build_datacube(surface, level)
    schema.validate_datacube(cube, require_static=False, require_satellite=False)


def test_build_datacube_accepts_bare_elevation_dataarray():
    surface, level, _, static = _parts()
    cube = build_datacube(surface, level, static=static["elevation"])
    assert "elevation" in cube


def test_build_datacube_rejects_bad_grid():
    surface, level, _, _ = _parts()
    with pytest.raises(schema.SchemaError):
        build_datacube(surface.isel(lat=slice(0, 10)), level)


def test_write_and_open_datacube_roundtrip(tmp_path):
    surface, level, satellite, static = _parts()
    cube = build_datacube(surface, level, satellite=satellite, static=static)
    store = tmp_path / "datacube.zarr"
    write_datacube(cube, store)
    reopened = open_datacube(store)
    schema.validate_datacube(reopened, require_static=True, require_satellite=True)
