"""Tests for hazard label construction (``nowcast.features.labels``)."""

from __future__ import annotations

import numpy as np

from nowcast import config, schema
from nowcast.features.labels import _upstream_accumulate, build_labels
from nowcast.testing import synthetic


def _cube(n_hours: int = 8, seed: int = 2):
    return synthetic.make_datacube(n_hours=n_hours, seed=seed).isel(
        lat=slice(0, 10), lon=slice(0, 10)
    )


def test_dims_coords_and_unit_range():
    labels = build_labels(_cube())
    assert tuple(labels.dims) == schema.TARGET_DIMS
    assert list(labels["hazard"].values) == list(config.HAZARDS)
    assert list(labels["lead"].values) == list(config.LEAD_TIMES_H)
    assert float(labels.min()) >= 0.0
    assert float(labels.max()) <= 1.0
    assert labels.sizes["time"] == 8


def test_heavy_rain_cell_makes_positive_cloudburst_label_at_matching_lead():
    cube = _cube(n_hours=10)
    precip = np.zeros_like(cube["precip"].values)
    precip[5, 3:8, 3:8] = 120.0  # well above the cloudburst rate threshold
    cube["precip"] = (("time", "lat", "lon"), precip)

    labels = build_labels(cube)
    hazard = config.HAZARDS.index("cloudburst")
    lead = config.LEAD_TIMES_H.index(2)  # occurrence at t=5 shows at t=3, lead=2

    assert float(labels.isel(time=3, hazard=hazard, lead=lead, lat=5, lon=5)) > 0.0
    assert float(labels.isel(time=0, hazard=hazard, lead=0, lat=0, lon=0)) == 0.0


def test_upstream_accumulate_sums_the_chain():
    # A 1-D chain draining east (D8 code 1); last cell is the outlet.
    flow_dir = np.ones((1, 5), dtype="float32")
    flow_dir[0, -1] = 0
    field = np.ones((1, 5))
    acc = _upstream_accumulate(field, flow_dir)
    assert acc[0, -1] == 5.0
    assert acc[0, 0] == 1.0


def test_flash_flood_with_routing_is_finite_and_bounded():
    cube = _cube(n_hours=4)
    routing = {
        "flow_direction": cube["flow_direction"],
        "flow_accumulation": cube["flow_accumulation"],
    }
    labels = build_labels(cube, routing=routing)
    flash = labels.isel(hazard=config.HAZARDS.index("flash_flood"))
    assert np.isfinite(flash.values).all()
    assert float(flash.max()) <= 1.0
    assert float(flash.min()) >= 0.0


def test_flash_flood_fallback_without_routing_runs():
    labels = build_labels(_cube(n_hours=4))
    flash = labels.isel(hazard=config.HAZARDS.index("flash_flood"))
    assert np.isfinite(flash.values).all()
