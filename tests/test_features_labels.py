"""Tests for hazard label construction (``nowcast.features.labels``)."""

from __future__ import annotations

import numpy as np
import pytest

from nowcast import config, schema
from nowcast.features.labels import (
    _upstream_accumulate,
    build_labels,
    valid_label_times,
)
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


def test_single_heavy_rain_cell_peak_normalises_to_one():
    # F1: an ISOLATED occurrence cell must reach 1.0, not the ~0.16 of a
    # unit-integral Gaussian kernel.
    cube = _cube(n_hours=12)
    precip = np.zeros_like(cube["precip"].values)
    precip[4, 5, 5] = 120.0  # one cell, one timestep
    cube["precip"] = (("time", "lat", "lon"), precip)

    labels = build_labels(cube)
    cloudburst = labels.sel(hazard="cloudburst")
    assert float(cloudburst.max()) == pytest.approx(1.0, abs=1e-4)
    # the peak sits at the occurrence cell, at the lead that maps t->4
    lead = config.LEAD_TIMES_H.index(2)
    assert float(cloudburst.isel(time=2, lead=lead, lat=5, lon=5)) == pytest.approx(
        1.0, abs=1e-4
    )


def test_heavy_rain_block_makes_positive_cloudburst_label_at_matching_lead():
    cube = _cube(n_hours=12)
    precip = np.zeros_like(cube["precip"].values)
    precip[5, 3:8, 3:8] = 120.0
    cube["precip"] = (("time", "lat", "lon"), precip)

    labels = build_labels(cube)
    hazard = config.HAZARDS.index("cloudburst")
    lead = config.LEAD_TIMES_H.index(2)  # occurrence at t=5 shows at t=3, lead=2

    positive = float(labels.isel(time=3, hazard=hazard, lead=lead, lat=5, lon=5))
    assert positive >= config.LABEL_OCCURRENCE_THRESHOLD
    assert float(labels.isel(time=0, hazard=hazard, lead=0, lat=0, lon=0)) == 0.0


def test_label_valid_coord_marks_horizon_and_matches_helper():
    labels = build_labels(_cube(n_hours=12))
    assert "label_valid" in labels.coords
    valid = labels["label_valid"].values.astype(bool)
    max_lead = max(config.LEAD_TIMES_H)
    # last max_lead contiguous hours cannot have a full horizon
    assert not valid[-max_lead:].any()
    assert valid[: 12 - max_lead].all()
    np.testing.assert_array_equal(
        valid, valid_label_times(labels["time"]).values.astype(bool)
    )


def test_valid_label_times_flags_time_gap():
    import pandas as pd

    times = list(pd.date_range("2018-05-02", periods=6, freq="1h")) + list(
        pd.date_range("2018-06-01", periods=6, freq="1h")
    )
    valid = valid_label_times(np.array(times, dtype="datetime64[ns]")).values
    # nothing before the gap has a full 6 h contiguous horizon
    assert not valid[:6].any()


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
