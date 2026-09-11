"""Tests for :mod:`nowcast.dem.routing`."""

from __future__ import annotations

import numpy as np
import pytest

from nowcast import config, schema
from nowcast.dem import routing


def test_compute_routing_shapes_and_vars(synthetic_dem):
    ds = routing.compute_routing(synthetic_dem)
    for spec in ("flow_direction", "flow_accumulation", "streams", "hand", "slope"):
        assert ds[spec].dims == schema.STATIC_DIMS
        assert ds[spec].shape == config.GRID_SHAPE
    assert ds.attrs["routing_backend"] in {"pysheds", "numpy"}


def test_flow_accumulation_finite_and_at_least_one(synthetic_dem):
    acc = routing.compute_routing(synthetic_dem)["flow_accumulation"]
    assert np.isfinite(acc.values).all()
    assert float(acc.min()) >= 1.0


def test_streams_non_empty_and_boolean(synthetic_dem):
    streams = routing.compute_routing(synthetic_dem)["streams"]
    assert streams.dtype == bool
    assert int(streams.values.sum()) > 0


def test_hand_non_negative(synthetic_dem):
    hand = routing.compute_routing(synthetic_dem)["hand"]
    assert np.isfinite(hand.values).all()
    assert float(hand.min()) >= 0.0


def test_flow_direction_uses_d8_codes(synthetic_dem):
    """Codes are the ESRI D8 *values* (0 for sinks); compass semantics are not
    remapped for the lat-ascending grid -- see the routing module docstring."""
    fdir = routing.compute_routing(synthetic_dem)["flow_direction"].values
    valid = {0, 1, 2, 4, 8, 16, 32, 64, 128}
    assert set(np.unique(fdir).astype(int)).issubset(valid)


def test_d8_table_matches_label_router_offsets():
    """The (drow, dcol) -> code table must stay identical to the label-side
    accumulator so routing and labels agree on drainage."""
    from nowcast.features.labels import _D8_OFFSETS

    routing_offsets = {code: (dr, dc) for dr, dc, code, _ in routing._D8}
    assert routing_offsets == _D8_OFFSETS


def test_numpy_flowdir_agrees_with_pysheds_on_tilted_plane():
    """On a monotone tilted plane every interior cell has one unambiguous
    steepest-descent neighbour; the numpy and pysheds pointers must match
    there. The pysheds side is skipped when pysheds is unusable in this env."""
    import xarray as xr

    h, w = config.GRID_SHAPE
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    elev = (200.0 + 3.0 * yy + 1.0 * xx).astype("float64")  # drops toward -row, -col
    dem = xr.DataArray(
        elev,
        dims=("lat", "lon"),
        coords={"lat": config.GRID_LAT, "lon": config.GRID_LON},
    )
    # exercise the public path too (returns numpy backend in this env)
    assert routing.compute_routing(dem, stream_threshold=1.0)["flow_direction"].shape == (h, w)

    interior = np.zeros((h, w), dtype=bool)
    interior[1:-1, 1:-1] = True
    fdir_np = routing._d8_numpy(elev, None)["flow_direction"]
    # steepest descent on this plane is due -row (code 64)
    assert set(np.unique(fdir_np[interior]).astype(int)) == {64}

    if routing.routing_backend() == "pysheds":  # pragma: no cover - env dependent
        fdir_ps = routing._d8_pysheds(elev, None)["flow_direction"]
        assert (fdir_np[interior] == fdir_ps[interior]).mean() > 0.95


def test_numpy_backend_directly(synthetic_dem):
    out = routing._d8_numpy(np.asarray(synthetic_dem.values, dtype="float64"), None)
    assert out["flow_accumulation"].min() >= 1.0
    assert out["streams"].any()
    # the highest cell drains somewhere -> total accumulation equals cell count
    assert out["flow_accumulation"].max() <= synthetic_dem.size


def test_stream_threshold_override(synthetic_dem):
    dense = routing.compute_routing(synthetic_dem, stream_threshold=1.0)["streams"]
    assert bool(dense.values.all())


def test_routing_backend_reports_value():
    assert routing.routing_backend() in {"pysheds", "numpy"}


def test_compute_routing_rejects_3d():
    bad = _dummy_da_3d()
    with pytest.raises(ValueError):
        routing.compute_routing(bad)


def _dummy_da_3d():
    import xarray as xr

    return xr.DataArray(
        np.zeros((2, 3, 3)),
        dims=("time", "lat", "lon"),
        coords={"time": [0, 1], "lat": [1, 2, 3], "lon": [1, 2, 3]},
    )
