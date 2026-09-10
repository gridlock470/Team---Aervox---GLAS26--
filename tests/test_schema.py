"""Contract tests for :mod:`nowcast.schema`."""

from __future__ import annotations

import pytest

from nowcast import config, schema
from nowcast.testing import synthetic


def test_feature_channels_unique():
    assert len(schema.FEATURE_CHANNELS) == len(set(schema.FEATURE_CHANNELS))


def test_static_channels_subset_of_feature_channels():
    assert set(schema.STATIC_CHANNELS) <= set(schema.FEATURE_CHANNELS)


def test_sample_shapes_match_config():
    assert schema.INPUT_SHAPE == (
        len(schema.FEATURE_CHANNELS),
        config.INPUT_SEQ_LEN,
        *config.GRID_SHAPE,
    )
    assert schema.TARGET_SHAPE == (
        len(config.HAZARDS),
        len(config.LEAD_TIMES_H),
        *config.GRID_SHAPE,
    )


def test_validate_datacube_accepts_synthetic():
    ds = synthetic.make_datacube(n_hours=12, seed=1)
    schema.validate_datacube(ds, require_satellite=True)


def test_validate_datacube_rejects_missing_var(synthetic_datacube):
    bad = synthetic_datacube.drop_vars("t2m")
    with pytest.raises(schema.SchemaError):
        schema.validate_datacube(bad)


def test_validate_datacube_rejects_wrong_grid():
    ds = synthetic.make_datacube(n_hours=6, seed=2).isel(lat=slice(0, 10))
    with pytest.raises(schema.SchemaError):
        schema.validate_datacube(ds)


def test_validate_sample_accepts_contract_shapes():
    x, y = synthetic.make_sample(seed=3)
    schema.validate_sample(x, y)


def test_validate_sample_rejects_bad_shape():
    x, y = synthetic.make_sample(seed=4)
    with pytest.raises(schema.SchemaError):
        schema.validate_sample(x[:-1], y)


def test_validate_sample_rejects_out_of_range_target():
    x, y = synthetic.make_sample(seed=5)
    y = y.copy()
    y[0, 0, 0, 0] = 2.5
    with pytest.raises(schema.SchemaError):
        schema.validate_sample(x, y)


def test_targets_in_unit_interval():
    tgt = synthetic.make_targets(n_hours=8, seed=6)
    assert float(tgt.min()) >= 0.0
    assert float(tgt.max()) <= 1.0
    assert tuple(tgt.dims) == schema.TARGET_DIMS
