"""Shared pytest fixtures — all backed by :mod:`nowcast.testing.synthetic`.

No test in this suite requires real data. When real IMDAA / MERA / INSAT
files land, add integration tests under ``tests/integration/`` guarded by a
``@pytest.mark.integration`` marker instead of changing these.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nowcast.testing import synthetic


@pytest.fixture
def synthetic_datacube():
    """A schema-valid raw datacube, 48 hours."""
    return synthetic.make_datacube(n_hours=48, seed=0)


@pytest.fixture
def synthetic_feature_datacube():
    """A datacube that also carries every ``FEATURE_CHANNELS`` name."""
    return synthetic.make_feature_datacube(n_hours=48, seed=0)


@pytest.fixture
def synthetic_dem():
    """A synthetic elevation DataArray with a drainage valley."""
    return synthetic.make_dem(seed=0)


@pytest.fixture
def synthetic_targets():
    """Schema-valid target probability maps."""
    return synthetic.make_targets(n_hours=48, seed=0)


@pytest.fixture
def synthetic_sample():
    """One ``(x, y)`` training pair with the contract shapes."""
    return synthetic.make_sample(seed=0)


@pytest.fixture
def tmp_zarr(tmp_path: Path) -> Path:
    """A path for a throwaway Zarr store inside pytest's tmp dir."""
    return tmp_path / "store.zarr"
