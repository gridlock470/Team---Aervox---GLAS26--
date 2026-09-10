"""Tests for LightGBM baseline training and evaluation."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from nowcast import schema
from nowcast.features.labels import build_labels
from nowcast.testing import synthetic

_HAS_METPY = importlib.util.find_spec("metpy") is not None
_HAS_LGBM = importlib.util.find_spec("lightgbm") is not None
pytestmark = pytest.mark.skipif(
    not (_HAS_METPY and _HAS_LGBM), reason="metpy and lightgbm are required"
)

if _HAS_METPY and _HAS_LGBM:
    from nowcast.baseline.evaluate import evaluate
    from nowcast.baseline.train_lgbm import train
    from nowcast.features.assemble import assemble_features, write_features
    from nowcast.features.labels import write_labels


@pytest.fixture(scope="module")
def artifact_paths(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("baseline")
    cube = synthetic.make_datacube(n_hours=6, seed=7).isel(
        lat=slice(0, 10), lon=slice(0, 10)
    )
    precip = cube["precip"].values.copy()
    precip[1:5, 2:9, 2:9] = 90.0  # strong, wide, persistent -> positives after smoothing
    cube["precip"] = (("time", "lat", "lon"), precip)

    features = assemble_features(cube)
    labels = build_labels(cube)
    features_path = write_features(features, workdir / "features.zarr")
    labels_path = write_labels(labels, workdir / "labels.nc")
    return workdir, features_path, labels_path


def test_train_produces_model_files(artifact_paths):
    workdir, features_path, labels_path = artifact_paths
    out_dir = workdir / "models"
    manifest = train(features_path, labels_path, out_dir, n_per_time=None)

    assert (out_dir / "manifest.json").exists()
    assert manifest["target_names"]
    assert manifest["feature_names"] == list(schema.FEATURE_CHANNELS)
    assert list(out_dir.glob("*.txt")), "expected at least one booster file"


def test_evaluate_returns_finite_metrics(artifact_paths):
    workdir, features_path, labels_path = artifact_paths
    out_dir = workdir / "models"
    train(features_path, labels_path, out_dir, n_per_time=None)
    metrics = evaluate(out_dir, features_path, labels_path)

    assert (out_dir / "metrics.json").exists()
    assert metrics["targets"]
    for target, values in metrics["targets"].items():
        for key in ("csi", "pod", "far", "pr_auc", "brier"):
            assert np.isfinite(values[key]), (target, key)
            assert values[key] >= 0.0
        assert isinstance(values["reliability"], list)
