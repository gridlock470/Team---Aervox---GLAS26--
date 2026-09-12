"""Tests for LightGBM baseline training and evaluation."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from nowcast import config, schema
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


def _heavy_rain_cube(n_hours, start, rain_windows):
    cube = synthetic.make_datacube(n_hours=n_hours, seed=7).isel(
        lat=slice(0, 8), lon=slice(0, 8)
    )
    cube = cube.assign_coords(
        time=pd.date_range(start, periods=n_hours, freq="1h")
    )
    precip = cube["precip"].values.copy()
    for lo, hi in rain_windows:
        precip[lo:hi, 1:7, 1:7] = 90.0  # wide, persistent -> positives after smoothing
    cube["precip"] = (("time", "lat", "lon"), precip)
    return cube


@pytest.fixture(scope="module")
def split_paths(tmp_path_factory):
    """Cube straddling the configured train/val boundary.

    The start is derived from ``config.VAL_DATE_RANGE`` rather than hard-coded,
    so the fixture keeps straddling the boundary wherever the splits are set.
    """
    workdir = tmp_path_factory.mktemp("baseline_split")
    boundary_start = (
        pd.Timestamp(config.VAL_DATE_RANGE[0]) - pd.Timedelta(days=1)
    ).strftime("%Y-%m-%dT%H:%M")
    cube = _heavy_rain_cube(
        54, boundary_start, rain_windows=[(1, 6), (26, 32)]
    )
    features = assemble_features(cube)
    labels = build_labels(cube)
    return (
        workdir,
        write_features(features, workdir / "features.zarr"),
        write_labels(labels, workdir / "labels.nc"),
    )


@pytest.fixture(scope="module")
def demo_paths(tmp_path_factory):
    """Single-period cube (synthetic default 2018 dates) for the --no-split path."""
    workdir = tmp_path_factory.mktemp("baseline_demo")
    cube = _heavy_rain_cube(16, "2018-05-02T00:00", rain_windows=[(1, 7)])
    features = assemble_features(cube)
    labels = build_labels(cube)
    return (
        workdir,
        write_features(features, workdir / "features.zarr"),
        write_labels(labels, workdir / "labels.nc"),
    )


def test_train_on_date_range_produces_model_files(split_paths):
    workdir, features_path, labels_path = split_paths
    out_dir = workdir / "models"
    manifest = train(features_path, labels_path, out_dir, n_per_time=None)

    assert (out_dir / "manifest.json").exists()
    assert manifest["split"] == "train_date_range"
    assert manifest["feature_names"] == list(schema.FEATURE_CHANNELS)
    assert list(out_dir.glob("*.txt")), "expected at least one booster file"


def test_evaluate_on_val_range_returns_finite_metrics(split_paths):
    workdir, features_path, labels_path = split_paths
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


def test_train_raises_and_writes_nothing_when_split_is_empty(demo_paths, tmp_path):
    """F5: a cube outside TRAIN_DATE_RANGE must not silently train on everything."""
    _, features_path, labels_path = demo_paths
    # relabel the demo cube's time to 2021 (outside every configured range)
    from nowcast.features.assemble import open_features
    from nowcast.features.labels import open_labels

    feats = open_features(features_path).assign_coords(
        time=pd.date_range("2021-05-02T00:00", periods=16, freq="1h")
    )
    labs = open_labels(labels_path).assign_coords(
        time=pd.date_range("2021-05-02T00:00", periods=16, freq="1h")
    )
    fp = write_features(feats, tmp_path / "f2021.zarr")
    lp = write_labels(labs, tmp_path / "l2021.nc")
    out_dir = tmp_path / "models_should_not_exist"

    with pytest.raises(ValueError, match="TRAIN_DATE_RANGE"):
        train(fp, lp, out_dir, n_per_time=None)
    assert not (out_dir / "manifest.json").exists()

    # --no-split rescues the degenerate/demo case
    manifest = train(fp, lp, out_dir, n_per_time=None, no_split=True)
    assert manifest["split"] == "none"
    assert (out_dir / "manifest.json").exists()


def test_evaluate_no_split_scores_single_period_cube(demo_paths):
    workdir, features_path, labels_path = demo_paths
    out_dir = workdir / "models"
    train(features_path, labels_path, out_dir, n_per_time=None, no_split=True)

    with pytest.raises(ValueError, match="VAL_DATE_RANGE"):
        evaluate(out_dir, features_path, labels_path)

    metrics = evaluate(out_dir, features_path, labels_path, no_split=True)
    assert metrics["targets"]
    assert all(
        np.isfinite(v["brier"]) for v in metrics["targets"].values()
    )


def test_labels_binarise_at_config_threshold(split_paths):
    _, features_path, labels_path = split_paths
    from nowcast.features.labels import open_labels

    labels = open_labels(labels_path)
    cloudburst = labels.sel(hazard="cloudburst")
    # peak-normalisation (F1) means the injected block clears the 0.5 threshold
    assert float(cloudburst.max()) >= config.LABEL_OCCURRENCE_THRESHOLD
