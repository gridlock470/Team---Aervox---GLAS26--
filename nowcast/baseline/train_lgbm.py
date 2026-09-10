"""Train one LightGBM classifier per (hazard, lead) on the pixel baseline.

Targets are binarised at 0.5. Boosters are saved as ``<target>.txt`` alongside a
``manifest.json`` describing feature order, params and per-target model kind.
MLflow logging is used when the package is importable, otherwise skipped.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from nowcast import config
from nowcast.baseline.dataset import make_pixel_dataset, split_by_year
from nowcast.features.assemble import open_features
from nowcast.features.labels import open_labels

_BIN_THRESHOLD = 0.5
_DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 100,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": config.RANDOM_SEED,
    "n_jobs": 1,
    "verbosity": -1,
}


def _try_mlflow():
    try:
        import mlflow

        return mlflow
    except Exception:
        return None


def train(
    features_path: str | Path,
    labels_path: str | Path,
    out_dir: str | Path,
    *,
    n_per_time: int | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit a classifier per (hazard, lead) and persist boosters + manifest.

    Returns the manifest dict.
    """
    import lightgbm as lgb

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    features = open_features(features_path)
    labels = open_labels(labels_path)

    train_ds, _ = split_by_year(features, labels, n_per_time=n_per_time)
    if train_ds.X.shape[0] == 0:
        train_ds = make_pixel_dataset(features, labels, n_per_time=n_per_time)

    lgb_params = dict(_DEFAULT_PARAMS)
    if params:
        lgb_params.update(params)

    mlflow = _try_mlflow()
    if mlflow is not None:
        mlflow.start_run()
        try:
            mlflow.log_params(lgb_params)
        except Exception:
            pass

    models: dict[str, Any] = {}
    for ti, target in enumerate(train_ds.target_names):
        y = (train_ds.y[:, ti] >= _BIN_THRESHOLD).astype(int)
        pos_rate = float(y.mean()) if y.size else 0.0
        if np.unique(y).size < 2:
            models[target] = {"kind": "constant", "value": pos_rate, "target": target}
        else:
            clf = lgb.LGBMClassifier(**lgb_params)
            clf.fit(train_ds.X, y)
            model_file = f"{target}.txt"
            clf.booster_.save_model(str(out_dir / model_file))
            models[target] = {
                "kind": "booster",
                "file": model_file,
                "positive_rate": pos_rate,
                "target": target,
            }
            if mlflow is not None:
                try:
                    mlflow.log_metric(f"{target}__train_pos_rate", pos_rate)
                except Exception:
                    pass

    manifest = {
        "feature_names": train_ds.feature_names,
        "target_names": train_ds.target_names,
        "hazards": list(config.HAZARDS),
        "leads": list(config.LEAD_TIMES_H),
        "bin_threshold": _BIN_THRESHOLD,
        "lgb_params": lgb_params,
        "n_train_rows": int(train_ds.X.shape[0]),
        "models": models,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

    if mlflow is not None:
        try:
            mlflow.log_artifact(str(manifest_path))
        except Exception:
            pass
        finally:
            mlflow.end_run()

    return manifest


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: ``python -m nowcast.baseline.train_lgbm ...``."""
    parser = argparse.ArgumentParser(description="Train the LightGBM pixel baseline.")
    parser.add_argument("--features", required=True, help="path to the feature Zarr store")
    parser.add_argument("--labels", required=True, help="path to the labels NetCDF file")
    parser.add_argument("--out-dir", required=True, help="directory for boosters + manifest")
    parser.add_argument("--n-per-time", type=int, default=None)
    args = parser.parse_args(argv)

    manifest = train(
        args.features, args.labels, args.out_dir, n_per_time=args.n_per_time
    )
    print(json.dumps(manifest["models"], indent=2, default=str))


if __name__ == "__main__":
    main()
