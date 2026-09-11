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
from nowcast.baseline.dataset import make_pixel_dataset, split_by_date_range
from nowcast.features.assemble import open_features
from nowcast.features.labels import open_labels

_BIN_THRESHOLD = float(config.LABEL_OCCURRENCE_THRESHOLD)
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
    no_split: bool = False,
) -> dict[str, Any]:
    """Fit a classifier per (hazard, lead) and persist boosters + manifest.

    By default the model trains on ``config.TRAIN_DATE_RANGE`` only; an empty
    slice raises. Pass ``no_split=True`` to train on the whole dataset (demo /
    single-period cubes). Returns the manifest dict.
    """
    import lightgbm as lgb

    features = open_features(features_path)
    labels = open_labels(labels_path)

    if no_split:
        train_ds = make_pixel_dataset(features, labels, n_per_time=n_per_time)
        if train_ds.X.shape[0] == 0:
            raise ValueError(
                "no rows after the label-horizon drop; dataset is shorter than "
                f"max(LEAD_TIMES_H)={max(config.LEAD_TIMES_H)} h"
            )
    else:
        train_ds, _ = split_by_date_range(features, labels, n_per_time=n_per_time)
        if train_ds.X.shape[0] == 0:
            raise ValueError(
                f"no rows in TRAIN_DATE_RANGE={config.TRAIN_DATE_RANGE} for this "
                "dataset; pass --no-split (no_split=True) to train on everything"
            )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
        "split": "none" if no_split else "train_date_range",
        "train_date_range": list(config.TRAIN_DATE_RANGE),
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
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="train on the whole dataset instead of config.TRAIN_DATE_RANGE",
    )
    args = parser.parse_args(argv)

    manifest = train(
        args.features,
        args.labels,
        args.out_dir,
        n_per_time=args.n_per_time,
        no_split=args.no_split,
    )
    print(json.dumps(manifest["models"], indent=2, default=str))


if __name__ == "__main__":
    main()
