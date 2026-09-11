"""Evaluate the LightGBM pixel baseline.

Per (hazard, lead) it reports CSI, POD, FAR (contingency scores at a 0.5
threshold), PR-AUC, the Brier score and reliability-curve points, writing the
result to ``metrics.json`` in the model directory. A reliability plot is drawn
only when ``matplotlib`` is importable and ``plot=True``.
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

_BIN = float(config.LABEL_OCCURRENCE_THRESHOLD)
_N_BINS = 10


def _contingency(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float, float]:
    pred = y_prob >= _BIN
    obs = y_true >= _BIN
    tp = float(np.sum(pred & obs))
    fp = float(np.sum(pred & ~obs))
    fn = float(np.sum(~pred & obs))
    csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    pod = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    far = fp / (tp + fp) if (tp + fp) > 0 else 0.0
    return csi, pod, far


def _pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    obs = (y_true >= _BIN).astype(int)
    if obs.sum() == 0 or obs.sum() == obs.size:
        return 0.0
    try:
        from sklearn.metrics import average_precision_score

        return float(average_precision_score(obs, y_prob))
    except Exception:
        return 0.0


def _reliability(y_true: np.ndarray, y_prob: np.ndarray) -> list[dict[str, float]]:
    obs = (y_true >= _BIN).astype(float)
    edges = np.linspace(0.0, 1.0, _N_BINS + 1)
    points: list[dict[str, float]] = []
    for b in range(_N_BINS):
        lo, hi = float(edges[b]), float(edges[b + 1])
        if b < _N_BINS - 1:
            in_bin = (y_prob >= lo) & (y_prob < hi)
        else:
            in_bin = (y_prob >= lo) & (y_prob <= hi)
        if not np.any(in_bin):
            continue
        points.append(
            {
                "bin_lower": lo,
                "bin_upper": hi,
                "mean_pred": float(np.mean(y_prob[in_bin])),
                "mean_obs": float(np.mean(obs[in_bin])),
                "count": int(np.sum(in_bin)),
            }
        )
    return points


def _predict(entry: dict[str, Any], model_dir: Path, X: np.ndarray) -> np.ndarray:
    if entry["kind"] == "constant":
        return np.full(X.shape[0], float(entry["value"]), dtype="float64")
    import lightgbm as lgb

    booster = lgb.Booster(model_file=str(model_dir / entry["file"]))
    return np.asarray(booster.predict(X), dtype="float64").ravel()


def evaluate(
    model_dir: str | Path,
    features_path: str | Path,
    labels_path: str | Path,
    *,
    plot: bool = False,
    no_split: bool = False,
) -> dict[str, Any]:
    """Return ``{"targets": {target: metrics}}`` and write ``metrics.json``.

    Evaluates on ``config.VAL_DATE_RANGE`` by default; an empty slice raises.
    Pass ``no_split=True`` to score against the whole dataset (demo cubes).
    """
    model_dir = Path(model_dir)
    manifest = json.loads((model_dir / "manifest.json").read_text())

    features = open_features(features_path)
    labels = open_labels(labels_path)

    if no_split:
        val_ds = make_pixel_dataset(features, labels)
        if val_ds.X.shape[0] == 0:
            raise ValueError(
                "no rows after the label-horizon drop; dataset is shorter than "
                f"max(LEAD_TIMES_H)={max(config.LEAD_TIMES_H)} h"
            )
    else:
        _, val_ds = split_by_date_range(features, labels)
        if val_ds.X.shape[0] == 0:
            raise ValueError(
                f"no rows in VAL_DATE_RANGE={config.VAL_DATE_RANGE} for this "
                "dataset; pass --no-split (no_split=True) to evaluate on everything"
            )

    results: dict[str, Any] = {}
    for ti, target in enumerate(manifest["target_names"]):
        entry = manifest["models"][target]
        y_true = val_ds.y[:, ti].astype("float64")
        y_prob = np.clip(_predict(entry, model_dir, val_ds.X), 0.0, 1.0)
        csi, pod, far = _contingency(y_true, y_prob)
        obs_bin = (y_true >= _BIN).astype(float)
        results[target] = {
            "csi": csi,
            "pod": pod,
            "far": far,
            "pr_auc": _pr_auc(y_true, y_prob),
            "brier": float(np.mean((y_prob - obs_bin) ** 2)),
            "n": int(y_true.size),
            "obs_pos_rate": float(np.mean(obs_bin)),
            "reliability": _reliability(y_true, y_prob),
        }

    metrics = {"targets": results}
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if plot:
        _maybe_plot(results, model_dir)
    return metrics


def _maybe_plot(results: dict[str, Any], out_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    for target, res in results.items():
        pts = res["reliability"]
        if pts:
            ax.plot(
                [p["mean_pred"] for p in pts],
                [p["mean_obs"] for p in pts],
                marker="o",
                label=target,
            )
    ax.set_xlabel("forecast probability")
    ax.set_ylabel("observed frequency")
    ax.set_title("Reliability")
    fig.savefig(out_dir / "reliability.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: ``python -m nowcast.baseline.evaluate ...``."""
    parser = argparse.ArgumentParser(description="Evaluate the LightGBM pixel baseline.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="evaluate on the whole dataset instead of config.VAL_DATE_RANGE",
    )
    args = parser.parse_args(argv)

    metrics = evaluate(
        args.model_dir,
        args.features,
        args.labels,
        plot=args.plot,
        no_split=args.no_split,
    )
    print(json.dumps(metrics["targets"], indent=2))


if __name__ == "__main__":
    main()
