"""Phase 5: fit temperature scaling on the real trained model.

    python scripts/calibrate_model.py

Loads the real Phase-3 checkpoint, runs it forward on real held-out data
(VAL to fit, TEST to evaluate -- the standard Guo et al. 2017 split, so the
reported improvement is on data the temperature was never fit against),
fits a single global `TemperatureScaler` (one scalar for all hazards/leads --
see nowcast/uncertainty/calibration.py; cloudburst/flash_flood have only 1-2
event dates in val/test, per nowcast/config.py, far too few to fit a separate
per-hazard scalar to), and reports before/after expected calibration error.

Same honesty note as docs/PHASE_4_FINDINGS.md: thunderstorm is the only
hazard with enough held-out events for its ECE number to mean anything;
cloudburst/flash_flood's per-hazard ECE is reported anyway, but caveat it the
same way.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nowcast import config  # noqa: E402
from nowcast.training.lit_module import LitNowcast  # noqa: E402
from nowcast.training.train import build, load_config  # noqa: E402
from nowcast.uncertainty.calibration import (  # noqa: E402
    TemperatureScaler,
    expected_calibration_error,
)

CONFIG_PATH = Path("runs/phase3_final/convlstm_real.yaml")
CHECKPOINT_PATH = Path(
    "runs/phase3_final/lightning_logs/version_1/checkpoints/epoch=24-step=18950.ckpt"
)
OUT_JSON = Path("runs/phase3_final/calibration.json")


def collect_logits(model: LitNowcast, loader) -> tuple[torch.Tensor, torch.Tensor]:
    """Run `model` over every batch in `loader`, return concatenated (logits, targets)."""
    all_logits, all_targets = [], []
    with torch.no_grad():
        for x, y, terrain in loader:
            out = model.model(x, terrain)
            logits = model.model.stack_logits(out)
            all_logits.append(logits)
            all_targets.append(y)
    return torch.cat(all_logits, dim=0), torch.cat(all_targets, dim=0)


def per_hazard_ece(probs: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    return {
        hazard: expected_calibration_error(probs[:, i], targets[:, i])
        for i, hazard in enumerate(config.HAZARDS)
    }


def main() -> int:
    print(f"loading config: {CONFIG_PATH}")
    cfg = load_config(CONFIG_PATH)
    datamodule, _ = build(cfg)  # discard the freshly-initialised module; real weights load below
    datamodule.setup()

    print(f"loading real trained checkpoint: {CHECKPOINT_PATH}")
    model = LitNowcast.load_from_checkpoint(str(CHECKPOINT_PATH), map_location="cpu")
    model.eval()

    print("running forward pass over VAL (fit split)...")
    val_logits, val_targets = collect_logits(model, datamodule.val_dataloader())
    print(f"  VAL: {val_logits.shape[0]} windows")

    print("running forward pass over TEST (evaluation split)...")
    test_logits, test_targets = collect_logits(model, datamodule.test_dataloader())
    print(f"  TEST: {test_logits.shape[0]} windows")

    test_probs_before = torch.sigmoid(test_logits)
    ece_before = expected_calibration_error(test_probs_before, test_targets)
    ece_before_per_hazard = per_hazard_ece(test_probs_before, test_targets)

    print("fitting TemperatureScaler on VAL logits...")
    scaler = TemperatureScaler().fit(val_logits, val_targets)
    temperature = float(scaler.temperature)
    print(f"  fitted temperature: {temperature:.4f}")

    test_probs_after = torch.sigmoid(scaler(test_logits).detach())
    ece_after = expected_calibration_error(test_probs_after, test_targets)
    ece_after_per_hazard = per_hazard_ece(test_probs_after, test_targets)

    print(f"\nTEST ECE before calibration: {ece_before:.4f}")
    print(f"TEST ECE after calibration:  {ece_after:.4f}")
    print("\nPer-hazard TEST ECE (thunderstorm is the only one with enough")
    print("held-out events for this number to be statistically meaningful --")
    print("cloudburst/flash_flood have 1-2 event dates in TEST, see")
    print("nowcast/config.py and docs/PHASE_4_FINDINGS.md):")
    for hazard in config.HAZARDS:
        before_h = ece_before_per_hazard[hazard]
        after_h = ece_after_per_hazard[hazard]
        print(f"  {hazard}: before={before_h:.4f}  after={after_h:.4f}")

    result = {
        "checkpoint": str(CHECKPOINT_PATH),
        "n_val_windows": int(val_logits.shape[0]),
        "n_test_windows": int(test_logits.shape[0]),
        "temperature": temperature,
        "ece_before": ece_before,
        "ece_after": ece_after,
        "ece_before_per_hazard": ece_before_per_hazard,
        "ece_after_per_hazard": ece_after_per_hazard,
        "caveat": "cloudburst/flash_flood have only 1-2 event dates in TEST; "
                  "their per-hazard ECE is not statistically meaningful yet.",
    }
    OUT_JSON.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
