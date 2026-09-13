# Phase 5 — calibrating the real Phase-3 model

## What ran

`scripts/calibrate_model.py` loaded the real trained checkpoint
(`runs/phase3_final/lightning_logs/version_1/checkpoints/
epoch=24-step=18950.ckpt`), ran it forward over real held-out data (151 VAL
windows, 1157 TEST windows), fit a single global `TemperatureScaler`
(`nowcast/uncertainty/calibration.py`, previously only exercised against
synthetic 1-D tensors) on VAL logits, and evaluated expected calibration
error on TEST before/after — the standard Guo et al. 2017 split, so TEST was
never seen by the fit. Full numbers in `runs/phase3_final/calibration.json`.

## Result

- **Fitted temperature: 1.3527** (>1, i.e. it softens the model's
  probabilities toward 0.5 — the direction you'd expect if the model is
  mildly overconfident).
- **Aggregate TEST ECE: 0.0046 before → 0.0068 after** — calibration made it
  *worse*, not better.
- **Per-hazard TEST ECE**: thunderstorm 0.0138 → 0.0206 (worse); cloudburst
  and flash_flood are 0.0000 → 0.0000 (no change) — both already-degenerate,
  not a real "well calibrated" result. At their base rates the model
  predicts near-zero everywhere, which trivially matches a near-zero true
  frequency almost everywhere; this is the same "predict nothing" pattern
  flagged in `docs/PHASE_4_FINDINGS.md`, not evidence of good calibration.

## Why calibration didn't help here — and why that's not surprising given Phase 4

This is a genuine, honest result, not a bug: I didn't spin it into a success.
Two things are true at once:

1. **Aggregate ECE was already small before calibration** (0.0046) — the
   model isn't badly miscalibrated to begin with, so there wasn't much room
   for temperature scaling to improve on, and a small fit-set noise can move
   the number the wrong way.
2. **The 151-window VAL split and the 1157-window TEST split cover
   different calendar periods** (VAL: Aug 6-12; TEST: Aug 13-Sep 30) that
   Phase 4 already established have different rare-hazard base rates (the
   "monsoon-intensification" jump `nowcast/config.py` documents). A
   temperature fit on VAL optimizes for VAL's specific mix of conditions,
   which doesn't necessarily transfer to TEST's different mix. This is the
   same root cause as Phase 4's finding, showing up in a different metric:
   **small, non-stationary held-out splits limit what any post-hoc
   statistical correction — loss-weighting in Phase 2, calibration here —
   can reliably achieve**, independent of the correction method.

## Recommendation

Don't chase a "fix" for this by trying more calibration variants (per-lead
temperatures, Platt scaling, etc.) on the same small splits — for the same
reason Phase 4 recommended against further loss-weighting tuning. The
model's raw probabilities are already close to as calibrated as this data
can tell us; re-run this exact script once more IMDAA event-years land
(2019/2020, per `[[sih-build-status]]`) to see whether a bigger, more
representative VAL split changes the answer.
