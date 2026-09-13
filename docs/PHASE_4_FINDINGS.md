# Phase 4 — diagnosing the cloudburst/flash-flood val_csi=0.0 result

## Starting point

Phase 3 (`runs/phase3_final`, 25 epochs, real 2018 data) finished with overall
`val_csi=0.025`, but:

- `val_csi/cloudburst = 0.0`
- `val_csi/flash_flood = 0.0`
- `val_csi/thunderstorm ≈ 0.03` (weak but real)

The question this phase answers: is that a fixable threshold/weighting
problem, or something more fundamental?

## Root cause: too few positive examples to *evaluate*, not just to train on

Both already-tried fixes fail identically:

- **Threshold sweeping** already exists (`nowcast/training/metrics.py`'s
  `best_csi()`), computed per hazard. It doesn't rescue cloudburst/
  flash_flood — there's no threshold at which they show real skill.
- **Loss-weighting ablation** (`runs/phase2_flat` vs `runs/phase2_invfreq`,
  documented directly in `runs/phase3_final/convlstm_real.yaml`'s own
  comments): inverse-frequency hazard weighting fit cloudburst/flash_flood
  *far* better on train (CSI 0.71/0.77 vs flat's 0.52/0.59) but produced the
  same `val_csi=0.0` as flat weighting — the textbook signature of
  overfitting a handful of event windows, and it actively hurt thunderstorm's
  real signal in the process (`val_csi` 0.026→0.003). Flat weighting was
  correctly kept.

The actual constraint is data volume. `nowcast/config.py`'s own inline
comment states it plainly: cloudburst fires on **8 dates** and flash_flood on
**3 dates** in the entire 2018 record. With `VAL_DATE_RANGE = 2018-08-06 to
08-12`, validation contains only **2 cloudburst dates and 1 flash_flood
date** — windowed (`INPUT_SEQ_LEN=12`, leads to 6h), that's a single-digit
count of positive validation windows. `config.py` already says outright:
*"one or two dates per rare hazard per split is token coverage, not a
statistically meaningful held-out score."* `val_csi=0.0` here reads as noise
from an underpowered evaluation, not a well-powered null result.

**Widening `VAL_DATE_RANGE` to catch more of the rare-event dates was
considered and rejected.** The remaining cloudburst/flash_flood dates cluster
right at the existing VAL/TEST boundary (08-11 through 08-17). Moving that
boundary *after already knowing where the positive dates are* is data
snooping — the split stays as originally reviewed and decided (F4, Round 1).

## Fix landed this phase: a real diagnostic gap, not a modeling gap

`NowcastMetrics.compute()` (`nowcast/training/metrics.py`) already computed
per-hazard `csi_best/<hazard>` and `csi_best_threshold/<hazard>`, but
`LitNowcast._epoch_end` (`nowcast/training/lit_module.py`) only logged the
*aggregate* versions — the per-hazard values were silently dropped before
reaching `metrics.csv`. Per-hazard `frequency_bias` wasn't computed at all
(only the cross-hazard aggregate). Until now, there was no way to tell from a
training run whether cloudburst/flash_flood were over- or under-firing, or
whether any threshold helped them, versus just seeing `0.0` on the headline
number.

Changed:
- `metrics.py`: `NowcastMetrics._core()` now also computes
  `frequency_bias/<hazard>` per hazard (mirrors the existing `csi/<hazard>`
  pattern).
- `lit_module.py`: `_epoch_end`'s per-hazard logging loop now also logs
  `csi_best/<hazard>`, `csi_best_threshold/<hazard>`, and
  `frequency_bias/<hazard>` for every hazard, every epoch.
- Verified against `tests/test_training_metrics.py` and
  `tests/test_lit_module.py` (22 tests, all green) before running anything.

A short (7-epoch, matching the Phase 2 ablation's own precedent) diagnostic
run, `runs/phase4_diag`, was launched with the fix to actually read these new
per-hazard numbers — not to seek a better final model, since Phase 3's full
25-epoch run already showed this exact config's cloudburst/flash_flood result
holds throughout training.

**`runs/phase4_diag` final-epoch (epoch 6/7) per-hazard numbers** (real run,
`runs/phase4_diag/lightning_logs/version_0/metrics.csv`):

| hazard | csi | csi_best | csi_best_threshold | frequency_bias |
|---|---|---|---|---|
| thunderstorm | 0.046 | 0.082 | 0.147 | 7.87 |
| cloudburst | 0.0 | 0.0 | **0.01** (sweep floor) | **0.0** |
| flash_flood | 0.0 | 0.0 | **0.01** (sweep floor) | **0.0** |

This is exactly the diagnosis the logging fix was meant to surface, now
confirmed numerically rather than inferred: cloudburst/flash_flood aren't
just weak at the default 0.5 threshold — `csi_best` is **0.0 even at 0.01**,
the lowest threshold in the entire sweep grid (`DEFAULT_CSI_THRESHOLDS`,
`metrics.py:31-33`), and `frequency_bias=0.0` (not `nan`) means the model
is predicting **zero positives for these two hazards at any threshold that
matters**, on real observed-positive batches (a `nan` would mean no
positives were ever observed to score against — this is a real, measured
zero-detection result). Thunderstorm, by contrast, has a real optimal
threshold (0.147, well below the naive 0.5) and a frequency_bias of 7.87 —
over-firing roughly 8x, but firing, which is a fixable calibration
direction rather than a null result. This matches the root-cause diagnosis
above precisely: there's no probability threshold that rescues cloudburst/
flash_flood on this data, because the model never learned to predict them
as anything other than absent.

## Recommendation

Do not spend further effort tuning loss weights, thresholds, or
architecture to raise cloudburst/flash_flood CSI on 2018-only data — two
independent weighting schemes already converged on the same null result, and
the diagnostic fix above confirms *why*: there isn't enough held-out signal
to measure, let alone improve. The real unlock is more independent
event-years (2019/2020 IMDAA ingestion — order already submitted and running
on RDS per `[[sih-build-status]]`). Until that lands:

- Report **thunderstorm** as the console's only hazard with a statistically
  meaningful held-out score.
- Keep cloudburst/flash_flood visible in the UI (they're real, documented
  physical hazards this system is built to eventually cover) but caveat
  their current skill explicitly wherever it's surfaced — the same honesty
  convention already used for the DGMR integration and the Model panel's
  "diagnostics are placeholder values" note.
