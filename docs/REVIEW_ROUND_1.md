# Review — Round 1

Reviewer: `ecc:mle-reviewer`, over commits `ebf583fd` (foundation), `b0899706`
(slice A / data), `b53f5cec` (slice B / features), `1bb1c371` (slice C / model).

Baseline reproduced: `ruff check nowcast tests` clean; `pytest -q` → **123 passed**.

All findings are against code validated on **synthetic data only** — no real
IMDAA/MERA/INSAT yet. Several blockers are precisely the places where synthetic
data hides a real-data bug.

## Target-label contract (decision — resolves F1 / F11)

1. `features.labels.build_labels` returns a **peak-normalised** smoothed
   occurrence probability: an isolated event cell reaches **1.0**, not ~0.16.
2. `config.LABEL_OCCURRENCE_THRESHOLD = 0.5` — a cell is a positive occurrence
   when `label >= threshold`.
3. `MultiTaskLoss` trains **BCE + Dice against the soft (probabilistic) target**.
4. **CSI/POD/FAR, the LightGBM baseline, and calibration** binarise the target
   at `LABEL_OCCURRENCE_THRESHOLD`.
5. `focal_loss` stays available; wire it into `MultiTaskLoss` as an option.

## Chronological split (decision — resolves F4)

`config` now exposes disjoint **date ranges** (`TRAIN_DATE_RANGE`,
`VAL_DATE_RANGE`, `TEST_DATE_RANGE`), not calendar-year tuples. `TRAIN_YEARS` /
`VAL_YEARS` / `TEST_YEARS` are **removed** — filter on the `time` coord.

---

## BLOCKING

| ID | Slice | File | Summary |
|----|-------|------|---------|
| **F1** | B (+C) | `features/labels.py:~177-189`; `training/metrics.py`; `uncertainty/calibration.py`; `baseline/train_lgbm.py:82` | Gaussian smoothing (kernel sums to 1) collapses isolated events to ~0.16 → every 0.5-binarised consumer sees **zero positives** for cloudburst & flash_flood; metrics/baseline/loss all degenerate. Fix per the label contract above (peak-normalise). |
| **F2** | C | `data/datamodule.py:70-75` | `NowcastDataset(terrain=None)` reads `schema.FLASH_FLOOD_EXTRA_CHANNELS` (incl. `flow_direction`) from the feature dataset, but `assemble_features` emits **only** the 20 `FEATURE_CHANNELS` — no `flow_direction`. Confirmed `KeyError` on the real path. Give the datamodule an explicit terrain source (DEM routing store / datacube static vars). |
| **F3** | C | `data/datamodule.py:88-97`; `training/train.py:41-45` | Real (non-synthetic) training path never builds/loads a `Normalizer` → raw `prmsl≈1e5`, `z≈1e5` into the ConvLSTM. Also terrain tensor is never normalised (`flow_accumulation` needs `log1p`; `flow_direction` is nominal). Add a train-years-only norm-stats step → `config.NORM_STATS_PATH`, load in `setup()`/`train.py`; normalise terrain. |
| **F4** | foundation/C | `config.py`; `data/datamodule.py:153-154` | `VAL_YEARS == TEST_YEARS == (2020,)` → val and test identical, no held-out set. Fixed in `config` (disjoint date ranges); datamodule + baseline must switch to date-range filtering. |
| **F5** | B | `baseline/train_lgbm.py:64-66`; `baseline/evaluate.py:97-99` | On an empty year-split the baseline **silently** falls back to the full dataset (train == eval, includes val/test). Raise a clear error instead; gate the degenerate case behind an explicit `--no-split` flag. |
| **F6** | A | `ingest/_util.py:30-38`; `ingest/names.py:62-68` | `accumulated_to_rate(cumulative=True)` does `diff("time").clip(min=0)` — at every forecast-cycle reset the negative diff is silently zeroed and the first step is emitted with no true-window division. IMDAA `APCP` default (`cumulative_accum=False`) is likely wrong. `precip` is the **only** label source → corrupts every label on real data. Detect resets, divide by the real interval, set correct per-source defaults. |
| **F7** | A | `pipelines/ingest_flow.py:67-88`; `ingest/datacube.py:86` | `xr.merge(..., join="exact")` will raise on INSAT's native half-hourly timestamps; precip reindex `method="nearest"` has **no tolerance** (stale precip onto gaps); IMERG half-hourly is never aggregated to `TIMESTEP="1h"`. Resample INSAT/IMERG to the datacube step, add reindex tolerance, align satellite time index. |

## ADVISORY (fix now where cheap; noted otherwise)

| ID | Slice | Summary | Plan |
|----|-------|---------|------|
| F8 | A | D8 compass labels inverted (topology still correct — offsets consistent everywhere). | Fix docstrings/`long_name`/test name; add numpy-vs-pysheds `flowdir` agreement test on a tilted-plane DEM. |
| F9 | A | Routing computed on the 0.1° (~11 km) grid → flash-flood "catchments" can't resolve real basins. | Make routing resolution configurable (`route_native_resolution`); document as a known limitation for the real DEM. |
| F10 | B | `_shift_occurrence` zero-pads the last 1–6 h of every block (served as samples) and shifts **before** the split → 2019 tail leaks 2020 occurrence into train. | Drop samples whose label horizon runs past the data or across a time gap; shift within each split. |
| F11 | B/C | Two target definitions (soft vs binarised). | Resolved by the label contract above; wire `focal_loss` into `MultiTaskLoss`. |
| F12 | A/B/C | Duplicated helpers (`_first_step_backfill`, `_grid_spacing_m`, `num_groups`). | Promote to `nowcast/common/` (A adds; B/C import). |
| F13 | A/B/C | Missing data silently → 0 precip / field-median / 0 m elevation, no validity mask. | A: emit a `valid_mask` in the datacube. B: thread it through `assemble`/`labels`. C: pass to `MultiTaskLoss(mask=…)`. |
| F14 | C | `TemperatureScaler` fits/scores against soft targets (should be hard labels / NLL). | Binarise targets at `LABEL_OCCURRENCE_THRESHOLD` for calibration + reliability. |
| F15 | C | `csi/pod/far/pr_auc` return `0.0` on degenerate cases — indistinguishable from a bad model. | Return `nan`; nan-aware aggregation; also log positive counts per (hazard, lead). |
| F16 | C | `from_synthetic` computes norm stats over train+val+test. | Compute from the train slice only. |
| F17 | B | 7-level IWV trapezoid omits the sub-1000 hPa boundary layer; IMDAA ships `tcwv` (PWAT). | Prefer `tcwv` when present; keep the integral as fallback. |
| F18 | — (orchestrator) | `mlruns/`, `.pytest-report.xml`, `nowcast.egg-info/` present in tree. | Confirm git-ignored (not tracked) — done. |

## Verdict

Architecture and slice boundaries are sound: schema-driven contract, synthetic
harness, model/loss/metric shapes, MetPy feature math (IWV sign/units,
convergence, bulk shear), ConvLSTM state handling, `stack_logits` hazard order,
and D8-table consistency between the numpy router and the label accumulator are
all correct. **Not yet safe to train on real data**: F1–F7 (label smoothing,
terrain wiring, real-path normalisation, split discipline, precip units, ingest
time-alignment) must land first. They are concentrated in the A↔B↔C seams, not
inside any one slice, and are a few days of coordinated work — no re-architecting.

## Cannot be assessed without real data

IMDAA/MERA `APCP` accumulation convention & window (F6); whether INSAT L1C HDF5
exposes `IMG_*_TEMP` or raw counts needing a LUT; pysheds routing branch + its
`dirmap` vs affine orientation (F8, pysheds broken in this env); whether 0.1°
routing yields usable catchments (F9); real event base rates vs the `config`
thresholds; CAPE/CIN/LI quality from 7 levels; INSAT/IMERG native time stepping
& correct 1 h aggregation (F7); real `flow_accumulation` magnitude range (F3).
