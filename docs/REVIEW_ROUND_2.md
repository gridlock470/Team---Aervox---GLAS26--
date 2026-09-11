# Review — Round 2

Reviewer: `ecc:mle-reviewer` (same instance as round 1), over commits
`477e7b28`/`9da70143`/`5a600bb1`/`2259159b` (round-1 fixes).
Baseline reproduced: `ruff check nowcast tests` clean; `pytest -q` → **149 passed**.

## F1–F18 status

| ID | Status | Note |
|---|---|---|
| F1 | CLOSED | Verified numerically: isolated event now peaks at 1.0. |
| F2 | CLOSED | Verified live against real `assemble_features()` output; `terrain=None` fallback returns `(2,H,W)`, no `KeyError`. |
| F3 | CLOSED (advisory residual) | `resolve_norm_stats`/`transform_terrain` wired; residual: the raw `flow_accumulation` **feature channel** (shared backbone) still gets plain z-score, not `log1p`, unlike its dedicated terrain-plane copy. |
| F4 | CLOSED (advisory residual) | Disjoint date ranges + raise-on-overlap in baseline; `NowcastDataModule._assert_disjoint` only checks VAL-vs-TEST, not TRAIN-vs-others (low risk — static constants). |
| F5 | CLOSED | Explicit `no_split` required in both CLIs; raises otherwise. |
| F6 | CLOSED | Verified numerically with a synthetic reset sawtooth. |
| F7 | CLOSED | Tolerance-bounded time alignment in `ingest/datacube.py` + `pipelines/ingest_flow.py`. |
| F8 | CLOSED | Docstrings corrected; array-space-vs-compass-name caveat documented. |
| F9 | OPEN (deferred) | Routing still computed post-regrid at 0.1°; unchanged, no regression. |
| F10 | **SUPERSEDED by F19** | Fixed correctly in the baseline; missed in the deep-model datamodule. |
| F11 | CLOSED | One documented threshold used consistently; BCE+Dice intentionally train on the soft target — coherent, not accidental. `focal_loss` implemented but unused (advisory, dead code). |
| F12 | OPEN (deferred) | Duplicated helpers still present, unchanged. |
| F13 | OPEN (deferred) | Silent zero/median fill unchanged; gaps now at least surface as NaN first (neutral-to-better) — doesn't close it. |
| F14 | CLOSED | Hard-binarised targets for calibration. |
| F15 | CLOSED | `nan` not `0.0` on degenerate cases; `n_pos/...` counters added. Trivial residual: `expected_calibration_error` still `0.0` on the empty-tensor edge case. |
| F16 | CLOSED | Train-only norm stats on both the synthetic and real paths. |
| F17 | CLOSED | `iwv` prefers `tcwv`, falls back to the pressure integral. |
| F18 | OPEN (housekeeping) | `.pytest-report.xml`/`mlruns/`/`nowcast.egg-info/` confirmed **not tracked** by git (gitignored); directories exist locally only, harmless. |

## NEW BLOCKING — F19

**Slice:** C (`nowcast/data/datamodule.py`) vs the F10 fix (B, `nowcast/features/labels.py`)

`valid_label_times()` (built for F10) is imported and used **only** in
`nowcast/baseline/dataset.py`. Zero references in `nowcast/data/datamodule.py`.
Reproduced directly: a synthetic heavy-rain event placed just after a
hypothetical train/val boundary produces a **train-side sample with label
1.0** at `lead=6` reaching into "val", while `valid_label_times` recomputed on
the correctly-sliced train series correctly flags that timestep invalid. The
Lightning/ConvLSTM path — the primary model — leaks future occurrence across
the chronological split boundary and serves fabricated zero-labels at each
split's tail. This inflates VAL/TEST CSI/PR-AUC in a way that won't be visible
without checking real event dates by hand.

**Fix:** `NowcastDataset.__init__` / `NowcastDataModule._subset` must recompute
`valid_label_times` on the **post-slice** `time` coordinate and exclude
invalid window-start indices from `__len__`/`__getitem__`, mirroring
`baseline/dataset.py:make_pixel_dataset` (which already does this correctly).

**Verify:** reproduce the round-2 repro (heavy-rain event at a split boundary)
through `NowcastDataset` directly (not `build_labels` alone) and assert the
leaking window index is absent — `__len__` shrinks accordingly.

## NEW ADVISORY — F20

`nowcast/data/transforms.py:resolve_norm_stats` — load-if-exists with no
version/hash key. A stale `norm_stats.json` from a prior experiment (different
`FEATURE_CHANNELS`, `TRAIN_DATE_RANGE`, or re-ingested cube) is silently
reused. Not blocking for a single-experiment prototype; will bite once two
configs are compared. Fix: key the cache by a hash of
(`FEATURE_CHANNELS`, `TRAIN_DATE_RANGE`, a data-version string) or add an
explicit `force_recompute` flag; log the cache file's mtime when reused.

## Verdict

Round 1 items that mattered for real-data correctness are closed: label
smoothing produces usable positives, the real training path is normalised and
no longer crashes on terrain, splits are chronologically disjoint with
fail-loud guards (baseline), precip accumulation/reset handling is verified,
ingest time-alignment is tolerance-bounded, D8 orientation is documented,
metrics/calibration are honest about degenerate cases, and norm stats are
train-only. The writers' unasked design calls (dropping `flow_direction` from
the terrain input; the `terrain=None` fallback reading straight off
`FEATURE_CHANNELS`) are sound.

**F19 is a genuine regression-class gap**, not a nit: the F10 leakage fix
landed in the baseline only, never reached the deep-model data path. **Not yet
safe to train the ConvLSTM/transformer model on real data until F19 is
fixed** — the LightGBM baseline path is sound end-to-end already (modulo the
still-open, lower-severity F9/F12/F13/F18/F20, all deferred).

Round 1's "cannot assess without real data" list is unchanged and still applies.
