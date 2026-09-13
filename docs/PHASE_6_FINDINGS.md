# Phase 6 — bias-correction: mechanism built, deployment layer still blocked

## The build-order step, and why it's not fully doable yet

[[sih-build-order]] step 7 is "bridge train-on-IMDAA / infer-on-operational
(NCUM/GFS-GDAS) gap" — at real deployment time the model would run on live
operational NWP output, which has systematically different statistics than
the reanalysis it trained on, so a bias-correction step is needed between
them. Checked directly: **zero bias-correction code existed anywhere**
(clean start) and **neither IMDAA nor GFS/NCUM/GDAS data exists on disk** —
`DataSet/organized/04_reanalysis_imdaa/` and the raw IMDAA directory are both
empty (the RDS order is still "Running", per `[[sih-build-status]]`), and
GFS/NCUM ingestion was never scheduled as a near-term step (only mentioned
as future operational-design prose in `scripts/make_overview_pdf.py`).

**Correction worth carrying forward**: this project's own `runs/phase3_final`
and this session's earlier write-ups call the training data "real 2018 IMDAA
data." That's imprecise — `nowcast/ingest/names.py` documents that **ERA5
currently stands in for IMDAA** in the pipeline ("same variable roles, 0.25°
instead of 0.12°") because the real order hasn't landed. Still real
reanalysis data, not synthetic — just not actually IMDAA yet. That also means
ERA5-vs-IMDAA isn't a usable two-source pair for this phase: right now
they're the same file playing both roles.

## What was actually built: quantile-mapping mechanism, proven on real data

`nowcast/bias_correction/` (new): `fit_quantile_map(source, target)` fits an
empirical CDF-matching correction; `QuantileMapper.apply(values)` applies it,
clamping (not extrapolating) outside the fitted range. Standard, dependency-free
(numpy only). Unit-tested against synthetic shifted distributions
(`tests/test_bias_correction_quantile_mapping.py`, 3 tests, all green) —
including that it generalizes to held-out samples, not just the fitting set.

**Proven against real data**, not just synthetic: the one real,
already-grid-aligned, independently-sourced pair that does exist —
**ERA5 reanalysis precip vs. IMERG satellite-observed precip**, both for
2018-09-08 (the same real date used elsewhere in this project) —
via `scripts/quantile_map_precip_demo.py`:

- ERA5: mean 0.227 mm/h across 44,688 real cell-hours.
- IMERG: mean 0.088 mm/h across the same cell-hours — ERA5 runs roughly
  2.6x wetter on average over this domain/date.
- **KS statistic (distributional match) before correction: 0.576.**
- **KS statistic after correction: 0.005** — a real, substantial
  improvement, not a synthetic toy result.

Caveat worth stating plainly: this number is fit-and-applied on the same
sample (one date), so it's closer to an in-sample check that the mechanism
works correctly than an out-of-sample generalization test — the unit tests
cover generalization to held-out synthetic samples, but a genuine
held-out-date check on real data (fit on one date, apply+evaluate on
another) hasn't been done, since a single real date is what's on hand.

## What this is not

This is a **precip-only, single-variable, single-date proof that the
quantile-mapping mechanism works correctly on real data** — not the
deployment bias-correction layer. That layer needs to correct all 13
IMDAA-sourced `FEATURE_CHANNELS` (`t2m, prmsl, tcwv, cape, cin, iwv,
iwv_tendency_1h, lifted_index, conv_850, shear_0_6km, shear_0_1km, u10, v10`
— the other 7 channels are INSAT/DEM-sourced observational or static fields
that don't need this treatment), and correct *toward* real operational
GFS/NCUM/GDAS output specifically, neither of which is available yet.

## Recommendation

The mechanism is ready to reuse as soon as real inputs land: once IMDAA
ingestion completes, re-run an equivalent of this demo as IMDAA-vs-current-
stand-in to validate the module against the real training source; once
GFS/NCUM/GDAS ingestion exists, apply the same `nowcast.bias_correction`
module per-channel between train-time IMDAA and inference-time operational
data. Until then, this step is correctly parked — building more
bias-correction machinery without real operational data to correct toward
would be speculative, not real progress.
