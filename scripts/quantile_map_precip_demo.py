"""Phase 6: quantile-mapping mechanism demo on real precipitation data.

    python scripts/quantile_map_precip_demo.py

THIS IS A PROOF OF MECHANISM, NOT THE DEPLOYMENT BIAS-CORRECTION LAYER.
The build-order step this serves ("bridge train-on-IMDAA / infer-on-
operational NCUM-GFS-GDAS") needs real IMDAA and real GFS/NCUM/GDAS data;
neither exists on disk yet (IMDAA order still queued at the provider -- see
docs/PHASE_6_FINDINGS.md). What *is* on disk: two independently-sourced,
already grid-aligned real estimates of the same physical quantity --
ERA5 reanalysis precipitation (which currently stands in for IMDAA in this
pipeline, per nowcast/ingest/names.py) and IMERG satellite-observed
precipitation, for the same real date this project already uses elsewhere
(2018-09-08, Uttarakhand). This script fits nowcast.bias_correction's
quantile-mapping module ERA5 -> IMERG and reports a real before/after
distributional-match number, proving the mechanism works on real data
without claiming it is the final operational correction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nowcast import config  # noqa: E402
from nowcast.bias_correction import fit_quantile_map  # noqa: E402
from nowcast.ingest.imdaa import load_imdaa_single_level  # noqa: E402
from nowcast.ingest.imerg import load_imerg  # noqa: E402

# Same real date used by scripts/make_dgmr_nowcast.py elsewhere in this project.
ERA5_FILE = config.DATA_DIR / "organized" / "02_reanalysis_era5" / "era5_sl_201809.nc"
IMERG_FILE = config.DATA_DIR / "organized" / "03_rainfall_imerg" / "imerg_20180908.nc"


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Max gap between two empirical CDFs -- 0 is a perfect distributional match."""
    grid = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(np.sort(a), grid, side="right") / a.size
    cdf_b = np.searchsorted(np.sort(b), grid, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def main() -> int:
    print(f"loading ERA5 (standing in for IMDAA) precip: {ERA5_FILE}")
    era5 = load_imdaa_single_level(ERA5_FILE)["precip"]
    era5 = era5.sel(time=slice("2018-09-08", "2018-09-08"))
    era5_vals = era5.values.reshape(-1)
    era5_vals = era5_vals[np.isfinite(era5_vals)]
    print(f"  ERA5 2018-09-08: {era5_vals.size} cell-hours, "
          f"min/max/mean = {era5_vals.min():.3f} / {era5_vals.max():.3f} / "
          f"{era5_vals.mean():.3f} mm/h")

    print(f"loading IMERG (satellite-observed) precip: {IMERG_FILE}")
    imerg = load_imerg(IMERG_FILE)["precip"]
    imerg_vals = imerg.values.reshape(-1)
    imerg_vals = imerg_vals[np.isfinite(imerg_vals)]
    print(f"  IMERG 2018-09-08: {imerg_vals.size} cell-hours, "
          f"min/max/mean = {imerg_vals.min():.3f} / {imerg_vals.max():.3f} / "
          f"{imerg_vals.mean():.3f} mm/h")

    ks_before = ks_statistic(era5_vals, imerg_vals)
    print(f"\nKS statistic before correction (ERA5 vs IMERG): {ks_before:.4f}")

    print("fitting quantile map ERA5 -> IMERG...")
    mapper = fit_quantile_map(era5_vals, imerg_vals)
    corrected = mapper.apply(era5_vals)

    ks_after = ks_statistic(corrected, imerg_vals)
    print(f"KS statistic after correction:                 {ks_after:.4f}")
    improved = ks_after < ks_before
    verdict = (
        "the corrected ERA5 distribution is closer to IMERG than raw ERA5 was."
        if improved
        else "unexpected -- investigate before trusting this mechanism."
    )
    print(f"\n{'IMPROVED' if improved else 'DID NOT IMPROVE'}: {verdict}")

    print("\nReminder: this is precip-only, one date, ERA5-vs-IMERG as a real-data")
    print("stand-in pair -- not the full 13-channel IMDAA-sourced bias-correction")
    print("layer, and not evidence it would work for GFS/NCUM operational inference.")
    print("See docs/PHASE_6_FINDINGS.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
