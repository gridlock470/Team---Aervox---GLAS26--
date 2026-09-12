"""Generate a demo AI nowcast overlay from the pretrained DGMR model.

    python scripts/make_dgmr_nowcast.py

Mirrors scripts/make_live_nowcast.py's pattern (real data -> static output the
console reads) but for a pretrained deep-learning nowcaster instead of the
project's own LightGBM baseline:

1. Loads real IMERG precipitation for 2018-09-08 over Uttarakhand via the
   project's own nowcast.ingest.imerg.load_imerg loader.
2. Takes the last 4 hourly frames as DGMR's input context.
3. Runs openclimatefix/dgmr (a pretrained Deep Generative Model of Radar,
   downloaded from Hugging Face and cached locally) to produce 18 forecast
   frames.
4. Colour-maps 4 evenly spaced output frames to PNGs and writes a manifest
   the frontend reads.

Honesty check, worth repeating here since it is easy to lose once this is
just "a PNG on a map": DGMR is trained on UK weather radar, a different
sensor and spatial scale than IMERG satellite precipitation, and the public
checkpoint is documented by its authors as trained on their sample dataset
only, not the full paper training run. This script produces a real model
output, not a placeholder -- but it is a demo of the integration, not a
validated India forecast. dgmrNowcast.json carries that disclaimer through
to the UI on purpose.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
# The editable `nowcast` install's finder points at a stale path, so running
# this as `python scripts/make_dgmr_nowcast.py` (script dir on sys.path, not
# cwd) can't resolve it -- add the project root explicitly instead of relying
# on that install.
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "skillful_nowcasting-main" / "skillful_nowcasting-main"))

from nowcast import config  # noqa: E402
from nowcast.ingest.imerg import load_imerg  # noqa: E402

IMERG_FILE = config.DATA_DIR / "organized" / "03_rainfall_imerg" / "imerg_20180908.nc"
OUT_JSON = config.PROJECT_ROOT / "frontend" / "src" / "data" / "dgmrNowcast.json"
OUT_FRAMES_DIR = config.PROJECT_ROOT / "frontend" / "public" / "dgmr_frames" / "uk"

FORECAST_OUT_INDICES = [0, 6, 12, 17]  # 4 of the model's 18 output steps
STEP_LABELS = ["AI nowcast, step 1 (experimental)", "AI nowcast, step 2 (experimental)",
               "AI nowcast, step 3 (experimental)", "AI nowcast, step 4 (experimental)"]

# DGMR's own grid-cell loss (dgmr/dgmr.py, weight_fn) caps rainfall intensity
# at 24 -- the clearest signal in the released code for the physical scale
# (mm/h-like) the model was trained to expect. No explicit input-normalizing
# function ships in the trimmed repo, so frames are fed at that raw scale,
# clipped for numerical stability, rather than rescaled to an arbitrary [0,1].
PRECIP_CAP_MM_H = 24.0

# Blue (light) -> cyan -> yellow -> red (heavy), the same warm-is-worse
# convention as the console's severity ramp elsewhere in the app.
_COLOR_STOPS = np.array([
    [8, 12, 40],
    [30, 110, 200],
    [80, 200, 210],
    [250, 220, 80],
    [220, 40, 40],
], dtype="float32")


def colormap(frame: np.ndarray, vmax: float) -> np.ndarray:
    """Map a 2D non-negative field to an RGB uint8 image."""
    t = np.clip(frame, 0.0, vmax) / max(vmax, 1e-6)
    n_stops = len(_COLOR_STOPS) - 1
    idx = np.clip((t * n_stops).astype(int), 0, n_stops - 1)
    frac = (t * n_stops - idx)[..., None]
    rgb = _COLOR_STOPS[idx] * (1 - frac) + _COLOR_STOPS[idx + 1] * frac
    return rgb.astype("uint8")


def resize(frame: np.ndarray, size: int = 256) -> np.ndarray:
    from PIL import Image

    img = Image.fromarray(frame.astype("float32"), mode="F")
    return np.asarray(img.resize((size, size), Image.BILINEAR))


def main() -> int:
    import torch
    from dgmr import DGMR

    print(f"loading real IMERG data: {IMERG_FILE}")
    da = load_imerg(IMERG_FILE)["precip"]
    # (lat, lon), north row first, matches how an image's row 0 reads as "top".
    da = da.transpose("time", "lat", "lon")
    context = da.isel(time=slice(-4, None)).values[:, ::-1, :]  # flip lat -> north-up
    print(f"input context: {context.shape}, min/max/mean = "
          f"{context.min():.3f} / {context.max():.3f} / {context.mean():.3f} mm/h")

    context_256 = np.stack([resize(f) for f in context])
    x = torch.from_numpy(context_256).float().clip(0, PRECIP_CAP_MM_H)
    x = x.unsqueeze(0).unsqueeze(2)  # (batch=1, time=4, channel=1, H, W)

    print("loading pretrained DGMR (cached locally after the first run)...")
    model = DGMR.from_pretrained("openclimatefix/dgmr")
    model.eval()

    print("running inference...")
    with torch.no_grad():
        out = model(x)  # (1, 18, 1, 256, 256)
    out_np = out.squeeze(0).squeeze(1).numpy()
    print(f"model output: {out_np.shape}, min/max/mean = "
          f"{out_np.min():.3f} / {out_np.max():.3f} / {out_np.mean():.3f}")

    OUT_FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    frames_meta = []
    from PIL import Image as PILImage

    for step, out_idx in enumerate(FORECAST_OUT_INDICES):
        field = np.clip(out_np[out_idx], 0, None)
        # Normalized per-frame, not against one shared physical scale: this
        # output is already far outside any physically meaningful range (see
        # the module docstring), and the GAN's magnitude swings wildly and
        # unpredictably between timesteps when run outside its trained
        # domain. A shared scale would just show whichever frame happened to
        # have the largest outlier and flatten the rest to near-black.
        vmax = max(float(np.percentile(field, 99)), 1e-3)
        rgb = colormap(field, vmax)
        fname = f"frame_{step}.png"
        PILImage.fromarray(rgb, mode="RGB").save(OUT_FRAMES_DIR / fname)
        frames_meta.append({
            "step": step,
            "label": STEP_LABELS[step],
            "file": f"/dgmr_frames/uk/{fname}",
            "model_output_index": out_idx,
        })
        print(f"  wrote {fname} (model output frame {out_idx})")

    manifest = {
        "region": "uk",
        "bbox": [config.BBOX_WEST, config.BBOX_SOUTH, config.BBOX_EAST, config.BBOX_NORTH],
        "generated_from": "Real IMERG precipitation, 2018-09-08, Uttarakhand",
        "model": {
            "name": "DGMR (Deep Generative Model of Radar)",
            "source": "openclimatefix/dgmr (Hugging Face)",
            "paper": "Ravuri et al. 2021, https://arxiv.org/abs/2104.00954",
        },
        "disclaimer": (
            "Demo of a real pretrained model producing real output, not a validated "
            "India forecast: DGMR is trained on UK weather radar (different sensor "
            "and spatial scale than IMERG), and the public checkpoint is documented "
            "as trained on a sample dataset only, not the full paper training run."
        ),
        "frames": frames_meta,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
