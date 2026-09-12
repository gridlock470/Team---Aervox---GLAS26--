# AI Forecast overlay (DGMR) — setup and usage

This document covers the "AI Forecast (beta)" feature on the map: a real
pretrained deep-learning nowcasting model (DGMR — DeepMind's Deep Generative
Model of Radar, via OpenClimateFix's `openclimatefix/dgmr` on Hugging Face)
producing real output from real IMERG precipitation data, wired into the
console's map as an overlay you can toggle on and off.

**What it is:** a real model, running on real data, end to end.
**What it is not:** a validated forecast for India. See "Honest limitations"
below before you show this to anyone — it matters for how you talk about it.

---

## 1. One-time setup

You need three things running: the Python environment (for regenerating
frames), the Node auth backend, and the React frontend. The AI frames
themselves are pre-generated static files already committed to the repo, so
**you do not need to re-run the Python step to see the feature** — only do
that if you want fresh/different frames.

### 1a. Python environment (only needed to regenerate frames)

From the project root:

```
.venv\Scripts\python.exe -m pip install huggingface_hub einops torchvision pytorch_msssim
```

(`torch` and `pytorch_lightning` should already be installed as part of the
project's main `requirements.txt`.)

The DGMR model code itself lives in `skillful_nowcasting-main/skillful_nowcasting-main/`
(the upstream `openclimatefix/skillful_nowcasting` repo, copied in locally) —
nothing to install for that part, the generation script adds it to `sys.path`
automatically.

### 1b. Backend (login) — first time only

```
cd backend
npm install
```

### 1c. Frontend — first time only

```
cd frontend
npm install
```

---

## 2. Running the app

Two ways, pick one:

**Double-click launchers** (Windows):
- `backend\start-server.bat`
- `frontend\start-dev.bat`

**Or manually, in two terminals:**
```
cd backend && npm start          # http://localhost:4000
cd frontend && npm run dev       # http://localhost:5173
```

Open `http://localhost:5173`. You'll land on the sign-in screen first — sign
up for an account, then sign in (signing up does not log you in automatically,
by design; you sign in separately). Once inside, you're on the main console.

---

## 3. Seeing the AI Forecast overlay

1. Make sure the region selector (top search box) is on **Uttarakhand** —
   the AI overlay is only generated for this region so far.
2. On the map, look for the **"AI Forecast (beta)"** button (top-right area,
   below the map's own zoom controls, above the recenter button's row).
3. Click it. A colour overlay appears on the map — that's DGMR's generated
   output for the current lead-time step.
4. The **+2h / +4h / +6h** buttons under the map switch between 4 different
   generated frames (mapped loosely, not as precise physical lead times —
   see below).
5. Click the button again to hide the overlay. A caption with the model's
   name and disclaimer appears whenever the overlay is on.

---

## 4. Regenerating the frames (optional)

The frames are pre-generated and committed, so this is optional — do it if
you want a fresh sample (DGMR is a **generative** model: it samples a random
latent vector each run, so re-running produces a different-looking, equally
valid result every time) or want to point it at different input data.

```
.venv\Scripts\python.exe scripts\make_dgmr_nowcast.py
```

This will:
1. Load real IMERG precipitation for 2018-09-08 over Uttarakhand
   (`DataSet/organized/03_rainfall_imerg/imerg_20180908.nc`).
2. Download the pretrained DGMR weights from Hugging Face **the first time
   only** (~394 MB, cached afterward at
   `~/.cache/huggingface/hub/models--openclimatefix--dgmr` — no re-download
   on later runs). Setting an `HF_TOKEN` environment variable speeds this up
   but isn't required.
3. Run inference (a few seconds on CPU) and write 4 colour-mapped PNGs to
   `frontend/public/dgmr_frames/uk/` plus the manifest at
   `frontend/src/data/dgmrNowcast.json`.

If you change the input data or region, you'll need to edit
`scripts/make_dgmr_nowcast.py` (see `IMERG_FILE` and the `"uk"`-specific
paths near the top) — it currently only targets the one bundled Uttarakhand
IMERG file.

---

## 5. Honest limitations (read before demoing)

- **Domain mismatch.** DGMR is trained on UK Met Office weather radar. Our
  input is GPM IMERG satellite precipitation over India — a different
  sensor, resolution, and geography than what the model ever saw in
  training. The output is a real model inference, not a fabricated image,
  but its accuracy for Indian monsoon convection is unproven.
- **Sample-dataset checkpoint.** OpenClimateFix's own documentation states
  the public `openclimatefix/dgmr` checkpoint was trained on their sample
  dataset, not the full paper-scale UK radar training run.
- **Colour scale is not physical.** Each frame is normalized independently
  (its own 99th-percentile) purely so the structure is visible — the raw
  model output values are numerically far outside any real rain-rate range
  (we observed values from -589 to +3551 on a variable that should read
  0–~25 mm/h), which is itself evidence of the domain mismatch above. Do not
  read colour intensity as a calibrated rainfall amount.
- **Lead-time labels are approximate.** DGMR's native output cadence is
  5-minute radar steps; our input is hourly IMERG. The 4 displayed frames
  are 4 of its 18 output steps, picked to loosely track the app's existing
  Now/+2h/+4h/+6h control — they are labeled "(experimental)" in the UI on
  purpose, not claimed as exact physical lead times.
- **Stochastic.** DGMR samples a random latent vector per run. Regenerating
  frames will give a different-looking (not wrong, just different) result
  each time.

None of this makes the integration fake or the demo dishonest — it's a real
pretrained SOTA model producing real output from real data, which is
genuinely worth showing. It just isn't a finished, validated forecasting
product yet. Say that plainly if asked; see `docs/JUDGE_DEMO_GUIDE.md` for
how to frame it.
