# Demoing the AI Forecast feature to judges

Read `docs/AI_FORECAST_README.md` first if you haven't — this file assumes
you know what the feature is. This one is about *how to talk about it*.

## The one-sentence framing

> "We integrated a real pretrained deep-learning nowcasting model — the same
> architecture DeepMind published in Nature — end to end with our real
> satellite data pipeline, as a proof of concept for where this system is
> headed, alongside the ConvLSTM model we're training ourselves."

That's the honest, defensible claim. Don't let it drift into "our AI predicts
Indian monsoon storms" — a judge who asks one follow-up question will find
the gap, and overclaiming costs you more credibility than the gap itself
would.

## Demo script (60–90 seconds)

1. **Show the sign-in gate first.** "The console is behind real
   authentication — a database-backed sign-up/sign-in, not a mockup."
2. **Land on the map.** Point out it's full-bleed, real hazard data from the
   region's stations, real hazard categories.
3. **Switch to Uttarakhand**, click **AI Forecast (beta)**. Let the overlay
   render. Pause here — this is the moment.
4. Say: *"That overlay isn't a static image — it's live output from a
   pretrained generative model, DGMR, running inference on real IMERG
   satellite precipitation data from our own pipeline right now [or:
   pre-computed from our real data]. Click the lead-time buttons and it
   shows different generated frames."*
5. Click +2h / +4h / +6h to show the frames changing.
6. **Proactively name the limitation before they ask:** *"This model was
   originally trained on UK weather radar, so we're transparent in the UI
   itself that this is a proof-of-integration, not a validated forecast for
   Indian conditions yet — that's what our own ConvLSTM model, trained on
   IMERG and ERA5 data specifically for this region, is for."*
7. Point at the disclaimer caption on screen as you say it — it's already
   there, showing you built this honestly rather than hiding the caveat.

## Why "we said the limitation ourselves" is a strength, not a weakness

Judges evaluating ML projects are specifically trained to probe for
overclaiming. A team that proactively states a model's limitation reads as
technically mature; a team that gets caught overclaiming when asked reads as
either not understanding their own system or hiding something. You have
already built the honest version — use it. Pointing at your own disclaimer
in the UI is a stronger move than hoping nobody asks.

## Likely questions, and how to answer them

**"How accurate is this?"**
> "We haven't validated it for India yet — it's trained on UK radar. That's
> exactly why it's labeled a beta/experimental overlay and why we're also
> training our own ConvLSTM model on the actual IMERG/ERA5 data for this
> region — that's [status of your own model, e.g. 'currently at epoch 7 of
> training']."

**"Why use a UK-trained model at all, then?"**
> "Speed and proof of architecture. Training a generative radar model from
> scratch takes the kind of compute and data DeepMind used across a TPU
> pod — not realistic in a hackathon timeframe. Using the pretrained
> checkpoint let us prove the whole pipeline — real data in, real deep
> model, real overlay on the map — end to end, while our own model trains in
> parallel on data that actually matches our domain."

**"Is the data real or simulated?"**
> "The input is real: GPM IMERG satellite precipitation for Uttarakhand,
> from our own ingestion pipeline (`nowcast/ingest`), for a real date
> (September 8, 2018) that we also use elsewhere in the project. The model
> output is a real forward pass through the pretrained network, not
> synthetic or hand-drawn."

**"Why does it look different if I click the button again / rerun it?"**
> "DGMR is a generative model — it samples a random latent vector each run,
> like other GAN-based generators. That's expected behavior, not a bug."

**"What would it take to make this production-ready?"**
> "Two things: swap in our own model once its training and validation are
> done, and if we wanted to keep using a UK-style architecture like DGMR,
> fine-tune it on Indian radar/IMERG data instead of relying on the public
> UK checkpoint."

## What NOT to say

- Don't say "this predicts Indian weather" or "this is our forecasting
  model" without the UK-training caveat attached in the same breath.
- Don't imply the colour intensity is a calibrated rainfall amount — it's
  normalized per frame for visibility, not a physical scale (see the README
  for the actual numbers, which are wildly outside physical range and are
  themselves the evidence of domain mismatch).
- Don't claim the four frames are precise +2h/+4h/+6h forecasts — they're
  approximate steps from an 18-frame output, picked to loosely track the
  UI's existing lead-time control.

## If something breaks live

- Overlay doesn't appear: confirm the region is set to **Uttarakhand** — the
  button only shows there.
- Frontend won't load: check both `backend\start-server.bat` and
  `frontend\start-dev.bat` are running (two separate windows).
- Worst case: the frames are static pre-generated PNGs already committed to
  the repo — nothing needs to run live in Python during the demo itself,
  only the two Node dev servers. If the map won't load at all, fall back to
  walking through `docs/AI_FORECAST_README.md`'s screenshots-in-your-head
  narration of what it does, and offer to show the generated PNGs directly
  from `frontend/public/dgmr_frames/uk/`.
