/* Severity-scaled alert siren, synthesised with the Web Audio API -- no
   external audio assets, so nothing to fetch/license/fail to load.

   A genuinely continuous tone (not a beep re-triggered on a setInterval,
   which always has an audible gap between bursts): one oscillator held
   running, with a second low-frequency oscillator modulating its pitch so it
   warbles rather than droning. Severity changes the wave shape (square is
   harsher/sharper than sine), the warble rate, and the sweep depth -- red
   is the fastest, sharpest, loudest. Green never sounds; there is nothing to
   attract attention to. */

let audioCtx = null
let active = null // { osc, lfo, gain, sev }

function getCtx() {
  if (typeof window === 'undefined') return null
  const Ctx = window.AudioContext || window.webkitAudioContext
  if (!Ctx) return null
  if (!audioCtx) audioCtx = new Ctx()
  if (audioCtx.state === 'suspended') audioCtx.resume()
  return audioCtx
}

/* Browsers only allow audio to start inside a direct user-gesture handler.
   Call this from the toggle button's own onClick (before any state update)
   so the context is created/resumed in that synchronous window, rather than
   inside a useEffect that fires a tick later. */
export function ensureAudioReady() {
  getCtx()
}

const SIREN = {
  // base/sweep in Hz, rate in warble cycles/sec, gain is peak linear volume.
  yellow: { base: 640, sweep: 70, rate: 1.4, wave: 'sine', gain: 0.11 },
  orange: { base: 760, sweep: 200, rate: 3.2, wave: 'square', gain: 0.14 },
  red: { base: 820, sweep: 320, rate: 5.5, wave: 'square', gain: 0.17 },
}

export function startSiren(sev) {
  const params = SIREN[sev]
  const ctx = getCtx()
  if (!ctx || !params) return
  if (active?.sev === sev) return // already running this exact siren
  stopSiren()

  const osc = ctx.createOscillator()
  osc.type = params.wave
  osc.frequency.value = params.base

  // The LFO's output modulates the main oscillator's frequency directly --
  // this is what makes the tone warble continuously instead of needing to
  // be re-started. lfoGain scales the modulation depth (Hz of sweep).
  const lfo = ctx.createOscillator()
  lfo.type = 'sine'
  lfo.frequency.value = params.rate
  const lfoGain = ctx.createGain()
  lfoGain.gain.value = params.sweep
  lfo.connect(lfoGain).connect(osc.frequency)

  const gain = ctx.createGain()
  gain.gain.setValueAtTime(0, ctx.currentTime)
  gain.gain.linearRampToValueAtTime(params.gain, ctx.currentTime + 0.05)

  osc.connect(gain).connect(ctx.destination)
  osc.start()
  lfo.start()

  active = { osc, lfo, gain, sev }
}

export function stopSiren() {
  if (!active) return
  const ctx = getCtx()
  const { osc, lfo, gain } = active
  if (ctx) {
    gain.gain.cancelScheduledValues(ctx.currentTime)
    gain.gain.setValueAtTime(gain.gain.value, ctx.currentTime)
    gain.gain.linearRampToValueAtTime(0, ctx.currentTime + 0.08) // fade, not a click
  }
  const stopAt = (ctx?.currentTime ?? 0) + 0.12
  try {
    osc.stop(stopAt)
    lfo.stop(stopAt)
  } catch {
    // Already stopped/closed -- nothing left to clean up.
  }
  active = null
}
