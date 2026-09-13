import { useEffect, useRef, useState } from 'react'
import './App.css'
import Header from './components/Header.jsx'
import TopNav from './components/TopNav.jsx'
import MapPanel, { FULLSCREEN_TRANSITION_MS } from './components/MapPanel.jsx'
import TimelineStrip from './components/TimelineStrip.jsx'
import InsightModal from './components/InsightModal.jsx'
import AuthGate from './components/AuthGate.jsx'
import AlertsPanel from './components/panels/AlertsPanel.jsx'
import PointsPanel from './components/panels/PointsPanel.jsx'
import DriversPanel from './components/panels/DriversPanel.jsx'
import ModelPanel from './components/panels/ModelPanel.jsx'
import CapPanel from './components/panels/CapPanel.jsx'
import TelemetryPanel from './components/panels/TelemetryPanel.jsx'
import EnginePanel from './components/panels/EnginePanel.jsx'
import { DATA, HAZARDS, regionPeak, timeAt } from './data/nowcastData.js'
import { sevFor } from './lib/severity.js'
import { startSiren, stopSiren } from './lib/alertSound.js'
import { fetchMe } from './lib/api.js'

const AUTH_KEY = 'nowcast-auth'

function readStoredAuth() {
  try {
    const stored = JSON.parse(localStorage.getItem(AUTH_KEY))
    return stored?.token && stored?.user ? stored : null
  } catch {
    return null
  }
}

/* Everything that is not the map, the hazard selector or the lead time lives
   behind a tab in the dock -- Telemetry included, so the map keeps the
   screen instead of sharing it with an always-visible section. One panel
   mounted at a time. */
const PANELS = {
  telemetry: TelemetryPanel,
  alerts: AlertsPanel,
  points: PointsPanel,
  drivers: DriversPanel,
  model: ModelPanel,
  cap: CapPanel,
  engine: EnginePanel,
}

/* FLIP-style grow/shrink transition: measure the element's rect before the
   layout-changing class flips, then animate the transform from that old rect
   back to identity, so the box reads as *growing* into its new box rather
   than popping. Used to take .stage from its in-flow grid cell to a
   fixed fullscreen box (and back) without a hard cut. */
function flip(el, applyClass, duration) {
  if (!el) return
  const before = el.getBoundingClientRect()
  applyClass()
  const after = el.getBoundingClientRect()

  const scaleX = before.width / after.width
  const scaleY = before.height / after.height
  const translateX = before.left - after.left
  const translateY = before.top - after.top

  el.style.transformOrigin = 'top left'
  el.style.transition = 'none'
  el.style.transform = `translate(${translateX}px, ${translateY}px) scale(${scaleX}, ${scaleY})`
  // Force layout so the browser registers the start transform before the
  // next write, otherwise both style writes get batched and there is
  // nothing to transition between.
  // eslint-disable-next-line no-unused-expressions
  el.offsetHeight
  el.style.transition = `transform ${duration}ms cubic-bezier(0.2, 0.7, 0.2, 1)`
  el.style.transform = 'translate(0, 0) scale(1, 1)'

  const cleanup = () => {
    el.style.transition = ''
    el.style.transform = ''
    el.style.transformOrigin = ''
    el.removeEventListener('transitionend', cleanup)
  }
  el.addEventListener('transitionend', cleanup)
  setTimeout(cleanup, duration + 80)
}

export default function App() {
  const [region, setRegion] = useState('uk')
  const [hazard, setHazard] = useState('flashflood')
  const [step, setStep] = useState(0)
  const [activePanel, setActivePanel] = useState(null)
  const [mapFullscreen, setMapFullscreen] = useState(false)
  const [soundAlerts, setSoundAlerts] = useState(false)
  const [auth, setAuth] = useState(readStoredAuth)
  const [authStatus, setAuthStatus] = useState(() => (readStoredAuth() ? 'checking' : 'anonymous'))
  const [lastUsedAt, setLastUsedAt] = useState(() => Date.now() - 5 * 60 * 1000)
  // Synthetic-injection state, lifted here from MapPanel (which still owns
  // the interval and the map-specific fly-to/siren behavior) so the other
  // dock panels can see the same event MapPanel already reacts to. Keyed by
  // the same "regionId-stationId" id MapPanel already builds for ALL_POINTS.
  const [injectionOverrides, setInjectionOverrides] = useState({})
  const [injectionAlerts, setInjectionAlerts] = useState([])
  const stageRef = useRef(null)
  const skipFirstRegionRef = useRef(true)

  // The Model panel's "Last used" reads as of this moment -- switching city
  // is treated as asking the model to run again for that region, so it
  // resets to "just now" rather than sitting on the mount-time value.
  useEffect(() => {
    if (skipFirstRegionRef.current) {
      skipFirstRegionRef.current = false
      return
    }
    setLastUsedAt(Date.now())
  }, [region])

  // A stored token is re-validated against the server rather than trusted
  // forever -- it can expire or the account can be gone. Nothing behind the
  // gate mounts while this is pending, so there is no flash of the
  // dashboard before bouncing back to sign-in.
  useEffect(() => {
    if (!auth?.token) return undefined
    let cancelled = false
    fetchMe(auth.token)
      .then(() => { if (!cancelled) setAuthStatus('authenticated') })
      .catch(() => {
        if (cancelled) return
        setAuth(null)
        setAuthStatus('anonymous')
        try {
          localStorage.removeItem(AUTH_KEY)
        } catch {
          // Nothing to clean up if storage was never writable.
        }
      })
    return () => { cancelled = true }
  }, [auth?.token])

  function handleAuthenticated({ token, user }) {
    setAuth({ token, user })
    setAuthStatus('authenticated')
    try {
      localStorage.setItem(AUTH_KEY, JSON.stringify({ token, user }))
    } catch {
      // Session still works for this tab even if it can't be remembered.
    }
  }

  function handleLogOut() {
    setAuth(null)
    setAuthStatus('anonymous')
    try {
      localStorage.removeItem(AUTH_KEY)
    } catch {
      // Nothing to clean up if storage was never writable.
    }
  }

  // MapPanel's synthetic-injection interval calls this every time it
  // actually flips a station (never on a no-op tick). Builds a synthetic
  // alert per newly orange/red station in the SAME shape AlertsPanel/CapPanel
  // already consume from DATA[region].alerts, so both render it unmodified.
  // Drivers/Model deliberately get no fabricated per-station number here --
  // drivers are region+hazard-global in the data model with no station
  // dimension, so Model's existing "Last used" reuse and Drivers' own
  // elevated-reading note (computed from injectionOverrides directly) are
  // the only honest ways those two panels can reflect this.
  function handleInjectionEvent(changes) {
    const hazardName = HAZARDS.find((h) => h.id === hazard)?.name ?? hazard
    const now = timeAt(step)

    setInjectionOverrides((prev) => {
      const next = { ...prev }
      changes.forEach((c) => { next[`${c.regionId}-${c.stationId}`] = c.sev })
      return next
    })

    const newAlerts = changes
      .filter((c) => c.sev === 'red' || c.sev === 'orange')
      .map((c) => ({
        region: c.regionId,
        sev: c.sev,
        id: `SYN-${c.stationId}-${Date.now()}`,
        headline: `${hazardName} signal detected near ${c.name}`,
        area: c.name,
        window: `${now} onward`,
        sent: now,
      }))
    if (newAlerts.length) {
      // Capped, not unbounded, but generously: ALL_POINTS spans every
      // region, so a tight cap on this combined cross-region list would let
      // noise from regions the operator isn't even viewing crowd out their
      // own region's alerts once the cap is hit. 300 is enough headroom for
      // a very long demo session (10s/tick, up to 3 per tick) without ever
      // practically truncating a single region's own view.
      setInjectionAlerts((prev) => [...newAlerts, ...prev].slice(0, 300))
    }

    // Same "the model was just asked to run again" semantics already used
    // on region switch -- only when the event actually touches the region
    // currently being viewed, so this keeps meaning "recently relevant"
    // rather than just "the interval ticked somewhere".
    if (changes.some((c) => c.regionId === region)) {
      setLastUsedAt(Date.now())
    }
  }

  // Turning the feed off reverts what it was actively changing (Telemetry's
  // live pct/status, Drivers' elevated-reading note -- both driven off
  // injectionOverrides) back to baseline, the same way the map's own dots
  // revert. injectionAlerts is deliberately NOT cleared here: a real
  // alerting/CAP log doesn't erase its history just because the live feed
  // that produced it went quiet, and lastUsedAt stays too -- the model
  // really was asked to run, that's a fact about the past, not a live value.
  function handleSyntheticToggle(active) {
    if (!active) setInjectionOverrides({})
  }

  const activeSeverity = sevFor(regionPeak(region, hazard, step))
  const isAlerting = soundAlerts && activeSeverity !== 'green'

  // Runs the severity-appropriate siren continuously (not re-triggered on an
  // interval, which always leaves an audible gap between bursts) while sound
  // alerts are on and the active hazard reads above "no warning". Re-armed
  // on every region/hazard/step change since the peak severity for the
  // current selection can change with any of them; stopSiren() is a no-op
  // if nothing is playing, so this is safe to call unconditionally on the
  // way out.
  useEffect(() => {
    if (!soundAlerts || activeSeverity === 'green') {
      stopSiren()
      return undefined
    }
    startSiren(activeSeverity)
    return () => stopSiren()
  }, [soundAlerts, activeSeverity])

  // Grow/shrink .stage (map + its overlays) between its grid cell and a
  // fullscreen box whenever mapFullscreen flips. Skipped under
  // prefers-reduced-motion, matching the console's existing convention.
  useEffect(() => {
    const el = stageRef.current
    if (!el) return
    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches

    if (reduceMotion) {
      el.classList.toggle('is-fullscreen', mapFullscreen)
      return
    }

    flip(
      el,
      () => el.classList.toggle('is-fullscreen', mapFullscreen),
      FULLSCREEN_TRANSITION_MS
    )
  }, [mapFullscreen])

  // Escape exits fullscreen from anywhere on the page.
  useEffect(() => {
    if (!mapFullscreen) return
    function onKey(e) {
      if (e.key === 'Escape') setMapFullscreen(false)
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [mapFullscreen])

  // MapPanel's ids span every region at once; re-key/filter down to just the
  // region currently in view before handing anything to the dock panels, so
  // they never need to know about MapPanel's cross-region id scheme, and so
  // switching regions naturally hides/reveals the right synthetic events --
  // same as the static baseline data already behaves.
  const regionInjectionOverrides = Object.fromEntries(
    Object.entries(injectionOverrides)
      .map(([key, sev]) => {
        const dash = key.indexOf('-')
        return [key.slice(0, dash), key.slice(dash + 1), sev]
      })
      .filter(([regionId]) => regionId === region)
      .map(([, stationId, sev]) => [stationId, sev])
  )
  const regionInjectionAlerts = injectionAlerts.filter((a) => a.region === region)

  const alertCount = DATA[region].alerts.length + regionInjectionAlerts.length
  // Ordered by operator urgency, not by when each feature was built: Alerts
  // and the CAP log they're backed by come first (safety-critical), then the
  // core situational-awareness views (Telemetry, Points), then explainability
  // (Drivers) and meta/diagnostic views (Model), with the explicitly
  // decorative Live Compute demo last regardless of anything else.
  const tabs = [
    { id: 'alerts', label: 'Alerts', badge: alertCount, icon: 'fa-solid fa-bell', group: 'records' },
    { id: 'cap', label: 'CAP log', icon: 'fa-solid fa-file-shield', group: 'records' },
    { id: 'telemetry', label: 'Telemetry', icon: 'fa-solid fa-gauge-high', group: 'diagnostics' },
    { id: 'points', label: 'Points', icon: 'fa-solid fa-location-dot', group: 'diagnostics' },
    { id: 'drivers', label: 'Drivers', icon: 'fa-solid fa-wind', group: 'diagnostics' },
    { id: 'model', label: 'Model', icon: 'fa-solid fa-brain', group: 'diagnostics' },
    { id: 'engine', label: 'Live Compute', icon: 'fa-solid fa-terminal', group: 'diagnostics' },
  ]

  const activeTab = tabs.find((t) => t.id === activePanel) ?? null
  const Panel = activeTab ? PANELS[activeTab.id] ?? AlertsPanel : null

  // The gate is the entire front screen -- nothing about the console (map,
  // data, even the shell chrome) mounts until a session is confirmed.
  if (authStatus === 'checking') {
    return <div className="auth-gate"><p style={{ color: 'var(--ink-dim)' }}>Checking session&hellip;</p></div>
  }
  if (authStatus !== 'authenticated') {
    return <AuthGate onAuthenticated={handleAuthenticated} />
  }

  return (
    <main className="shell">
      <Header
        region={region}
        onRegionChange={setRegion}
        soundAlerts={soundAlerts}
        onToggleSoundAlerts={() => setSoundAlerts((v) => !v)}
        activeSeverity={activeSeverity}
        user={auth.user}
        onLogOut={handleLogOut}
      />
      <TopNav
        region={region}
        hazard={hazard}
        step={step}
        onHazardChange={setHazard}
        tabs={tabs}
        onOpenPanel={setActivePanel}
      />

      <div className={`stage${mapFullscreen ? ' is-fullscreen' : ''}`} ref={stageRef}>
        <MapPanel
          region={region}
          hazard={hazard}
          step={step}
          fullscreen={mapFullscreen}
          onToggleFullscreen={() => setMapFullscreen((prev) => !prev)}
          onInjectionEvent={handleInjectionEvent}
          onSyntheticToggle={handleSyntheticToggle}
        />
        <div className="stage-overlay">
          <TimelineStrip step={step} onStepChange={setStep} />
        </div>
      </div>

      <InsightModal
        open={!!activeTab}
        tabs={tabs}
        activeId={activeTab?.id}
        onSelect={setActivePanel}
        onClose={() => setActivePanel(null)}
      >
        {Panel && (
          <Panel
            region={region}
            hazard={hazard}
            step={step}
            lastUsedAt={lastUsedAt}
            injectionOverrides={regionInjectionOverrides}
            injectionAlerts={regionInjectionAlerts}
          />
        )}
      </InsightModal>

      {/* Fixed, full-viewport, pointer-events:none -- a second alarm channel
          alongside the siren rather than a themed severity indicator, so it
          reads as urgent from across a room even with the sound muted. */}
      {isAlerting && <div className="alert-border-glow" aria-hidden="true" />}

      {/* Text equivalent of the siren/border-glow above for anyone not
          relying on sound or a glance at the screen -- announced once per
          severity transition, not on every render. */}
      <div className="sr-only" role="status" aria-live="polite">
        {isAlerting
          ? `Current hazard reading: ${activeSeverity} warning level.`
          : 'Current hazard reading: no active warning.'}
      </div>
    </main>
  )
}
