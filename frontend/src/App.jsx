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
import { DATA, regionPeak } from './data/nowcastData.js'
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
  const stageRef = useRef(null)

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

  const alertCount = DATA[region].alerts.length
  const tabs = [
    { id: 'telemetry', label: 'Telemetry', icon: 'fa-solid fa-gauge-high' },
    { id: 'alerts', label: 'Alerts', badge: alertCount, icon: 'fa-solid fa-bell' },
    { id: 'points', label: 'Points', icon: 'fa-solid fa-location-dot' },
    { id: 'drivers', label: 'Drivers', icon: 'fa-solid fa-wind' },
    { id: 'model', label: 'Model', icon: 'fa-solid fa-brain' },
    { id: 'cap', label: 'CAP log', icon: 'fa-solid fa-file-shield' },
  ]

  const activeTab = tabs.find((t) => t.id === activePanel) ?? null
  const Panel = activeTab ? PANELS[activeTab.id] ?? TelemetryPanel : null

  // The gate is the entire front screen -- nothing about the console (map,
  // data, even the shell chrome) mounts until a session is confirmed.
  if (authStatus === 'checking') {
    return <div className="auth-gate"><p style={{ color: 'var(--ink-dim)' }}>Checking session&hellip;</p></div>
  }
  if (authStatus !== 'authenticated') {
    return <AuthGate onAuthenticated={handleAuthenticated} />
  }

  return (
    <div className="shell">
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
        />
        <div className="stage-overlay">
          <TimelineStrip step={step} onStepChange={setStep} />
        </div>
      </div>

      <InsightModal
        open={!!activeTab}
        title={activeTab?.label}
        icon={activeTab?.icon}
        onClose={() => setActivePanel(null)}
      >
        {Panel && <Panel region={region} hazard={hazard} step={step} />}
      </InsightModal>

      {/* Fixed, full-viewport, pointer-events:none -- a second alarm channel
          alongside the siren rather than a themed severity indicator, so it
          reads as urgent from across a room even with the sound muted. */}
      {isAlerting && <div className="alert-border-glow" aria-hidden="true" />}
    </div>
  )
}
