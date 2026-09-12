import { useEffect, useRef, useState } from 'react'
import './App.css'
import Header from './components/Header.jsx'
import NavRail from './components/NavRail.jsx'
import MapPanel, { FULLSCREEN_TRANSITION_MS } from './components/MapPanel.jsx'
import TimelineStrip from './components/TimelineStrip.jsx'
import Dock from './components/Dock.jsx'
import AlertsPanel from './components/panels/AlertsPanel.jsx'
import PointsPanel from './components/panels/PointsPanel.jsx'
import DriversPanel from './components/panels/DriversPanel.jsx'
import ModelPanel from './components/panels/ModelPanel.jsx'
import CapPanel from './components/panels/CapPanel.jsx'
import TelemetryPanel from './components/panels/TelemetryPanel.jsx'
import { DATA } from './data/nowcastData.js'

/* One panel is mounted at a time -- the dock renders whichever the active tab
   selects. Keeping the map unobstructed is the point: everything that is not
   the map, the hazard selector or the lead time lives behind a tab.
   Telemetry (the pasted AEROCAST mock's stat cards + station table) is the
   default view; Alerts/Points/Drivers/Model/CAP are the console's own
   panels, kept reachable behind their own tabs rather than dropped. */
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
  const [tab, setTab] = useState('telemetry')
  const [collapsed, setCollapsed] = useState(false)
  const [mapFullscreen, setMapFullscreen] = useState(false)
  const stageRef = useRef(null)

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
    { id: 'telemetry', label: 'Telemetry' },
    { id: 'alerts', label: 'Alerts', badge: alertCount },
    { id: 'points', label: 'Points' },
    { id: 'drivers', label: 'Drivers' },
    { id: 'model', label: 'Model' },
    { id: 'cap', label: 'CAP log' },
  ]

  const Panel = PANELS[tab] ?? TelemetryPanel

  return (
    <div className="shell">
      <Header region={region} onRegionChange={setRegion} />
      <NavRail region={region} hazard={hazard} step={step} onHazardChange={setHazard} />

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

      <Dock
        tabs={tabs}
        active={tab}
        onActiveChange={setTab}
        collapsed={collapsed}
        onCollapsedChange={setCollapsed}
      >
        <Panel region={region} hazard={hazard} step={step} />
      </Dock>
    </div>
  )
}
