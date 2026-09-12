import { useState } from 'react'
import './App.css'
import Header from './components/Header.jsx'
import NavRail from './components/NavRail.jsx'
import MapPanel from './components/MapPanel.jsx'
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

export default function App() {
  const [region, setRegion] = useState('uk')
  const [hazard, setHazard] = useState('flashflood')
  const [step, setStep] = useState(0)
  const [tab, setTab] = useState('telemetry')
  const [collapsed, setCollapsed] = useState(false)

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

      <div className="stage">
        <MapPanel region={region} hazard={hazard} step={step} />
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
