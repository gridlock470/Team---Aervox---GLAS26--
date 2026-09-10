import { useState } from 'react'
import './App.css'
import Header from './components/Header.jsx'
import NavRail from './components/NavRail.jsx'
import MapPanel from './components/MapPanel.jsx'
import XaiRail from './components/XaiRail.jsx'
import TimelineStrip from './components/TimelineStrip.jsx'

export default function App() {
  const [region, setRegion] = useState('uk')
  const [hazard, setHazard] = useState('flashflood')
  const [step, setStep] = useState(0)

  return (
    <div className="shell">
      <Header region={region} onRegionChange={setRegion} />
      <NavRail region={region} hazard={hazard} step={step} onHazardChange={setHazard} />
      <MapPanel region={region} hazard={hazard} step={step} />
      <XaiRail region={region} hazard={hazard} step={step} />
      <TimelineStrip step={step} onStepChange={setStep} />
    </div>
  )
}
