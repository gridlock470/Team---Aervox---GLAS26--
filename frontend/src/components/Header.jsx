import { useEffect, useState } from 'react'
import { REGION_META } from '../data/nowcastData.js'
import './Header.css'

const BASE_HOUR = 7, BASE_MIN = 42, BASE_SEC = 18
const BASE_TOTAL_SECONDS = BASE_HOUR * 3600 + BASE_MIN * 60 + BASE_SEC

function formatClock(totalSeconds) {
  const h = Math.floor(totalSeconds / 3600) % 24
  const m = Math.floor(totalSeconds / 60) % 60
  const s = totalSeconds % 60
  return `${h < 10 ? '0' : ''}${h}:${m < 10 ? '0' : ''}${m}:${s < 10 ? '0' : ''}${s} IST`
}

export default function Header({ region, onRegionChange }) {
  const [clockSeconds, setClockSeconds] = useState(BASE_TOTAL_SECONDS)

  useEffect(() => {
    const id = setInterval(() => {
      setClockSeconds((prev) => prev + 1)
    }, 1000)
    return () => clearInterval(id)
  }, [])

  return (
    <header className="console-header">
      <div className="brand">
        <h1>Hyperlocal Nowcast Console</h1>
        <p>Precursor detection for severe thunderstorms, cloudbursts and flash floods, 2&ndash;6 hours ahead of onset.</p>
      </div>
      <div className="header-controls">
        <div className="segmented" role="group" aria-label="Region">
          {REGION_META.map((meta) => (
            <button
              key={meta.id}
              type="button"
              aria-pressed={region === meta.id}
              onClick={() => onRegionChange(meta.id)}
            >
              {meta.label}
            </button>
          ))}
        </div>
        <div className="status-block">
          <span className="live-pill"><span className="live-dot"></span>Live &mdash; refreshed every 10 min</span>
          <span className="clock mono">{formatClock(clockSeconds)}</span>
        </div>
      </div>
    </header>
  )
}
