import { useEffect, useState } from 'react'
import { REGION_META } from '../data/nowcastData.js'
import './Header.css'

const BASE_HOUR = 7, BASE_MIN = 42, BASE_SEC = 18
const BASE_TOTAL_SECONDS = BASE_HOUR * 3600 + BASE_MIN * 60 + BASE_SEC

const THEME_KEY = 'nowcast-theme'

/* Dark is the default: this is a control-room display, usually run at night.
   localStorage is guarded because it throws outright in some privacy modes
   rather than returning null. */
function readStoredTheme() {
  try {
    const stored = localStorage.getItem(THEME_KEY)
    return stored === 'light' || stored === 'dark' ? stored : 'dark'
  } catch {
    return 'dark'
  }
}

function formatClock(totalSeconds) {
  const h = Math.floor(totalSeconds / 3600) % 24
  const m = Math.floor(totalSeconds / 60) % 60
  const s = totalSeconds % 60
  return `${h < 10 ? '0' : ''}${h}:${m < 10 ? '0' : ''}${m}:${s < 10 ? '0' : ''}${s} IST`
}

export default function Header({ region, onRegionChange }) {
  const [clockSeconds, setClockSeconds] = useState(BASE_TOTAL_SECONDS)
  const [theme, setTheme] = useState(readStoredTheme)

  useEffect(() => {
    const id = setInterval(() => {
      setClockSeconds((prev) => prev + 1)
    }, 1000)
    return () => clearInterval(id)
  }, [])

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    try {
      localStorage.setItem(THEME_KEY, theme)
    } catch {
      // Private browsing can refuse writes; the theme still applies for this
      // session, it just will not be remembered.
    }
  }, [theme])

  return (
    <header className="console-header">
      <div className="brand">
        <span className="brand-mark" aria-hidden="true"><i className="fa-solid fa-cloud-bolt"></i></span>
        <div className="brand-text">
          <h1>AERO<strong>CAST</strong> <span className="brand-sub">Hyperlocal Nowcast Console</span></h1>
          <p>Precursor detection for severe thunderstorms, cloudbursts and flash floods, 2&ndash;6 hours ahead of onset.</p>
        </div>
      </div>
      <div className="header-controls">
        <div className="area-selector" role="group" aria-label="Region">
          <span className="area-label"><i className="fa-solid fa-crosshairs" aria-hidden="true"></i> Region</span>
          <div className="segmented">
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
        </div>
        <button
          type="button"
          className="theme-toggle"
          aria-pressed={theme === 'light'}
          title={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
          onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
        >
          <span className="theme-glyph" aria-hidden="true">
            <i className={theme === 'dark' ? 'fa-solid fa-moon' : 'fa-solid fa-sun'}></i>
          </span>
          {theme === 'dark' ? 'Dark Obsidian' : 'Alabaster Gold'}
        </button>
        <div className="status-block">
          <span className="live-pill"><span className="live-dot"></span>Live &mdash; refreshed every 10 min</span>
          <span className="clock mono">{formatClock(clockSeconds)}</span>
        </div>
        <button type="button" className="login-btn" title="Operator sign-in is not wired up in this console yet">
          <i className="fa-solid fa-right-to-bracket" aria-hidden="true"></i> Log In
        </button>
      </div>
    </header>
  )
}
