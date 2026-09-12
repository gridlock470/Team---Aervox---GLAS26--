import { useEffect, useMemo, useRef, useState } from 'react'
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
  const [query, setQuery] = useState('')
  const [open, setOpen] = useState(false)
  const searchRef = useRef(null)

  const activeMeta = REGION_META.find((m) => m.id === region)

  const matches = useMemo(() => {
    const q = query.trim().toLowerCase()
    if (!q) return REGION_META
    return REGION_META.filter((m) => m.label.toLowerCase().includes(q))
  }, [query])

  function pick(meta) {
    onRegionChange(meta.id)
    setQuery('')
    setOpen(false)
  }

  function handleSubmit(e) {
    e.preventDefault()
    if (matches.length) pick(matches[0])
  }

  // Close the suggestion list on an outside click -- a search box that stays
  // open until you click its own suggestion is a search box that traps focus.
  useEffect(() => {
    function onDocClick(e) {
      if (searchRef.current && !searchRef.current.contains(e.target)) setOpen(false)
    }
    document.addEventListener('mousedown', onDocClick)
    return () => document.removeEventListener('mousedown', onDocClick)
  }, [])

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

      <form className="region-search" ref={searchRef} role="search" onSubmit={handleSubmit}>
        <i className="fa-solid fa-magnifying-glass" aria-hidden="true"></i>
        <input
          type="text"
          value={open ? query : query || activeMeta?.label || ''}
          placeholder="Search region or city&hellip;"
          aria-label="Search region or city"
          onFocus={() => { setOpen(true); setQuery('') }}
          onChange={(e) => { setQuery(e.target.value); setOpen(true) }}
        />
        {open && (
          <ul className="region-search-list" role="listbox">
            {matches.length === 0 && <li className="region-search-empty">No region matches &ldquo;{query}&rdquo;</li>}
            {matches.map((meta) => (
              <li key={meta.id}>
                <button
                  type="button"
                  role="option"
                  aria-selected={region === meta.id}
                  onClick={() => pick(meta)}
                >
                  <i className="fa-solid fa-location-dot" aria-hidden="true"></i> {meta.label}
                </button>
              </li>
            ))}
          </ul>
        )}
      </form>

      <div className="header-controls">
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
