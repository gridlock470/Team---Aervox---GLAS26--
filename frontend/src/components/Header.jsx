import { useEffect, useMemo, useRef, useState } from 'react'
import { REGION_META } from '../data/nowcastData.js'
import { ensureAudioReady } from '../lib/alertSound.js'
import './Header.css'

const THEME_KEY = 'nowcast-theme'
const CLOCK_FORMAT_KEY = 'nowcast-clock-format'

// The console reports everything in IST, so the header clock has to as well
// regardless of the viewer's own system timezone -- Intl handles the +05:30
// offset correctly (and it never observes DST) rather than hand-rolling the
// arithmetic. Two formatters, not one reconfigured on the fly, since 12h
// needs a different locale to get "hh:mm:ss AM/PM" rather than en-GB's
// lowercase "hh:mm:ss am".
const IST_CLOCK_24H = new Intl.DateTimeFormat('en-GB', {
  timeZone: 'Asia/Kolkata',
  hour: '2-digit',
  minute: '2-digit',
  second: '2-digit',
  hour12: false,
})
const IST_CLOCK_12H = new Intl.DateTimeFormat('en-US', {
  timeZone: 'Asia/Kolkata',
  hour: '2-digit',
  minute: '2-digit',
  second: '2-digit',
  hour12: true,
})

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

function readStoredClockFormat() {
  try {
    const stored = localStorage.getItem(CLOCK_FORMAT_KEY)
    return stored === '12h' || stored === '24h' ? stored : '24h'
  } catch {
    return '24h'
  }
}

function formatClock(date, clockFormat) {
  const formatter = clockFormat === '12h' ? IST_CLOCK_12H : IST_CLOCK_24H
  return `${formatter.format(date)} IST`
}

export default function Header({ region, onRegionChange, soundAlerts, onToggleSoundAlerts, activeSeverity, user, onLogOut }) {
  const [now, setNow] = useState(() => new Date())
  const [theme, setTheme] = useState(readStoredTheme)
  const [clockFormat, setClockFormat] = useState(readStoredClockFormat)
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
    const id = setInterval(() => setNow(new Date()), 1000)
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

  useEffect(() => {
    try {
      localStorage.setItem(CLOCK_FORMAT_KEY, clockFormat)
    } catch {
      // Private browsing can refuse writes; the choice still applies for
      // this session, it just will not be remembered.
    }
  }, [clockFormat])

  return (
    <header className="console-header">
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
          className={soundAlerts ? 'sound-toggle is-on' : 'sound-toggle'}
          data-sev={activeSeverity}
          aria-pressed={soundAlerts}
          title={soundAlerts ? 'Mute severity alert sounds' : 'Enable severity alert sounds'}
          onClick={() => { ensureAudioReady(); onToggleSoundAlerts?.() }}
        >
          <span className="sound-glyph" aria-hidden="true">
            <i className={soundAlerts ? 'fa-solid fa-bell' : 'fa-solid fa-bell-slash'}></i>
          </span>
          {soundAlerts ? 'Alert Sound On' : 'Alert Sound Off'}
        </button>
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
          <div className="clock-row">
            <span className="clock mono">{formatClock(now, clockFormat)}</span>
            <button
              type="button"
              className="clock-format-toggle"
              aria-pressed={clockFormat === '12h'}
              title={clockFormat === '24h' ? 'Switch to 12-hour clock' : 'Switch to 24-hour clock'}
              onClick={() => setClockFormat((f) => (f === '24h' ? '12h' : '24h'))}
            >
              {clockFormat === '24h' ? '24h' : '12h'}
            </button>
          </div>
        </div>
        <div className="account-block">
          <span className="account-name">
            <i className="fa-solid fa-user" aria-hidden="true"></i> {user.username}
          </span>
          <button type="button" className="login-btn is-logout" onClick={onLogOut}>
            <i className="fa-solid fa-right-from-bracket" aria-hidden="true"></i> Log Out
          </button>
        </div>
      </div>
    </header>
  )
}
