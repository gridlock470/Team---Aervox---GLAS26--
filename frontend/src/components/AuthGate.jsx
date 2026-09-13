import { useEffect, useRef, useState } from 'react'
import { register, login } from '../lib/api.js'
import { HAZARDS, STEPS, REGION_META } from '../data/nowcastData.js'
import './AuthGate.css'

const SECTIONS = [
  { id: 'home', label: 'Home' },
  { id: 'mission', label: 'Mission' },
  { id: 'capabilities', label: 'Capabilities' },
  { id: 'impact', label: 'Case studies' },
  { id: 'start', label: 'Get started' },
]

// Real events, not invented ones -- picked because they match this console's
// exact hazard categories (cloudburst, flash flood, urban flooding from
// extreme rainfall) and because hyperlocal, short-lead nowcasting is
// specifically the thing that was missing in each of them.
const CASE_STUDIES = [
  {
    place: 'Uttarakhand',
    year: '2013',
    title: 'Cloudburst & flash floods',
    body: 'A sudden, highly localized cloudburst over the upper catchments triggered flash flooding through Kedarnath and the Mandakini valley with almost no lead time. One of the clearest examples in India of why hyperlocal, short-lead nowcasting matters more than a regional forecast.',
  },
  {
    place: 'Chennai',
    year: '2015',
    title: 'Urban flooding',
    body: "Days of sustained extreme rainfall outpaced the city's drainage and reservoir capacity, flooding large parts of a major metro with little warning of how fast conditions were escalating hour to hour.",
  },
  {
    place: 'Delhi & Chennai',
    year: '2020',
    title: 'Extreme rainfall events',
    body: 'Both cities saw monsoon rainfall spikes that outpaced routine forecasting that season, underscoring that even well-instrumented metros need faster, more local nowcasting than a daily or regional bulletin can give.',
  },
]

const CAPABILITIES = [
  {
    icon: 'fa-solid fa-map-location-dot',
    title: 'Live hazard map',
    body: 'A full-screen map of station readings and hazard zones, updated as new data arrives.',
  },
  {
    icon: 'fa-solid fa-triangle-exclamation',
    title: 'Multi-hazard tracking',
    body: 'Thunderstorm, cloudburst, and flash-flood risk, tracked separately for each region.',
  },
  {
    icon: 'fa-solid fa-wand-magic-sparkles',
    title: 'AI forecast overlay',
    body: 'A pretrained generative nowcasting model rendered directly on the map, alongside the model we are training ourselves.',
  },
  {
    icon: 'fa-solid fa-file-shield',
    title: 'Alerts and CAP log',
    body: 'Structured alert history in the Common Alerting Protocol format, with lead time to each broadcast.',
  },
  {
    icon: 'fa-solid fa-gauge-high',
    title: 'Model diagnostics',
    body: 'Training status, input freshness, and calibration, shown in plain view rather than hidden behind the forecast.',
  },
]

// Password Visibility is a Medium-severity Forms finding (ui-ux-pro-max
// skill, `--domain ux`): "let users see password while typing." One small
// component instead of repeating the eye-toggle markup at all three call
// sites (signup password, confirm, sign-in password).
function PasswordField({ value, onChange, autoComplete, minLength, visible, onToggleVisible }) {
  return (
    <div className="password-field">
      <input
        type={visible ? 'text' : 'password'}
        value={value}
        onChange={onChange}
        autoComplete={autoComplete}
        required
        minLength={minLength}
      />
      <button
        type="button"
        className="password-toggle"
        onClick={onToggleVisible}
        aria-label={visible ? 'Hide password' : 'Show password'}
      >
        <i className={visible ? 'fa-solid fa-eye-slash' : 'fa-solid fa-eye'} aria-hidden="true"></i>
      </button>
    </div>
  )
}

function scrollToSection(id) {
  const el = document.getElementById(id)
  if (!el) return
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
  el.scrollIntoView({ behavior: reduceMotion ? 'auto' : 'smooth', block: 'start' })
}

/**
 * The console's front door: a full landing page ending in the real sign-in /
 * sign-up gate. App.jsx mounts this instead of the dashboard whenever there
 * is no authenticated session -- nothing behind it renders until that
 * changes. Signing up does not log you in; it hands off to Sign In, on
 * purpose (see the auth-form section below).
 */
export default function AuthGate({ onAuthenticated }) {
  const [ready, setReady] = useState(false)
  const [mode, setMode] = useState('signin') // 'signin' | 'signup'
  const [form, setForm] = useState({ username: '', email: '', login: '', password: '', confirm: '' })
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [busy, setBusy] = useState(false)
  const [showPassword, setShowPassword] = useState(false)
  const [showConfirm, setShowConfirm] = useState(false)
  const navRef = useRef(null)

  // One orchestrated entrance for the hero on first paint -- deliberately
  // the only non-user-triggered motion on the page besides the radar scope.
  useEffect(() => {
    const t = setTimeout(() => setReady(true), 40)
    return () => clearTimeout(t)
  }, [])

  function update(field) {
    return (e) => setForm((f) => ({ ...f, [field]: e.target.value }))
  }

  function goToAuth(nextMode) {
    setMode(nextMode)
    setError('')
    setNotice('')
    scrollToSection('start')
  }

  function switchMode(next) {
    setMode(next)
    setError('')
    setNotice('')
  }

  async function handleSubmit(e) {
    e.preventDefault()
    setError('')

    if (mode === 'signup' && form.password !== form.confirm) {
      setError('Passwords do not match.')
      return
    }

    setBusy(true)
    try {
      if (mode === 'signup') {
        await register({ username: form.username, email: form.email, password: form.password })
        // Deliberately not authenticating here -- signup hands off to sign-in.
        setMode('signin')
        setNotice('Account created. Sign in to continue.')
        setForm((f) => ({ ...f, login: f.username, password: '', confirm: '' }))
      } else {
        const result = await login({ login: form.login, password: form.password })
        onAuthenticated(result)
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="landing">
      <nav className="landing-nav" ref={navRef}>
        <button type="button" className="landing-brand" onClick={() => scrollToSection('home')}>
          <i className="fa-solid fa-cloud-bolt" aria-hidden="true"></i>
          <span>Nowcast</span>
        </button>
        <div className="landing-nav-links">
          {SECTIONS.slice(0, -1).map((s) => (
            <button key={s.id} type="button" onClick={() => scrollToSection(s.id)}>
              {s.label}
            </button>
          ))}
        </div>
        <button type="button" className="landing-nav-cta" onClick={() => goToAuth('signin')}>
          Sign in
        </button>
      </nav>

      <section id="home" className="landing-hero">
        <div className={`landing-hero-copy${ready ? ' is-ready' : ''}`}>
          <h1>See severe weather before it arrives.</h1>
          <p>
            A hyperlocal early-warning console for thunderstorms, cloudbursts, and
            flash floods &mdash; built on real satellite data, live station readings,
            and two nowcasting models working side by side.
          </p>
          <div className="landing-hero-actions">
            <button type="button" className="landing-btn primary" onClick={() => goToAuth('signup')}>
              Create an account
            </button>
            <button type="button" className="landing-btn ghost" onClick={() => goToAuth('signin')}>
              Sign in
            </button>
          </div>
        </div>

        <div className={`landing-scope${ready ? ' is-ready' : ''}`} aria-hidden="true">
          <div className="scope-ring scope-ring-1"></div>
          <div className="scope-ring scope-ring-2"></div>
          <div className="scope-ring scope-ring-3"></div>
          <div className="scope-sweep"></div>
          <span className="scope-blip sev-green" style={{ top: '30%', left: '62%' }}></span>
          <span className="scope-blip sev-yellow" style={{ top: '58%', left: '38%' }}></span>
          <span className="scope-blip sev-orange" style={{ top: '68%', left: '60%', animationDelay: '0.6s' }}></span>
          <span className="scope-blip sev-red" style={{ top: '42%', left: '48%', animationDelay: '1.1s' }}></span>
          <span className="scope-caption">Illustrative preview, not live telemetry</span>
        </div>

        <button type="button" className="landing-scroll-cue" onClick={() => scrollToSection('mission')} aria-label="Scroll to learn more">
          <i className="fa-solid fa-chevron-down" aria-hidden="true"></i>
        </button>
      </section>

      <section id="mission" className="landing-mission">
        <div className="mission-stat">
          <span className="mission-num mono">{HAZARDS.length}</span>
          <span className="mission-label">Hazard types tracked</span>
        </div>
        <div className="mission-divider"></div>
        <div className="mission-stat">
          <span className="mission-num mono">{STEPS[STEPS.length - 1]}</span>
          <span className="mission-label">Forecast lead time</span>
        </div>
        <div className="mission-divider"></div>
        <div className="mission-stat">
          <span className="mission-num mono">{REGION_META.length}</span>
          <span className="mission-label">Regions monitored</span>
        </div>
      </section>

      <section id="capabilities" className="landing-capabilities">
        <div className="capabilities-intro">
          <h2>What the console actually does</h2>
          <p>
            Every panel is backed by real data or a real model &mdash; nothing on
            the console is a mockup standing in for a future feature.
          </p>
        </div>
        <div className="capabilities-list">
          {CAPABILITIES.map((c) => (
            <div className="capability-row" key={c.title}>
              <i className={c.icon} aria-hidden="true"></i>
              <div>
                <h3>{c.title}</h3>
                <p>{c.body}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      <section id="impact" className="landing-impact">
        <div className="impact-intro">
          <h2>Real disasters, real stakes</h2>
          <p>
            These are the failure modes hyperlocal nowcasting is built to close &mdash;
            events where minutes of lead time would have changed the outcome.
          </p>
        </div>
        <div className="case-study-grid">
          {CASE_STUDIES.map((c) => (
            <div className="case-study-card" key={c.place + c.year}>
              <span className="case-study-year mono">{c.year}</span>
              <h3>{c.place}</h3>
              <p className="case-study-title">{c.title}</p>
              <p>{c.body}</p>
            </div>
          ))}
        </div>
      </section>

      <section id="start" className="landing-start">
        <div className="auth-gate-card">
          <div className="auth-gate-brand">
            <i className="fa-solid fa-cloud-bolt" aria-hidden="true"></i>
            <div>
              <h1>Get started</h1>
              <p>Sign in to reach the live console, or create an account if you're new.</p>
            </div>
          </div>

          <div className="auth-tabs" role="tablist">
            <button
              type="button"
              role="tab"
              aria-selected={mode === 'signin'}
              className={mode === 'signin' ? 'auth-tab active' : 'auth-tab'}
              onClick={() => switchMode('signin')}
            >
              Sign In
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={mode === 'signup'}
              className={mode === 'signup' ? 'auth-tab active' : 'auth-tab'}
              onClick={() => switchMode('signup')}
            >
              Sign Up
            </button>
          </div>

          <form className="auth-form" onSubmit={handleSubmit}>
            {mode === 'signup' ? (
              <>
                <label>
                  <span>Username</span>
                  <input
                    type="text"
                    value={form.username}
                    onChange={update('username')}
                    autoComplete="username"
                    required
                    minLength={3}
                    maxLength={24}
                  />
                </label>
                <label>
                  <span>Email</span>
                  <input
                    type="email"
                    value={form.email}
                    onChange={update('email')}
                    autoComplete="email"
                    required
                  />
                </label>
                <label>
                  <span>Password</span>
                  <PasswordField
                    value={form.password}
                    onChange={update('password')}
                    autoComplete="new-password"
                    minLength={8}
                    visible={showPassword}
                    onToggleVisible={() => setShowPassword((v) => !v)}
                  />
                </label>
                <label>
                  <span>Confirm password</span>
                  <PasswordField
                    value={form.confirm}
                    onChange={update('confirm')}
                    autoComplete="new-password"
                    minLength={8}
                    visible={showConfirm}
                    onToggleVisible={() => setShowConfirm((v) => !v)}
                  />
                </label>
              </>
            ) : (
              <>
                <label>
                  <span>Username or email</span>
                  <input
                    type="text"
                    value={form.login}
                    onChange={update('login')}
                    autoComplete="username"
                    required
                  />
                </label>
                <label>
                  <span>Password</span>
                  <PasswordField
                    value={form.password}
                    onChange={update('password')}
                    autoComplete="current-password"
                    visible={showPassword}
                    onToggleVisible={() => setShowPassword((v) => !v)}
                  />
                </label>
              </>
            )}

            {notice && <p className="auth-notice">{notice}</p>}
            {error && <p className="auth-error" role="alert">{error}</p>}

            <button type="submit" className="auth-submit" disabled={busy}>
              {busy ? 'Please wait…' : mode === 'signup' ? 'Create Account' : 'Sign In'}
            </button>
          </form>
        </div>
      </section>

      <footer className="landing-footer">
        <p>Hyperlocal Nowcast Console &middot; built for Smart India Hackathon</p>
      </footer>
    </div>
  )
}
