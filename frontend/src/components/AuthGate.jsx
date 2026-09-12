import { useState } from 'react'
import { register, login } from '../lib/api.js'
import './AuthGate.css'

/**
 * The console's front door. Nothing behind this renders until a session
 * exists: App.jsx mounts this instead of the dashboard whenever there is no
 * authenticated user. Signing up does NOT log you in -- it hands you back to
 * the Sign In tab, because "create an account" and "enter the application"
 * are two separate, deliberate steps here.
 */
export default function AuthGate({ onAuthenticated }) {
  const [mode, setMode] = useState('signin') // 'signin' | 'signup'
  const [form, setForm] = useState({ username: '', email: '', login: '', password: '', confirm: '' })
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [busy, setBusy] = useState(false)

  function update(field) {
    return (e) => setForm((f) => ({ ...f, [field]: e.target.value }))
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
    <div className="auth-gate">
      <div className="auth-gate-card">
        <div className="auth-gate-brand">
          <i className="fa-solid fa-cloud-bolt" aria-hidden="true"></i>
          <div>
            <h1>Hyperlocal Nowcast Console</h1>
            <p>Sign in to reach the live console.</p>
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
                  autoFocus
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
                <input
                  type="password"
                  value={form.password}
                  onChange={update('password')}
                  autoComplete="new-password"
                  required
                  minLength={8}
                />
              </label>
              <label>
                <span>Confirm password</span>
                <input
                  type="password"
                  value={form.confirm}
                  onChange={update('confirm')}
                  autoComplete="new-password"
                  required
                  minLength={8}
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
                  autoFocus
                />
              </label>
              <label>
                <span>Password</span>
                <input
                  type="password"
                  value={form.password}
                  onChange={update('password')}
                  autoComplete="current-password"
                  required
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
    </div>
  )
}
