import { useEffect, useRef, useState } from 'react'
import { register, login } from '../lib/api.js'
import './AuthModal.css'

/**
 * Popup opened from the header's Log In button, offering the two entry
 * points an operator can take: create a new account, or sign in to an
 * existing one. Both hit the auth server's SQLite-backed user table.
 */
export default function AuthModal({ open, onClose, onAuthenticated }) {
  const [mode, setMode] = useState('signin') // 'signin' | 'signup'
  const [form, setForm] = useState({ username: '', email: '', login: '', password: '', confirm: '' })
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const dialogRef = useRef(null)

  useEffect(() => {
    if (!open) return undefined
    setError('')
    setForm({ username: '', email: '', login: '', password: '', confirm: '' })
    function onKey(e) {
      if (e.key === 'Escape') onClose()
    }
    document.addEventListener('keydown', onKey)
    dialogRef.current?.focus()
    return () => document.removeEventListener('keydown', onKey)
  }, [open, mode, onClose])

  if (!open) return null

  function update(field) {
    return (e) => setForm((f) => ({ ...f, [field]: e.target.value }))
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
      const result =
        mode === 'signup'
          ? await register({ username: form.username, email: form.email, password: form.password })
          : await login({ login: form.login, password: form.password })
      onAuthenticated(result)
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="auth-modal-backdrop" onMouseDown={onClose}>
      <div
        className="auth-modal"
        role="dialog"
        aria-modal="true"
        aria-label={mode === 'signup' ? 'Create account' : 'Sign in'}
        ref={dialogRef}
        tabIndex={-1}
        onMouseDown={(e) => e.stopPropagation()}
      >
        <div className="auth-modal-head">
          <div className="auth-tabs" role="tablist">
            <button
              type="button"
              role="tab"
              aria-selected={mode === 'signin'}
              className={mode === 'signin' ? 'auth-tab active' : 'auth-tab'}
              onClick={() => setMode('signin')}
            >
              Sign In
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={mode === 'signup'}
              className={mode === 'signup' ? 'auth-tab active' : 'auth-tab'}
              onClick={() => setMode('signup')}
            >
              Create Account
            </button>
          </div>
          <button type="button" className="auth-modal-close" aria-label="Close" onClick={onClose}>
            <i className="fa-solid fa-xmark" aria-hidden="true"></i>
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

          {error && <p className="auth-error" role="alert">{error}</p>}

          <button type="submit" className="auth-submit" disabled={busy}>
            {busy ? 'Please wait…' : mode === 'signup' ? 'Create ID & Password' : 'Sign In'}
          </button>
        </form>
      </div>
    </div>
  )
}
