const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:4000'

async function request(path, options) {
  let res
  try {
    res = await fetch(`${API_BASE}${path}`, {
      headers: { 'Content-Type': 'application/json' },
      ...options,
    })
  } catch {
    throw new Error('Cannot reach the auth server. Is it running?')
  }

  const body = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(body.error || 'Something went wrong.')
  return body
}

export function register({ username, email, password }) {
  return request('/api/auth/register', {
    method: 'POST',
    body: JSON.stringify({ username, email, password }),
  })
}

export function login({ login, password }) {
  return request('/api/auth/login', {
    method: 'POST',
    body: JSON.stringify({ login, password }),
  })
}

export function fetchMe(token) {
  return request('/api/auth/me', {
    headers: { Authorization: `Bearer ${token}` },
  })
}

// Not built on request() -- that helper hardcodes a JSON Content-Type,
// which is wrong for a multipart upload (the browser must set its own
// Content-Type with the multipart boundary; setting it manually breaks it).
export async function uploadNetcdfInsights(token, file) {
  const formData = new FormData()
  formData.append('file', file)

  let res
  try {
    res = await fetch(`${API_BASE}/api/insights/netcdf`, {
      method: 'POST',
      headers: { Authorization: `Bearer ${token}` },
      body: formData,
    })
  } catch {
    throw new Error('Cannot reach the server. Is it running?')
  }

  const body = await res.json().catch(() => ({}))
  if (!res.ok || body.error) throw new Error(body.error || 'Something went wrong.')
  return body
}
