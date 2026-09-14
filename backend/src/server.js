import express from 'express'
import cors from 'cors'
import multer from 'multer'
import { execFile } from 'node:child_process'
import { randomUUID } from 'node:crypto'
import { unlink } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { createUser, findUserByUsername, findUserByEmail, findUserByLogin, findUserById } from './db.js'
import { hashPassword, verifyPassword, signToken, requireAuth } from './auth.js'

const app = express()
app.use(cors())
app.use(express.json())

// backend/src -> backend -> project root, where scripts/ and .venv/ live.
const PROJECT_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..')
const PYTHON_PATH = path.join(PROJECT_ROOT, '.venv', 'Scripts', 'python.exe')
const INSPECT_SCRIPT = path.join(PROJECT_ROOT, 'scripts', 'inspect_netcdf.py')

// Real uploads are 0.5-8 MB (IMERG daily / ERA5 monthly) -- 100 MB is
// generous headroom, not a tight guess, and diskStorage (not memory) since
// the Python script needs a real file path to open, not a buffer.
const upload = multer({
  storage: multer.diskStorage({
    destination: (req, file, cb) => cb(null, os.tmpdir()),
    filename: (req, file, cb) => cb(null, `${randomUUID()}.nc`),
  }),
  limits: { fileSize: 100 * 1024 * 1024 },
})

const USERNAME_RE = /^[a-zA-Z0-9_.-]{3,24}$/
const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/

function publicUser(user) {
  return { id: user.id, username: user.username, email: user.email, createdAt: user.created_at }
}

app.post('/api/auth/register', async (req, res) => {
  const { username, email, password } = req.body || {}

  if (typeof username !== 'string' || !USERNAME_RE.test(username)) {
    return res.status(400).json({ error: 'Username must be 3-24 characters (letters, numbers, _ . -).' })
  }
  if (typeof email !== 'string' || !EMAIL_RE.test(email)) {
    return res.status(400).json({ error: 'Enter a valid email address.' })
  }
  if (typeof password !== 'string' || password.length < 8) {
    return res.status(400).json({ error: 'Password must be at least 8 characters.' })
  }

  if (findUserByUsername(username)) {
    return res.status(409).json({ error: 'That username is already taken.' })
  }
  if (findUserByEmail(email)) {
    return res.status(409).json({ error: 'An account with that email already exists.' })
  }

  const passwordHash = await hashPassword(password)
  const user = createUser({ username, email, passwordHash })
  const token = signToken(user)
  res.status(201).json({ token, user: publicUser(user) })
})

app.post('/api/auth/login', async (req, res) => {
  const { login, password } = req.body || {}

  if (typeof login !== 'string' || !login.trim() || typeof password !== 'string') {
    return res.status(400).json({ error: 'Enter your username/email and password.' })
  }

  const user = findUserByLogin(login.trim())
  const ok = user && (await verifyPassword(password, user.password_hash))
  if (!ok) {
    return res.status(401).json({ error: 'Incorrect username/email or password.' })
  }

  const token = signToken(user)
  res.json({ token, user: publicUser(user) })
})

app.get('/api/auth/me', requireAuth, (req, res) => {
  const user = findUserById(req.user.sub)
  if (!user) return res.status(404).json({ error: 'Account no longer exists.' })
  res.json({ user: publicUser(user) })
})

app.post('/api/insights/netcdf', requireAuth, upload.single('file'), async (req, res) => {
  const file = req.file
  if (!file) {
    return res.status(400).json({ error: 'No file was uploaded.' })
  }
  if (!/\.nc$/i.test(file.originalname)) {
    await unlink(file.path).catch(() => {})
    return res.status(400).json({ error: 'Only .nc (NetCDF) files are supported.' })
  }

  execFile(
    PYTHON_PATH,
    [INSPECT_SCRIPT, file.path],
    { timeout: 60000, maxBuffer: 10 * 1024 * 1024 },
    async (err, stdout) => {
      await unlink(file.path).catch(() => {})

      if (err) {
        return res.status(500).json({ error: 'Could not analyze this file.' })
      }
      let result
      try {
        result = JSON.parse(stdout)
      } catch {
        return res.status(500).json({ error: 'Could not analyze this file.' })
      }
      res.json(result)
    }
  )
})

const PORT = process.env.PORT || 4000
app.listen(PORT, () => {
  console.log(`nowcast-auth-server listening on http://localhost:${PORT}`)
})
