import bcrypt from 'bcryptjs'
import jwt from 'jsonwebtoken'

// A prototype needs a stable secret across restarts without asking the
// operator to configure one; production deployments should set JWT_SECRET
// themselves and this fallback is only ever reached in dev.
const JWT_SECRET = process.env.JWT_SECRET || 'nowcast-dev-secret-change-me'
const TOKEN_TTL = '7d'

export async function hashPassword(password) {
  const salt = await bcrypt.genSalt(10)
  return bcrypt.hash(password, salt)
}

export async function verifyPassword(password, hash) {
  return bcrypt.compare(password, hash)
}

export function signToken(user) {
  return jwt.sign({ sub: user.id, username: user.username }, JWT_SECRET, { expiresIn: TOKEN_TTL })
}

export function requireAuth(req, res, next) {
  const header = req.headers.authorization || ''
  const token = header.startsWith('Bearer ') ? header.slice(7) : null
  if (!token) return res.status(401).json({ error: 'Missing authorization token.' })

  try {
    req.user = jwt.verify(token, JWT_SECRET)
    next()
  } catch {
    res.status(401).json({ error: 'Invalid or expired session.' })
  }
}
