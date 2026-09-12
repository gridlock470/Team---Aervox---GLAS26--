import { DatabaseSync } from 'node:sqlite'
import { fileURLToPath } from 'node:url'
import path from 'node:path'
import fs from 'node:fs'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const dataDir = path.join(__dirname, '..', 'data')
fs.mkdirSync(dataDir, { recursive: true })

const db = new DatabaseSync(path.join(dataDir, 'auth.sqlite'))

db.exec(`
  CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
  );
`)

const insertUserStmt = db.prepare(
  'INSERT INTO users (username, email, password_hash, created_at) VALUES (?, ?, ?, ?)'
)
const findByUsernameStmt = db.prepare('SELECT * FROM users WHERE username = ?')
const findByEmailStmt = db.prepare('SELECT * FROM users WHERE email = ?')
const findByIdStmt = db.prepare('SELECT * FROM users WHERE id = ?')
const findByLoginStmt = db.prepare('SELECT * FROM users WHERE username = ? OR email = ?')

export function createUser({ username, email, passwordHash }) {
  const info = insertUserStmt.run(username, email, passwordHash, new Date().toISOString())
  return findByIdStmt.get(info.lastInsertRowid)
}

export function findUserByUsername(username) {
  return findByUsernameStmt.get(username)
}

export function findUserByEmail(email) {
  return findByEmailStmt.get(email)
}

export function findUserById(id) {
  return findByIdStmt.get(id)
}

// Sign-in accepts either the username or the email in the same field.
export function findUserByLogin(login) {
  return findByLoginStmt.get(login, login)
}
