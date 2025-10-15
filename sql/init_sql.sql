PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS users(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  tg_id TEXT UNIQUE NOT NULL,
  plan TEXT DEFAULT 'trial'
);

CREATE TABLE IF NOT EXISTS documents(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  owner_id INTEGER REFERENCES users(id),
  kind TEXT NOT NULL,
  path TEXT NOT NULL,
  pages INTEGER DEFAULT 0,
  created_at TEXT DEFAULT (datetime('now'))
);

-- эмбеддинги храним как BLOB(float32[])
CREATE TABLE IF NOT EXISTS chunks(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  doc_id INTEGER REFERENCES documents(id),
  owner_id INTEGER REFERENCES users(id),
  page INTEGER,
  section_path TEXT,
  text TEXT,
  embedding BLOB
);

CREATE INDEX IF NOT EXISTS idx_chunks_owner_doc ON chunks(owner_id, doc_id);
