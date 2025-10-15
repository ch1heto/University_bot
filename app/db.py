import sqlite3
from pathlib import Path
from typing import Optional, Iterable, Tuple
from .config import Cfg

# Текущая версия индексатора (повышай при изменении логики парсинга/индексации)
# Было 2; повышаем до 3 из-за новых типов чанков/attrs и префиксов.
CURRENT_INDEXER_VERSION = 3

# Базовые таблицы (минимальный набор столбцов, совместимый со старыми БД)
_BASE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    tg_id  TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS documents (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_id        INTEGER NOT NULL,
    kind            TEXT,
    path            TEXT NOT NULL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chunks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id          INTEGER NOT NULL,
    owner_id        INTEGER NOT NULL,
    page            INTEGER,
    section_path    TEXT,
    text            TEXT,
    embedding       BLOB NOT NULL
);
"""


# ----------------------------- helpers -----------------------------

def _table_info(con: sqlite3.Connection, table: str) -> set[str]:
    cur = con.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cur.fetchall()}

def _try_exec_many(con: sqlite3.Connection, sqls: Iterable[str]) -> None:
    cur = con.cursor
    cur = con.cursor()
    for s in sqls:
        cur.execute(s)
    con.commit()


# ----------------------------- migrations -----------------------------

def _ensure_columns(con: sqlite3.Connection) -> None:
    """Добавляем недостающие колонки (безопасно для старых БД)."""
    cur = con.cursor()

    # documents: content_sha256, file_uid, indexer_version, layout_profile
    dcols = _table_info(con, "documents")
    if "content_sha256" not in dcols:
        cur.execute("ALTER TABLE documents ADD COLUMN content_sha256 TEXT")
    if "file_uid" not in dcols:
        cur.execute("ALTER TABLE documents ADD COLUMN file_uid TEXT")
    if "indexer_version" not in dcols:
        cur.execute("ALTER TABLE documents ADD COLUMN indexer_version INTEGER")
    if "layout_profile" not in dcols:
        cur.execute("ALTER TABLE documents ADD COLUMN layout_profile TEXT")

    # chunks: element_type, attrs
    ccols = _table_info(con, "chunks")
    if "element_type" not in ccols:
        cur.execute("ALTER TABLE chunks ADD COLUMN element_type TEXT")
    if "attrs" not in ccols:
        cur.execute("ALTER TABLE chunks ADD COLUMN attrs TEXT")

    con.commit()


def _ensure_indexes(con: sqlite3.Connection) -> None:
    """Создаём индексы; зависящие от новых колонок — только если колонки уже есть."""
    cur = con.cursor()

    # Общие индексы
    cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS uix_users_tg_id ON users(tg_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_owner ON documents(owner_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_owner ON chunks(doc_id, owner_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_owner ON chunks(owner_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_owner_page ON chunks(doc_id, owner_id, page)")

    # Индексы по новым колонкам (если они появились)
    dcols = _table_info(con, "documents")
    if {"owner_id", "content_sha256"}.issubset(dcols):
        cur.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS uix_documents_owner_sha "
            "ON documents(owner_id, content_sha256)"
        )
    if {"owner_id", "file_uid"}.issubset(dcols):
        cur.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS uix_documents_owner_fileid "
            "ON documents(owner_id, file_uid)"
        )
    if "indexer_version" in dcols:
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_indexer_version "
            "ON documents(indexer_version)"
        )

    ccols = _table_info(con, "chunks")
    if "element_type" in ccols:
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_type ON chunks(element_type)")
    if "section_path" in ccols:
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_section_path ON chunks(section_path)")

    con.commit()


def _ensure_fts(con: sqlite3.Connection) -> None:
    """
    Опционально создаём FTS5-таблицу для лексического поиска.
    Если FTS5 недоступен в сборке SQLite — просто пропускаем.
    """
    cur = con.cursor()
    try:
        # content=chunks, content_rowid=id — будем синхронизировать триггерами
        cur.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts "
            "USING fts5(text, section_path, content='chunks', content_rowid='id');"
        )

        # Триггеры синхронизации
        cur.executescript("""
        CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(rowid, text, section_path)
            VALUES (new.id, new.text, new.section_path);
        END;
        CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, text, section_path)
            VALUES('delete', old.id, old.text, old.section_path);
        END;
        CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, text, section_path)
            VALUES('delete', old.id, old.text, old.section_path);
            INSERT INTO chunks_fts(rowid, text, section_path)
            VALUES (new.id, new.text, new.section_path);
        END;
        """)
        con.commit()

        # Первичная инициализация индекса (на случай существующих строк)
        cur.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
        con.commit()
    except sqlite3.OperationalError:
        # FTS5 может быть недоступен — игнорируем
        con.rollback()


def _ensure_schema() -> None:
    """Создаёт/мигрирует схему безопасно для уже существующих БД."""
    db_path = Path(Cfg.SQLITE_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path.as_posix(), check_same_thread=False)
    try:
        # 1) Базовые таблицы
        con.executescript(_BASE_SCHEMA_SQL)
        con.commit()
        # 2) Миграции (новые столбцы)
        _ensure_columns(con)
        # 3) Индексы
        _ensure_indexes(con)
        # 4) FTS5 (необязательно)
        _ensure_fts(con)
    finally:
        con.close()


# ----------------------------- public API: connections -----------------------------

def get_conn() -> sqlite3.Connection:
    _ensure_schema()
    con = sqlite3.connect(Cfg.SQLITE_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    # Немного тюнинга SQLite
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con


# ----------------------------- public API: users -----------------------------

def ensure_user(tg_id: str) -> int:
    con = get_conn()
    cur = con.cursor()
    cur.execute("INSERT OR IGNORE INTO users(tg_id) VALUES (?)", (tg_id,))
    cur.execute("SELECT id FROM users WHERE tg_id = ?", (tg_id,))
    uid = cur.fetchone()["id"]
    con.commit()
    con.close()
    return uid


# ----------------------------- public API: documents -----------------------------

def set_document_indexer_version(doc_id: int, version: int = CURRENT_INDEXER_VERSION) -> None:
    """Фиксируем версию индексатора для документа (используй после успешной индексации)."""
    con = get_conn()
    cur = con.cursor()
    cur.execute("UPDATE documents SET indexer_version=? WHERE id=?", (version, doc_id))
    con.commit()
    con.close()


def get_document_indexer_version(doc_id: int) -> Optional[int]:
    """Читаем сохранённую версию индексатора для документа."""
    con = get_conn()
    cur = con.cursor()
    cur.execute("SELECT indexer_version FROM documents WHERE id=?", (doc_id,))
    row = cur.fetchone()
    con.close()
    return (row["indexer_version"] if row and row["indexer_version"] is not None else None)


def find_existing_document(owner_id: int,
                           content_sha256: Optional[str],
                           file_uid: Optional[str]) -> Optional[int]:
    """
    Ищем уже загруженный документ пользователя по хэшу содержимого и/или file_unique_id Телеграма.
    Работает, только если соответствующие колонки есть (миграции выполняются автоматически).
    """
    con = get_conn()
    cur = con.cursor()
    cols = _table_info(con, "documents")
    if not {"content_sha256", "file_uid", "owner_id"}.issubset(cols):
        con.close()
        return None

    if content_sha256 and file_uid:
        cur.execute(
            "SELECT id FROM documents WHERE owner_id=? AND (content_sha256=? OR file_uid=?) "
            "ORDER BY id DESC LIMIT 1",
            (owner_id, content_sha256, file_uid),
        )
    elif content_sha256:
        cur.execute(
            "SELECT id FROM documents WHERE owner_id=? AND content_sha256=? "
            "ORDER BY id DESC LIMIT 1",
            (owner_id, content_sha256),
        )
    elif file_uid:
        cur.execute(
            "SELECT id FROM documents WHERE owner_id=? AND file_uid=? "
            "ORDER BY id DESC LIMIT 1",
            (owner_id, file_uid),
        )
    else:
        con.close()
        return None

    row = cur.fetchone()
    con.close()
    return (row["id"] if row else None)


def insert_document(owner_id: int,
                    kind: str,
                    path: str,
                    *,
                    content_sha256: Optional[str] = None,
                    file_uid: Optional[str] = None) -> int:
    """Создаём запись о документе (если есть доп. колонки — пишем и их)."""
    con = get_conn()
    cur = con.cursor()
    cols = _table_info(con, "documents")
    if {"content_sha256", "file_uid"}.issubset(cols):
        cur.execute(
            "INSERT INTO documents(owner_id, kind, path, content_sha256, file_uid) VALUES(?,?,?,?,?)",
            (owner_id, kind, path, content_sha256, file_uid),
        )
    else:
        cur.execute(
            "INSERT INTO documents(owner_id, kind, path) VALUES(?,?,?)",
            (owner_id, kind, path),
        )
    doc_id = cur.lastrowid
    con.commit()
    con.close()
    return doc_id


def update_document_meta(doc_id: int,
                         *,
                         path: Optional[str] = None,
                         content_sha256: Optional[str] = None,
                         file_uid: Optional[str] = None,
                         layout_profile: Optional[str] = None) -> None:
    """Обновляем путь/хэш/UID/профиль документа (гибко, только те поля, что переданы)."""
    con = get_conn()
    cur = con.cursor()
    sets = []
    vals: list = []
    if path is not None:
        sets.append("path=?"); vals.append(path)
    if content_sha256 is not None:
        sets.append("content_sha256=?"); vals.append(content_sha256)
    if file_uid is not None:
        sets.append("file_uid=?"); vals.append(file_uid)
    if layout_profile is not None:
        sets.append("layout_profile=?"); vals.append(layout_profile)
    if not sets:
        con.close()
        return
    vals.append(doc_id)
    cur.execute(f"UPDATE documents SET {', '.join(sets)} WHERE id=?", vals)
    con.commit()
    con.close()


# ----------------------------- public API: chunks -----------------------------

def delete_document_chunks(doc_id: int, owner_id: int) -> int:
    """Удаляет все чанки документа пользователя; возвращает количество удалённых строк."""
    con = get_conn()
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) AS c FROM chunks WHERE doc_id=? AND owner_id=?", (doc_id, owner_id))
    cnt = cur.fetchone()["c"]
    cur.execute("DELETE FROM chunks WHERE doc_id=? AND owner_id=?", (doc_id, owner_id))
    con.commit()
    con.close()
    return int(cnt)


def count_document_chunks(doc_id: int, owner_id: int) -> int:
    con = get_conn()
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) AS c FROM chunks WHERE doc_id=? AND owner_id=?", (doc_id, owner_id))
    cnt = cur.fetchone()["c"]
    con.close()
    return int(cnt)


# ----------------------------- public API: FTS utils -----------------------------

def has_fts() -> bool:
    try:
        con = get_conn()
        cur = con.cursor()
        cur.execute("SELECT 1 FROM chunks_fts LIMIT 1")
        _ = cur.fetchone()
        con.close()
        return True
    except Exception:
        return False

def fts_rebuild() -> None:
    """Полная перестройка FTS индекса (на случай массовых изменений)."""
    try:
        con = get_conn()
        cur = con.cursor()
        cur.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
        con.commit()
        con.close()
    except Exception:
        # Если FTS нет — просто тихо выходим
        pass
