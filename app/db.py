# app/db.py
import sqlite3
import json
from pathlib import Path
from typing import Optional, Iterable, Any, Dict, List, Tuple
from datetime import datetime, timedelta

from .config import Cfg, ProcessingState

# Текущая версия индексатора (повышаем до 5 — появились сохранение структурного индекса секций
# и усиленный «кеш по хэшу» с проверкой соответствия версии индексатора).
CURRENT_INDEXER_VERSION = 5

# Базовые таблицы (минимальный набор столбцов, совместимый со старыми БД)
# Добавлена таблица document_sections для сохранения структурированного индекса.
_BASE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    tg_id             TEXT NOT NULL UNIQUE,
    active_doc_id     INTEGER
);

CREATE TABLE IF NOT EXISTS documents (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_id          INTEGER NOT NULL,
    kind              TEXT,
    path              TEXT NOT NULL,
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    -- добавляются миграциями:
    -- content_sha256 TEXT,
    -- file_uid       TEXT,
    -- indexer_version INTEGER,
    -- layout_profile TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id            INTEGER NOT NULL,
    owner_id          INTEGER NOT NULL,
    page              INTEGER,
    section_path      TEXT,
    text              TEXT,
    embedding         BLOB NOT NULL
    -- добавляются миграциями:
    -- element_type TEXT,
    -- attrs       TEXT
);

-- НОВОЕ: структурированный индекс секций документа (как спарсили — так и сохраняем).
CREATE TABLE IF NOT EXISTS document_sections (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id        INTEGER NOT NULL,
    ord           INTEGER NOT NULL,          -- порядок следования секций
    title         TEXT,
    level         INTEGER,
    page          INTEGER,
    section_path  TEXT,
    element_type  TEXT,
    text          TEXT,
    attrs         TEXT,                      -- JSON (в т.ч. ocr_text, anchors и т.п.)
    FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
);
"""


# ----------------------------- helpers -----------------------------

def _table_info(con: sqlite3.Connection, table: str) -> set[str]:
    cur = con.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cur.fetchall()}

def _try_exec_many(con: sqlite3.Connection, sqls: Iterable[str]) -> None:
    cur = con.cursor()
    for s in sqls:
        cur.execute(s)
    con.commit()

def _utc_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def _queue_prune_ttl(queue: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Удаляет элементы очереди, вышедшие за TTL."""
    if not queue:
        return queue
    ttl = timedelta(seconds=Cfg.PENDING_QUEUE_TTL_SEC)
    now = datetime.utcnow()
    kept: List[Dict[str, Any]] = []
    for item in queue:
        try:
            ts = item.get("ts")
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else None
            if dt and (now - dt.replace(tzinfo=None)) <= ttl:
                kept.append(item)
        except Exception:
            # Не валидный ts — выбрасываем элемент
            continue
    return kept


# ----------------------------- migrations -----------------------------

def _ensure_columns(con: sqlite3.Connection) -> None:
    """Добавляем недостающие колонки (безопасно для старых БД)."""
    cur = con.cursor()

    # users: активный документ плюс состояние FSM/очередь
    ucols = _table_info(con, "users")
    if "active_doc_id" not in ucols:
        cur.execute("ALTER TABLE users ADD COLUMN active_doc_id INTEGER")
    if "processing_state" not in ucols:
        cur.execute("ALTER TABLE users ADD COLUMN processing_state TEXT DEFAULT 'idle'")
    if "ingest_job_id" not in ucols:
        cur.execute("ALTER TABLE users ADD COLUMN ingest_job_id TEXT")
    if "ingest_started_at" not in ucols:
        cur.execute("ALTER TABLE users ADD COLUMN ingest_started_at TIMESTAMP")
    if "last_ready_at" not in ucols:
        cur.execute("ALTER TABLE users ADD COLUMN last_ready_at TIMESTAMP")
    if "last_error" not in ucols:
        cur.execute("ALTER TABLE users ADD COLUMN last_error TEXT")
    if "pending_queue" not in ucols:
        cur.execute("ALTER TABLE users ADD COLUMN pending_queue TEXT")  # JSON-массив

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

    # document_sections: таблица создаётся в базовой схеме; дополнительных колонок пока нет.
    con.commit()


def _ensure_indexes(con: sqlite3.Connection) -> None:
    """Создаём индексы; зависящие от новых колонок — только если колонки уже есть."""
    cur = con.cursor()

    # Общие индексы
    cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS uix_users_tg_id ON users(tg_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_users_active_doc ON users(active_doc_id)")

    # FSM индексы
    ucols = _table_info(con, "users")
    if "processing_state" in ucols:
        cur.execute("CREATE INDEX IF NOT EXISTS idx_users_state ON users(processing_state)")
    if {"active_doc_id", "processing_state"}.issubset(ucols):
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_users_doc_state ON users(active_doc_id, processing_state)"
        )
    if "ingest_job_id" in ucols:
        cur.execute("CREATE INDEX IF NOT EXISTS idx_users_ingest_job ON users(ingest_job_id)")

    cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_owner ON documents(owner_id)")

    dcols = _table_info(con, "documents")
    if {"owner_id", "content_sha256"}.issubset(dcols):
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_owner_sha "
            "ON documents(owner_id, content_sha256)"
        )
    if {"owner_id", "file_uid"}.issubset(dcols):
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_owner_fileid "
            "ON documents(owner_id, file_uid)"
        )
    if "indexer_version" in dcols:
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_indexer_version "
            "ON documents(indexer_version)"
        )

    ccols = _table_info(con, "chunks")
    if {"owner_id", "doc_id"}.issubset(ccols):
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_owner_doc "
            "ON chunks(owner_id, doc_id)"
        )
    if {"owner_id", "doc_id", "section_path"}.issubset(ccols):
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_owner_doc_section "
            "ON chunks(owner_id, doc_id, section_path)"
        )
    if "element_type" in ccols:
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_type ON chunks(element_type)")
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_owner_doc_type "
            "ON chunks(owner_id, doc_id, element_type)"
        )
    if "section_path" in ccols:
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_section_path ON chunks(section_path)")

    # Индексы для document_sections
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_sections_doc_ord ON document_sections(doc_id, ord)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_sections_doc_type ON document_sections(doc_id, element_type)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_sections_doc_path ON document_sections(doc_id, section_path)"
    )

    con.commit()


def _ensure_fts(con: sqlite3.Connection) -> None:
    """
    Опционально создаём FTS5-таблицу для лексического поиска.
    Если FTS5 недоступен в сборке SQLite — просто пропускаем.
    Rebuild делаем только при первичном создании.
    """
    cur = con.cursor()
    try:
        cur.execute("SELECT COUNT(1) FROM sqlite_master WHERE type='table' AND name='chunks_fts'")
        existed = bool(cur.fetchone()[0])

        cur.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts "
            "USING fts5(text, section_path, content='chunks', content_rowid='id');"
        )

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

        if not existed:
            cur.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
            con.commit()
    except sqlite3.OperationalError:
        con.rollback()


def _ensure_schema() -> None:
    """Создаёт/мигрирует схему безопасно для уже существующих БД."""
    db_path = Path(Cfg.SQLITE_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path.as_posix(), check_same_thread=False)
    try:
        # 1) Базовые таблицы (включая document_sections)
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
    con = sqlite3.connect(Cfg.SQLITE_PATH, check_same_thread=False, isolation_level=None)
    con.row_factory = sqlite3.Row
    # Немного тюнинга SQLite
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute("PRAGMA foreign_keys=ON;")
    return con


# ----------------------------- public API: users -----------------------------

def ensure_user(tg_id: str) -> int:
    con = get_conn()
    cur = con.cursor()
    cur.execute("BEGIN IMMEDIATE;")
    try:
        cur.execute("INSERT OR IGNORE INTO users(tg_id) VALUES (?)", (tg_id,))
        cur.execute("SELECT id FROM users WHERE tg_id = ?", (tg_id,))
        row = cur.fetchone()
        if not row:
            cur.execute("INSERT OR IGNORE INTO users(tg_id) VALUES (?)", (tg_id,))
            cur.execute("SELECT id FROM users WHERE tg_id = ?", (tg_id,))
            row = cur.fetchone()
        uid = int(row["id"])
        cur.execute("COMMIT;")
        return uid
    except Exception:
        cur.execute("ROLLBACK;")
        raise
    finally:
        con.close()


# ----------------------------- public API: documents -----------------------------

def set_document_indexer_version(doc_id: int, version: int = CURRENT_INDEXER_VERSION) -> None:
    """Фиксируем версию индексатора для документа (используй после успешной индексации)."""
    con = get_conn()
    cur = con.cursor()
    cur.execute("UPDATE documents SET indexer_version=? WHERE id=?", (version, doc_id))
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


# НОВОЕ: «умный» поиск пригодного к переиспользованию документа по хэшу
# с учётом версии индексатора и наличия данных индекса (chunks/sections).
def find_reusable_document(owner_id: int,
                           content_sha256: Optional[str],
                           file_uid: Optional[str],
                           *,
                           required_indexer_version: int = CURRENT_INDEXER_VERSION) -> Optional[int]:
    """
    Возвращает doc_id только если найден документ с совпадающим хэшем/UID,
    у него indexer_version >= required_indexer_version и имеются как минимум
    чанки или секции (не пустой индекс). Иначе — None.
    """
    doc_id = find_existing_document(owner_id, content_sha256, file_uid)
    if not doc_id:
        return None

    con = get_conn()
    cur = con.cursor()
    cur.execute("SELECT indexer_version FROM documents WHERE id=?", (doc_id,))
    row = cur.fetchone()
    ver = int(row["indexer_version"]) if row and row["indexer_version"] is not None else 0

    if ver < int(required_indexer_version):
        con.close()
        return None

    # есть ли индекс? (chunks или sections)
    cur.execute("SELECT EXISTS(SELECT 1 FROM chunks WHERE doc_id=? LIMIT 1)", (doc_id,))
    has_chunks = bool(cur.fetchone()[0])
    cur.execute("SELECT EXISTS(SELECT 1 FROM document_sections WHERE doc_id=? LIMIT 1)", (doc_id,))
    has_sections = bool(cur.fetchone()[0])
    con.close()

    return doc_id if (has_chunks or has_sections) else None


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
    con.close()
    return int(doc_id)


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
    con.close()


# ----------------------------- public API: sections (NEW) -----------------------------

def upsert_document_sections(doc_id: int, sections: List[Dict[str, Any]]) -> int:
    """
    Перезаписывает структурированный индекс секций документа.
    Возвращает количество вставленных секций.
    """
    con = get_conn()
    cur = con.cursor()
    cur.execute("BEGIN IMMEDIATE;")
    try:
        cur.execute("DELETE FROM document_sections WHERE doc_id=?", (doc_id,))
        ord_no = 0
        for sec in sections or []:
            ord_no += 1
            cur.execute(
                "INSERT INTO document_sections(doc_id, ord, title, level, page, section_path, element_type, text, attrs) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    doc_id,
                    ord_no,
                    sec.get("title"),
                    sec.get("level"),
                    sec.get("page"),
                    sec.get("section_path"),
                    sec.get("element_type"),
                    sec.get("text"),
                    json.dumps(sec.get("attrs") or {}, ensure_ascii=False),
                ),
            )
        cur.execute("COMMIT;")
        return ord_no
    except Exception:
        cur.execute("ROLLBACK;")
        raise
    finally:
        con.close()


def get_document_sections(doc_id: int) -> List[Dict[str, Any]]:
    """Возвращает сохранённый структурированный индекс секций документа (в порядке ord)."""
    con = get_conn()
    cur = con.cursor()
    cur.execute(
        "SELECT ord, title, level, page, section_path, element_type, text, attrs "
        "FROM document_sections WHERE doc_id=? ORDER BY ord ASC",
        (doc_id,),
    )
    rows = cur.fetchall()
    con.close()
    out: List[Dict[str, Any]] = []
    for r in rows or []:
        try:
            attrs = json.loads(r["attrs"]) if r["attrs"] else {}
        except Exception:
            attrs = {}
        out.append({
            "title": r["title"],
            "level": r["level"],
            "page": r["page"],
            "section_path": r["section_path"],
            "element_type": r["element_type"],
            "text": r["text"],
            "attrs": attrs,
            "ord": r["ord"],
        })
    return out


def delete_document_sections(doc_id: int) -> int:
    """Удаляет все секции структурированного индекса документа. Возвращает число удалённых строк."""
    con = get_conn()
    cur = con.cursor()
    cur.execute("SELECT COUNT(1) AS c FROM document_sections WHERE doc_id=?", (doc_id,))
    cnt = int(cur.fetchone()["c"])
    cur.execute("DELETE FROM document_sections WHERE doc_id=?", (doc_id,))
    con.close()
    return cnt


def count_document_sections(doc_id: int) -> int:
    con = get_conn()
    cur = con.cursor()
    cur.execute("SELECT COUNT(1) AS c FROM document_sections WHERE doc_id=?", (doc_id,))
    cnt = int(cur.fetchone()["c"])
    con.close()
    return cnt


# ----------------------------- public API: chunks -----------------------------

def delete_document_chunks(doc_id: int, owner_id: int) -> int:
    """Удаляет все чанки документа пользователя; возвращает количество удалённых строк."""
    con = get_conn()
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) AS c FROM chunks WHERE doc_id=? AND owner_id=?", (doc_id, owner_id))
    cnt = int(cur.fetchone()["c"])
    cur.execute("DELETE FROM chunks WHERE doc_id=? AND owner_id=?", (doc_id, owner_id))
    con.close()
    return cnt


def count_document_chunks(doc_id: int, owner_id: int) -> int:
    con = get_conn()
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) AS c FROM chunks WHERE doc_id=? AND owner_id=?", (doc_id, owner_id))
    cnt = int(cur.fetchone()["c"])
    con.close()
    return cnt


# ----------------------------- public API: FTS utils -----------------------------

def has_fts() -> bool:
    try:
        con = get_conn()
        cur = con.cursor()
        cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunks_fts'")
        ok = bool(cur.fetchone()[0])
        con.close()
        return ok
    except Exception:
        return False

def fts_rebuild() -> None:
    """Полная перестройка FTS индекса (на случай массовых изменений)."""
    try:
        con = get_conn()
        cur = con.cursor()
        cur.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
        con.close()
    except Exception:
        pass


# ----------------------------- FSM / user processing state -----------------------------

def set_user_active_doc(user_id: int, doc_id: Optional[int]) -> None:
    """Сохраняем ID «активного» документа для пользователя (NULL, чтобы очистить)."""
    con = get_conn()
    cur = con.cursor()
    cur.execute("UPDATE users SET active_doc_id=? WHERE id=?", (doc_id, user_id))
    con.close()

def get_user_active_doc(user_id: int) -> Optional[int]:
    """Возвращаем ID активного документа пользователя (или None)."""
    con = get_conn()
    cur = con.cursor()
    cur.execute("SELECT active_doc_id FROM users WHERE id=?", (user_id,))
    row = cur.fetchone()
    con.close()
    val = row["active_doc_id"] if row else None
    return (int(val) if val is not None else None)


def get_user_state(user_id: int) -> Dict[str, Any]:
    con = get_conn()
    cur = con.cursor()
    cur.execute("""
        SELECT processing_state, ingest_job_id, ingest_started_at, last_ready_at, last_error,
               active_doc_id, pending_queue
        FROM users WHERE id=?
    """, (user_id,))
    row = cur.fetchone()
    con.close()
    if not row:
        return {
            "processing_state": ProcessingState.IDLE.value,
            "ingest_job_id": None,
            "ingest_started_at": None,
            "last_ready_at": None,
            "last_error": None,
            "active_doc_id": None,
            "pending_queue": [],
        }
    queue_raw = row["pending_queue"]
    try:
        q = json.loads(queue_raw) if queue_raw else []
        q = _queue_prune_ttl(q)
    except Exception:
        q = []
    return {
        "processing_state": row["processing_state"] or ProcessingState.IDLE.value,
        "ingest_job_id": row["ingest_job_id"],
        "ingest_started_at": row["ingest_started_at"],
        "last_ready_at": row["last_ready_at"],
        "last_error": row["last_error"],
        "active_doc_id": row["active_doc_id"],
        "pending_queue": q,
    }


def _set_user_fields_atomic(user_id: int, fields: Dict[str, Any]) -> None:
    if not fields:
        return
    con = get_conn()
    cur = con.cursor()
    cur.execute("BEGIN IMMEDIATE;")
    try:
        sets = []
        vals: List[Any] = []
        for k, v in fields.items():
            sets.append(f"{k}=?")
            if isinstance(v, list) or isinstance(v, dict):
                vals.append(json.dumps(v, ensure_ascii=False))
            else:
                vals.append(v)
        vals.append(user_id)
        cur.execute(f"UPDATE users SET {', '.join(sets)} WHERE id=?", vals)
        cur.execute("COMMIT;")
    except Exception:
        cur.execute("ROLLBACK;")
        raise
    finally:
        con.close()


def transition_state(user_id: int, to_state: ProcessingState, *, allow_from: Optional[Iterable[ProcessingState]] = None) -> Tuple[str, str]:
    """
    Безопасный переход состояния. Можно задать допустимые начальные состояния.
    Возвращает (old_state, new_state).
    """
    con = get_conn()
    cur = con.cursor()
    cur.execute("BEGIN IMMEDIATE;")
    try:
        cur.execute("SELECT processing_state FROM users WHERE id=?", (user_id,))
        row = cur.fetchone()
        old_state = (row["processing_state"] if row and row["processing_state"] else ProcessingState.IDLE.value)
        if allow_from:
            allowed = {s.value if isinstance(s, ProcessingState) else str(s) for s in allow_from}
            if old_state not in allowed:
                # не валидный переход — просто выходим без изменений
                cur.execute("COMMIT;")
                con.close()
                return old_state, old_state
        cur.execute("UPDATE users SET processing_state=? WHERE id=?", (to_state.value, user_id))
        cur.execute("COMMIT;")
        return old_state, to_state.value
    except Exception:
        cur.execute("ROLLBACK;")
        raise
    finally:
        con.close()


def start_downloading(user_id: int) -> None:
    transition_state(user_id, ProcessingState.DOWNLOADING,
                     allow_from=[ProcessingState.IDLE, ProcessingState.READY])

def start_indexing(user_id: int, *, doc_id: Optional[int], ingest_job_id: Optional[str]) -> None:
    con = get_conn()
    cur = con.cursor()
    cur.execute("BEGIN IMMEDIATE;")
    try:
        # блокируем параллелизм на пользователя
        cur.execute("SELECT processing_state FROM users WHERE id=?", (user_id,))
        row = cur.fetchone()
        old = (row["processing_state"] if row and row["processing_state"] else ProcessingState.IDLE.value)
        if old not in {ProcessingState.DOWNLOADING.value, ProcessingState.IDLE.value, ProcessingState.READY.value}:
            # уже есть активный пайплайн — выходим
            cur.execute("COMMIT;")
            return
        cur.execute("""
            UPDATE users
               SET processing_state=?,
                   active_doc_id=?,
                   ingest_job_id=?,
                   ingest_started_at=?,
                   last_error=NULL
             WHERE id=?
        """, (ProcessingState.INDEXING.value, doc_id, ingest_job_id, _utc_iso(), user_id))
        cur.execute("COMMIT;")
    except Exception:
        cur.execute("ROLLBACK;")
        raise
    finally:
        con.close()


def finish_indexing_success(user_id: int, *, doc_id: Optional[int]) -> None:
    con = get_conn()
    cur = con.cursor()
    cur.execute("BEGIN IMMEDIATE;")
    try:
        cur.execute("""
            UPDATE users
               SET processing_state=?,
                   active_doc_id=?,
                   ingest_job_id=NULL,
                   ingest_started_at=NULL,
                   last_ready_at=?,
                   last_error=NULL
             WHERE id=?
        """, (ProcessingState.READY.value, doc_id, _utc_iso(), user_id))
        cur.execute("COMMIT;")
    except Exception:
        cur.execute("ROLLBACK;")
        raise
    finally:
        con.close()


def finish_indexing_error(user_id: int, *, error_message: str) -> None:
    # ограничим размер сообщения об ошибке
    msg = (error_message or "").strip()
    if len(msg) > 2000:
        msg = msg[:2000] + "…"
    con = get_conn()
    cur = con.cursor()
    cur.execute("BEGIN IMMEDIATE;")
    try:
        cur.execute("""
            UPDATE users
               SET processing_state=?,
                   ingest_job_id=NULL,
                   ingest_started_at=NULL,
                   last_error=?,
                   -- active_doc_id не трогаем: пусть остаётся предыдущий успешный документ
                   last_ready_at=last_ready_at
             WHERE id=?
        """, (ProcessingState.IDLE.value, msg, user_id))
        cur.execute("COMMIT;")
    except Exception:
        cur.execute("ROLLBACK;")
        raise
    finally:
        con.close()


# ----------------------------- Pending queue (барьер READY) -----------------------------

def enqueue_pending_query(user_id: int, text: str, *, meta: Optional[Dict[str, Any]] = None) -> int:
    """
    Кладём запрос в очередь пользователя, если документ ещё не READY.
    Возвращаем длину очереди после добавления (с учётом TTL и лимитов).
    """
    con = get_conn()
    cur = con.cursor()
    cur.execute("BEGIN IMMEDIATE;")
    try:
        cur.execute("SELECT pending_queue FROM users WHERE id=?", (user_id,))
        row = cur.fetchone()
        raw = row["pending_queue"] if row else None
        try:
            queue = json.loads(raw) if raw else []
        except Exception:
            queue = []

        queue = _queue_prune_ttl(queue)

        item = {
            "ts": _utc_iso(),
            "text": text,
            "meta": (meta or {}),
        }
        queue.append(item)
        # ограничение длины
        if Cfg.PENDING_QUEUE_MAX >= 0:
            queue = queue[-Cfg.PENDING_QUEUE_MAX:]

        cur.execute("UPDATE users SET pending_queue=? WHERE id=?", (json.dumps(queue, ensure_ascii=False), user_id))
        cur.execute("COMMIT;")
        return len(queue)
    except Exception:
        cur.execute("ROLLBACK;")
        raise
    finally:
        con.close()


def dequeue_all_pending_queries(user_id: int) -> List[Dict[str, Any]]:
    """
    Забираем и очищаем всю очередь запросов пользователя после перехода в READY.
    Возвращаем список элементов (сортированы по возрастанию времени).
    """
    con = get_conn()
    cur = con.cursor()
    cur.execute("BEGIN IMMEDIATE;")
    try:
        cur.execute("SELECT pending_queue FROM users WHERE id=?", (user_id,))
        row = cur.fetchone()
        raw = row["pending_queue"] if row else None
        try:
            queue = json.loads(raw) if raw else []
        except Exception:
            queue = []

        queue = _queue_prune_ttl(queue)
        # сортировка по ts
        try:
            queue.sort(key=lambda x: x.get("ts", ""))
        except Exception:
            pass

        cur.execute("UPDATE users SET pending_queue=NULL WHERE id=?", (user_id,))
        cur.execute("COMMIT;")
        return queue
    except Exception:
        cur.execute("ROLLBACK;")
        raise
    finally:
        con.close()


def clear_pending_queue(user_id: int) -> None:
    con = get_conn()
    cur = con.cursor()
    cur.execute("UPDATE users SET pending_queue=NULL WHERE id=?", (user_id,))
    con.close()


def get_processing_state(user_id: int) -> str:
    con = get_conn()
    cur = con.cursor()
    cur.execute("SELECT processing_state FROM users WHERE id=?", (user_id,))
    row = cur.fetchone()
    con.close()
    return (row["processing_state"] if row and row["processing_state"] else ProcessingState.IDLE.value)


def set_processing_state(user_id: int, state: ProcessingState) -> None:
    _set_user_fields_atomic(user_id, {"processing_state": state.value})


def get_last_error(user_id: int) -> Optional[str]:
    con = get_conn()
    cur = con.cursor()
    cur.execute("SELECT last_error FROM users WHERE id=?", (user_id,))
    row = cur.fetchone()
    con.close()
    return (row["last_error"] if row else None)


def get_ready_status(user_id: int) -> Dict[str, Optional[str]]:
    """
    Удобный снэпшот для барьера:
    - текущее состояние
    - когда начался ingest
    - когда последний раз стало READY
    """
    con = get_conn()
    cur = con.cursor()
    cur.execute("""
        SELECT processing_state, ingest_started_at, last_ready_at, active_doc_id
          FROM users WHERE id=?
    """, (user_id,))
    row = cur.fetchone()
    con.close()
    if not row:
        return {
            "processing_state": ProcessingState.IDLE.value,
            "ingest_started_at": None,
            "last_ready_at": None,
            "active_doc_id": None,
        }
    return {
        "processing_state": row["processing_state"] or ProcessingState.IDLE.value,
        "ingest_started_at": row["ingest_started_at"],
        "last_ready_at": row["last_ready_at"],
        "active_doc_id": row["active_doc_id"],
    }
