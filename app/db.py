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
# Добавлена таблица document_sections для сохранения структурированного индекса,
# а также таблицы figures и figure_analysis для универсального индекса рисунков.
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

-- НОВОЕ: базовый каталог рисунков документа.
CREATE TABLE IF NOT EXISTS figures (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id        INTEGER NOT NULL,
    figure_label  TEXT,           -- '2.3', 'Рис. 2.3' и т.п.
    page          INTEGER,
    image_path    TEXT,           -- относительный путь к файлу png/jpg
    caption       TEXT,           -- подпись под рисунком (как в документе)
    kind          TEXT,           -- предварительный тип: bar/pie/org_chart/...
    attrs         TEXT,           -- доп. JSON (bounding box, source и т.п.)
    FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
);

-- НОВОЕ: результаты универсального анализа рисунков.
CREATE TABLE IF NOT EXISTS figure_analysis (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    figure_id      INTEGER NOT NULL,   -- FK на figures.id
    kind           TEXT,               -- итоговый тип (может отличаться от предварительного)
    data_json      TEXT,               -- JSON со структурой чисел/текстовых блоков
    exact_numbers  INTEGER,            -- 0/1 — гарантированы ли точные числа
    exact_text     INTEGER,            -- 0/1 — гарантирован ли точный текст
    confidence     REAL,               -- уверенность 0..1
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (figure_id) REFERENCES figures(id) ON DELETE CASCADE
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

    # Индексы для figures / figure_analysis (универсальный индекс рисунков)
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_figures_doc ON figures(doc_id)"
    )
    cur.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uix_figures_doc_label ON figures(doc_id, figure_label)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_figure_analysis_figure ON figure_analysis(figure_id)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_figure_analysis_kind ON figure_analysis(kind)"
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


# ----------------------------- public API: figures (универсальный индекс) -----------------------------

def upsert_figure(
    doc_id: int,
    figure_label: Optional[str],
    page: Optional[int],
    image_path: Optional[str],
    caption: Optional[str],
    *,
    kind: Optional[str] = None,
    attrs: Optional[Dict[str, Any]] = None,
) -> int:
    """
    Создаёт или обновляет запись о рисунке.
    Пытается найти существующую запись по (doc_id, figure_label) если label задан,
    иначе по (doc_id, image_path). Возвращает id строки в таблице figures.
    """
    con = get_conn()
    cur = con.cursor()
    cur.execute("BEGIN IMMEDIATE;")
    try:
        row = None
        if figure_label is not None:
            cur.execute(
                "SELECT id FROM figures WHERE doc_id=? AND figure_label=? LIMIT 1",
                (doc_id, figure_label),
            )
            row = cur.fetchone()
        elif image_path is not None:
            cur.execute(
                "SELECT id FROM figures WHERE doc_id=? AND image_path=? LIMIT 1",
                (doc_id, image_path),
            )
            row = cur.fetchone()

        attrs_json = json.dumps(attrs or {}, ensure_ascii=False)

        if row:
            fid = int(row["id"])
            cur.execute(
                """
                UPDATE figures
                   SET page      = ?,
                       image_path= ?,
                       caption   = ?,
                       kind      = ?,
                       attrs     = ?,
                       figure_label = COALESCE(?, figure_label)
                 WHERE id = ?
                """,
                (page, image_path, caption, kind, attrs_json, figure_label, fid),
            )
        else:
            cur.execute(
                """
                INSERT INTO figures(doc_id, figure_label, page, image_path, caption, kind, attrs)
                VALUES(?,?,?,?,?,?,?)
                """,
                (doc_id, figure_label, page, image_path, caption, kind, attrs_json),
            )
            fid = int(cur.lastrowid)

        cur.execute("COMMIT;")
        return fid
    except Exception:
        cur.execute("ROLLBACK;")
        raise
    finally:
        con.close()


def get_figures_for_doc(doc_id: int) -> List[Dict[str, Any]]:
    """
    Возвращает список рисунков документа вместе с результатами анализа (если они есть).
    """
    con = get_conn()
    cur = con.cursor()
    cur.execute(
        """
        SELECT
            f.id              AS figure_id,
            f.doc_id          AS doc_id,
            f.figure_label    AS figure_label,
            f.page            AS page,
            f.image_path      AS image_path,
            f.caption         AS caption,
            f.kind            AS figure_kind,
            f.attrs           AS figure_attrs,
            a.kind            AS analysis_kind,
            a.data_json       AS data_json,
            a.exact_numbers   AS exact_numbers,
            a.exact_text      AS exact_text,
            a.confidence      AS confidence,
            a.created_at      AS created_at,
            a.updated_at      AS updated_at
        FROM figures f
        LEFT JOIN figure_analysis a ON a.figure_id = f.id
        WHERE f.doc_id = ?
        ORDER BY f.page NULLS FIRST, f.id ASC
        """,
        (doc_id,),
    )
    rows = cur.fetchall()
    con.close()

    out: List[Dict[str, Any]] = []
    for r in rows or []:
        try:
            attrs = json.loads(r["figure_attrs"]) if r["figure_attrs"] else {}
        except Exception:
            attrs = {}
        try:
            data = json.loads(r["data_json"]) if r["data_json"] else None
        except Exception:
            data = None

        out.append(
            {
                "figure_id": r["figure_id"],
                "doc_id": r["doc_id"],
                "label": r["figure_label"],
                "page": r["page"],
                "image_path": r["image_path"],
                "caption": r["caption"],
                "figure_kind": r["figure_kind"],
                "analysis_kind": r["analysis_kind"],
                "data": data,
                "exact_numbers": bool(r["exact_numbers"]) if r["exact_numbers"] is not None else None,
                "exact_text": bool(r["exact_text"]) if r["exact_text"] is not None else None,
                "confidence": r["confidence"],
                "attrs": attrs,
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
            }
        )
    return out


def delete_figures_for_doc(doc_id: int) -> int:
    """
    Удаляет все рисунки документа (анализ удалится каскадно благодаря FK ON DELETE CASCADE).
    Возвращает число удалённых строк из figures.
    """
    con = get_conn()
    cur = con.cursor()
    cur.execute("SELECT COUNT(1) AS c FROM figures WHERE doc_id=?", (doc_id,))
    cnt = int(cur.fetchone()["c"])
    cur.execute("DELETE FROM figures WHERE doc_id=?", (doc_id,))
    con.close()
    return cnt


def upsert_figure_analysis(
    figure_id: int,
    *,
    kind: Optional[str],
    data: Optional[Any],
    exact_numbers: Optional[bool],
    exact_text: Optional[bool],
    confidence: Optional[float],
) -> int:
    """
    Создаёт или обновляет запись в figure_analysis для указанного рисунка.
    Возвращает id строки в таблице figure_analysis.
    """
    con = get_conn()
    cur = con.cursor()
    cur.execute("BEGIN IMMEDIATE;")
    try:
        cur.execute("SELECT id FROM figure_analysis WHERE figure_id=? LIMIT 1", (figure_id,))
        row = cur.fetchone()

        if isinstance(data, (dict, list)):
            data_json = json.dumps(data, ensure_ascii=False)
        elif data is None:
            data_json = None
        else:
            data_json = str(data)

        exact_numbers_int = int(bool(exact_numbers)) if exact_numbers is not None else None
        exact_text_int = int(bool(exact_text)) if exact_text is not None else None

        if row:
            fa_id = int(row["id"])
            cur.execute(
                """
                UPDATE figure_analysis
                   SET kind          = ?,
                       data_json     = ?,
                       exact_numbers = ?,
                       exact_text    = ?,
                       confidence    = ?,
                       updated_at    = CURRENT_TIMESTAMP
                 WHERE id = ?
                """,
                (kind, data_json, exact_numbers_int, exact_text_int, confidence, fa_id),
            )
        else:
            cur.execute(
                """
                INSERT INTO figure_analysis(figure_id, kind, data_json, exact_numbers, exact_text, confidence)
                VALUES(?,?,?,?,?,?)
                """,
                (figure_id, kind, data_json, exact_numbers_int, exact_text_int, confidence),
            )
            fa_id = int(cur.lastrowid)

        cur.execute("COMMIT;")
        return fa_id
    except Exception:
        cur.execute("ROLLBACK;")
        raise
    finally:
        con.close()


def get_figure_analysis(figure_id: int) -> Optional[Dict[str, Any]]:
    """
    Возвращает результат анализа рисунка по его figure_id (или None).
    """
    con = get_conn()
    cur = con.cursor()
    cur.execute(
        """
        SELECT id, figure_id, kind, data_json, exact_numbers, exact_text, confidence,
               created_at, updated_at
          FROM figure_analysis
         WHERE figure_id = ?
         LIMIT 1
        """,
        (figure_id,),
    )
    row = cur.fetchone()
    con.close()
    if not row:
        return None

    try:
        data = json.loads(row["data_json"]) if row["data_json"] else None
    except Exception:
        data = None

    return {
        "id": row["id"],
        "figure_id": row["figure_id"],
        "kind": row["kind"],
        "data": data,
        "exact_numbers": bool(row["exact_numbers"]) if row["exact_numbers"] is not None else None,
        "exact_text": bool(row["exact_text"]) if row["exact_text"] is not None else None,
        "confidence": row["confidence"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def get_figure_analysis_by_label(doc_id: int, figure_label: str) -> Optional[Dict[str, Any]]:
    """
    Удобный хелпер: найти анализ по (doc_id, figure_label).
    """
    con = get_conn()
    cur = con.cursor()
    cur.execute(
        """
        SELECT a.id, a.figure_id, a.kind, a.data_json, a.exact_numbers, a.exact_text,
               a.confidence, a.created_at, a.updated_at
          FROM figures f
          JOIN figure_analysis a ON a.figure_id = f.id
         WHERE f.doc_id = ? AND f.figure_label = ?
         LIMIT 1
        """,
        (doc_id, figure_label),
    )
    row = cur.fetchone()
    con.close()
    if not row:
        return None

    try:
        data = json.loads(row["data_json"]) if row["data_json"] else None
    except Exception:
        data = None

    return {
        "id": row["id"],
        "figure_id": row["figure_id"],
        "kind": row["kind"],
        "data": data,
        "exact_numbers": bool(row["exact_numbers"]) if row["exact_numbers"] is not None else None,
        "exact_text": bool(row["exact_text"]) if row["exact_text"] is not None else None,
        "confidence": row["confidence"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


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
