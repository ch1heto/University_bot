# app/bot.py
import re
import os
import html
import json
import logging
from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import Command

from .config import Cfg
from .db import (
    ensure_user, get_conn,
    set_document_indexer_version, get_document_indexer_version,
    CURRENT_INDEXER_VERSION,
    update_document_meta, delete_document_chunks,
)
from .parsing import parse_docx, parse_pdf, parse_doc, save_upload
from .indexing import index_document
from .retrieval import (
    retrieve, build_context, invalidate_cache,
    _mk_table_pattern, _mk_figure_pattern, keyword_find,  # могут не понадобиться, но оставим
)
from .ace import ace_once, agent_no_context
from .polza_client import probe_embedding_dim  # используем для профиля эмбеддингов

# утилиты
from .utils import safe_filename, sha256_bytes, split_for_telegram, infer_doc_kind

# гибридный контекст: семантика + FTS/LIKE
from .lexsearch import best_context


# --------------------- форматирование и отправка ---------------------

_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)

def _to_html(text: str) -> str:
    """Конвертируем **bold** в <b>...</b> и экранируем HTML."""
    if not text:
        return ""
    text = _BOLD_RE.sub(r"<b>\1</b>", text)
    text = html.escape(text)
    return text.replace("&lt;b&gt;", "<b>").replace("&lt;/b&gt;", "</b>")

async def _send(m: types.Message, text: str):
    """Бережно отправляем длинный текст частями в HTML-режиме."""
    for chunk in split_for_telegram(text or "", 3900):
        await m.answer(_to_html(chunk), parse_mode="HTML", disable_web_page_preview=True)

# summarizer (мягкий импорт)
try:
    from .summarizer import is_summary_intent, overview_context
except Exception:
    def is_summary_intent(text: str) -> bool:
        return bool(re.search(r"\b(суть|кратко|основн|главн|summary|overview|итог|вывод)\w*\b",
                              text or "", re.IGNORECASE))

    def overview_context(owner_id: int, doc_id: int, max_chars: int = 6000) -> str:
        con = get_conn()
        cur = con.cursor()
        cur.execute(
            """
            SELECT page, section_path, text
            FROM chunks
            WHERE owner_id=? AND doc_id=? 
              AND (text LIKE '[Заголовок]%%'
                   OR text LIKE '%%Цель%%'
                   OR text LIKE '%%Задач%%'
                   OR text LIKE '%%Введен%%'
                   OR text LIKE '%%Заключен%%'
                   OR text LIKE '%%Вывод%%')
            ORDER BY page ASC, id ASC
            LIMIT 14
            """,
            (owner_id, doc_id),
        )
        rows = cur.fetchall()
        con.close()
        if not rows:
            return ""
        parts, total = [], 0
        for r in rows:
            block = f"{(r['text'] or '').strip()}"
            if total + len(block) > max_chars:
                break
            parts.append(block)
            total += len(block)
        return "\n\n".join(parts)

# ГОСТ-валидатор (мягкий импорт)
try:
    from .validators_gost import validate_gost, render_report
except Exception:
    validate_gost = None
    render_report = None


bot = Bot(Cfg.TG_TOKEN)
dp = Dispatcher()

# «Активный документ» в памяти процесса
ACTIVE_DOC: dict[int, int] = {}  # user_id -> doc_id


# ------------------------ Гардрейлы ------------------------

_BANNED_PATTERNS = [
    r"jail ?break|system\s*prompt|developer\s*mode|dan\b|ignore (all|previous) (rules|instructions)",
    r"\bвзлом|хаки?|кейген|кряк|социальн(ая|ые) инженерия",
    r"\bвирус|вредонос|эксплойт|ботнет|ddos\b",
    r"\bоружи|взрывчат|бомб|наркот|порно|эротик|18\+",
    r"\bпаспорт|снилс|инн\b.*(сгенер|поддел)",
    r"\bобой(д|ти)\b.*антиплаг|антиплагиат|antiplagiat",
    r"sql.?инъек|инъекци(я|и) sql",
]

def safety_check(text: str) -> str | None:
    t = (text or "").lower()
    for p in _BANNED_PATTERNS:
        if re.search(p, t, flags=re.IGNORECASE):
            return ("Запрос нарушает правила безопасности "
                    "(взлом/вредонос/обход ограничений/NSFW/личные данные).")
    return None

_ALLOWED_HINT_WORDS = [
    "вкр", "диплом", "курсов", "методолог", "литератур", "литобзор",
    "гипотез", "цель", "задач", "введение", "заключен", "обзор",
    "оформлен", "гост", "таблиц", "рисунк", "антиплаг", "плагиат",
    "презентац", "защиту", "опрос", "анкета", "методы", "статистик",
]

def topical_check(text: str) -> str | None:
    """
    Мягкое тематическое ограничение — используем ТОЛЬКО как подсказку,
    даже если нет активного документа.
    """
    t = (text or "").lower()
    if not any(w in t for w in _ALLOWED_HINT_WORDS):
        return ("Подсказка: я сильнее отвечаю по теме ВКР (структура, методология, ГОСТ, "
                "литобзор, антиплагиат). Если пришлёте файл диплома — смогу отвечать по содержанию.")
    return None


# --------------------- БД / утилиты ---------------------

def _table_has_columns(con, table: str, cols: list[str]) -> bool:
    cur = con.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    have = {row[1] for row in cur.fetchall()}
    return all(c in have for c in cols)

def _find_existing_doc(con, owner_id: int, sha256: str | None, file_uid: str | None):
    """Поиск уже загруженного документа пользователя по sha/file_unique_id (если колонки есть)."""
    if not _table_has_columns(con, "documents", ["content_sha256", "file_uid"]):
        return None
    cur = con.cursor()
    if sha256 and file_uid:
        cur.execute(
            "SELECT id FROM documents WHERE owner_id=? AND (content_sha256=? OR file_uid=?) "
            "ORDER BY id DESC LIMIT 1",
            (owner_id, sha256, file_uid),
        )
    elif sha256:
        cur.execute(
            "SELECT id FROM documents WHERE owner_id=? AND content_sha256=? "
            "ORDER BY id DESC LIMIT 1",
            (owner_id, sha256),
        )
    elif file_uid:
        cur.execute(
            "SELECT id FROM documents WHERE owner_id=? AND file_uid=? "
            "ORDER BY id DESC LIMIT 1",
            (owner_id, file_uid),
        )
    row = cur.fetchone()
    return (row["id"] if row else None)

def _insert_document(con, owner_id: int, kind: str, path: str,
                     sha256: str | None, file_uid: str | None) -> int:
    cur = con.cursor()
    if _table_has_columns(con, "documents", ["content_sha256", "file_uid"]):
        cur.execute(
            "INSERT INTO documents(owner_id, kind, path, content_sha256, file_uid) VALUES(?,?,?,?,?)",
            (owner_id, kind, path, sha256, file_uid),
        )
    else:
        cur.execute(
            "INSERT INTO documents(owner_id, kind, path) VALUES(?,?,?)",
            (owner_id, kind, path),
        )
    doc_id = cur.lastrowid
    con.commit()
    return doc_id


# --------------------- Таблицы: парсинг/нормализация ---------------------

_TABLE_ANY = re.compile(r"\bтаблиц\w*|\bтабл\.\b|\bтаблица\w*|(?:^|\s)table(s)?\b", re.IGNORECASE)
_TABLE_TITLE_RE = re.compile(r"(?i)\bтаблица\s+(\d+(?:[.,]\d+)*)(?:\s*[—\-–]\s*(.+))?")
_COUNT_HINT = re.compile(r"\bсколько\b|how many", re.IGNORECASE)
_WHICH_HINT = re.compile(r"\bкаки(е|х)\b|\bсписок\b|\bперечисл\w*\b|\bназов\w*\b", re.IGNORECASE)

def _plural_tables(n: int) -> str:
    n_abs = abs(n) % 100
    n1 = n_abs % 10
    if 11 <= n_abs <= 14:
        return "таблиц"
    if n1 == 1:
        return "таблица"
    if 2 <= n1 <= 4:
        return "таблицы"
    return "таблиц"

def _strip_table_prefix(s: str) -> str:
    return re.sub(r"^\[\s*таблица\s*\]\s*", "", s or "", flags=re.IGNORECASE)

def _last_segment(name: str) -> str:
    s = (name or "").strip()
    if "/" in s:
        s = s.split("/")[-1].strip()
    s = _strip_table_prefix(s)
    s = re.sub(r"\s*[-–—]\s*", " — ", s)
    s = re.sub(r"\s+", " ", s).strip(" .")
    return s

def _parse_table_title(text: str) -> tuple[str | None, str | None]:
    t = (text or "").strip()
    m = _TABLE_TITLE_RE.search(t)
    if not m:
        return (None, None)
    num = (m.group(1) or "").strip() or None
    title = (m.group(2) or "").strip() or None
    return (num, title)

def _shorten(s: str, limit: int = 120) -> str:
    s = (s or "").strip()
    if len(s) <= limit:
        return s
    return s[:limit - 1].rstrip() + "…"


# -------- Таблицы: подсчёт и список (совместимо со старыми БД) --------

def _distinct_table_basenames(uid: int, doc_id: int) -> list[str]:
    """
    Собираем «базовые» имена таблиц (section_path без хвоста ' [row …]').
    Работает и с новыми индексами (table_row) и со старыми.
    """
    con = get_conn()
    cur = con.cursor()

    # сначала пробуем опереться на типы
    if _table_has_columns(con, "chunks", ["element_type"]):
        cur.execute(
            """
            SELECT DISTINCT
                CASE
                    WHEN instr(section_path, ' [row ')>0
                        THEN substr(section_path, 1, instr(section_path,' [row ')-1)
                    ELSE section_path
                END AS base_name
            FROM chunks
            WHERE doc_id=? AND owner_id=? AND element_type IN ('table','table_row')
            """,
            (doc_id, uid),
        )
    else:
        # очень старый индекс — эвристика
        cur.execute(
            """
            SELECT DISTINCT
                CASE
                    WHEN instr(section_path, ' [row ')>0
                        THEN substr(section_path, 1, instr(section_path,' [row ')-1)
                    ELSE section_path
                END AS base_name
            FROM chunks
            WHERE doc_id=? AND owner_id=? AND (
                  lower(section_path) LIKE '%таблица%'
               OR lower(text)        LIKE '%таблица%'
               OR section_path LIKE 'Таблица %' COLLATE NOCASE
               OR text        LIKE '[Таблица]%' COLLATE NOCASE
               OR lower(section_path) LIKE '%table %'
               OR lower(text)        LIKE '%table %'
            )
            """,
            (doc_id, uid),
        )

    base_items = [r["base_name"] for r in cur.fetchall() if r["base_name"]]
    con.close()
    base_items = sorted(set(base_items), key=lambda s: s.lower())
    return base_items

def _count_tables(uid: int, doc_id: int) -> int:
    return len(_distinct_table_basenames(uid, doc_id))

def _compose_display_from_attrs(attrs_json: str | None, base: str, first_row_text: str | None) -> str:
    """
    Правила отображения:
      1) есть caption_num → 'Таблица N — tail' (если tail есть; иначе просто 'Таблица N').
      2) нет номера → показываем только описание: caption_tail или header_preview.
      3) фолбэк: парсим номер и хвост из base ('Таблица N — ...') и показываем с номером.
      4) если ничего не вышло — берём первую строку или короткий base без служебных слов.
    """
    num = None
    tail = None
    header_preview = None
    if attrs_json:
        try:
            a = json.loads(attrs_json or "{}")
            num = a.get("caption_num") or a.get("label")
            tail = a.get("caption_tail") or a.get("title")
            header_preview = a.get("header_preview")
        except Exception:
            pass

    if num:
        num = str(num).replace(",", ".").strip()
        if tail:
            return f"Таблица {num} — {_shorten(str(tail), 160)}"
        if header_preview:
            return f"Таблица {num} — {_shorten(str(header_preview), 160)}"
        return f"Таблица {num}"

    # без номера в attrs — пробуем распарсить из base и показать С номером
    num_b, title_b = _parse_table_title(_last_segment(base))
    if num_b:
        return f"Таблица {num_b}" + (f" — {_shorten(title_b, 160)}" if title_b else "")

    # без номера — только описание
    if tail:
        return _shorten(str(tail), 160)
    if header_preview:
        return _shorten(str(header_preview), 160)

    if first_row_text:
        return _shorten(first_row_text, 160)

    s = _last_segment(base)
    s = re.sub(r"(?i)^\s*таблица\s+\d+(?:\.\d+)*\s*", "", s).strip(" —–-")
    return _shorten(s or "Таблица", 160)


# ------------------------------ Источники ------------------------------

_SOURCES_HINT = re.compile(
    r"\b(источник(?:и|ов)?|список\s+литературы|библиограф\w*|references?)\b",
    re.IGNORECASE
)
_REF_LINE_RE = re.compile(r"^\s*(?:\[\d+\]|\d+[.)])\s+.+", re.MULTILINE)

def _count_sources(uid: int, doc_id: int) -> int:
    con = get_conn()
    cur = con.cursor()
    has_type = _table_has_columns(con, "chunks", ["element_type"])
    total = 0

    if has_type:
        cur.execute(
            "SELECT COUNT(*) AS c FROM chunks WHERE owner_id=? AND doc_id=? AND element_type='reference'",
            (uid, doc_id),
        )
        row = cur.fetchone()
        total = int(row["c"] or 0)

    if total == 0:
        # fallback: ищем нумерованные строки в разделе «источники»
        items = set()
        cur.execute(
            "SELECT section_path, text FROM chunks WHERE owner_id=? AND doc_id=? ORDER BY page ASC, id ASC",
            (uid, doc_id),
        )
        for r in cur.fetchall():
            sec = (r["section_path"] or "").lower()
            t = (r["text"] or "")
            if ("источник" in sec or "литератур" in sec or "библиограф" in sec or "reference" in sec):
                for line in _REF_LINE_RE.findall(t):
                    items.add(re.sub(r"\s+", " ", line).strip().rstrip(".").lower())
        if not items:
            cur.execute("SELECT text FROM chunks WHERE owner_id=? AND doc_id=?", (uid, doc_id))
            for r in cur.fetchall():
                t = (r["text"] or "")
                for line in _REF_LINE_RE.findall(t):
                    items.add(re.sub(r"\s+", " ", line).strip().rstrip(".").lower())
        total = len(items)

    con.close()
    return total


# --------- быстрый ответ: наличие практической части ---------
_PRACTICAL_Q = re.compile(r"(есть ли|наличие|присутствует ли|имеется ли).{0,40}практическ", re.IGNORECASE)

def _has_practical_part(uid: int, doc_id: int) -> bool:
    con = get_conn()
    cur = con.cursor()
    cur.execute(
        """
        SELECT 1
        FROM chunks
        WHERE owner_id=? AND doc_id=? AND (
            lower(section_path) LIKE '%практическ%' OR
            lower(text)         LIKE '%практическ%'
        )
        LIMIT 1
        """,
        (uid, doc_id),
    )
    row = cur.fetchone()
    con.close()
    return row is not None


# ------------- ГОСТ-интент и проверка ------------- 

_GOST_HINT = re.compile(r"\b(гост|оформлени|шрифт|межстроч|кегл|выравнивани|поля|оформить)\w*\b", re.IGNORECASE)

async def _maybe_run_gost(m: types.Message, uid: int, doc_id: int, text: str) -> bool:
    """Если похоже, что просят проверку оформления — запускаем валидатор ГОСТ. Возвращаем True, если ответили."""
    if not validate_gost or not render_report:
        return False
    if not _GOST_HINT.search(text or ""):
        return False

    con = get_conn()
    cur = con.cursor()
    cur.execute("SELECT path FROM documents WHERE id=? AND owner_id=?", (doc_id, uid))
    row = cur.fetchone()
    con.close()
    if not row:
        return False

    path = row["path"]
    try:
        sections = _parse_by_ext(path)
    except Exception:
        return False

    report = validate_gost(sections)
    text_rep = render_report(report, max_issues=25)
    await _send(m, text_rep)
    return True

# ------------------------------ helpers ------------------------------

def _parse_by_ext(path: str) -> list[dict]:
    fname = (os.path.basename(path) or "").lower()
    if fname.endswith(".docx"):
        return parse_docx(path)
    if fname.endswith(".doc"):
        return parse_doc(path)
    if fname.endswith(".pdf"):
        return parse_pdf(path)
    raise RuntimeError("Поддерживаю .doc, .docx и .pdf.")

def _first_chunks_context(owner_id: int, doc_id: int, n: int = 10, max_chars: int = 6000) -> str:
    con = get_conn()
    cur = con.cursor()
    cur.execute(
        "SELECT page, section_path, text FROM chunks "
        "WHERE owner_id=? AND doc_id=? "
        "ORDER BY page ASC, id ASC LIMIT ?",
        (owner_id, doc_id, n)
    )
    rows = cur.fetchall()
    con.close()
    if not rows:
        return ""
    parts, total = [], 0
    for r in rows:
        block = f"{(r['text'] or '').strip()}"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)

# ---------- verbatim fallback по цитате (шинглы + LIKE/NOCASE) ----------

def _normalize_for_like(s: str) -> str:
    s = (s or "")
    s = s.replace("\u00A0", " ")  # NBSP -> пробел
    s = s.replace("«", '"').replace("»", '"').replace("“", '"').replace("”", '"')
    s = s.replace("’", "'").replace("‘", "'")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _make_shingles(s: str, min_len: int = 30, max_len: int = 90, step: int = 25) -> list[str]:
    s = _normalize_for_like(s)
    if not s:
        return []
    if len(s) <= max_len:
        return [s]
    out = []
    i = 0
    while i < len(s):
        chunk = s[i:i + max_len]
        if len(chunk) >= min_len:
            out.append(chunk)
        i += step
    return out[:6]

def verbatim_find(owner_id: int, doc_id: int, q_text: str, max_hits: int = 3) -> list[dict]:
    shingles = _make_shingles(q_text)
    if not shingles:
        return []
    con = get_conn()
    cur = con.cursor()
    hits: list[dict] = []
    for sh in shingles:
        pattern = f"%{sh}%"
        cur.execute(
            """
            SELECT page, section_path, text FROM chunks
            WHERE owner_id=? AND doc_id=? AND
                  replace(text, char(160), ' ') LIKE ? COLLATE NOCASE
            ORDER BY page ASC, id ASC
            LIMIT ?
            """,
            (owner_id, doc_id, pattern, max_hits - len(hits)),
        )
        for r in cur.fetchall():
            t = (r["text"] or "")
            t_norm = _normalize_for_like(t)
            pos = t_norm.lower().find(_normalize_for_like(sh).lower())
            if pos >= 0:
                s = max(pos - 120, 0)
                e = min(pos + len(sh) + 120, len(t_norm))
                hits.append({
                    "page": r["page"],
                    "section_path": r["section_path"],
                    "snippet": t_norm[s:e].strip(),
                })
            if len(hits) >= max_hits:
                con.close()
                return hits
    con.close()
    return hits


# ------------------------------ /start ------------------------------

@dp.message(Command("start"))
async def start(m: types.Message):
    ensure_user(str(m.from_user.id))
    await _send(m,
        "Привет! Я ассистент по ВКР. Пришли файл диплома (.doc/.docx/.pdf) — я его проиндексирую и буду отвечать по содержанию.\n"
        "Можно прикрепить подпись-вопрос к файлу или задавать вопросы позже отдельными сообщениями.\n"
        "Без файла отвечаю как агент по ВКР (структура, методология, ГОСТ, антиплагиат, подготовка к защите)."
    )


# ======================= НОВАЯ МУЛЬТИ-ИНТЕНТ ЛОГИКА =======================

_NUM_IN_TEXT = re.compile(r"(?i)\bтабл(?:ица)?\.?\s*(\d+(?:[.,]\d+)*)\b")

def _detect_intents(text: str) -> dict:
    """
    Из одного сообщения вытаскиваем все задачи, чтобы ответ был один, но полный.
    """
    t = (text or "").strip()
    intents = {
        "language": "ru",
        "tables": {"want": False, "count": False, "list": False, "describe": [], "limit": 25},
        "sources": {"want": False, "count": False, "list": False, "limit": 25},
        "figures": {"want": False, "list": False, "limit": 25},
        "summary": bool(is_summary_intent(t)),
        "practical": bool(_PRACTICAL_Q.search(t or "")),
        "general_question": None,
    }

    # Язык (очень грубо)
    if re.search(r"[a-z]{3,}", t) and not re.search(r"[а-я]{3,}", t, re.IGNORECASE):
        intents["language"] = "en"

    # Таблицы
    if _TABLE_ANY.search(t):
        intents["tables"]["want"] = True
        if _COUNT_HINT.search(t):
            intents["tables"]["count"] = True
        if _WHICH_HINT.search(t) or re.search(r"\b(какие таблиц|список таблиц)\b", t, re.IGNORECASE):
            intents["tables"]["list"] = True
        # Явные номера таблиц
        nums = [n.replace(",", ".") for n in _NUM_IN_TEXT.findall(t)]
        if nums:
            intents["tables"]["describe"] = sorted(set(nums), key=lambda x: [int(p) if p.isdigit() else p for p in x.split(".")])

    # Источники
    if _SOURCES_HINT.search(t):
        intents["sources"]["want"] = True
        if _COUNT_HINT.search(t):
            intents["sources"]["count"] = True
        if _WHICH_HINT.search(t) or "список" in t.lower():
            intents["sources"]["list"] = True

    # Фигуры (простая эвристика)
    if re.search(r"\bрисунк\w*|figure\b", t, re.IGNORECASE):
        intents["figures"]["want"] = True
        if _WHICH_HINT.search(t):
            intents["figures"]["list"] = True

    # Остаток как общий вопрос
    intents["general_question"] = t

    logging.debug("INTENTS: %s", json.dumps(intents, ensure_ascii=False))
    return intents


def _gather_facts(uid: int, doc_id: int, intents: dict) -> dict:
    """
    Собираем ТОЛЬКО факты из БД/индекса, без генерации текста.
    """
    facts: dict[str, object] = {"doc_id": doc_id}

    # ----- Таблицы -----
    if intents["tables"]["want"]:
        # count
        total_tables = _count_tables(uid, doc_id)
        # list -> массив "Таблица N — Название"
        basenames = _distinct_table_basenames(uid, doc_id)
        con = get_conn()
        cur = con.cursor()
        items: list[str] = []
        for base in basenames:
            # attrs из первой строки table_row (в новых индексах они тут)
            cur.execute(
                """
                SELECT attrs FROM chunks
                WHERE owner_id=? AND doc_id=? AND element_type='table_row'
                  AND section_path LIKE ? || ' [row %'
                ORDER BY id ASC LIMIT 1
                """,
                (uid, doc_id, base),
            )
            r = cur.fetchone()
            attrs_json = r["attrs"] if r else None

            # первая строка — fallback
            cur.execute(
                """
                SELECT text FROM chunks
                WHERE owner_id=? AND doc_id=? AND element_type='table_row'
                  AND section_path LIKE ? || ' [row %'
                ORDER BY id ASC LIMIT 1
                """,
                (uid, doc_id, base),
            )
            r_first = cur.fetchone()
            first_row_text = (r_first["text"] or "").split("\n")[0] if r_first and r_first["text"] else None
            if first_row_text:
                first_row_text = " — ".join([c.strip() for c in first_row_text.split(" | ") if c.strip()])

            title = _compose_display_from_attrs(attrs_json, base, first_row_text)
            title = _strip_table_prefix(title)
            items.append(title)

        con.close()
        facts["tables"] = {
            "count": total_tables,
            "list": items[:intents["tables"]["limit"]],
            "more": max(0, len(items) - intents["tables"]["limit"]),
            "describe": [],
        }

        # describe по конкретным номерам
        desc_cards = []
        if intents["tables"]["describe"]:
            con = get_conn()
            cur = con.cursor()
            for num in intents["tables"]["describe"]:

                # по attrs
                like1 = f'%\"caption_num\": \"{num}\"%'
                like2 = f'%\"label\": \"{num}\"%'
                cur.execute(
                    """
                    SELECT page, section_path, attrs FROM chunks
                    WHERE owner_id=? AND doc_id=? AND element_type IN ('table','table_row')
                      AND (attrs LIKE ? OR attrs LIKE ?)
                    ORDER BY id ASC LIMIT 1
                    """,
                    (uid, doc_id, like1, like2),
                )
                row = cur.fetchone()

                if not row:
                    # по имени секции
                    cur.execute(
                        """
                        SELECT page, section_path, attrs FROM chunks
                        WHERE owner_id=? AND doc_id=? AND element_type IN ('table','table_row')
                          AND section_path LIKE ? COLLATE NOCASE
                        ORDER BY id ASC LIMIT 1
                        """,
                        (uid, doc_id, f'%Таблица {num}%'),
                    )
                    row = cur.fetchone()

                if not row:
                    continue

                # заголовки/хвосты
                attrs_json = row["attrs"] if row else None
                # 1-2 первых строки как highlights
                cur.execute(
                    """
                    SELECT text FROM chunks
                    WHERE owner_id=? AND doc_id=? AND element_type='table_row'
                      AND (section_path=? OR section_path LIKE ? || ' [row %')
                    ORDER BY id ASC LIMIT 2
                    """,
                    (uid, doc_id, row["section_path"], row["section_path"]),
                )
                rows = cur.fetchall()
                highlights = []
                for r in rows or []:
                    first_line = (r["text"] or "").split("\n")[0]
                    if first_line:
                        highlights.append(" — ".join([c.strip() for c in first_line.split(" | ") if c.strip()]))

                # заголовок для карточки
                base = row["section_path"]
                first_row_text = highlights[0] if highlights else None
                display = _compose_display_from_attrs(attrs_json, base, first_row_text)
                display = _strip_table_prefix(display)

                desc_cards.append({
                    "num": num,
                    "display": display,
                    "where": {"page": row["page"], "section_path": row["section_path"]},
                    "highlights": highlights,
                })
            con.close()

        facts["tables"]["describe"] = desc_cards

    # ----- Источники -----
    if intents["sources"]["want"]:
        # Собираем массив строк источников (как в старой _list_sources, но возвращаем данные)
        con = get_conn()
        cur = con.cursor()
        has_type = _table_has_columns(con, "chunks", ["element_type", "attrs"])
        items: list[tuple[int | None, str]] = []

        if has_type:
            cur.execute(
                "SELECT text, attrs FROM chunks WHERE owner_id=? AND doc_id=? AND element_type='reference' ORDER BY id ASC",
                (uid, doc_id),
            )
            for r in cur.fetchall():
                txt = (r["text"] or "").strip()
                idx = None
                attrs_val = r["attrs"] if r is not None else None
                if attrs_val:
                    try:
                        a = json.loads(attrs_val)
                        v = a.get("ref_index")
                        idx = int(v) if isinstance(v, int) or (isinstance(v, str) and str(v).isdigit()) else None
                    except Exception:
                        idx = None
                items.append((idx, txt))
            items.sort(key=lambda x: (999999 if x[0] is None else x[0]))
        else:
            cur.execute(
                "SELECT section_path, text FROM chunks WHERE owner_id=? AND doc_id=? ORDER BY page ASC, id ASC",
                (uid, doc_id),
            )
            for r in cur.fetchall():
                sec = (r["section_path"] or "").lower()
                t = (r["text"] or "")
                if ("источник" in sec or "литератур" in sec or "библиограф" in sec or "reference" in sec):
                    for line in _REF_LINE_RE.findall(t):
                        line = re.sub(r"\s+", " ", line).strip().rstrip(".")
                        m = re.match(r"^\s*(?:\[(\d+)\]|(\d+)[.)])\s+(.*)$", line)
                        if m:
                            n = m.group(1) or m.group(2)
                            try:
                                idx = int(n)
                            except Exception:
                                idx = None
                            items.append((idx, m.group(3).strip()))
        con.close()

        # дедуп по тексту
        seen = set()
        list_sources = []
        for idx, txt in items:
            k = (txt or "").strip().lower()
            if not k or k in seen:
                continue
            seen.add(k)
            list_sources.append(txt)

        facts["sources"] = {
            "count": len(list_sources),
            "list": list_sources[:intents["sources"]["limit"]],
            "more": max(0, len(list_sources) - intents["sources"]["limit"]),
        }

    # ----- Практическая часть -----
    if intents.get("practical"):
        facts["practical_present"] = _has_practical_part(uid, doc_id)

    # ----- Summary -----
    if intents.get("summary"):
        s = overview_context(uid, doc_id, max_chars=6000) or _first_chunks_context(uid, doc_id, n=12, max_chars=6000)
        if s:
            facts["summary_text"] = s

    # ----- Общий вопрос: контекст + цитаты -----
    if intents.get("general_question"):
        vb = verbatim_find(uid, doc_id, intents["general_question"], max_hits=3)
        ctx = best_context(uid, doc_id, intents["general_question"], max_chars=6000)
        if not ctx:
            hits = retrieve(uid, doc_id, intents["general_question"], top_k=8)
            if hits:
                ctx = build_context(hits)
        if not ctx:
            ctx = _first_chunks_context(uid, doc_id, n=10, max_chars=6000)
        if ctx:
            facts["general_ctx"] = ctx
        if vb:
            facts["verbatim_hits"] = vb

    # логируем маленький срез фактов (без огромных текстов)
    log_snapshot = dict(facts)
    if "general_ctx" in log_snapshot and isinstance(log_snapshot["general_ctx"], str):
        log_snapshot["general_ctx"] = log_snapshot["general_ctx"][:300] + "…" if len(log_snapshot["general_ctx"]) > 300 else log_snapshot["general_ctx"]
    if "summary_text" in log_snapshot and isinstance(log_snapshot["summary_text"], str):
        log_snapshot["summary_text"] = log_snapshot["summary_text"][:300] + "…" if len(log_snapshot["summary_text"]) > 300 else log_snapshot["summary_text"]
    logging.debug("FACTS: %s", json.dumps(log_snapshot, ensure_ascii=False))
    return facts


_RULES_MD = (
    "1) Ответь одним сообщением, закрой все подпункты вопроса.\n"
    "2) Заголовки таблиц: если есть номер → «Таблица N — Название»; если номера нет — только название.\n"
    "3) Не выводи служебные метки и размеры (никаких [Таблица], «ряд 1», «(6×7)»).\n"
    "4) В списках покажи не более 25 строк, затем «… и ещё M», если есть.\n"
    "5) Не придумывай факты вне блока Facts; если данных нет — скажи честно.\n"
)

def _compose_answer(question: str, facts: dict, lang: str = "ru") -> str:
    """Готовим markdown-контекст для модели и просим её красиво «сшить» ответ."""
    def md_list(arr: list[str], max_show: int, more: int | None) -> str:
        out = []
        for x in (arr or [])[:max_show]:
            out.append(f"- {x}")
        if more and more > 0:
            out.append(f"… и ещё {more}")
        return "\n".join(out)

    parts = []

    # Таблицы
    tables = facts.get("tables") or {}
    if tables:
        block = []
        if "count" in tables:
            block.append(f"count: {tables.get('count', 0)}")
        if tables.get("list"):
            block.append("list:\n" + md_list(tables["list"], 25, tables.get("more", 0)))
        if tables.get("describe"):
            cards = []
            for c in tables["describe"]:
                cards.append({
                    "num": c.get("num"),
                    "display": c.get("display"),
                    "where": c.get("where"),
                    "highlights": c.get("highlights", [])[:2],
                })
            block.append("describe:\n" + json.dumps(cards, ensure_ascii=False, indent=2))
        parts.append("- Tables:\n  " + "\n  ".join(block))

    # Источники
    sources = facts.get("sources") or {}
    if sources:
        block = []
        block.append(f"count: {sources.get('count', 0)}")
        if sources.get("list"):
            block.append("list:\n" + md_list(sources["list"], 25, sources.get("more", 0)))
        parts.append("- Sources:\n  " + "\n  ".join(block))

    # Практическая часть
    if "practical_present" in facts:
        parts.append(f"- PracticalPartPresent: {bool(facts['practical_present'])}")

    # Краткое содержание (если просили)
    if "summary_text" in facts:
        parts.append("- Summary:\n  " + (facts["summary_text"][:1200] + ("…" if len(facts["summary_text"]) > 1200 else "")).replace("\n", "\n  "))

    # Вербатим-цитаты
    if facts.get("verbatim_hits"):
        hits_md = []
        for h in facts["verbatim_hits"]:
            page = h.get('page')
            sec = (h.get('section_path') or "").strip()
            page_str = (str(page) if page is not None else "?")
            where = f'в разделе «{sec}», стр. {page_str}' if sec else f'на стр. {page_str}'
            hits_md.append(f"- Match {where}: «{h['snippet']}»")
        parts.append("- Citations:\n  " + "\n  ".join(hits_md))

    # Общий контекст (для ответа на общий вопрос)
    if "general_ctx" in facts:
        parts.append("- Context:\n  " + (facts["general_ctx"][:1500] + ("…" if len(facts["general_ctx"]) > 1500 else "")).replace("\n", "\n  "))

    facts_md = "[Facts]\n" + "\n".join(parts) + "\n\n[Rules]\n" + _RULES_MD

    # Один вызов модели — она пишет красивый связный ответ на основе фактов.
    reply = ace_once(question, facts_md)
    return reply


# ------------------------------ основной ответчик ------------------------------

async def respond_with_answer(m: types.Message, uid: int, doc_id: int, q_text: str):
    q_text = (q_text or "").strip()
    logging.debug(f"Получен запрос от пользователя: {q_text}")
    if not q_text:
        await _send(m, "Вопрос пустой. Напишите, что именно вас интересует по ВКР.")
        return

    # Безопасность — всегда
    viol = safety_check(q_text)
    if viol:
        await _send(m, viol + " Задайте корректный вопрос по ВКР.")
        return

    # ГОСТ-проверка по запросу
    if await _maybe_run_gost(m, uid, doc_id, q_text):
        return

    # Единый мульти-интент пайплайн
    intents = _detect_intents(q_text)
    facts = _gather_facts(uid, doc_id, intents)
    reply = _compose_answer(q_text, facts, lang=intents.get("language", "ru"))
    await _send(m, reply)


# ------------------------------ эмбеддинг-профиль ------------------------------

def _current_embedding_profile() -> str:
    dim = probe_embedding_dim(None)
    if dim:
        return f"emb={Cfg.POLZA_EMB}|dim={dim}"
    return f"emb={Cfg.POLZA_EMB}"

def _needs_reindex_by_embeddings(con, doc_id: int) -> bool:
    if not _table_has_columns(con, "documents", ["layout_profile"]):
        return True
    cur = con.cursor()
    cur.execute("SELECT layout_profile FROM documents WHERE id=?", (doc_id,))
    row = cur.fetchone()
    stored = (row["layout_profile"] or "") if row else ""
    if not stored:
        return True
    cur_model = Cfg.POLZA_EMB.strip().lower()
    stored_model = ""
    stored_dim = None
    for part in stored.split("|"):
        part = (part or "").strip().lower()
        if part.startswith("emb="):
            stored_model = part[4:]
        if part.startswith("dim="):
            try:
                stored_dim = int(part[4:])
            except Exception:
                stored_dim = None
    if stored_model and stored_model != cur_model:
        return True
    cur_dim = probe_embedding_dim(None)
    if cur_dim and stored_dim and stored_dim != cur_dim:
        return True
    return False


# ------------------------------ загрузка файла ------------------------------

@dp.message(F.document)
async def handle_doc(m: types.Message):
    uid = ensure_user(str(m.from_user.id))
    doc = m.document

    # 1) скачиваем
    file = await bot.get_file(doc.file_id)
    stream = await bot.download_file(file.file_path)
    data = stream.read()
    stream.close()

    # 2) дедуп по содержимому + file_unique_id
    sha256 = sha256_bytes(data)
    file_uid = getattr(doc, "file_unique_id", None)

    con = get_conn()
    existing_id = _find_existing_doc(con, uid, sha256, file_uid)

    if existing_id:
        existing_ver = get_document_indexer_version(existing_id) or 0

        need_reindex = False
        try:
            if _needs_reindex_by_embeddings(con, existing_id):
                need_reindex = True
        except Exception:
            need_reindex = True

        if existing_ver < CURRENT_INDEXER_VERSION:
            need_reindex = True

        if need_reindex:
            filename = safe_filename(f"{m.from_user.id}_{doc.file_name}")
            path = save_upload(data, filename, Cfg.UPLOAD_DIR)
            update_document_meta(existing_id, path=path, content_sha256=sha256, file_uid=file_uid)
            con.close()

            try:
                sections = _parse_by_ext(path)
                if sum(len(s.get("text") or "") for s in sections) < 500:
                    await _send(m, "Похоже, файл пустой или это скан-PDF без текста.")
                    return

                delete_document_chunks(existing_id, uid)
                index_document(uid, existing_id, sections)
                invalidate_cache(uid, existing_id)

                set_document_indexer_version(existing_id, CURRENT_INDEXER_VERSION)
                update_document_meta(existing_id, layout_profile=_current_embedding_profile())

            except Exception as e:
                await _send(m, f"Не удалось переиндексировать документ #{existing_id}: {e}")
                return

            ACTIVE_DOC[uid] = existing_id
            caption = (m.caption or "").strip()
            await _send(m, f"Документ #{existing_id} переиндексирован. Готов отвечать.")
            if caption:
                await respond_with_answer(m, uid, existing_id, caption)
            return

        con.close()
        ACTIVE_DOC[uid] = existing_id
        caption = (m.caption or "").strip()
        await _send(m, f"Этот файл уже загружен ранее как документ #{existing_id}. Использую его.")
        if caption:
            await respond_with_answer(m, uid, existing_id, caption)
        return

    # 3) сохраняем
    filename = safe_filename(f"{m.from_user.id}_{doc.file_name}")
    path = save_upload(data, filename, Cfg.UPLOAD_DIR)

    # 4) парсим
    try:
        sections = _parse_by_ext(path)
    except Exception as e:
        await _send(m, f"Не удалось обработать файл: {e}")
        return

    # 5) пустой/скан
    if sum(len(s.get("text") or "") for s in sections) < 500:
        await _send(m, "Похоже, файл пустой или это скан-PDF без текста. "
                       "Загрузите, пожалуйста, DOC/DOCX или текстовый PDF.")
        return

    # 6) документ → БД и индексация
    kind = infer_doc_kind(doc.file_name)
    doc_id = _insert_document(con, uid, kind, path, sha256, file_uid)
    con.close()

    index_document(uid, doc_id, sections)
    invalidate_cache(uid, doc_id)
    set_document_indexer_version(doc_id, CURRENT_INDEXER_VERSION)
    update_document_meta(doc_id, layout_profile=_current_embedding_profile())

    ACTIVE_DOC[uid] = doc_id

    caption = (m.caption or "").strip()
    if caption:
        await _send(m, f"Документ #{doc_id} проиндексирован. Отвечаю на ваш вопрос из подписи…")
        await respond_with_answer(m, uid, doc_id, caption)
    else:
        await _send(m, f"Готово. Документ #{doc_id} проиндексирован. Можете задавать вопросы по работе.")


# ------------------------------ обычный текст ------------------------------

@dp.message(F.text & ~F.via_bot)
async def qa(m: types.Message):
    uid = ensure_user(str(m.from_user.id))
    doc_id = ACTIVE_DOC.get(uid)
    text = (m.text or "").strip()

    if not doc_id:
        # Мягкая подсказка, но не блокируем ответ
        hint = topical_check(text)
        if hint:
            await _send(m, hint)
        reply = agent_no_context(text)
        await _send(m, reply)
        return

    await respond_with_answer(m, uid, doc_id, text)
