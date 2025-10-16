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
    _mk_table_pattern, _mk_figure_pattern, keyword_find,
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
        # fallback-обзор через заголовки/цель/задачи (без ссылочных пометок)
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
    """Тематическое ограничение — используем ТОЛЬКО когда нет активного документа."""
    t = (text or "").lower()
    if not any(w in t for w in _ALLOWED_HINT_WORDS):
        return ("Я отвечаю только по теме ВКР: структура, методология, оформление по ГОСТ, "
                "литобзор, антиплагиат, подготовка к защите.")
    return None


# --------------------- БД / дедуп ---------------------

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


# ------------- Подсказки: таблицы и источники -------------

_TABLE_QUERY_HINT = re.compile(r"\bтаблиц", re.IGNORECASE)

def _plural_tables(n: int) -> str:
    # простая русская морфология для "таблица"
    n_abs = abs(n) % 100
    n1 = n_abs % 10
    if 11 <= n_abs <= 14:
        return "таблиц"
    if n1 == 1:
        return "таблица"
    if 2 <= n1 <= 4:
        return "таблицы"
    return "таблиц"

def _last_segment(name: str) -> str:
    """Берём последний сегмент после / и чистим мусор."""
    s = (name or "").strip()
    if "/" in s:
        s = s.split("/")[-1].strip()
    # убираем префикс [Таблица] если такой есть
    if s.startswith("[Таблица]"):
        s = s[len("[Таблица]"):].strip()
    # нормализуем тире
    s = re.sub(r"\s*[-–—]\s*", " — ", s)
    s = re.sub(r"\s+", " ", s).strip(" .")
    return s

_TABLE_TITLE_RE = re.compile(r"(?i)\bтаблица\s+(\d+(?:\.\d+)*)(?:\s*[—\-–]\s*(.+))?")

def _parse_table_title(text: str) -> tuple[str | None, str | None]:
    """
    Вернёт (номер, заголовок) если удалось распарсить.
    """
    t = (text or "").strip()
    m = _TABLE_TITLE_RE.search(t)
    if not m:
        return (None, None)
    num = m.group(1)
    title = (m.group(2) or "").strip()
    return (num, title or None)

def _shorten(s: str, limit: int = 120) -> str:
    s = (s or "").strip()
    if len(s) <= limit:
        return s
    return s[:limit-1].rstrip() + "…"

async def _list_tables_if_asked(uid: int, doc_id: int, q_text: str) -> str | None:
    """
    На общий вопрос «какие/сколько таблиц» выдаём короткие строки:
    • Таблица 2.1 — Название (r×c)
    """
    if not _TABLE_QUERY_HINT.search(q_text) or re.search(r"\b\d+([.,]\d+)?\b", q_text):
        return None

    con = get_conn()
    cur = con.cursor()
    has_type = _table_has_columns(con, "chunks", ["element_type", "attrs"])
    if has_type:
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
                  text LIKE '[Таблица]%' COLLATE NOCASE
               OR section_path LIKE 'Таблица %' COLLATE NOCASE
               OR lower(text) LIKE '%таблица%'
               OR lower(section_path) LIKE '%таблица%'
               OR lower(text) LIKE '%table %'
               OR lower(section_path) LIKE '%table %'
            )
            """,
            (doc_id, uid),
        )
    base_items = [r["base_name"] for r in cur.fetchall() if r["base_name"]]
    base_items = sorted(set(base_items), key=lambda s: s.lower())

    lines = []
    for base in base_items:
        # пробуем вытащить «короткий заголовок» и размеры
        short = _last_segment(base)

        num, title = (None, None)
        attrs = None
        # сперва возьмём корневой элемент таблицы, если есть
        cur.execute(
            """
            SELECT text, attrs FROM chunks
            WHERE owner_id=? AND doc_id=? AND element_type='table'
              AND (section_path=? OR section_path LIKE ? || ' [row %')
            ORDER BY id ASC LIMIT 1
            """,
            (uid, doc_id, base, base),
        )
        row = cur.fetchone()
        sample_text = row["text"] if row else ""
        if row and row["attrs"]:
            try:
                attrs = json.loads(row["attrs"])
            except Exception:
                attrs = None

        # если из текста ничего не поняли — попробуем из base/short
        num, title = _parse_table_title(sample_text)
        if not num:
            num, title = _parse_table_title(short)
        if not num:
            # как минимум вернём короткое имя
            display = short
        else:
            display = f"Таблица {num}" + (f" — {title}" if title else "")

        # добавим краткое описание по размерам (если знаем)
        if isinstance(attrs, dict):
            r = attrs.get("n_rows")
            c = attrs.get("n_cols")
            if isinstance(r, int) and isinstance(c, int) and r > 0 and c > 0:
                display += f" ({r}×{c})"

        lines.append("• " + _shorten(display, 140))

    con.close()

    if not lines:
        return "В документе таблиц не найдено."

    total = len(lines)
    shown = lines[:30]
    tail = total - len(shown)
    header = f"Нашёл {total} { _plural_tables(total) }:"
    if tail > 0:
        shown.append(f"… и ещё {tail}")
    return header + "\n" + "\n".join(shown)


# --- Источники (список литературы / references) ---

_SOURCES_HINT = re.compile(
    r"\b(источник(?:и|ов)?|список\s+литературы|библиограф\w*|references?)\b",
    re.IGNORECASE
)
_REF_LINE_RE = re.compile(r"^\s*(?:\[\d+\]|\d+\.)\s+.+", re.MULTILINE)

async def _list_sources_if_asked(uid: int, doc_id: int, q_text: str) -> str | None:
    if not _SOURCES_HINT.search(q_text or ""):
        return None

    con = get_conn()
    cur = con.cursor()

    # 1) сначала пробуем разделы с соответствующим названием
    cur.execute(
        """
        SELECT text FROM chunks
        WHERE owner_id=? AND doc_id=? AND (
            lower(section_path) LIKE '%источник%'
         OR lower(section_path) LIKE '%литератур%'
         OR lower(section_path) LIKE '%библиограф%'
         OR lower(section_path) LIKE '%reference%'
        )
        ORDER BY page ASC, id ASC
        """,
        (uid, doc_id),
    )
    rows = cur.fetchall()
    items: list[str] = []
    for r in rows:
        t = (r["text"] or "").strip()
        if not t:
            continue
        # из блока текста вытащим «построчно» библиографические записи
        for line in _REF_LINE_RE.findall(t):
            line = re.sub(r"\s+", " ", line).strip().rstrip(".")
            items.append(line)

    # 2) если раздел не нашли — пробуем во всём документе по шаблону ссылок
    if not items:
        cur.execute(
            "SELECT text FROM chunks WHERE owner_id=? AND doc_id=? ORDER BY page ASC, id ASC",
            (uid, doc_id),
        )
        for r in cur.fetchall():
            t = (r["text"] or "")
            for line in _REF_LINE_RE.findall(t):
                line = re.sub(r"\s+", " ", line).strip().rstrip(".")
                items.append(line)

    con.close()

    # нормализуем и отфильтруем дубли
    seen = set()
    normed = []
    for it in items:
        k = it.lower()
        if k in seen:
            continue
        seen.add(k)
        normed.append(it)

    if not normed:
        return "Раздел «Источники/Список литературы» не найден или пуст."

    total = len(normed)
    shown = normed[:30]
    tail = total - len(shown)
    body = "\n".join(f"{i+1}. {shown[i]}" for i in range(len(shown)))
    header = f"Нашёл {total} источников:"
    if tail > 0:
        body += f"\n… и ещё {tail}"
    return header + "\n" + body


# --------- быстрый ответ: наличие практической части ---------
_PRACTICAL_Q = re.compile(r"(есть ли|наличие|присутствует ли|имеется ли).{0,40}практическ", re.IGNORECASE)

def _has_practical_part(uid: int, doc_id: int) -> bool:
    """Есть ли в документе явные упоминания практической части (в тексте или названиях разделов)."""
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
    """Супер-fallback: берём первые фрагменты по порядку страниц (без ссылочных хвостов)."""
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
    """
    Ищем несколько шинглов вопроса как подстроки в chunks.text (LIKE, NOCASE, с заменой NBSP).
    Возвращаем [{page, section_path, snippet}]
    """
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

    # Быстрый ответ: «есть ли практическая часть?»
    if _PRACTICAL_Q.search(q_text):
        if _has_practical_part(uid, doc_id):
            await _send(m, "В документе найдены упоминания практической части (по вхождению «практическ…» в тексте/разделах).")
        else:
            await _send(m, "В документе явных упоминаний практической части не найдено.")
        return

    # --- СПИСОК ИСТОЧНИКОВ ---
    src_hint = await _list_sources_if_asked(uid, doc_id, q_text)
    if src_hint:
        await _send(m, src_hint)
        return  # завершаем — пользователю нужен именно список источников

    # --- СПИСОК ТАБЛИЦ ---
    tbl_hint = await _list_tables_if_asked(uid, doc_id, q_text)
    if tbl_hint:
        await _send(m, tbl_hint)
        return  # завершаем — не запускаем лишний RAG/ACE

    # 0) Точный verbatim-fallback по цитате (до всего остального)
    vb_hits = verbatim_find(uid, doc_id, q_text)
    if vb_hits:
        parts = []
        for h in vb_hits:
            page = h.get('page')
            sec = (h.get('section_path') or "").strip()
            page_str = (str(page) if page is not None else "?")
            where = f'в разделе «{sec}», стр. {page_str}' if sec else f'на стр. {page_str}'
            parts.append(f"Нашёл совпадение {where}:\n«{h['snippet']}»")
        await _send(m, "\n\n".join(parts))
        return

    # 1) keyword-fallback по номерам таблиц/рисунков
    pat = _mk_table_pattern(q_text) or _mk_figure_pattern(q_text)
    if pat:
        hits_kw = keyword_find(uid, doc_id, pat)
        if hits_kw:
            parts = []
            for h in hits_kw:
                page = h.get('page')
                sec = (h.get('section_path') or "").strip()
                page_str = (str(page) if page is not None else "?")
                where = f'в разделе «{sec}», стр. {page_str}' if sec else f'на стр. {page_str}'
                parts.append(f"Нашёл упоминание {where}:\n«{h['snippet']}»")
            await _send(m, "\n\n".join(parts))
            return

    # 2) «суть/кратко» — обзорный контекст и сразу ACE
    if is_summary_intent(q_text):
        ctx = overview_context(uid, doc_id, max_chars=6000)
        if not ctx:
            ctx = _first_chunks_context(uid, doc_id, n=12, max_chars=6000)
        if ctx:
            reply = ace_once(q_text, ctx)
            await _send(m, reply)
            return

    # 3) Гибридный контекст (lex + semantic)
    ctx = best_context(uid, doc_id, q_text, max_chars=6000)

    # 4) fallback: чистый retrieve БЕЗ жёсткого порога
    if not ctx:
        hits = retrieve(uid, doc_id, q_text, top_k=8)
        if hits:
            ctx = build_context(hits)

    # 5) супер-fallback: первые куски по страницам
    if not ctx:
        ctx = _first_chunks_context(uid, doc_id, n=10, max_chars=6000)

    if not ctx:
        await _send(m, "В загруженной работе нет достаточных оснований ответить. "
                       "Уточните раздел или добавьте нужные фрагменты.")
        return

    # 6) Ответ — ACE (строго по контексту)
    reply = ace_once(q_text, ctx)
    await _send(m, reply)


# ------------------------------ эмбеддинг-профиль ------------------------------

def _current_embedding_profile() -> str:
    """
    Подпись текущей эмбеддинг-модели для сохранения в documents.layout_profile.
    Формат: 'emb=<MODEL>|dim=<D>' (dim может отсутствовать, если не удалось пробить).
    """
    dim = probe_embedding_dim(None)
    if dim:
        return f"emb={Cfg.POLZA_EMB}|dim={dim}"
    return f"emb={Cfg.POLZA_EMB}"

def _needs_reindex_by_embeddings(con, doc_id: int) -> bool:
    """
    Возвращает True, если документ должен быть переиндексирован из-за смены
    модели/размерности эмбеддингов (или отсутствия профиля).
    """
    if not _table_has_columns(con, "documents", ["layout_profile"]):
        return True  # старая БД — лучше переиндексировать
    cur = con.cursor()
    cur.execute("SELECT layout_profile FROM documents WHERE id=?", (doc_id,))
    row = cur.fetchone()
    stored = (row["layout_profile"] or "") if row else ""
    if not stored:
        return True  # профиль не записан — обновим
    # сравним модель
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
    # сравним размерность, если можем её пробить
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

        # проверка профиля эмбеддингов
        need_reindex = False
        try:
            if _needs_reindex_by_embeddings(con, existing_id):
                need_reindex = True
        except Exception:
            # на всякий случай — если что-то пошло не так с проверкой, безопаснее переиндексировать
            need_reindex = True

        if existing_ver < CURRENT_INDEXER_VERSION:
            need_reindex = True

        if need_reindex:
            # обновим путь/хэши и переиндексируем
            filename = safe_filename(f"{m.from_user.id}_{doc.file_name}")
            path = save_upload(data, filename, Cfg.UPLOAD_DIR)
            update_document_meta(existing_id, path=path, content_sha256=sha256, file_uid=file_uid)
            con.close()

            try:
                sections = _parse_by_ext(path)
                if sum(len(s.get("text") or "") for s in sections) < 500:
                    await _send(m, "Похоже, файл пустой или это скан-PDF без текста.")
                    return

                # очистим старые чанки и проиндексируем заново
                delete_document_chunks(existing_id, uid)
                index_document(uid, existing_id, sections)
                invalidate_cache(uid, existing_id)

                # обновим версии/профиль
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
        # Агент без документа: safety + тематичность
        viol = safety_check(text) or topical_check(text)
        if viol:
            await _send(m, viol + " Я могу помочь по структуре/методологии/ГОСТ/антиплагиату и т.п.")
            return

        reply = agent_no_context(text)
        await _send(m, reply)
        return

    # Есть документ — обычный путь с гибридным поиском + ACE
    await respond_with_answer(m, uid, doc_id, text)
