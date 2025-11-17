# app/summarizer.py
from __future__ import annotations
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from .db import get_conn
from .polza_client import chat_with_gpt, vision_describe  # текст + vision-описания
from .vision_analyzer import analyze_figure as va_analyze_figure  # единый анализатор диаграмм

# ----------------------- СИСТЕМНЫЙ ПРОМТ ДЛЯ РЕЗЮМЕ -----------------------

SYS_SUMMARY = (
    "Ты русскоязычный РЕПЕТИТОР по ВКР. Тебе даны фрагменты дипломной работы (контекст). "
    "Сделай краткое, но содержательное резюме, опираясь ТОЛЬКО на эти фрагменты, без внешних знаний.\n\n"
    "Формат:\n"
    "— 2–3 предложения: тема, цель, объект/предмет, общий подход.\n"
    "— 4–7 пунктов: задачи, данные/методы, ключевые результаты/числа, вклад.\n"
    "— (если есть) 2–4 пункта по таблицам/рисункам — что именно в них показано.\n"
    "— В конце перечисли использованные маркеры контекста вида [Источник N].\n\n"
    "Правила:\n"
    "- Не выдумывай фактов; если сведений мало — явно укажи, чего не хватает.\n"
    "- Не придумывай номера страниц/таблиц/рисунков.\n"
    "- Пиши по-русски, ясно и по делу."
)

# ----------------------- МАЛЫЕ УТИЛИТЫ ДЛЯ БД -----------------------

def _has_columns(con, table: str, cols: List[str]) -> bool:
    cur = con.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    have = {row[1] for row in cur.fetchall()}
    return all(c in have for c in cols)

def _fetchall(con, sql: str, params: Tuple[Any, ...]) -> List[Dict[str, Any]]:
    cur = con.cursor()
    cur.execute(sql, params)
    rows = cur.fetchall()
    return [dict(r) for r in rows]

def _fetchone(con, sql: str, params: Tuple[Any, ...]) -> Optional[Dict[str, Any]]:
    cur = con.cursor()
    cur.execute(sql, params)
    row = cur.fetchone()
    return (dict(row) if row else None)

# убираем служебные префиксы ([Таблица]/[Рисунок]/[Заголовок]), хвосты вида "[row 1]" и размеры "(6×7)"
_CLEAN_MARKERS_RE = re.compile(
    r"^\s*\[(?:таблица|рисунок|заголовок|страница)\]\s*|"
    r"\s*\[\s*row\s*\d+\s*\]\s*|"
    r"\s*\(\d+\s*[×x]\s*\d+\)\s*$",
    re.IGNORECASE
)
def _clean(s: str) -> str:
    t = (s or "").replace("\xa0", " ")
    t = _CLEAN_MARKERS_RE.sub("", t)
    t = t.replace("–", "-").replace("—", "-")
    t = re.sub(r"\s+\|\s+", " — ", t)   # "a | b" -> "a — b"
    return re.sub(r"\s+", " ", t).strip()

# ----------------------- СБОР «ПОЛНОГО» ТЕКСТА -----------------------

def _collect_full_text(owner_id: int, doc_id: int) -> str:
    """Собирает весь текст документа в исходном порядке."""
    con = get_conn()
    cur = con.cursor()
    cur.execute(
        "SELECT text FROM chunks WHERE owner_id=? AND doc_id=? ORDER BY page ASC, id ASC",
        (owner_id, doc_id)
    )
    rows = cur.fetchall()
    con.close()
    return "\n\n".join([_clean(row["text"]) for row in rows if row["text"]])

def make_full_context(owner_id: int, doc_id: int, max_chars: int = 6000) -> str:
    """Формирует полный контекст (обрезаем до max_chars)."""
    return _collect_full_text(owner_id, doc_id)[:max_chars]

# ----------------------- СЕМАНТИЧЕСКАЯ ВЫБОРКА ДЛЯ ОБЗОРА -----------------------

# Интент на «краткое содержание»
_SUMMARY_HINT_RE = re.compile(
    r"\b(суть|кратко|основн|главн|summary|overview|итог|выводы?|резюме|реферат|аннотац|автореферат)\w*\b",
    re.IGNORECASE
)

def is_summary_intent(text: str) -> bool:
    return bool(_SUMMARY_HINT_RE.search(text or ""))

# Ключевые маркеры по содержанию
_KEY_HINTS = [
    "Цель", "Задач", "Объект", "Предмет", "Гипотез",
    "Метод", "Методик", "Материал", "Данные", "Вывод", "Заключен", "Введен",
    "Актуальн", "Новизн", "Практическ[а-я]* значим"  # частые разделы во ВКР
]
_KEY_RE = re.compile("|".join([fr"{h}" for h in _KEY_HINTS]), re.IGNORECASE)

def _collect_headings(owner_id: int, doc_id: int, limit: int = 50) -> List[Dict[str, Any]]:
    """Заголовки (новые индексы: element_type='heading', старые: префикс)."""
    con = get_conn()
    has_type = _has_columns(con, "chunks", ["element_type"])
    if has_type:
        rows = _fetchall(
            con,
            "SELECT id, page, section_path, text FROM chunks "
            "WHERE owner_id=? AND doc_id=? AND element_type='heading' "
            "ORDER BY page ASC, id ASC LIMIT ?",
            (owner_id, doc_id, limit)
        )
    else:
        rows = _fetchall(
            con,
            "SELECT id, page, section_path, text FROM chunks "
            "WHERE owner_id=? AND doc_id=? AND text LIKE '[Заголовок]%' "
            "ORDER BY page ASC, id ASC LIMIT ?",
            (owner_id, doc_id, limit)
        )
    con.close()
    return rows

def _collect_keylines(owner_id: int, doc_id: int, limit: int = 30) -> List[Dict[str, Any]]:
    """Строки с целями/задачами/методами/выводами и т.п. (быстрые эвристики)."""
    con = get_conn()
    rows = _fetchall(
        con,
        "SELECT id, page, section_path, text FROM chunks "
        "WHERE owner_id=? AND doc_id=? "
        "ORDER BY page ASC, id ASC",
        (owner_id, doc_id)
    )
    con.close()
    out = []
    for r in rows:
        t = (r.get("text") or "")
        if _KEY_RE.search(t):
            out.append(r)
            if len(out) >= limit:
                break
    return out

def _collect_tables(owner_id: int, doc_id: int, max_tables: int = 8, sample_rows: int = 2) -> List[Dict[str, Any]]:
    """Список таблиц + первые строки (совместимость со старыми индексами)."""
    con = get_conn()
    has_type = _has_columns(con, "chunks", ["element_type"])

    if has_type:
        tables = _fetchall(
            con,
            "SELECT DISTINCT "
            "CASE WHEN instr(section_path, ' [row ')>0 "
            "     THEN substr(section_path, 1, instr(section_path,' [row ')-1) "
            "     ELSE section_path END AS base_name "
            "FROM chunks "
            "WHERE owner_id=? AND doc_id=? AND element_type IN ('table','table_row')",
            (owner_id, doc_id)
        )
    else:
        tables = _fetchall(
            con,
            "SELECT DISTINCT "
            "CASE WHEN instr(section_path, ' [row ')>0 "
            "     THEN substr(section_path, 1, instr(section_path,' [row ')-1) "
            "     ELSE section_path END AS base_name "
            "FROM chunks "
            "WHERE owner_id=? AND doc_id=? AND text LIKE '[Таблица]%'",
            (owner_id, doc_id)
        )

    base_names = [t["base_name"] for t in tables if t.get("base_name")]
    base_names = sorted(set(base_names), key=lambda s: s.lower())[:max_tables]

    result: List[Dict[str, Any]] = []
    conn2 = get_conn()
    cur = conn2.cursor()
    for name in base_names:
        samples: List[str] = []
        for r in range(1, sample_rows + 1):
            cur.execute(
                "SELECT text, page FROM chunks WHERE owner_id=? AND doc_id=? AND section_path=? "
                "ORDER BY id ASC LIMIT 1",
                (owner_id, doc_id, f"{name} [row {r}]")
            )
            row = cur.fetchone()
            if row and row["text"]:
                samples.append(_clean(row["text"]))
        result.append({"name": name, "samples": samples})
    conn2.close()
    return result

def _collect_figures(owner_id: int, doc_id: int, max_figs: int = 8) -> List[Dict[str, Any]]:
    """Список «рисунков»: сначала element_type='figure', затем подписи, затем JSON-attrs (caption_num/label)."""
    con = get_conn()
    has_type_and_attrs = _has_columns(con, "chunks", ["element_type", "attrs"])

    rows: List[Dict[str, Any]] = []
    if has_type_and_attrs:
        # 1) normal figure
        rows = _fetchall(
            con,
            "SELECT DISTINCT page, section_path, text, attrs FROM chunks "
            "WHERE owner_id=? AND doc_id=? AND element_type='figure' "
            "ORDER BY id ASC LIMIT ?",
            (owner_id, doc_id, max_figs),
        )

        # 2) подписи RU/EN (с NBSP-нормализацией)
        if not rows:
            rows = _fetchall(
                con,
                "SELECT DISTINCT page, section_path, text, attrs FROM chunks "
                "WHERE owner_id=? AND doc_id=? AND ("
                "  replace(section_path, char(160),' ') LIKE '%Рис%'     COLLATE NOCASE OR "
                "  replace(text,        char(160),' ') LIKE '%Рис%'     COLLATE NOCASE OR "
                "  replace(section_path, char(160),' ') LIKE '%Схем%'    COLLATE NOCASE OR "
                "  replace(text,        char(160),' ') LIKE '%Схем%'    COLLATE NOCASE OR "
                "  replace(section_path, char(160),' ') LIKE '%Картин%'  COLLATE NOCASE OR "
                "  replace(text,        char(160),' ') LIKE '%Картин%'  COLLATE NOCASE OR "
                "  replace(section_path, char(160),' ') LIKE '%Fig%'     COLLATE NOCASE OR "
                "  replace(text,        char(160),' ') LIKE '%Fig%'     COLLATE NOCASE OR "
                "  replace(section_path, char(160),' ') LIKE '%Picture%' COLLATE NOCASE OR "
                "  replace(text,        char(160),' ') LIKE '%Picture%' COLLATE NOCASE OR "
                "  replace(section_path, char(160),' ') LIKE '%Pic%'     COLLATE NOCASE OR "
                "  replace(text,        char(160),' ') LIKE '%Pic%'     COLLATE NOCASE "
                ") ORDER BY id ASC LIMIT ?",
                (owner_id, doc_id, max_figs),
            )

        # 3) JSON-attrs: если есть только номер/лейбл
        if not rows:
            rows = _fetchall(
                con,
                "SELECT DISTINCT page, section_path, text, attrs FROM chunks "
                "WHERE owner_id=? AND doc_id=? AND element_type='figure' "
                "AND (attrs LIKE '%\"caption_num\": %' OR attrs LIKE '%\"label\": %') "
                "ORDER BY id ASC LIMIT ?",
                (owner_id, doc_id, max_figs),
            )
    else:
        # старые индексы: только по подписям, attrs может НЕ существовать
        rows = _fetchall(
            con,
            "SELECT DISTINCT page, section_path, text, NULL AS attrs FROM chunks "
            "WHERE owner_id=? AND doc_id=? AND ("
            "  replace(section_path, char(160),' ') LIKE '%Рис%'     COLLATE NOCASE OR "
            "  replace(text,        char(160),' ') LIKE '%Рис%'     COLLATE NOCASE OR "
            "  replace(section_path, char(160),' ') LIKE '%Схем%'    COLLATE NOCASE OR "
            "  replace(text,        char(160),' ') LIKE '%Схем%'    COLLATE NOCASE OR "
            "  replace(section_path, char(160),' ') LIKE '%Картин%'  COLLATE NOCASE OR "
            "  replace(text,        char(160),' ') LIKE '%Картин%'  COLLATE NOCASE OR "
            "  replace(section_path, char(160),' ') LIKE '%Fig%'     COLLATE NOCASE OR "
            "  replace(text,        char(160),' ') LIKE '%Fig%'     COLLATE NOCASE OR "
            "  replace(section_path, char(160),' ') LIKE '%Picture%' COLLATE NOCASE OR "
            "  replace(text,        char(160),' ') LIKE '%Picture%' COLLATE NOCASE OR "
            "  replace(section_path, char(160),' ') LIKE '%Pic%'     COLLATE NOCASE OR "
            "  replace(text,        char(160),' ') LIKE '%Pic%'     COLLATE NOCASE "
            ") ORDER BY id ASC LIMIT ?",
            (owner_id, doc_id, max_figs),
        )

    con.close()
    return rows


def _select_first_paragraphs(owner_id: int, doc_id: int, limit: int = 12) -> List[Dict[str, Any]]:
    """Ранние абзацы как общий фолбэк контекста."""
    con = get_conn()
    has_type = _has_columns(con, "chunks", ["element_type"])
    if has_type:
        rows = _fetchall(
            con,
            "SELECT page, section_path, text FROM chunks "
            "WHERE owner_id=? AND doc_id=? AND (element_type IS NULL OR element_type IN ('paragraph','text','page')) "
            "ORDER BY page ASC, id ASC LIMIT ?",
            (owner_id, doc_id, limit)
        )
    else:
        rows = _fetchall(
            con,
            "SELECT page, section_path, text FROM chunks "
            "WHERE owner_id=? AND doc_id=? "
            "ORDER BY page ASC, id ASC LIMIT ?",
            (owner_id, doc_id, limit)
        )
    con.close()
    # уберём пустые
    return [r for r in rows if (r.get("text") or "").strip()][:limit]

def _build_context_block(rows: List[Dict[str, Any]], max_chars: int = 6000) -> str:
    """
    Формирует контекст вида:
    [Источник N] ...текст...
    (стр. X • Раздел)
    """
    parts: List[str] = []
    total = 0
    i = 1
    for r in rows:
        page = r.get("page")
        sec = (r.get("section_path") or "").strip()
        text = _clean(r.get("text") or "")
        if not text:
            continue
        block = f"[Источник {i}] {text}"
        loc = []
        if page is not None:
            loc.append(f"стр. {page}")
        if sec:
            loc.append(sec)
        if loc:
            block += f"  ({' • '.join(loc)})"
        # лимит
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
        i += 1
    return "\n\n".join(parts)

def overview_context(owner_id: int, doc_id: int, max_chars: int = 6000) -> str:
    """
    «Умный» контекст для резюме: заголовки, цели/задачи/методы/выводы,
    краткие сведения о таблицах и рисунках + немного первых абзацев.
    Всегда с маркерами [Источник N] для удобного цитирования в ответе.
    """
    rows: List[Dict[str, Any]] = []

    # 1) Заголовки верхних уровней
    rows += _collect_headings(owner_id, doc_id, limit=20)

    # 2) Ключевые строки (цели/задачи/методы/выводы…)
    rows += _collect_keylines(owner_id, doc_id, limit=30)

    # 3) Таблицы: склеим имя секции + первую строку
    for t in _collect_tables(owner_id, doc_id, max_tables=8, sample_rows=2):
        name = t["name"]
        samples = [s for s in (t.get("samples") or []) if s]
        if not samples:
            continue
        first = samples[0].splitlines()[0]
        rows.append({
            "page": None,
            "section_path": name,
            "text": f"{name}: {first}"
        })

    # 4) Рисунки: возьмём подпись целиком (section_path или text)
    for f in _collect_figures(owner_id, doc_id, max_figs=20):
        cap = (f.get("text") or f.get("section_path") or "").strip()
        if not cap:
            continue
        rows.append({
            "page": f.get("page"),
            "section_path": f.get("section_path"),
            "text": cap
        })

    # 5) Общий фолбэк (первые абзацы)
    rows += _select_first_paragraphs(owner_id, doc_id, limit=12)

    if not rows:
        return ""

    return _build_context_block(rows, max_chars=max_chars)

# ----------------------- РЕЗЮМЕ ДОКУМЕНТА -----------------------

def summarize_document(owner_id: int, doc_id: int) -> str:
    """
    Делаем краткое резюме дипломной работы:
    — строим «умный» контекст overview_context,
    — отправляем в модель с SYS_SUMMARY,
    — возвращаем ответ.
    """
    ctx = overview_context(owner_id, doc_id, max_chars=6000)
    if not ctx.strip():
        # крайний случай — шлём просто первые ~6000 символов
        ctx = make_full_context(owner_id, doc_id, max_chars=6000)

    if not ctx.strip():
        return ("Не удалось собрать достаточно контекста для резюме. "
                "Проверьте, что в работе есть текстовые разделы (цели/задачи/методы/выводы).")

    reply = chat_with_gpt(
        [
            {"role": "system", "content": SYS_SUMMARY},
            {"role": "assistant", "content": f"Контекст:\n{ctx}"},
            {"role": "user", "content": "Сформулируй краткое резюме работы по приведённым фрагментам."}
        ],
        temperature=0.2,
    )
    return (reply or "").strip()

# ======================================================================
#                  VISION: ОПИСАНИЕ РИСУНКОВ ПО НОМЕРАМ
# ======================================================================

# Регекс для «Рисунок 2.1 — ...» / «Рис. 2,1 ...» / «Figure 3»
_FIG_CAP_RE = re.compile(
    r"(?i)\b(?:рис(?:\.|унок)?|схем(?:а|ы)?|картин(?:ка|ки)?|figure|fig\.?|picture|pic\.?)\s*(?:№\s*)?(\d+(?:[.,]\d+)*)"
    r"(?:\s*[—\-–:]\s*(.+))?"
)


def _norm_num(n: Optional[str]) -> Optional[str]:
    if not n:
        return None
    return n.replace(",", ".").strip()

def _parse_figure_title_line(s: str) -> Tuple[Optional[str], Optional[str]]:
    m = _FIG_CAP_RE.search(s or "")
    if not m:
        return (None, None)
    return (_norm_num(m.group(1)), (m.group(2) or "").strip() or None)


def _figure_row_by_number(owner_id: int, doc_id: int, num: str) -> Optional[Dict[str, Any]]:
    """Ищем чанк рисунка по номеру: учитываем NBSP и случаи, когда номер лежит только в attrs (caption_num/label)."""
    con = get_conn()
    want = _norm_num(num) or ""
    row: Optional[Dict[str, Any]] = None

    pats = [
        # RU
        (f"%Рисунок {want}%", f"%Рисунок {want}%"),
        (f"%Рис. {want}%",    f"%Рис. {want}%"),
        (f"%Рис {want}%",     f"%Рис {want}%"),
        (f"%Рисунок{want}%",  f"%Рисунок{want}%"),
        (f"%Рис.{want}%",     f"%Рис.{want}%"),
        (f"%Рис{want}%",      f"%Рис{want}%"),
        (f"%Схема {want}%",   f"%Схема {want}%"),
        (f"%Схема{want}%",    f"%Схема{want}%"),
        (f"%Картинка {want}%",f"%Картинка {want}%"),
        (f"%Картинка{want}%", f"%Картинка{want}%"),
        # EN
        (f"%Figure {want}%",  f"%Figure {want}%"),
        (f"%Figure{want}%",   f"%Figure{want}%"),
        (f"%Fig. {want}%",    f"%Fig. {want}%"),
        (f"%Fig.{want}%",     f"%Fig.{want}%"),
        (f"%Fig {want}%",     f"%Fig {want}%"),
        (f"%Fig{want}%",      f"%Fig{want}%"),
        (f"%Picture {want}%", f"%Picture {want}%"),
        (f"%Picture{want}%",  f"%Picture{want}%"),
        (f"%Pic. {want}%",    f"%Pic. {want}%"),
        (f"%Pic.{want}%",     f"%Pic.{want}%"),
        (f"%Pic {want}%",     f"%Pic {want}%"),
        (f"%Pic{want}%",      f"%Pic{want}%"),
    ]

    has_new = _has_columns(con, "chunks", ["element_type", "attrs"])

    # 1) «Настоящий» figure: сравнение через REPLACE(char(160),' ')
    if has_new:
        for p1, p2 in pats:
            row = _fetchone(
                con,
                "SELECT id, page, section_path, text, attrs FROM chunks "
                "WHERE owner_id=? AND doc_id=? AND element_type='figure' "
                "AND (replace(section_path, char(160),' ') LIKE ? COLLATE NOCASE "
                "     OR replace(text,        char(160),' ') LIKE ? COLLATE NOCASE) "
                "ORDER BY id ASC LIMIT 1",
                (owner_id, doc_id, p1, p2),
            )
            if row:
                break

        # 1.b) JSON-attrs: caption_num/label
        if not row:
            like1 = f'%\"caption_num\": \"{want}\"%'
            like2 = f'%\"label\": \"{want}\"%'
            row = _fetchone(
                con,
                "SELECT id, page, section_path, text, attrs FROM chunks "
                "WHERE owner_id=? AND doc_id=? AND element_type='figure' AND (attrs LIKE ? OR attrs LIKE ?) "
                "ORDER BY id ASC LIMIT 1",
                (owner_id, doc_id, like1, like2),
            )

    # 2) Фолбэк: любая секция с подходящей подписью
    if not row:
        for p1, p2 in pats:
            base_sql = (
                "SELECT id, page, section_path, text, " +
                ("attrs" if has_new else "NULL AS attrs") +
                " FROM chunks WHERE owner_id=? AND doc_id=? "
                "AND (replace(section_path, char(160),' ') LIKE ? COLLATE NOCASE "
                "     OR replace(text,        char(160),' ') LIKE ? COLLATE NOCASE) "
            )
            # В старых схемах element_type может отсутствовать — не используем его в ORDER BY
            order_sql = (
                "ORDER BY CASE WHEN element_type='figure' THEN 0 ELSE 1 END, id ASC LIMIT 1"
                if has_new else
                "ORDER BY id ASC LIMIT 1"
            )
            row = _fetchone(
                con,
                base_sql + order_sql,
                (owner_id, doc_id, p1, p2),
            )
            if row:
                break

    con.close()
    return row


def _extract_images_from_attrs(attrs_raw: Any) -> List[str]:
    """Извлекаем список путей изображений из JSON-attrs."""
    try:
        if not attrs_raw:
            return []
        if isinstance(attrs_raw, dict):
            obj = attrs_raw
        else:
            obj = json.loads(attrs_raw)
        imgs = obj.get("images") or []
        out: List[str] = []
        for p in imgs:
            if not p:
                continue
            sp = str(p)
            if Path(sp).exists():
                out.append(sp)
        return out
    except Exception:
        return []

def _collect_images_for_section(owner_id: int, doc_id: int, section_path: str) -> List[str]:
    """
    Подтягиваем images из attrs по всем чанкам данной секции.
    Нужен для случаев, когда картинки лежат не в других чанках.
    """
    con = get_conn()
    try:
        # В старых схемах колонки attrs может не быть — тогда просто возвращаем пусто
        if not _has_columns(con, "chunks", ["attrs"]):
            return []

        cur = con.cursor()
        cur.execute(
            """
            SELECT attrs FROM chunks
            WHERE owner_id=? AND doc_id=? AND section_path=? AND attrs IS NOT NULL
            ORDER BY id ASC LIMIT 12
            """,
            (owner_id, doc_id, section_path),
        )
        rows = cur.fetchall() or []
    finally:
        con.close()

    images: List[str] = []
    for rr in rows:
        try:
            obj = json.loads(rr["attrs"] or "{}")
        except Exception:
            obj = {}
        for p in (obj.get("images") or []):
            if p and p not in images and Path(str(p)).exists():
                images.append(str(p))
    return images


def _format_figure_sentence(num: Optional[str], vision_text: str, tail: Optional[str]) -> str:
    """
    Финальный формат строки ответа. Мини-очистка вводных конструкций.
    """
    n = num or "—"
    vt = (vision_text or "").strip()
    vt = re.sub(r"(?i)^(на\s+изображении|на\s+рисунке|на\s+фото)\s+(представлен[оы]?|показан[оы]?|изображен[оы]?)\s*", "", vt)
    vt = vt.rstrip(".").strip()

    if vt:
        base = f"На этом рисунке (Рисунок {n}) изображено {vt}."
    elif tail:
        base = f"На этом рисунке (Рисунок {n}) изображено: {tail.strip()}."
    else:
        base = f"На этом рисунке (Рисунок {n}) представлено содержание, соответствующее подписи автора."
    return base

def describe_figures(owner_id: int, doc_id: int, numbers: List[str]) -> str:
    """
    Принимает список номеров рисунков ['2.1', '3', ...] и возвращает по каждому
    короткое академичное описание. Использует общий vision-клиент (polza_client.vision_describe).
    """
    if not numbers:
        return "Не указаны номера рисунков. Например: 2.1, 3.2, 5."

    lines: List[str] = []
    for raw in numbers:
        num = _norm_num(raw)
        row = _figure_row_by_number(owner_id, doc_id, num or "")
        if not row:
            lines.append(f"Рисунок {num or raw}: не найден в документе.")
            continue

        # Подпись-намёк (из section_path или text)
        sec = row.get("section_path") or ""
        txt = row.get("text") or ""
        n_sec, tail_sec = _parse_figure_title_line(sec)
        n_txt, tail_txt = _parse_figure_title_line(txt)
        tail = tail_sec or tail_txt

        # Картинки: из attrs найденного чанка + по всей секции
        img_paths = _extract_images_from_attrs(row.get("attrs"))
        if sec:
            extra = _collect_images_for_section(owner_id, doc_id, sec)
            for p in extra:
                if p not in img_paths:
                    img_paths.append(p)

        # Описание через общий vision-клиент
        vision_text = ""
        if img_paths:
            try:
                desc = vision_describe(image_or_images=img_paths, lang="ru")
                vision_text = (desc.get("description") or "").strip()
            except Exception:
                vision_text = ""

        # Ответная строка
        out = _format_figure_sentence(num or n_sec or n_txt, vision_text, tail)
        if not img_paths and not vision_text:
            out += " (Картинка не была извлечена из файла; описание дано по подписи/контексту.)"

        lines.append(out)

    return "\n".join(lines)

# Обёртка для одного номера
def describe_figure(owner_id: int, doc_id: int, number: str) -> str:
    return describe_figures(owner_id, doc_id, [number])


# ======================================================================
#         НОВОЕ: ВЫТАСКИВАЕМ ТОЧНЫЕ ЗНАЧЕНИЯ С КАРТИНОК (FIG_VALUES)
# ======================================================================


# ======================================================================
#         НОВОЕ: ВЫТАСКИВАЕМ ТОЧНЫЕ ЗНАЧЕНИЯ С КАРТИНОК (FIG_VALUES)
# ======================================================================


def _format_values_markdown(obj: Dict[str, Any]) -> str:
    """
    Простой формат вывода числовых значений диаграммы:
    - Markdown-таблица «Категория | Серия | Значение | Ед.»;
    - CSV для копирования/экспорта.
    Без текстовой интерпретации и без пересказа графика.
    Используем в первую очередь поле exact_numbers (если есть), затем data.
    """
    rows = obj.get("exact_numbers") or obj.get("data") or []
    warnings = obj.get("warnings") or []

    if not rows:
        return "_Нет структурированных числовых данных; возможно, это не диаграмма с числами._"

    lines: List[str] = []
    lines.append("**Точные значения (как в документе):**")
    lines.append("| Категория | Серия | Значение | Ед. |")
    lines.append("|---|---|---|---|")

    csv_lines: List[str] = ["label,series,value,unit"]

    for r in rows:
        label = str(
            r.get("label")
            or r.get("category")
            or r.get("name")
            or r.get("x")
            or ""
        ).strip()
        series = str(
            r.get("series")
            or r.get("group")
            or r.get("legend")
            or ""
        ).strip()

        val = r.get("value")
        if val is None:
            val = r.get("y") or r.get("x") or r.get("count")
        value_str = "" if val is None else str(val)

        unit = str(r.get("unit") or r.get("units") or "").strip()

        lines.append(
            f"| {label or '—'} | {series or '—'} | {value_str or '—'} | {unit or '—'} |"
        )

        def _csv_escape(s: str) -> str:
            return '"' + s.replace('"', '""') + '"'

        csv_lines.append(
            ",".join(
                [
                    _csv_escape(label),
                    _csv_escape(series),
                    _csv_escape(value_str),
                    _csv_escape(unit),
                ]
            )
        )

    lines.append("")
    lines.append("CSV (для экспорта):")
    lines.append("```")
    lines.extend(csv_lines)
    lines.append("```")

    if warnings:
        lines.append("")
        lines.append("Предупреждения: " + "; ".join(str(w) for w in warnings))

    return "\n".join(lines).strip()


def extract_figure_values(owner_id: int, doc_id: int, numbers: List[str]) -> str:
    """
    Публичная функция: извлечь ЧИСЛОВЫЕ значения с указанных рисунков.

    Она НЕ даёт интерпретацию, а возвращает по каждому рисунку блок вида:

    **Рисунок N.** Подпись (если есть)
    **Точные значения (как в документе):**
    | Категория | Серия | Значение | Ед. |
    ...

    Приоритет источников данных:
    1) attrs.chart_matrix / attrs.chart_data (OOXML-данные диаграммы из индекса);
    2) если их нет — va_analyze_figure (vision_analyzer.analyze_figure).
    Никакой генерации GPT здесь нет.
    """
    if not numbers:
        return "Не указаны номера рисунков. Например: 2.1, 3.2, 5."

    outputs: List[str] = []

    for raw in numbers:
        num = _norm_num(raw)
        row = _figure_row_by_number(owner_id, doc_id, num or "")
        if not row:
            outputs.append(f"Рисунок {num or raw}: не найден в документе.")
            continue

        # подпись/название
        sec = row.get("section_path") or ""
        txt = row.get("text") or ""
        n_sec, tail_sec = _parse_figure_title_line(sec)
        n_txt, tail_txt = _parse_figure_title_line(txt)
        title_tail = tail_sec or tail_txt
        shown_num = num or n_sec or n_txt or "—"

        # ---------- 1. Пытаемся вытащить числовые данные напрямую из attrs.chart_matrix / attrs.chart_data ----------
        attrs_raw = row.get("attrs")
        try:
            attrs_obj = attrs_raw if isinstance(attrs_raw, dict) else (json.loads(attrs_raw) if attrs_raw else {})
        except Exception:
            attrs_obj = {}

        chart_matrix = attrs_obj.get("chart_matrix")
        chart_rows = attrs_obj.get("chart_data")

        analysis: Optional[Dict[str, Any]] = None

        # 1.a) Нормализованный matrix → список числовых строк
        if isinstance(chart_matrix, dict):
            cats = chart_matrix.get("categories") or []
            series_list = chart_matrix.get("series") or []
            unit_common = chart_matrix.get("unit")
            exact_rows: List[Dict[str, Any]] = []

            for ri, cat in enumerate(cats):
                cat_label = str(cat).strip() if cat is not None else ""
                for s in series_list or []:
                    vals = (s or {}).get("values") or []
                    unit = (s or {}).get("unit") or unit_common
                    ser_name = (s or {}).get("name")
                    v = vals[ri] if ri < len(vals) else None
                    if v is None:
                        continue
                    exact_rows.append(
                        {
                            "label": cat_label,
                            "series": (ser_name or "").strip(),
                            "value": v,
                            "unit": unit,
                        }
                    )

            if exact_rows:
                analysis = {"exact_numbers": exact_rows}

        # 1.b) Фолбэк: плоский список chart_data (старый/универсальный формат),
        # только если по matrix ничего не собрали
        if not analysis and isinstance(chart_rows, list) and chart_rows:
            exact_rows: List[Dict[str, Any]] = []
            for r in chart_rows:
                if not isinstance(r, dict):
                    continue
                label = (r.get("category") or r.get("label") or "").strip()
                series_name = (r.get("series_name") or r.get("series") or "").strip()
                val = r.get("value")
                unit = r.get("unit")
                if label or (val is not None):
                    exact_rows.append(
                        {
                            "label": label,
                            "series": series_name,
                            "value": val,
                            "unit": unit,
                        }
                    )
            if exact_rows:
                analysis = {"exact_numbers": exact_rows}

        # Если удалось собрать числовые данные из OOXML — сразу формируем таблицу, без vision
        if isinstance(analysis, dict) and (analysis.get("exact_numbers") or analysis.get("data")):
            header = (
                f"**Рисунок {shown_num}.** {title_tail}"
                if title_tail
                else f"**Рисунок {shown_num}.**"
            )
            values_block = _format_values_markdown(analysis)
            outputs.append(header + "\n" + values_block)
            continue

        # ---------- 2. Если в attrs нет структурированных данных — переходим к vision-анализу по картинкам ----------
        # собираем файлы изображений
        img_paths = _extract_images_from_attrs(attrs_obj)
        if sec:
            extra = _collect_images_for_section(owner_id, doc_id, sec)
            for p in extra:
                if p not in img_paths:
                    img_paths.append(p)

        if not img_paths:
            outputs.append(
                f"Рисунок {shown_num}: не удалось извлечь файл изображения "
                f"(могу опираться только на подпись: {title_tail or '—'})."
            )
            continue

        # Берём первую картинку как основную для анализа
                # Берём первую картинку как основную для анализа
        image_path = img_paths[0]

        try:
            # Используем тот же стиль вызова, что и в bot.py,
            # чтобы не ловить TypeError по неправильным именам параметров.
            analysis = va_analyze_figure(
                image_path,
                caption_hint=title_tail,
                lang="ru",
            )
        except Exception:
            analysis = None

        rows = None
        if isinstance(analysis, dict):
            rows = analysis.get("exact_numbers") or analysis.get("data")

        if not rows:
            outputs.append(
                f"Рисунок {shown_num}: не удалось структурировать числовые значения автоматически."
            )
            continue

        header = (
            f"**Рисунок {shown_num}.** {title_tail}"
            if title_tail
            else f"**Рисунок {shown_num}.**"
        )
        values_block = _format_values_markdown(analysis)
        outputs.append(header + "\n" + values_block)

    return "\n\n".join(outputs).strip()


def extract_figure_value(owner_id: int, doc_id: int, number: str) -> str:
    """
    Обёртка для одного номера рисунка.

    ВАЖНО: первый ряд ответа всегда начинается с строки
    '**Рисунок N.** ...', чтобы bot.py мог отделить заголовок от таблицы.
    """
    return extract_figure_values(owner_id, doc_id, [number])
