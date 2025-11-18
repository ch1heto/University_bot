# app/summarizer.py
from __future__ import annotations
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from .db import get_conn
from .polza_client import chat_with_gpt  # только GPT для текстовых резюме

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
