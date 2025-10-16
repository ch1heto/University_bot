from __future__ import annotations
import re
from typing import List, Dict, Any, Optional, Tuple

from .db import get_conn
from .polza_client import chat_with_gpt  # Используем GPT-5 для генерации резюме

# ----------------------- СИСТЕМНЫЙ ПРОМТ ДЛЯ РЕЗЮМЕ -----------------------

SYS_SUMMARY = (
    "Ты русскоязычный ассистент по ВКР. Тебе даны фрагменты дипломной работы (контекст). "
    "Сделай краткое, но содержательное резюме работы БЕЗ выдумываний, опираясь только на эти фрагменты.\n\n"
    "Формат:\n"
    "— 2–3 предложения с сутью работы (тема, цель, объект/предмет, общий подход).\n"
    "— 4–7 пунктов: задачи, данные/методы, ключевые результаты/числа, вклад.\n"
    "— (если есть) 2–4 пункта по таблицам/рисункам, что в них показано.\n"
    "— В конце перечисли источники вида [Источник N], которые ты использовал.\n\n"
    "Правила:\n"
    "- Никаких фактов вне контекста; если не хватает данных — скажи об этом кратко.\n"
    "- Не придумывай номера страниц/таблиц/рисунков; не используй внешние знания.\n"
    "- Пиши чётко и по делу."
)

# ----------------------- ВСПОМОГАЛКИ ДЛЯ БД/ПОИСКА -----------------------

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

# ----------------------- СБОР СНИППЕТОВ ДЛЯ ОБЗОРА -----------------------

# Сбор всего текста документа
def _collect_full_text(owner_id: int, doc_id: int) -> str:
    """
    Собирает весь текст документа, включая все части (главы, параграфы, таблицы и т. д.).
    """
    con = get_conn()
    cur = con.cursor()
    cur.execute(
        "SELECT text FROM chunks WHERE owner_id=? AND doc_id=? ORDER BY page ASC, id ASC",
        (owner_id, doc_id)
    )
    rows = cur.fetchall()
    con.close()

    # Собираем весь текст в одну строку
    full_text = "\n\n".join([row["text"] for row in rows if row["text"]])

    return full_text

# ----------------------- ПОСТРОЕНИЕ КОНТЕКСТА -----------------------

def _clean(s: str) -> str:
    return (s or "").replace("\xa0", " ").strip()

def _build_context_block(rows: List[Dict[str, Any]], max_chars: int = 6000) -> str:
    """
    Формирует контекст вида:
    [Источник N] ...текст... (стр. X • Раздел)
    """
    parts, total = [], 0
    for i, r in enumerate(rows, 1):
        page = r.get("page")
        sec = r.get("section_path")
        text = _clean(r.get("text") or "")
        block = f"[Источник {i}] {text}  (стр. {page} • {sec})"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)

def make_full_context(owner_id: int, doc_id: int, max_chars: int = 6000) -> str:
    """
    Формирует полный контекст, включающий весь текст документа.
    """
    full_text = _collect_full_text(owner_id, doc_id)

    # Ограничиваем размер контекста
    return full_text[:max_chars]  # Чтобы не превысить лимит символов

# ----------------------- СБОР СНИППЕТОВ ДЛЯ ОБЗОРА -----------------------

# ключевые маркеры для «основы» резюме
_KEY_HINTS = [
    "Цель", "Задач", "Объект", "Предмет", "Гипотез",
    "Метод", "Методик", "Материал", "Данные", "Вывод", "Заключен", "Введен"
]
_KEY_RE = re.compile("|".join([fr"{h}" for h in _KEY_HINTS]), re.IGNORECASE)

def _collect_headings(owner_id: int, doc_id: int, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Возвращает заголовки в исходном порядке. Работает и со старым индексом:
    — новый индекс: element_type='heading'
    — старый индекс: text LIKE '[Заголовок]%'
    """
    con = get_conn()
    has_type = _has_columns(con, "chunks", ["element_type"])
    if has_type:
        rows = _fetchall(
            con,
            "SELECT id, page, section_path, text FROM chunks "
            "WHERE owner_id=? AND doc_id=? AND element_type='heading' "
            "ORDER BY page ASC, id ASC",
            (owner_id, doc_id)
        )
    else:
        rows = _fetchall(
            con,
            "SELECT id, page, section_path, text FROM chunks "
            "WHERE owner_id=? AND doc_id=? AND text LIKE '[Заголовок]%' "
            "ORDER BY page ASC, id ASC",
            (owner_id, doc_id)
        )
    con.close()
    return rows[:limit]

def _collect_keylines(owner_id: int, doc_id: int, limit: int = 30) -> List[Dict[str, Any]]:
    """
    Ищет строки с целями/задачами/методами/выводами и т.п.
    """
    con = get_conn()
    rows = _fetchall(
        con,
        "SELECT id, page, section_path, text FROM chunks "
        "WHERE owner_id=? AND doc_id=? AND (text LIKE '%'||?||'%' OR text LIKE '%'||?||'%' "
        "OR text LIKE '%'||?||'%' OR text LIKE '%'||?||'%' OR text LIKE '%'||?||'%' "
        "OR text LIKE '%'||?||'%' OR text LIKE '%'||?||'%' OR text LIKE '%'||?||'%' "
        "OR text LIKE '%'||?||'%' OR text LIKE '%'||?||'%') "
        "ORDER BY page ASC, id ASC "
        "LIMIT ?",
        (
            owner_id, doc_id,
            "Цель", "Задач", "Объект", "Предмет", "Гипотез",
            "Метод", "Методик", "Данные", "Вывод", "Заключен",
            limit
        )
    )
    con.close()
    # допфильтрация по regex, чтобы убрать случайные совпадения
    out = []
    for r in rows:
        t = (r.get("text") or "")
        if _KEY_RE.search(t):
            out.append(r)
    return out[:limit]

def _collect_tables(owner_id: int, doc_id: int, max_tables: int = 8, sample_rows: int = 2) -> List[Dict[str, Any]]: 
    """
    Собирает список таблиц (уникальные base_name) и несколько первых строк каждой.
    Совместимо со старым индексом: ищем префикс '[Таблица]' и section_path с суффиксом ' [row N]'.
    """
    con = get_conn()
    has_type = _has_columns(con, "chunks", ["element_type"])

    # шаг 1: собрать базовые имена таблиц
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

    # шаг 2: для каждой таблицы — взять первые sample_rows с [row 1], [row 2], ...
    result: List[Dict[str, Any]] = []
    cur = get_conn().cursor()
    for name in base_names:
        samples: List[str] = []
        for r in range(1, sample_rows + 1):
            cur.execute(
                "SELECT text FROM chunks "
                "WHERE owner_id=? AND doc_id=? AND section_path=? "
                "ORDER BY id ASC LIMIT 1",
                (owner_id, doc_id, f"{name} [row {r}]")
            )
            row = cur.fetchone()
            if row and row["text"]:
                samples.append(row["text"])
        result.append({"name": name, "samples": samples})
    cur.connection.close()

    return result

def _collect_figures(owner_id: int, doc_id: int, max_figs: int = 8) -> List[Dict[str, Any]]: 
    """
    Грубый сбор «рисунков» — по префиксу '[Рисунок]' (совместимость со старым индексом).
    """
    con = get_conn()
    has_type = _has_columns(con, "chunks", ["element_type"])
    if has_type:
        rows = _fetchall(
            con,
            "SELECT DISTINCT section_path FROM chunks "
            "WHERE owner_id=? AND doc_id=? AND element_type='figure' "
            "ORDER BY section_path ASC LIMIT ?",
            (owner_id, doc_id, max_figs)
        )
    else:
        rows = _fetchall(
            con,
            "SELECT DISTINCT section_path FROM chunks "
            "WHERE owner_id=? AND doc_id=? AND text LIKE '[Рисунок]%' "
            "ORDER BY section_path ASC LIMIT ?",
            (owner_id, doc_id, max_figs)
        )
    con.close()
    return [{"name": r["section_path"]} for r in rows if r.get("section_path")]

# ----------------------- ВЫСОКОУРОВНЕВЫЕ API -----------------------

def summarize_document(owner_id: int, doc_id: int) -> str:
    """
    Делаем краткое резюме дипломной работы:
    — строим полный контекст,
    — отправляем в модель с SYS_SUMMARY,
    — возвращаем ответ.
    """
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
    return reply
