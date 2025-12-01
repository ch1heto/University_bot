# app/answer_builder.py
from __future__ import annotations

import re
import json
import asyncio
from typing import Dict, List, Any, Optional, AsyncIterable, Iterable, Union

# --- Прямой доступ к модели (без ACE) ---
try:
    from .polza_client import chat_with_gpt, chat_with_gpt_stream  # type: ignore
except Exception:
    chat_with_gpt = None          # type: ignore
    chat_with_gpt_stream = None   # type: ignore

# Опциональные аналитические «подсказки» по таблицам (если модуль есть)
try:
    from .analytics import analyze_table_by_num  # type: ignore
except Exception:
    analyze_table_by_num = None  # type: ignore

# Новые: для полной выгрузки таблиц
from .config import Cfg
from .db import get_conn

# Хелперы распознавания номеров/флагов/подсказок из intents (чтобы не дублировать регексы)
try:
    from .intents import extract_table_numbers, extract_section_hints, extract_figure_numbers  # type: ignore
except Exception:
    extract_table_numbers = None  # type: ignore
    extract_figure_numbers = None  # type: ignore


    # Фолбэк: примитивный парсер подсказок разделов «глава/раздел/§/section 2.2»
    _SECTION_HINT_RE = re.compile(
        r"(?i)\b(глава|раздел|пункт|подраздел|§|section|chapter|clause)\s*([A-Za-zА-Яа-я]?\s*\d+(?:[.,]\d+)*)"
    )

    def _normalize_num_fallback(s: str) -> str:
        s = (s or "").replace(" ", "").replace(",", ".")
        s = re.sub(r"([A-Za-zА-Яа-я])[\.\-]?(?=\d)", r"\1", s)
        return s.strip()

    def extract_section_hints(text: str) -> List[str]:  # type: ignore
        out: List[str] = []
        for m in _SECTION_HINT_RE.findall(text or ""):
            val = _normalize_num_fallback(m[1] or "")
            if val:
                out.append(val)
        # уникализируем, сохраняя порядок
        seen = set()
        uniq: List[str] = []
        for v in out:
            if v not in seen:
                seen.add(v)
                uniq.append(v)
        return uniq

try:
    from .intents import TABLE_ALL_VALUES_RE, TABLE_ROWS_LIMIT_RE, EXACT_NUMBERS_RE  # type: ignore
except Exception:
    TABLE_ALL_VALUES_RE = re.compile(
        r"(?i)\b(все|всю|целиком|полностью|полная)\b.*\b(таблиц\w*|таблица|значени\w*|данн\w*|строк\w*|колон\w*)\b"
        r"|(?:\ball\b.*\b(table|values|rows|columns)\b|\bfull\s+(table|values)\b|\bentire\s+table\b)"
    )
    TABLE_ROWS_LIMIT_RE = re.compile(
        r"(?i)(?:покаж[ий]|выведи|дай|отобрази|верни|первые)\s+(\d{1,4})\s+(строк\w*|rows?)\b"
        r"|(^|\s)(\d{1,4})\s+(строк\w*|rows?)\b"
        r"|top\s+(\d{1,4})\b"
    )
    EXACT_NUMBERS_RE = re.compile(
        r"(?i)(точн\w+\s+как\s+в\s+(документе|тексте|файле)|ровно\s+как\s+в\s+(документе|тексте|файле)|как\s+есть|"
        r"без\s+округлен\w+|не\s*(меняй|изменяй)\s*(формат|запят|разделител)|сохрани\w*\s*(формат|вид|разделител|запят)|"
        r"без\s+нормализ\w+|exact(ly)?\s+as\s+in\s+(doc(ument)?|file|text)|keep\s+format|do\s+not\s+change\s+(format|commas|separators)|"
        r"no\s+round(ing)?|as-is)"
    )

# ----------------------------- Константы лимитов -----------------------------
FULL_TABLE_MAX_ROWS: int = getattr(Cfg, "FULL_TABLE_MAX_ROWS", 500)
FULL_TABLE_MAX_COLS: int = getattr(Cfg, "FULL_TABLE_MAX_COLS", 40)
FULL_TABLE_MAX_CHARS: int = getattr(Cfg, "FULL_TABLE_MAX_CHARS", 20_000)

# Сколько таблиц подтягивать с «главного» раздела, если номер не указан
SECTION_TABLES_MAX: int = getattr(Cfg, "SECTION_TABLES_MAX", 3)

# ----------------------------- Нормализация текста -----------------------------

_NUM_TOKEN = re.compile(r"(?<!\d)(\d{1,3}(?:[ \u00A0]\d{3})+|\d+)([.,]\d+)?\s*(%?)")

def _normalize_numbers(s: str) -> str:
    """
    Нормализация чисел в БЛОКЕ ФАКТОВ (не в финальном ответе!):
      - 12 345 → 12345 (убираем пробелы тысяч);
      - 1,25 → 1.25 (десятичная точка);
      - '  %' → '%'.
    """
    def repl(m: re.Match) -> str:
        int_part = (m.group(1) or "").replace(" ", "").replace("\u00A0", "")
        frac = (m.group(2) or "").replace(",", ".")
        pct = (m.group(3) or "")
        return f"{int_part}{frac}{pct}"

    s = _NUM_TOKEN.sub(repl, s or "")
    # унификация длинных тире/минусов
    s = s.replace("–", "—").replace("--", "—")
    return s

def _shorten(s: str, limit: int = 300) -> str:
    s = (s or "").strip()
    return s if len(s) <= limit else (s[:limit - 1].rstrip() + "…")

def _md_list(arr: List[str], max_show: int, more: Optional[int], *, norm_numbers: bool = True) -> str:
    out = []
    for x in (arr or [])[:max_show]:
        out.append(f"- {_normalize_numbers(x) if norm_numbers else x}")
    if more and more > 0:
        out.append(f"… и ещё {more}")
    return "\n".join(out)

# ----------------------------- Правила для модели -----------------------------

_DEFAULT_RULES = (
    "1) Если в секции [Items] перечислены подпункты — отвечай по КАЖДОМУ подпункту отдельно, строго по порядку, "
    "сохраняя нумерацию 1), 2), 3). Не смешивай ответы по разным подпунктам. "
    "Если подпункты не перечислены — закрой все смысловые пункты вопроса последовательно.\n"
    "2) Не используй Markdown-заголовки (#, ##, ###). Для подзаголовков используй жирный текст и двоеточие: **Пункт 2.3:** ...\n"
    "3) Заголовки таблиц: если есть номер → «Таблица N — Название»; если номера нет — только название. "
    "Заголовки рисунков/диаграмм: если есть номер → «Рисунок N — Название».\n"
    "4) Не выводи служебные метки и размеры (никаких [Таблица], «ряд 1», «(6×7)»).\n"
    "5) В списках покажи не более 25 строк, затем «… и ещё M», если есть.\n"
    "6) Не придумывай факты вне блока Facts (Tables/Figures/TablesRaw/values); если данных нет — скажи честно.\n"
    "7) Если пользователь просит точные значения «как в документе/без нормализации» — сохраняй исходный вид чисел (разделители тысяч, запятая/точка, дефисы).\n"
    "8) Если вопрос про рисунки/диаграммы/графики — опирайся на раздел [Figures] (включая describe, describe_cards и значения диаграмм), "
    "не придумывай числа, которых там нет.\n"
    "9) Для диаграмм и графиков сначала коротко поясни, что показывают оси, легенда и единицы измерения, как устроен масштаб (шаг шкалы, начало отсчёта). "
    "Если в [Figures]/values нет явного описания масштаба, не придумывай его: прямо указывай, что масштаб можно только приблизительно оценить по подписям.\n"
    "10) Структура ответа: сначала дай блок '[документ]' — ответ строго по фактам из [Facts]; затем блок '[модель]' — короткие общие пояснения, "
    "интерпретации и советы по теме вопроса. В блоке '[модель]' не добавляй новых «фактов» про документ, только общий контекст и рекомендации. "
    "Если дополнительных пояснений нет, всё равно выведи строку '[модель] дополнительных комментариев нет'.\n"
    "11) Если в вопросе упоминаются конкретные элементы документа (например, «рисунок 2.2, таблица 2.1 и глава 1»), "
    "оформляй ответ отдельными блоками по каждому объекту: сначала «Рисунок 2.2 — …», затем «Таблица 2.1 — …», затем «Глава 1 — …». "
    "Внутри каждого блока опирайся только на факты, относящиеся к этому объекту.\n"
)

# конвертируем строки-заголовки в **жирный**
_HEAD_TO_BOLD_RE = re.compile(r"(?m)^\s*#{1,6}\s*(.+?)\s*$")
def _headings_to_bold(text: str) -> str:
    return _HEAD_TO_BOLD_RE.sub(lambda m: f"**{m.group(1).strip()}**", text or "")

# ----------------------------- Вспомогалки для табличных данных -----------------------------

_CAP_RE = re.compile(
    r"(?i)\bтаблица\s+([A-Za-zА-Яа-я]?\s*\d+(?:[.,]\d+)*)\b(?:\s*[—\-–:\u2013\u2014]\s*(.+))?"
)

def _normalize_num(s: str) -> str:
    s = (s or "").replace(" ", "").replace(",", ".")
    return s.strip()

# --- НОВОЕ: нормализация номеров рисунков / фокус по одному рисунку ---

def _normalize_fig_num_local(s: str) -> str:
    """
    Локальная нормализация номера рисунка для сравнения:
      - убираем пробелы;
      - запятую → точку;
      - склеиваем литеру и цифру: 'A.1'/'A-1' → 'A1'.
    """
    s = (s or "").strip()
    s = s.replace(" ", "").replace(",", ".")
    s = re.sub(r"([A-Za-zА-Яа-я])[\.\-]?(?=\d)", r"\1", s)
    return s

def _figure_focus_from_facts(facts: Dict[str, Any]) -> List[str]:
    """
    Возвращает нормализованный список номеров рисунков, на которых нужно сфокусироваться.
    Основные сценарии:
      - intents['figures']['single_only'] == True и есть describe с номерами;
      - facts['figures']['single_only'] == True и есть describe_nums;
      - fallback: единственная карточка в describe_cards → её num.
    """
    nums: List[str] = []

    # 1) Из intents, если они проброшены в facts
    try:
        intents = (facts or {}).get("intents") or {}
        fig_i = (intents.get("figures") or {})
        if fig_i.get("single_only") and fig_i.get("describe"):
            for n in fig_i.get("describe") or []:
                v = _normalize_fig_num_local(str(n))
                if v:
                    nums.append(v)
    except Exception:
        pass

    # 2) Из facts['figures'], если туда явно положили флаг/номера
    if not nums:
        try:
            fig_f = (facts or {}).get("figures") or {}
            if fig_f.get("single_only"):
                for n in (fig_f.get("describe_nums") or []):
                    v = _normalize_fig_num_local(str(n))
                    if v:
                        nums.append(v)
        except Exception:
            pass

    # 3) Fallback: только одна карточка рисунка → считаем, что вопрос про неё
    if not nums:
        try:
            fig_f = (facts or {}).get("figures") or {}
            cards = fig_f.get("describe_cards") or []
            if isinstance(cards, list) and len(cards) == 1:
                n = cards[0].get("num")
                if n:
                    v = _normalize_fig_num_local(str(n))
                    if v:
                        nums.append(v)
        except Exception:
            pass

    # дедуп с сохранением порядка
    seen = set()
    out: List[str] = []
    for v in nums:
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out

def _filter_lines_for_figure_focus(lines: List[str], focus_nums: List[str]) -> List[str]:
    """
    Оставляем только те строки, которые явно ссылаются на нужный(е) номер(а) рисунка:
      - парсим номера через extract_figure_numbers, нормализуем;
      - если в строке нет номеров → в single-режиме выбрасываем (чтобы не протащить чужой рисунок);
      - если есть номера, но среди них есть чужие → строку выкидываем.
    """
    if not focus_nums:
        return [str(x).strip() for x in (lines or []) if str(x).strip()]

    focus_set = {n for n in focus_nums if n}
    out: List[str] = []

    for x in (lines or []):
        s = str(x or "").strip()
        if not s:
            continue

        nums: List[str] = []
        if extract_figure_numbers:
            try:
                nums = extract_figure_numbers(s) or []  # type: ignore
            except Exception:
                nums = []

        if not nums:
            # в режиме «только Рисунок N» не тащим строки без явного номера
            continue

        norm_in_line = {_normalize_fig_num_local(n) for n in nums}
        if norm_in_line.issubset(focus_set):
            out.append(s)

    return out

def _parse_table_title(text: str) -> tuple[Optional[str], Optional[str]]:
    m = _CAP_RE.search(text or "")
    if not m:
        return (None, None)
    raw_num = (m.group(1) or "").replace(" ", "")
    return (_normalize_num(raw_num), (m.group(2) or "").strip() or None)

def _last_segment(name: str) -> str:
    s = (name or "").strip()
    if "/" in s:
        s = s.split("/")[-1].strip()
    s = re.sub(r"^\[\s*таблица\s*\]\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*[-–—]\s*", " — ", s)
    s = re.sub(r"\s+", " ", s).strip(" .")
    return s

def _compose_display(attrs_json: Optional[str], base: str, first_row_text: Optional[str]) -> str:
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
        tail_like = (tail or header_preview or first_row_text or "").strip()
        return f"Таблица {num}" + (f" — {_shorten(tail_like, 160)}" if tail_like else "")

    num_b, title_b = _parse_table_title(_last_segment(base))
    if num_b:
        text_tail = title_b or first_row_text or header_preview
        return f"Таблица {num_b}" + (f" — {_shorten(text_tail, 160)}" if text_tail else "")

    if tail:
        return _shorten(str(tail), 160)
    if header_preview:
        return _shorten(str(header_preview), 160)
    if first_row_text:
        return _shorten(first_row_text, 160)

    s = _last_segment(base)
    s = re.sub(r"(?i)^\s*таблица\s+\d+(?:\.\d+)*\s*", "", s).strip(" —–-")
    return _shorten(s or "Таблица", 160)

def _table_has_columns(con, table: str, cols: List[str]) -> bool:
    cur = con.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    have = {row[1] for row in cur.fetchall()}
    return all(c in have for c in cols)

def _find_table_anchor(uid: int, doc_id: int, num: str) -> Optional[Dict[str, Any]]:
    want = _normalize_num(num)
    con = get_conn()
    cur = con.cursor()
    has_ext = _table_has_columns(con, "chunks", ["element_type", "attrs"])

    row = None
    if has_ext:
        like1 = f'%\"caption_num\": \"{want}\"%'
        like2 = f'%\"label\": \"{want}\"%'
        cur.execute(
            """
            SELECT page, section_path, attrs
            FROM chunks
            WHERE owner_id=? AND doc_id=? AND element_type IN ('table','table_row')
              AND (attrs LIKE ? OR attrs LIKE ?)
            ORDER BY id ASC LIMIT 1
            """,
            (uid, doc_id, like1, like2),
        )
        row = cur.fetchone()
        if not row:
            cur.execute(
                """
                SELECT page, section_path, attrs
                FROM chunks
                WHERE owner_id=? AND doc_id=? AND element_type IN ('table','table_row')
                  AND section_path LIKE ? COLLATE NOCASE
                ORDER BY id ASC LIMIT 1
                """,
                (uid, doc_id, f'%Таблица {want}%'),
            )
            row = cur.fetchone()
    else:
        cur.execute(
            """
            SELECT page, section_path, NULL AS attrs
            FROM chunks
            WHERE owner_id=? AND doc_id=? 
              AND (lower(section_path) LIKE '%%таблица %%' OR text LIKE '[Таблица]%%')
              AND section_path LIKE ? COLLATE NOCASE
            ORDER BY id ASC LIMIT 1
            """,
            (uid, doc_id, f'%Таблица {want}%'),
        )
        row = cur.fetchone()

    con.close()
    return dict(row) if row else None

def _base_from_section(section_path: str) -> str:
    s = section_path or ""
    i = s.find(" [row ")
    return s if i < 0 else s[:i]

def _split_cells(line: str) -> List[str]:
    s = (line or "").strip()
    if " | " in s:
        cells = [c.strip() for c in s.split(" | ")]
    elif "\t" in s:
        cells = [c.strip() for c in s.split("\t")]
    else:
        cells = [c.strip() for c in re.split(r"\s{2,}", s)]
    return [c for c in cells if c != ""]

def _detect_header_and_matrix(rows_texts: List[str]) -> tuple[List[str], List[List[str]]]:
    raw_rows = [_split_cells((t or "").splitlines()[0]) for t in rows_texts if (t or "").strip()]
    if not raw_rows:
        return ([], [])

    def frac_numeric(cells: List[str]) -> float:
        if not cells:
            return 0.0
        num_re = re.compile(r"^[+-]?\d{1,3}(?:[\s\u00A0]\d{3})*(?:[.,]\d+)?%?$|^[+-]?\d+(?:[.,]\d+)?%?$")
        n = sum(1 for c in cells if num_re.match(c.strip()))
        return n / max(1, len(cells))

    header_candidates = raw_rows[0]
    data_candidates = raw_rows[1:] if len(raw_rows) > 1 else []
    use_header = False
    if data_candidates:
        if frac_numeric(header_candidates) <= 0.25 and any(frac_numeric(r) >= 0.4 for r in data_candidates[:3]):
            use_header = True

    if use_header:
        headers = header_candidates
        data_rows = data_candidates
    else:
        max_len = max(len(r) for r in raw_rows)
        headers = [f"Колонка {i+1}" for i in range(max_len)]
        data_rows = raw_rows

    W = min(len(headers), FULL_TABLE_MAX_COLS)
    headers = headers[:W]
    matrix: List[List[str]] = []
    for r in data_rows:
        row = (r + [""] * W)[:W]
        matrix.append(row)
    headers = [_shorten(h, 80) for h in headers]
    return (headers, matrix)

def _fetch_table_rows(uid: int, doc_id: int, base: str) -> List[Dict[str, Any]]:
    con = get_conn()
    cur = con.cursor()
    if _table_has_columns(con, "chunks", ["element_type"]):
        sql = """
            SELECT text, page FROM chunks
            WHERE owner_id=? AND doc_id=? AND element_type='table_row'
              AND (section_path=? OR section_path LIKE ? || ' [row %')
            ORDER BY id ASC
        """
    else:
        sql = """
            SELECT text, page FROM chunks
            WHERE owner_id=? AND doc_id=? AND (section_path=? OR section_path LIKE ? || ' [row %')
            ORDER BY id ASC
        """
    cur.execute(sql, (uid, doc_id, base, base))
    rows = cur.fetchall() or []
    con.close()
    return [{"text": (r["text"] or ""), "page": r["page"]} for r in rows]

def _apply_rows_and_char_limits(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Обрезаем по строкам/символам, чтобы не вывалиться за разумные лимиты промпта.
    """
    rows: List[List[str]] = payload.get("rows") or []
    headers: List[str] = payload.get("headers") or []
    total_rows = len(rows)
    truncated = False

    # Лимит строк
    if total_rows > payload.get("_rows_limit_effective", FULL_TABLE_MAX_ROWS):
        rows = rows[: payload.get("_rows_limit_effective", FULL_TABLE_MAX_ROWS)]
        truncated = True

    # Лимит символов (грубая оценка на json.dumps)
    obj = {"headers": headers, "rows": rows}
    text = json.dumps(obj, ensure_ascii=False)
    if len(text) > FULL_TABLE_MAX_CHARS:
        # грубо уменьшаем, пока не влезет
        step = max(5, len(rows) // 10)
        n = len(rows)
        while n > 0 and len(json.dumps({"headers": headers, "rows": rows[:n]}, ensure_ascii=False)) > FULL_TABLE_MAX_CHARS:
            n -= step
        rows = rows[: max(1, n)]
        truncated = True

    payload["rows"] = rows
    payload["total_rows"] = int(total_rows)
    payload["truncated"] = bool(truncated or total_rows > len(rows))
    payload.pop("_rows_limit_effective", None)
    return payload

# ---------- НОВОЕ: таблицы по подсказке «в главе/разделе» (без номера) ----------

def _distinct_table_bases_by_section(uid: int, doc_id: int, sect_hint: str) -> List[str]:
    """
    Возвращает уникальные «базовые» имена таблиц (section_path без хвоста ' [row …]')
    для разделов, в секционных путях которых встречается `sect_hint` (например: '2.2', '3', 'A1').
    """
    con = get_conn()
    cur = con.cursor()
    has_type = _table_has_columns(con, "chunks", ["element_type"])

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
            WHERE owner_id=? AND doc_id=? AND element_type IN ('table','table_row')
              AND section_path LIKE ? COLLATE NOCASE
            """,
            (uid, doc_id, f"%{sect_hint}%"),
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
            WHERE owner_id=? AND doc_id=? 
              AND (lower(section_path) LIKE '%таблица %' OR lower(text) LIKE '[таблица]%')
              AND section_path LIKE ? COLLATE NOCASE
            """,
            (uid, doc_id, f"%{sect_hint}%"),
        )
    rows = cur.fetchall() or []
    con.close()

    bases = [r["base_name"] for r in rows if r and r["base_name"]]
    # Нормализуем и уникализируем, с сохранением порядка
    seen = set()
    uniq: List[str] = []
    for b in bases:
        k = b.strip()
        if not k:
            continue
        if k not in seen:
            seen.add(k)
            uniq.append(k)
    return uniq

def _tables_raw_by_bases(uid: int, doc_id: int, bases: List[str], rows_limit_effective: int) -> List[Dict[str, Any]]:
    """
    Собирает JSON-представление таблиц по списку базовых section_path.
    """
    out: List[Dict[str, Any]] = []
    for base in bases:
        try:
            row_objs = _fetch_table_rows(uid, doc_id, base)
            if not row_objs:
                continue
            # первая строка — для красивого display
            first_line = None
            txt0 = (row_objs[0]["text"] or "").split("\n")[0]
            if txt0:
                first_line = " — ".join([c.strip() for c in _split_cells(txt0) if c.strip()])

            # попробуем достать attrs из первой подходящей записи table/table_row
            con = get_conn()
            cur = con.cursor()
            has_attrs = _table_has_columns(con, "chunks", ["attrs"])
            if has_attrs:
                cur.execute(
                    """
                    SELECT attrs, page FROM chunks
                    WHERE owner_id=? AND doc_id=? AND (section_path=? OR section_path LIKE ? || ' [row %')
                    ORDER BY id ASC LIMIT 1
                    """,
                    (uid, doc_id, base, base),
                )
            else:
                cur.execute(
                    """
                    SELECT page FROM chunks
                    WHERE owner_id=? AND doc_id=? AND (section_path=? OR section_path LIKE ? || ' [row %')
                    ORDER BY id ASC LIMIT 1
                    """,
                    (uid, doc_id, base, base),
                )
            r = cur.fetchone()
            con.close()

            attrs_json = (r["attrs"] if (has_attrs and r and "attrs" in r.keys()) else None)
            page = r["page"] if r else None
            if isinstance(attrs_json, dict):
                attrs_json = json.dumps(attrs_json, ensure_ascii=False)

            display = _compose_display(attrs_json, base, first_line)
            rows_texts = [ro["text"] for ro in row_objs]
            headers, matrix = _detect_header_and_matrix(rows_texts)

            payload = {
                "num": None,  # номер неизвестен (не просили по номеру)
                "display": display,
                "where": {"page": page, "section_path": base},
                "headers": headers,
                "rows": matrix,
                "_rows_limit_effective": rows_limit_effective,
            }
            out.append(_apply_rows_and_char_limits(payload))
        except Exception:
            continue
    return out

def _make_tables_raw_for_prompt(
    owner_id: Optional[int],
    doc_id: Optional[int],
    question: str,
    include_all: Optional[bool] = None,
    rows_limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Возвращает массив таблиц (в виде JSON-пакетов) для секции [TablesRaw] промпта.

    Сценарии:
      1) Вопрос содержит конкретные номера таблиц → тащим по номерам.
      2) Номеров нет, но упомянута «глава/раздел/пункт §…» и «таблица» → тащим все таблицы из указанного раздела(ов).
    """
    if owner_id is None or doc_id is None:
        return []

    q = question or ""

    # если флаги не передали явно — определяем их из текста вопроса
    if include_all is None or rows_limit is None:
        inc_all_auto, rows_limit_auto = _extract_fulltable_request(q)
        if include_all is None:
            include_all = inc_all_auto
        if rows_limit is None:
            rows_limit = rows_limit_auto

    rows_limit_effective = min(
        FULL_TABLE_MAX_ROWS,
        rows_limit if rows_limit is not None else (FULL_TABLE_MAX_ROWS if include_all else 80),
    )

    # --- 1) Попытка по номерам
    want_nums: List[str] = []
    if extract_table_numbers:
        try:
            want_nums = extract_table_numbers(q) or []
        except Exception:
            want_nums = []
    out: List[Dict[str, Any]] = []
    if want_nums:
        for num in want_nums:
            try:
                anchor = _find_table_anchor(owner_id, doc_id, num)
                if not anchor:
                    continue
                page = anchor.get("page")
                sec = anchor.get("section_path") or ""
                base = _base_from_section(sec)

                row_objs = _fetch_table_rows(owner_id, doc_id, base)
                first_line = None
                if row_objs:
                    txt = (row_objs[0]["text"] or "").split("\n")[0]
                    if txt:
                        first_line = " — ".join([c.strip() for c in _split_cells(txt) if c.strip()])
                attrs_json = anchor.get("attrs")
                if isinstance(attrs_json, dict):
                    attrs_json = json.dumps(attrs_json, ensure_ascii=False)
                display = _compose_display(attrs_json, base, first_line)

                rows_texts = [r["text"] for r in row_objs]
                headers, matrix = _detect_header_and_matrix(rows_texts)

                payload = {
                    "num": str(num),
                    "display": display,
                    "where": {"page": page, "section_path": base},
                    "headers": headers,
                    "rows": matrix,
                    "_rows_limit_effective": rows_limit_effective,
                }
                out.append(_apply_rows_and_char_limits(payload))
            except Exception:
                continue
        if out:
            return out  # если нашли по номерам — этого достаточно

    # --- 2) Номеров нет → пробуем по подсказкам «в главе/разделе/§ …»
    sects: List[str] = []
    try:
        sects = extract_section_hints(q) or []
    except Exception:
        sects = []
    if not sects:
        return []  # нет ни номеров, ни подсказок раздела — ничего не тащим

    # для каждого хинта подтянем до SECTION_TABLES_MAX таблиц
    collected: List[Dict[str, Any]] = []
    for hint in sects:
        try:
            bases = _distinct_table_bases_by_section(owner_id, doc_id, hint)
            if not bases:
                continue
            bases = bases[:SECTION_TABLES_MAX]
            pack = _tables_raw_by_bases(owner_id, doc_id, bases, rows_limit_effective)
            collected.extend(pack)
        except Exception:
            continue

    return collected

# ----------------------------- Сборка блока фактов -----------------------------

def _cards_for_tables(
    table_describe: List[Dict[str, Any]],
    *,
    owner_id: Optional[int] = None,
    doc_id: Optional[int] = None,
    lang: str = "ru",
    insights_top_k: int = 3,
    norm_numbers: bool = True,
) -> List[Dict[str, Any]]:
    """
    Приводим карточки таблиц к компактному формату; при наличии analytics и идентификаторов
    документа аккуратно добавляем вычисленные инсайты (короткий текст + небольшие топы),
    не нарушая принципа «только из фактов».
    """
    cards: List[Dict[str, Any]] = []

    for c in (table_describe or []):
        card = {
            "num": c.get("num"),
            "display": _normalize_numbers(c.get("display") or "") if norm_numbers else (c.get("display") or ""),
            "where": c.get("where") or {},
            "highlights": [
                _normalize_numbers(h) if norm_numbers else h
                for h in (c.get("highlights") or [])
            ][:2],
        }

        # Мягкая интеграция аналитики — только если есть owner_id/doc_id и корректная функция.
        num = (c.get("num") or "").strip() if isinstance(c.get("num"), str) else None
        if analyze_table_by_num and owner_id is not None and doc_id is not None and num:
            try:
                res = analyze_table_by_num(owner_id, doc_id, num, top_k=max(1, insights_top_k), lang=lang)  # type: ignore
                if isinstance(res, dict) and res.get("ok"):
                    insight: Dict[str, Any] = {}

                    # Короткий текст отчёта (ужимаем)
                    text = res.get("text") or ""
                    if text:
                        insight["text"] = (_normalize_numbers if norm_numbers else (lambda x: x))(
                            _shorten(str(text), 420)
                        )

                    # Топы по числовым колонкам (до 2 колонок, каждый топ — до insights_top_k строк)
                    tops: List[Dict[str, Any]] = []
                    for col in (res.get("numeric_summary") or [])[:2]:
                        col_name = str(col.get("col") or "").strip() or "Колонка"
                        col_top = col.get("top") or []
                        small = []
                        for item in col_top[:insights_top_k]:
                            row_lbl = (_normalize_numbers if norm_numbers else (lambda x: x))(
                                str(item.get("row") or "")
                            )
                            val = item.get("value")
                            try:
                                small.append({"row": row_lbl, "value": float(val) if val is not None else None})
                            except Exception:
                                small.append({"row": row_lbl, "value": val})
                        if small:
                            tops.append({"col": (_normalize_numbers(col_name) if norm_numbers else col_name), "top": small})
                    if tops:
                        insight["tops"] = tops

                    if insight:
                        card["insights"] = insight
            except Exception:
                # Мягкая деградация — без инсайтов
                pass

        cards.append(card)
    return cards


def facts_to_prompt(
    facts: Dict[str, Any],
    *,
    rules: Optional[str] = None,
    owner_id: Optional[int] = None,
    doc_id: Optional[int] = None,
    lang: str = "ru",
    tables_raw: Optional[List[Dict[str, Any]]] = None,
    norm_numbers: bool = True,
) -> str:
    """
    Превращает dict `facts` (как собирается в bot._gather_facts) в устойчивый markdown-блок
    для модели: [Facts] ... [Rules] ...
    Дополнительно может включать [TablesRaw] с полной матрицей таблиц по запросу пользователя.
    """
    parts: List[str] = []

    # ----- Подпункты (Items) -----
    # Поддерживаем два варианта: facts['coverage']['items'] (как возвращает retrieve_coverage)
    # и плоский facts['items'] (если заполняется напрямую слоем-оркестратором).
    items_src = (
        ((facts or {}).get("coverage") or {}).get("items")
        or (facts or {}).get("items")
        or []
    )
    if items_src:
        lines: List[str] = []
        for it in items_src[:25]:
            if isinstance(it, dict):
                idx = it.get("id") or it.get("index")
                text = it.get("ask") or it.get("text") or ""
                if idx is not None:
                    lines.append(f"{idx}) {(_normalize_numbers(text) if norm_numbers else text)}")
                else:
                    lines.append(f"- {(_normalize_numbers(text) if norm_numbers else text)}")
            else:
                s = str(it)
                lines.append(f"- {(_normalize_numbers(s) if norm_numbers else s)}")
        # Отдельная секция, чтобы модель явно понимала структуру ответа
        parts.append("- Items:\n  " + "\n  ".join(lines))

    # ----- Таблицы -----
    raw_tables = (facts or {}).get("tables") or {}
    tables: Dict[str, Any] = raw_tables if isinstance(raw_tables, dict) else {}

    # есть ли ХОТЬ КАКИЕ-ТО реальные табличные факты?
    has_table_facts = False
    if tables:
        if (tables.get("count") or 0) > 0:
            has_table_facts = True
        if tables.get("list"):
            has_table_facts = True
        if tables.get("describe"):
            has_table_facts = True

    # на будущее: если где-то отдельно положили tables_raw в facts
    if not has_table_facts and (facts or {}).get("tables_raw"):
        has_table_facts = True

    if has_table_facts:
        block: List[str] = []
        if "count" in tables:
            block.append(f"count: {int(tables.get('count') or 0)}")
        if tables.get("list"):
            block.append(
                "list:\n"
                + _md_list(tables["list"], 25, tables.get("more", 0), norm_numbers=norm_numbers)
            )
        if tables.get("describe"):
            cards = _cards_for_tables(
                tables.get("describe") or [],
                owner_id=owner_id,
                doc_id=doc_id,
                lang=lang,
                norm_numbers=norm_numbers,
            )
            block.append("describe:\n" + json.dumps(cards, ensure_ascii=False, indent=2))
        parts.append("- Tables:\n  " + "\n  ".join(block))


        # ----- Рисунки -----
    figures = (facts or {}).get("figures") or {}
    cards = figures.get("describe_cards") if figures else None
    describe_lines = (figures.get("describe") or []) if figures else []

    if cards or describe_lines:
        # Если в интентах/фактах стоит флаг single_only — сфокусируемся только на нужном(ых) номере(ах)
        try:
            focus_nums = _figure_focus_from_facts(facts)
        except Exception:
            focus_nums = []

        # фильтрация карточек по номерам, если есть фокус
        if cards and focus_nums:
            focus_set = {n for n in focus_nums if n}
            filtered_cards: List[Dict[str, Any]] = []
            for c in cards:
                num = c.get("num") or c.get("label")
                if not num:
                    continue
                norm = _normalize_fig_num_local(str(num))
                if norm in focus_set:
                    filtered_cards.append(c)
            if filtered_cards:
                cards = filtered_cards

        block: List[str] = []

        # Карточки от vision_analyzer (describe_cards)
        if cards:
            for c in cards[:25]:
                title = c.get("title") or c.get("display") or "Рисунок"
                desc = c.get("description") or ""
                values = c.get("values") or []  # новый формат vision_analyzer

                # Заголовок
                block.append(f"- **{title}**")

                # Значения (если есть явный список)
                if values:
                    vals = ", ".join(values)
                    block.append(f"  Значения: {vals}")

                # Описание
                if desc:
                    block.append(f"  Описание: {desc}")

        # Текстовый формат facts['figures']['describe'] (старый/резервный путь)
        if describe_lines:
            # если включён режим single_only — оставим только строки про нужный(е) номер(а)
            lines = _filter_lines_for_figure_focus(describe_lines, focus_nums or [])
            for s in lines[:25]:
                txt = _normalize_numbers(str(s)) if norm_numbers else str(s)
                block.append(f"- {txt}")

        if block:
            parts.append("- Figures:\n  " + "\n  ".join(block))




    # ----- Источники -----
    sources = (facts or {}).get("sources") or {}
    if sources:
        block = [f"count: {int(sources.get('count') or 0)}"]
        if sources.get("list"):
            block.append("list:\n" + _md_list(sources.get("list") or [], 25, sources.get("more", 0), norm_numbers=norm_numbers))
        parts.append("- Sources:\n  " + "\n  ".join(block))

    # ----- Практическая часть -----
    if "practical_present" in (facts or {}):
        parts.append(f"- PracticalPartPresent: {bool(facts.get('practical_present'))}")

    # ----- Краткое содержание -----
    if (facts or {}).get("summary_text"):
        st = str(facts["summary_text"])
        # Оставляем полный структурированный вывод от summarizer
        parts.append("- Summary:\n  " + st.replace("\n", "\n  "))

    # ----- Вербатим-цитаты (шинглы) -----
    if (facts or {}).get("verbatim_hits"):
        hits_md = []
        for h in facts["verbatim_hits"]:
            page = h.get("page")
            sec = (h.get("section_path") or "").strip()
            page_str = (str(page) if page is not None else "?")
            where = f'в разделе «{sec}», стр. {page_str}' if sec else f'на стр. {page_str}'
            snippet = (_normalize_numbers(h.get("snippet") or "") if norm_numbers else (h.get("snippet") or ""))
            hits_md.append(f"- Match {where}: «{snippet}»")
        parts.append("- Citations:\n  " + "\n  ".join(hits_md))

    # ----- Общий контекст -----
    if (facts or {}).get("general_ctx"):
        ctx = str(facts.get("general_ctx") or "")
        ctx = (_normalize_numbers(_shorten(ctx, 1500)) if norm_numbers else _shorten(ctx, 1500))
        parts.append("- Context:\n  " + ctx.replace("\n", "\n  "))

    # ----- Полные таблицы по запросу -----
    if tables_raw:
        try:
            payload = json.dumps(tables_raw, ensure_ascii=False)  # уже ограничено по символам/строкам
        except Exception:
            payload = json.dumps([], ensure_ascii=False)
        parts.append("- TablesRaw:\n  " + payload.replace("\n", "\n  "))

    facts_md = "[Facts]\n" + "\n".join(parts) + "\n\n[Rules]\n" + (rules or _DEFAULT_RULES)
    return facts_md

# ----------------------------- Вспомогательные стрим-утилы -----------------------------

def _extract_ids_from_facts(facts: Dict[str, Any]) -> tuple[Optional[int], Optional[int]]:
    owner_id = (
        facts.get("owner_id")
        or facts.get("_owner_id")
        or (facts.get("meta") or {}).get("owner_id")
    )
    doc_id = (
        facts.get("doc_id")
        or facts.get("_doc_id")
        or (facts.get("meta") or {}).get("doc_id")
    )
    try:
        owner_id = int(owner_id) if owner_id is not None else None
    except Exception:
        owner_id = None
    try:
        doc_id = int(doc_id) if doc_id is not None else None
    except Exception:
        doc_id = None
    return owner_id, doc_id

def _smart_cut_point(s: str, limit: int) -> int:
    if len(s) <= limit:
        return len(s)
    cut = s.rfind("\n", 0, limit)
    if cut == -1:
        cut = s.rfind(". ", 0, limit)
    if cut == -1:
        cut = s.rfind(" ", 0, limit)
    if cut == -1:
        cut = limit
    return max(1, cut)

def _chunk_text(s: str, maxlen: int = 480) -> Iterable[str]:
    s = s or ""
    i = 0
    n = len(s)
    if n == 0:
        return []
    while i < n:
        cut = _smart_cut_point(s[i:], maxlen)
        yield s[i:i+cut]
        i += cut

async def _aiter_any(obj: Union[str, Iterable[str], AsyncIterable[str]]) -> AsyncIterable[str]:
    """
    Нормализуем любой источник в асинхронный поток строк.
    """
    if isinstance(obj, str):
        for part in _chunk_text(obj, 480):
            yield part
            await asyncio.sleep(0)
        return

    if hasattr(obj, "__aiter__"):
        async for x in obj:  # type: ignore
            if x:
                yield str(x)
        return

    if hasattr(obj, "__iter__"):
        for x in obj:  # type: ignore
            if x:
                yield str(x)
            await asyncio.sleep(0)
        return

# ----------------------------- Публичное API -----------------------------

# Вопрос вообще про таблицу?
# Вопрос вообще про таблицу?
_ASKS_TABLE_RE = re.compile(r"(?i)\bтаблиц[а-я]*\b|\btable\b")

def _asks_about_table(question: str) -> bool:
    q = (question or "").strip()
    return bool(_ASKS_TABLE_RE.search(q))


def _has_any_table_facts(facts: Dict[str, Any]) -> bool:
    """
    Понимаем, есть ли в фактах хоть какая-то конкретика по таблицам.
    Нужен, чтобы не отдавать в модель вопрос про таблицу без таблиц.
    """
    if not isinstance(facts, dict):
        return False

    tables = (facts.get("tables") or {})
    if not isinstance(tables, dict):
        return False

    if (tables.get("count") or 0) > 0:
        return True
    if tables.get("list"):
        return True
    if tables.get("describe"):
        return True

    # на будущее: если где-то вручную положили tables_raw
    if facts.get("tables_raw"):
        return True

    return False


def _has_any_figure_facts(facts: Dict[str, Any]) -> bool:
    """
    Мягкий аналог _has_any_table_facts для рисунков.
    Используем как «страховку»: если вопрос формулируют как про таблицу,
    но в фактах есть только блок по рисункам (например, таблица как картинка),
    не блокируем ответ.
    """
    if not isinstance(facts, dict):
        return False

    figures = (facts.get("figures") or {})
    if not isinstance(figures, dict):
        return False

    if (figures.get("count") or 0) > 0:
        return True
    if figures.get("list"):
        return True
    if figures.get("describe_cards"):
        return True
    if figures.get("describe"):
        return True

    return False

def _extract_fulltable_request(question: str) -> tuple[bool, Optional[int]]:
    """
    Определяем, просили ли «все значения/полностью» и/или лимит строк.
    """
    q = question or ""
    include_all = bool(TABLE_ALL_VALUES_RE.search(q))
    rows_limit = None
    m = TABLE_ROWS_LIMIT_RE.search(q)
    if m:
        for g in m.groups():
            if g and g.isdigit():
                try:
                    n = int(g)
                    if n > 0:
                        rows_limit = n
                        break
                except Exception:
                    pass
    return include_all, rows_limit

def _want_exact_numbers(question: str, facts: Dict[str, Any]) -> bool:
    """
    Определяем, просили ли вывести числа «как в документе» (без нормализации).
    Источники сигнала:
      - фраза в вопросе (EXACT_NUMBERS_RE),
      - либо явный флаг facts['exact_numbers']=True (на будущее).
    """
    if isinstance(facts, dict) and bool(facts.get("exact_numbers")):
        return True
    q = (question or "")
    try:
        return bool(EXACT_NUMBERS_RE.search(q))
    except Exception:
        return False

def generate_answer(
    question: str,
    facts: Dict[str, Any],
    *,
    language: str = "ru",
    pass_score: int = 85,  # оставлен для совместимости, сейчас не используется
    rules_override: Optional[str] = None,
) -> str:
    """
    Универсальный билдер финального ответа (нестримовый путь):
      1) по умолчанию нормализует числа в блоке фактов, но отключает это при запросе «как в документе»;
      2) собирает устойчивый блок промпта (Facts + Rules), включая TablesRaw при необходимости;
      3) обращается напрямую к chat_with_gpt без ACE;
      4) возвращает финальный текст.
    """
    q = (question or "").strip()
    if not q:
        return "Вопрос пустой. Сформулируйте, пожалуйста, что именно требуется разобрать по ВКР."

    # 💡 Гард: пользователь спрашивает про таблицу, а фактов по таблицам нет вообще.
    # Вместо фантазий модели сразу просим уточнить номер.
    if _asks_about_table(q) and not _has_any_table_facts(facts):
        if not _has_any_figure_facts(facts):
            return (
                "Я не вижу в фактах, к какой именно таблице можно привязаться.\n"
                "Пожалуйста, укажите номер таблицы (например, «таблица 2.4») "
                "или пришлите скрин / точный заголовок."
            )

    owner_id, doc_id = _extract_ids_from_facts(facts)
    want_exact = _want_exact_numbers(q, facts)

    # ❗ Тянем TablesRaw (по номерам или по «главе/разделу»)
    include_all, rows_limit = _extract_fulltable_request(q)
    tables_raw: List[Dict[str, Any]] = _make_tables_raw_for_prompt(
        owner_id,
        doc_id,
        q,
        include_all=include_all,
        rows_limit=rows_limit,
    )

    ctx = facts_to_prompt(
        facts,
        rules=rules_override,
        owner_id=owner_id,
        doc_id=doc_id,
        lang=language,
        tables_raw=tables_raw if tables_raw else None,
        norm_numbers=not want_exact,
    )

    # --- Прямой вызов модели без ACE ---
    system_prompt = (
        "Ты — универсальный умный ассистент по дипломным работам и научным текстам.\n"
        "Тебе дан блок фактов [Facts] из работы и правила ответа.\n"
        "Твоя задача — дать максимально полный, связный и понятный ответ на запрос пользователя.\n"
        "В ПЕРВУЮ ОЧЕРЕДЬ опирайся на факты из [Facts]. Если каких-то деталей там нет,\n"
        "ты можешь аккуратно использовать общие знания по теме, но не приписывай документу\n"
        "того, чего в нём нет.\n"
        "Структурируй ответ под запрос: используй списки, подпункты и подзаголовки, когда это уместно.\n"
        "Если в вопросе упомянуты конкретные таблицы, рисунки или главы (например: «опиши рисунок 2.2, "
        "таблицу 2.1 и главу 1»), делай отдельные подпункты для каждого объекта в указанном порядке и "
        "поясняй их содержательно."
    )



    if chat_with_gpt is None:
        # жёсткий фолбэк — просто возвращаем дайджест фактов
        fallback = [
            "Не удалось обратиться к модели. Ниже — краткий конспект найденных фактов:",
            "",
            ctx[:4000],
        ]
        return "\n".join(fallback)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": ctx},
        {"role": "user", "content": q},
    ]

    try:
        reply = chat_with_gpt(  # type: ignore
            messages,
            temperature=0.2,
            max_tokens=getattr(Cfg, "FINAL_MAX_TOKENS", 1600),
        )
        return _headings_to_bold((reply or "").strip())
    except Exception:
        fallback = [
            "Не удалось сгенерировать ответ моделью. Ниже — краткий конспект найденных фактов:",
            "",
            ctx[:4000],
        ]
        return "\n".join(fallback)


async def generate_answer_stream(
    question: str,
    facts: Dict[str, Any],
    *,
    language: str = "ru",
    pass_score: int = 85,  # оставлен для совместимости
    rules_override: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = getattr(Cfg, "FINAL_MAX_TOKENS", 1600),
) -> AsyncIterable[str]:
    """
    Стриминговый билдер финального ответа (корутина, возвращает async-итератор):
      - собирает Facts+Rules (+ TablesRaw если нужно, по номерам или по «главе/разделу»);
      - учитывает запрос на «точные числа как в документе» (отключает нормализацию в промпте);
      - стримит напрямую через chat_with_gpt_stream (без ACE);
      - если стрим недоступен — эмулирует стрим, нарезая результат generate_answer.
    """
    q = (question or "").strip()
    if not q:
        async def _empty():
            yield "Вопрос пустой. Сформулируйте, пожалуйста, что именно требуется разобрать по ВКР."
        return _empty()

    # 💡 Тот же гард, что и в синхронной версии:
    # если про таблицу спрашивают, а таблиц в фактах нет — сразу просим уточнить номер.
    if _asks_about_table(q) and not _has_any_table_facts(facts):
        if not _has_any_figure_facts(facts):
            async def _need_table_num():
                yield (
                    "Я не вижу в фактах, к какой именно таблице можно привязаться.\n"
                    "Пожалуйста, укажите номер таблицы (например, «таблица 2.4») "
                    "или пришлите скрин / точный заголовок."
                )
            return _need_table_num()

    owner_id, doc_id = _extract_ids_from_facts(facts)
    want_exact = _want_exact_numbers(q, facts)

    # Полные таблицы: по номеру или по подсказке «в главе/разделе»
    include_all, rows_limit = _extract_fulltable_request(q)
    tables_raw: List[Dict[str, Any]] = _make_tables_raw_for_prompt(
        owner_id,
        doc_id,
        q,
        include_all=include_all,
        rows_limit=rows_limit,
    )

    ctx = facts_to_prompt(
        facts,
        rules=rules_override,
        owner_id=owner_id,
        doc_id=doc_id,
        lang=language,
        tables_raw=tables_raw if tables_raw else None,
        norm_numbers=not want_exact,
    )

    system_prompt = (
        "Ты — универсальный умный ассистент по дипломным работам и научным текстам.\n"
        "Тебе дан блок фактов [Facts] из работы и правила ответа.\n"
        "Твоя задача — дать максимально полный, связный и понятный ответ на запрос пользователя.\n"
        "В ПЕРВУЮ ОЧЕРЕДЬ опирайся на факты из [Facts]. Если каких-то деталей там нет,\n"
        "ты можешь аккуратно использовать общие знания по теме, но не приписывай документу\n"
        "того, чего в нём нет.\n"
        "Структурируй ответ под запрос: используй списки, подпункты и подзаголовки, когда это уместно.\n"
        "Если в вопросе упомянуты конкретные таблицы, рисунки или главы (например: «опиши рисунок 2.2, "
        "таблицу 2.1 и главу 1»), делай отдельные подпункты для каждого объекта в указанном порядке и "
        "поясняй их содержательно."
    )



    # 1) Нормальный путь — стрим напрямую из модели
    if chat_with_gpt_stream is not None:
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "assistant", "content": ctx},
                {"role": "user", "content": q},
            ]
            stream_obj = chat_with_gpt_stream(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,  # type: ignore
            )

            async def _fmt():
                async for chunk in _aiter_any(stream_obj):
                    yield _headings_to_bold(chunk)
            return _fmt()

        except Exception:
            # пойдём в эмулированный стрим
            pass

    # 2) Фолбэк — эмулируем стрим через generate_answer
    async def _emulated() -> AsyncIterable[str]:  # type: ignore
        try:
            final = generate_answer(
                question=q,
                facts=facts,
                language=language,
                pass_score=pass_score,
                rules_override=rules_override,
            )
        except Exception:
            final = "Не удалось сгенерировать ответ."
        for part in _chunk_text(final, 480):
            yield part
            await asyncio.sleep(0)
    return _emulated()


def debug_digest(facts: Dict[str, Any]) -> str:
    """
    Короткий диагностический дайджест по фактам — удобно логировать/показывать в /diag.
    """
    parts = []
    tbl = (facts or {}).get("tables") or {}
    fig = (facts or {}).get("figures") or {}
    src = (facts or {}).get("sources") or {}

    parts.append(f"Таблиц: {int(tbl.get('count') or 0)} (показано: {len(tbl.get('list') or [])})")
    parts.append(f"Рисунков: {int(fig.get('count') or 0)} (показано: {len(fig.get('list') or [])})")
    parts.append(f"Источников: {int(src.get('count') or 0)} (показано: {len(src.get('list') or [])})")

    if facts.get("practical_present") is not None:
        parts.append(f"Практическая часть: {'да' if facts.get('practical_present') else 'нет'}")

    if facts.get("summary_text"):
        parts.append("Есть краткое содержание (summary_text).")

    if facts.get("general_ctx"):
        parts.append("Есть общий контекст (general_ctx).")

    if facts.get("verbatim_hits"):
        parts.append(f"Есть точные совпадения (verbatim_hits): {len(facts.get('verbatim_hits') or [])}")

    return "\n".join(parts)
