# app/analytics.py
from __future__ import annotations

import re
import json
from typing import List, Dict, Any, Optional, Tuple

from .db import get_conn

# ---------------------------- Вспомогательные утилиты ----------------------------

def _table_has_columns(con, table: str, cols: List[str]) -> bool:
    cur = con.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    have = {row[1] for row in cur.fetchall()}
    return all(c in have for c in cols)


# Разделители тысяч и пробелы (обычный, неразрывный, узкий неразрывный, тонкий)
_THIN_SPACES = "\u00A0\u202F\u2009"
# Негатив в бухгалтерском стиле: (1 234,56)
_PARENS_RE = re.compile(r"^\(\s*(.+?)\s*\)$")
# Хвостовые/встроенные единицы и валюты
_UNIT_PCT_RE = re.compile(r"%")
_UNIT_PERMILLE_RE = re.compile(r"‰")
_UNIT_PP_RE = re.compile(r"(?i)\bп\.?п\.?|\bpp\b")
_CURRENCY_RE = re.compile(r"(?i)(?:\s*(?:₽|руб\.?|rur|rub|€|eur|\$|usd))\b")
# Мультипликаторы (тыс/млн/млрд, k/m/b/bn)
_MULTIPLIER_PATTERNS = [
    (re.compile(r"(?i)\b(тыс\.?|thous\.?|k)\b"), 1_000.0),
    (re.compile(r"(?i)\b(млн\.?|mln|million|m)\b"), 1_000_000.0),
    (re.compile(r"(?i)\b(млрд\.?|bln|bn|billion|b)\b"), 1_000_000_000.0),
]

# Число: допускаем пробелы/апострофы как разделители тысяч, запятую/точку как десятичный.
# Проценты/‰/п.п./валюта/мультипликаторы могут присутствовать, но будут сняты.
_NUM_CORE_RE = re.compile(
    rf"""
    ^\s*
    [+-]?
    (?:
        (?:\d{{1,3}}(?:[ \'{_THIN_SPACES}]\d{{3}})+|\d+)
        (?:[.,]\d+)?     # дробная часть
    |
        \d*(?:[.,]\d+)
    )
    \s*$
    """,
    re.X,
)


def _strip_spaces(s: str) -> str:
    return (s or "").replace("\u00A0", " ").replace("\u202F", " ").replace("\u2009", " ").strip()


def _detect_multiplier(text: str) -> Tuple[str, float]:
    """Возвращает (text_без_мультипликатора, множитель)."""
    t = text
    mul = 1.0
    for pat, k in _MULTIPLIER_PATTERNS:
        m = pat.search(t)
        if m:
            t = pat.sub("", t).strip()
            mul = k
            break
    return t, mul


def _canon_decimal(raw: str) -> Optional[float]:
    """
    Преобразуем текстовое число к float без потери величины:
    - удаляем пробелы/апострофы как разделители тысяч
    - определяем десятичный разделитель как последний из {'.', ','}
    """
    s = _strip_spaces(raw)
    # Бухгалтерские скобки
    negative = False
    pm = _PARENS_RE.match(s)
    if pm:
        s = pm.group(1)
        negative = True

    # Оставляем только знаки, цифры, . , и апостроф/пробел как разделители тысяч
    s = re.sub(_CURRENCY_RE, "", s).strip()

    # Выделяем «числовую» часть (если вокруг есть лишние слова)
    # Например: "1 234,56 тыс." → оставляем "1 234,56"
    num_match = re.search(r"[+-]?\s*(?:\d+[ '\u00A0\u202F\u2009]?)*\d(?:[.,]\d+)?", s)
    if not num_match:
        return None
    s_num = num_match.group(0)

    # Если и точка, и запятая — последняя из них считается десятичной
    if "," in s_num and "." in s_num:
        if s_num.rfind(",") > s_num.rfind("."):
            s_num = s_num.replace(".", "").replace(",", ".")
        else:
            s_num = s_num.replace(",", "")
    else:
        s_num = s_num.replace(",", ".")

    # Убираем разделители тысяч (пробелы/апострофы)
    s_num = s_num.replace(" ", "").replace("'", "")
    if not _NUM_CORE_RE.match(s_num):
        return None
    try:
        v = float(s_num)
    except Exception:
        return None
    return -v if negative else v


def _parse_number_with_unit(s: str) -> Tuple[Optional[float], str, bool]:
    """
    Возвращает (value, unit, had_unit_flag).
    unit ∈ {"%", "permille", "pp", ""}.
    value:
      - для "%" приводим к доле (0..1) — это важно для консистентности;
      - для permille/pp/"" возвращаем «как есть» (после учёта множителей).
    """
    if s is None:
        return (None, "", False)
    text = _strip_spaces(str(s))

    # Фиксируем наличие единиц
    had_percent = bool(_UNIT_PCT_RE.search(text))
    had_permille = bool(_UNIT_PERMILLE_RE.search(text))
    had_pp = bool(_UNIT_PP_RE.search(text))

    # Снимаем валюту/обозначения
    text = _CURRENCY_RE.sub("", text)

    # Мультипликатор (тыс/млн/млрд/k/m/b/bn)
    text, mul = _detect_multiplier(text)

    # Снимаем явные единицы из текста (для корректного парса числа)
    text = _UNIT_PCT_RE.sub("", text)
    text = _UNIT_PERMILLE_RE.sub("", text)
    text = _UNIT_PP_RE.sub("", text)

    base = _canon_decimal(text)
    if base is None:
        return (None, "", False)

    base *= mul

    if had_percent:
        # Для % приводим к доле
        return (base / 100.0, "%", True)
    if had_permille:
        # Оставляем в промилле «как есть», но помечаем тип
        return (base, "permille", True)
    if had_pp:
        return (base, "pp", True)
    return (base, "", False)


def _to_float(s: str) -> Tuple[Optional[float], bool]:
    """
    Back-compat враппер: (value, is_percent_column_unit_sign_present).
    Для процентов — возвращает долю (0..1).
    """
    v, unit, had = _parse_number_with_unit(s)
    return (v, unit == "%")


def _normalize_num(s: str) -> str:
    return (s or "").replace(",", ".").replace(" ", "").strip()


def _shorten(s: str, limit: int = 140) -> str:
    s = (s or "").strip()
    return s if len(s) <= limit else (s[:limit - 1].rstrip() + "…")


def _last_segment(name: str) -> str:
    s = (name or "").strip()
    if "/" in s:
        s = s.split("/")[-1].strip()
    s = re.sub(r"^\[\s*таблица\s*\]\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*[-–—]\s*", " — ", s)
    s = re.sub(r"\s+", " ", s).strip(" .")
    return s


_CAP_RE = re.compile(
    r"(?i)\bтаблица\s+([A-Za-zА-Яа-я]?\s*\d+(?:[.,]\d+)*)\b(?:\s*[—\-–:\u2013\u2014]\s*(.+))?"
)

def _parse_table_title(text: str) -> Tuple[Optional[str], Optional[str]]:
    m = _CAP_RE.search(text or "")
    if not m:
        return (None, None)
    raw_num = (m.group(1) or "").replace(" ", "")
    return (_normalize_num(raw_num), (m.group(2) or "").strip() or None)


def _compose_display_from_attrs(attrs_json: Optional[str], base: str, first_row_text: Optional[str]) -> str:
    """
    Стратегия формирования заголовка (совместима с bot.py):
      1) attrs.caption_num → 'Таблица N — tail/header/firstrow'
      2) иначе парсим номер из base
      3) иначе описание без номера
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
        tail_like = (tail or header_preview or first_row_text or "").strip()
        return f"Таблица {num}" + (f" — {_shorten(tail_like, 160)}" if tail_like else "")

    # без номера — пробуем распарсить из base и показать С номером
    num_b, title_b = _parse_table_title(_last_segment(base))
    if num_b:
        text_tail = title_b or first_row_text or header_preview
        return f"Таблица {num_b}" + (f" — {_shorten(text_tail, 160)}" if text_tail else "")

    # только описание
    if tail:
        return _shorten(str(tail), 160)
    if header_preview:
        return _shorten(str(header_preview), 160)
    if first_row_text:
        return _shorten(first_row_text, 160)

    s = _last_segment(base)
    s = re.sub(r"(?i)^\s*таблица\s+\d+(?:\.\d+)*\s*", "", s).strip(" —–-")
    return _shorten(s or "Таблица", 160)


def _find_table_anchor(uid: int, doc_id: int, num: str) -> Optional[Dict[str, Any]]:
    """
    Ищем «якорь» таблицы по номеру:
      1) по attrs.caption_num / attrs.label
      2) по section_path LIKE '%Таблица N%'
    Возвращаем первый подходящий чанк (table или table_row).
    """
    want = _normalize_num(num)
    con = get_conn()
    cur = con.cursor()
    has_ext = _table_has_columns(con, "chunks", ["element_type", "attrs"])

    row = None
    if has_ext:
        # 1) По JSON-атрибутам
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

        # 2) По section_path (при наличии element_type)
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
        # Ветка без element_type/attrs в схеме — безопасный фолбэк
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


def _fetch_table_rows(uid: int, doc_id: int, base: str, sample_rows_limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Возвращает список строк таблицы (в порядке id ASC):
    [{ "text": "...", "page": int }]
    Если sample_rows_limit задан — ограничиваем количество строк.
    """
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
        # старый индекс: ориентируемся по section_path
        sql = """
            SELECT text, page FROM chunks
            WHERE owner_id=? AND doc_id=? AND (section_path=? OR section_path LIKE ? || ' [row %')
            ORDER BY id ASC
        """
    cur.execute(sql, (uid, doc_id, base, base))
    rows = cur.fetchall() or []
    con.close()
    out = [{"text": (r["text"] or ""), "page": r["page"]} for r in rows]
    return out[:sample_rows_limit] if (sample_rows_limit is not None) else out


def _split_cells(line: str) -> List[str]:
    """
    Парсим строку в ячейки. В индексе строки таблицы обычно в формате 'c1 | c2 | c3'.
    Фолбэки: табы, множественные пробелы.
    """
    s = (line or "").strip()
    if " | " in s:
        cells = [c.strip() for c in s.split(" | ")]
    elif "\t" in s:
        cells = [c.strip() for c in s.split("\t")]
    else:
        # разбивка по 2+ пробелам как последний фолбэк
        cells = [c.strip() for c in re.split(r"\s{2,}", s)]
    return [c for c in cells if c != ""]


def _detect_header_and_make_matrix(rows_texts: List[str]) -> Tuple[List[str], List[List[str]]]:
    """
    Возвращает (headers, matrix_rows). Если явного заголовка нет — генерируем 'Колонка i'.
    """
    raw_rows = [_split_cells((t or "").splitlines()[0]) for t in rows_texts if (t or "").strip()]
    if not raw_rows:
        return ([], [])

    # Грубая эвристика: первая строка — заголовок, если она содержит <=25% числовых ячеек,
    # а следующая строка явно содержит числа.
    def frac_numeric(cells: List[str]) -> float:
        if not cells:
            return 0.0
        n = 0
        for c in cells:
            v, _ = _to_float(c)
            if v is not None:
                n += 1
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
        # нет очевидного заголовка — сгенерируем
        max_len = max(len(r) for r in raw_rows)
        headers = [f"Колонка {i+1}" for i in range(max_len)]
        data_rows = raw_rows

    # Выравниваем строки по длине заголовка
    W = len(headers)
    normalized: List[List[str]] = []
    for r in data_rows:
        row = (r + [""] * W)[:W]
        normalized.append(row)

    # Укорачиваем слишком длинные хэдеры
    headers = [_shorten(h, 80) for h in headers]
    return (headers, normalized)


def _normalize_percent_column(values: List[Optional[float]], had_pct_flags: List[bool]) -> List[Optional[float]]:
    """
    Делает значения в колонке консистентными для %:
    - если ячейка содержала '%' — в values уже доля (0..1)
    - если без '%', но число > 1 → считаем, что это "15" (%), делим на 100
    - если без '%', и число в [0..1] → уже доля
    """
    out: List[Optional[float]] = []
    for v, had in zip(values, had_pct_flags):
        if v is None:
            out.append(None)
            continue
        if had:
            out.append(v)  # уже доля
        else:
            out.append(v / 100.0 if v > 1.0 else v)
    return out


def _numeric_columns(headers: List[str], rows: List[List[str]]) -> List[Dict[str, Any]]:
    """
    Для каждого столбца решаем, числовой ли он; если да — собираем числа и базовую статистику.
    Возвращает список словарей по столбцам (включая нечисловые с флагом is_numeric=False).
    Обеспечивает 100% консистентность значений для процентных колонок.
    """
    cols: List[Dict[str, Any]] = []
    W = len(headers)
    H = len(rows)

    for j in range(W):
        raw_vals: List[Optional[float]] = []
        had_pct: List[bool] = []
        for i in range(H):
            v, is_pct = _to_float(rows[i][j])
            raw_vals.append(v)
            had_pct.append(is_pct)

        # критерий «числовой»: >=60% значимых чисел в колонке
        numeric_vals = [x for x in raw_vals if x is not None]
        is_numeric = (len(numeric_vals) >= max(1, int(0.6 * len(rows))))

        if not is_numeric or not numeric_vals:
            cols.append({
                "index": j,
                "name": headers[j],
                "is_numeric": False,
                "is_percent": False,
                "count": 0,
                "sum": None,
                "mean": None,
                "min": None,
                "max": None,
                "values": [],
                "values_norm": [None] * H,
            })
            continue

        # Решаем, что это колонка процентов:
        #  - явные '%' в >= половины числовых ячеек
        #  - либо нет '%', но >=80% значений находятся в [0..1] и max<=1.0
        pct_count = sum(1 for i, v in enumerate(raw_vals) if v is not None and had_pct[i])
        frac_count = sum(1 for v in numeric_vals if 0.0 <= v <= 1.0)
        nnum = len(numeric_vals)
        max_v = max(numeric_vals) if numeric_vals else 0.0

        is_percent_col = (pct_count >= (nnum + 1) // 2) or (pct_count == 0 and nnum >= 3 and frac_count >= int(0.8 * nnum) and max_v <= 1.0)

        # Нормализуем значения, если процентовая колонка
        if is_percent_col:
            norm_vals = _normalize_percent_column([v if v is not None else None for v in raw_vals], had_pct)
            use_vals = [v for v in norm_vals if v is not None]
        else:
            norm_vals = [v if v is not None else None for v in raw_vals]
            use_vals = numeric_vals

        n = len(use_vals)
        s = sum(use_vals) if use_vals else None
        mn = min(use_vals) if use_vals else None
        mx = max(use_vals) if use_vals else None
        mean = (s / n) if use_vals else None

        cols.append({
            "index": j,
            "name": headers[j],
            "is_numeric": True,
            "is_percent": bool(is_percent_col),
            "count": n,
            "sum": s,
            "mean": mean,
            "min": mn,
            "max": mx,
            "values": use_vals,        # агрегированные (без None), оставлено для совместимости
            "values_norm": norm_vals,  # массив по строкам (None там, где пусто)
        })
    return cols


def _top_rows_by_column(rows: List[List[str]],
                        col_idx: int,
                        values_norm_by_row: List[Optional[float]],
                        row_label_idx: int,
                        top_k: int) -> List[Dict[str, Any]]:
    """
    Возвращает топ-строки по нормализованному значению столбца col_idx.
    В качестве подписи строки берём row_label_idx (обычно 0) или '#строка i'.
    """
    pairs: List[Tuple[int, float]] = []
    for i, v in enumerate(values_norm_by_row):
        if v is None:
            continue
        pairs.append((i, float(v)))
    pairs.sort(key=lambda x: -x[1])

    out: List[Dict[str, Any]] = []
    for i, v in pairs[:max(1, top_k)]:
        label = (rows[i][row_label_idx].strip() if row_label_idx < len(rows[i]) and rows[i][row_label_idx].strip() else f"Строка {i+1}")
        out.append({"row": _shorten(label, 80), "value": float(v)})
    return out


def _format_number(x: Optional[float], is_percent: bool, lang: str) -> str:
    if x is None:
        return "—"
    val = x * 100.0 if is_percent else x
    if lang == "en":
        return f"{val:,.2f}" + ("%" if is_percent else "")
    # ru: запятая как десятичный разделитель
    s = f"{val:,.2f}"
    s = s.replace(",", " ").replace(".", ",")
    return s + ("%" if is_percent else "")


def _render_text_report(display: str,
                        headers: List[str],
                        rows: List[List[str]],
                        cols: List[Dict[str, Any]],
                        top_k: int,
                        lang: str) -> str:
    if not rows:
        return ("Таблица пуста или не распознаны строки."
                if lang != "en" else "The table is empty or rows were not recognized.")

    # Выберем «подпись строки» — чаще всего это первый столбец
    row_label_idx = 0

    lines: List[str] = []
    if lang == "en":
        lines.append(f"Summary for {display}:")
    else:
        lines.append(f"Краткий разбор для {display}:")

    # По каждому числовому столбцу — базовая статистика и топы
    for col in cols:
        if not col["is_numeric"]:
            continue
        name = col["name"] or f"Колонка {col['index']+1}"
        c_count = col["count"]
        c_min = _format_number(col["min"], col["is_percent"], lang)
        c_max = _format_number(col["max"], col["is_percent"], lang)
        c_mean = _format_number(col["mean"], col["is_percent"], lang)
        c_sum = _format_number(col["sum"], col["is_percent"], lang)

        if lang == "en":
            lines.append(f"• Column «{name}»: n={c_count}, min={c_min}, max={c_max}, mean={c_mean}, sum={c_sum}.")
        else:
            lines.append(f"• Колонка «{name}»: n={c_count}, минимум={c_min}, максимум={c_max}, среднее={c_mean}, сумма={c_sum}.")

        # Топ строки (по нормализованным значениям)
        top_rows = _top_rows_by_column(rows, col["index"], col["values_norm"], row_label_idx, top_k)
        if top_rows:
            if lang == "en":
                lines.append("  Top rows:")
            else:
                lines.append("  Топ-строки:")
            for tr in top_rows:
                val = _format_number(tr["value"], col["is_percent"], lang)
                lines.append(f"  – {tr['row']}: {val}")

    if len(lines) == 1:
        # не нашлось числовых столбцов
        if lang == "en":
            return f"{lines[0]}\nThere are no numeric columns to analyze."
        return f"{lines[0]}\nЧисловые столбцы для анализа не обнаружены."
    return "\n".join(lines)


# ---------------------------- Публичный API ----------------------------

def analyze_table_by_num(uid: int,
                         doc_id: int,
                         number: str,
                         top_k: int = 5,
                         lang: str = "ru",
                         *,
                         rows_limit: Optional[int] = 10,
                         include_all_values: bool = False) -> Dict[str, Any]:
    """
    Главная функция аналитики таблицы по номеру.
    Возвращает словарь:
    {
      "ok": bool,
      "error": str|None,
      "num": "2.1",
      "display": "Таблица 2.1 — ...",
      "where": {"page": int|None, "section_path": str|None},
      "headers": [str,...],
      "rows_preview": [ [cell,...], ... ],   # по умолчанию до 10 строк (rows_limit)
      "rows_all": [ [cell,...], ... ]        # (опц.) при include_all_values=True — вся таблица без усечения
      "rows_count": int,                     # общее кол-во строк данных
      "columns_count": int,                  # число колонок
      "numeric_summary": [
          { "col": "Название", "is_percent": bool, "count": int, "min": float|None, "max": float|None,
            "mean": float|None, "sum": float|None, "top": [ {"row": str, "value": float}, ... ] }
      ],
      "text": "Краткий текст отчёта"
    }
    """
    num = _normalize_num(number)
    if not num:
        return {"ok": False, "error": "Не указан номер таблицы.", "num": number}

    anchor = _find_table_anchor(uid, doc_id, num)
    if not anchor:
        return {"ok": False, "error": f"Таблица {number} не найдена.", "num": number}

    page = anchor.get("page")
    sec = anchor.get("section_path") or ""
    base = _base_from_section(sec)

    # Сформируем display с учётом attrs и первой строки
    # Берём первую строку таблицы (если есть) для превью
    row_objs = _fetch_table_rows(uid, doc_id, base)
    first_row_text = None
    if row_objs:
        first_line = (row_objs[0]["text"] or "").split("\n")[0]
        if first_line:
            first_row_text = " — ".join([c.strip() for c in _split_cells(first_line) if c.strip()])

    attrs_json = anchor.get("attrs")
    if isinstance(attrs_json, dict):
        attrs_json = json.dumps(attrs_json, ensure_ascii=False)
    display = _compose_display_from_attrs(attrs_json, base, first_row_text)

    # Матрица
    rows_texts = [r["text"] for r in row_objs]
    headers, matrix = _detect_header_and_make_matrix(rows_texts)

    # Числовые столбцы (с консистентной нормализацией процентов)
    cols = _numeric_columns(headers, matrix)

    # Готовим краткий текстовый отчёт
    text = _render_text_report(display, headers, matrix, cols, top_k=top_k, lang=lang)

    # Сериализуем numeric_summary
    numeric_summary: List[Dict[str, Any]] = []
    for c in cols:
        if not c["is_numeric"]:
            continue
        top_rows = _top_rows_by_column(matrix, c["index"], c["values_norm"], 0, top_k)
        numeric_summary.append({
            "col": c["name"],
            "is_percent": bool(c["is_percent"]),
            "count": int(c["count"]),
            "min": c["min"],
            "max": c["max"],
            "mean": c["mean"],
            "sum": c["sum"],
            "top": top_rows,
        })

    # Превью с лимитом строк (по умолчанию 10)
    # Важно: это влияет только на превью, не на расчёты.
    preview_limit = None if rows_limit is None or rows_limit < 0 else int(rows_limit)
    rows_preview = matrix if preview_limit is None else matrix[:preview_limit]
    rows_preview = [row[:len(headers)] for row in rows_preview]

    result: Dict[str, Any] = {
        "ok": True,
        "error": None,
        "num": num,
        "display": display,
        "where": {"page": page, "section_path": base},
        "headers": headers,
        "rows_preview": rows_preview,
        "rows_count": len(matrix),
        "columns_count": len(headers),
        "numeric_summary": numeric_summary,
        "text": text,
    }

    # При необходимости отдаём всю таблицу без усечения
    if include_all_values:
        result["rows_all"] = [row[:len(headers)] for row in matrix]

    return result


def extract_table_matrix_by_num(uid: int,
                                doc_id: int,
                                number: str) -> Dict[str, Any]:
    """
    Утилита для извлечения всей матрицы таблицы (без аналитики).
    Возвращает:
    {
      "ok": bool,
      "error": str|None,
      "num": "2.1",
      "display": "Таблица 2.1 — ...",
      "where": {"page": int|None, "section_path": str|None},
      "headers": [str,...],
      "rows": [ [cell,...], ... ],
      "rows_count": int,
      "columns_count": int
    }
    """
    num = _normalize_num(number)
    if not num:
        return {"ok": False, "error": "Не указан номер таблицы.", "num": number}

    anchor = _find_table_anchor(uid, doc_id, num)
    if not anchor:
        return {"ok": False, "error": f"Таблица {number} не найдена.", "num": number}

    page = anchor.get("page")
    sec = anchor.get("section_path") or ""
    base = _base_from_section(sec)

    row_objs = _fetch_table_rows(uid, doc_id, base)
    rows_texts = [r["text"] for r in row_objs]

    # Для display можно попытаться собрать хвост из первой строки
    first_row_text = None
    if row_objs:
        first_line = (row_objs[0]["text"] or "").split("\n")[0]
        if first_line:
            first_row_text = " — ".join([c.strip() for c in _split_cells(first_line) if c.strip()])

    attrs_json = anchor.get("attrs")
    if isinstance(attrs_json, dict):
        attrs_json = json.dumps(attrs_json, ensure_ascii=False)
    display = _compose_display_from_attrs(attrs_json, base, first_row_text)

    headers, matrix = _detect_header_and_make_matrix(rows_texts)

    return {
        "ok": True,
        "error": None,
        "num": num,
        "display": display,
        "where": {"page": page, "section_path": base},
        "headers": headers,
        "rows": [row[:len(headers)] for row in matrix],
        "rows_count": len(matrix),
        "columns_count": len(headers),
    }
