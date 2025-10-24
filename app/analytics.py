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

_NUM_RE = re.compile(r"^[+-]?\d{1,3}(?:[\s\u00A0]\d{3})*(?:[.,]\d+)?%?$|^[+-]?\d+(?:[.,]\d+)?%?$")

def _to_float(s: str) -> Tuple[Optional[float], bool]:
    """
    Пробуем распарсить число. Возвращаем (value, is_percent).
    Понимаем пробелы-разделители тысяч и запятую как десятичный разделитель.
    """
    if s is None:
        return (None, False)
    raw = str(s).strip().replace("\u00A0", " ")
    if not _NUM_RE.match(raw):
        return (None, False)
    is_percent = raw.endswith("%")
    raw = raw.rstrip("%").strip()
    raw = raw.replace(" ", "")
    # если есть и точка, и запятая, оставляем последнюю как десятичный разделитель
    if "," in raw and "." in raw:
        if raw.rfind(",") > raw.rfind("."):
            raw = raw.replace(".", "")
            raw = raw.replace(",", ".")
        else:
            raw = raw.replace(",", "")
    else:
        raw = raw.replace(",", ".")
    try:
        val = float(raw)
        if is_percent:
            # Для согласованности приводим к 0..1 (а в отчёте показываем как %)
            return (val / 100.0, True)
        return (val, False)
    except Exception:
        return (None, False)

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
        n = sum(1 for c in cells if _to_float(c)[0] is not None)
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

def _numeric_columns(headers: List[str], rows: List[List[str]]) -> List[Dict[str, Any]]:
    """
    Для каждого столбца решаем, числовой ли он; если да — собираем числа и базовую статистику.
    Возвращает список словарей по столбцам (включая нечисловые с флагом is_numeric=False).
    """
    cols: List[Dict[str, Any]] = []
    W = len(headers)
    for j in range(W):
        col_vals: List[Optional[float]] = []
        col_is_percent = 0
        for r in rows:
            v, is_percent = _to_float(r[j])
            if is_percent:
                col_is_percent += 1
            col_vals.append(v)

        # критерий «числовой»: >=60% значимых чисел в колонке
        numeric_vals = [x for x in col_vals if x is not None]
        is_numeric = (len(numeric_vals) >= max(1, int(0.6 * len(rows))))

        if is_numeric and numeric_vals:
            n = len(numeric_vals)
            s = sum(numeric_vals)
            mn = min(numeric_vals)
            mx = max(numeric_vals)
            mean = s / n if n else None
            cols.append({
                "index": j,
                "name": headers[j],
                "is_numeric": True,
                "is_percent": (col_is_percent > len(rows) / 2),
                "count": n,
                "sum": s,
                "mean": mean,
                "min": mn,
                "max": mx,
                "values": numeric_vals,  # оставим для топов
            })
        else:
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
            })
    return cols

def _top_rows_by_column(rows: List[List[str]], col_idx: int, values: List[float], row_label_idx: int, top_k: int) -> List[Dict[str, Any]]:
    """
    Возвращает топ-строки по значению столбца col_idx.
    В качестве подписи строки берём первую колонку (row_label_idx) или '#строка i'.
    """
    pairs: List[Tuple[int, float]] = []
    for i, r in enumerate(rows):
        v, _ = _to_float(r[col_idx])
        if v is None:
            continue
        pairs.append((i, v))
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

        # Топ строки
        top_rows = _top_rows_by_column(rows, col["index"], col["values"], row_label_idx, top_k)
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

    # Числовые столбцы
    cols = _numeric_columns(headers, matrix)

    # Готовим краткий текстовый отчёт
    text = _render_text_report(display, headers, matrix, cols, top_k=top_k, lang=lang)

    # Сериализуем numeric_summary (без «values»)
    numeric_summary: List[Dict[str, Any]] = []
    for c in cols:
        if not c["is_numeric"]:
            continue
        top_rows = _top_rows_by_column(matrix, c["index"], c["values"], 0, top_k)
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
