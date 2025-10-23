# app/table_reconstruct.py
from __future__ import annotations
import re
import json
from typing import List, Dict, Any, Optional, Tuple

from .db import get_conn
from .vision_tables_ocr import ocr_tables_by_numbers, ocr_table_section

# =============================== ВСПОМОГАТЕЛЬНЫЕ УТИЛИТЫ ===============================

def _table_has_columns(con, table: str, cols: List[str]) -> bool:
    cur = con.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    have = {row[1] for row in cur.fetchall()}
    return all(c in have for c in cols)

def _shorten(s: str, limit: int = 120) -> str:
    s = (s or "").strip()
    if len(s) <= limit:
        return s
    return s[:limit - 1].rstrip() + "…"

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

# «Таблица 2.1 — …», «Table 3: …»
_TABLE_TITLE_RE = re.compile(
    r"(?i)\b(?:таблица|table)\s+([A-Za-zА-Яа-я]?\s*\d+(?:[.,]\d+)*)\b(?:\s*[—\-–:]\s*(.+))?"
)

def _parse_table_title(text: str) -> Tuple[Optional[str], Optional[str]]:
    t = (text or "").strip()
    m = _TABLE_TITLE_RE.search(t)
    if not m:
        return (None, None)
    raw = (m.group(1) or "").replace(" ", "")
    num = raw.replace(",", ".") or None
    title = (m.group(2) or "").strip() or None
    return (num, title)

def _compose_display_from_attrs(attrs_json: Optional[str], base: str, first_row_text: Optional[str]) -> str:
    """
    Правила отображения (как в bot.py):
      1) есть caption_num → 'Таблица N — tail/header/firstrow'.
      2) иначе без номера → только описание.
      3) fallback — парсим из base.
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

def _base_from_section_path(section_path: str) -> str:
    """
    Отрезаем хвост ' [row N]' → получаем «базовое имя таблицы».
    """
    s = section_path or ""
    i = s.find(" [row ")
    if i > 0:
        return s[:i]
    return s

def _row_index_from_section_path(section_path: str) -> int:
    """
    Извлекаем номер строки из '... [row N]'. Если нет — 10**9 (чтобы ушло в конец).
    """
    m = re.search(r"\[row\s+(\d+)\]", section_path or "", re.IGNORECASE)
    if not m:
        return 10**9
    try:
        return int(m.group(1))
    except Exception:
        return 10**9

def _split_cells(row_text: str) -> List[str]:
    """
    Индексатор для строк таблиц использует формат вида "a | b | c".
    Аккуратно разбиваем и чистим пробелы.
    """
    t = (row_text or "").strip()
    # заменим множественные разделители на единый
    parts = [c.strip() for c in t.split(" | ")]
    # fallback: если нет " | ", попробуем 2+ пробелов
    if len(parts) == 1:
        parts = [c.strip() for c in re.split(r"\s{2,}", t)]
    return [p for p in parts if p != ""]

def _to_markdown(headers: List[str], rows: List[List[str]]) -> str:
    if not headers and not rows:
        return ""
    cols = max(len(headers), max((len(r) for r in rows), default=0))
    if cols == 0:
        return ""
    # нормализуем ширину
    hh = headers + [""] * (cols - len(headers))
    norm_rows: List[List[str]] = []
    for r in rows:
        rr = list(r) + [""] * (cols - len(r))
        norm_rows.append(rr)
    # собираем Markdown
    out: List[str] = []
    out.append("| " + " | ".join(hh) + " |")
    out.append("| " + " | ".join(["---"] * cols) + " |")
    for r in norm_rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)

def _to_csv(headers: List[str], rows: List[List[str]]) -> str:
    def esc(x: str) -> str:
        x = x.replace('"', '""')
        if ("," in x) or ('"' in x) or ("\n" in x):
            return f'"{x}"'
        return x
    cols = max(len(headers), max((len(r) for r in rows), default=0))
    if cols == 0:
        return ""
    hh = headers + [""] * (cols - len(headers))
    out = [",".join(esc(c) if c is not None else "" for c in hh)]
    for r in rows:
        rr = list(r) + [""] * (cols - len(r))
        out.append(",".join(esc(c) if c is not None else "" for c in rr))
    return "\n".join(out)

# =============================== ЧТЕНИЕ ИЗ БД: TABLE_ROW ===============================

def _fetch_table_rows(owner_id: int, doc_id: int, base_section: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Читает все строки таблицы (element_type='table_row') по base_section.
    Возвращает (rows, attrs_json_first_row).
      rows: [{section_path, text, page, attrs?}, ...] — отсортированы по [row N]
    """
    con = get_conn()
    cur = con.cursor()
    has_ext = _table_has_columns(con, "chunks", ["element_type", "attrs"])
    if has_ext:
        cur.execute(
            """
            SELECT section_path, text, page, attrs
            FROM chunks
            WHERE owner_id=? AND doc_id=? AND element_type='table_row'
              AND section_path LIKE ? || ' [row %'
            ORDER BY id ASC
            """,
            (owner_id, doc_id, base_section),
        )
    else:
        # fallback для старых БД: ищем по маске на секцию + по тексту prefix '[Таблица]'
        cur.execute(
            """
            SELECT section_path, text, page, NULL as attrs
            FROM chunks
            WHERE owner_id=? AND doc_id=? 
              AND section_path LIKE ? || ' [row %'
            ORDER BY id ASC
            """,
            (owner_id, doc_id, base_section),
        )
    rows = cur.fetchall() or []
    con.close()

    py_rows = [dict(r) for r in rows]
    # сортируем по номеру row
    py_rows.sort(key=lambda r: _row_index_from_section_path(r.get("section_path") or ""))

    # attrs первой строки (для header_rows/header_preview и т.д.)
    attrs_first = None
    for r in py_rows:
        if r.get("attrs"):
            attrs_first = r["attrs"]
            break
    return py_rows, attrs_first

def _find_table_base_by_number(owner_id: int, doc_id: int, num: str) -> Optional[Tuple[str, Optional[int], Optional[str]]]:
    """
    Ищем «базовое имя» таблицы по номеру.
    Возвращает (base_section, page, attrs_json_first_row) или None.
    """
    con = get_conn()
    cur = con.cursor()
    has_ext = _table_has_columns(con, "chunks", ["element_type", "attrs"])

    # Сначала пробуем attrs.caption_num/label на table_row
    found = None
    if has_ext:
        cur.execute(
            """
            SELECT section_path, page, attrs FROM chunks
            WHERE owner_id=? AND doc_id=? AND element_type='table_row'
              AND (attrs LIKE ? OR attrs LIKE ?)
            ORDER BY id ASC LIMIT 1
            """,
            (owner_id, doc_id, f'%\"caption_num\": \"{num}\"%', f'%\"label\": \"{num}\"%'),
        )
        found = cur.fetchone()

    if not found:
        # Ищем по подписи в section_path (любая строка таблицы)
        cur.execute(
            """
            SELECT section_path, page, """ + ("attrs" if has_ext else "NULL as attrs") + """
            FROM chunks
            WHERE owner_id=? AND doc_id=? AND section_path LIKE ? COLLATE NOCASE
            ORDER BY id ASC LIMIT 1
            """,
            (owner_id, doc_id, f"%Таблица {num}%"),
        )
        found = cur.fetchone()

    con.close()
    if not found:
        return None

    sec = found["section_path"]
    base = _base_from_section_path(sec)
    return (base, found["page"], found["attrs"] if "attrs" in found.keys() else None)

def _headers_from_attrs(attrs_json: Optional[str]) -> Tuple[int, List[str]]:
    """
    Пытаемся понять, сколько «верхних» строк являются заголовком.
    Ожидаемые поля (не строго): header_rows: int, header_preview: "A | B | C".
    Возвращает (header_rows, header_preview_cells).
    """
    if not attrs_json:
        return (1, [])
    try:
        a = json.loads(attrs_json or "{}")
    except Exception:
        return (1, [])
    header_rows = 1
    try:
        if isinstance(a.get("header_rows"), int) and a["header_rows"] > 0:
            header_rows = a["header_rows"]
    except Exception:
        pass
    header_preview_cells: List[str] = []
    hp = a.get("header_preview")
    if isinstance(hp, str) and hp.strip():
        header_preview_cells = _split_cells(hp)
    return (header_rows, header_preview_cells)

def _display_from_any(attrs_json: Optional[str], base: str, rows: List[Dict[str, Any]]) -> str:
    first_row_text = None
    if rows:
        first_line = (rows[0].get("text") or "").split("\n")[0]
        if first_line:
            first_row_text = " — ".join([c.strip() for c in first_line.split(" | ") if c.strip()])
    return _compose_display_from_attrs(attrs_json, base, first_row_text)

# =============================== ОСНОВНАЯ СБОРКА ТАБЛИЦЫ ===============================

def _assemble_table_from_rows(base: str,
                              page: Optional[int],
                              rows: List[Dict[str, Any]],
                              attrs_json: Optional[str]) -> Dict[str, Any]:
    """
    Превращаем список строк таблицы в структуру с headers/rows/markdown/csv.
    """
    # Заголовочные строки
    header_rows_count, header_preview_cells = _headers_from_attrs(attrs_json)

    # Разбираем все строки в ячейки
    parsed: List[List[str]] = []
    for r in rows:
        txt = (r.get("text") or "").split("\n", 1)[0]  # только первая строка чанка
        cells = _split_cells(txt)
        parsed.append(cells)

    headers: List[str] = []
    body: List[List[str]] = []

    if parsed:
        if header_rows_count >= 1 and len(parsed) >= header_rows_count:
            # Склеиваем несколько строк заголовка (если их несколько) — по ячейкам с переносом
            acc = parsed[0]
            for i in range(1, header_rows_count):
                if i < len(parsed):
                    add = parsed[i]
                    # расширяем по ширине
                    width = max(len(acc), len(add))
                    acc = acc + [""] * (width - len(acc))
                    add = add + [""] * (width - len(add))
                    acc = [f"{a}\n{b}".strip() if b else a for a, b in zip(acc, add)]
            headers = acc
            body = parsed[header_rows_count:]
        else:
            headers = parsed[0]
            body = parsed[1:]

    # Если headers пусты — попробуем из header_preview
    if not headers and header_preview_cells:
        headers = header_preview_cells

    # Markdown/CSV
    md = _to_markdown(headers, body)
    csv_text = _to_csv(headers, body)

    display = _display_from_any(attrs_json, base, rows)
    card = {
        "source": "rows",
        "display": _strip_table_prefix(display),
        "where": {"page": page, "section_path": base},
        "headers": headers,
        "rows": body,
        "markdown": md,
        "csv": csv_text,
        "images": _collect_images_for_section_paths(rows),
        "note": "",
    }
    return card

def _collect_images_for_section_paths(rows: List[Dict[str, Any]]) -> List[str]:
    """Пытаемся вытянуть изображения из attrs любой из строк (на случай таблиц-сканов)."""
    images: List[str] = []
    for r in rows:
        attrs_raw = r.get("attrs")
        if not attrs_raw:
            continue
        try:
            a = json.loads(attrs_raw or "{}")
            for p in (a.get("images") or []):
                if p and p not in images:
                    images.append(p)
        except Exception:
            continue
    return images

# =============================== ПУБЛИЧНЫЕ ФУНКЦИИ ===============================

def reconstruct_table_by_section(owner_id: int,
                                 doc_id: int,
                                 section_path: str,
                                 *,
                                 lang: str = "ru",
                                 prefer_markdown: bool = True) -> Dict[str, Any]:
    """
    Реконструирует таблицу по её «базовой» секции (без ' [row N]').
    Если в БД нет строк — fallback на vision/OCR по всей секции.
    """
    base = _base_from_section_path(section_path)
    rows, attrs_first = _fetch_table_rows(owner_id, doc_id, base)
    page = rows[0]["page"] if rows else None

    if rows:
        return _assemble_table_from_rows(base, page, rows, attrs_first)

    # Фолбэк: OCR/Vision по секции
    ocr = ocr_table_section(owner_id, doc_id, base, lang=lang, prefer_markdown=prefer_markdown)
    # Приведём к общей форме
    return {
        "source": "vision_ocr",
        "display": ocr.get("display") or _last_segment(base) or "Таблица",
        "where": ocr.get("where") or {"page": page, "section_path": base},
        "headers": [],
        "rows": [],
        "markdown": ocr.get("markdown") or "",
        "csv": ocr.get("csv") or "",
        "images": ocr.get("images") or [],
        "note": ocr.get("note") or "Текстовые строки таблицы в индексе не найдены; использован OCR.",
    }

def reconstruct_table_by_number(owner_id: int,
                                doc_id: int,
                                num: str,
                                *,
                                lang: str = "ru",
                                prefer_markdown: bool = True) -> Dict[str, Any]:
    """
    Реконструирует таблицу по её номеру (например, '2.1' или 'A.1').
    Порядок:
      1) ищем table_row и собираем;
      2) если нет — OCR/Vision по номеру.
    """
    # Нормализуем номер: 2,1 → 2.1
    want = (num or "").replace(",", ".").replace(" ", "")
    if not want:
        return {
            "source": "none",
            "display": f"Таблица {num}",
            "where": {"page": None, "section_path": ""},
            "headers": [],
            "rows": [],
            "markdown": "",
            "csv": "",
            "images": [],
            "note": "Не указан номер таблицы.",
        }

    found = _find_table_base_by_number(owner_id, doc_id, want)
    if found:
        base, page, attrs_first = found
        rows, _attrs_again = _fetch_table_rows(owner_id, doc_id, base)
        if rows:
            # attrs_first мог быть None, возьмём то что есть
            attrs_json = attrs_first
            if not attrs_json:
                for r in rows:
                    if r.get("attrs"):
                        attrs_json = r["attrs"]
                        break
            return _assemble_table_from_rows(base, page, rows, attrs_json)

    # Фолбэк: OCR/Vision по номеру
    cards = ocr_tables_by_numbers(owner_id, doc_id, [want], lang=lang, prefer_markdown=prefer_markdown)
    if cards:
        c = cards[0]
        return {
            "source": "vision_ocr",
            "display": c.get("display") or f"Таблица {want}",
            "where": c.get("where") or {"page": None, "section_path": ""},
            "headers": [],
            "rows": [],
            "markdown": c.get("markdown") or "",
            "csv": c.get("csv") or "",
            "images": c.get("images") or [],
            "note": c.get("note") or "Текстовые строки таблицы в индексе не найдены; использован OCR.",
        }

    return {
        "source": "none",
        "display": f"Таблица {want}",
        "where": {"page": None, "section_path": ""},
        "headers": [],
        "rows": [],
        "markdown": "",
        "csv": "",
        "images": [],
        "note": "Таблица с таким номером не найдена ни в индексированных строках, ни через OCR.",
    }

def reconstruct_table(owner_id: int,
                      doc_id: int,
                      *,
                      number: Optional[str] = None,
                      section_path: Optional[str] = None,
                      lang: str = "ru",
                      prefer_markdown: bool = True) -> Dict[str, Any]:
    """
    Универсальная обёртка:
      - если задан number — пробуем по номеру;
      - иначе по section_path;
      - иначе возвращаем пустую карточку.
    """
    if number:
        return reconstruct_table_by_number(owner_id, doc_id, number, lang=lang, prefer_markdown=prefer_markdown)
    if section_path:
        return reconstruct_table_by_section(owner_id, doc_id, section_path, lang=lang, prefer_markdown=prefer_markdown)
    return {
        "source": "none",
        "display": "Таблица",
        "where": {"page": None, "section_path": ""},
        "headers": [],
        "rows": [],
        "markdown": "",
        "csv": "",
        "images": [],
        "note": "Не задан номер или секция таблицы.",
    }


__all__ = [
    "reconstruct_table",
    "reconstruct_table_by_number",
    "reconstruct_table_by_section",
]
