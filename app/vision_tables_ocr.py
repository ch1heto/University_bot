# app/vision_tables_ocr.py
from __future__ import annotations
import re
import json
import base64
import mimetypes
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from .db import get_conn
from .polza_client import chat_with_gpt  # единая Chat API (поддерживает image_url)

# Опциональные локальные OCR-зависимости (мягкий импорт)
try:
    import pytesseract  # type: ignore
    from PIL import Image  # type: ignore
    _HAS_LOCAL_OCR = True
except Exception:
    _HAS_LOCAL_OCR = False


# ============================== ВНУТРЕННИЕ УТИЛИТЫ ==============================

def _has_columns(con, table: str, cols: List[str]) -> bool:
    cur = con.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    have = {row[1] for row in cur.fetchall()}
    return all(c in have for c in cols)

def _fetchone(con, sql: str, params: Tuple[Any, ...]) -> Optional[Dict[str, Any]]:
    cur = con.cursor()
    cur.execute(sql, params)
    row = cur.fetchone()
    return (dict(row) if row else None)

def _fetchall(con, sql: str, params: Tuple[Any, ...]) -> List[Dict[str, Any]]:
    cur = con.cursor()
    cur.execute(sql, params)
    rows = cur.fetchall()
    return [dict(r) for r in rows]

def _guess_mime(path: str) -> str:
    mt, _ = mimetypes.guess_type(path)
    return mt or "image/png"

def _file_to_data_url(path: str) -> Optional[str]:
    """Локальный файл → data URL (base64), чтобы передать изображение через image_url."""
    try:
        p = Path(path)
        if not p.exists() or not p.is_file():
            return None
        b = p.read_bytes()
        b64 = base64.b64encode(b).decode("ascii")
        mime = _guess_mime(path)
        return f"data:{mime};base64,{b64}"
    except Exception:
        return None

def _extract_images_from_attrs(attrs_raw: Any) -> List[str]:
    """Извлекаем список путей изображений из JSON-attrs (если они там есть)."""
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
            sp = str(p)
            if sp and Path(sp).exists():
                out.append(sp)
        return out
    except Exception:
        return []

def _collect_images_for_section(owner_id: int, doc_id: int, section_path: str) -> List[str]:
    """
    Подтягиваем images из attrs по всем чанкам данной секции.
    Нужен для случаев, когда картинки лежат не в первом чанке.
    """
    con = get_conn()
    try:
        if not _has_columns(con, "chunks", ["attrs"]):
            return []
        rows = _fetchall(
            con,
            "SELECT attrs FROM chunks "
            "WHERE owner_id=? AND doc_id=? AND section_path=? AND attrs IS NOT NULL "
            "ORDER BY id ASC LIMIT 16",
            (owner_id, doc_id, section_path),
        )
    finally:
        con.close()
    images: List[str] = []
    for rr in rows:
        for p in _extract_images_from_attrs(rr.get("attrs")):
            if p not in images:
                images.append(p)
    return images

# «Таблица 2.1 — ...», «Table 5: ...»
_TABLE_TITLE_RE = re.compile(
    r"(?i)\b(?:таблица|table)\s+([A-Za-zА-Яа-я]?\s*\d+(?:[.,]\d+)*)\b(?:\s*[—\-–:]\s*(.+))?"
)

def _normalize_num(n: str | None) -> Optional[str]:
    if not n:
        return None
    return n.replace(",", ".").replace(" ", "").strip() or None

def _parse_table_title(text_or_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Извлекаем номер и хвост подписи из строки вида 'Таблица 2.3 — Название'.
    Возвращает (num, title).
    """
    t = (text_or_path or "").strip()
    m = _TABLE_TITLE_RE.search(t)
    if not m:
        return (None, None)
    num = _normalize_num(m.group(1) or "")
    title = (m.group(2) or "").strip() or None
    return (num, title)

def _last_segment(name: str) -> str:
    s = (name or "").strip()
    if "/" in s:
        s = s.split("/")[-1].strip()
    s = re.sub(r"^\[\s*таблица\s*\]\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*[-–—]\s*", " — ", s)
    s = re.sub(r"\s+", " ", s).strip(" .")
    return s


# ============================== ПОИСК ТАБЛИЦ В БД ==============================

def _find_table_chunk_by_number(owner_id: int, doc_id: int, num: str) -> Optional[Dict[str, Any]]:
    """
    Ищем первый чанк «таблицы» по номеру:
      1) attrs.caption_num / attrs.label == num
      2) section_path или text содержит «Таблица {num}» (фолбэк)
    Возвращает {page, section_path, text, attrs} или None.
    """
    want = _normalize_num(num) or ""
    con = get_conn()
    try:
        has_ext = _has_columns(con, "chunks", ["element_type", "attrs"])

        # По attrs.caption_num / attrs.label
        found = None
        if has_ext:
            # Ищем и строковое и числовое представление (на всякий случай)
            like1 = f'%\"caption_num\": \"{want}\"%'
            like2 = f'%\"label\": \"{want}\"%'
            like3 = f'%\"caption_num\": {want}%'  # если поле было числом
            like4 = f'%\"label\": {want}%'
            found = _fetchone(
                con,
                "SELECT page, section_path, text, attrs FROM chunks "
                "WHERE owner_id=? AND doc_id=? AND element_type IN ('table','table_row') "
                "AND (attrs LIKE ? OR attrs LIKE ? OR attrs LIKE ? OR attrs LIKE ?) "
                "ORDER BY id ASC LIMIT 1",
                (owner_id, doc_id, like1, like2, like3, like4),
            )

        # По подписи в section_path/text
        if not found:
            sel = "SELECT page, section_path, text" + (", attrs" if has_ext else "")
            found = _fetchone(
                con,
                f"{sel} FROM chunks "
                "WHERE owner_id=? AND doc_id=? AND "
                "(section_path LIKE ? COLLATE NOCASE OR text LIKE ? COLLATE NOCASE) "
                "ORDER BY id ASC LIMIT 1",
                (owner_id, doc_id, f"%Таблица {want}%", f"%Таблица {want}%"),
            )
        return found
    finally:
        con.close()


# ============================== VISION: ПРЕОБРАЗОВАНИЕ В ТАБЛИЦУ ==============================

def _vision_table_markdown_for_images(image_paths: List[str], *, lang: str = "ru",
                                      max_cols: int = 14) -> Dict[str, str]:
    """
    Отдаём 1–2 изображения в vision-модель через chat_with_gpt + image_url.
    Просим вывести ЧИСТО Markdown-таблицу (и дублируем в CSV).
    Возвращает {"markdown": "...", "csv": "..."} (поля могут быть пустыми).
    """
    if not image_paths:
        return {"markdown": "", "csv": ""}

    chosen = [p for p in image_paths if p][:2]
    image_inputs = []
    for p in chosen:
        data = _file_to_data_url(p)
        if data:
            image_inputs.append({"type": "image_url", "image_url": {"url": data}})

    if not image_inputs:
        return {"markdown": "", "csv": ""}

    # Инструкции для VLM: только таблица, без пояснений
    sys_prompt = (
        "Ты помощник по академическим текстам. Извлеки из изображения таблицу и верни ТОЛЬКО её в виде Markdown.\n"
        f"- Язык заголовков и значений: {'русский' if lang.startswith('ru') else 'English'}.\n"
        "- Сохраняй структуру: заголовки колонок, объединения ячеек — по возможности.\n"
        "- Если в таблице больше столбцов, чем помещается, используй переносы строк внутри ячеек.\n"
        "- НИЧЕГО лишнего не пиши — только саму Markdown-таблицу (без пояснений и комментариев)."
    )

    user_text = (
        "Преврати таблицу на изображении в Markdown-таблицу. "
        f"Ограничение: не более {max_cols} столбцов. Не выдумывай значения."
    )

    # 1) Markdown
    md = ""
    try:
        md = chat_with_gpt(
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": [{"type": "text", "text": user_text}] + image_inputs},
            ],
            temperature=0.0,
            max_tokens=900,
        ) or ""
        md = (md or "").strip()
    except Exception:
        md = ""

    # 2) CSV (преобразуем в отдельном шаге, чтобы не ломать Markdown)
    csv_prompt = (
        "Возьми таблицу, которую ты только что сгенерировал в Markdown, "
        "и верни ЕЁ ЖЕ как CSV (разделитель — запятая). "
        "Не добавляй пояснений, возвращай только CSV."
    )
    csv_out = ""
    if md:
        try:
            csv_out = chat_with_gpt(
                [
                    {"role": "system", "content": "Ты конвертор таблиц Markdown → CSV. Выводи только CSV, без пояснений."},
                    {"role": "assistant", "content": md},
                    {"role": "user", "content": csv_prompt},
                ],
                temperature=0.0,
                max_tokens=900,
            ) or ""
            csv_out = (csv_out or "").strip()
        except Exception:
            csv_out = ""

    return {"markdown": md, "csv": csv_out}


def _local_ocr_csv(image_paths: List[str]) -> str:
    """
    Очень грубый локальный фолбэк: OCR каждого изображения → строки текста,
    затем «псевдо-таблица» CSV по эвристике разделителей.
    Работает только если есть pytesseract + Pillow.
    """
    if not _HAS_LOCAL_OCR or not image_paths:
        return ""

    rows: List[List[str]] = []
    for p in image_paths:
        try:
            img = Image.open(p)
            text = pytesseract.image_to_string(img, lang="rus+eng")
        except Exception:
            continue
        # Делим по строкам и пытаемся найти «табличные» разделители
        for line in (text or "").splitlines():
            line = line.strip()
            if not line:
                continue
            # простая эвристика: '|' или несколько пробелов как разделитель
            if "|" in line:
                parts = [c.strip() for c in line.split("|")]
            else:
                parts = [c.strip() for c in re.split(r"\s{2,}", line)]
            rows.append(parts if parts else [line])

    # Нормируем ширину
    width = max((len(r) for r in rows), default=0)
    if width == 0:
        return ""
    norm = []
    for r in rows:
        if len(r) < width:
            r = r + [""] * (width - len(r))
        norm.append(r)

    # Собираем CSV
    out_lines: List[str] = []
    for r in norm:
        esc = []
        for cell in r:
            cell = cell.replace('"', '""')
            if ("," in cell) or ('"' in cell):
                esc.append(f'"{cell}"')
            else:
                esc.append(cell)
        out_lines.append(",".join(esc))
    return "\n".join(out_lines)


# ============================== ПУБЛИЧНЫЕ API ==============================

def ocr_table_section(owner_id: int,
                      doc_id: int,
                      section_path: str,
                      *,
                      lang: str = "ru",
                      prefer_markdown: bool = True) -> Dict[str, Any]:
    """
    OCR/Vision для таблицы, известной по section_path.
    Возвращает:
      {
        "display": "Таблица 2.3 — Название",
        "where": {"page": <int|None>, "section_path": <str>},
        "images": [<paths>],
        "markdown": "<md-table>|",
        "csv": "a,b,c\n...",
        "note": "..."  # краткая пометка о качестве/фолбэке
      }
    """
    # 1) Найдём любой чанк этой секции (чтобы узнать страницу и attrs)
    con = get_conn()
    try:
        has_ext = _has_columns(con, "chunks", ["attrs"])
        row = _fetchone(
            con,
            "SELECT page, section_path, text, " + ("attrs" if has_ext else "NULL AS attrs") +
            " FROM chunks WHERE owner_id=? AND doc_id=? AND section_path=? ORDER BY id ASC LIMIT 1",
            (owner_id, doc_id, section_path),
        )
    finally:
        con.close()

    if not row:
        return {
            "display": _last_segment(section_path) or "Таблица",
            "where": {"page": None, "section_path": section_path},
            "images": [],
            "markdown": "",
            "csv": "",
            "note": "Не удалось найти секцию в индексе.",
        }

    # 2) Вытянем картинки
    images = []
    images += _extract_images_from_attrs(row.get("attrs"))
    images_extra = _collect_images_for_section(owner_id, doc_id, section_path)
    for p in images_extra:
        if p not in images:
            images.append(p)

    # 3) Сформируем display (по подписи секции или тексту)
    tail = _last_segment(section_path)
    n1, t1 = _parse_table_title(tail)
    if not (n1 or t1):
        n2, t2 = _parse_table_title(row.get("text") or "")
    else:
        n2, t2 = (None, None)
    num = n1 or n2
    title = t1 or t2
    display = f"Таблица {num}" + (f" — {title}" if (num and title) else "")
    if not num and not title:
        display = tail or "Таблица"

    # 4) Vision или локальный OCR
    note = ""
    md, csv_text = "", ""

    if images:
        vis = _vision_table_markdown_for_images(images, lang=lang)
        md = vis.get("markdown") or ""
        csv_text = vis.get("csv") or ""
        if not (md or csv_text):
            # Пробуем локальный OCR как фолбэк
            csv_text = _local_ocr_csv(images)
            if csv_text:
                note = "Использован локальный OCR (качество может отличаться)."
            else:
                note = "Не удалось восстановить таблицу из изображения."
    else:
        note = "Картинки не извлечены из секции; OCR невозможен."

    # 5) Если просят Markdown, а его нет — попробуем из CSV восстановить простую Markdown-таблицу
    if prefer_markdown and not md and csv_text:
        try:
            # Попросим модель конвертировать CSV → Markdown
            md = chat_with_gpt(
                [
                    {"role": "system", "content": "Ты конвертор CSV → Markdown-таблица. Верни только Markdown-таблицу."},
                    {"role": "user", "content": csv_text},
                ],
                temperature=0.0,
                max_tokens=600,
            ) or ""
            md = (md or "").strip()
        except Exception:
            md = ""

    return {
        "display": display,
        "where": {"page": row.get("page"), "section_path": section_path},
        "images": images,
        "markdown": md,
        "csv": csv_text,
        "note": note,
    }


def ocr_tables_by_numbers(owner_id: int,
                          doc_id: int,
                          numbers: List[str],
                          *,
                          lang: str = "ru",
                          prefer_markdown: bool = True) -> List[Dict[str, Any]]:
    """
    OCR/Vision для набора таблиц по их номерам ['2.1', '3', 'A.1' ...].
    Возвращает список карточек:
      {
        "num": "2.1",
        "display": "Таблица 2.1 — ...",
        "where": {...},
        "images": [...],
        "markdown": "...",
        "csv": "...",
        "note": "..."
      }
    """
    cards: List[Dict[str, Any]] = []
    if not numbers:
        return cards

    # нормализуем номера
    want = []
    seen = set()
    for n in numbers:
        nn = _normalize_num(n)
        if nn and nn not in seen:
            seen.add(nn)
            want.append(nn)

    for num in want:
        row = _find_table_chunk_by_number(owner_id, doc_id, num)
        if not row:
            cards.append({
                "num": num,
                "display": f"Таблица {num}",
                "where": {"page": None, "section_path": ""},
                "images": [],
                "markdown": "",
                "csv": "",
                "note": "Таблица с таким номером не найдена.",
            })
            continue

        sec = row.get("section_path") or ""
        tail = _last_segment(sec)
        n1, t1 = _parse_table_title(tail)
        display = f"Таблица {n1 or num}" + (f" — {t1}" if t1 else "")

        images = []
        images += _extract_images_from_attrs(row.get("attrs"))
        for p in _collect_images_for_section(owner_id, doc_id, sec):
            if p not in images:
                images.append(p)

        note = ""
        md, csv_text = "", ""
        if images:
            vis = _vision_table_markdown_for_images(images, lang=lang)
            md = vis.get("markdown") or ""
            csv_text = vis.get("csv") or ""
            if not (md or csv_text):
                csv_text = _local_ocr_csv(images)
                if csv_text:
                    note = "Использован локальный OCR (качество может отличаться)."
                else:
                    note = "Не удалось восстановить таблицу из изображения."
        else:
            note = "Картинки не извлечены из секции; OCR невозможен."

        if prefer_markdown and not md and csv_text:
            try:
                md = chat_with_gpt(
                    [
                        {"role": "system", "content": "Ты конвертор CSV → Markdown-таблица. Верни только Markdown-таблицу."},
                        {"role": "user", "content": csv_text},
                    ],
                    temperature=0.0,
                    max_tokens=600,
                ) or ""
                md = (md or "").strip()
            except Exception:
                md = ""

        cards.append({
            "num": num,
            "display": display,
            "where": {"page": row.get("page"), "section_path": sec},
            "images": images,
            "markdown": md,
            "csv": csv_text,
            "note": note,
        })

    return cards


def vision_table_markdown(image_paths: List[str], *, lang: str = "ru", max_cols: int = 14) -> str:
    """
    Упрощённая обёртка: вернуть только Markdown-таблицу из списка путей изображений.
    Полезно для единичных вызовов.
    """
    res = _vision_table_markdown_for_images(image_paths, lang=lang, max_cols=max_cols)
    return res.get("markdown", "") or ""


# ============================== ЭВРИСТИКА: «КАРТИНКА С ТАБЛИЦЕЙ» ==============================

# Быстрые подсказки — слова, указывающие на табличность
_TABLE_HINT_WORDS_RE = re.compile(r"(?i)\b(табл(?:ица)?|table|табл\.)\b")
_TABLE_CAPTION_RE = re.compile(
    r"(?i)\bтабл(?:ица)?\.?\s*(?:№\s*)?(?:[A-Za-zА-Яа-я]\.?[\s-]*\d+(?:[.,]\d+)*|\d+(?:[.,]\d+)*)"
)

def _looks_like_table_filename(path: str) -> bool:
    """Эвристика по имени файла картинки."""
    s = Path(path).name.lower()
    return any(k in s for k in ("table", "таблица", "tabl", "tab_"))

def is_table_image_section(sec: Dict[str, Any]) -> bool:
    """
    Эвристика для figure-секций: пометить картинку как «табличную» (скан таблицы).
    Используется в ingest_orchestrator для последующего OCR.

    Триггеры (достаточно одного):
      1) В attrs.caption_tail/title/section_path встречается «Таблица …» или слова table/таблица/табл.
      2) section_path/текст содержит явную подпись «Таблица N».
      3) Среди путей изображений есть имена с «table/таблица».
    """
    if not isinstance(sec, dict):
        return False

    et = (sec.get("element_type") or "").lower()
    txt = (sec.get("text") or "").strip()
    sp = (sec.get("section_path") or "").strip()
    title = (sec.get("title") or "").strip()
    attrs = sec.get("attrs") or {}

    # Должно быть изображение/figure-подобное
    is_figure_like = (et == "figure") or txt.startswith("[Рисунок]") or re.search(r"(?i)\bris[.\s]|figure", sp or "")
    if not is_figure_like:
        return False

    # 1) Явные текстовые/путевые маркеры
    for candidate in (sp, title, txt, attrs.get("caption_tail") or "", attrs.get("title") or ""):
        if not candidate:
            continue
        if _TABLE_HINT_WORDS_RE.search(candidate) or _TABLE_CAPTION_RE.search(candidate):
            return True

    # 2) Имена файлов изображений
    imgs = attrs.get("images") or []
    if any(_looks_like_table_filename(p) for p in imgs if p):
        return True

    return False


__all__ = [
    "ocr_table_section",
    "ocr_tables_by_numbers",
    "vision_table_markdown",
    "is_table_image_section",
]
