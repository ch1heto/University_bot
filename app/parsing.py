# app/parsing.py
from docx import Document as Docx
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.text.run import Run
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
import pdfplumber
from pathlib import Path
import subprocess
import tempfile
import platform
import shutil
import os
import re
from typing import List, Dict, Any, Optional, Tuple

# мягкая зависимость для извлечения изображений из PDF
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None  # без него делаем только текстовые подписи

# локальные утилиты/конфиг
try:
    from .config import Cfg
    DEFAULT_UPLOAD_DIR = Cfg.UPLOAD_DIR
    CFG_FIG_WINDOW = int(getattr(Cfg, "FIG_NEIGHBOR_WINDOW", 4))
    CFG_PDF_EXTRACT = bool(getattr(Cfg, "PDF_EXTRACT_IMAGES", True))
except Exception:
    DEFAULT_UPLOAD_DIR = "./uploads"
    CFG_FIG_WINDOW = 4
    CFG_PDF_EXTRACT = True

try:
    from .utils import save_bytes, safe_filename, sha256_bytes
except Exception:
    # минимальные заглушки (на всякий случай)
    import hashlib
    def sha256_bytes(data: bytes) -> str:
        h = hashlib.sha256(); h.update(data or b""); return h.hexdigest()
    def save_bytes(data: bytes, filename: str, dirpath: str = "./uploads") -> str:
        p = Path(dirpath); p.mkdir(parents=True, exist_ok=True)
        fp = p / filename; fp.write_bytes(data); return str(fp)
    def safe_filename(name: str, max_len: int = 180) -> str:
        name = re.sub(r"[^0-9A-Za-zА-Яа-яЁё _.\-]+", "_", name or "file").strip(" .")
        return name[:max_len]

# ----------------------------- helpers (common) -----------------------------

def _clean(s: str) -> str:
    return re.sub(r"[ \t]+", " ", (s or "").replace("\xa0", " ")).strip()

def _len_pt(x) -> Optional[float]:
    """Length -> pt (float) | None."""
    try:
        return float(getattr(x, "pt", None)) if x is not None else None
    except Exception:
        return None

def _align_name(a) -> Optional[str]:
    if a is None:
        return None
    try:
        if a == WD_ALIGN_PARAGRAPH.LEFT:
            return "left"
        if a == WD_ALIGN_PARAGRAPH.CENTER:
            return "center"
        if a == WD_ALIGN_PARAGRAPH.RIGHT:
            return "right"
        if a == WD_ALIGN_PARAGRAPH.JUSTIFY:
            return "justify"
    except Exception:
        pass
    return str(a)

def _extract_numbers(text: str) -> List[str]:
    # ловим целые/десятичные/проценты
    return re.findall(r"\b\d+(?:[.,]\d+)?%?\b", text or "")

# ----------------------------- DOCX block iterator -----------------------------

def _iter_block_items(doc: Docx):
    """Итерируем абзацы и таблицы в порядке следования в документе."""
    body = doc._element.body
    for child in body.iterchildren():
        if child.tag.endswith("p"):
            yield Paragraph(child, doc)
        elif child.tag.endswith("tbl"):
            yield Table(child, doc)

# ----------------------------- lists / numbering -----------------------------

def _paragraph_list_info(p: Paragraph) -> dict:
    """Пытаемся извлечь инфо о списке (numId/ilvl)."""
    info = {"is_list": False, "level": None, "num_id": None}
    try:
        pPr = p._p.pPr
        if pPr is not None and pPr.numPr is not None:
            numPr = pPr.numPr
            ilvl = numPr.ilvl
            numId = numPr.numId
            if numId is not None:
                info["is_list"] = True
                info["num_id"] = int(numId.val)
                info["level"] = int(ilvl.val) if ilvl is not None else 0
    except Exception:
        pass
    return info

# ----------------------------- inline images flag -----------------------------

NS = {
    "a":   "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r":   "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
    "wp":  "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
}

def _paragraph_has_image(p: Paragraph) -> bool:
    """Грубая эвристика: есть ли в абзаце встроенная картинка."""
    try:
        pics = p._p.xpath(".//pic:pic", namespaces=NS)
        return bool(pics)
    except Exception:
        return False

def _extract_paragraph_images(doc: Docx, p: Paragraph, uploads_dir: str = DEFAULT_UPLOAD_DIR) -> List[str]:
    """
    Возвращает список абсолютных путей к сохранённым картинкам, найденным в абзаце p (inline/anchored).
    """
    paths: List[str] = []
    try:
        blips = p._p.xpath(".//a:blip", namespaces=NS)
        for b in blips:
            rid = b.get(qn("r:embed"))
            if not rid:
                continue
            part = doc.part.related_parts.get(rid)
            if not part:
                continue
            data = getattr(part, "blob", None)
            if not data:
                continue
            h = sha256_bytes(data)
            # расширение из content_type
            ext = ".png"
            ct = (getattr(part, "content_type", "") or "").lower()
            if "jpeg" in ct or "jpg" in ct:
                ext = ".jpg"
            elif "png" in ct:
                ext = ".png"
            elif "gif" in ct:
                ext = ".gif"
            elif "bmp" in ct:
                ext = ".bmp"
            elif "tiff" in ct or "tif" in ct:
                ext = ".tiff"
            name = safe_filename(f"docx_fig_{h[:12]}{ext}")
            fp = save_bytes(data, name, uploads_dir)
            if fp not in paths:
                paths.append(fp)
    except Exception:
        pass
    return paths

def _collect_neighbor_images(doc: Docx, block: Paragraph, window: int = CFG_FIG_WINDOW, uploads_dir: str = DEFAULT_UPLOAD_DIR) -> List[str]:
    """
    Смотрим соседние блоки (±window) и собираем картинки.
    Это покрывает случаи: картинка сверху, подпись отдельно; или наоборот.
    """
    try:
        body = doc._element.body
        children = list(body.iterchildren())
        # находим индекс текущего абзаца
        idx = None
        for i, el in enumerate(children):
            if getattr(block, "_p", None) is el:
                idx = i
                break
        if idx is None:
            return []
        out: List[str] = []
        lo = max(0, idx - window)
        hi = min(len(children) - 1, idx + window)
        for j in range(lo, hi + 1):
            el = children[j]
            if el.tag.endswith("p"):
                p2 = Paragraph(el, doc)
                out += _extract_paragraph_images(doc, p2, uploads_dir)
        # дедуп
        uniq: List[str] = []
        for pth in out:
            if pth not in uniq:
                uniq.append(pth)
        return uniq
    except Exception:
        return []

# ----------------------------- runs formatting -----------------------------

def _runs_style_summary(runs: List[Run]) -> dict:
    """Сводка по run'ам: встречались ли bold/italic/underline, список шрифтов/размеров."""
    any_bold = any(bool(r.bold) for r in runs if r is not None)
    any_italic = any(bool(r.italic) for r in runs if r is not None)
    any_ul = any(bool(r.underline) for r in runs if r is not None)
    fonts, sizes = [], []
    for r in runs:
        try:
            if r.font and r.font.name:
                fonts.append(r.font.name)
        except Exception:
            pass
        try:
            sz = _len_pt(getattr(r.font, "size", None))
            if sz:
                sizes.append(sz)
        except Exception:
            pass
    fonts = list(dict.fromkeys([f for f in fonts if f]))  # uniq, keep order
    sizes = sorted(set(round(s, 2) for s in sizes))
    return {"bold": any_bold, "italic": any_italic, "underline": any_ul,
            "fonts": fonts, "font_sizes_pt": sizes}

def _paragraph_attrs(p: Paragraph) -> dict:
    pf = p.paragraph_format
    attrs = {
        "style": (p.style.name if p.style else None),
        "alignment": _align_name(getattr(pf, "alignment", None)),
        "left_indent_pt": _len_pt(getattr(pf, "left_indent", None)),
        "first_line_indent_pt": _len_pt(getattr(pf, "first_line_indent", None)),
        "space_before_pt": _len_pt(getattr(pf, "space_before", None)),
        "space_after_pt": _len_pt(getattr(pf, "space_after", None)),
        "line_spacing": getattr(pf, "line_spacing", None),
        "has_inline_image": _paragraph_has_image(p),
    }
    attrs.update(_paragraph_list_info(p))
    attrs.update(_runs_style_summary(p.runs))
    return attrs

# ----------------------------- tables -----------------------------

def _table_row_strings(tbl: Table) -> List[str]:
    """Возвращает строки таблицы: одна строка = ряд, ячейки через ' | ' (с учётом merged)."""
    lines: List[str] = []
    try:
        for row in tbl.rows:
            cells = []
            for c in row.cells:
                cells.append(_clean(c.text))
            # убираем повторы смежных ячеек (merge-эффект)
            dedup = []
            for i, x in enumerate(cells):
                if i == 0 or x != cells[i - 1]:
                    dedup.append(x)
            line = " | ".join([x for x in dedup if x]).strip(" |")
            if line:
                lines.append(line)
    except Exception:
        pass
    return lines

def _table_to_text(tbl: Table) -> str:
    lines = _table_row_strings(tbl)
    return "\n".join(lines)

def _table_header_preview(tbl: Table, limit: int = 160) -> Optional[str]:
    """Краткое описание из первой непустой строки таблицы."""
    lines = _table_row_strings(tbl)
    if not lines:
        return None
    hdr = lines[0].strip()
    if not hdr:
        return None
    if len(hdr) > limit:
        hdr = hdr[:limit - 1].rstrip() + "…"
    return hdr

def _table_attrs(tbl: Table) -> dict:
    try:
        n_rows = len(tbl.rows)
    except Exception:
        n_rows = None
    try:
        n_cols = len(tbl.columns)
    except Exception:
        n_cols = None
    return {"n_rows": n_rows, "n_cols": n_cols, "header_preview": _table_header_preview(tbl)}

# ----------------------------- caption helpers -----------------------------

# «Таблица 2.1», «Табл. 5», «Таблица А.1», «Таблица П1.2», «Таблица № 7», «Таблица 1. Название»
CAPTION_RE_TABLE = re.compile(
    r"""^\s*
        (?:табл(?:ица)?|table)\s*\.?\s*
        (?:№\s*)?
        (
          (?:[A-Za-zА-Яа-я]\.?[\s-]*\d+(?:[.,]\d+)*)   # А.1 / A.1 / П1.2
          |
          (?:\d+(?:[.,]\d+)*)                          # 2.1 / 3
        )
        (?:\s*(?:[.\-—:\u2013\u2014])\s*(.*))?         # ← допускаем точку-тире-двоеточие
        \s*$""",
    re.IGNORECASE | re.VERBOSE
)

# «Рис. 1.», «Рисунок 2,1 — ...», «Fig. 3: ...», «Figure 4 ...», «Рис. А.1 ...»
CAPTION_RE_FIG = re.compile(
    r"""^\s*
        (?:
            рис(?:\.|унок)?      # 'Рис.' или 'Рисунок'
            | figure             # 'Figure'
            | fig\.?             # 'Fig.' или 'Fig'
        )
        \s*
        (?:№\s*)?                # опциональное '№'
        (
          (?:[A-Za-zА-Яа-я]\.?[\s-]*\d+(?:[.,]\d+)*)   # А.1 / П1.2
          |
          (?:\d+(?:[.,]\d+)*)                          # 1 / 2.1 / 3,2 / 2.1.3
        )
        (?:\s*(?:[.\-—:\u2013\u2014])\s*(.*))?         # ← разрешаем и точку/двоеточие/тире
        \s*$""",
    re.IGNORECASE | re.VERBOSE
)

def _classify_caption(text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Возвращает (kind, number_label, tail) или (None, None, None).
    number_label в формате '2.1' / 'А.1' / 'П1.2', tail — подпись (может быть пустой).
    """
    t = (text or "").strip()
    m = CAPTION_RE_TABLE.match(t)
    if m:
        num = (m.group(1) or "").replace(",", ".").replace(" ", "")
        tail = (m.group(2) or "").strip()
        return "table", num, tail
    m = CAPTION_RE_FIG.match(t)
    if m:
        num = (m.group(1) or "").replace(",", ".").replace(" ", "")
        tail = (m.group(2) or "").strip()
        return "figure", num, tail
    return None, None, None

def _compose_table_title(num: Optional[str], tail: Optional[str]) -> str:
    if num and tail:
        return f"Таблица {num} — {tail}"
    if num:
        return f"Таблица {num}"
    if tail:
        return tail  # без слова «Таблица», если номера нет
    return "Таблица"

def _compose_figure_title(num: Optional[str], tail: Optional[str]) -> str:
    if num and tail:
        return f"Рисунок {num} — {tail}"
    if num:
        return f"Рисунок {num}"
    if tail:
        return tail  # без слова «Рисунок», если номера нет
    return "Рисунок"

def _is_title_candidate(text: str, attrs: dict) -> bool:
    """Эвристика: короткий центрированный абзац — кандидат в название таблицы/рисунка."""
    t = (text or "").strip()
    if not t:
        return False
    if CAPTION_RE_TABLE.match(t) or CAPTION_RE_FIG.match(t):
        return False  # это не «название», это сам номер
    if len(t) < 3 or len(t) > 200:
        return False
    al = (attrs or {}).get("alignment")
    if al not in {"center", "right", "left"}:
        return False
    # не списочный элемент
    if (attrs or {}).get("is_list"):
        return False
    # слишком «цифровая» строка — не подпись
    digits = len(re.findall(r"\d", t))
    if digits > max(8, len(t) // 3):
        return False
    return True

# ----------------------------- sources / appendix helpers -----------------------------

SOURCES_TITLE_RE = re.compile(
    r"""(?ix)
    (
      \bсписок\s+(?:использ\w+\s+)?(?:литератур(?:а|ы)\b|источни\w*\b)
      |
      \bбиблиограф\w*\b
      |
      \breferences?\b
      |
      \bbibliograph\w*\b
      |
      \bисточни\w+\b
      |
      \bлитератур(?:а|ы)\b
    )
    """
)

APPENDIX_TITLE_RE = re.compile(
    r"\b(приложени[ея]|appendix)\b", re.IGNORECASE
)

# разрешаем «12. », «12) », «12 - », «12 — », «[12] », И ТЕПЕРЬ ТАКЖЕ «12 <пробел>»
REF_LINE_RE = re.compile(r"^\s*(?:\[(\d+)\]|(\d+)(?:[\.\)\-–—:]\s*|\s+))\s*(.+)$")

def _is_sources_title(s: str) -> bool:
    return bool(SOURCES_TITLE_RE.search(s or ""))

def _is_appendix_title(s: str) -> bool:
    return bool(APPENDIX_TITLE_RE.search(s or ""))

def _parse_reference_line(s: str) -> Tuple[Optional[int], str]:
    """
    Из строки источника выделяет номер (если есть) и «хвост».
    Поддерживает формы: "[12] …", "12. …", "12) …", "12 — …", "12 …".
    """
    t = _clean(s)
    m = REF_LINE_RE.match(t)
    if not m:
        return None, t
    num = m.group(1) or m.group(2)
    try:
        idx = int(num)
    except Exception:
        idx = None
    return idx, (m.group(3) or "").strip()

# ----------------------------- section-path helpers (НОВОЕ: нумерация) -----------------------------

def _hpath(ids: List[str], titles: List[str]) -> str:
    parts = []
    for i, t in enumerate(titles):
        if not t:
            continue
        sid = ids[i] if i < len(ids) else None
        parts.append((f"{sid} {t}".strip() if sid else t).strip())
    return " / ".join(parts)

def _make_section_id(counters: List[int], upto: Optional[int] = None) -> str:
    if upto is None or upto > len(counters):
        upto = len(counters)
    return ".".join(str(x) for x in counters[:upto] if x > 0)

# ----------------------------- DOCX main -----------------------------

def parse_docx(path: str) -> List[Dict[str, Any]]:
    """
    Парсим .docx с поддержкой заголовков, таблиц, базового форматирования и
    аккуратной обработкой подписей к таблицам/рисункам.
    Также извлекаем изображения и привязываем к секциям element_type='figure'.
    ДОБАВЛЕНО: нумерация разделов (ID вида 1, 1.1, 1.2...) и передача scope_id/path.
    """
    doc = Docx(path)

    sections: List[Dict[str, Any]] = []
    buf: List[str] = []
    last_para_attrs: Optional[dict] = None

    # текущий заголовок и стек заголовков
    cur_title, cur_level = "Документ", 0

    # НОВОЕ: счётчики для нумерации и стек для id/заголовков
    outline_counters: List[int] = []        # [1, 2, ...] по уровням
    heading_stack_titles: List[str] = []    # названия на уровнях
    heading_stack_ids: List[str] = []       # текстовые ID на уровнях ("1", "1.2", ...)

    def _current_scope_id() -> str:
        return _make_section_id(outline_counters)

    def _current_scope_path() -> str:
        return _hpath(heading_stack_ids, heading_stack_titles)

    def _enter_heading(level: int, title: str) -> str:
        """Обновляет счётчики и стеки под новый заголовок уровня level, возвращает section_id."""
        nonlocal outline_counters, heading_stack_titles, heading_stack_ids, cur_title, cur_level
        if level < 1:
            level = 1
        # расширяем/усекаем счётчики до нужного уровня
        while len(outline_counters) < level:
            outline_counters.append(0)
        outline_counters = outline_counters[:level]
        # инкремент уровня и сброс глубже
        outline_counters[level - 1] += 1

        # обновляем стеки названий и id
        if len(heading_stack_titles) < level:
            heading_stack_titles += [""] * (level - len(heading_stack_titles))
        if len(heading_stack_ids) < level:
            heading_stack_ids += [""] * (level - len(heading_stack_ids))

        heading_stack_titles = heading_stack_titles[:level]
        heading_stack_ids = heading_stack_ids[:level]

        sid = _make_section_id(outline_counters, upto=level)
        heading_stack_titles[level - 1] = (title or "Без названия")
        heading_stack_ids[level - 1] = sid

        cur_title = title or "Без названия"
        cur_level = level
        return sid

    # состояние: находимся ли в разделе «Источники»
    in_sources = False

    # состояние по подписям
    pending_tbl_num: Optional[str] = None          # например '3.2', 'А.1'
    pending_tbl_tail: Optional[str] = None         # текст подписи (если есть)
    awaiting_tail: bool = False                    # видели «Таблица N», ждём подпись строкой ниже
    last_title_candidate: Optional[str] = None     # запомненный короткий центрированный абзац
    last_title_candidate_age: int = 999            # «возраст» кандидата (в блоках)

    table_counter = 0
    figure_counter = 0

    def flush_text_section():
        nonlocal buf, cur_title, cur_level, last_para_attrs
        if buf:
            text = "\n".join(buf)
            sections.append({
                "title": cur_title,
                "level": cur_level,
                "text": text,
                "page": None,
                "section_path": _current_scope_path() or cur_title,
                "element_type": "paragraph",
                "attrs": {
                    "numbers": _extract_numbers(text),
                    "section_scope_id": _current_scope_id(),
                    "section_scope_path": _current_scope_path(),
                    **(last_para_attrs or {})
                }
            })
            buf.clear()
            last_para_attrs = None

    def consume_title_candidate() -> Optional[str]:
        """Забираем и сбрасываем сохранённый заголовок-кандидат."""
        nonlocal last_title_candidate, last_title_candidate_age
        t = last_title_candidate
        last_title_candidate = None
        last_title_candidate_age = 999
        return t

    def consider_flush_stale_candidate():
        """Если кандидат «залежался», возвращаем его в обычный текст."""
        nonlocal last_title_candidate, last_title_candidate_age, buf
        if last_title_candidate and last_title_candidate_age > 3:
            buf.append(last_title_candidate)
            last_title_candidate = None
            last_title_candidate_age = 999

    for block in _iter_block_items(doc):
        # возраст кандидатного заголовка увеличиваем каждый блок
        last_title_candidate_age = last_title_candidate_age + 1

        if isinstance(block, Paragraph):
            p_text = _clean(block.text)
            p_style_name = (block.style.name if block.style else "") or ""
            p_style = p_style_name.lower()

            # Заголовок раздела (поддержка RU/EN локалей)
            if re.match(r"^(heading|заголовок)\s*\d+", p_style):
                consider_flush_stale_candidate()
                flush_text_section()
                title = p_text or "Без названия"
                try:
                    # извлечь номер уровня из имени стиля
                    m = re.search(r"(\d+)", p_style)
                    lvl = max(1, int(m.group(1))) if m else 1
                except Exception:
                    lvl = 1

                sec_id = _enter_heading(lvl, title)

                in_sources = _is_sources_title(title)
                if _is_appendix_title(title):
                    in_sources = False  # раздел «Приложения» заканчивает блок источников

                sections.append({
                    "title": title,
                    "level": lvl,
                    "text": "",
                    "page": None,
                    "section_path": _current_scope_path(),
                    "element_type": "heading",
                    "attrs": {"style": p_style_name, "section_id": sec_id}
                })
                continue

            # Подзаголовок «Приложение …» без стиля Heading
            attrs_here = _paragraph_attrs(block)
            if _is_appendix_title(p_text) and _is_title_candidate(p_text, attrs_here):
                consider_flush_stale_candidate()
                flush_text_section()
                title = p_text or "Приложение"
                lvl = max(1, cur_level + 1)
                sec_id = _enter_heading(lvl, title)

                in_sources = False  # «Приложение» завершает список источников

                sections.append({
                    "title": title,
                    "level": lvl,
                    "text": "",
                    "page": None,
                    "section_path": _current_scope_path(),
                    "element_type": "heading",
                    "attrs": {"style": "pseudo-heading-appendix", "section_id": sec_id, **attrs_here}
                })
                continue

            # Заголовок-кандидат (короткий центрированный/не списочный абзац) — как подпись
            if _is_title_candidate(p_text, attrs_here) and not _is_sources_title(p_text):
                last_title_candidate = p_text
                last_title_candidate_age = 0
                # не пишем в обычный текст — дождёмся таблицу/рисунок
                continue

            # Раздел «Источники»
            if in_sources and p_text:
                consider_flush_stale_candidate()
                flush_text_section()
                idx, tail = _parse_reference_line(p_text)
                attrs = _paragraph_attrs(block)
                attrs.update({
                    "numbers": _extract_numbers(tail),
                    "section_scope_id": _current_scope_id(),
                    "section_scope_path": _current_scope_path()
                })
                if idx is not None:
                    attrs["ref_index"] = idx
                ref_title = f"Источник {idx}" if idx is not None else "Источник"
                sections.append({
                    "title": ref_title,
                    "level": max(1, cur_level + 1),
                    "text": tail,
                    "page": None,
                    "section_path": _hpath(heading_stack_ids, heading_stack_titles + [ref_title]),
                    "element_type": "reference",
                    "attrs": attrs,
                })
                continue

            # Попытка распознать подписи к таблицам/рисункам
            kind, num, tail = _classify_caption(p_text or "")
            if kind == "table":
                # встретили строку «Таблица N [—/./: подпись]»
                pending_tbl_num = num
                if tail:
                    pending_tbl_tail = tail
                    awaiting_tail = False
                else:
                    # ждём подпись отдельной строкой ниже
                    pending_tbl_tail = None
                    awaiting_tail = True
                # если прямо перед этим был центрированный кандидат — используем его как хвост
                if awaiting_tail and last_title_candidate and last_title_candidate_age <= 1:
                    pending_tbl_tail = consume_title_candidate()
                    awaiting_tail = False
                # не добавляем этот абзац в обычный текст
                continue
            elif kind == "figure":
                # Если хвоста нет, а сразу перед этим был центрированный кандидат — используем его как хвост подписи рисунка.
                if (not tail) and last_title_candidate and last_title_candidate_age <= 1:
                    tail = consume_title_candidate()

                consider_flush_stale_candidate()
                flush_text_section()
                figure_counter += 1
                fig_title = _compose_figure_title(num, tail)
                attrs_here_fig = _paragraph_attrs(block)

                # извлечём изображения из текущего и соседних абзацев (±CFG_FIG_WINDOW)
                imgs: List[str] = []
                imgs += _extract_paragraph_images(doc, block, DEFAULT_UPLOAD_DIR)  # из текущего
                neigh = _collect_neighbor_images(doc, block, window=CFG_FIG_WINDOW, uploads_dir=DEFAULT_UPLOAD_DIR)
                for pth in neigh:
                    if pth not in imgs:
                        imgs.append(pth)

                attrs_here_fig["caption_num"] = num
                attrs_here_fig["caption_tail"] = tail
                attrs_here_fig["label"] = num
                attrs_here_fig["title"] = tail
                attrs_here_fig["numbers"] = _extract_numbers(p_text or "")
                if imgs:
                    attrs_here_fig["images"] = imgs

                # добавляем привязку к текущему разделу и стабильный якорь
                attrs_here_fig["section_scope"] = _current_scope_path()
                attrs_here_fig["section_scope_id"] = _current_scope_id()
                attrs_here_fig["anchor"] = f"fig-{(num or figure_counter)}"

                sections.append({
                    "title": fig_title,
                    "level": max(1, cur_level + 1),
                    "text": p_text or "",
                    "page": None,
                    "section_path": _hpath(heading_stack_ids, heading_stack_titles + [fig_title]),
                    "element_type": "figure",
                    "attrs": attrs_here_fig
                })

                # сброс «табличного» состояния
                pending_tbl_num = None
                pending_tbl_tail = None
                awaiting_tail = False
                continue

            # Кандидат на заголовок «Список литературы/Источники» без стиля Heading
            if _is_sources_title(p_text) and _is_title_candidate(p_text, attrs_here):
                consider_flush_stale_candidate()
                flush_text_section()
                title = p_text or "Список литературы"
                lvl = max(1, cur_level + 1)
                sec_id = _enter_heading(lvl, title)

                in_sources = True  # включаем режим разбора источников

                sections.append({
                    "title": title,
                    "level": lvl,
                    "text": "",
                    "page": None,
                    "section_path": _current_scope_path(),
                    "element_type": "heading",
                    "attrs": {"style": "pseudo-heading-sources", "section_id": sec_id, **attrs_here}
                })
                continue

            # Обычный текст — копим в буфер,
            # а «залежавшийся» кандидат вернём в текст
            consider_flush_stale_candidate()
            if p_text:
                last_para_attrs = _paragraph_attrs(block)
                buf.append(p_text)
            continue

        # Таблица
        if isinstance(block, Table):
            consider_flush_stale_candidate()
            flush_text_section()
            table_counter += 1

            # 1) составляем подпись таблицы по приоритетам
            caption_source = "none"
            num_final: Optional[str] = None
            tail_final: Optional[str] = None

            if pending_tbl_num:
                num_final = pending_tbl_num
                if pending_tbl_tail:
                    tail_final = pending_tbl_tail
                    caption_source = "single_line" if not awaiting_tail else "two_lines_after"
                elif last_title_candidate and last_title_candidate_age <= 1:
                    tail_final = consume_title_candidate()
                    caption_source = "two_lines_after"
            elif last_title_candidate and last_title_candidate_age <= 1:
                # заголовок перед таблицей, номера нет
                tail_final = consume_title_candidate()
                caption_source = "two_lines_before"

            # 2) фолбэк: если названия всё ещё нет — взять первую строку таблицы
            t_text = _table_to_text(block) or "(пустая таблица)"
            attrs_tbl = _table_attrs(block) | {"numbers": _extract_numbers(t_text)}

            if not tail_final and attrs_tbl.get("header_preview"):
                tail_final = attrs_tbl["header_preview"]
                if caption_source == "none":
                    caption_source = "header_row"

            # 3) Компоновка «человеческого» заголовка и атрибутов
            t_title = _compose_table_title(num_final, tail_final)
            attrs_tbl["caption_num"] = num_final
            attrs_tbl["caption_tail"] = tail_final
            attrs_tbl["label"] = num_final
            attrs_tbl["title"] = tail_final
            attrs_tbl["caption_source"] = caption_source
            # привязка к разделу + стабильный якорь
            attrs_tbl["section_scope"] = _current_scope_path()
            attrs_tbl["section_scope_id"] = _current_scope_id()
            attrs_tbl["anchor"] = f"tbl-{(num_final or table_counter)}"

            sections.append({
                "title": t_title or f"Таблица {table_counter}",
                "level": max(1, cur_level + 1),
                "text": t_text,
                "page": None,
                "section_path": _hpath(heading_stack_ids, heading_stack_titles + [t_title or f"Таблица {table_counter}"]),
                "element_type": "table",
                "attrs": attrs_tbl
            })

            # сброс состояния
            pending_tbl_num = None
            pending_tbl_tail = None
            awaiting_tail = False
            last_title_candidate = None
            last_title_candidate_age = 999
            continue

    # «Хвост» текста
    if last_title_candidate:
        buf.append(last_title_candidate)
    flush_text_section()

    return sections

# ----------------------------- PDF -----------------------------

def _group_words_into_lines(words: List[dict], y_tol: float = 2.0) -> List[Tuple[float, str]]:
    """
    Группируем слова pdfplumber в строки по близкому Y.
    Возвращает список (y_center, line_text) в порядке сверху-вниз.
    """
    if not words:
        return []
    # сортируем по Y, затем X
    words = sorted(words, key=lambda w: (w.get("top", 0), w.get("x0", 0)))
    lines: List[Tuple[float, List[dict]]] = []
    for w in words:
        if not w.get("text"):
            continue
        t = float(w.get("top", 0.0))
        if not lines:
            lines.append((t, [w]))
            continue
        last_y, last_words = lines[-1]
        if abs(t - last_y) <= y_tol:
            last_words.append(w)
        else:
            lines.append((t, [w]))
    out: List[Tuple[float, str]] = []
    for y, ws in lines:
        ws = sorted(ws, key=lambda z: z.get("x0", 0.0))
        text = " ".join([_clean(z.get("text", "")) for z in ws if _clean(z.get("text", ""))])
        if text:
            # y_center прикидываем по top + median высоты
            tops = [float(z.get("top", y)) for z in ws]
            bottoms = [float(z.get("bottom", y+10)) for z in ws]
            y_center = (min(tops) + max(bottoms)) / 2.0
            out.append((y_center, text))
    # сверху-вниз
    out.sort(key=lambda x: x[0])
    return out

def _fitz_images_with_bbox(pg) -> List[Dict[str, Any]]:
    """
    Возвращает список изображений страницы PyMuPDF с bbox и xref.
    """
    images: List[Dict[str, Any]] = []
    try:
        info = pg.get_text("dict")
        for b in info.get("blocks", []):
            if b.get("type") == 1:
                bbox = b.get("bbox")
                xref = b.get("xref") or b.get("image")
                if bbox and xref:
                    x0, y0, x1, y1 = bbox
                    images.append({
                        "xref": int(xref),
                        "bbox": (float(x0), float(y0), float(x1), float(y1)),
                        "yc": (float(y0) + float(y1)) / 2.0  # ВАЖНО: в системе fitz (ноль внизу!)
                    })
    except Exception:
        # мягко пропускаем
        return []
    return images

def parse_pdf(path: str) -> List[Dict[str, Any]]:
    """
    Расширенный парсинг PDF:
    - постраничный текст (element_type='page')
    - попытка извлечения простых таблиц (element_type='table')
    - эвристическое извлечение списка литературы (element_type='reference')
    - извлечение подписей «Рисунок N — …» и (если возможно) привязка изображений по близости (а не «по порядку»)
    """
    out: List[Dict[str, Any]] = []

    # Откроем PyMuPDF один раз (если доступен и включён)
    pdf_fitx = None
    if fitz is not None and CFG_PDF_EXTRACT:
        try:
            pdf_fitx = fitz.open(path)
        except Exception:
            pdf_fitx = None

    xref_cache: Dict[int, str] = {}  # xref -> saved path

    def _save_xref(xref: int) -> Optional[str]:
        if pdf_fitx is None:
            return None
        if xref in xref_cache:
            return xref_cache[xref]
        try:
            base = pdf_fitx.extract_image(xref)
            data = base.get("image", None)
            if not data:
                return None
            ext = "." + (base.get("ext") or "png")
            h = sha256_bytes(data)
            fp = save_bytes(data, safe_filename(f"pdf_img_{h[:12]}{ext}"), DEFAULT_UPLOAD_DIR)
            xref_cache[xref] = fp
            return fp
        except Exception:
            return None

    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            # 1) Строки и текст
            try:
                words = page.extract_words() or []
            except Exception:
                words = []
            lines = _group_words_into_lines(words)
            text_lines = [t for (_, t) in lines]
            plain_text = "\n".join(text_lines) or (page.extract_text() or "")

            out.append({
                "title": f"Стр. {i}",
                "level": 1,
                "text": plain_text,
                "page": i,
                "section_path": f"p.{i}",
                "element_type": "page",
                "attrs": {"numbers": _extract_numbers(plain_text)}
            })

            # 2) Простые таблицы (если распознаны pdfplumber'ом)
            try:
                tables = page.extract_tables()
            except Exception:
                tables = None
            if tables:
                for ti, rows in enumerate(tables, 1):
                    try:
                        lines_tbl = []
                        max_cols = 0
                        for row in rows or []:
                            row = [_clean(c) for c in (row or [])]
                            max_cols = max(max_cols, len(row))
                            line = " | ".join([c for c in row if c])
                            if line:
                                lines_tbl.append(line)
                        t_text = "\n".join(lines_tbl)
                        if t_text.strip():
                            label = f"{i}.{ti}"
                            header_preview = (lines_tbl[0] if lines_tbl else None)
                            if header_preview and len(header_preview) > 160:
                                header_preview = header_preview[:159].rstrip() + "…"
                            attrs = {
                                "caption_num": label,
                                "caption_tail": header_preview,
                                "label": label,
                                "title": header_preview,
                                "caption_source": "header_row" if header_preview else "none",
                                "n_rows": len(rows or []),
                                "n_cols": max_cols if max_cols > 0 else None,
                                "header_preview": header_preview,
                                "numbers": _extract_numbers(t_text),
                            }
                            human = _compose_table_title(label, header_preview)
                            out.append({
                                "title": human,
                                "level": 2,
                                "text": t_text,
                                "page": i,
                                "section_path": f"Стр. {i} / {human}",
                                "element_type": "table",
                                "attrs": attrs
                            })
                    except Exception:
                        pass

            # 3) Подписи к рисункам (по строкам с координатами) + привязка изображений по близости
            fig_caps: List[Tuple[str, str, float]] = []  # [(num, tail, y_center_top)]
            for y, line_text in lines:
                k, n, t = _classify_caption(line_text.strip())
                if k == "figure":
                    fig_caps.append((n or "", t or "", y))

            # подготовим список изображений с bbox для этой страницы
            images_with_bbox: List[Dict[str, Any]] = []
            if pdf_fitx is not None:
                try:
                    pg = pdf_fitx[i - 1]  # 0-based
                    images_with_bbox = _fitz_images_with_bbox(pg)
                except Exception:
                    images_with_bbox = []

            # ВАЖНО: согласуем системы координат (fitz: ноль внизу → в top-систему pdfplumber)
            page_height = float(page.height or 0.0)
            for im in images_with_bbox:
                try:
                    im["yc_top"] = page_height - float(im.get("yc", 0.0))
                except Exception:
                    im["yc_top"] = None

            # создаём секции figure
            if fig_caps:
                for (num, tail, ycap_top) in fig_caps:
                    title = _compose_figure_title(num, tail)
                    attrs = {
                        "caption_num": num,
                        "caption_tail": tail,
                        "label": num,
                        "title": tail,
                        "numbers": _extract_numbers(title),
                    }

                    # сопоставление по близости: ближайшее по вертикали изображение (в согласованной системе)
                    chosen_path: Optional[str] = None
                    if images_with_bbox:
                        best = None
                        for im in images_with_bbox:
                            if im.get("yc_top") is None:
                                continue
                            dist = abs(float(im["yc_top"]) - float(ycap_top))
                            if (best is None) or (dist < best[0]):
                                best = (dist, im)
                        if best is not None:
                            # разумный порог — 400pt по вертикали (примерно 14 см)
                            if best[0] <= 400:
                                xref = int(best[1]["xref"])
                                chosen_path = _save_xref(xref)

                    if chosen_path:
                        attrs["images"] = [chosen_path]

                    out.append({
                        "title": title,
                        "level": 2,
                        "text": title,
                        "page": i,
                        "section_path": f"Стр. {i} / {title}",
                        "element_type": "figure",
                        "attrs": attrs
                    })

            # 4) Эвристика для списка литературы на странице
            has_sources_kw = bool(SOURCES_TITLE_RE.search(plain_text))
            ref_lines = []
            for line in (plain_text.splitlines() or []):
                line = line.strip()
                if not line:
                    continue
                m = REF_LINE_RE.match(line)
                if m:
                    try:
                        idx = int(m.group(1) or m.group(2))
                    except Exception:
                        continue
                    tail = (m.group(3) or "").strip()
                    ref_lines.append((idx, tail))

            if has_sources_kw and len(ref_lines) >= 3:
                for idx, tail in ref_lines:
                    out.append({
                        "title": f"Источник {idx}",
                        "level": 2,
                        "text": tail,
                        "page": i,
                        "section_path": f"Стр. {i} / Источник {idx}",
                        "element_type": "reference",
                        "attrs": {"ref_index": idx, "numbers": _extract_numbers(tail)}
                    })

    # закрываем PyMuPDF, если открывали
    try:
        if pdf_fitx is not None:
            pdf_fitx.close()
    except Exception:
        pass

    return out

# -------- .doc -> .docx без офисов (Aspose.Words) + альтернативы --------

def _convert_doc_to_docx_with_aspose(doc_path: str, outdir: Path) -> Path:
    """Конвертация .doc -> .docx через Aspose.Words (pip install aspose-words)."""
    try:
        import aspose.words as aw
    except Exception as e:
        raise RuntimeError(f"Aspose.Words не установлен: {e}. Установите пакет 'aspose-words'.")
    doc = aw.Document(doc_path)
    out_file = outdir / (Path(doc_path).stem + ".docx")
    doc.save(str(out_file), aw.SaveFormat.DOCX)
    if not out_file.exists():
        raise RuntimeError("Aspose.Words не создал .docx файл.")
    return out_file

def _find_soffice() -> Optional[str]:
    cand = shutil.which("soffice") or shutil.which("soffice.com") or shutil.which("soffice.exe")
    if cand:
        return cand
    env = os.getenv("SOFFICE_PATH")
    if env and Path(env).exists():
        return env
    system = platform.system()
    candidates = []
    if system == "Windows":
        candidates += [
            r"C:\Program Files\LibreOffice\program\soffice.com",
            r"C:\Program Files\LibreOffice\program\soffice.exe",
            r"C:\Program Files (x86)\LibreOffice\program\soffice.com",
            r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
        ]
    elif system == "Darwin":
        candidates += ["/Applications/LibreOffice.app/Contents/MacOS/soffice"]
    else:
        candidates += ["/usr/bin/soffice", "/usr/local/bin/soffice"]
    for p in candidates:
        if Path(p).exists():
            return p
    return None

def _convert_doc_to_docx_with_soffice(doc_path: str, outdir: Path) -> Path:
    soffice_bin = _find_soffice()
    if not soffice_bin:
        raise RuntimeError("LibreOffice не найден (ни в PATH, ни по стандартным путям, ни в SOFFICE_PATH).")
    cmd = [soffice_bin, "--headless", "--convert-to", "docx", "--outdir", str(outdir), doc_path]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out = outdir / (Path(doc_path).stem + ".docx")
    if not out.exists():
        found = list(outdir.glob("*.docx"))
        if not found:
            raise RuntimeError("Конвертация прошла, но .docx не найден.")
        return found[0]
    return out

def _convert_doc_to_docx_with_word(doc_path: str, outdir: Path) -> Path:
    try:
        import win32com.client  # pywin32
    except Exception as e:
        raise RuntimeError(f"pywin32 недоступен: {e}")
    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False
    try:
        doc = word.Documents.Open(str(Path(doc_path).resolve()))
        out_file = outdir / (Path(doc_path).stem + ".docx")
        doc.SaveAs(str(out_file), FileFormat=16)  # 16 = wdFormatXMLDocument
        doc.Close(False)
    finally:
        word.Quit()
    if not out_file.exists():
        raise RuntimeError("MS Word не создал .docx файл.")
    return out_file

def parse_doc(path: str) -> List[Dict[str, Any]]:
    """
    Парсит .doc: 1) Aspose.Words, 2) LibreOffice (если есть), 3) Word COM (если есть),
    затем переиспользует parse_docx().
    """
    with tempfile.TemporaryDirectory() as tmp:
        outdir = Path(tmp)
        errors: List[str] = []

        try:
            out_path = _convert_doc_to_docx_with_aspose(path, outdir)
            return parse_docx(str(out_path))
        except Exception as e:
            errors.append(f"Aspose: {e}")

        try:
            out_path = _convert_doc_to_docx_with_soffice(path, outdir)
            return parse_docx(str(out_path))
        except Exception as e:
            errors.append(f"LibreOffice: {e}")

        if platform.system() == "Windows":
            try:
                out_path = _convert_doc_to_docx_with_word(path, outdir)
                return parse_docx(str(out_path))
            except Exception as e:
                errors.append(f"Word COM: {e}")

        raise RuntimeError("Не удалось обработать .doc. " + " | ".join(errors))

# ----------------------------- SAVE -----------------------------

def save_upload(raw: bytes, filename: str, upload_dir: str = DEFAULT_UPLOAD_DIR) -> str:
    p = Path(upload_dir)
    p.mkdir(parents=True, exist_ok=True)
    fp = p / filename
    fp.write_bytes(raw)
    return str(fp)
