# app/parsing.py
from docx import Document as Docx
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.text.run import Run
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
try:
    from lxml import etree as ET
except Exception:
    import xml.etree.ElementTree as ET
import pdfplumber
from pathlib import Path
import subprocess
import tempfile
import platform
import shutil
import os
import re
from typing import List, Dict, Any, Optional, Tuple
from decimal import Decimal, InvalidOperation
# мягкий импорт OOXML-индекса (для вытаскивания chart_data по номерам рисунков)
try:
    from . import ooxml_lite as _ooxml_lite
except Exception:
    _ooxml_lite = None

_IMG_CACHE: Dict[str, str] = {}

# мягкая зависимость для извлечения изображений из PDF
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None  # без него делаем только текстовые подписи

# ----------------------------- config -----------------------------
try:
    from .config import Cfg
    DEFAULT_UPLOAD_DIR = Cfg.UPLOAD_DIR
    # большее окно поиска соседних блоков
    CFG_FIG_WINDOW = int(getattr(Cfg, "FIG_NEIGHBOR_WINDOW", 24))
    # явный look-ahead вперёд от подписи
    CFG_FIG_LOOKAHEAD = int(getattr(Cfg, "FIG_LOOKAHEAD", 60))
    CFG_PDF_EXTRACT = bool(getattr(Cfg, "PDF_EXTRACT_IMAGES", True))
    CFG_MIN_IMAGE_EMU = int(getattr(Cfg, "MIN_IMAGE_EMU", 600_000))
    # NEW: управление векторным рендером возле подписи
    CFG_PDF_VECTOR_RASTERIZE = bool(getattr(Cfg, "PDF_VECTOR_RASTERIZE", True))
    CFG_PDF_VECTOR_DPI = int(getattr(Cfg, "PDF_VECTOR_DPI", 360))
    CFG_PDF_VECTOR_MIN_WIDTH_PX = int(getattr(Cfg, "PDF_VECTOR_MIN_WIDTH_PX", 1200))
    CFG_PDF_VECTOR_PAD_PX = int(getattr(Cfg, "PDF_VECTOR_PAD_PX", 16))
    CFG_PDF_VECTOR_MAX_DPI = int(getattr(Cfg, "PDF_VECTOR_MAX_DPI", 600))
    CFG_PDF_CAPTION_DY = int(getattr(Cfg, "PDF_CAPTION_MAX_DISTANCE_PX", 300))
except Exception:
    DEFAULT_UPLOAD_DIR = "./uploads"
    CFG_FIG_WINDOW = 24
    CFG_FIG_LOOKAHEAD = 60
    CFG_PDF_EXTRACT = True
    CFG_MIN_IMAGE_EMU = 600_000  # ~1.5 см

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

# ----------------------------- inline images & units -----------------------------

NS = {
    "a":   "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r":   "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
    "wp":  "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "c":   "http://schemas.openxmlformats.org/drawingml/2006/chart",
    "v":   "urn:schemas-microsoft-com:vml",
    "w":   "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
}

def _paragraph_has_image(p: Paragraph) -> bool:
    try:
        return bool(p._p.xpath(".//pic:pic | .//w:pict//v:imagedata", namespaces=NS))
    except Exception:
        return False

# EMU converters
EMU_PER_INCH, EMU_PER_CM, EMU_PER_PT, EMU_PER_PX = 914400, 360000, 12700, 9525

def _to_emu(val: str) -> Optional[int]:
    try:
        s = val.strip().lower().replace(",", ".")
        if s.endswith("cm"): return int(float(s[:-2]) * EMU_PER_CM)
        if s.endswith("in"): return int(float(s[:-2]) * EMU_PER_INCH)
        if s.endswith("pt"): return int(float(s[:-2]) * EMU_PER_PT)
        if s.endswith("px"): return int(float(s[:-2]) * EMU_PER_PX)
        return int(float(s) * EMU_PER_PT)
    except Exception:
        return None

def _emu_from_vml_style(style: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
    if not style:
        return None, None
    w = _to_emu(re.search(r"width\s*:\s*([^;]+)", style, re.I).group(1)) if re.search(r"width\s*:\s*([^;]+)", style, re.I) else None
    h = _to_emu(re.search(r"height\s*:\s*([^;]+)", style, re.I).group(1)) if re.search(r"height\s*:\s*([^;]+)", style, re.I) else None
    return w, h

# ----------------------------- image extraction -----------------------------

def _extract_images_from_container(doc: Docx, container, uploads_dir: str) -> List[str]:
    out: List[str] = []
    try:
        items: List[Tuple[str, str, Optional[int], Optional[int]]] = []  # (kind, rid, cx, cy)

        # DrawingML (r:embed + ближайший wp:extent)
        for blip in container.xpath(".//a:blip", namespaces=NS):
            rid = blip.get(qn("r:embed"))
            if not rid:
                continue
            ext = blip.xpath("./ancestor::wp:inline[1]/wp:extent | ./ancestor::wp:anchor[1]/wp:extent", namespaces=NS)
            cx = cy = None
            if ext:
                try:
                    cx = int(ext[0].get("cx") or 0)
                    cy = int(ext[0].get("cy") or 0)
                except Exception:
                    cx = cy = None
            items.append(("dml", rid, cx, cy))

        # VML (<w:pict><v:imagedata r:id="…">)
        for node in container.xpath(".//w:pict//v:imagedata | .//v:shape//v:imagedata", namespaces=NS):
            rid = node.get(qn("r:id"))
            if not rid:
                continue
            shape = node.getparent()
            while shape is not None and shape.tag.split('}')[-1].lower() not in ("shape", "pict"):
                shape = shape.getparent()
            style = shape.get("style") if shape is not None else None
            cx, cy = _emu_from_vml_style(style)
            items.append(("vml", rid, cx, cy))

        for _, rid, cx, cy in items:
            # фильтруем «мелочь», если известен размер
            if cx is not None and cy is not None and min(cx, cy) < CFG_MIN_IMAGE_EMU:
                continue

            part = doc.part.related_parts.get(rid)
            if not part:
                continue
            data = getattr(part, "blob", None)
            if not data:
                continue

            h = sha256_bytes(data)
            if h in _IMG_CACHE:
                path = _IMG_CACHE[h]
                if path not in out:
                    out.append(path)
                continue

            ct = (getattr(part, "content_type", "") or "").lower()
            ext = ".png"
            if "jpeg" in ct or "jpg" in ct: ext = ".jpg"
            elif "png" in ct: ext = ".png"
            elif "gif" in ct: ext = ".gif"
            elif "bmp" in ct: ext = ".bmp"
            elif "tiff" in ct or "tif" in ct: ext = ".tiff"
            elif "svg"  in ct: ext = ".svg"
            elif "emf"  in ct: ext = ".emf"
            elif "wmf"  in ct: ext = ".wmf"

            fp = str(Path(save_bytes(data, safe_filename(f"docx_fig_{h[:12]}{ext}"), uploads_dir)).resolve())
            _IMG_CACHE[h] = fp
            out.append(fp)
    except Exception:
        pass
    return out

def _extract_paragraph_images(doc: Docx, p: Paragraph, uploads_dir: str = DEFAULT_UPLOAD_DIR) -> List[str]:
    return _extract_images_from_container(doc, p._p, uploads_dir)

def _extract_table_images(doc: Docx, tbl: Table, uploads_dir: str = DEFAULT_UPLOAD_DIR) -> List[str]:
    return _extract_images_from_container(doc, tbl._tbl, uploads_dir)

def _collect_neighbor_images(doc: Docx, block: Paragraph, window: int = CFG_FIG_WINDOW,
                             uploads_dir: str = DEFAULT_UPLOAD_DIR) -> List[str]:
    """Собираем картинки из блоков в диапазоне [idx-window, idx+window]."""
    try:
        body = doc._element.body
        children = list(body.iterchildren())
        idx = next((i for i, el in enumerate(children) if getattr(block, "_p", None) is el), None)
        if idx is None:
            return []
        out: List[str] = []
        for j in range(max(0, idx - window), min(len(children) - 1, idx + window) + 1):
            el = children[j]
            if el.tag.endswith("p"):
                out += _extract_paragraph_images(doc, Paragraph(el, doc), uploads_dir)
            elif el.tag.endswith("tbl"):
                out += _extract_table_images(doc, Table(el, doc), uploads_dir)
        # дедуп
        uniq: List[str] = []
        for pth in out:
            if pth not in uniq:
                uniq.append(pth)
        return uniq
    except Exception:
        return []

def _first_images_after(doc: Docx, block: Paragraph, steps: int = CFG_FIG_LOOKAHEAD,
                        uploads_dir: str = DEFAULT_UPLOAD_DIR) -> List[str]:
    """Жёсткий look-ahead: найдём первую(ые) картинки после подписи в пределах steps."""
    try:
        body = doc._element.body
        children = list(body.iterchildren())
        idx = next((i for i, el in enumerate(children) if getattr(block, "_p", None) is el), None)
        if idx is None:
            return []
        out: List[str] = []
        hi = min(len(children) - 1, idx + steps)
        for j in range(idx + 1, hi + 1):
            el = children[j]
            if el.tag.endswith("p"):
                out += _extract_paragraph_images(doc, Paragraph(el, doc), uploads_dir)
            elif el.tag.endswith("tbl"):
                out += _extract_table_images(doc, Table(el, doc), uploads_dir)
            if out:
                # как только нашли — завершаем и дедупим
                uniq: List[str] = []
                for pth in out:
                    if pth not in uniq:
                        uniq.append(pth)
                return uniq
        return []
    except Exception:
        return []


# --- bbox / vector helpers (NEW) ---
def _rect_union(rects: List[Tuple[float, float, float, float]]) -> Optional[Tuple[float, float, float, float]]:
    if not rects:
        return None
    x0 = min(r[0] for r in rects); y0 = min(r[1] for r in rects)
    x1 = max(r[2] for r in rects); y1 = max(r[3] for r in rects)
    return (x0, y0, x1, y1)

def _bbox_vdist_generic(a, b) -> float:
    ax0, ay0, ax1, ay1 = a; bx0, by0, bx1, by1 = b
    if by0 >= ay1: return abs(by0 - ay1)
    if by1 <= ay0: return abs(ay0 - by1)
    return 0.0

def _fitz_vector_rects(pg) -> List[Tuple[float, float, float, float]]:
    """bbox'ы векторных примитивов (lines/paths/rects) на странице."""
    try:
        drawings = pg.get_drawings() or []
    except Exception:
        return []
    rects = []
    for d in drawings:
        r = d.get("rect")
        if r:
            rects.append((float(r.x0), float(r.y0), float(r.x1), float(r.y1)))
    return rects

def _render_clip_png(pg, clip_rect: Tuple[float, float, float, float],
                     dpi_base: int, min_w_px: int, max_dpi: int, pad_px: int) -> Optional[bytes]:
    try:
        import fitz
    except Exception:
        return None
    # расчёт zoom: базовый DPI + гарантия минимальной ширины
    zoom = max(1.0, float(dpi_base) / 72.0)
    width_pt = max(1.0, clip_rect[2] - clip_rect[0])
    need_zoom = float(min_w_px) / width_pt
    zoom = min(max(zoom, need_zoom), float(max_dpi) / 72.0)
    # паддинг в поинтах
    pad_pt = pad_px / zoom
    page_rect = pg.rect
    clip = fitz.Rect(
        max(page_rect.x0, clip_rect[0] - pad_pt),
        max(page_rect.y0, clip_rect[1] - pad_pt),
        min(page_rect.x1, clip_rect[2] + pad_pt),
        min(page_rect.y1, clip_rect[3] + pad_pt),
    )
    try:
        pix = pg.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip, alpha=True)
        return pix.tobytes("png")
    except Exception:
        return None

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

# ----------------------------- tables helpers -----------------------------

def _table_row_strings(tbl: Table) -> List[str]:
    """Возвращает строки таблицы: одна строка = ряд, ячейки через ' | ' (с учётом merged)."""
    lines: List[str] = []
    try:
        for row in tbl.rows:
            cells: List[str] = []
            for c in row.cells:
                cells.append(_clean(c.text))

            # Важно: НЕ удаляем соседние дубликаты — это ломает реальные данные
            # (например, 35% и 35% в соседних столбцах).
            # Убираем только хвостовые пустые ячейки.
            while cells and not (cells[-1] or "").strip():
                cells.pop()

            # если строка полностью пустая — пропускаем
            if not any((x or "").strip() for x in cells):
                continue

            # Склеиваем с сохранением порядка (пустые внутри не выкидываем)
            line = " | ".join(cells).strip(" |")
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
        (?:\s*(?:[.\-—:\u2013\u2014])\s*(.*))?         # точка/двоеточие/тире + подпись
        \s*$""",
    re.IGNORECASE | re.VERBOSE
)

# «Рис. 1.», «Рисунок 2,1 — ...», «Fig. 3: ...», «Figure 4 ...», «Рис. А.1 ...»
CAPTION_RE_FIG = re.compile(
    r"""^\s*
        (?:
            рис(?:\.|унок)?
            | figure
            | fig\.?
        )
        \s*
        (?:№\s*)?
        (
          (?:[A-Za-zА-Яа-я]\.?[\s-]*\d+(?:[.,]\d+)*)   # А.1 / П1.2
          |
          (?:\d+(?:[.,]\d+)*)                          # 1 / 2.1 / 3,2 / 2.1.3
        )
        (?:\s*(?:[.\-—:\u2013\u2014])\s*(.*))?
        \s*$""",
    re.IGNORECASE | re.VERBOSE
)

def _classify_caption(text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Возвращает (kind, number_label, tail) или (None, None, None).

    number_label всегда в нормализованном виде (без буквенных префиксов, пробелов и запятых).
    """
    t = (text or "").strip()
    m = CAPTION_RE_TABLE.match(t)
    if m:
        raw_num = (m.group(1) or "")
        num = _norm_caption_num(raw_num)
        tail = (m.group(2) or "").strip()
        return "table", num, tail
    m = CAPTION_RE_FIG.match(t)
    if m:
        raw_num = (m.group(1) or "")
        num = _norm_caption_num(raw_num)
        tail = (m.group(2) or "").strip()
        return "figure", num, tail
    return None, None, None


def _norm_caption_num(s: Any) -> Optional[str]:
    """
    Нормализуем строковый номер подписи:
    'А. 2,3' -> '2.3'
    'Рисунок А 2.3' -> '2.3'
    '2.3 '   -> '2.3'
    """
    if s is None:
        return None
    try:
        s = str(s)
    except Exception:
        return None
    s = s.replace("\u00A0", " ").strip()
    if not s:
        return None
    # убираем ведущую литеру/буквы с точками/пробелами/тире
    s = re.sub(r"^[A-Za-zА-Яа-яЁё]\.?[\s\-]*", "", s)
    s = s.replace(" ", "")
    s = s.replace(",", ".")
    return s or None

def _compose_table_title(num: Optional[str], tail: Optional[str]) -> str:
    if num and tail:
        return f"Таблица {num} — {tail}"
    if num:
        return f"Таблица {num}"
    if tail:
        return tail
    return "Таблица"

def _compose_figure_title(num: Optional[str], tail: Optional[str]) -> str:
    if num and tail:
        return f"Рисунок {num} — {tail}"
    if num:
        return f"Рисунок {num}"
    if tail:
        return tail
    return "Рисунок"

def _is_title_candidate(text: str, attrs: dict) -> bool:
    """Эвристика: короткий (до 200) центрированный/не списочный — кандидат в подпись."""
    t = (text or "").strip()
    if not t:
        return False
    if CAPTION_RE_TABLE.match(t) or CAPTION_RE_FIG.match(t):
        return False
    if len(t) < 3 or len(t) > 200:
        return False
    al = (attrs or {}).get("alignment")
    if al not in {"center", "right", "left"}:
        return False
    if (attrs or {}).get("is_list"):
        return False
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

APPENDIX_TITLE_RE = re.compile(r"\b(приложени[ея]|appendix)\b", re.IGNORECASE)

# НОВОЕ: текстовый детектор глав/разделов типа «ГЛАВА 1», «Раздел 2.3», «Chapter 1»
CHAPTER_TITLE_RE = re.compile(
    r"""^\s*
        (глава|раздел|chapter)\s+
        \d+(?:[.,]\d+)*
        """,
    re.IGNORECASE | re.VERBOSE,
)

# разрешаем «12. », «12) », «12 - », «12 — », «[12] », и «12 <пробел>»
REF_LINE_RE = re.compile(r"^\s*(?:\[(\d+)\]|(\d+)(?:[\.\)\-–—:]\s*|\s+))\s*(.+)$")

def _is_sources_title(s: str) -> bool:
    return bool(SOURCES_TITLE_RE.search(s or ""))

def _is_appendix_title(s: str) -> bool:
    return bool(APPENDIX_TITLE_RE.search(s or ""))

def _parse_reference_line(s: str) -> Tuple[Optional[int], str]:
    """Из строки источника выделяет номер (если есть) и хвост."""
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


def _safe_decimal(text: str) -> Optional[Decimal]:
    """
    Аккуратно парсим число из OOXML-строки:
    - убираем неразрывные пробелы и обычные пробелы
    - меняем запятую на точку
    - отрезаем '%' на конце
    Возвращаем Decimal или None, если парсинг не удался.
    """
    try:
        s = (text or "").strip()
        s = s.replace("\u00A0", " ").replace(" ", "")
        if not s:
            return None
        if s.endswith("%"):
            s = s[:-1]
        s = s.replace(",", ".")
        if not s:
            return None
        return Decimal(s)
    except (InvalidOperation, ValueError):
        return None

# ----------------------------- charts (docx) -----------------------------

def _paragraph_chart_rids(p: Paragraph) -> List[str]:
    """RID'ы на встроенные диаграммы в абзаце (w:drawing/a:graphicData/c:chart)."""
    try:
        nodes = p._p.xpath(".//a:graphic//a:graphicData//c:chart", namespaces=NS)
        rids = []
        for n in nodes:
            rid = n.get(qn("r:id")) if hasattr(n, "get") else n.attrib.get("{%s}id" % NS["r"])
            if rid:
                rids.append(rid)
        return rids
    except Exception:
        return []

def _table_chart_rids(tbl: Table) -> List[str]:
    """RID'ы на встроенные диаграммы внутри таблицы (w:tbl)."""
    try:
        nodes = tbl._tbl.xpath(".//a:graphic//a:graphicData//c:chart", namespaces=NS)
        rids: List[str] = []
        for n in nodes:
            rid = n.get(qn("r:id")) if hasattr(n, "get") else n.attrib.get("{%s}id" % NS["r"])
            if rid:
                rids.append(rid)
        return rids
    except Exception:
        return []

def _series_name(ser) -> Optional[str]:
    try:
        v = ser.find(".//c:tx/c:v", NS)
        if v is not None and (v.text or "").strip():
            return v.text.strip()
        v = ser.find(".//c:tx/c:strRef/c:strCache/c:pt/c:v", NS)
        if v is not None and (v.text or "").strip():
            return v.text.strip()
    except Exception:
        pass
    return None

def _cache_points(root, path: str) -> List[Tuple[int, str]]:
    """Возвращает [(idx, value_text)] из numCache/strCache по указанному XPath."""
    pts: List[Tuple[int, str]] = []
    for pt in root.findall(path + "/c:pt", NS):
        try:
            idx = int(pt.attrib.get("idx", "0"))
        except Exception:
            continue
        val_el = pt.find("c:v", NS)
        val = (val_el.text if val_el is not None else "") or ""
        pts.append((idx, val))
    pts.sort(key=lambda x: x[0])
    return pts

def _parse_chart_xml(xml_bytes: bytes) -> Tuple[Optional[str], List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Возвращает (chart_type, rows, chart_matrix).

    rows — плоский список словарей вида:
      {
        "label":       str,            # подпись категории + (опц.) имя серии
        "value":       float | None,   # числовое значение (для процентов уже в шкале 0–100)
        "unit":        str | None,     # "%", если это проценты, иначе None
        "value_raw":   str,            # исходная строка из XML
        "series_name": str | None,     # имя серии (tx)
        "category":    str,            # подпись категории без префикса серии
      }

    chart_matrix — нормализованное представление:
      {
        "chart_type": str,
        "categories": [str, ...],
        "series": [
          {"name": str | None, "values": [float | None, ...], "unit": str | None}
        ],
        "unit": str | None,   # общая единица измерения (если возможна)
        "meta": {
          "bar_dir": str | None,          # "bar" / "col" для barChart
          "cat_orientation": str | None,  # "minMax"/"maxMin" (ось категорий)
          "val_orientation": str | None,  # "minMax"/"maxMin" (ось значений)
        }
      }

    Поддерживаем Pie/Bar/Column/Line. Если парсинг не удался — (None, [], None).
    """
    try:
        root = ET.fromstring(xml_bytes)
    except Exception:
        return None, [], None

    def _first(*paths):
        for p in paths:
            el = root.find(".//" + p, NS)
            if el is not None:
                return el
        return None

    chart_el = _first(
        "c:pieChart",
        "c:barChart",
        "c:bar3DChart",
        "c:colChart",
        "c:col3DChart",
        "c:lineChart",
    )
    if chart_el is None:
        return None, [], None

    tag = chart_el.tag.split("}")[-1]
    ctype = {
        "pieChart": "PieChart",
        "barChart": "BarChart",
        "bar3DChart": "BarChart",
        "colChart": "BarChart",
        "col3DChart": "BarChart",
        "lineChart": "LineChart",
    }.get(tag, tag)

    # --- оси и ориентация (для bar/column/line) ---
    bar_dir: Optional[str] = None
    if tag in {"barChart", "bar3DChart", "colChart", "col3DChart"}:
        bar_dir_el = chart_el.find("./c:barDir", NS)
        if bar_dir_el is not None:
            bar_dir = (bar_dir_el.get("val") or "").strip() or None

    cat_orientation: Optional[str] = None
    val_orientation: Optional[str] = None
    plot_area = root.find(".//c:plotArea", NS)
    if plot_area is not None:
        cat_ax = plot_area.find("./c:catAx", NS)
        if cat_ax is not None:
            o_el = cat_ax.find("./c:scaling/c:orientation", NS)
            if o_el is not None:
                cat_orientation = (o_el.get("val") or "").strip() or None
        val_ax = plot_area.find("./c:valAx", NS)
        if val_ax is not None:
            o_el = val_ax.find("./c:scaling/c:orientation", NS)
            if o_el is not None:
                val_orientation = (o_el.get("val") or "").strip() or None

    out: List[Dict[str, Any]] = []

    # Для matrix
    categories: Optional[List[str]] = None
    matrix_series: List[Dict[str, Any]] = []
    matrix_unit: Optional[str] = None

    series = chart_el.findall("./c:ser", NS) or []
    multi_series = len(series) > 1

    for si, ser in enumerate(series):
        sname = _series_name(ser)

        # категории
        cat_pts = _cache_points(ser, ".//c:cat/c:strRef/c:strCache") or \
                  _cache_points(ser, ".//c:cat/c:numRef/c:numCache")

        # значения (idx -> raw string)
        val_pts = _cache_points(ser, ".//c:val/c:numRef/c:numCache") or \
                  _cache_points(ser, ".//c:val/c:numCache")

        # карта idx -> категория (запасной вариант)
        cats_by_idx = {i: (l or "").strip() for i, l in cat_pts}
        # последовательности по порядку
        cat_seq = [(l or "").strip() for _, l in cat_pts]
        val_seq = [(idx, (v or "")) for idx, v in val_pts]

        if si == 0:
            # первая серия задаёт «главный» список категорий
            categories = cat_seq[:]

        # формат серии: ищем numFmt с '%'
        is_percent_fmt = False
        for nm in ser.findall(".//c:numFmt", NS):
            fc = (nm.get("formatCode") or "").lower()
            if "%" in fc:
                is_percent_fmt = True
                break

        # собираем Decimal'ы для эвристики по значениям
        dec_vals: List[Decimal] = []
        for _, raw_v in val_seq:
            d = _safe_decimal(raw_v)
            if d is not None:
                dec_vals.append(d)

        # эвристика: если формат не явно процентный, но сумма ≈1 и max≤1 — считаем долями
        is_percent_by_values = False
        if dec_vals:
            max_val = max(dec_vals)
            s_val = sum(dec_vals)
            if max_val >= Decimal("0") and max_val <= Decimal("1.05"):
                if abs(s_val - Decimal("1")) <= Decimal("0.05"):
                    is_percent_by_values = True

        is_percent = is_percent_fmt or is_percent_by_values
        unit = "%" if is_percent else None
        if unit and matrix_unit is None:
            matrix_unit = unit

        # --- плоский список точек (rows) ---
        for pos, (idx, raw_v) in enumerate(val_seq):
            raw_str = (raw_v or "").strip()

            # основное сопоставление — по порядку (позиции)
            if pos < len(cat_seq):
                cat_label = cat_seq[pos]
            else:
                # запасной вариант — по индексу, как раньше
                cat_label = cats_by_idx.get(idx, "").strip()

            base_label = cat_label
            label = base_label
            if multi_series and sname:
                label = f"{sname}: {base_label}" if base_label else sname

            d = _safe_decimal(raw_str)
            value_num: Optional[float] = None
            if d is not None:
                if is_percent and abs(d) <= Decimal("1.05"):
                    # доля 0–1 → шкала 0–100
                    value_num = float(d * Decimal("100"))
                else:
                    value_num = float(d)

            if not label and value_num is None and not raw_str:
                continue

            out.append(
                {
                    "label": label,
                    "value": value_num,
                    "unit": unit,
                    "value_raw": raw_str,
                    "series_name": sname,
                    "category": cat_label,
                }
            )

        # --- значения, выровненные по категориям (для matrix) ---
        if categories:
            values_aligned: List[Optional[float]] = []
            # map idx -> Decimal once
            idx_to_dec: Dict[int, Optional[Decimal]] = {}
            for idx, raw_v in val_seq:
                idx_to_dec[idx] = _safe_decimal(raw_v or "")

            # основной сценарий — одинаковый порядок категорий
            for pos_cat, cat in enumerate(categories):
                if pos_cat < len(val_seq):
                    _, raw_v = val_seq[pos_cat]
                    d = _safe_decimal(raw_v or "")
                else:
                    # если вдруг длины не совпали — пробуем по индексу
                    d = idx_to_dec.get(pos_cat, None)

                if d is not None:
                    if is_percent and abs(d) <= Decimal("1.05"):
                        v_num = float(d * Decimal("100"))
                    else:
                        v_num = float(d)
                else:
                    v_num = None
                values_aligned.append(v_num)

            matrix_series.append(
                {
                    "name": sname,
                    "values": values_aligned,
                    "unit": unit,
                }
            )

    chart_matrix: Optional[Dict[str, Any]] = None
    if categories and matrix_series:
        chart_matrix = {
            "chart_type": ctype,
            "categories": categories[:],
            "series": matrix_series,
            "unit": matrix_unit,
            "meta": {
                "bar_dir": bar_dir,
                "cat_orientation": cat_orientation,
                "val_orientation": val_orientation,
            },
        }
        # если ось категорий идёт max→min — разворачиваем матрицу
        if chart_matrix["meta"].get("cat_orientation") == "maxMin":
            chart_matrix["categories"] = list(reversed(chart_matrix["categories"]))
            for s in chart_matrix["series"]:
                s["values"] = list(reversed(s["values"]))

    return ctype, out, chart_matrix


def _chart_data_from_rid(doc: Docx, rid: str) -> Tuple[Optional[str], List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Достаём chart*.xml по rId и парсим."""
    try:
        part = doc.part.related_parts.get(rid)
        if not part:
            return None, [], None
        xml_bytes = getattr(part, "blob", None)
        if not xml_bytes:
            return None, [], None
        return _parse_chart_xml(xml_bytes)
    except Exception:
        return None, [], None


def _extract_chart_data_from_paragraph(doc: Docx, p: Paragraph) -> Tuple[Optional[str], List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Для абзаца возвращает первую найденную (тип, data, matrix) диаграмму."""
    for rid in _paragraph_chart_rids(p):
        ctype, rows, matrix = _chart_data_from_rid(doc, rid)
        if rows or matrix:
            return ctype, rows, matrix
    return None, [], None

def _extract_chart_data_from_table(doc: Docx, tbl: Table) -> Tuple[Optional[str], List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Для таблицы возвращает первую найденную (тип, data, matrix) диаграмму."""
    for rid in _table_chart_rids(tbl):
        ctype, rows, matrix = _chart_data_from_rid(doc, rid)
        if rows or matrix:
            return ctype, rows, matrix
    return None, [], None

def _collect_neighbor_chart_data(doc: Docx, block: Paragraph, window: int = CFG_FIG_WINDOW) -> Tuple[Optional[str], List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Ищем диаграмму в диапазоне блоков вокруг anchor-параграфа."""
    try:
        body = doc._element.body
        children = list(body.iterchildren())
        idx = None
        for i, el in enumerate(children):
            if getattr(block, "_p", None) is el:
                idx = i
                break
        if idx is None:
            return None, [], None
        lo = max(0, idx - window)
        hi = min(len(children) - 1, idx + window)
        for j in range(lo, hi + 1):
            el = children[j]
            if el.tag.endswith("p"):
                p2 = Paragraph(el, doc)
                ctype, rows, matrix = _extract_chart_data_from_paragraph(doc, p2)
                if rows or matrix:
                    return ctype, rows, matrix
            elif el.tag.endswith("tbl"):
                t2 = Table(el, doc)
                ctype, rows, matrix = _extract_chart_data_from_table(doc, t2)
                if rows or matrix:
                    return ctype, rows, matrix
        return None, [], None
    except Exception:
        return None, [], None

# ----------------------------- section-path helpers -----------------------------

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

    NEW:
      - Детект номерных псевдо-заголовков вида "2.1 Название" (часто не heading-стиль)
      - Для таблиц сохраняем полный grid/TSV (все колонки), чтобы downstream не терял значения
    """
    # Пытаемся заранее построить OOXML-индекс (если модуль доступен).
    oox_index: Optional[Dict[str, Any]] = None
    if _ooxml_lite is not None:
        try:
            oox_index = _ooxml_lite.build_index(path)
        except Exception:
            oox_index = None

    doc = Docx(path)

    sections: List[Dict[str, Any]] = []
    buf: List[str] = []
    last_para_attrs: Optional[dict] = None

    cur_title, cur_level = "Документ", 0
    outline_counters: List[int] = []
    heading_stack_titles: List[str] = []
    heading_stack_ids: List[str] = []

    def _current_scope_id() -> str:
        return _make_section_id(outline_counters)

    def _current_scope_path() -> str:
        return _hpath(heading_stack_ids, heading_stack_titles)

    def _enter_heading(level: int, title: str) -> str:
        nonlocal outline_counters, heading_stack_titles, heading_stack_ids, cur_title, cur_level
        if level < 1:
            level = 1
        while len(outline_counters) < level:
            outline_counters.append(0)
        outline_counters = outline_counters[:level]
        outline_counters[level - 1] += 1

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

    # NEW: номерной заголовок вида "2.1 Название", "3.2.1 ..." (чтобы секции не терялись)
    NUM_HEADING_RE = re.compile(r"^\s*(\d+(?:[.,]\d+){0,6})\s+(.+?)\s*$")

    def _is_numbered_heading(text: str, attrs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Пытаемся отличить заголовок "2.1 ..." от нумерованных списков "1) ..." и обычного текста.
        Возвращает dict {num, title, level} или None.
        """
        t = (text or "").strip()
        if not t:
            return None

        m = NUM_HEADING_RE.match(t)
        if not m:
            return None

        num_raw = (m.group(1) or "").strip()
        title = (m.group(2) or "").strip()
        if not title:
            return None

        # Нормализуем номер
        num = num_raw.replace(",", ".").strip().strip(".")
        if not num:
            return None

        # Против нумерованных списков "1) ..."
        if re.match(r"^\d+\)\s+", t):
            return None

        # Заголовки обычно короче
        if len(title) > 140:
            return None

        looks_like_title = _is_title_candidate(t, attrs)
        if not looks_like_title:
            # всё равно встречается "2.1 ..." как обычный абзац без форматирования —
            # но тогда строка должна быть короткая и "не похожа на обычный текст"
            if len(t) > 80:
                return None
            # слишком много точек (кроме номера) — похоже на обычный текст
            if t.count(".") > num.count(".") + 1:
                return None

        level = 1 + num.count(".")
        level = max(1, min(level, 8))
        return {"num": num, "title": title, "level": level}

    def _enter_numbered_heading(level: int, title: str, num_id: str) -> str:
        """
        Входим в заголовок, но ID секции делаем равным реальному номеру ("2.1"),
        чтобы потом легко матчить по sec/section_path.
        """
        nonlocal outline_counters, heading_stack_titles, heading_stack_ids, cur_title, cur_level

        if level < 1:
            level = 1

        # синхронизируем outline_counters с реальным номером (если он "чисто числовой")
        parts = [p for p in (num_id or "").split(".") if p.strip()]
        parsed: List[int] = []
        ok = True
        for p in parts:
            if p.isdigit():
                parsed.append(int(p))
            else:
                ok = False
                break

        if ok and parsed:
            outline_counters = parsed[:]  # теперь _current_scope_id будет совпадать с "2.1"
        else:
            # fallback на старую логику
            while len(outline_counters) < level:
                outline_counters.append(0)
            outline_counters = outline_counters[:level]
            outline_counters[level - 1] += 1

        # растягиваем стеки
        if len(heading_stack_titles) < level:
            heading_stack_titles += [""] * (level - len(heading_stack_titles))
        if len(heading_stack_ids) < level:
            heading_stack_ids += [""] * (level - len(heading_stack_ids))

        heading_stack_titles[:] = heading_stack_titles[:level]
        heading_stack_ids[:] = heading_stack_ids[:level]

        # проставляем id текущего уровня = реальный номер
        heading_stack_titles[level - 1] = (title or "Без названия")
        heading_stack_ids[level - 1] = num_id

        # если у родителя пусто — заполним префиксами "2", "2.1"…
        if ok and parsed:
            for i in range(level - 1):
                if not heading_stack_ids[i]:
                    heading_stack_ids[i] = ".".join(str(x) for x in parsed[: i + 1])

        cur_title = title or "Без названия"
        cur_level = level
        return num_id

    def _ooxml_chart_by_caption(num: Optional[str]) -> Optional[Dict[str, Any]]:
        """
        Ищем в OOXML-индексе фигуру с таким же номером подписи и вытаскиваем оттуда chart_data.
        Возвращаем словарь с ключами chart_type / chart_data / chart_full или None.
        """
        if not oox_index or not num:
            return None
        norm = _norm_caption_num(num)
        if not norm:
            return None
        try:
            figs = oox_index.get("figures", []) or []
        except Exception:
            return None

        for f in figs:
            cap_num = f.get("caption_num") or f.get("n")
            if _norm_caption_num(cap_num) != norm:
                continue

            chart = f.get("chart") or {}
            chart_type = chart.get("type") or chart.get("chart_type")
            chart_data = f.get("chart_data") or chart.get("chart_data") or chart.get("data")
            if not chart_data:
                continue

            return {
                "chart_type": chart_type,
                "chart_data": chart_data,
                "chart_full": chart,
            }
        return None

    in_sources = False

    pending_tbl_num: Optional[str] = None
    pending_tbl_tail: Optional[str] = None
    awaiting_tail: bool = False
    last_title_candidate: Optional[str] = None
    last_title_candidate_age: int = 999

    # NEW: последняя встреченная inline-ссылка на таблицу (см. табл. 6)
    last_table_ref_num: Optional[str] = None
    last_table_ref_age: int = 999  # счётчик «старения» ссылки

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
        nonlocal last_title_candidate, last_title_candidate_age
        t = last_title_candidate
        last_title_candidate = None
        last_title_candidate_age = 999
        return t

    def consider_flush_stale_candidate():
        nonlocal last_title_candidate, last_title_candidate_age, buf
        if last_title_candidate and last_title_candidate_age > 3:
            buf.append(last_title_candidate)
            last_title_candidate = None
            last_title_candidate_age = 999

    # NEW: helper to keep full table grid (all columns)
    def _table_to_grid(tbl: Table) -> List[List[str]]:
        grid: List[List[str]] = []
        try:
            for row in tbl.rows:
                r: List[str] = []
                for cell in row.cells:
                    r.append(_clean(cell.text or ""))
                grid.append(r)
        except Exception:
            return []
        return grid

    for block in _iter_block_items(doc):
        last_title_candidate_age = last_title_candidate_age + 1
        last_table_ref_age = last_table_ref_age + 1

        if isinstance(block, Paragraph):
            p_text = _clean(block.text)
            p_style_name = (block.style.name if block.style else "") or ""
            p_style = p_style_name.lower()

            # Heading*
            if re.match(r"^(heading|заголовок)\s*\d+", p_style):
                consider_flush_stale_candidate()
                flush_text_section()
                title = p_text or "Без названия"
                try:
                    m = re.search(r"(\d+)", p_style)
                    lvl = max(1, int(m.group(1))) if m else 1
                except Exception:
                    lvl = 1
                sec_id = _enter_heading(lvl, title)

                in_sources = _is_sources_title(title)
                if _is_appendix_title(title):
                    in_sources = False

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

            attrs_here = _paragraph_attrs(block)

            # НОВОЕ: псевдо-заголовки введения/заключения/литературы,
            # даже если они не в Heading-стиле (иначе intro теряется).
            if _is_title_candidate(p_text, attrs_here):
                is_intro = re.search(r"(?i)\b(введение|вступление)\b", p_text or "")
                is_concl = re.search(r"(?i)\b(заключение|выводы\s+и\s+рекомендации|общие\s+выводы)\b", p_text or "")
                is_refs = re.search(r"(?i)\b(список\s+литературы|библиограф(ия|ический)\s+список|источники|литература)\b", p_text or "")

                if is_intro or is_concl or is_refs:
                    consider_flush_stale_candidate()
                    flush_text_section()

                    title = p_text.strip()
                    lvl = 1
                    sec_id = _enter_heading(lvl, title)

                    in_sources = bool(is_refs)
                    if _is_appendix_title(title):
                        in_sources = False

                    sections.append({
                        "title": title,
                        "level": lvl,
                        "text": "",
                        "page": None,
                        "section_path": _current_scope_path(),
                        "element_type": "heading",
                        "attrs": {"style": "pseudo-heading-special", "section_id": sec_id, **attrs_here}
                    })
                    continue

            # НОВОЕ: текстовый псевдо-заголовок главы/раздела («ГЛАВА 1 ...», «Раздел 2.3 ...»)
            if CHAPTER_TITLE_RE.match(p_text) and _is_title_candidate(p_text, attrs_here):
                consider_flush_stale_candidate()
                flush_text_section()
                title = p_text or "Глава"
                # Для простоты считаем такие заголовки уровнем 1
                lvl = 1
                sec_id = _enter_heading(lvl, title)

                in_sources = _is_sources_title(title)
                if _is_appendix_title(title):
                    in_sources = False

                sections.append({
                    "title": title,
                    "level": lvl,
                    "text": "",
                    "page": None,
                    "section_path": _current_scope_path(),
                    "element_type": "heading",
                    "attrs": {"style": "pseudo-heading-chapter", "section_id": sec_id, **attrs_here}
                })
                continue

            # Appendix pseudo-heading
            if _is_appendix_title(p_text) and _is_title_candidate(p_text, attrs_here):
                consider_flush_stale_candidate()
                flush_text_section()
                title = p_text or "Приложение"
                lvl = max(1, cur_level + 1)
                sec_id = _enter_heading(lvl, title)
                in_sources = False
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

            # NEW: номерной псевдо-заголовок типа "2.1 Организация исследования"
            nh = _is_numbered_heading(p_text, attrs_here)
            if nh:
                consider_flush_stale_candidate()
                flush_text_section()

                num = nh["num"]
                title = p_text or nh["title"]
                lvl = nh["level"]

                sec_id = _enter_numbered_heading(lvl, title, num)

                in_sources = _is_sources_title(title)
                if _is_appendix_title(title):
                    in_sources = False

                sections.append({
                    "title": title,
                    "level": lvl,
                    "text": "",
                    "page": None,
                    "section_path": _current_scope_path(),
                    "element_type": "heading",
                    "attrs": {
                        "style": "pseudo-heading-numbered",
                        "section_id": sec_id,
                        "section_num": num,
                        **attrs_here
                    }
                })
                continue

            # Short centered candidate (potential caption tail)
            if _is_title_candidate(p_text, attrs_here) and not _is_sources_title(p_text):
                last_title_candidate = p_text
                last_title_candidate_age = 0
                continue

            # Sources section items
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

            # Captions
            kind, num, tail = _classify_caption(p_text or "")
            if kind == "table":
                pending_tbl_num = num
                if tail:
                    pending_tbl_tail = tail
                    awaiting_tail = False
                else:
                    pending_tbl_tail = None
                    awaiting_tail = True
                # позволяем хвосту быть не строго следующим блоком, а с одним "посредником" (пустой абзац и т.п.)
                if awaiting_tail and last_title_candidate and last_title_candidate_age <= 2:
                    pending_tbl_tail = consume_title_candidate()
                    awaiting_tail = False
                continue

            elif kind == "figure":
                if (not tail) and last_title_candidate and last_title_candidate_age <= 1:
                    tail = consume_title_candidate()

                consider_flush_stale_candidate()
                flush_text_section()
                figure_counter += 1

                # НОВОЕ: нормализованный номер подписи
                norm_num: Optional[str] = _norm_caption_num(num) if num else None
                fig_title = _compose_figure_title(num, tail)
                attrs_here_fig = _paragraph_attrs(block)

                # 1) изображения рядом
                imgs: List[str] = []
                imgs += _extract_paragraph_images(doc, block, DEFAULT_UPLOAD_DIR)
                neigh = _collect_neighbor_images(doc, block, window=CFG_FIG_WINDOW, uploads_dir=DEFAULT_UPLOAD_DIR)
                for pth in neigh:
                    if pth not in imgs:
                        imgs.append(pth)

                # 2) диаграмма рядом (по inline-XML в .docx)
                chart_type, chart_rows, chart_matrix = _extract_chart_data_from_paragraph(doc, block)
                if (not chart_rows) and (chart_matrix is None):
                    chart_type, chart_rows, chart_matrix = _collect_neighbor_chart_data(
                        doc, block, window=CFG_FIG_WINDOW
                    )

                # 2a) если есть OOXML-индекс — пробуем забрать chart_data оттуда по номеру рисунка
                oox_chart = _ooxml_chart_by_caption(num)

                # 3) фолбэк: если не нашли ни картинку, ни диаграмму — жёсткий look-ahead вперёд
                if not imgs and not chart_rows and chart_matrix is None and not (oox_chart and oox_chart.get("chart_data")):
                    imgs = _first_images_after(doc, block, CFG_FIG_LOOKAHEAD, DEFAULT_UPLOAD_DIR)

                attrs_here_fig["caption_num"] = num
                attrs_here_fig["caption_tail"] = tail
                attrs_here_fig["label"] = num
                attrs_here_fig["title"] = tail
                attrs_here_fig["origin"] = "docx"
                attrs_here_fig["numbers"] = _extract_numbers(p_text or "")

                if imgs:
                    attrs_here_fig.setdefault("images", imgs)

                # Приоритет: данные из OOXML-индекса, затем локальный разбор chart*.xml
                if oox_chart and oox_chart.get("chart_data"):
                    attrs_here_fig["chart_type"] = (oox_chart.get("chart_type") or chart_type or "Chart")
                    attrs_here_fig["chart_data"] = oox_chart["chart_data"]
                    attrs_here_fig["chart_origin"] = "ooxml_index"
                    # полный блок диаграммы из индекса — отдельным полем для отладки/расширений
                    if oox_chart.get("chart_full") is not None:
                        attrs_here_fig["chart_ooxml"] = oox_chart["chart_full"]
                elif chart_rows:
                    attrs_here_fig["chart_type"] = chart_type or "Chart"
                    attrs_here_fig["chart_data"] = chart_rows
                    attrs_here_fig["chart_origin"] = "docx_chart_xml"

                if chart_matrix is not None:
                    attrs_here_fig["chart_matrix"] = chart_matrix

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

                pending_tbl_num = None
                pending_tbl_tail = None
                awaiting_tail = False
                continue

            # Sources pseudo-heading
            if _is_sources_title(p_text) and _is_title_candidate(p_text, attrs_here):
                consider_flush_stale_candidate()
                flush_text_section()
                title = p_text or "Список литературы"
                lvl = max(1, cur_level + 1)
                sec_id = _enter_heading(lvl, title)
                in_sources = True
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

            # Orphan inline images/charts (no explicit text)
            imgs_inline = _extract_paragraph_images(doc, block, DEFAULT_UPLOAD_DIR)
            chart_type_adhoc, chart_rows_adhoc, chart_matrix_adhoc = _extract_chart_data_from_paragraph(doc, block)
            if (imgs_inline or chart_rows_adhoc or chart_matrix_adhoc) and not _clean(p_text):
                consider_flush_stale_candidate()
                flush_text_section()
                figure_counter += 1
                tail_auto = consume_title_candidate() if last_title_candidate and last_title_candidate_age <= 1 else None
                fig_title = _compose_figure_title(None, tail_auto or "Рисунок без подписи")

                attrs_here_fig = _paragraph_attrs(block)
                if imgs_inline:
                    attrs_here_fig["images"] = imgs_inline
                if chart_rows_adhoc:
                    attrs_here_fig["chart_type"] = chart_type_adhoc or "Chart"
                    attrs_here_fig["chart_data"] = chart_rows_adhoc
                    attrs_here_fig["chart_origin"] = "docx_chart_xml"
                if chart_matrix_adhoc is not None:
                    attrs_here_fig["chart_matrix"] = chart_matrix_adhoc
                attrs_here_fig["section_scope"] = _current_scope_path()
                attrs_here_fig["section_scope_id"] = _current_scope_id()
                attrs_here_fig["anchor"] = f"fig-{figure_counter}"

                sections.append({
                    "title": fig_title,
                    "level": max(1, cur_level + 1),
                    "text": p_text or "",
                    "page": None,
                    "section_path": _hpath(heading_stack_ids, heading_stack_titles + [fig_title]),
                    "element_type": "figure",
                    "attrs": attrs_here_fig
                })
                continue

            # Обычный текст
            consider_flush_stale_candidate()
            if p_text:
                last_para_attrs = _paragraph_attrs(block)
                buf.append(p_text)

                # NEW: ловим inline-ссылку на таблицу: «см. табл. 6», «см. таблицу 3» и т.п.
                m_ref = re.search(
                    r"(?:см\.?\s*)?табл(?:ица)?\s*\.?\s*(?:№\s*)?(\d+(?:[.,]\d+)*)",
                    p_text,
                    re.IGNORECASE,
                )
                if m_ref:
                    # нормализуем номер как в подписи («6», «2.3» и т.п.)
                    last_table_ref_num = _norm_caption_num(m_ref.group(1))
                    last_table_ref_age = 0

            continue

        # Таблица
        if isinstance(block, Table):
            consider_flush_stale_candidate()
            flush_text_section()
            table_counter += 1

            caption_source = "none"
            num_final: Optional[str] = None      # raw-строка из подписи/ссылки
            tail_final: Optional[str] = None

            if pending_tbl_num:
                # Явная подпись «Таблица N …» — приоритетнее всего
                num_final = pending_tbl_num
                if pending_tbl_tail:
                    tail_final = pending_tbl_tail
                    caption_source = "single_line" if not awaiting_tail else "two_lines_after"
                elif last_title_candidate and last_title_candidate_age <= 2:
                    tail_final = consume_title_candidate()
                    caption_source = "two_lines_after"

            elif last_table_ref_num and last_table_ref_age <= 3:
                # NEW: нет подписи, но недавно была inline-ссылка «см. табл. N»
                num_final = last_table_ref_num
                caption_source = "by_inline_ref"

            elif last_title_candidate and last_title_candidate_age <= 2:
                # Короткая строка над таблицей — хвост подписи без номера
                tail_final = consume_title_candidate()
                caption_source = "two_lines_before"

            # --- НОВОЕ: нормализованный номер для поиска/ID ---
            norm_num: Optional[str] = _norm_caption_num(num_final) if num_final else None

            t_text = _table_to_text(block) or "(пустая таблица)"

            # NEW: сохраняем "сырой" grid таблицы (все колонки), чтобы дальше нигде не отрезалось
            grid = _table_to_grid(block)
            if grid:
                # уберём полностью пустые хвостовые колонки одинаково для всех строк (часто артефакты)
                max_cols = max(len(r) for r in grid)
                norm = [r + [""] * (max_cols - len(r)) for r in grid]
                while max_cols > 0 and all((nr[max_cols - 1] or "").strip() == "" for nr in norm):
                    max_cols -= 1
                if max_cols > 0:
                    norm = [nr[:max_cols] for nr in norm]
                grid = norm

            attrs_tbl = _table_attrs(block) | {"numbers": _extract_numbers(t_text)}

            # NEW: порядковый номер таблицы в документе (для fallback-поиска)
            attrs_tbl["order_index"] = table_counter

            # NEW: кладём полный вид в attrs (для TablesRaw/LLM)
            if grid:
                attrs_tbl["table_grid"] = grid
                attrs_tbl["table_rows"] = len(grid)
                attrs_tbl["table_cols"] = max((len(r) for r in grid), default=0)
                attrs_tbl["table_tsv"] = "\n".join("\t".join(r) for r in grid)

            if not tail_final and attrs_tbl.get("header_preview"):
                tail_final = attrs_tbl["header_preview"]
                if caption_source == "none":
                    caption_source = "header_row"

            # Для отображения можно использовать нормализованный номер; raw сохраняем отдельно
            t_title = _compose_table_title(norm_num, tail_final)

            attrs_tbl["caption_raw"] = num_final
            attrs_tbl["caption_num"] = norm_num
            attrs_tbl["caption_tail"] = tail_final
            attrs_tbl["label"] = norm_num
            attrs_tbl["title"] = tail_final
            attrs_tbl["caption_source"] = caption_source
            attrs_tbl["section_scope"] = _current_scope_path()
            attrs_tbl["section_scope_id"] = _current_scope_id()
            attrs_tbl["anchor"] = f"tbl-{(norm_num or table_counter)}"

            # Явно помечаем происхождение таблицы
            attrs_tbl["origin"] = "docx"
            attrs_tbl["source"] = "docx_table"
            attrs_tbl["is_table"] = True

            sections.append({
                "title": t_title or f"Таблица {table_counter}",
                "level": max(1, cur_level + 1),
                "text": t_text,
                "page": None,
                "section_path": _hpath(
                    heading_stack_ids,
                    heading_stack_titles + [t_title or f"Таблица {table_counter}"]
                ),
                "element_type": "table",
                "attrs": attrs_tbl
            })

            # изображения внутри таблицы → отдельные figure
            tbl_imgs = _extract_table_images(doc, block, DEFAULT_UPLOAD_DIR)
            for k, pth in enumerate(tbl_imgs or [], 1):
                figure_counter += 1
                ft = _compose_figure_title(None, f"Изображение в таблице {attrs_tbl.get('caption_num') or table_counter} ({k})")
                sections.append({
                    "title": ft,
                    "level": max(1, cur_level + 1),
                    "text": "",
                    "page": None,
                    "section_path": _hpath(heading_stack_ids, heading_stack_titles + [ft]),
                    "element_type": "figure",
                    "attrs": {
                        "images": [pth],
                        "section_scope": _current_scope_path(),
                        "section_scope_id": _current_scope_id(),
                        "anchor": f"tbl-{(attrs_tbl.get('caption_num') or table_counter)}-img-{k}",
                    }
                })

            # встроенная диаграмма в таблице → отдельный figure
            chart_type_t, chart_rows_t, chart_matrix_t = _extract_chart_data_from_table(doc, block)
            if chart_rows_t or chart_matrix_t is not None:
                figure_counter += 1
                ft = _compose_figure_title(
                    None,
                    attrs_tbl.get("caption_tail") or attrs_tbl.get("header_preview") or "Рисунок без подписи"
                )
                fig_attrs: Dict[str, Any] = {
                    "section_scope": _current_scope_path(),
                    "section_scope_id": _current_scope_id(),
                    "anchor": f"fig-{figure_counter}",
                }
                if chart_rows_t:
                    fig_attrs["chart_type"] = chart_type_t or "Chart"
                    fig_attrs["chart_data"] = chart_rows_t
                    fig_attrs["chart_origin"] = "docx_chart_xml"
                if chart_matrix_t is not None:
                    fig_attrs["chart_matrix"] = chart_matrix_t

                sections.append({
                    "title": ft,
                    "level": max(1, cur_level + 1),
                    "text": "",
                    "page": None,
                    "section_path": _hpath(heading_stack_ids, heading_stack_titles + [ft]),
                    "element_type": "figure",
                    "attrs": fig_attrs,
                })

            pending_tbl_num = None
            pending_tbl_tail = None
            awaiting_tail = False
            last_title_candidate = None
            last_title_candidate_age = 999

            # NEW: inline-ссылка «см. табл. N» сработала → обнуляем
            last_table_ref_num = None
            last_table_ref_age = 999

            continue

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
            tops = [float(z.get("top", y)) for z in ws]
            bottoms = [float(z.get("bottom", y+10)) for z in ws]
            y_center = (min(tops) + max(bottoms)) / 2.0
            out.append((y_center, text))
    out.sort(key=lambda x: x[0])
    return out

def _fitz_images_with_bbox(pg) -> List[Dict[str, Any]]:
    """Возвращает список изображений страницы PyMuPDF с bbox и xref."""
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
                        "yc": (float(y0) + float(y1)) / 2.0  # в системе fitz (ноль внизу!)
                    })
    except Exception:
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

    pdf_fitx = None
    if fitz is not None and CFG_PDF_EXTRACT:
        try:
            pdf_fitx = fitz.open(path)
        except Exception:
            pdf_fitx = None

    xref_cache: Dict[int, str] = {}

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
            fp = str(Path(save_bytes(data, safe_filename(f"pdf_img_{h[:12]}{ext}"), DEFAULT_UPLOAD_DIR)).resolve())
            xref_cache[xref] = fp
            return fp
        except Exception:
            return None

    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            # 1) текст страницы
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

            # 2) простые таблицы
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

            # 3) подписи к рисункам + ближайшие изображения (вертикальная близость)
            fig_caps: List[Tuple[str, str, float]] = []
            for y, line_text in lines:
                k, n, t = _classify_caption(line_text.strip())
                if k == "figure":
                    fig_caps.append((n or "", t or "", y))

            images_with_bbox: List[Dict[str, Any]] = []
            if pdf_fitx is not None:
                try:
                    pg = pdf_fitx[i - 1]
                    images_with_bbox = _fitz_images_with_bbox(pg)
                except Exception:
                    images_with_bbox = []

            page_height = float(page.height or 0.0)
            for im in images_with_bbox:
                try:
                    im["yc_top"] = page_height - float(im.get("yc", 0.0))
                except Exception:
                    im["yc_top"] = None

            used_xrefs = set()
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

                    chosen_path: Optional[str] = None
                    if images_with_bbox:
                        best = None
                        for im in images_with_bbox:
                            if im.get("yc_top") is None:
                                continue
                            dist = abs(float(im["yc_top"]) - float(ycap_top))
                            if (best is None) or (dist < best[0]):
                                best = (dist, im)
                        if best is not None and best[0] <= max(0, CFG_PDF_CAPTION_DY):  # ~14 см
                            xref = int(best[1]["xref"])
                            chosen_path = _save_xref(xref)

                    if chosen_path:
                        attrs["images"] = [chosen_path]
                        attrs["origin"] = "pdf"                      # << добавлено
                        attrs["image_origin"] = "pdf_xref"           # << добавлено
                        attrs["pdf_page"] = i    
                        try:
                            used_xrefs.add(int(best[1]["xref"]))
                        except Exception:
                            pass
                    else:
                        # NEW: векторный рендер вокруг подписи
                        if pdf_fitx is not None and CFG_PDF_VECTOR_RASTERIZE:
                            try:
                                pg = pdf_fitx[i - 1]
                                vrects = _fitz_vector_rects(pg)
                            except Exception:
                                vrects = []
                            if vrects:
                                # фильтруем векторные bbox'ы по вертикальной близости к подписи
                                near = [r for r in vrects if _bbox_vdist_generic(
                                    r, (0.0, ycap_top - 1.0, float(pg.rect.x1), ycap_top + 1.0)
                                ) <= max(0, CFG_PDF_CAPTION_DY)]
                                cluster = _rect_union(near)
                                if cluster:
                                    png = _render_clip_png(
                                        pg, cluster,
                                        dpi_base=CFG_PDF_VECTOR_DPI,
                                        min_w_px=CFG_PDF_VECTOR_MIN_WIDTH_PX,
                                        max_dpi=CFG_PDF_VECTOR_MAX_DPI,
                                        pad_px=CFG_PDF_VECTOR_PAD_PX
                                    )
                                    if png:
                                        h = sha256_bytes(png)
                                        saved = str(Path(save_bytes(png, safe_filename(f"pdf_vec_{h[:12]}.png"), DEFAULT_UPLOAD_DIR)).resolve())
                                        attrs["images"] = [saved]
                                        attrs["origin"] = "pdf"                      # << добавлено
                                        attrs["image_origin"] = "pdf_vector_render"  # << добавлено
                                        attrs["pdf_page"] = i 

                    out.append({
                        "title": title,
                        "level": 2,
                        "text": title,
                        "page": i,
                        "section_path": f"Стр. {i} / {title}",
                        "element_type": "figure",
                        "attrs": attrs
                    })

            if images_with_bbox:
                for idx_im, im in enumerate(images_with_bbox, 1):
                    xref = int(im.get("xref", 0)) if im.get("xref") is not None else None
                    if xref and xref in used_xrefs:
                        continue
                    saved = _save_xref(xref) if xref else None
                    if not saved:
                        continue
                    title = _compose_figure_title(None, f"Изображение без подписи {i}.{idx_im}")
                    out.append({
                        "title": title,
                        "level": 2,
                        "text": title,
                        "page": i,
                        "section_path": f"Стр. {i} / {title}",
                        "element_type": "figure",
                        "attrs": {
                            "images": [saved],
                            "origin": "pdf",                 # << добавлено
                            "image_origin": "pdf_xref",      # << добавлено
                            "pdf_page": i                    # << добавлено
                        }
                    })
            
            # после блока с fig_caps / images_with_bbox
            if CFG_PDF_VECTOR_RASTERIZE and not fig_caps:
                try:
                    pg = pdf_fitx[i - 1]
                    vrects = _fitz_vector_rects(pg)
                except Exception:
                    vrects = []
                if vrects:
                    # берём крупный кластер на странице как эвристический «рисунок без подписи»
                    # (срезаем мелочь: шире 40pt и выше 40pt)
                    big = [r for r in vrects if (r[2]-r[0]) >= 40 and (r[3]-r[1]) >= 40]
                    cluster = _rect_union(big)
                    if cluster:
                        png = _render_clip_png(pg, cluster,
                                            dpi_base=CFG_PDF_VECTOR_DPI,
                                            min_w_px=CFG_PDF_VECTOR_MIN_WIDTH_PX,
                                            max_dpi=CFG_PDF_VECTOR_MAX_DPI,
                                            pad_px=CFG_PDF_VECTOR_PAD_PX)
                        if png:
                            h = sha256_bytes(png)
                            saved = str(Path(save_bytes(png, safe_filename(f"pdf_vec_{h[:12]}.png"), DEFAULT_UPLOAD_DIR)).resolve())
                            title = _compose_figure_title(None, f"Изображение без подписи {i}.V")
                            out.append({
                                "title": title,
                                "level": 2,
                                "text": title,
                                "page": i,
                                "section_path": f"Стр. {i} / {title}",
                                "element_type": "figure",
                                "attrs": {
                                    "images": [saved],
                                    "origin": "pdf",                     # << добавлено
                                    "image_origin": "pdf_vector_render", # << добавлено
                                    "pdf_page": i                        # << добавлено
                                }
                            })


            # 4) эвристика списка литературы на странице
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

    try:
        if pdf_fitx is not None:
            pdf_fitx.close()
    except Exception:
        pass

    return out

# -------- .doc -> .docx (Aspose / LibreOffice / Word COM) --------

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
