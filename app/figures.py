# app/figures.py
# -*- coding: utf-8 -*-
"""
Извлечение иллюстраций из документов и индексирование “Рисунок N”.
Поддержка:
  - DOCX (python-docx обязателен)
  - PDF (через PyMuPDF = fitz; если нет — тихо пропускаем)

ENV:
  FIGURES_ROOT=./runtime/figures           # корневая папка для всех наборов рисунков
  DOCX_EXTRACT_IMAGES=true                 # извлекать из DOCX
  PDF_EXTRACT_IMAGES=true                  # извлекать из PDF (если есть PyMuPDF)
  DOCX_PDF_FALLBACK=true                   # если из DOCX 0 штук — конвертнуть в PDF и попробовать PDF-путь
  DOCX_TO_PDF_BACKEND=auto                 # auto|docx2pdf|libreoffice|off
  FIG_NEIGHBOR_WINDOW=10                   # сколько абзацев искать подпись вокруг картинки (DOCX)
  PDF_CAPTION_MAX_DISTANCE_PX=250          # макс. вертикальная дистанция (px) от картинки/вектора до подписи на странице
  PDF_MIN_IMAGE_AREA=20000                 # отсечь мелкие пиктограммы (минимальная площадь в пикселях)
  FIG_CLEANUP_ON_NEW_DOC=true              # очищать FIGURES_ROOT при новом документе

  # Новые флаги для векторных диаграмм и поведения DOCX→PDF
  PDF_VECTOR_RASTERIZE=true               # включить рендер векторных примитивов рядом с подписью
  PDF_VECTOR_DPI=360                      # базовый DPI для растра векторной вырезки
  PDF_VECTOR_MIN_WIDTH_PX=1200            # гарантировать минимальную ширину результата (поднимем zoom при необходимости)
  PDF_VECTOR_PAD_PX=16                    # паддинг вокруг объединённого bbox при рендере
  PDF_VECTOR_MAX_DPI=600                  # не повышать DPI выше этого
  DOCX_PDF_ALWAYS=false                   # всегда прогонять DOCX→PDF ветку (вдобавок к обычному извлечению из DOCX)

Публичный API:
  - index_document(path: str) -> dict
  - load_index(fig_dir: str) -> dict
  - query_by_number(index, "4"|"2.1") -> list[dict]
  - query_by_caption(index, text, top_k=3) -> list[dict]
  - query_by_chapter(index, "название главы", top_k=3) -> list[dict]
  - find_figure(index, number=None, caption_query=None, chapter=None) -> list[dict]

Каждая запись фигуры имеет поля:
  id, doc_id, rel_path, abs_path, order, number, title, caption, section, anchors[]
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# --- Логер ---
log = logging.getLogger("figures")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    log.addHandler(h)
log.setLevel(logging.INFO)

# --- ENV helpers ---

def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}

def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    try:
        return int(val) if val is not None else default
    except Exception:
        return default

def _env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return default if v is None else str(v).strip()

FIG_ROOT = Path(os.getenv("FIGURES_ROOT", "./runtime/figures")).resolve()
DOCX_ON = _env_bool("DOCX_EXTRACT_IMAGES", True)
PDF_ON = _env_bool("PDF_EXTRACT_IMAGES", True)
DOCX_FALLBACK = _env_bool("DOCX_PDF_FALLBACK", True)
DOCX_PDF_BACKEND = _env_str("DOCX_TO_PDF_BACKEND", "auto").lower()
NEIGH = _env_int("FIG_NEIGHBOR_WINDOW", 10)
PDF_CAPTION_MAX_DY = _env_int("PDF_CAPTION_MAX_DISTANCE_PX", 250)
PDF_MIN_IMAGE_AREA = _env_int("PDF_MIN_IMAGE_AREA", 20000)
CLEAN_ON_NEW = _env_bool("FIG_CLEANUP_ON_NEW_DOC", True)

# Новые ENV
PDF_VECTOR_ON = _env_bool("PDF_VECTOR_RASTERIZE", True)
PDF_VECTOR_DPI = _env_int("PDF_VECTOR_DPI", 360)
PDF_VECTOR_MIN_W = _env_int("PDF_VECTOR_MIN_WIDTH_PX", 1200)
PDF_VECTOR_PAD_PX = _env_int("PDF_VECTOR_PAD_PX", 16)
PDF_VECTOR_MAX_DPI = _env_int("PDF_VECTOR_MAX_DPI", 600)
DOCX_PDF_ALWAYS = _env_bool("DOCX_PDF_ALWAYS", False)

# --- Утиль ---

_rx_ws = re.compile(r"\s+")
_rx_not_word = re.compile(r"[^0-9A-Za-zА-Яа-яЁё\-_. ]+", re.UNICODE)

def _normalize_text(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("\xa0", " ")
    s = _rx_ws.sub(" ", s)
    return s

def _slugify(s: str, max_len: int = 80) -> str:
    s = _normalize_text(s)
    s = _rx_not_word.sub("", s)
    s = s.strip().lower()
    s = s.replace(" ", "-")
    return s[:max_len] if s else "untitled"

def _short_hash(b: bytes, n: int = 10) -> str:
    import hashlib as _hh
    return _hh.sha1(b).hexdigest()[:n]

def _file_hash(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _doc_id_for(path: Path) -> str:
    stem = _slugify(path.stem)
    h = _file_hash(path)[:8]
    return f"{stem}-{h}"

def _ensure_clean_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _maybe_cleanup_root():
    if CLEAN_ON_NEW and FIG_ROOT.exists():
        shutil.rmtree(FIG_ROOT, ignore_errors=True)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)

# --- Разбор подписи "Рисунок N" ---

_CAPTION_PREFIXES = [
    r"рисунок", r"рис\.", r"рис",
    r"figure", r"fig\.", r"fig"
]
_rx_caption = re.compile(
    rf"^\s*(?P<prefix>(?:{'|'.join(_CAPTION_PREFIXES)}))\s*"
    r"(?P<num>\d+(?:\.\d+)*)\s*[:\-—]?\s*(?P<title>.*)$",
    re.IGNORECASE | re.UNICODE
)

def parse_caption_line(text: str) -> Tuple[Optional[str], Optional[str]]:
    t = _normalize_text(text)
    m = _rx_caption.match(t)
    if not m:
        return None, None
    num = (m.group("num") or "").replace(",", ".").strip()
    title = _normalize_text(m.group("title"))
    return num, (title or None)

def looks_like_caption(text: str) -> bool:
    num, _ = parse_caption_line(text)
    if num:
        return True
    t = _normalize_text(text).lower()
    return any(t.startswith(p) for p in ["рис", "figure", "fig"])

# --- Модель данных ---

@dataclass
class FigureRecord:
    id: str
    doc_id: str
    rel_path: str
    abs_path: str
    order: int
    number: Optional[str] = None
    title: Optional[str] = None
    caption: Optional[str] = None
    section: Optional[str] = None
    anchors: Optional[List[str]] = None

    def build_anchors(self) -> List[str]:
        out: List[str] = []
        if self.number:
            n = self.number
            out += [f"рисунок {n}", f"рис. {n}", f"figure {n}", f"fig {n}", n]
        if self.title:
            out.append(_normalize_text(self.title).lower())
        if self.caption:
            out.append(_normalize_text(self.caption).lower())
        if self.section:
            out.append(_normalize_text(self.section).lower())
        seen = set()
        uniq = []
        for a in out:
            a = a.strip().lower()
            if a and a not in seen:
                uniq.append(a)
                seen.add(a)
        return uniq

# --- DOCX helpers ---

def _iter_docx_paragraphs(document):
    """Итерируем параграфы верхнего уровня и из таблиц, сохраняя линейный порядок."""
    from docx.text.paragraph import Paragraph
    from docx.table import _Cell, Table
    from docx.document import Document as _Document

    def iter_block_items(parent):
        if isinstance(parent, _Document):
            parent_elm = parent.element.body
        elif isinstance(parent, _Cell):
            parent_elm = parent._tc
        else:
            return
        for child in parent_elm.iterchildren():
            if child.tag.endswith('}p'):
                yield Paragraph(child, parent)
            elif child.tag.endswith('}tbl'):
                table = Table(child, parent)
                for row in table.rows:
                    for cell in row.cells:
                        for item in iter_block_items(cell):
                            yield item

    for item in iter_block_items(document):
        yield item

def _paragraph_style_name(p) -> str:
    try:
        return (p.style.name or "").strip()
    except Exception:
        return ""

def _is_heading_style(name: str) -> Optional[int]:
    n = (name or "").strip().lower()
    for prefix in ("heading ", "заголовок "):
        if n.startswith(prefix):
            try:
                lvl = int(n.split()[-1])
                if 1 <= lvl <= 6:
                    return lvl
            except Exception:
                pass
    return None

def _extract_images_from_run(run, doc_part) -> List[Tuple[bytes, str]]:
    blobs: List[Tuple[bytes, str]] = []
    drawing_elems = run._r.xpath(".//w:drawing//a:blip")
    for blip in drawing_elems:
        rid = blip.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed")
        if not rid:
            continue
        part = doc_part.related_parts.get(rid)
        if not part:
            continue
        blob = part.blob
        ct = getattr(part, "content_type", "") or ""
        ext = ".png"
        if "jpeg" in ct:
            ext = ".jpg"
        elif "png" in ct:
            ext = ".png"
        elif "gif" in ct:
            ext = ".gif"
        elif "tiff" in ct or "tif" in ct:
            ext = ".tif"
        blobs.append((blob, ext))
    return blobs

def _nearest_caption(paragraphs: List[Any], idx: int, window: int) -> Tuple[Optional[str], Optional[str]]:
    best = None
    best_dist = 10**9
    for delta in range(0, window + 1):
        for sign in (-1, 1) if delta > 0 else (1,):
            j = idx + sign * delta
            if j < 0 or j >= len(paragraphs):
                continue
            t = _normalize_text(paragraphs[j].text or "")
            if looks_like_caption(t):
                d = abs(j - idx)
                if d < best_dist:
                    best = t
                    best_dist = d
        if best is not None:
            break
    return best, None

def extract_docx(docx_path: Path, out_dir: Path, neighbor_window: int = NEIGH) -> List[FigureRecord]:
    try:
        from docx import Document
    except Exception as e:
        log.error("python-docx не установлен: %s", e)
        return []

    doc = Document(str(docx_path))
    paragraphs = list(_iter_docx_paragraphs(doc))

    section_path = []
    section_path_by_idx: Dict[int, str] = {}

    for i, p in enumerate(paragraphs):
        lvl = _is_heading_style(_paragraph_style_name(p))
        if lvl:
            if len(section_path) >= lvl:
                section_path = section_path[:lvl - 1]
            section_path.append(_normalize_text(p.text))
        section_path_by_idx[i] = " > ".join([s for s in section_path if s])

    figures: List[FigureRecord] = []
    order = 0

    for i, p in enumerate(paragraphs):
        all_blobs: List[Tuple[bytes, str]] = []
        for run in p.runs:
            all_blobs.extend(_extract_images_from_run(run, doc.part))
        if not all_blobs:
            continue

        caption_text, _ = _nearest_caption(paragraphs, i, neighbor_window)
        fig_number, fig_title = (None, None)
        if caption_text:
            n, ttl = parse_caption_line(caption_text)
            fig_number, fig_title = n, ttl

        for blob, ext in all_blobs:
            order += 1
            shash = _short_hash(blob, 10)
            if fig_number:
                fname = f"fig-{fig_number.replace('.', '-')}-{shash}{ext}"
            else:
                fname = f"fig-{order:03d}-{shash}{ext}"
            dst = out_dir / fname
            with open(dst, "wb") as f:
                f.write(blob)

            rec = FigureRecord(
                id=f"{fig_number or order}-{shash}",
                doc_id=out_dir.name,
                rel_path=str(dst.relative_to(out_dir)),
                abs_path=str(dst),
                order=order,
                number=fig_number,
                title=fig_title,
                caption=caption_text,
                section=section_path_by_idx.get(i) or None,
            )
            rec.anchors = rec.build_anchors()
            figures.append(rec)

    return figures

# --- DOCX → PDF конвертация (постоянный кэш) ---

def _wait_until_readable(path: Path, timeout_sec: float = 8.0) -> bool:
    """Ждём, пока файл можно открыть на чтение и размер стабилен (для Windows/COM)."""
    t0 = time.time()
    last_size = -1
    while time.time() - t0 < timeout_sec:
        try:
            if path.exists():
                sz = path.stat().st_size
                if sz > 0 and sz == last_size:
                    with open(path, "rb"):
                        return True
                last_size = sz
        except Exception:
            pass
        time.sleep(0.15)
    return path.exists()

def _docx_to_pdf_persistent(docx_path: Path, out_dir: Path) -> Optional[Path]:
    """
    Конвертирует DOCX → PDF в ПЕРСИСТЕНТНЫЙ файл в каталоге набора рисунков:
      <out_dir>/source.pdf
    Без TemporaryDirectory(), чтобы избежать WinError 32 при удалении.
    """
    out_pdf = out_dir / "source.pdf"
    # заранее уберём старый, если был
    try:
        if out_pdf.exists():
            out_pdf.unlink()
    except Exception:
        # если занят — перезапишем поверх
        pass

    backend = DOCX_PDF_BACKEND

    def _try_docx2pdf() -> bool:
        try:
            from docx2pdf import convert
        except Exception as e:
            log.info("docx2pdf недоступен: %s", e)
            return False
        try:
            convert(str(docx_path), str(out_pdf))
            ok = _wait_until_readable(out_pdf, timeout_sec=12.0)
            return ok and out_pdf.exists()
        except Exception as e:
            log.info("docx2pdf не смог сконвертировать: %s", e)
            return False

    def _try_libreoffice() -> bool:
        try:
            cmd = [
                "soffice", "--headless",
                "--convert-to", "pdf",
                "--outdir", str(out_dir),
                str(docx_path),
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
            if res.returncode != 0:
                log.info("LibreOffice вернул код %s: %s", res.returncode, res.stderr.decode(errors="ignore"))
            ok = _wait_until_readable(out_pdf, timeout_sec=8.0)
            return ok and out_pdf.exists()
        except Exception as e:
            log.info("LibreOffice (soffice) не смог сконвертировать: %s", e)
            return False

    if DOCX_PDF_BACKEND == "off":
        return None

    if backend in {"auto", "docx2pdf"}:
        if _try_docx2pdf():
            log.info("DOCX→PDF (docx2pdf): %s", out_pdf)
            return out_pdf
        if backend == "docx2pdf":
            return None

    if backend in {"auto", "libreoffice"}:
        if _try_libreoffice():
            log.info("DOCX→PDF (LibreOffice): %s", out_pdf)
            return out_pdf

    return None

# --- PDF (через PyMuPDF) ---

def _page_text_blocks(page) -> List[Tuple[Tuple[float, float, float, float], str]]:
    """[(bbox, text)] для текстовых блоков страницы."""
    blocks: List[Tuple[Tuple[float, float, float, float], str]] = []
    try:
        d = page.get_text("dict")
    except Exception:
        txt = page.get_text("text") or ""
        if txt.strip():
            blocks.append(((0, 0, 0, 0), txt))
        return blocks

    for b in d.get("blocks", []):
        if b.get("type") == 0:
            bbox = tuple(b.get("bbox", (0, 0, 0, 0)))
            parts = []
            for line in b.get("lines", []):
                for span in line.get("spans", []):
                    parts.append(span.get("text", ""))
                parts.append("\n")
            text = _normalize_text("".join(parts))
            if text.strip():
                blocks.append((bbox, text))
    return blocks

def _page_image_blocks(page) -> List[Tuple[int, Tuple[float, float, float, float]]]:
    """[(xref, bbox)] для 'видимых' изображений из словаря текста."""
    items: List[Tuple[int, Tuple[float, float, float, float]]] = []
    try:
        d = page.get_text("dict")
    except Exception:
        return items
    for b in d.get("blocks", []):
        if b.get("type") == 1:
            xref = b.get("xref") or b.get("image")
            if not xref:
                continue
            bbox = tuple(b.get("bbox", (0, 0, 0, 0)))
            try:
                xref = int(xref)
            except Exception:
                continue
            items.append((xref, bbox))
    return items

def _bbox_vdist(img_bbox, cap_bbox) -> float:
    """Вертикальная дистанция между картинкой/вектором и подписью."""
    ix0, iy0, ix1, iy1 = img_bbox
    cx0, cy0, cx1, cy1 = cap_bbox
    # непересекаются: расстояние по вертикали
    if cy0 >= iy1:
        return abs(cy0 - iy1)
    if cy1 <= iy0:
        return abs(iy0 - cy1)
    # пересекаются по вертикали
    return 0.0

# --- Вспомогательные для вектора ---

def _rect_union(rects: List[Tuple[float, float, float, float]]) -> Optional[Tuple[float, float, float, float]]:
    if not rects:
        return None
    x0 = min(r[0] for r in rects)
    y0 = min(r[1] for r in rects)
    x1 = max(r[2] for r in rects)
    y1 = max(r[3] for r in rects)
    return (x0, y0, x1, y1)

def _rect_expand(rect: Tuple[float, float, float, float],
                 pad_pt: float,
                 page_rect: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x0, y0, x1, y1 = rect
    px0, py0, px1, py1 = page_rect
    return (
        max(px0, x0 - pad_pt),
        max(py0, y0 - pad_pt),
        min(px1, x1 + pad_pt),
        min(py1, y1 + pad_pt),
    )

def _page_vector_rects(page) -> List[Tuple[float, float, float, float]]:
    """Собираем bbox'ы векторных примитивов (линиии/path/rect), доступные через get_drawings()."""
    rects: List[Tuple[float, float, float, float]] = []
    try:
        drawings = page.get_drawings()
    except Exception:
        return rects
    for d in drawings or []:
        r = d.get("rect")
        if r:
            # PyMuPDF Rect → tuple
            rects.append((float(r.x0), float(r.y0), float(r.x1), float(r.y1)))
    return rects

def _render_clip_to_png(page, clip_rect: Tuple[float, float, float, float],
                        dpi_base: int,
                        min_width_px: int,
                        max_dpi: int,
                        pad_px: int) -> Optional[bytes]:
    """
    Рендер вырезки страницы в PNG с гарантированной минимальной шириной.
    """
    try:
        import fitz
    except Exception:
        return None

    # Ограничим клип пейдж-ректом и добавим паддинг
    page_rect = (float(page.rect.x0), float(page.rect.y0), float(page.rect.x1), float(page.rect.y1))
    zoom = max(1.0, float(dpi_base) / 72.0)
    pad_pt = pad_px / zoom  # pad в поинтах
    clip = _rect_expand(clip_rect, pad_pt, page_rect)

    width_pt = max(1.0, clip[2] - clip[0])
    # увеличим zoom, если не добираем минимальную ширину в px
    needed_zoom = float(min_width_px) / width_pt
    zoom = max(zoom, needed_zoom)
    # но не выше max_dpi
    zoom = min(zoom, float(max_dpi) / 72.0)

    try:
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, clip=fitz.Rect(*clip), alpha=True)
        img_bytes = pix.tobytes("png")
        return img_bytes
    except Exception as e:
        log.debug("Не удалось отрендерить вырезку: %s", e)
        return None

def extract_pdf(pdf_path: Path, out_dir: Path) -> List[FigureRecord]:
    try:
        import fitz  # PyMuPDF
    except Exception:
        log.info("PyMuPDF не найден — пропускаю извлечение изображений из PDF.")
        return []

    fig_records: List[FigureRecord] = []
    doc = fitz.open(str(pdf_path))
    order = 0
    seen_xref_on_page: set[Tuple[int, int]] = set()

    for page_index in range(len(doc)):
        page = doc[page_index]

        # --- подписи
        text_blocks = _page_text_blocks(page)
        captions: List[Tuple[int, Tuple[float, float, float, float], str, str, Optional[str]]] = []
        cap_idx = 0
        for bbox, txt in text_blocks:
            num, title = parse_caption_line(txt)
            if num:
                captions.append((cap_idx, bbox, txt, num, title))
                cap_idx += 1

        # --- растровые изображения
        img_blocks = _page_image_blocks(page)
        images = page.get_images(full=True)

        xref_meta: Dict[int, Tuple[int, int]] = {}
        for img in images:
            xref = img[0]
            try:
                w, h = img[2], img[3]
            except Exception:
                w, h = 0, 0
            xref_meta[xref] = (w, h)

        matched_caption_ids: set[int] = set()

        # Случай, когда dict не вернул имидж-блоки — вытянем все xref'ы
        if not img_blocks:
            for img in images:
                xref = img[0]
                if (page_index, xref) in seen_xref_on_page:
                    continue
                seen_xref_on_page.add((page_index, xref))
                w, h = xref_meta.get(xref, (0, 0))
                if w * h and (w * h) < PDF_MIN_IMAGE_AREA:
                    continue
                try:
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n > 4:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    ext = ".png" if pix.alpha or pix.n in (4, 5) else ".jpg"
                    img_bytes = pix.tobytes(output="png" if ext == ".png" else "jpg")
                except Exception:
                    continue

                order += 1
                shash = _short_hash(img_bytes, 10)
                fname = f"fig-p{page_index+1:03d}-{order:03d}-{shash}{ext}"
                dst = out_dir / fname
                with open(dst, "wb") as f:
                    f.write(img_bytes)

                rec = FigureRecord(
                    id=f"p{page_index+1}-{order}-{shash}",
                    doc_id=out_dir.name,
                    rel_path=str(dst.relative_to(out_dir)),
                    abs_path=str(dst),
                    order=order,
                    number=None,
                    title=None,
                    caption=None,
                    section=f"Стр. {page_index+1}",
                )
                rec.anchors = rec.build_anchors()
                fig_records.append(rec)
            # Не выходим — ниже ещё пойдёт векторный рендер по подписям

        # Привязка видимых имидж-блоков к подписям
        for xref, bbox in img_blocks:
            if (page_index, xref) in seen_xref_on_page:
                continue
            seen_xref_on_page.add((page_index, xref))

            w, h = xref_meta.get(xref, (0, 0))
            if w * h and (w * h) < PDF_MIN_IMAGE_AREA:
                continue

            best = None
            best_dy = 10**9
            best_cap_id = None
            for cap_id, cap_bbox, cap_text, cap_num, cap_title in captions:
                dy = _bbox_vdist(bbox, cap_bbox)
                if dy < best_dy:
                    best = (cap_text, cap_num, cap_title)
                    best_dy = dy
                    best_cap_id = cap_id

            cap_text, cap_num, cap_title = (None, None, None)
            if best and best_dy <= max(0, PDF_CAPTION_MAX_DY):
                cap_text, cap_num, cap_title = best
                if best_cap_id is not None:
                    matched_caption_ids.add(best_cap_id)

            try:
                pix = fitz.Pixmap(doc, xref)
                if pix.n > 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                ext = ".png" if pix.alpha or pix.n in (4, 5) else ".jpg"
                img_bytes = pix.tobytes(output="png" if ext == ".png" else "jpg")
            except Exception:
                continue

            order += 1
            shash = _short_hash(img_bytes, 10)
            base_name = f"fig-{cap_num.replace('.', '-')}-" if cap_num else f"fig-p{page_index+1:03d}-"
            fname = f"{base_name}{order:03d}-{shash}{ext}"
            dst = out_dir / fname
            with open(dst, "wb") as f:
                f.write(img_bytes)

            rec = FigureRecord(
                id=f"p{page_index+1}-{order}-{shash}",
                doc_id=out_dir.name,
                rel_path=str(dst.relative_to(out_dir)),
                abs_path=str(dst),
                order=order,
                number=cap_num,
                title=cap_title,
                caption=cap_text,
                section=f"Стр. {page_index+1}",
            )
            rec.anchors = rec.build_anchors()
            fig_records.append(rec)

        # --- Рендер векторных диаграмм возле подпиcей, если нет привязанной растровой картинки
        if PDF_VECTOR_ON and captions:
            vector_rects = _page_vector_rects(page)
            if vector_rects:
                for cap_id, cap_bbox, cap_text, cap_num, cap_title in captions:
                    # если к этой подписи уже привязали растровую картинку — пропускаем
                    if cap_id in matched_caption_ids:
                        continue
                    # собираем все векторные прямоугольники, достаточно близкие по вертикали
                    related = []
                    for r in vector_rects:
                        dy = _bbox_vdist(r, cap_bbox)
                        if dy <= max(0, PDF_CAPTION_MAX_DY):
                            # отсекаем абсурдно маленькие штрихи, но оставляем оси: >=2pt по обеим осям
                            if (r[2] - r[0]) >= 2.0 and (r[3] - r[1]) >= 2.0:
                                related.append(r)
                    if not related:
                        continue
                    cluster = _rect_union(related)
                    if not cluster:
                        continue

                    img_bytes = _render_clip_to_png(
                        page,
                        clip_rect=cluster,
                        dpi_base=PDF_VECTOR_DPI,
                        min_width_px=PDF_VECTOR_MIN_W,
                        max_dpi=PDF_VECTOR_MAX_DPI,
                        pad_px=PDF_VECTOR_PAD_PX,
                    )
                    if not img_bytes:
                        continue

                    order += 1
                    shash = _short_hash(img_bytes, 10)
                    # префикс 'figv' — чтобы отличать от вытащенных xref-изображений
                    base_name = f"figv-{cap_num.replace('.', '-')}-" if cap_num else f"figv-p{page_index+1:03d}-"
                    fname = f"{base_name}{order:03d}-{shash}.png"
                    dst = out_dir / fname
                    try:
                        with open(dst, "wb") as f:
                            f.write(img_bytes)
                    except Exception as e:
                        log.debug("Не удалось сохранить векторный рендер: %s", e)
                        continue

                    rec = FigureRecord(
                        id=f"p{page_index+1}-{order}-{shash}",
                        doc_id=out_dir.name,
                        rel_path=str(dst.relative_to(out_dir)),
                        abs_path=str(dst),
                        order=order,
                        number=cap_num,
                        title=cap_title,
                        caption=cap_text,
                        section=f"Стр. {page_index+1}",
                    )
                    rec.anchors = rec.build_anchors()
                    fig_records.append(rec)

    doc.close()
    return fig_records

# --- Индексация / загрузка ---

def _write_index(out_dir: Path, records: List[FigureRecord]) -> Dict[str, Any]:
    idx = {
        "doc_id": out_dir.name,
        "count": len(records),
        "figures": [asdict(r) for r in records],
    }
    with open(out_dir / "figures_index.json", "w", encoding="utf-8") as f:
        json.dump(idx, f, ensure_ascii=False, indent=2)
    return idx

def load_index(fig_dir: str | Path) -> Dict[str, Any]:
    p = Path(fig_dir)
    with open(p / "figures_index.json", "r", encoding="utf-8") as f:
        return json.load(f)

def index_document(path: str | Path) -> Dict[str, Any]:
    """
    Главная точка входа: извлечь картинки из файла и построить индекс.
    Возвращает dict-индекс.
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    _maybe_cleanup_root()
    doc_id = _doc_id_for(path)
    out_dir = (FIG_ROOT / doc_id)
    _ensure_clean_dir(out_dir)

    records: List[FigureRecord] = []

    if path.suffix.lower() == ".docx":
        if not DOCX_ON:
            log.info("DOCX_EXTRACT_IMAGES=false — пропускаю %s", path.name)
        else:
            log.info("Извлекаю изображения из DOCX: %s", path.name)
            records += extract_docx(path, out_dir, neighbor_window=NEIGH)

            # NEW: извлекаем встроенные диаграммы из chartXML и рендерим PNG без LibreOffice
            try:
                chart_records = _extract_docx_charts_to_png(
                    path, out_dir, neighbor_window=NEIGH, order_base=len(records)
                )
                if chart_records:
                    log.info("Добавил %d PNG из встроенных диаграмм DOCX.", len(chart_records))
                    records += chart_records
            except Exception as e:
                log.info("Не удалось извлечь диаграммы из DOCX: %s", e)

        # DOCX→PDF — только если совсем ничего не нашли, либо если явно включён ALWAYS
        if PDF_ON and DOCX_PDF_BACKEND != "off" and (DOCX_PDF_ALWAYS or (DOCX_FALLBACK and len(records) == 0)):
            if DOCX_PDF_ALWAYS and len(records) > 0:
                log.info("DOCX_PDF_ALWAYS=true — дополнительно запускаю PDF-вытяжку для векторных диаграмм.")
            elif len(records) == 0:
                log.info("DOCX дал 0 изображений — пробую конвертацию в PDF…")
            pdf_path = _docx_to_pdf_persistent(path, out_dir)
            if pdf_path and pdf_path.exists():
                log.info("Извлекаю изображения из PDF: %s", pdf_path.name)
                records += extract_pdf(pdf_path, out_dir)
            else:
                log.info("Не удалось выполнить DOCX→PDF.")


    elif path.suffix.lower() == ".pdf":
        if not PDF_ON:
            log.info("PDF_EXTRACT_IMAGES=false — пропускаю %s", path.name)
        else:
            log.info("Извлекаю изображения из PDF: %s", path.name)
            records += extract_pdf(path, out_dir)

    else:
        log.warning("Неподдерживаемый тип: %s", path.suffix)

    # сортировка: по номеру (если есть) → по порядку
    def _sort_key(r: FigureRecord):
        if r.number:
            try:
                nums = tuple(int(x) for x in r.number.split("."))
            except Exception:
                nums = (10**9,)
            return (0, nums, r.order)
        return (1, (r.order,), r.order)

    records.sort(key=_sort_key)
    idx = _write_index(out_dir, records)
    log.info("Готово: %d рисунков, индекс: %s", len(records), out_dir / "figures_index.json")
    return idx

# --- Поиск ---

def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def _best_by_score(items: Iterable[Tuple[Any, float]], top_k: int) -> List[Any]:
    arr = sorted(items, key=lambda x: x[1], reverse=True)
    return [x for x, _ in arr[:top_k]]

def query_by_number(index: Dict[str, Any], number: str) -> List[Dict[str, Any]]:
    number = _normalize_text(number).replace(",", ".")
    figs = index.get("figures", [])
    out = [f for f in figs if (str(f.get("number") or "").strip() == number)]
    return out

def query_by_caption(index: Dict[str, Any], text: str, top_k: int = 3) -> List[Dict[str, Any]]:
    q = _normalize_text(text).lower()
    figs = index.get("figures", [])
    scored = []
    for f in figs:
        pool = []
        for field in ("title", "caption"):
            if f.get(field):
                pool.append(_normalize_text(f[field]))
        if not pool:
            continue
        score = max(_ratio(q, p.lower()) for p in pool)
        scored.append((f, score))
    return _best_by_score(scored, top_k)

def query_by_chapter(index: Dict[str, Any], chapter: str, top_k: int = 3) -> List[Dict[str, Any]]:
    q = _normalize_text(chapter).lower()
    figs = index.get("figures", [])
    scored = []
    for f in figs:
        sec = f.get("section") or ""
        if not sec:
            continue
        score = _ratio(q, sec.lower())
        scored.append((f, score))
    return _best_by_score(scored, top_k)

def find_figure(
    index: Dict[str, Any],
    number: Optional[str] = None,
    caption_query: Optional[str] = None,
    chapter: Optional[str] = None,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """
    Универсальный подбор:
      - если задан number, пробуем точный номер
      - если chapter, фильтруем по главе
      - если caption_query, делаем нечёткий матч по подписи/названию
    """
    figs = index.get("figures", [])
    if not figs:
        return []

    cand = figs
    if number:
        cand = query_by_number(index, number)
        if cand:
            return cand

    if chapter:
        cand = query_by_chapter(index, chapter, top_k=max(top_k, 5))

    if caption_query:
        base = {"figures": cand} if cand != figs else index
        return query_by_caption(base, caption_query, top_k=top_k)

    return figs[:top_k]

# --- Утилита для “подготовки к vision” ---

def open_image_bytes(fig: Dict[str, Any]) -> bytes:
    p = Path(fig["abs_path"])
    with open(p, "rb") as f:
        return f.read()

def figure_display_name(fig: Dict[str, Any]) -> str:
    num = fig.get("number")
    title = fig.get("title")
    if num and title:
        return f"Рисунок {num}: {title}"
    if num:
        return f"Рисунок {num}"
    return Path(fig["rel_path"]).name

# --- CLI для локальной отладки ---

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Extract and index figures from documents")
    ap.add_argument("path", help="Путь к .docx или .pdf")
    ap.add_argument("--neighbor", type=int, default=NEIGH, help="Окно поиска подписи (DOCX)")
    args = ap.parse_args()

    idx = index_document(args.path)
    print(json.dumps(idx, ensure_ascii=False, indent=2))

# --- DOCX chart namespaces (NEW) ---
_DOCX_NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "c": "http://schemas.openxmlformats.org/drawingml/2006/chart",
}

def _parse_chart_xml_bytes(xml_bytes: bytes) -> Optional[Dict[str, Any]]:
    """Извлечь тип диаграммы и серии (label/value) из chartX.xml (без LibreOffice)."""
    from xml.etree import ElementTree as ET
    try:
        root = ET.fromstring(xml_bytes)
    except Exception:
        return None

    NS = {"c": _DOCX_NS["c"]}
    plot = root.find(".//c:plotArea", NS)
    if plot is None:
        return None

    types = ["pieChart", "pie3DChart", "barChart", "bar3DChart", "colChart",
             "col3DChart", "lineChart", "areaChart", "doughnutChart"]
    chart_node = None
    chart_type = None
    for t in types:
        node = plot.find(f"c:{t}", NS)
        if node is not None:
            chart_node, chart_type = node, t
            break
    if chart_node is None:
        return None

    def _str_pts(node):
        pts = []
        for pt in node.findall(".//c:pt", NS):
            idx = int(pt.attrib.get("idx", "0"))
            v = pt.find("c:v", NS)
            pts.append((idx, (v.text if v is not None else "")))
        pts.sort(key=lambda x: x[0])
        return [v for _, v in pts]

    def _num_pts(node):
        pts = []
        for pt in node.findall(".//c:pt", NS):
            idx = int(pt.attrib.get("idx", "0"))
            v = pt.find("c:v", NS)
            try:
                val = float(v.text) if v is not None else None
            except Exception:
                val = None
            pts.append((idx, val))
        pts.sort(key=lambda x: x[0])
        return [v for _, v in pts]

    series = []
    for ser in chart_node.findall("c:ser", NS):
        tx = ser.find(".//c:tx//c:v", NS)
        ser_name = tx.text if tx is not None else None

        cat = ser.find(".//c:cat", NS)
        val = ser.find(".//c:val", NS)
        cats, vals = [], []

        if cat is not None:
            strCache = cat.find(".//c:strCache", NS)
            if strCache is not None:
                cats = _str_pts(strCache)
        if val is not None:
            numCache = val.find(".//c:numCache", NS)
            if numCache is not None:
                vals = _num_pts(numCache)

        items = []
        for i, label in enumerate(cats):
            items.append((label, vals[i] if i < len(vals) else None))

        series.append({"name": ser_name, "items": items})

    return {"type": chart_type, "series": series} if series else None

def _render_chart_png(chart_info: Dict[str, Any], out_path: Path) -> Optional[str]:
    """Простейший рендер диаграммы в PNG (matplotlib)."""
    if not chart_info or not (chart_info.get("series") or []):
        return None
    try:
        import matplotlib.pyplot as plt  # локальный импорт
    except Exception as e:
        log.info("matplotlib недоступен, пропускаю рендер chart→PNG: %s", e)
        return None

    ser = chart_info["series"][0]
    labels = [l or "" for (l, _) in ser.get("items") or []]
    values = [float(v) if v is not None else 0.0 for (_, v) in ser.get("items") or []]

    plt.figure()
    t = (chart_info.get("type") or "").lower()
    if "pie" in t:
        vpos = [max(0.0, v) for v in values]
        if sum(vpos) == 0:
            vpos = [1 for _ in vpos]
        plt.pie(vpos, labels=labels, autopct="%1.1f%%")
        plt.title(ser.get("name") or "Pie chart")
    elif "line" in t:
        plt.plot(range(len(values)), values, marker="o")
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
        plt.title(ser.get("name") or "Line chart")
    else:
        plt.bar(range(len(values)), values)
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
        plt.title(ser.get("name") or "Bar chart")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()
    return str(out_path)

def _extract_docx_charts_to_png(
    docx_path: Path,
    out_dir: Path,
    neighbor_window: int = NEIGH,
    order_base: int = 0,
) -> List[FigureRecord]:
    """
    Находит встроенные диаграммы (c:chart) в .docx, парсит chartX.xml и рендерит PNG.
    Возвращает FigureRecord так же, как обычные растровые картинки.
    """
    try:
        from docx import Document
    except Exception as e:
        log.info("python-docx недоступен — пропускаю извлечение диаграмм: %s", e)
        return []

    doc = Document(str(docx_path))
    paragraphs = list(_iter_docx_paragraphs(doc))

    # карта "индекс параграфа → цепочка заголовков"
    section_path: List[str] = []
    section_path_by_idx: Dict[int, str] = {}
    for i, p in enumerate(paragraphs):
        lvl = _is_heading_style(_paragraph_style_name(p))
        if lvl:
            if len(section_path) >= lvl:
                section_path = section_path[:lvl - 1]
            section_path.append(_normalize_text(p.text))
        section_path_by_idx[i] = " > ".join([s for s in section_path if s])

    records: List[FigureRecord] = []
    order = order_base

    for i, p in enumerate(paragraphs):
        # ищем chart-объекты в runs текущего параграфа
        for run in getattr(p, "runs", []):
            # lxml-узел run._r доступен в python-docx
            for chart_el in run._r.xpath(".//w:drawing//a:graphic//a:graphicData/c:chart", namespaces=_DOCX_NS):
                rid = chart_el.get(f"{{{_DOCX_NS['r']}}}id")
                if not rid:
                    continue
                part = doc.part.related_parts.get(rid)
                if not part or not getattr(part, "blob", None):
                    continue

                info = _parse_chart_xml_bytes(part.blob)
                if not info:
                    continue  # ничего не извлекли из XML

                # подпись и номер рядом
                caption_text, _ = _nearest_caption(paragraphs, i, neighbor_window)
                fig_number, fig_title = (None, None)
                if caption_text:
                    n, ttl = parse_caption_line(caption_text)
                    fig_number, fig_title = n, ttl

                # рендерим PNG
                name_base = f"figc-{fig_number.replace('.', '-')}" if fig_number else f"figc-{order + 1:03d}"
                png_name = f"{name_base}-" + _short_hash(json.dumps(info, ensure_ascii=False).encode("utf-8"), 10) + ".png"
                png_path = out_dir / png_name
                img_path = _render_chart_png(info, png_path)
                if not img_path:
                    continue

                order += 1
                rec = FigureRecord(
                    id=f"{fig_number or order}-{_short_hash(png_name.encode(), 8)}",
                    doc_id=out_dir.name,
                    rel_path=str(png_path.relative_to(out_dir)),
                    abs_path=str(png_path),
                    order=order,
                    number=fig_number,
                    title=fig_title,
                    caption=caption_text,
                    section=section_path_by_idx.get(i) or None,
                )
                rec.anchors = rec.build_anchors()
                records.append(rec)

    return records
