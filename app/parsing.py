# app/parsing.py
from docx import Document as Docx
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.text.run import Run
from docx.enum.text import WD_ALIGN_PARAGRAPH
import pdfplumber
from pathlib import Path
import subprocess
import tempfile
import platform
import shutil
import os
import re
from typing import List, Dict, Any, Optional, Tuple

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

def _table_to_text(tbl: Table) -> str:
    """Таблица → текст: одна строка = ряд, ячейки через ' | ' (удаляем повторы от merged)."""
    lines = []
    for row in tbl.rows:
        cells = []
        for c in row.cells:
            cells.append(_clean(c.text))
        # убираем повторы смежных ячеек (merge-эффект)
        dedup = []
        for i, x in enumerate(cells):
            if i == 0 or x != cells[i - 1]:
                dedup.append(x)
        line = " | ".join(dedup).strip(" |")
        if line:
            lines.append(line)
    return "\n".join(lines)

def _table_attrs(tbl: Table) -> dict:
    try:
        n_rows = len(tbl.rows)
    except Exception:
        n_rows = None
    try:
        n_cols = len(tbl.columns)
    except Exception:
        n_cols = None
    return {"n_rows": n_rows, "n_cols": n_cols}

# ----------------------------- caption helpers -----------------------------

CAPTION_RE_TABLE = re.compile(
    r"^\s*(табл(?:ица)?|table)\.?\s*(\d+(?:[.,]\d+)?)(?:\s*[-—:]\s*(.*))?\s*$",
    re.IGNORECASE
)
CAPTION_RE_FIG = re.compile(
    r"^\s*(рис(?:унок)?|figure)\.?\s*(\d+(?:[.,]\d+)?)(?:\s*[-—:]\s*(.*))?\s*$",
    re.IGNORECASE
)

def _classify_caption(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Возвращает (kind, normalized_title):
      kind ∈ {'table','figure', None}
      normalized_title вида 'Таблица 2.1 — Подпись' / 'Рисунок 1.3 — Подпись'
    """
    t = (text or "").strip()
    m = CAPTION_RE_TABLE.match(t)
    if m:
        num, tail = m.group(2), (m.group(3) or "").strip()
        title = f"Таблица {num}" + (f" — {tail}" if tail else "")
        return "table", title
    m = CAPTION_RE_FIG.match(t)
    if m:
        num, tail = m.group(2), (m.group(3) or "").strip()
        title = f"Рисунок {num}" + (f" — {tail}" if tail else "")
        return "figure", title
    return None, None

# ----------------------------- sources helpers -----------------------------

SOURCES_TITLE_RE = re.compile(r"(источник|список\s+литератур|библиограф|references?)", re.IGNORECASE)
REF_LINE_RE = re.compile(r"^\s*(?:\[(\d+)\]|(\d+)[\.\)])\s*(.+)$")

def _is_sources_title(s: str) -> bool:
    return bool(SOURCES_TITLE_RE.search(s or ""))

def _parse_reference_line(s: str) -> Tuple[Optional[int], str]:
    """
    Из строки источника выделяет номер (если есть) и «хвост».
    Поддерживает формы: "[12] …", "12. …", "12) …".
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

# ----------------------------- section-path helpers -----------------------------

def _hpath_str(stack: List[str]) -> str:
    return " / ".join([s for s in stack if s])

# ----------------------------- DOCX main -----------------------------

def parse_docx(path: str) -> List[Dict[str, Any]]:
    """
    Парсим .docx с поддержкой заголовков, таблиц, базового форматирования, пометок о картинках и
    корректной обработкой подписей (таблица/рисунок). Для секций формируем иерархический section_path
    по стеку заголовков: 'Глава 1 / 1.1 Введение / Таблица 1.1 — ...'.
    Также распознаём раздел «Источники/Список литературы/Библиография/References» и помечаем его
    элементы как element_type='reference'.
    """
    doc = Docx(path)

    sections: List[Dict[str, Any]] = []
    buf: List[str] = []
    last_para_attrs: Optional[dict] = None

    # текущий заголовок и стек заголовков для section_path
    title, level = "Документ", 0
    heading_stack: List[str] = []

    # состояние: находимся ли в разделе «Источники»
    in_sources = False

    # ожидаемая подпись
    pending_caption_tbl: Optional[str] = None
    pending_caption_fig: Optional[str] = None
    table_counter = 0
    figure_counter = 0

    def flush_text_section():
        nonlocal buf, title, level, last_para_attrs
        if buf:
            text = "\n".join(buf)
            sections.append({
                "title": title,
                "level": level,
                "text": text,
                "page": None,
                "section_path": _hpath_str(heading_stack) or title,
                "element_type": "paragraph",
                "attrs": {
                    "numbers": _extract_numbers(text),
                    **(last_para_attrs or {})
                }
            })
            buf.clear()
            last_para_attrs = None

    for block in _iter_block_items(doc):
        if isinstance(block, Paragraph):
            p_text = _clean(block.text)
            p_style = ((block.style and block.style.name) or "").lower()

            # Заголовок раздела
            if p_style.startswith("heading"):
                flush_text_section()
                title = p_text or "Без названия"
                try:
                    level = max(1, int(p_style.replace("heading", "")))
                except Exception:
                    level = 1
                # обновляем стек заголовков
                if len(heading_stack) < level:
                    heading_stack += [""] * (level - len(heading_stack))
                heading_stack = heading_stack[:level]
                heading_stack[level - 1] = title

                # режим «Источники»
                in_sources = _is_sources_title(title)

                # сбрасываем ожидаемые подписи
                pending_caption_tbl = None
                pending_caption_fig = None

                # отдельная секция для самого заголовка
                sections.append({
                    "title": title,
                    "level": level,
                    "text": "",
                    "page": None,
                    "section_path": _hpath_str(heading_stack),
                    "element_type": "heading",
                    "attrs": {"style": p_style}
                })
                continue

            # Если мы в разделе Источников — каждый непустой абзац трактуем как запись reference
            if in_sources and p_text:
                flush_text_section()
                idx, tail = _parse_reference_line(p_text)
                attrs = _paragraph_attrs(block)
                attrs.update({"numbers": _extract_numbers(tail)})
                if idx is not None:
                    attrs["ref_index"] = idx
                ref_title = f"Источник {idx}" if idx is not None else "Источник"
                sections.append({
                    "title": ref_title,
                    "level": max(1, level + 1),
                    "text": tail,
                    "page": None,
                    "section_path": _hpath_str(heading_stack + [ref_title]),
                    "element_type": "reference",
                    "attrs": attrs,
                })
                continue

            # Подпись (caption) — разруливаем двусмысленность
            if p_style == "caption" or CAPTION_RE_TABLE.match(p_text or "") or CAPTION_RE_FIG.match(p_text or ""):
                kind, norm = _classify_caption(p_text or "")
                if kind == "table":
                    pending_caption_tbl = norm or p_text or None
                    pending_caption_fig = None
                elif kind == "figure":
                    pending_caption_fig = norm or p_text or None
                    pending_caption_tbl = None
                else:
                    # caption без ключевого слова — игнорируем
                    pass
                continue

            # Абзац с картинкой → фиксируем как «Рисунок»
            if _paragraph_has_image(block):
                flush_text_section()
                figure_counter += 1
                fig_title = pending_caption_fig or f"Рисунок {figure_counter}"
                sections.append({
                    "title": fig_title,
                    "level": max(1, level + 1),
                    "text": p_text or "",
                    "page": None,
                    "section_path": _hpath_str(heading_stack + [fig_title]),
                    "element_type": "figure",
                    "attrs": _paragraph_attrs(block)
                })
                pending_caption_fig = None
                continue

            # Обычный текст — копим в буфер
            if p_text:
                last_para_attrs = _paragraph_attrs(block)
                buf.append(p_text)
            continue

        # Таблица
        if isinstance(block, Table):
            flush_text_section()
            table_counter += 1
            t_title = pending_caption_tbl or f"Таблица {table_counter}"
            t_text = _table_to_text(block) or "(пустая таблица)"
            sections.append({
                "title": t_title,
                "level": max(1, level + 1),
                "text": t_text,
                "page": None,
                "section_path": _hpath_str(heading_stack + [t_title]),
                "element_type": "table",
                "attrs": _table_attrs(block) | {"numbers": _extract_numbers(t_text)}
            })
            pending_caption_tbl = None
            continue

    # «Хвост» текста
    flush_text_section()

    return sections

# ----------------------------- PDF -----------------------------

def parse_pdf(path: str) -> List[Dict[str, Any]]:
    """
    Простой парсинг PDF:
    - постраничный текст (element_type='page')
    - попытка извлечения простых таблиц (element_type='table')
    - эвристическое извлечение списка литературы (element_type='reference')
    """
    out: List[Dict[str, Any]] = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""

            out.append({
                "title": f"Стр. {i}",
                "level": 1,
                "text": text,
                "page": i,
                "section_path": f"p.{i}",
                "element_type": "page",
                "attrs": {"numbers": _extract_numbers(text)}
            })

            # простые таблицы (если распознаны)
            try:
                tables = page.extract_tables()
            except Exception:
                tables = None
            if tables:
                for ti, rows in enumerate(tables, 1):
                    try:
                        lines = []
                        for row in rows or []:
                            row = [_clean(c) for c in (row or [])]
                            line = " | ".join([c for c in row if c])
                            if line:
                                lines.append(line)
                        t_text = "\n".join(lines)
                        if t_text.strip():
                            out.append({
                                "title": f"Таблица {i}.{ti}",
                                "level": 2,
                                "text": t_text,
                                "page": i,
                                "section_path": f"Стр. {i} / Таблица {i}.{ti}",
                                "element_type": "table",
                                "attrs": {"numbers": _extract_numbers(t_text)}
                            })
                    except Exception:
                        pass

            # эвристика для списка литературы на странице
            has_sources_kw = bool(SOURCES_TITLE_RE.search(text))
            ref_lines = []
            for line in (text.splitlines() or []):
                line = line.strip()
                if not line:
                    continue
                m = REF_LINE_RE.match(line)
                if m:
                    idx = int(m.group(1) or m.group(2))
                    tail = (m.group(3) or "").strip()
                    ref_lines.append((idx, tail))

            # если есть явные признаки — добавим как отдельные элементы
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

def save_upload(raw: bytes, filename: str, upload_dir: str = "./uploads") -> str:
    p = Path(upload_dir)
    p.mkdir(parents=True, exist_ok=True)
    fp = p / filename
    fp.write_bytes(raw)
    return str(fp)
