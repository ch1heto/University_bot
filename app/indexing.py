# app/indexing.py
import re
import json
import hashlib
import numpy as np
from typing import List, Dict, Any, Iterable, Optional

from .db import get_conn
from .polza_client import embeddings
from .chunking import split_into_chunks
from .config import Cfg

# мягкие зависимости для OCR
try:
    import pytesseract  # type: ignore
    from PIL import Image  # type: ignore
except Exception:
    pytesseract = None
    Image = None

# --------- OCR конфиг с безопасными дефолтами ---------
try:
    OCR_ENABLED = bool(getattr(Cfg, "OCR_ENABLED", True))
    OCR_LANG = getattr(Cfg, "OCR_LANG", "rus+eng")
    OCR_MIN_CHARS = int(getattr(Cfg, "OCR_MIN_CHARS", 12))
    OCR_MAX_IMAGES = int(getattr(Cfg, "OCR_MAX_IMAGES_PER_SECTION", 6))
except Exception:
    OCR_ENABLED = True
    OCR_LANG = "eng"
    OCR_MIN_CHARS = 12
    OCR_MAX_IMAGES = 6


# ---------- helpers ----------

def _norm(s: str | None) -> str:
    return (s or "").strip()


def _make_anchor_id(section_path: str, page: int | None, title: str | None) -> str:
    base = f"{_norm(section_path)}|p={{}}|t={_norm(title)}".format(page or "")
    h = hashlib.sha1(base.encode("utf-8")).hexdigest()
    return f"anch-{h[:16]}"


def _prefix(section: Dict[str, Any]) -> str:
    et = (section.get("element_type") or "").lower()
    title = _norm(section.get("title")) or "Документ"
    page = section.get("page")

    tag = "[Текст]"
    if et == "heading":
        tag = "[Заголовок]"
    elif et == "table":
        tag = "[Таблица]"
    elif et == "figure":
        tag = "[Рисунок]"
    elif et == "page":
        tag = "[Страница]"

    parts = [tag]
    if page is not None:
        parts.append(f"стр.{page}")
    parts.append(title)
    return " ".join(parts)


def _attach_anchors(attrs: dict, *, section_path: str, page: int | None, title: str | None) -> dict:
    out = dict(attrs or {})
    out.setdefault("section_title", _norm(title))
    out.setdefault("section_path_norm", _norm(section_path))
    out.setdefault("loc", {"page": page, "section_path": _norm(section_path)})
    out.setdefault("anchor_id", _make_anchor_id(section_path, page, title))
    return out


def _chunks_table_has(con, cols: List[str]) -> bool:
    cur = con.cursor()
    cur.execute("PRAGMA table_info(chunks)")
    have = {row[1] for row in cur.fetchall()}
    return all(c in have for c in cols)


def _batched(items: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(items), n):
        yield items[i:i + n]


def _limit_table_row_columns(row: str, max_cols: int) -> str:
    if max_cols <= 0:
        return row
    parts = [p.strip() for p in (row or "").split(" | ")]
    if len(parts) <= max_cols:
        return " | ".join([p for p in parts if p])
    return " | ".join([p for p in parts[:max_cols] if p])


# ---------- OCR & CHART synthesis ----------

def _run_ocr_on_images_if_needed(base_attrs: dict) -> Optional[str]:
    """
    Если нет ocr_text, но есть images — делаем OCR.
    Возвращает строку (может быть пустой) или None, если OCR не выполнялся.
    """
    if not OCR_ENABLED or pytesseract is None or Image is None:
        return None

    if not isinstance(base_attrs, dict):
        return None

    ocr_text = (base_attrs.get("ocr_text") or "").strip()
    if ocr_text:
        return ocr_text  # уже есть

    images = base_attrs.get("images") or []
    if not isinstance(images, (list, tuple)) or not images:
        return None

    out_lines: List[str] = []
    processed = 0
    for pth in images:
        if processed >= max(1, OCR_MAX_IMAGES):
            break
        try:
            img = Image.open(pth)
        except Exception:
            continue
        try:
            txt = pytesseract.image_to_string(img, lang=OCR_LANG) or ""
        except Exception:
            txt = ""
        txt = txt.replace("\x0c", "").strip()
        if txt:
            out_lines.append(txt)
        processed += 1

    merged = "\n".join([t for t in out_lines if t.strip()]).strip()
    if merged and len(merged) >= OCR_MIN_CHARS:
        # положим в attrs, чтобы переиспользовалось в следующих шагах/повторах
        base_attrs["ocr_text"] = merged
        return merged
    return "" if merged else None


def _synth_chart_text(base_attrs: dict, section: Dict[str, Any]) -> Optional[str]:
    """
    Из attrs.chart_data собираем читабельный текст.
    Формат: «метка — значение» построчно + лёгкая сводка (мин/макс/сумма, если числовые).
    """
    rows = base_attrs.get("chart_data")
    if not isinstance(rows, list) or not rows:
        return None

    lines: List[str] = []
    numeric_vals: List[float] = []

    # Заголовок/контекст
    cap_num = base_attrs.get("caption_num") or base_attrs.get("label")
    cap_tail = base_attrs.get("caption_tail") or base_attrs.get("title")
    head = None
    if (section.get("element_type") or "").lower() == "figure":
        if cap_num and cap_tail:
            head = f"Данные диаграммы «Рисунок {cap_num} — {cap_tail}»"
        elif cap_num:
            head = f"Данные диаграммы «Рисунок {cap_num}»"
        elif cap_tail:
            head = f"Данные диаграммы: {cap_tail}"
    if head:
        lines.append(head)

    for r in rows:
        label = _norm(str(r.get("label", "")))
        val = r.get("value", None)
        if label == "" and val is None:
            continue
        if isinstance(val, float) or isinstance(val, int):
            numeric_vals.append(float(val))
            vstr = f"{val}".rstrip("0").rstrip(".") if isinstance(val, float) else str(val)
        else:
            vstr = str(val) if val is not None else ""
        if label and vstr:
            lines.append(f"{label} — {vstr}")
        elif label:
            lines.append(label)
        elif vstr:
            lines.append(vstr)

    # простая сводка
    if numeric_vals:
        s = sum(numeric_vals)
        mn = min(numeric_vals)
        mx = max(numeric_vals)
        lines.append(f"Сумма = {s:g}; мин = {mn:g}; макс = {mx:g}")

    txt = "\n".join([ln for ln in lines if ln.strip()]).strip()
    if txt:
        base_attrs["chart_text"] = txt  # сохраним для дебага/повтора
        return txt
    return None


def _yield_ocr_chunks_if_any(
    section: Dict[str, Any],
    base_attrs: dict,
) -> Iterable[tuple[str, Dict[str, Any], str, dict]]:
    """
    Если у секции есть ocr_text — режем и отдаём.
    Если ocr_text нет, но есть images — пробуем сделать OCR и также отдаём.
    subtype = "ocr".
    """
    et = (section.get("element_type") or "").lower()
    if not isinstance(base_attrs, dict):
        return

    # OCR при необходимости
    ocr_text_existing = (base_attrs.get("ocr_text") or "").strip()
    if not ocr_text_existing:
        maybe = _run_ocr_on_images_if_needed(base_attrs)
        if isinstance(maybe, str):
            ocr_text_existing = maybe.strip()

    if not ocr_text_existing:
        return

    page = section.get("page")
    section_path = _norm(section.get("section_path")) or _norm(section.get("title")) or "Документ"
    for ch in split_into_chunks(ocr_text_existing):
        if not ch.strip():
            continue
        attrs = dict(base_attrs)
        attrs["subtype"] = "ocr"
        yield ch, {"page": page, "section_path": section_path}, (et or "figure"), attrs


def _yield_chart_chunks_if_any(
    section: Dict[str, Any],
    base_attrs: dict,
) -> Iterable[tuple[str, Dict[str, Any], str, dict]]:
    """
    Если у секции есть chart_data — синтезируем текст и отдаём отдельные чанки.
    subtype = "chart".
    """
    et = (section.get("element_type") or "").lower()
    if not isinstance(base_attrs, dict):
        return

    chart_txt = base_attrs.get("chart_text")
    if not chart_txt:
        chart_txt = _synth_chart_text(base_attrs, section)

    if not chart_txt:
        return

    page = section.get("page")
    section_path = _norm(section.get("section_path")) or _norm(section.get("title")) or "Документ"
    for ch in split_into_chunks(chart_txt):
        if not ch.strip():
            continue
        attrs = dict(base_attrs)
        attrs["subtype"] = "chart"
        yield ch, {"page": page, "section_path": section_path}, (et or "figure"), attrs


def _yield_chunks_for_section(
    section: Dict[str, Any],
    *,
    max_table_rows: Optional[int] = None,
    max_table_cols: Optional[int] = None,
) -> Iterable[tuple[str, Dict[str, Any], str, dict]]:
    et = (section.get("element_type") or "").lower()
    title = _norm(section.get("title")) or "Документ"
    section_path = _norm(section.get("section_path")) or title
    page = section.get("page")
    base_attrs = dict(section.get("attrs") or {})
    text = section.get("text") or ""

    # якоря
    base_attrs = _attach_anchors(base_attrs, section_path=section_path, page=page, title=title)

    # заголовок
    if et == "heading":
        head_txt = _prefix(section)
        yield head_txt, {"page": page, "section_path": section_path}, "heading", base_attrs
        return

    # источник
    if et == "reference":
        ref_text = (text or "").strip()
        if ref_text:
            yield ref_text, {"page": page, "section_path": section_path}, "reference", base_attrs
        return

    # таблица
    if et == "table":
        cap_tail = base_attrs.get("caption_tail") or base_attrs.get("title")
        header_preview = base_attrs.get("header_preview")
        tail_from_title = None
        m = re.search(
            r"(?i)\bтабл(?:ица)?\.?\s*(?:№\s*)?(?:[A-Za-zА-Яа-я]\.?[\s-]*\d+(?:[.,]\d+)*|\d+(?:[.,]\d+)*)\s*[—\-–:\u2013\u2014]\s*(.+)",
            _norm(title)
        )
        if m:
            tail_from_title = _norm(m.group(1))

        root_text = cap_tail or header_preview or tail_from_title or "(таблица)"
        yield root_text, {"page": page, "section_path": section_path}, "table", base_attrs

        lines = [ln.strip() for ln in (text or "").splitlines() if ln and ln.strip()]
        if not lines:
            # даже если нет текстовых строк — попробуем OCR по картинкам в attrs.images
            yield from _yield_ocr_chunks_if_any(section, base_attrs)
            # и текст из диаграммы внутри таблицы (если парсер положил chart_data)
            yield from _yield_chart_chunks_if_any(section, base_attrs)
            return

        n_max_rows = Cfg.FULL_TABLE_MAX_ROWS if max_table_rows is None else max_table_rows
        n_max_cols = Cfg.FULL_TABLE_MAX_COLS if max_table_cols is None else max_table_cols

        for i, row in enumerate(lines[: max(1, n_max_rows)], 1):
            trimmed = _limit_table_row_columns(row, max(0, n_max_cols or 0))
            attrs = dict(base_attrs)
            attrs["row_index"] = i
            row_section_path = f"{section_path} [row {i}]"
            attrs = _attach_anchors(attrs, section_path=row_section_path, page=page, title=title)
            yield trimmed, {"page": page, "section_path": row_section_path}, "table_row", attrs

        # OCR и диаграмма-текст
        yield from _yield_ocr_chunks_if_any(section, base_attrs)
        yield from _yield_chart_chunks_if_any(section, base_attrs)
        return

    # фигуры / страницы / обычные параграфы
    base = text if isinstance(text, str) else str(text)
    if et == "figure" and not base.strip():
        cap_num = base_attrs.get("caption_num") or base_attrs.get("label")
        cap_tail = base_attrs.get("caption_tail") or base_attrs.get("title")
        if cap_num and cap_tail:
            base = f"Рисунок {cap_num} — {cap_tail}"
        elif cap_num:
            base = f"Рисунок {cap_num}"
        elif cap_tail:
            base = str(cap_tail)
        else:
            base = "Рисунок"

    prefix = _prefix(section)
    for ch in split_into_chunks(base):
        if not ch.strip():
            continue
        out_et = et if et in {"page", "figure", "paragraph"} else ("paragraph" if et == "" else et)
        yield f"{prefix}\n{ch}", {"page": page, "section_path": section_path}, out_et, base_attrs

    # добавим OCR-чанки и данные диаграмм
    yield from _yield_ocr_chunks_if_any(section, base_attrs)
    yield from _yield_chart_chunks_if_any(section, base_attrs)


# ---------- API ----------

def index_document(
    owner_id: int,
    doc_id: int,
    sections: List[Dict[str, Any]],
    *,
    batch_size: int = 128
) -> None:
    """
    Индексирует документ:
    - Источники -> element_type='reference' (только текст записи).
    - Таблицы -> корневой чанк + построчно с ограничениями Cfg.FULL_TABLE_MAX_ROWS/COLS.
    - Заголовки -> отдельные короткие чанки.
    - Фигуры/страницы/текст -> обычные чанки с контекстным префиксом.
    - **НОВОЕ**: OCR картинок (attrs.images) прямо здесь (subtype="ocr"), если нет attrs.ocr_text.
    - **НОВОЕ**: Текст из диаграмм (attrs.chart_data) превращается в чанки (subtype="chart").
    - Эмбеддинги считаются батчами.
    """
    rows_text: List[str] = []
    rows_meta: List[Dict[str, Any]] = []
    rows_type: List[str] = []
    rows_attrs: List[dict] = []

    for s in sections or []:
        for txt, meta, etype, attrs in _yield_chunks_for_section(
            s,
            max_table_rows=Cfg.FULL_TABLE_MAX_ROWS,
            max_table_cols=Cfg.FULL_TABLE_MAX_COLS,
        ):
            if not txt.strip():
                continue
            rows_text.append(txt)
            rows_meta.append(meta)
            rows_type.append(etype)
            rows_attrs.append(attrs or {})

    if not rows_text:
        return

    con = get_conn()
    cur = con.cursor()
    has_extended_cols = _chunks_table_has(con, ["element_type", "attrs"])

    idx = 0
    for batch in _batched(rows_text, batch_size):
        vecs = embeddings(batch)
        if not vecs or len(vecs) != len(batch):
            raise RuntimeError("embeddings() вернул неожиданный размер результата.")
        for j, vec in enumerate(vecs):
            k = idx + j
            meta = rows_meta[k]
            blob = np.asarray(vec, dtype=np.float32).tobytes()

            if has_extended_cols:
                cur.execute(
                    "INSERT INTO chunks(doc_id, owner_id, page, section_path, text, element_type, attrs, embedding) "
                    "VALUES(?,?,?,?,?,?,?,?)",
                    (
                        doc_id,
                        owner_id,
                        meta.get("page"),
                        meta.get("section_path"),
                        batch[j],
                        rows_type[k],
                        json.dumps(rows_attrs[k], ensure_ascii=False),
                        blob,
                    ),
                )
            else:
                cur.execute(
                    "INSERT INTO chunks(doc_id, owner_id, page, section_path, text, embedding) "
                    "VALUES(?,?,?,?,?,?)",
                    (doc_id, owner_id, meta.get("page"), meta.get("section_path"), batch[j], blob),
                )
        idx += len(batch)

    con.commit()
    con.close()
