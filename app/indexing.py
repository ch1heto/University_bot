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


# ---------- helpers ----------

def _norm(s: str | None) -> str:
    return (s or "").strip()


def _make_anchor_id(section_path: str, page: int | None, title: str | None) -> str:
    """
    Стабильный якорь для ссылки на место в тексте.
    Основан на section_path + page + title (безопасно для старых индексов: хранится только в attrs).
    """
    base = f"{_norm(section_path)}|p={page or ''}|t={_norm(title)}"
    h = hashlib.sha1(base.encode("utf-8")).hexdigest()  # короткий и достаточно стабильный
    return f"anch-{h[:16]}"


def _prefix(section: Dict[str, Any]) -> str:
    """
    Короткий префикс контекста для большинства чанков (повышает качество поиска).
      [Заголовок] Введение
      [Таблица] стр.3 Таблица 2.1 — Состав выборки
      [Рисунок] стр.7 Рисунок 3.2 — Архитектура
      [Текст] стр.5 Глава 2. Методика исследования

    ВАЖНО: для element_type='reference' префиксы НЕ используются — в text пишем только саму запись источника.
    """
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
    """
    Ненавязчиво обогащаем attrs служебными полями для прозрачных ссылок.
    Не ломает старые пайплайны (всё внутри JSON attrs).
    """
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
    """
    Ограничивает число колонок в строке таблицы (split по ' | ').
    """
    if max_cols <= 0:
        return row
    parts = [p.strip() for p in (row or "").split(" | ")]
    if len(parts) <= max_cols:
        return " | ".join([p for p in parts if p])
    return " | ".join([p for p in parts[:max_cols] if p])


def _yield_ocr_chunks_if_any(
    section: Dict[str, Any],
    base_attrs: dict,
) -> Iterable[tuple[str, Dict[str, Any], str, dict]]:
    """
    Если секция (обычно figure) содержит attrs.ocr_text — порождаем отдельные OCR-чанки.
    Тип оставляем тем же (figure/page/...) и помечаем subtype="ocr" в attrs.
    """
    et = (section.get("element_type") or "").lower()
    if not isinstance(base_attrs, dict):
        return
    ocr_text = (base_attrs.get("ocr_text") or "").strip()
    if not ocr_text:
        return

    page = section.get("page")
    section_path = _norm(section.get("section_path")) or _norm(section.get("title")) or "Документ"
    # OCR-текст режем самостоятельными чанками (без префикса, чтобы не раздувать текст).
    # Но добавим небольшой OCR-тег в attrs.
    for ch in split_into_chunks(ocr_text):
        if not ch.strip():
            continue
        attrs = dict(base_attrs)
        attrs["subtype"] = "ocr"
        yield ch, {"page": page, "section_path": section_path}, et or "figure", attrs


def _yield_chunks_for_section(
    section: Dict[str, Any],
    *,
    max_table_rows: Optional[int] = None,
    max_table_cols: Optional[int] = None,
) -> Iterable[tuple[str, Dict[str, Any], str, dict]]:
    """
    Генерирует чанки для одной секции.
    Возвращает кортежи: (text, meta, element_type, attrs_dict)

    - Таблицы -> сначала корневой чанк (element_type='table' с описанием),
                 затем одна строка = один чанк (element_type='table_row', attrs.row_index).
    - Заголовки -> отдельный малый чанк (element_type='heading').
    - Источники -> один чанк на запись (element_type='reference'); в text — ТОЛЬКО текст записи.
    - Остальной текст (включая figure/page) -> split_into_chunks(...) с префиксом в начале.
      Для figure, если text пустой, синтезируем «Рисунок N — Хвост», чтобы чанк не потерялся.
    - Если в attrs присутствует ocr_text — добавляем отдельные OCR-чанки (subtype="ocr").
    """
    et = (section.get("element_type") or "").lower()
    title = _norm(section.get("title")) or "Документ"
    section_path = _norm(section.get("section_path")) or title
    page = section.get("page")
    base_attrs = dict(section.get("attrs") or {})
    text = section.get("text") or ""

    # Добавим прозрачные якоря (безопасно: всё уходит в attrs JSON)
    base_attrs = _attach_anchors(base_attrs, section_path=section_path, page=page, title=title)

    # Заголовок — отдельный небольшой чанк с префиксом
    if et == "heading":
        head_txt = _prefix(section)
        yield head_txt, {"page": page, "section_path": section_path}, "heading", base_attrs
        return

    # Источник — отдельный чанк: в text ТОЛЬКО содержимое записи без служебных префиксов
    if et == "reference":
        ref_text = (text or "").strip()
        if ref_text:
            yield ref_text, {"page": page, "section_path": section_path}, "reference", base_attrs
        return

    # Таблица — корневой чанк + построчно
    if et == "table":
        cap_tail = base_attrs.get("caption_tail") or base_attrs.get("title")
        header_preview = base_attrs.get("header_preview")
        # попробуем вытащить хвост из заголовка "Таблица N — Хвост"
        tail_from_title = None
        m = re.search(
            r"(?i)\bтабл(?:ица)?\.?\s*(?:№\s*)?(?:[A-Za-zА-Яа-я]\.?[\s-]*\d+(?:[.,]\d+)*|\d+(?:[.,]\d+)*)\s*[—\-–:\u2013\u2014]\s*(.+)",
            _norm(title)
        )
        if m:
            tail_from_title = _norm(m.group(1))

        root_text = cap_tail or header_preview or tail_from_title or "(таблица)"
        yield root_text, {"page": page, "section_path": section_path}, "table", base_attrs

        # Построчные чанки: с учётом лимитов из конфигурации
        lines = [ln.strip() for ln in (text or "").splitlines() if ln and ln.strip()]
        if not lines:
            # Таблица без текста (только мета/attrs) — всё равно OCR-чанки добавим, если есть
            yield from _yield_ocr_chunks_if_any(section, base_attrs)
            return

        n_max_rows = Cfg.FULL_TABLE_MAX_ROWS if max_table_rows is None else max_table_rows
        n_max_cols = Cfg.FULL_TABLE_MAX_COLS if max_table_cols is None else max_table_cols

        for i, row in enumerate(lines[: max(1, n_max_rows) ], 1):
            trimmed = _limit_table_row_columns(row, max(0, n_max_cols or 0))
            attrs = dict(base_attrs)
            attrs["row_index"] = i
            row_section_path = f"{section_path} [row {i}]"
            attrs = _attach_anchors(attrs, section_path=row_section_path, page=page, title=title)
            yield trimmed, {"page": page, "section_path": row_section_path}, "table_row", attrs

        # OCR по табличным картинкам (если есть)
        yield from _yield_ocr_chunks_if_any(section, base_attrs)
        return

    # Фигуры / страницы / обычные абзацы — делим на чанки, добавляем префикс как контекст
    # Для figure при пустом текстe синтезируем строку из подписи (важно для поиска «Рисунок N …»).
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

    # Добавим OCR-чанки (если есть ocr_text)
    yield from _yield_ocr_chunks_if_any(section, base_attrs)


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
    - Источники: element_type='reference', text = чистый текст записи, attrs.ref_index сохраняется как есть.
    - Таблицы: корневой чанк с описанием (element_type='table') + построчно (element_type='table_row', attrs.row_index),
      при этом строки/колонки ограничиваются по Cfg.FULL_TABLE_MAX_ROWS / FULL_TABLE_MAX_COLS.
    - Заголовки: короткие чанки (element_type='heading').
    - Фигуры: текст подписи и (если есть) OCR — отдельными чанками (subtype="ocr" в attrs).
    - Остальное: обычные текстовые чанки с префиксом в начале, чтобы улучшить поиск.
    - Эмбеддинги считаются пакетами (batch_size) и пишутся в таблицу chunks.
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

    # Считаем эмбеддинги батчами
    idx = 0
    for batch in _batched(rows_text, batch_size):
        vecs = embeddings(batch)  # list[list[float]] или np.ndarray
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
