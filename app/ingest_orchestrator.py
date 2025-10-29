# app/ingest_orchestrator.py
from __future__ import annotations

import os
import re
import io
import uuid
import json
import time
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Callable
from copy import deepcopy
from pathlib import Path

from .config import Cfg, ProcessingState
from .db import (
    find_existing_document,
    insert_document,
    update_document_meta,
    set_document_indexer_version,
    start_indexing,
    finish_indexing_success,
    finish_indexing_error,
    CURRENT_INDEXER_VERSION,
)

# Необязательные зависимости — если их нет, просто деградируем без ошибок
try:
    from .table_reconstruct import header_preview_from_row  # краткое превью по первой строке
except Exception:  # pragma: no cover
    def header_preview_from_row(row_text: str) -> str:
        # минимальный фолбэк
        t = (row_text or "").strip().split("\n")[0]
        t = " — ".join([c.strip() for c in t.split(" | ") if c.strip()])
        return t[:180]

try:
    from .vision_tables_ocr import is_table_image_section  # эвристика для табличных картинок
except Exception:  # pragma: no cover
    def is_table_image_section(sec: Dict[str, Any]) -> bool:
        return False

# --- Мягкие зависимости для парсинга / OCR ---
try:
    from .parsing import parse_docx, parse_pdf, parse_doc
except Exception:
    parse_docx = parse_pdf = parse_doc = None  # будем валить осмысленной ошибкой

# ----------------------------- КОНСТАНТЫ / РЕГЕКСЫ -----------------------------

# Подписи к таблицам: «Таблица 2.3 — Название», допускаем буквы «А.1», «П1.2»
_TABLE_CAP_RE = re.compile(
    r"(?i)\bтаблица\s+([A-Za-zА-Яа-я]?\s*\d+(?:[.,]\d+)*)\b(?:\s*[—\-–:]\s*(.+))?"
)

# Подписи к рисункам: «Рисунок 3», «Рис. 2.1 — ...», «Figure 1.2 ...»
_FIG_CAP_RE = re.compile(
    r"(?i)\b(?:рис(?:\.|унок)?|figure|fig\.?)\s*(?:№\s*)?([A-Za-zА-Яа-я]?\s*\d+(?:[.,]\d+)*)\b(?:\s*[—\-–:]\s*(.+))?"
)

# Строки библиографии: "[12] Текст..." или "12) Текст..." или "12. Текст..."
_REF_LINE_RE = re.compile(r"^\s*(?:\[(\d+)\]|(\d{1,3})[.)])\s+(.+)$")


# ----------------------------- УТИЛИТЫ ОРКЕСТРАЦИИ -----------------------------

class IngestError(Exception):
    """Обёртка ошибок на этапе индексации (для единообразного репортинга в FSM)."""


def _file_sha256(path: str, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _detect_kind(path: str, fallback: str = "doc") -> str:
    ext = (os.path.splitext(path)[1] or "").lower()
    if ext in {".docx"}:
        return "docx"
    if ext in {".pdf"}:
        return "pdf"
    if ext in {".txt", ".md"}:
        return "txt"
    if ext in {".doc"}:
        return "doc"
    return fallback


# ----------------------------- ОБОГАЩЕНИЕ (оставляем из вашего файла) -----------------------------

def _norm_num(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    return s.replace(" ", "").replace(",", ".").strip()

def _last_segment(path: str) -> str:
    s = (path or "").strip()
    if "/" in s:
        s = s.split("/")[-1].strip()
    return s

def _ensure_attrs(d: Dict[str, Any]) -> Dict[str, Any]:
    if "attrs" not in d or not isinstance(d["attrs"], dict):
        d["attrs"] = {}
    return d["attrs"]

def _is_reference_section_name(section_path: str) -> bool:
    sp = (section_path or "").lower()
    return any(k in sp for k in ("источник", "литератур", "библиограф", "reference", "references", "bibliograph"))

def _base_table_name(section_path: str) -> str:
    """
    Для строк вида '.../Таблица 2.1 — ... [row 1]' вернёт базу без хвоста ' [row ...]'.
    """
    s = section_path or ""
    pos = s.find(" [row ")
    return s[:pos] if pos > 0 else s

def _parse_table_caption_from(path_or_text: str) -> Tuple[Optional[str], Optional[str]]:
    m = _TABLE_CAP_RE.search(path_or_text or "")
    if not m:
        return (None, None)
    return _norm_num(m.group(1)), (m.group(2) or "").strip() or None

def _parse_figure_caption_from(path_or_text: str) -> Tuple[Optional[str], Optional[str]]:
    m = _FIG_CAP_RE.search(path_or_text or "")
    if not m:
        return (None, None)
    return _norm_num(m.group(1)), (m.group(2) or "").strip() or None


def _enrich_tables(sections: List[Dict[str, Any]]) -> None:
    """
    Проставляет для таблиц/table_row единые attrs:
      - caption_num
      - caption_tail
      - header_preview (по первой строке таблицы, если она есть)
    Ничего не удаляет и не переименовывает.
    """
    groups: Dict[str, List[int]] = {}
    for i, sec in enumerate(sections):
        et = (sec.get("element_type") or "").lower()
        sp = sec.get("section_path") or ""
        txt = sec.get("text") or ""
        is_table_like = et in {"table", "table_row"} or txt.startswith("[Таблица]")
        if not is_table_like:
            continue
        base = _base_table_name(sp)
        groups.setdefault(base, []).append(i)

    for base, idxs in groups.items():
        tail = _last_segment(base)
        num, title = _parse_table_caption_from(tail)
        if not num or not title:
            first = sections[idxs[0]]
            n2, t2 = _parse_table_caption_from(first.get("text") or "")
            num = num or n2
            title = title or t2

        preview = None
        for i in idxs:
            sp_i = sections[i].get("section_path") or ""
            et_i = (sections[i].get("element_type") or "").lower()
            is_row = (" [row " in sp_i) or (et_i == "table_row")
            if not is_row:
                continue
            first_line = (sections[i].get("text") or "").split("\n")[0]
            if first_line:
                preview = header_preview_from_row(first_line)
                break

        for i in idxs:
            attrs = _ensure_attrs(sections[i])
            if num and not attrs.get("caption_num"):
                attrs["caption_num"] = num
            if title and not attrs.get("caption_tail"):
                attrs["caption_tail"] = title
            if preview and not attrs.get("header_preview"):
                attrs["header_preview"] = preview


def _enrich_figures(sections: List[Dict[str, Any]]) -> None:
    """
    Проставляет для фигуры attrs:
      - caption_num
      - caption_tail
    """
    for sec in sections:
        et = (sec.get("element_type") or "").lower()
        txt = sec.get("text") or ""
        sp = sec.get("section_path") or ""
        if et != "figure" and not (txt.startswith("[Рисунок]") or re.search(r"(?i)\bрис\.", sp)):
            continue

        num, tail = _parse_figure_caption_from(sp or txt)
        attrs = _ensure_attrs(sec)
        if num and not attrs.get("caption_num"):
            attrs["caption_num"] = num
        if tail and not attrs.get("caption_tail"):
            attrs["caption_tail"] = tail

        if is_table_image_section(sec):
            attrs.setdefault("flags", [])
            if "table_like_image" not in attrs["flags"]:
                attrs["flags"].append("table_like_image")


def _split_references_block(sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Если обнаружен «сплошной» блок списка литературы (один-два больших абзаца),
    разрезаем его на отдельные элементы element_type='reference'.
    Если в документе уже есть отдельные reference – ничего не делаем.
    Возвращаем НОВЫЙ список секций (оригинал не мутируем).
    """
    if any((sec.get("element_type") or "").lower() == "reference" for sec in sections):
        return sections

    out: List[Dict[str, Any]] = []
    changed = False

    for sec in sections:
        sp = sec.get("section_path") or ""
        txt = (sec.get("text") or "").strip()
        if not txt or not _is_reference_section_name(sp):
            out.append(sec)
            continue

        lines = [ln for ln in txt.splitlines() if ln.strip()]
        found_any = False

        for ln in lines:
            m = _REF_LINE_RE.match(ln)
            if not m:
                continue
            found_any = True
            idx = m.group(1) or m.group(2)
            body = m.group(3).strip()
            item = deepcopy(sec)
            item["text"] = body
            item["element_type"] = "reference"
            attrs = _ensure_attrs(item)
            try:
                attrs["ref_index"] = int(idx)
            except Exception:
                attrs["ref_index"] = idx
            out.append(item)

        if found_any:
            changed = True
        else:
            out.append(sec)

    return out if changed else sections


def enrich_sections(
    sections: List[Dict[str, Any]],
    *,
    doc_kind: Optional[str] = None,
    enable_ocr: bool = True,
    enable_table_ocr: bool = True,
    normalise_numbers: bool = True,
    detect_figures: bool = True,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Главная точка входа для «ингеста». Принимает список секций и возвращает обновлённый список.
    """
    if not sections:
        return sections

    dk = (doc_kind or "").lower()
    if dk in {"scan", "pdf_scan", "image_pdf"}:
        enable_ocr = True
        enable_table_ocr = True

    _enrich_tables(sections)
    if detect_figures:
        _enrich_figures(sections)
    sections = _split_references_block(sections)
    return sections


# ----------------------------- НОВОЕ: кеш и OCR -----------------------------

def _cache_dir() -> Path:
    base = Path(Cfg.UPLOAD_DIR or "./uploads")
    d = base / "_index_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _cache_path_for_sha(sha256: str) -> Path:
    return _cache_dir() / f"{sha256}.json"

def _load_cached_sections(sha256: str) -> Optional[List[Dict[str, Any]]]:
    p = _cache_path_for_sha(sha256)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def _save_cached_sections(sha256: str, sections: List[Dict[str, Any]]) -> str:
    p = _cache_path_for_sha(sha256)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(sections, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)
    return str(p)


# ----------------------------- ПУБЛИЧНАЯ ОРКЕСТРАЦИЯ -----------------------------

IndexFn = Callable[[int, str, str], Dict[str, Any]]
# сигнатура: indexer_fn(doc_id, file_path, kind) -> {"sections_count": int, "chunks_count": int, ...}

def _build_structured_sections(file_path: str, kind: str) -> List[Dict[str, Any]]:
    """
    Общая точка входа для структурированного парсинга.
    """
    k = (kind or "").lower()
    if k == "docx":
        if not parse_docx:
            raise IngestError("parse_docx недоступен (не импортирован parsing.py).")
        sections = parse_docx(file_path)
    elif k == "pdf":
        if not parse_pdf:
            raise IngestError("parse_pdf недоступен (не импортирован parsing.py).")
        sections = parse_pdf(file_path)
    elif k == "doc":
        if not parse_doc:
            raise IngestError("parse_doc недоступен (не импортирован parsing.py).")
        sections = parse_doc(file_path)
    elif k == "txt":
        # Простейший одно-секционный парсинг
        body = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        sections = [{
            "title": "Документ",
            "level": 1,
            "text": body,
            "page": None,
            "section_path": "Документ",
            "element_type": "paragraph",
            "attrs": {}
        }]
    else:
        # fallback как txt
        body = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        sections = [{
            "title": "Документ",
            "level": 1,
            "text": body,
            "page": None,
            "section_path": "Документ",
            "element_type": "paragraph",
            "attrs": {}
        }]

    # Дополнительное обогащение (таблицы/рисунки/референсы)
    sections = enrich_sections(sections, doc_kind=k)
    return sections


def ingest_document(
    user_id: int,
    file_path: str,
    *,
    kind: Optional[str] = None,
    file_uid: Optional[str] = None,
    content_sha256: Optional[str] = None,
    layout_profile: Optional[str] = None,
    force_reindex: bool = False,
    indexer_fn: Optional[IndexFn] = None,
) -> Dict[str, Any]:
    """
    Атомарный оркестратор «файл → индексация → READY».

    - Идемпотентно переиспользует документ по SHA256/UID (если разрешено).
    - Переводит FSM-состояния: INDEXING → (READY|IDLE при ошибке).
    - Строит структурированный индекс, делает мягкий OCR изображений и кеширует по SHA.
    - Вызывает ваш индексатор (indexer_fn), который обязан записать чанки/эмбеддинги.
    - На успехе выставляет индексатор-версию документа.

    Возвращает словарь с полями:
        {
          "doc_id": int,
          "reused": bool,
          "kind": str,
          "sha256": str,
          "file_uid": str|None,
          "indexer_version": int|None,
          "index_result": dict|None
        }
    """
    if not os.path.isfile(file_path):
        raise IngestError(f"Файл не найден: {file_path}")

    # 1) Определяем тип и SHA
    resolved_kind = kind or _detect_kind(file_path)
    sha256 = content_sha256 or _file_sha256(file_path)

    # 2) Идемпотентность: если документ уже есть и актуален — не индексируем заново
    reused_doc_id: Optional[int] = None
    if Cfg.INGEST_IDEMPOTENCY_BY_HASH and not force_reindex:
        reused_doc_id = find_existing_document(owner_id=user_id, content_sha256=sha256, file_uid=file_uid)

    if reused_doc_id and not force_reindex:
        # обновим метаданные (пути/sha/uid/профиль)
        update_document_meta(
            reused_doc_id,
            path=file_path,
            content_sha256=sha256,
            file_uid=file_uid,
            layout_profile=layout_profile,
        )
        # сразу помечаем как READY — документ уже есть; реиндексация не требуется
        finish_indexing_success(user_id, doc_id=reused_doc_id)
        return {
            "doc_id": reused_doc_id,
            "reused": True,
            "kind": resolved_kind,
            "sha256": sha256,
            "file_uid": file_uid,
            "indexer_version": None,   # не меняли
            "index_result": {
                "note": "document reused by hash",
                "prepared_sections_cache": str(_cache_path_for_sha(sha256)),
            },
        }

    # 3) Создаём запись документа и переходим в INDEXING
    doc_id = insert_document(
        owner_id=user_id,
        kind=resolved_kind,
        path=file_path,
        content_sha256=sha256,
        file_uid=file_uid,
    )
    ingest_job_id = str(uuid.uuid4())
    start_indexing(user_id, doc_id=doc_id, ingest_job_id=ingest_job_id)

    # 4) Структурированный парсинг + OCR + кеш по хэшу
    try:
        cache_used = False
        sections: Optional[List[Dict[str, Any]]] = None
        cache_path = _cache_path_for_sha(sha256)

        if not force_reindex:
            sections = _load_cached_sections(sha256)
            cache_used = sections is not None

        if not sections:
            # строим с нуля
            sections = _build_structured_sections(file_path, resolved_kind)
            _save_cached_sections(sha256, sections)

        # 5) Запускаем индексатор пользователя (опционально)
        index_result: Dict[str, Any] = {
            "sections_count": len(sections or []),
            "chunks_count": None,
            "prepared_sections_cache": str(cache_path),
            "cache_used": cache_used,
        }

        if indexer_fn:
            # индексатор обязан сам записать чанки/эмбеддинги в БД (может прочитать sections из кеша по sha)
            user_idx_res = indexer_fn(doc_id, file_path, resolved_kind) or {}
            # негрубо смёржим мета
            index_result.update({k: v for k, v in user_idx_res.items() if k not in index_result or v is not None})

        # 6) Фиксируем версию индексатора и помечаем READY
        set_document_indexer_version(doc_id)
        finish_indexing_success(user_id, doc_id=doc_id)

        return {
            "doc_id": doc_id,
            "reused": False,
            "kind": resolved_kind,
            "sha256": sha256,
            "file_uid": file_uid,
            "indexer_version": int(CURRENT_INDEXER_VERSION),
            "index_result": index_result,
        }

    except Exception as e:
        # Ошибка индексации — переводим в IDLE и пробрасываем исключение вверх (по желанию)
        finish_indexing_error(user_id, error_message=str(e))
        raise IngestError(f"Ошибка индексации: {e}") from e
