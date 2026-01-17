# app/ingest_orchestrator.py
from __future__ import annotations

import os
import re
import uuid
import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Callable
from copy import deepcopy
from pathlib import Path

from .config import Cfg, ProcessingState  # ProcessingState импортирован для внешней совместимости
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
    from .parsing_new import parse_docx, parse_doc, save_upload
except Exception:
    parse_docx = parse_pdf = parse_doc = None  # будем валить осмысленной ошибкой

# ----------------------------- КОНСТАНТЫ / РЕГЕКСЫ -----------------------------

# Подписи к таблицам: «Таблица 2.3 — Название», допускаем буквы «А.1», «П1.2»
_TABLE_CAP_RE = re.compile(
    r"(?i)\bтаблица\s+([A-Za-zА-Яа-я]?\s*\d+(?:[.,]\d+)*)\b(?:\s*[—\-–:]\s*(.+))?"
)

# Подписи к рисункам/диаграммам: «Рис. 2.1», «Диаграмма 3», «График 1.2», «Figure 1.2», «Chart 4» и т.п.
_FIG_CAP_RE = re.compile(
    r"(?i)\b(?:рис(?:\.|унок)?|диаграмма|график|схема|figure|fig\.?|chart|graph|diagram|plot)\s*(?:№\s*)?"
    r"([A-Za-zА-Яа-я]?\s*\d+(?:[.,]\d+)*)\b(?:\s*[—\-–:]\s*(.+))?"
)

# Строки библиографии: "[12] Текст..." или "12) Текст..." или "12. Текст..."
_REF_LINE_RE = re.compile(r"^\s*(?:\[(\d+)\]|(\d{1,3})[.)])\s+(.+)$")

# Версия логики подготовки секций (для инвалидации кеша)
_PREPARED_SECTIONS_VERSION = 4

# ----------------------------- УТИЛИТЫ ОРКЕСТРАЦИИ -----------------------------

class IngestError(Exception):
    """Обёртка ошибок на этапе индексации (для единообразного репортинга в FSM)."""


def _file_sha256(path: str, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
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
    if ext in {".doc", ".rtf"}:
        return "doc"
    return fallback


# ----------------------------- ОБОГАЩЕНИЕ -----------------------------

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
    return any(k in sp for k in ("источник", "источники", "литератур", "библиограф", "reference", "references", "bibliograph"))

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

def _classify_chart_from_tail(tail: str) -> Tuple[str, Optional[str]]:
    """
    Возвращает (figure_kind, chart_type_hint) по тексту хвоста подписи.
    figure_kind: 'chart' | 'figure'
    chart_type_hint: 'pie'|'bar'|'line'|'stacked_bar'|'histogram'|None
    """
    t = (tail or "").lower()
    if any(k in t for k in ("кругов", "кольцев", "pie", "donut", "ring")):
        return "chart", "pie"
    if any(k in t for k in ("столбч", "bar", "column", "колонн", "diagram bar", "bar chart")):
        return "chart", "bar"
    if any(k in t for k in ("линейн", "line", "trend", "time series")):
        return "chart", "line"
    if any(k in t for k in ("stacked", "составн", "100%", "100 %", "накоп", "stacked bar")):
        return "chart", "stacked_bar"
    if any(k in t for k in ("гистограм", "histogram", "freq", "распределен")):
        return "chart", "histogram"
    if any(w in t for w in ("диаграм", "график", "chart", "graph", "diagram", "plot")):
        return "chart", None
    return "figure", None


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
    Проставляет для figure/diagram атрибуты:
      - caption_num
      - caption_tail
      - figure_kind: 'chart' | 'figure'
      - chart_type_hint: 'pie'|'bar'|'line'|'stacked_bar'|'histogram'|None
      - flags: включает 'table_like_image' по эвристике
    """
    for sec in sections:
        et = (sec.get("element_type") or "").lower()
        txt = sec.get("text") or ""
        sp = sec.get("section_path") or ""
        if et != "figure" and not (txt.startswith("[Рисунок]") or re.search(r"(?i)\bрис\.", sp) or re.search(r"(?i)\b(диаграмма|график|chart|figure|diagram|plot)\b", sp)):
            continue

        num, tail = _parse_figure_caption_from(sp or txt)
        attrs = _ensure_attrs(sec)
        if num and not attrs.get("caption_num"):
            attrs["caption_num"] = num
        if tail and not attrs.get("caption_tail"):
            attrs["caption_tail"] = tail

        fk, hint = _classify_chart_from_tail(tail or "")
        attrs.setdefault("figure_kind", fk)
        if hint and not attrs.get("chart_type_hint"):
            attrs["chart_type_hint"] = hint

        if is_table_image_section(sec):
            attrs.setdefault("flags", [])
            if "table_like_image" not in attrs["flags"]:
                attrs["flags"].append("table_like_image")


def _split_references_block(sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Если обнаружен «сплошной» блок списка литературы (один-два больших абзаца),
    разрезаем его на отдельные элементы element_type='reference' (поддержка многострочных источников).
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
        current = None  # {"idx": "...", "body": ["..."]}

        def _flush():
            nonlocal current, out, changed
            if not current:
                return
            item = deepcopy(sec)
            item["text"] = " ".join(current["body"]).strip()
            item["element_type"] = "reference"
            attrs = _ensure_attrs(item)
            try:
                attrs["ref_index"] = int(current["idx"])
            except Exception:
                attrs["ref_index"] = current["idx"]
            attrs.setdefault("source", "split_block")
            out.append(item)
            current = None

        for ln in lines:
            m = _REF_LINE_RE.match(ln)
            if m:
                # новая запись
                _flush()
                found_any = True
                idx = (m.group(1) or m.group(2) or "").strip()
                body = (m.group(3) or "").strip()
                current = {"idx": idx, "body": [body] if body else []}
            else:
                # продолжение предыдущей записи
                if current:
                    current["body"].append(ln.strip())

        _flush()
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

    # 1) Сначала обогащаем (могут появиться новые секции)
    _enrich_tables(sections)
    if detect_figures:
        _enrich_figures(sections)
    sections = _split_references_block(sections)

    # 2) В конце — единая нормализация ролей (после всех изменений списка)
    sections = _assign_section_roles(sections)
    return sections


def _assign_section_roles(sections):
    """
    Проставляет sections[i]["attrs"]["role"]:
      - intro
      - chapter_1, chapter_2, ...
      - conclusion
      - references
      - appendix

    Важно: функция самодостаточная — все regex и константы внутри,
    чтобы не ловить "NameError/Undefined name" после вставки.
    """
    import re

    if not sections:
        return sections

    # роли
    ROLE_INTRO = "intro"
    ROLE_CONCLUSION = "conclusion"
    ROLE_REFERENCES = "references"
    ROLE_APPENDIX = "appendix"
    ROLE_CH_PREFIX = "chapter_"

    # паттерны
    INTRO_RE = re.compile(
        r"(?i)\b("
        r"введение|вступление|"
        r"общая\s+характеристика\s+работы|"
        r"общая\s+характеристика\s+вкр|"
        r"общая\s+характеристика\s+исследования"
        r")\b"
    )
    CONCLUSION_RE = re.compile(
        r"(?i)\b("
        r"заключение|"
        r"выводы\s+и\s+рекомендации|"
        r"итоги\s+работы|"
        r"общие\s+выводы|"
        r"результаты\s+исследования"
        r")\b"
    )
    REFERENCES_RE = re.compile(r"(?i)\b(список\s+литературы|библиограф(ия|ический)\s+список|источники|литература)\b")
    APPENDIX_RE = re.compile(r"(?i)\b(приложени[ея]|appendix)\b")
    CHAPTER_RE = re.compile(
        r"(?i)\b("
        r"глава|раздел|chapter|section|"
        r"часть|part"
        r")\s*([0-9]{1,2})\b"
    )

    LEADING_NUM_RE = re.compile(
        r"^\s*([0-9]{1,2})(?:[.\s]+|$)"
    )

        # НОВОЕ: пересчитываем роли “с нуля”, чтобы кеш/предыдущие значения не мешали
    for sec in sections:
        if "attrs" not in sec or not isinstance(sec["attrs"], dict):
            sec["attrs"] = {}
        sec["attrs"].pop("role", None)
        sec["attrs"].pop("chapter_num", None)

    current_role = None

    for sec in sections:
        attrs = sec["attrs"]

        title = (sec.get("title") or "").strip()
        spath = (sec.get("section_path") or "").strip()
        hay = f"{title}\n{spath}".strip().lower()

        detected_role = None

        # 1) Приоритет: “служебные” разделы (они важнее, чем leading-number)
        if INTRO_RE.search(hay):
            detected_role = ROLE_INTRO
        elif CONCLUSION_RE.search(hay):
            detected_role = ROLE_CONCLUSION
        elif REFERENCES_RE.search(hay):
            detected_role = ROLE_REFERENCES
        elif APPENDIX_RE.search(hay):
            detected_role = ROLE_APPENDIX
        else:
            # 2) Затем главы
            m = CHAPTER_RE.search(hay)
            if m:
                detected_role = f"{ROLE_CH_PREFIX}{m.group(2)}"
            else:
                # 3) Фолбэк: ведущая цифра в начале заголовка
                m2 = LEADING_NUM_RE.match(title)
                if m2:
                    detected_role = f"{ROLE_CH_PREFIX}{m2.group(1)}"

        if detected_role:
            current_role = detected_role

        # Всегда протягиваем текущую роль (после “обнуления” выше)
        if current_role:
            attrs["role"] = current_role
            if current_role.startswith(ROLE_CH_PREFIX):
                try:
                    attrs["chapter_num"] = int(current_role.replace(ROLE_CH_PREFIX, ""))
                except Exception:
                    pass

        sec["attrs"] = attrs

    return sections

# ----------------------------- НОВОЕ: кеш и OCR -----------------------------

def _cache_dir() -> Path:
    base = Path(Cfg.UPLOAD_DIR or "./uploads")
    d = base / "_index_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _cache_path_for_sha(sha256: str) -> Path:
    return _cache_dir() / f"{sha256}.v{_PREPARED_SECTIONS_VERSION}.json"

def _load_cached_sections(sha256: str) -> Optional[List[Dict[str, Any]]]:
    p = _cache_path_for_sha(sha256)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            return None
        # простая валидация минимальных ключей
        for x in data[:50]:
            if not isinstance(x, dict):
                return None
        return data
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
            sections = _build_structured_sections(file_path, resolved_kind)

        # НОВОЕ: единый путь нормализации/обогащения секций
        sections = enrich_sections(
            sections,
            doc_kind=resolved_kind,
            enable_ocr=True,
            enable_table_ocr=True,
            detect_figures=True,
        )

        # Сохраняем уже “готовые” секции
        _save_cached_sections(sha256, sections)

        # 5) Запускаем индексатор пользователя (опционально)
                # 5) Запускаем индексатор пользователя (опционально)
        has_chart_data = any(
            isinstance(sec.get("attrs"), dict) and (
                "chart_data" in sec["attrs"] or "chart_matrix" in sec["attrs"]
            )
            for sec in (sections or [])
        )

        index_result: Dict[str, Any] = {
            "sections_count": len(sections or []),
            "chunks_count": None,
            "prepared_sections_cache": str(cache_path),
            "cache_used": cache_used,
            # новый флаг: в подготовленных секциях есть данные диаграмм
            "has_chart_data": has_chart_data,
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
