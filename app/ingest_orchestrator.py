from __future__ import annotations

import re
from typing import List, Dict, Any, Optional, Tuple
from copy import deepcopy

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


# ----------------------------- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ -----------------------------

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


# ----------------------------- ОБОГАЩЕНИЕ ТАБЛИЦ -----------------------------

def _enrich_tables(sections: List[Dict[str, Any]]) -> None:
    """
    Проставляет для таблиц/table_row единые attrs:
      - caption_num
      - caption_tail
      - header_preview (по первой строке таблицы, если она есть)
    Ничего не удаляет и не переименовывает.
    """
    # Сгруппируем по базовому имени секции таблицы
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
        # 1) Извлекаем номер/название из хвоста базового имени или из текста
        tail = _last_segment(base)
        num, title = _parse_table_caption_from(tail)
        if not num or not title:
            # fallback: ищем по первому из группы
            first = sections[idxs[0]]
            n2, t2 = _parse_table_caption_from(first.get("text") or "")
            num = num or n2
            title = title or t2

        # 2) Берём превью по первой строке (ищем первую секцию row)
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

        # 3) Проставляем attrs для всех секций группы
        for i in idxs:
            attrs = _ensure_attrs(sections[i])
            if num and not attrs.get("caption_num"):
                attrs["caption_num"] = num
            if title and not attrs.get("caption_tail"):
                attrs["caption_tail"] = title
            if preview and not attrs.get("header_preview"):
                attrs["header_preview"] = preview


# ----------------------------- ОБОГАЩЕНИЕ РИСУНКОВ -----------------------------

def _enrich_figures(sections: List[Dict[str, Any]]) -> None:
    """
    Проставляет для фигуры attrs:
      - caption_num
      - caption_tail
    Изображения не трогаем (предполагается, что парсер положил их в attrs.images;
    если нет — оставляем как есть, чтобы не ломать пайплайн).
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

        # Если это явно картинка таблицы (скан), помечаем флаг — downstream-логика может включить OCR
        if is_table_image_section(sec):
            attrs.setdefault("flags", [])
            if "table_like_image" not in attrs["flags"]:
                attrs["flags"].append("table_like_image")


# ----------------------------- ОБОГАЩЕНИЕ ИСТОЧНИКОВ -----------------------------

def _split_references_block(sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Если обнаружен «сплошной» блок списка литературы (один-два больших абзаца),
    разрезаем его на отдельные элементы element_type='reference'.
    Если в документе уже есть отдельные reference – ничего не делаем.
    Возвращаем НОВЫЙ список секций (оригинал не мутируем).
    """
    # Если уже есть элементы reference — выходим
    if any((sec.get("element_type") or "").lower() == "reference" for sec in sections):
        return sections

    out: List[Dict[str, Any]] = []
    changed = False

    for sec in sections:
        sp = sec.get("section_path") or ""
        txt = (sec.get("text") or "").strip()

        # не подходит — просто переносим
        if not txt or not _is_reference_section_name(sp):
            out.append(sec)
            continue

        # пробуем построчно распарсить
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
            # строковая разметка не найдена — переносим как есть
            out.append(sec)

    return out if changed else sections


# ----------------------------- ПУБЛИЧНОЕ API -----------------------------

def enrich_sections(
    sections: List[Dict[str, Any]],
    *,
    doc_kind: Optional[str] = None,          # тип документа (docx/pdf/pdf_scan и т.п.)
    enable_ocr: bool = True,                 # флаги на будущее; не ломают старую логику
    enable_table_ocr: bool = True,
    normalise_numbers: bool = True,
    detect_figures: bool = True,
    **kwargs,                                # чтобы не падать от «неожиданных» аргументов
) -> List[Dict[str, Any]]:
    """
    Главная точка входа для «ингеста». Принимает список секций (как возвращают parse_docx/pdf/doc)
    и возвращает обновлённый список секций.
    Параметры doc_kind/флаги опциональны и безопасно игнорируются, если не используются.
    """
    if not sections:
        return sections

    # лёгкая настройка по типу документа: для скан-PDF усиливаем эвристику таблиц
    dk = (doc_kind or "").lower()
    if dk in {"scan", "pdf_scan", "image_pdf"}:
        enable_ocr = True
        enable_table_ocr = True

    # обогащение таблиц/рисунков как и раньше (флаги пока информативные)
    _enrich_tables(sections)
    if detect_figures:
        _enrich_figures(sections)

    # разбиение «Списка литературы» на элементы reference
    sections = _split_references_block(sections)
    return sections
