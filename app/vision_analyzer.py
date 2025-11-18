# app/vision_analyzer.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .polza_client import vision_describe, vision_extract_values

log = logging.getLogger("vision_analyzer")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    log.addHandler(h)
log.setLevel(logging.INFO)


def _safe_clean(text: Optional[str]) -> str:
    if not text:
        return ""
    return text.replace("\xa0", " ").strip()


def analyze_figure(fig, lang: str = "ru") -> Dict[str, Any]:
    """
    Универсальный анализ изображения:
    - title
    - values (если таблица или диаграмма — из DOCX; иначе — из vision_extract_values)
    - description (vision_describe)
    """

    # Определяем тип
    if getattr(fig, "chart_data", None):
        kind = "chart"
    elif getattr(fig, "table_data", None):
        kind = "table"
    else:
        kind = "other"

    title = _safe_clean(fig.title or fig.caption or "")
    caption = _safe_clean(fig.caption)
    description = ""
    values = []
    chart_source = None
    vision_raw = None
    warnings = []

    # ---- 1️⃣ Описание изображения (VISION)
    if fig.image_path:
        try:
            v = vision_describe(fig.image_path, lang=lang)
            description = _safe_clean(v.get("description", "")) or caption
            vision_raw = v
        except Exception as e:
            log.warning(f"vision_describe failed: {e}")
            description = caption or "Описание не удалось извлечь."
    else:
        description = caption or ""

    # ---- 2️⃣ Числовые данные
    if kind in ("chart", "table"):
        # Точные данные из DOCX
        if getattr(fig, "chart_data", None):
            values = fig.chart_data
            chart_source = "docx"
        elif getattr(fig, "table_data", None):
            values = fig.table_data
            chart_source = "docx"
    else:
        # Попробовать OCR-извлечение чисел
        try:
            vvals = vision_extract_values(fig.image_path, caption_hint=caption)
            values = vvals.get("data", [])
            vision_raw = vvals
        except Exception:
            values = []

    # ---- 3️⃣ Итоговая структура
    return {
        "id": fig.id,
        "doc_id": fig.doc_id,
        "kind": kind,
        "title": title,
        "caption": caption,
        "description": description,
        "values": values,
        "chart_source": chart_source,
        "vision_raw": vision_raw,
        "warnings": warnings,
    }
