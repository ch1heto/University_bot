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
    values: Any = []
    chart_source: Optional[str] = None
    vision_desc_raw: Optional[Dict[str, Any]] = None
    vision_values_raw: Optional[Dict[str, Any]] = None
    warnings: list[str] = []

    # ---- 1️⃣ Описание изображения (VISION)
    if getattr(fig, "image_path", None):
        try:
            v = vision_describe(fig.image_path, lang=lang)
            # если модель дала описание — используем его; иначе падаем обратно на подпись
            description = _safe_clean(v.get("description", "")) or caption
            vision_desc_raw = v
        except Exception as e:
            log.warning("vision_describe failed for %s: %s", getattr(fig, "image_path", None), e)
            description = caption or "Описание не удалось извлечь."
            warnings.append("vision_describe_failed")
    else:
        description = caption or ""
        if not description:
            warnings.append("no_description")


    # ---- 2️⃣ Числовые данные
        # ---- 2️⃣ Числовые данные
    if kind in ("chart", "table"):
        used_docx = False

        # 2.1. Точные данные из DOCX (приоритетнее OCR)
        if getattr(fig, "chart_data", None):
            values = fig.chart_data
            chart_source = "docx-chart"
            used_docx = True
        elif getattr(fig, "table_data", None):
            values = fig.table_data
            chart_source = "docx-table"
            used_docx = True

        # 2.2. Фолбэк в OCR, если в DOCX данных нет
        if not used_docx:
            if getattr(fig, "image_path", None):
                try:
                    vvals = vision_extract_values(
                        fig.image_path,
                        caption_hint=caption,
                        lang=lang,
                    )
                    values = vvals.get("data", []) or []
                    chart_source = chart_source or "vision"
                    vision_values_raw = vvals
                    if not values:
                        warnings.append("vision_no_values")
                except Exception as e:
                    log.warning("vision_extract_values failed for %s: %s", fig.image_path, e)
                    warnings.append("vision_extract_failed")
                    values = []
            else:
                # ни DOCX-данных, ни изображения для OCR
                warnings.append("no_source_for_values")
                values = []
    else:
        # Попробовать OCR-извлечение чисел для произвольного изображения
        if getattr(fig, "image_path", None):
            try:
                vvals = vision_extract_values(
                    fig.image_path,
                    caption_hint=caption,
                    lang=lang,
                )
                values = vvals.get("data", []) or []
                chart_source = "vision" if values else None
                vision_values_raw = vvals
                if not values:
                    warnings.append("vision_no_values")
            except Exception as e:
                log.warning("vision_extract_values failed for %s: %s", fig.image_path, e)
                warnings.append("vision_extract_failed")
                values = []
        else:
            values = []
            warnings.append("no_image_for_values")


    # ---- 3️⃣ Итоговая структура
    return {
        "id": getattr(fig, "id", None),
        "doc_id": getattr(fig, "doc_id", None),
        "kind": kind,
        "title": title,
        "caption": caption,
        "description": description,
        "values": values,
        "chart_source": chart_source,
        # для обратной совместимости: raw от описания
        "vision_raw": vision_desc_raw,
        # отдельно сырые данные от OCR-извлечения чисел
        "vision_values_raw": vision_values_raw,
        "warnings": warnings,
    }

