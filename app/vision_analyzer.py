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


def _safe_clean(text: Any) -> str:
    if not text:
        return ""
    if callable(text):
        return ""
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            return ""
    return text.replace("\xa0", " ").strip()



def analyze_figure(fig, lang: str = "ru") -> Dict[str, Any]:
    """
    Универсальный анализ изображения:
    - title
    - values (если таблица или диаграмма — из DOCX; иначе — из vision_extract_values)
    - description (vision_describe)

    Поддерживает fig как:
    - объект с атрибутами (title/caption/image_path/chart_data/table_data)
    - dict с ключами
    - str (путь к изображению)
    """

    # --- Нормализуем вход (fig может быть объектом / dict / str)
    image_path: Optional[str] = None
    chart_data = None
    table_data = None
    title_raw = None
    caption_raw = None
    fig_id = None
    fig_doc_id = None

    if isinstance(fig, str):
        image_path = fig
    elif isinstance(fig, dict):
        image_path = fig.get("image_path") or fig.get("abs_path") or fig.get("path")
        chart_data = fig.get("chart_data")
        table_data = fig.get("table_data")
        title_raw = fig.get("title")
        caption_raw = fig.get("caption")
        fig_id = fig.get("id")
        fig_doc_id = fig.get("doc_id")
    else:
        image_path = (
            getattr(fig, "image_path", None)
            or getattr(fig, "abs_path", None)
            or getattr(fig, "path", None)
        )
        chart_data = getattr(fig, "chart_data", None)
        table_data = getattr(fig, "table_data", None)
        title_raw = getattr(fig, "title", None)
        caption_raw = getattr(fig, "caption", None)
        fig_id = getattr(fig, "id", None)
        fig_doc_id = getattr(fig, "doc_id", None)

    # --- Определяем тип
    if chart_data:
        kind = "chart"
    elif table_data:
        kind = "table"
    else:
        kind = "other"

    # Важно: title_raw может оказаться callable (например, если fig == str и кто-то лезет в fig.title)
    title = _safe_clean(title_raw or caption_raw or "")
    caption = _safe_clean(caption_raw)

    description = ""
    values: Any = []
    chart_source: Optional[str] = None
    vision_desc_raw: Optional[Dict[str, Any]] = None
    vision_values_raw: Optional[Dict[str, Any]] = None
    warnings: list[str] = []

    # ---- 1️⃣ Описание изображения (VISION)
    if image_path:
        try:
            v = vision_describe(image_path, lang=lang)
            vision_desc_raw = v
            # если модель дала описание — используем его; иначе падаем обратно на подпись
            description = _safe_clean(v.get("description", "")) or caption
            if not description:
                warnings.append("vision_empty_description")
        except Exception as e:
            log.warning("vision_describe failed for %s: %s", image_path, e)
            description = caption or "Описание не удалось извлечь."
            warnings.append("vision_describe_failed")
    else:
        description = caption or ""
        if not description:
            warnings.append("no_description")

    # ---- 2️⃣ Числовые данные
    if kind in ("chart", "table"):
        used_docx = False

        # 2.1. Точные данные из DOCX (приоритетнее OCR)
        if chart_data:
            values = chart_data
            chart_source = "docx-chart"
            used_docx = True
        elif table_data:
            values = table_data
            chart_source = "docx-table"
            used_docx = True

        # 2.2. Фолбэк в OCR, если в DOCX данных нет
        if not used_docx:
            if image_path:
                try:
                    vvals = vision_extract_values(
                        image_path,
                        caption_hint=caption,
                        lang=lang,
                    )
                    vision_values_raw = vvals
                    values = vvals.get("data", []) or []
                    chart_source = chart_source or "vision"
                    if not values:
                        warnings.append("vision_no_values")
                except Exception as e:
                    log.warning("vision_extract_values failed for %s: %s", image_path, e)
                    warnings.append("vision_extract_failed")
                    values = []
            else:
                # ни DOCX-данных, ни изображения для OCR
                warnings.append("no_source_for_values")
                values = []
    else:
        # Попробовать OCR-извлечение чисел для произвольного изображения
        if image_path:
            try:
                vvals = vision_extract_values(
                    image_path,
                    caption_hint=caption,
                    lang=lang,
                )
                vision_values_raw = vvals
                values = vvals.get("data", []) or []
                chart_source = "vision" if values else None
                if not values:
                    warnings.append("vision_no_values")
            except Exception as e:
                log.warning("vision_extract_values failed for %s: %s", image_path, e)
                warnings.append("vision_extract_failed")
                values = []
        else:
            values = []
            warnings.append("no_image_for_values")

    # ---- 3️⃣ Итоговая структура
    return {
        "id": fig_id,
        "doc_id": fig_doc_id,
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
