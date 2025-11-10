# app/templates.py
# -*- coding: utf-8 -*-
"""
Шаблоны промптов для мультимодального анализа изображений.
Два основных сценария:
  1) Describe — осмысленное текстовое описание (без выдуманных чисел).
  2) Extract  — извлечение структурированных значений (label → value/unit/conf).

Каждый билд-функция возвращает dict с двумя полями:
  {
    "system": <строка для системного сообщения>,
    "user":   <строка для пользовательского сообщения>
  }

Политика:
  - RU/EN поддержка по флагу lang.
  - Жёсткая просьба вернуть ТОЛЬКО JSON в Extract/Describe (упрощает парсинг).
  - Контекст и подпись аккуратно укорачиваются, чтобы не раздувать промпт.

Использование (пример):
  from .templates import build_describe_prompt, build_extract_prompt
  msg = build_describe_prompt(caption="Рисунок 4 — ...", context="Раздел 2.3 ...")
  # msg["system"], msg["user"] → дальше передать в polza_client вместе с картинкой.
"""

from __future__ import annotations

from typing import Dict, Optional
from .config import Cfg

# ----------------------------- helpers -----------------------------

def _clean_text(s: Optional[str]) -> str:
    if not s:
        return ""
    return " ".join(str(s).replace("\xa0", " ").split()).strip()

def _shorten(s: str, max_len: int) -> str:
    s = _clean_text(s)
    if not s:
        return ""
    return (s[: max_len - 1].rstrip() + "…") if len(s) > max_len else s

def _lang_norm(lang: Optional[str]) -> str:
    l = (lang or Cfg.VISION_LANG or "ru").lower().strip()
    return "ru" if l.startswith("ru") else "en"

# ----------------------------- SYSTEM messages -----------------------------

_SYSTEM_DESCRIBE_RU = (
    "Ты помощник по анализу графиков и диаграмм. Отвечай по-русски. "
    "Не придумывай чисел, если их не видно на изображении; избегай фантазий. "
    "Если это не диаграмма, дай осмысленное краткое описание ключевого содержания."
)

_SYSTEM_DESCRIBE_EN = (
    "You are an assistant for analyzing charts and diagrams. Answer in English. "
    "Do not fabricate numbers if they are not visible in the image; avoid speculation. "
    "If it is not a chart, provide a concise meaningful description of the key content."
)

_SYSTEM_EXTRACT_RU = (
    "Ты извлекаешь значения с диаграмм по изображению и возвращаешь строго корректный JSON. "
    "Не добавляй никакого текста вне JSON. Не придумывай значения."
)

_SYSTEM_EXTRACT_EN = (
    "You extract values from charts in an image and return a strictly valid JSON. "
    "Do not add any text outside of JSON. Do not fabricate values."
)

# ----------------------------- JSON schemas (string examples) -----------------------------

def _describe_json_schema(lang: str) -> str:
    if lang == "ru":
        # двойные {{ }} чтобы оставить фигурные скобки в f-string
        return (
            "{{\n"
            '  "chart_type": "Pie|Bar|Line|Table|Image|Unknown",\n'
            '  "summary": "2–4 предложения связного описания без списков и выдуманных чисел.",\n'
            '  "comparisons": ["строка", "строка"],\n'
            '  "trends": ["строка", "строка"],\n'
            '  "caveats": ["строка"]\n'
            "}}"
        )
    else:
        return (
            "{{\n"
            '  "chart_type": "Pie|Bar|Line|Table|Image|Unknown",\n'
            '  "summary": "2–4 sentences of coherent description without lists or made-up numbers.",\n'
            '  "comparisons": ["string", "string"],\n'
            '  "trends": ["string", "string"],\n'
            '  "caveats": ["string"]\n'
            "}}"
        )

def _extract_json_schema(lang: str, max_items: int) -> str:
    if lang == "ru":
        return (
            "{{\n"
            '  "values": [\n'
            '    { "label": "строка", "value": 12.3, "unit": "%|шт|ед|null", "conf": 0.0 }\n'
            f"  ],\n  \"notes\": [\"до {max_items} элементов; value — число с точкой; unit для процентов — '%'\"]\n"
            "}}"
        )
    else:
        return (
            "{{\n"
            '  "values": [\n'
            '    { "label": "string", "value": 12.3, "unit": "%|pcs|units|null", "conf": 0.0 }\n'
            f"  ],\n  \"notes\": [\"up to {max_items} items; value is a number with dot; unit for percentages is '%'\"]\n"
            "}}"
        )

# ----------------------------- BUILDERS -----------------------------

def build_describe_prompt(
    caption: Optional[str] = None,
    context: Optional[str] = None,
    intent: str = "describe",
    lang: Optional[str] = None,
    sentences_min: Optional[int] = None,
    sentences_max: Optional[int] = None,
    require_json: Optional[bool] = None,
    max_context_len: int = 600,
) -> Dict[str, str]:
    """
    Собирает промпт для шага Describe.
    Возвращает dict {"system": ..., "user": ...}
    """
    L = _lang_norm(lang)
    sys = _SYSTEM_DESCRIBE_RU if L == "ru" else _SYSTEM_DESCRIBE_EN

    smin = sentences_min if sentences_min is not None else Cfg.VISION_DESCRIBE_SENTENCES_MIN
    smax = sentences_max if sentences_max is not None else Cfg.VISION_DESCRIBE_SENTENCES_MAX
    strict_json = Cfg.VISION_JSON_STRICT if require_json is None else bool(require_json)

    cap = _shorten(caption or "", 400)
    ctx = _shorten(context or "", max_context_len)
    intent = _clean_text(intent or "describe")

    if L == "ru":
        header = (
            f"Задача: кратко описать изображение ({smin}–{smax} предложений), "
            "без выдуманных чисел и без списков."
        )
        ctx_block = (
            f'- Подпись: "{cap or "(нет)"}"\n'
            f'- Окружение: "{ctx or "(нет)"}"\n'
            f"- Интент: {intent or 'describe'}"
        )
        req = (
            "Требования:"
            "\n- Определи тип диаграммы (если возможно) и суть: что сравнивается/распределяется."
            "\n- Укажи, что больше/меньше/какой общий тренд, без точных значений."
            "\n- Если это не диаграмма, опиши ключевой визуальный смысл."
        )
        schema = "Схема ответа{}:\n".format(" (ТОЛЬКО JSON)" if strict_json else "") + _describe_json_schema(L)
    else:
        header = (
            f"Task: briefly describe the image ({smin}–{smax} sentences), "
            "without fabricated numbers and without bullet lists."
        )
        ctx_block = (
            f'- Caption: "{cap or "(none)"}"\n'
            f'- Context: "{ctx or "(none)"}"\n'
            f"- Intent: {intent or 'describe'}"
        )
        req = (
            "Requirements:"
            "\n- Identify the chart type (if possible) and the main idea."
            "\n- State what is larger/smaller or the overall trend, without exact numbers."
            "\n- If it is not a chart, describe the key content of the image."
        )
        schema = "Response schema{}:\n".format(" (JSON ONLY)" if strict_json else "") + _describe_json_schema(L)

    user = "\n".join([header, "", "Контекст:" if L == "ru" else "Context:", ctx_block, "", req, "", schema])

    return {"system": sys, "user": user}


def build_extract_prompt(
    caption: Optional[str] = None,
    context: Optional[str] = None,
    lang: Optional[str] = None,
    max_items: Optional[int] = None,
    conf_min: Optional[float] = None,
    percent_decimals: Optional[int] = None,
    require_json: Optional[bool] = None,
    max_context_len: int = 600,
) -> Dict[str, str]:
    """
    Собирает промпт для шага Extract (извлечение значений).
    Возвращает dict {"system": ..., "user": ...}
    """
    L = _lang_norm(lang)
    sys = _SYSTEM_EXTRACT_RU if L == "ru" else _SYSTEM_EXTRACT_EN

    cap = _shorten(caption or "", 400)
    ctx = _shorten(context or "", max_context_len)
    k_max = max_items if max_items is not None else Cfg.VISION_EXTRACT_MAX_ITEMS
    cmin = conf_min if conf_min is not None else Cfg.VISION_EXTRACT_CONF_MIN
    pdec = percent_decimals if percent_decimals is not None else Cfg.VISION_PERCENT_DECIMALS
    strict_json = Cfg.VISION_JSON_STRICT if require_json is None else bool(require_json)

    if L == "ru":
        header = "Задача: извлечь пары «метка → значение/процент» из изображения диаграммы."
        hints = (
            f"- Подпись: \"{cap or '(нет)'}\"\n"
            f"- Окружение: \"{ctx or '(нет)'}\""
        )
        rules = (
            "Правила:\n"
            f"- Верни не более {k_max} элементов; допускай value только как число (разделитель — точка).\n"
            f"- Проценты размещай в value (например 13.5) и unit=\"%\"; округление до {pdec} знаков.\n"
            "- Для каждого элемента добавь conf в [0..1]. Игнорируй элементы с низкой уверенностью.\n"
            "- Не добавляй легенды/оси как отдельные значения."
        )
        extras = (
            "Проверки:\n"
            f"- Для круговых диаграмм сумма процентов ≈ {Cfg.VISION_PIE_SUM_TARGET}% "
            f"(допуск ±{Cfg.VISION_PIE_SUM_TOLERANCE_PP} п.п.)."
        )
        schema = "Схема ответа{}:\n".format(" (ТОЛЬКО JSON)" if strict_json else "") + _extract_json_schema(L, k_max)
    else:
        header = "Task: extract pairs \"label → value/percent\" from the chart image."
        hints = (
            f"- Caption: \"{cap or '(none)'}\"\n"
            f"- Context: \"{ctx or '(none)'}\""
        )
        rules = (
            "Rules:\n"
            f"- Return at most {k_max} items; value must be a number (dot as decimal separator).\n"
            f"- Percentages go into value (e.g., 13.5) with unit=\"%\"; round to {pdec} decimals.\n"
            "- Add conf within [0..1] for each item. Ignore low-confidence items.\n"
            "- Do not include axes/legend as separate values."
        )
        extras = (
            "Checks:\n"
            f"- For pie charts, sum of percentages ≈ {Cfg.VISION_PIE_SUM_TARGET}% "
            f"(tolerance ±{Cfg.VISION_PIE_SUM_TOLERANCE_PP} p.p.)."
        )
        schema = "Response schema{}:\n".format(" (JSON ONLY)" if strict_json else "") + _extract_json_schema(L, k_max)

    user = "\n".join([header, "", "Контекст:" if L == "ru" else "Context:", hints, "", rules, extras, "", schema])

    return {"system": sys, "user": user}

# ----------------------------- Public API -----------------------------

__all__ = [
    "build_describe_prompt",
    "build_extract_prompt",
]
