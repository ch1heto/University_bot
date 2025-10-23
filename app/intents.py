# app/intents.py
from __future__ import annotations

import re
import json
import logging
from typing import Dict, List, Tuple, Optional

# Пытаемся использовать is_summary_intent из summarizer (если есть),
# иначе — локальный фолбэк с теми же эвристиками.
try:
    from .summarizer import is_summary_intent as _is_summary_intent_external  # type: ignore
except Exception:
    _is_summary_intent_external = None  # type: ignore


# --------------------------- Регексы/константы ---------------------------

# Таблицы
TABLE_ANY_RE = re.compile(r"\bтаблиц\w*|\bтабл\.\b|\bтаблица\w*|(?:^|\s)table(s)?\b", re.IGNORECASE)
# Поддерживаем: 2.1, 3, A.1, А.1, П1.2
TABLE_NUM_RE = re.compile(
    r"(?i)\bтабл(?:ица)?\.?\s*([a-zа-я]\.?[\s-]?\d+(?:[.,]\d+)*|\d+(?:[.,]\d+)*)\b"
)

# Рисунки
FIG_ANY_RE = re.compile(
    r"\b(рисун\w*|рис(?:\.|унок)?|figure|fig\.?|картин\w*|изображен\w*|диаграмм\w*|график\w*|схем\w*|иллюстрац\w*)\b",
    re.IGNORECASE
)
FIG_NUM_RE = re.compile(
    r"(?i)\b(?:рис(?:\.|унок)?|figure|fig\.?|картин\w*)\s*(?:№\s*)?(\d+(?:[.,]\d+)*)\b"
)

# Источники
SOURCES_HINT_RE = re.compile(
    r"\b(источник(?:и|ов)?|список\s+литературы|список\s+источников|библиограф\w*|references?|bibliograph\w*)\b",
    re.IGNORECASE
)

# Прочие триггеры
COUNT_HINT_RE = re.compile(r"\bсколько\b|how many", re.IGNORECASE)
WHICH_HINT_RE = re.compile(r"\bкаки(е|х)\b|\bсписок\b|\bперечисл\w*\b|\bназов\w*\b", re.IGNORECASE)

# Практическая часть (быстрый флаг)
PRACTICAL_RE = re.compile(r"(есть ли|наличие|присутствует ли|имеется ли).{0,40}практическ", re.IGNORECASE)

# ГОСТ/оформление (для удобства — отдельный флаг в интентах; обработка остаётся в bot.py)
GOST_RE = re.compile(r"\b(гост|оформлени|шрифт|межстроч|кегл|выравнивани|поля|оформить)\w*\b", re.IGNORECASE)


# --------------------------- Вспомогалки ---------------------------

def _is_summary_intent_local(text: str) -> bool:
    return bool(re.search(r"\b(суть|кратко|основн|главн|summary|overview|итог|вывод)\w*\b",
                          text or "", re.IGNORECASE))

def is_summary_intent(text: str) -> bool:
    if _is_summary_intent_external:
        try:
            return bool(_is_summary_intent_external(text))
        except Exception:
            pass
    return _is_summary_intent_local(text)

def _language_guess(text: str) -> str:
    """
    Очень грубая эвристика: если в тексте есть латинские слова и нет кириллицы — считаем en.
    Иначе ru.
    """
    t = (text or "")
    has_lat = bool(re.search(r"[a-zA-Z]{3,}", t))
    has_cyr = bool(re.search(r"[а-яА-ЯёЁ]{3,}", t))
    return "en" if (has_lat and not has_cyr) else "ru"

def _normalize_num(s: str) -> str:
    """
    'A . 1,2' -> 'A.1.2' (буквенные префиксы допускаются), пробелы убираем, запятую -> точку.
    """
    s = (s or "")
    s = s.replace(" ", "")
    s = s.replace(",", ".")
    # унифицируем возможный «A-1.2» => "A1.2"
    s = re.sub(r"([A-Za-zА-Яа-я])[\.\-]?(?=\d)", r"\1", s)
    return s.strip()

def _sort_key_for_dotted(num: str):
    """
    Ключ сортировки для «2.10» < «2.2», а также для 'A.1'/'П1.2'.
    """
    parts = [p for p in str(num).split(".") if p != ""]
    key = []
    for p in parts:
        if p.isdigit():
            key.append((0, int(p)))
        else:
            key.append((1, p))
    return key


# --------------------------- Публичные парсеры ---------------------------

def extract_table_numbers(text: str) -> List[str]:
    raw = [m for m in TABLE_NUM_RE.findall(text or "")]
    if not raw:
        return []
    normed = {_normalize_num(n) for n in raw if n and str(n).strip()}
    return sorted(normed, key=_sort_key_for_dotted)

def extract_figure_numbers(text: str) -> List[str]:
    raw = [m for m in FIG_NUM_RE.findall(text or "")]
    if not raw:
        return []
    normed = {_normalize_num(n) for n in raw if n and str(n).strip()}
    return sorted(normed, key=_sort_key_for_dotted)


# --------------------------- Главный распознаватель ---------------------------

def detect_intents(text: str, *, list_limit: int = 25) -> Dict:
    """
    Универсальный распознаватель интентов для нашего бота.
    Возвращает словарь вида:
    {
      "language": "ru" | "en",
      "tables":  { "want": bool, "count": bool, "list": bool, "describe": [str], "limit": int },
      "figures": { "want": bool, "count": bool, "list": bool, "describe": [str], "limit": int },
      "sources": { "want": bool, "count": bool, "list": bool, "limit": int },
      "summary": bool,
      "practical": bool,
      "gost": bool,
      "general_question": str
    }
    """
    q = (text or "").strip()
    lang = _language_guess(q)

    # Базовая структура
    intents: Dict = {
        "language": lang,
        "tables":  {"want": False, "count": False, "list": False, "describe": [], "limit": int(list_limit)},
        "figures": {"want": False, "count": False, "list": False, "describe": [], "limit": int(list_limit)},
        "sources": {"want": False, "count": False, "list": False, "limit": int(list_limit)},
        "summary": is_summary_intent(q),
        "practical": bool(PRACTICAL_RE.search(q)),
        "gost": bool(GOST_RE.search(q)),
        "general_question": q,
    }

    # Таблицы
    if TABLE_ANY_RE.search(q):
        intents["tables"]["want"] = True
        if COUNT_HINT_RE.search(q):
            intents["tables"]["count"] = True
        if WHICH_HINT_RE.search(q) or re.search(r"\b(какие таблиц|список таблиц)\b", q, re.IGNORECASE):
            intents["tables"]["list"] = True
        nums = extract_table_numbers(q)
        if nums:
            intents["tables"]["describe"] = nums

    # Источники
    if SOURCES_HINT_RE.search(q):
        intents["sources"]["want"] = True
        if COUNT_HINT_RE.search(q):
            intents["sources"]["count"] = True
        if WHICH_HINT_RE.search(q) or ("список" in q.lower()):
            intents["sources"]["list"] = True

    # Рисунки
    if FIG_ANY_RE.search(q):
        intents["figures"]["want"] = True
        if COUNT_HINT_RE.search(q):
            intents["figures"]["count"] = True
        if WHICH_HINT_RE.search(q) or re.search(r"\b(какие рисунк|список рисунк)\w*\b", q, re.IGNORECASE):
            intents["figures"]["list"] = True
        nums_f = extract_figure_numbers(q)
        if nums_f:
            intents["figures"]["describe"] = nums_f

    logging.debug("INTENTS: %s", json.dumps(intents, ensure_ascii=False))
    return intents


__all__ = [
    "detect_intents",
    "extract_table_numbers",
    "extract_figure_numbers",
    "is_summary_intent",
    # экспортируем полезные регексы на случай использования снаружи
    "TABLE_ANY_RE", "TABLE_NUM_RE",
    "FIG_ANY_RE", "FIG_NUM_RE",
    "SOURCES_HINT_RE", "PRACTICAL_RE", "GOST_RE",
]
