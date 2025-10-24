# app/intents.py
from __future__ import annotations

import re
import json
import logging
from typing import Dict, List, Tuple, Optional

try:
    from .summarizer import is_summary_intent as _is_summary_intent_external  # type: ignore
except Exception:
    _is_summary_intent_external = None  # type: ignore


# --------------------------- Регексы/константы ---------------------------

# Любое упоминание таблиц
TABLE_ANY_RE = re.compile(
    r"\bтаблиц\w*|\bтабл\.\b|\bтаблица\w*|(?:^|\s)table(s)?\b",
    re.IGNORECASE
)

# Номер таблицы (RU/EN, с поддержкой №/no., и префикс-буквы: A.1, П1.2, А.1)
TABLE_NUM_RE = re.compile(
    r"(?i)\b(?:табл(?:ица)?|table)\s*(?:№|no\.?)?\s*"
    r"([A-Za-zА-Яа-я]\.?[\s-]?\d+(?:[.,]\d+)*|\d+(?:[.,]\d+)*)\b"
)

# Рисунки: упоминания и номера
FIG_ANY_RE = re.compile(
    r"\b(рисун\w*|рис(?:\.|унок)?|figure|fig\.?|картин\w*|изображен\w*|диаграмм\w*|график\w*|схем\w*|иллюстрац\w*)\b",
    re.IGNORECASE
)
FIG_NUM_RE = re.compile(
    r"(?i)\b(?:рис(?:\.|унок)?|figure|fig\.?|картин\w*)\s*(?:№\s*|no\.?\s*)?(\d+(?:[.,]\d+)*)\b"
)

# Секции/главы — подсказки, где искать таблицу («глава 2.2», «раздел 3», «§ 1.4», «section 4.1»)
SECTION_HINT_RE = re.compile(
    r"(?i)\b(глава|раздел|пункт|подраздел|§|section|chapter|clause)\s*([A-Za-zА-Яа-я]?\s*\d+(?:[.,]\d+)*)"
)

# «В главе есть таблица» — мягкие формулировки наличия
SECTION_HAS_TABLE_RE = re.compile(
    r"(?i)\b(в|на)\s+(?:этой\s+)?(главе|разделе|пункте|подразделе|section|chapter)\b.*?\b(есть|имеется|присутствует|contains?|has)\b.*?\b(таблиц\w*|table)\b"
)

SOURCES_HINT_RE = re.compile(
    r"\b(источник(?:и|ов)?|список\s+литературы|список\s+источников|библиограф\w*|references?|bibliograph\w*)\b",
    re.IGNORECASE
)

COUNT_HINT_RE = re.compile(r"\bсколько\b|how many", re.IGNORECASE)
WHICH_HINT_RE = re.compile(r"\bкаки(е|х)\b|\bсписок\b|\bперечисл\w*\b|\bназов\w*\b", re.IGNORECASE)

# Все значения/полная таблица + лимит строк
TABLE_ALL_VALUES_RE = re.compile(
    r"(?i)\b(все|всю|целиком|полностью|полная)\b.*\b(таблиц\w*|таблица|значени\w*|данн\w*|строк\w*|колон\w*)\b"
    r"|(?:\ball\b.*\b(table|values|rows|columns)\b|\bfull\s+(table|values)\b|\bentire\s+table\b)"
)
TABLE_ROWS_LIMIT_RE = re.compile(
    r"(?i)(?:покаж[ий]|выведи|дай|отобрази|верни|первые)\s+(\d{1,4})\s+(строк\w*|rows?)\b"
    r"|(^|\s)(\d{1,4})\s+(строк\w*|rows?)\b"
    r"|top\s+(\d{1,4})\b"
)

PRACTICAL_RE = re.compile(r"(есть ли|наличие|присутствует ли|имеется ли).{0,40}практическ", re.IGNORECASE)
GOST_RE = re.compile(r"\b(гост|оформлени|шрифт|межстроч|кегл|выравнивани|поля|оформить)\w*\b", re.IGNORECASE)

FIG_VISION_HINT_RE = re.compile(
    r"(?i)\b(опиш\w+|что\s+на|что\s+изображено|explain|describe)\b.*\b(рисунк|картинк|figure|image|diagram|graph|plot)\w*\b"
)

# НОВОЕ: «точные числа, как в документе / не меняй формат»
EXACT_NUMBERS_RE = re.compile(
    r"(?i)"
    r"(точн\w+\s+(как\s+в\s+(документе|тексте|файле)))|"
    r"(ровно\s+как\s+в\s+(документе|тексте|файле))|"
    r"(как\s+есть)|"
    r"(без\s+округлен\w+)|"
    r"(не\s*(меняй|изменяй)\s*(формат|запят\w*|разделител\w*|числ\w*))|"
    r"(сохрани\w*\s*(формат|вид|написан\w*|разделител\w*|запят\w*))|"
    r"(без\s+нормализ\w+)|"
    r"(exact(ly)?\s+as\s+in\s+(doc(ument)?|file|text))|"
    r"(keep\s+(the\s+)?format(ting)?)|"
    r"(do\s+not\s+change\s+(the\s+)?(format|commas|separators|numbers))|"
    r"(no\s+round(ing)?)|"
    r"(as-is)"
)


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
    t = (text or "")
    has_lat = bool(re.search(r"[a-zA-Z]{3,}", t))
    has_cyr = bool(re.search(r"[а-яА-ЯёЁ]{3,}", t))
    return "en" if (has_lat and not has_cyr) else "ru"

def _normalize_num(s: str) -> str:
    s = (s or "")
    s = s.replace(" ", "")
    s = s.replace(",", ".")
    # убираем точку/дефис между литерой и цифрой: "A.1" -> "A1" (для поиска по label/caption_num)
    s = re.sub(r"([A-Za-zА-Яа-я])[\.\-]?(?=\d)", r"\1", s)
    return s.strip()

def _sort_key_for_dotted(num: str):
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
    """
    Ищет «Таблица/табл./Table № A.1 / 2.3 / П1.2» и возвращает нормализованные номера:
    - литеры без разделителя с цифрами (A1), дробные части через точку (2.3)
    """
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

def extract_section_hints(text: str) -> List[str]:
    """
    Достаёт подсказки разделов («глава 2.2», «раздел 3», «§ 1.4», «section 4.1»),
    нормализует: '2.2', '3', '1.4', 'A1' и т.п.
    """
    out: List[str] = []
    for m in SECTION_HINT_RE.findall(text or ""):
        # m = (label, value)
        val = _normalize_num(m[1] or "")
        if val:
            out.append(val)
    # уникализируем, сохраняем порядок
    seen = set()
    uniq: List[str] = []
    for v in out:
        if v not in seen:
            seen.add(v)
            uniq.append(v)
    return uniq


def _extract_rows_limit_and_full(text: str) -> Tuple[Optional[int], bool]:
    t = text or ""
    if TABLE_ALL_VALUES_RE.search(t):
        return (None, True)
    m = TABLE_ROWS_LIMIT_RE.search(t)
    if m:
        for g in m.groups():
            if g and g.isdigit():
                try:
                    n = int(g)
                    if n > 0:
                        return (n, False)
                except Exception:
                    pass
    return (None, False)


# --------------------------- Главный распознаватель ---------------------------

def detect_intents(text: str, *, list_limit: int = 25) -> Dict:
    """
    Возвращает структуру намерений:
    {
      "language": "ru" | "en",
      "tables":  {
         "want": bool, "count": bool, "list": bool,
         "describe": [str], "limit": int,
         "include_all_values": bool, "rows_limit": int|None,
         "section_hints": [str],
         "from_section": bool              # НОВОЕ — «в главе есть таблица» → искать по разделу и тянуть целиком
      },
      "figures": {
         "want": bool, "count": bool, "list": bool,
         "describe": [str], "limit": int, "want_vision": bool
      },
      "sources": {"want": bool, "count": bool, "list": bool, "limit": int},
      "summary": bool,
      "practical": bool,
      "gost": bool,
      "exact_numbers": bool,
      "general_question": str
    }
    """
    q = (text or "").strip()
    lang = _language_guess(q)

    intents: Dict = {
        "language": lang,
        "tables":  {
            "want": False,
            "count": False,
            "list": False,
            "describe": [],
            "limit": int(list_limit),
            "include_all_values": False,
            "rows_limit": None,
            "section_hints": [],
            "from_section": False,   # НОВОЕ
        },
        "figures": {
            "want": False,
            "count": False,
            "list": False,
            "describe": [],
            "limit": int(list_limit),
            "want_vision": False
        },
        "sources": {"want": False, "count": False, "list": False, "limit": int(list_limit)},
        "summary": is_summary_intent(q),
        "practical": bool(PRACTICAL_RE.search(q)),
        "gost": bool(GOST_RE.search(q)),
        "exact_numbers": bool(EXACT_NUMBERS_RE.search(q)),
        "general_question": q,
    }

    # --- Таблицы
    if TABLE_ANY_RE.search(q):
        intents["tables"]["want"] = True

        if COUNT_HINT_RE.search(q):
            intents["tables"]["count"] = True
        if WHICH_HINT_RE.search(q) or re.search(r"\b(какие таблиц|список таблиц)\b", q, re.IGNORECASE):
            intents["tables"]["list"] = True

        nums = extract_table_numbers(q)
        if nums:
            intents["tables"]["describe"] = nums

        rows_limit, include_all = _extract_rows_limit_and_full(q)
        intents["tables"]["include_all_values"] = bool(include_all)
        if rows_limit is not None:
            intents["tables"]["rows_limit"] = int(rows_limit)

        # Подсказки, в какой главе/разделе искать таблицу
        sects = extract_section_hints(q)
        if sects:
            intents["tables"]["section_hints"] = sects

        # НОВОЕ: «в главе есть таблица» (без номера) — тянем таблицу целиком по разделу
        if (not nums) and sects and (SECTION_HAS_TABLE_RE.search(q) or True):
            # Если есть явные секции и нет номера — считаем, что хотят таблицу(ы) из секции целиком
            intents["tables"]["from_section"] = True
            # если пользователь явно не просил только первые N строк — включаем выгрузку целиком
            if rows_limit is None:
                intents["tables"]["include_all_values"] = True

    # --- Источники
    if SOURCES_HINT_RE.search(q):
        intents["sources"]["want"] = True
        if COUNT_HINT_RE.search(q):
            intents["sources"]["count"] = True
        if WHICH_HINT_RE.search(q) or ("список" in q.lower()):
            intents["sources"]["list"] = True

    # --- Рисунки
    if FIG_ANY_RE.search(q):
        intents["figures"]["want"] = True
        if COUNT_HINT_RE.search(q):
            intents["figures"]["count"] = True
        if WHICH_HINT_RE.search(q) or re.search(r"\b(какие рисунк|список рисунк)\w*\b", q, re.IGNORECASE):
            intents["figures"]["list"] = True

        nums_f = extract_figure_numbers(q)
        if nums_f:
            intents["figures"]["describe"] = nums_f
        if FIG_VISION_HINT_RE.search(q):
            intents["figures"]["want_vision"] = True

    logging.debug("INTENTS: %s", json.dumps(intents, ensure_ascii=False))
    return intents


__all__ = [
    "detect_intents",
    "extract_table_numbers",
    "extract_figure_numbers",
    "extract_section_hints",
    "is_summary_intent",
    "TABLE_ANY_RE", "TABLE_NUM_RE",
    "FIG_ANY_RE", "FIG_NUM_RE",
    "SECTION_HINT_RE", "SECTION_HAS_TABLE_RE",
    "SOURCES_HINT_RE", "PRACTICAL_RE", "GOST_RE",
    "TABLE_ALL_VALUES_RE", "TABLE_ROWS_LIMIT_RE",
    "EXACT_NUMBERS_RE",
]
