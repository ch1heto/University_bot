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

# Ловим все русские формы слова «таблица»: таблица, таблицу, таблицей, таблицы, таблицах…
TABLE_ANY_RE = re.compile(
    r"\bтаблиц[а-я]*|\bтабл\.\b|\btable(s)?\b",
    re.IGNORECASE
)

# Номер таблицы (RU/EN, с поддержкой №/no./номер, и префикс-буквы: A.1, П1.2, А.1)
TABLE_NUM_RE = re.compile(
    r"(?i)\b(?:табл(?:ица)?|таблиц\w*|table)\s*(?:№|no\.?|номер\s*)?\s*"
    r"([A-Za-zА-Яа-я]\.?[\s-]?\d+(?:[.,]\d+)*|\d+(?:[.,]\d+)*)\b"
)

# Рисунки: упоминания и номера
FIG_ANY_RE = re.compile(
    r"\b(рисун\w*|рис(?:\.|унок)?|figure|fig(?:\.|ure)?|картин\w*|изображен\w*|диаграмм\w*|график\w*|схем\w*|иллюстрац\w*)\b",
    re.IGNORECASE
)
# теперь поддерживаем:
# - "Рисунок 6", "Рис. 6", "Рис.6", "Figure 6", "Fig. 6"
# - "Рисунок 2.3", "Рисунок 2,3"
# - буквенный префикс: "Рисунок А.1", "Figure A1"
FIG_NUM_RE = re.compile(
    r"(?i)\b(?:рис(?:\.|унок)?|figure|fig(?:\.|ure)?|картин\w*)"
    r"\s*(?:№\s*|no\.?\s*|номер\s*)?"
    r"([A-Za-zА-Яа-я]?\s*\d+(?:[.,]\d+)*)\.?"
)


# Секции/главы — подсказки, где искать («глава 2.2», «раздел 3», «§ 1.4», «section 4.1», «appendix A.1»)
SECTION_HINT_RE = re.compile(
    r"(?i)\b(глава|раздел|пункт|подраздел|§|section|subsection|chapter|clause|appendix)\s*([A-Za-zА-Яа-я]?\s*\d+(?:[.,]\d+)*)"
)

# «в главе есть таблица» (ранее)
SECTION_HAS_TABLE_RE = re.compile(
    r"(?i)\b(в|на)\s+(?:этой\s+)?(главе|разделе|пункте|подразделе|section|chapter|subsection)\b.*?\b(есть|имеется|присутствует|contains?|has)\b.*?\b(таблиц\w*|table)\b"
)

# «в главе есть рисунок/рисунки» (ранее)
SECTION_HAS_FIG_RE = re.compile(
    r"(?i)\b(в|на)\s+(?:этой\s+)?(главе|разделе|пункте|подразделе|section|chapter|subsection)\b.*?\b(есть|имеется|присутствует|contains?|has)\b.*?\b(рисун\w*|figure|fig\.?|диаграмм\w*|график\w*|схем\w*|image|diagram|graph|plot)\b"
)

# НОВОЕ: явные конструкции «рисунок(и)/таблица(ы) в/из/к главе/разделе X» — даже без слова «есть»
FIG_IN_SECTION_RE = re.compile(
    r"(?i)\b(рисун\w*|figure|fig\.?|диаграмм\w*|график\w*|схем\w*|image|diagram|graph|plot)\b.*?\b(в|из|к)\s+(главе|разделе|пункте|подразделе|section|chapter|subsection)\s+[A-Za-zА-Яа-я]?\s*\d+(?:[.,]\d+)*"
)
TABLE_IN_SECTION_RE = re.compile(
    r"(?i)\b(таблиц\w*|табл\.|table)\b.*?\b(в|из|к)\s+(главе|разделе|пункте|подразделе|section|chapter|subsection)\s+[A-Za-zА-Яа-я]?\s*\d+(?:[.,]\d+)*"
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

# НОВОЕ: запрос «прозрачности ссылок/якорей» — показать место в тексте/страницу/раздел
LINKS_HINT_RE = re.compile(
    r"(?i)\b(покаж\w*\s+где\s+(это|сказано|написано)|"
    r"укаж\w*\s+(страниц\w*|раздел|главу|место)|"
    r"на\s+какой\s+страниц\w+|"
    r"с\s+ссылк\w+\s+(на|в)\s+текст\w*|"
    r"дай\s+ссылк\w*\s+на\s+место|"
    r"(show|give|provide)\s+(page|section)\s+(number|path)|"
    r"(where)\s+in\s+(the\s+)?(text|document))\b"
)

# Буллеты/нумерация для многочастных запросов
BULLET_SPLIT_RE = re.compile(
    r"(?m)^\s*(?:\d{1,2}[.)]\s+|\(\d{1,2}\)\s+|[-–—•●▪︎►»]\s+)"
)
# Разделители в одну строку ";", " ; ", " | " и множественные "?".
INLINE_SPLIT_RE = re.compile(r"(?:\s*;\s*|\s*\|\s*|\?\s+)(?!\d)")


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
    """
    Нормализация номера таблицы/рисунка так, чтобы он совпадал с caption_num из parsing.py:

    - убираем неразрывные пробелы и обычные пробелы по краям
    - отрезаем ведущую литеру с точкой/дефисом/пробелом: "А.1" / "A 1.2" / "П-2.3" -> "1" / "1.2" / "2.3"
    - убираем внутренние пробелы
    - запятую превращаем в точку: "2,3" -> "2.3"
    """
    s = (s or "")
    s = s.replace("\u00A0", " ").strip()
    if not s:
        return ""
    # убираем ведущую литеру/букву (как в _norm_caption_num из parsing.py)
    s = re.sub(r"^[A-Za-zА-Яа-я]\.?[\s\-]*", "", s)
    s = s.replace(" ", "")
    s = s.replace(",", ".")
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

def _clean_piece(s: str) -> str:
    s = (s or "").strip()
    # убираем хвостовые лишние разделители
    s = s.strip(" ;|—–-•●▪︎►»")
    # одиночный финальный вопрос/точка оставим, но множественные — подрежем
    s = re.sub(r"([?.!])\1+$", r"\1", s)
    return s


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
    """
    Ищет «Рисунок/Рис./Figure/Fig. 6/2.3/A.1» и возвращает нормализованные номера:
    - литеры склеиваются с цифрой: 'A.1' -> 'A1'
    - дробные части через точку: '2,3' -> '2.3'
    """
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


# --------------------------- Многочастные запросы ---------------------------

def _split_into_subitems(text: str) -> List[str]:
    """
    Делит запрос на подпункты:
      1) По явным буллетам/нумерации в начале строк.
      2) По ; / | / '? ' как безопасным разделителям.
    Не делим по одиночной 'и', чтобы не ломать нормальные фразы.
    """
    t = (text or "").strip()
    if not t:
        return []

    # 1) Буллеты/нумерация — если минимум два подпункта
    bullets = BULLET_SPLIT_RE.findall(t)
    if len(bullets) >= 2:
        parts = BULLET_SPLIT_RE.split(t)
        # split возвращает чередование "шум/пункты": фильтруем пустое и мусор
        cand = [p for p in (p.strip() for p in parts) if p]
        # Защита от ложных срабатываний: если весь текст короткий — не делим
        if len(" ".join(cand)) >= 20 and len(cand) >= 2:
            return [_clean_piece(p) for p in cand]

    # 2) В одну строку — ';', '|', '? ' (несколько вопросов подряд)
    inline = [p for p in INLINE_SPLIT_RE.split(t) if p and not re.fullmatch(r"(;|\||\?)", p)]
    if len(inline) >= 2:
        # склеим короткие хвосты обратно (например, аббревиатуры)
        items: List[str] = []
        buf = ""
        for seg in inline:
            seg = _clean_piece(seg)
            if not seg:
                continue
            if len(seg) < 4 and items:
                items[-1] = (items[-1] + " " + seg).strip()
            else:
                items.append(seg)
        if len(items) >= 2:
            return items

    # 3) По нескольким '?'
    q_splits = [s for s in re.split(r"\?\s*", t) if s]
    if len(q_splits) >= 2:
        items = [_clean_piece(s + "?") for s in q_splits[:-1]]
        tail = _clean_piece(q_splits[-1])
        if tail:
            items.append(tail)
        return [i for i in items if i]

    return []


def _classify_item(text: str) -> Dict[str, any]:
    """
    Лёгкая классификация подпункта по модальностям (без рекурсии detect_intents).
    Возвращаем подсказки, с которыми потом сможет работать пайплайн.
    """
    item = (text or "").strip()
    tables_want = bool(TABLE_ANY_RE.search(item))
    figures_want = bool(FIG_ANY_RE.search(item))
    sources_want = bool(SOURCES_HINT_RE.search(item))

    table_nums = extract_table_numbers(item) if tables_want else []
    figure_nums = extract_figure_numbers(item) if figures_want else []
    sect_hints = extract_section_hints(item)

    rows_limit, include_all = _extract_rows_limit_and_full(item)

    # Один конкретный рисунок (опиши рисунок 6 / что на рисунке 2.3 / figure 4)
    single_fig = False
    if figures_want:
        if len(figure_nums) == 1 and not (COUNT_HINT_RE.search(item) or WHICH_HINT_RE.search(item)):
            single_fig = True

    return {
        "ask": item,
        "tables": {
            "want": tables_want,
            "describe": table_nums,
            "section_hints": sect_hints if tables_want else [],
            "from_section": (
                tables_want and not table_nums and
                (bool(sect_hints) and (bool(SECTION_HAS_TABLE_RE.search(item)) or bool(TABLE_IN_SECTION_RE.search(item))))
            ),
            "include_all_values": bool(include_all) if tables_want else False,
            "rows_limit": rows_limit if tables_want else None,
        },
        "figures": {
            "want": figures_want,
            "describe": figure_nums,
            "section_hints": sect_hints if figures_want else [],
            "from_section": (
                figures_want and not figure_nums and
                (bool(sect_hints) and (bool(SECTION_HAS_FIG_RE.search(item)) or bool(FIG_IN_SECTION_RE.search(item))))
            ),
            "want_vision": bool(FIG_VISION_HINT_RE.search(item)) if figures_want else False,
            "single_only": single_fig,
            "exact_numbers": bool(EXACT_NUMBERS_RE.search(item)) if figures_want else False,
        },
        "sources": {
            "want": sources_want,
        },
        "summary": is_summary_intent(item),
        "practical": bool(PRACTICAL_RE.search(item)),
        "gost": bool(GOST_RE.search(item)),
        "exact_numbers": bool(EXACT_NUMBERS_RE.search(item)),
        "want_links": bool(LINKS_HINT_RE.search(item)),  # new
    }


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
         "from_section": bool
      },
      "figures": {
         "want": bool, "count": bool, "list": bool,
         "describe": [str], "limit": int, "want_vision": bool,
         "section_hints": [str], "from_section": bool
      },
      "sources": {"want": bool, "count": bool, "list": bool, "limit": int},
      "summary": bool,
      "practical": bool,
      "gost": bool,
      "exact_numbers": bool,
      "want_links": bool,
      "general_question": str,
      "subitems": [{"id": int, "ask": str, ... модальные подсказки ...}],
      "multi_want": bool
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
            "from_section": False,
        },
        "figures": {
            "want": False,
            "count": False,
            "list": False,
            "describe": [],
            "limit": int(list_limit),
            "want_vision": False,
            "section_hints": [],
            "from_section": False,
            "single_only": False,   # NEW: фокус на одном рисунке
            "exact_numbers": False, # NEW: нужны исходные числа для него
        },
        "sources": {"want": False, "count": False, "list": False, "limit": int(list_limit)},
        "summary": is_summary_intent(q),
        "practical": bool(PRACTICAL_RE.search(q)),
        "gost": bool(GOST_RE.search(q)),
        "exact_numbers": bool(EXACT_NUMBERS_RE.search(q)),
        "want_links": bool(LINKS_HINT_RE.search(q)),
        "general_question": q,
        "subitems": [],
        "multi_want": False,
    }


    # ---- Многочастные запросы (план подпунктов — на будущее для пайплайна)
    subparts = _split_into_subitems(q)
    if len(subparts) >= 2:
        intents["multi_want"] = True
        intents["subitems"] = []
        for i, sp in enumerate(subparts, start=1):
            cls = _classify_item(sp)
            cls["id"] = i
            intents["subitems"].append(cls)

    # --- Таблицы (общий слой)
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

        # «В (этой) главе есть таблица …» ИЛИ «таблица в главе …» — без конкретного номера
        if (not nums) and sects and (SECTION_HAS_TABLE_RE.search(q) or TABLE_IN_SECTION_RE.search(q)):
            intents["tables"]["from_section"] = True
            if rows_limit is None and not intents["tables"]["list"] and not intents["tables"]["count"]:
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

        # Определяем, нацелен ли вопрос на один конкретный рисунок
        if nums_f and len(nums_f) == 1 and not intents["figures"]["list"] and not intents["figures"]["count"]:
            intents["figures"]["single_only"] = True

        if FIG_VISION_HINT_RE.search(q):
            intents["figures"]["want_vision"] = True

        # «точные числа» для фигур (например: "дай точные значения по рисунку 6")
        if EXACT_NUMBERS_RE.search(q):
            intents["figures"]["exact_numbers"] = True

        # Секционные подсказки
        sects_f = extract_section_hints(q)
        if sects_f:
            intents["figures"]["section_hints"] = sects_f

        # «В (этой) главе есть рисунки …» ИЛИ «рисунок(и) в главе …» — без номера
        if (not nums_f) and sects_f and (SECTION_HAS_FIG_RE.search(q) or FIG_IN_SECTION_RE.search(q)):
            intents["figures"]["from_section"] = True


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
    "EXACT_NUMBERS_RE", "LINKS_HINT_RE",
]
