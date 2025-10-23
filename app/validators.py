# app/validators.py
from __future__ import annotations
import re
import html
from typing import Optional, List, Tuple

"""
Общие валидаторы/подсказки и мелкие утилиты для пайплайна.

Можно импортировать точечно:
    from .validators import (
        safety_check, topical_hint, is_gost_request,
        extract_table_numbers, extract_figure_numbers,
        normalize_caption_num, sanitize_html, is_scanned_like
    )
"""

# ------------------------ Гардрейлы (безопасность) ------------------------

# Запрещённые шаблоны (джейлбрейки, вредонос, незаконное и NSFW и т.п.)
_BANNED_PATTERNS = [
    r"jail ?break|system\s*prompt|developer\s*mode|dan\b|ignore (all|previous) (rules|instructions)",
    r"\bвзлом|хаки?|кейген|кряк|социальн(ая|ые) инженерия",
    r"\bвирус|вредонос|эксплойт|ботнет|ddos\b",
    r"\bоружи|взрывчат|бомб|наркот|порно|эротик|18\+",
    r"\bпаспорт|снилс|инн\b.*(сгенер|поддел)",
    r"\bобой(д|ти)\b.*антиплаг|антиплагиат|antiplagiat",
    r"sql.?инъек|инъекци(я|и) sql",
]

def safety_check(text: str) -> Optional[str]:
    """
    Возвращает строку-пояснение, если запрос нарушает правила безопасности; иначе None.
    """
    t = (text or "").lower()
    for p in _BANNED_PATTERNS:
        if re.search(p, t, flags=re.IGNORECASE):
            return ("Запрос нарушает правила безопасности "
                    "(взлом/вредонос/обход ограничений/NSFW/личные данные).")
    return None


# ------------------------ Топик-хинт (мягкая подсказка по теме) ------------------------

_ALLOWED_HINT_WORDS = [
    "вкр", "диплом", "курсов", "методолог", "литератур", "литобзор",
    "гипотез", "цель", "задач", "введение", "заключен", "обзор",
    "оформлен", "гост", "таблиц", "рисунк", "антиплаг", "плагиат",
    "презентац", "защиту", "опрос", "анкета", "методы", "статистик",
]

def topical_hint(text: str) -> Optional[str]:
    """
    Возвращает мягкую подсказку, если вопрос совсем вне «дипломной» тематики.
    Это не блокировка, лишь дружелюбный хелп.
    """
    t = (text or "").lower()
    if not any(w in t for w in _ALLOWED_HINT_WORDS):
        return ("Подсказка: я сильнее отвечаю по теме ВКР (структура, методология, ГОСТ, "
                "литобзор, антиплагиат). Если пришлёте файл диплома — смогу отвечать по содержанию.")
    return None


# ------------------------ ГОСТ-интент ------------------------

_GOST_HINT = re.compile(r"\b(гост|оформлени|шрифт|межстроч|кегл|выравнивани|поля|оформить)\w*\b", re.IGNORECASE)

def is_gost_request(text: str) -> bool:
    """Грубая эвристика: похоже ли, что пользователь просит проверку оформления (ГОСТ)."""
    return bool(_GOST_HINT.search(text or ""))


# ------------------------ Извлечение номеров таблиц/рисунков ------------------------

# Поддерживаем: 2.1, 3, A.1, А.1, П1.2 и т.п.
_TABLE_NUM_RE = re.compile(r"(?i)\bтабл(?:ица)?\.?\s*([a-zа-я]\.?[\s-]?\d+(?:[.,]\d+)*|\d+(?:[.,]\d+)*)\b")
_FIG_NUM_RE   = re.compile(r"(?i)\b(?:рис(?:\.|унок)?|figure|fig\.?)\s*(?:№\s*)?([a-zа-я]?\s*\d+(?:[.,]\d+)*)\b")

def normalize_caption_num(num: str) -> str:
    """N, M → N.M; убираем пробелы вокруг буквенного префикса."""
    num = (num or "").strip()
    num = num.replace(",", ".")
    num = re.sub(r"\s+", "", num)
    # П1.2 → П1.2; A . 1.2 → A1.2
    return num

def _sort_key_versionlike(v: str):
    # Для сортировки вида "A.1.2" / "3.10" / "3.2"
    parts = []
    for p in v.split("."):
        if p.isdigit():
            parts.append(int(p))
        else:
            parts.append(p)
    return parts

def extract_table_numbers(text: str) -> List[str]:
    """
    Возвращает список нормализованных номеров таблиц из строки.
    Пример: ["2.1", "A.3"]
    """
    nums = [normalize_caption_num(n) for n in _TABLE_NUM_RE.findall(text or "")]
    # Удаляем префикс вида "а." / "п." → "А." / "П." (кириллица/латиница остаётся как есть)
    nums = [re.sub(r"(?i)^([a-zа-я])\.", r"\1.", n) for n in nums if n]
    # dedup + сортировка «версионированием»
    return sorted(set(nums), key=_sort_key_versionlike)

def extract_figure_numbers(text: str) -> List[str]:
    """
    Возвращает список нормализованных номеров рисунков из строки.
    Пример: ["3", "2.4", "А.1"]
    """
    raw = [normalize_caption_num(n) for n in _FIG_NUM_RE.findall(text or "")]
    return sorted(set([n for n in raw if n]), key=_sort_key_versionlike)


# ------------------------ Санитайз/утилиты текста ------------------------

def sanitize_html(text: str) -> str:
    """
    Экранирует HTML, но сохраняет <b>..</b> если текст был в **bold** (маркидаун-замена).
    Полезно перед отправкой в Telegram с parse_mode="HTML".
    """
    if not text:
        return ""
    # Маркидаун **bold** → <b>..</b> (упрощённо)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text, flags=re.DOTALL)
    text = html.escape(text)
    return text.replace("&lt;b&gt;", "<b>").replace("&lt;/b&gt;", "</b>")

def is_scanned_like(sections: List[dict], *, min_chars: int = 500) -> bool:
    """
    Грубая эвристика «похоже на скан/PDF без текста».
    Возвращает True, если суммарный объём текста слишком мал.
    """
    total = 0
    for s in sections or []:
        total += len((s.get("text") or "").strip())
        if total >= min_chars:
            return False
    return True


# ------------------------ Публичный интерфейс ------------------------

__all__ = [
    "safety_check",
    "topical_hint",
    "is_gost_request",
    "extract_table_numbers",
    "extract_figure_numbers",
    "normalize_caption_num",
    "sanitize_html",
    "is_scanned_like",
]
