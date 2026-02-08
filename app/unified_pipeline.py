# app/unified_pipeline.py
"""
Единая точка входа для ВСЕХ типов вопросов.
Решает проблему дублирования запросов к LLM API.

БЫЛО:
- answer_builder.py делает запрос
- document_calc_agent.py делает запрос  
- document_semantic_planner.py делает запрос
= 3 запроса на один вопрос!

СТАЛО:
- unified_pipeline.py делает ОДИН запрос
= 1 запрос на один вопрос!
"""

from __future__ import annotations
import re
import json
import asyncio
import logging
from typing import Optional, Dict, Any, List
from functools import lru_cache

logger = logging.getLogger(__name__)

# Импорты
try:
    from .config import Cfg
    from .retrieval import (
        retrieve,
        build_context,
        retrieve_coverage,
        build_context_coverage,
        get_table_context_for_numbers,
        get_figure_context_for_numbers,
        # НОВОЕ: для мультимодальности
        build_multimodal_context,
        get_figure_record_with_image,
    )
    from .polza_client import (
        chat_with_gpt, 
        chat_with_gpt_stream,
        # НОВОЕ: для мультимодальности
        chat_with_gpt_multimodal,
        chat_with_gpt_stream_multimodal,
    )
    from .db import get_user_active_doc
    from .figures import analyze_figure_with_vision
except Exception as e:
    logger.warning(f"Импорты не загрузились (возможно, это тест): {e}")
    # Заглушки для тестирования
    def retrieve(*args, **kwargs): return []
    def build_context(*args, **kwargs): return ""
    def chat_with_gpt(*args, **kwargs): return "Заглушка"
    async def chat_with_gpt_stream(*args, **kwargs): yield "Заглушка"


# ==================== ОПРЕДЕЛЕНИЕ ТИПА ВОПРОСА ====================

class QuestionType:
    """Типы вопросов"""
    CALC = "calc"           # Вычислительный вопрос (работа с числами)
    TABLE = "table"         # Вопрос про таблицу
    FIGURE = "figure"       # Вопрос про рисунок/график
    MIXED = "mixed"         # Вопрос про таблицу И рисунок (сравнение)
    SECTION = "section"
    CHAPTER = "chapter"     # Вопрос про конкретную главу/раздел
    SLOTS = "slots"
    SEMANTIC = "semantic"   # Обычный смысловой вопрос
    GOST = "gost"          # Вопрос про оформление/ГОСТ


def detect_question_type(question: str) -> str:
    """
    Определяет тип вопроса по ключевым словам.
    """
    q_lower = (question or "").lower()

    # Нормализуем словесные числительные для лучшего определения типа
    q_norm = _normalize_word_numbers(q_lower)

    # 1) ВЫЧИСЛИТЕЛЬНЫЕ вопросы
    calc_keywords = [
        'сколько', 'посчитай', 'вычисли', 'сумма', 'среднее',
        'процент', 'доля', 'количество', 'всего', 'итого'
    ]
    if any(kw in q_lower for kw in calc_keywords):
        return QuestionType.CALC

    # 2) Таблица / рисунок / mixed
    has_table = bool(re.search(r'\b(таблиц\w*|табл\.?|table)\b', q_lower))
    has_figure = bool(re.search(r'\b(рисун\w*|рис\.|рис\b|график\w*|диаграмм\w*|figure|fig\.|chart)\b', q_lower))

    if has_table and has_figure:
        return QuestionType.MIXED
    if has_table:
        return QuestionType.TABLE
    if has_figure:
        return QuestionType.FIGURE

    # 3) CHAPTER — конкретная глава/раздел с номером
    # Проверяем ПЕРЕД SLOTS, чтобы "опиши главу 1" не уходило в слоты
    has_chapter_with_num = bool(re.search(
        r'\b(глав[а-яё]*|раздел[а-яё]*|подраздел[а-яё]*|параграф[а-яё]*|пункт[а-яё]*)\s*(?:№\s*)?\d',
        q_norm
    )) or bool(re.search(
        r'\b\d+\s*[--–—]?\s*(?:й|я|е|го|му)?\s*(глав[а-яё]*|раздел[а-яё]*)',
        q_norm
    ))
    if has_chapter_with_num:
        # Но если это "цель главы" / "задачи раздела" — это SLOTS
        slot_triggers_strict = ('цель', 'задач', 'объект', 'предмет', 'метод')
        if not any(t in q_lower for t in slot_triggers_strict):
            return QuestionType.CHAPTER

    # 4) СЛОТЫ / паспорт / краткий план
    slot_triggers = ('цель', 'задач', 'объект', 'предмет', 'метод', 'результ', 'вывод', 'заключен')
    slot_hits = sum(1 for k in slot_triggers if k in q_lower)

    # "паспорт/краткий план" — всегда SLOTS
    if any(t in q_lower for t in ('краткий план', 'план вкр', 'паспорт')):
        return QuestionType.SLOTS

    # КЛЮЧЕВОЙ ФИКС:
    # даже ОДИНОЧНЫЙ слот (например "опиши цель ВКР") должен идти в SLOTS,
    # иначе SECTION/SEMANTIC будут собирать контекст хуже.
    if slot_hits >= 1:
        return QuestionType.SLOTS

    # 5) РАЗДЕЛЫ/ГЛАВЫ/СТРУКТУРА (про структуру/главы — без номера)
    section_keywords = [
        'глав', 'раздел', 'параграф', 'пункт', 'введение', 'заключение',
        'структур', 'содержани', 'оглавлен', 'подглав'
    ]
    if any(kw in q_lower for kw in section_keywords):
        return QuestionType.SECTION

    # 6) ГОСТ/ОФОРМЛЕНИЕ
    gost_keywords = ['гост', 'оформлени', 'шрифт', 'межстроч', 'поля', 'кегл']
    if any(kw in q_lower for kw in gost_keywords):
        return QuestionType.GOST

    return QuestionType.SEMANTIC


# ─── Russian word-number normalization (ordinals + cardinals, 1-20) ───────────
_ORD_SUFFIXES = r'(?:ый|ая|ое|ого|ой|ому|ую|ым|ом|ые|ых|ыми)'
_ORD3_SUFFIXES = r'(?:ий|ья|ье|ьего|ьей|ьему|ью|ьим|ьем|ьи|ьих|ьими)'

_WORD_NUMBER_RULES: list[tuple[re.Pattern, str]] = [
    # ── teens and 20 (longer stems first to avoid substring conflicts) ──
    (re.compile(r'\bдвадцат' + _ORD_SUFFIXES + r'\b'), '20'),
    (re.compile(r'\bдвадцат[ьи]\b'), '20'),
    (re.compile(r'\bдевятнадцат' + _ORD_SUFFIXES + r'\b'), '19'),
    (re.compile(r'\bдевятнадцат[ьи]\b'), '19'),
    (re.compile(r'\bвосемнадцат' + _ORD_SUFFIXES + r'\b'), '18'),
    (re.compile(r'\bвосемнадцат[ьи]\b'), '18'),
    (re.compile(r'\bсемнадцат' + _ORD_SUFFIXES + r'\b'), '17'),
    (re.compile(r'\bсемнадцат[ьи]\b'), '17'),
    (re.compile(r'\bшестнадцат' + _ORD_SUFFIXES + r'\b'), '16'),
    (re.compile(r'\bшестнадцат[ьи]\b'), '16'),
    (re.compile(r'\bпятнадцат' + _ORD_SUFFIXES + r'\b'), '15'),
    (re.compile(r'\bпятнадцат[ьи]\b'), '15'),
    (re.compile(r'\bчетырнадцат' + _ORD_SUFFIXES + r'\b'), '14'),
    (re.compile(r'\bчетырнадцат[ьи]\b'), '14'),
    (re.compile(r'\bтринадцат' + _ORD_SUFFIXES + r'\b'), '13'),
    (re.compile(r'\bтринадцат[ьи]\b'), '13'),
    (re.compile(r'\bдвенадцат' + _ORD_SUFFIXES + r'\b'), '12'),
    (re.compile(r'\bдвенадцат[ьи]\b'), '12'),
    (re.compile(r'\bодиннадцат' + _ORD_SUFFIXES + r'\b'), '11'),
    (re.compile(r'\bодиннадцат[ьи]\b'), '11'),

    # ── 1-10 ordinals ──
    (re.compile(r'\bдесят' + _ORD_SUFFIXES + r'\b'), '10'),
    (re.compile(r'\bдевят' + _ORD_SUFFIXES + r'\b'), '9'),
    (re.compile(r'\bвосьм' + _ORD_SUFFIXES + r'\b'), '8'),
    (re.compile(r'\bседьм' + _ORD_SUFFIXES + r'\b'), '7'),
    (re.compile(r'\bшест' + _ORD_SUFFIXES + r'\b'), '6'),
    (re.compile(r'\bпят' + _ORD_SUFFIXES + r'\b'), '5'),
    (re.compile(r'\bчетв[её]рт' + _ORD_SUFFIXES + r'\b'), '4'),
    (re.compile(r'\bтрет' + _ORD3_SUFFIXES + r'\b'), '3'),
    (re.compile(r'\bвтор' + _ORD_SUFFIXES + r'\b'), '2'),
    (re.compile(r'\bперв' + _ORD_SUFFIXES + r'\b'), '1'),

    # ── 1-10 cardinals ──
    (re.compile(r'\bдесять\b'), '10'),
    (re.compile(r'\bдесяти\b'), '10'),
    (re.compile(r'\bдесятью\b'), '10'),
    (re.compile(r'\bдевять\b'), '9'),
    (re.compile(r'\bдевяти\b'), '9'),
    (re.compile(r'\bдевятью\b'), '9'),
    (re.compile(r'\bвосемь\b'), '8'),
    (re.compile(r'\bвосьми\b'), '8'),
    (re.compile(r'\bвосемью\b'), '8'),
    (re.compile(r'\bвосьмью\b'), '8'),
    (re.compile(r'\bсемь\b'), '7'),
    (re.compile(r'\bсеми\b'), '7'),
    (re.compile(r'\bсемью\b'), '7'),
    (re.compile(r'\bшесть\b'), '6'),
    (re.compile(r'\bшести\b'), '6'),
    (re.compile(r'\bшестью\b'), '6'),
    (re.compile(r'\bпять\b'), '5'),
    (re.compile(r'\bпяти\b'), '5'),
    (re.compile(r'\bпятью\b'), '5'),
    (re.compile(r'\bчетыр[её]х\b'), '4'),
    (re.compile(r'\bчетыр[её]м\b'), '4'),
    (re.compile(r'\bчетырьмя\b'), '4'),
    (re.compile(r'\bчетыре\b'), '4'),
    (re.compile(r'\bтр[её]х\b'), '3'),
    (re.compile(r'\bтр[её]м\b'), '3'),
    (re.compile(r'\bтремя\b'), '3'),
    (re.compile(r'\bтри\b'), '3'),
    (re.compile(r'\bдвух\b'), '2'),
    (re.compile(r'\bдвум\b'), '2'),
    (re.compile(r'\bдвумя\b'), '2'),
    (re.compile(r'\bдве\b'), '2'),
    (re.compile(r'\bдва\b'), '2'),
    (re.compile(r'\bодного\b'), '1'),
    (re.compile(r'\bодной\b'), '1'),
    (re.compile(r'\bодному\b'), '1'),
    (re.compile(r'\bодним\b'), '1'),
    (re.compile(r'\bодном\b'), '1'),
    (re.compile(r'\bодну\b'), '1'),
    (re.compile(r'\bодна\b'), '1'),
    (re.compile(r'\bодно\b'), '1'),
    (re.compile(r'\bодин\b'), '1'),
]


def _normalize_word_numbers(text: str) -> str:
    """Заменяет русские словесные числительные (1-20) на цифры.

    Поддерживает порядковые (первый, второй...) и количественные (один, два...)
    во всех родах и падежах.
    Примеры: 'первую главу' → '1 главу', 'рисунок два' → 'рисунок 2'
    """
    result = text
    for pattern, digit in _WORD_NUMBER_RULES:
        result = pattern.sub(digit, result)
    return result


def extract_numbers_from_question(question: str, entity_type: str = "figure") -> List[str]:
    """
    Извлекает номера (рисунков/таблиц/разделов) из вопроса.
    Поддерживает номера с точкой: 1.1, 2.3, 3.1
    Поддерживает словесные числительные: первый, второй, два, три и т.д.

    entity_type: "figure", "table", "section"
    """
    # Нормализуем словесные числительные → цифры
    q_lower = _normalize_word_numbers(question.lower())
    numbers = []

    _NUM = r'(\d+(?:\.\d+)?)'

    if entity_type == "figure":
        # Паттерны для рисунков во ВСЕХ падежах и с номерами типа 1.1
        patterns = [
            r'рисун[а-яё]*\s*(?:№\s*)?' + _NUM,          # рисунок 1.1, рисунке 2.3
            r'рис\.?\s*(?:№\s*)?' + _NUM,                 # рис. 1.1, рис 2
            r'график[а-яё]*\s*(?:№\s*)?' + _NUM,          # график 1.1
            r'диаграмм[а-яё]*\s*(?:№\s*)?' + _NUM,        # диаграмма 2.1
            r'figure\s*(?:№\s*)?' + _NUM,                 # figure 1.1
            r'fig\.?\s*(?:№\s*)?' + _NUM,                 # fig. 1.1
            # Обратный порядок: "2 рисунок", "второй рисунок" (после нормализации)
            _NUM + r'\s+рисун[а-яё]*',
            _NUM + r'\s+рис\b',
            _NUM + r'\s+график[а-яё]*',
            _NUM + r'\s+диаграмм[а-яё]*',
        ]
    elif entity_type == "table":
        # Паттерны для таблиц с номерами типа 2.1
        patterns = [
            r'таблиц[а-яё]*\s*(?:№\s*)?' + _NUM,         # таблица 2.1, таблице 3
            r'табл\.?\s*(?:№\s*)?' + _NUM,                # табл. 2.1
            r'table\s*(?:№\s*)?' + _NUM,                  # table 2.1
            # Обратный порядок: "2 таблица", "вторая таблица" (после нормализации)
            _NUM + r'\s+таблиц[а-яё]*',
            _NUM + r'\s+табл\b',
        ]
    else:  # section
        patterns = [
            r'глав[а-яё]*\s*(?:№\s*)?' + _NUM,
            r'раздел[а-яё]*\s*(?:№\s*)?' + _NUM,
            r'пункт[а-яё]*\s*(?:№\s*)?' + _NUM,
            # Обратный порядок: "1 глава", "первая глава" (после нормализации)
            _NUM + r'\s+глав[а-яё]*',
            _NUM + r'\s+раздел[а-яё]*',
            _NUM + r'\s+пункт[а-яё]*',
        ]

    for pattern in patterns:
        matches = re.findall(pattern, q_lower)
        numbers.extend(matches)

    # Fallback: ищем номер после предлога
    if not numbers:
        fallback_patterns = {
            "figure": r'(?:на|про|в|о|об)\s+рис\w*\s+' + _NUM,
            "table": r'(?:в|из|на|про)\s+табл\w*\s+' + _NUM,
        }
        if entity_type in fallback_patterns:
            fallback_match = re.search(fallback_patterns[entity_type], q_lower)
            if fallback_match:
                numbers.append(fallback_match.group(1))

    print(f"[DEBUG] extract_numbers('{question[:50]}...', '{entity_type}') -> {numbers}")

    return list(set(numbers))

# ==================== SUB-QUESTION QUERY ENGINE (МНОГОСЛОТНЫЕ ВОПРОСЫ) ====================

_SLOT_ORDER = [
    "goal", "tasks", "object", "subject", "methods", "results", "conclusions",
    "chapter", "table", "figure"
]

_SLOTS = {
    "goal": {
        "title": "Цель",
        "keywords": ["цель", "целью"],
        "subq": "Какая цель работы? Дай точную цитату и краткий вывод.",
        "force_type": QuestionType.SECTION,  # чтобы сработали meta/введение
    },
    "tasks": {
        "title": "Задачи",
        "keywords": ["задач"],
        "subq": "Какие задачи работы/исследования указаны? Перечисли списком. Если нет — скажи: «Не указано в документе».",
        "force_type": QuestionType.SECTION,
    },
    "object": {
        "title": "Объект",
        "keywords": ["объект"],
        "subq": "Какой объект исследования/работы указан? Дай формулировку из текста.",
        "force_type": QuestionType.SECTION,
    },
    "subject": {
        "title": "Предмет",
        "keywords": ["предмет"],
        "subq": "Какой предмет исследования/работы указан? Дай формулировку из текста.",
        "force_type": QuestionType.SECTION,
    },
    "methods": {
        "title": "Методы",
        "keywords": ["метод", "методолог"],
        "subq": "Какие методы указаны в работе? Перечисли только то, что явно написано в документе.",
        "force_type": QuestionType.SECTION,
    },
    "results": {
        "title": "Результаты",
        "keywords": ["результат", "итог"],
        "subq": "Какие результаты/итоги работы указаны (аннотация/заключение)? Приведи цитаты, если есть.",
        "force_type": QuestionType.SECTION,
    },
    "conclusions": {
        "title": "Выводы",
        "keywords": ["вывод", "заключен"],
        "subq": "Какие выводы представлены в работе (в заключении и/или выводы по главам)?",
        "force_type": QuestionType.SECTION,
    },
    "chapter": {
        "title": "Глава/раздел",
        "keywords": ["глав", "раздел", "подглав", "параграф", "пункт"],
        "subq": None,
        "force_type": QuestionType.CHAPTER,
    },
    "table": {
        "title": "Таблица",
        "keywords": ["таблиц", "табл"],
        "subq": None,
        "force_type": QuestionType.TABLE,
    },
    "figure": {
        "title": "Рисунок/график",
        "keywords": ["рисунок", "рис.", "рис ", "диаграмм", "график", "chart", "figure"],
        "subq": None,
        "force_type": QuestionType.FIGURE,
    },
}

def _extract_slots_in_order(question: str) -> List[str]:
    q = (question or "").lower()
    hits = []
    for slot in _SLOT_ORDER:
        cfg = _SLOTS.get(slot)
        if not cfg:
            continue
        for kw in cfg["keywords"]:
            idx = q.find(kw)
            if idx != -1:
                hits.append((idx, slot))
                break
    hits.sort(key=lambda x: x[0])

    seen = set()
    ordered = []
    for _, slot in hits:
        if slot not in seen:
            ordered.append(slot)
            seen.add(slot)
    return ordered

_INTENTS = {
    "slots":   ["цель", "задач", "объект", "предмет", "метод", "результ", "вывод", "заключен", "паспорт", "краткий план", "план вкр"],
    "chapter": ["глав", "раздел", "подглав", "параграф", "пункт", "введени", "оглавлен", "содержани", "структур"],
    "table":   ["таблиц", "табл", "table"],
    "figure":  ["рисун", "рис.", "рис ", "диаграмм", "график", "figure", "fig.", "chart"],
    "gost":    ["гост", "оформлени", "шрифт", "межстроч", "поля", "кегл"],
    "calc":    ["сколько", "посчитай", "вычисли", "сумма", "среднее", "процент", "доля", "итого"],
}

def _extract_intents_in_order(question: str) -> List[str]:
    q = (question or "").lower()
    hits = []
    for intent, kws in _INTENTS.items():
        best = None
        for kw in kws:
            idx = q.find(kw)
            if idx != -1 and (best is None or idx < best):
                best = idx
        if best is not None:
            hits.append((best, intent))
    hits.sort(key=lambda x: x[0])

    seen = set()
    ordered = []
    for _, intent in hits:
        if intent not in seen:
            ordered.append(intent)
            seen.add(intent)
    return ordered

def _looks_like_multi_slot(question: str) -> bool:
    intents = _extract_intents_in_order(question)

    # 2+ разных намерений (слоты + глава/рисунок/таблица/...) — это multi
    if len(intents) >= 2:
        return True

    # “чистые слоты” тоже бывают multi: цель+объект+... даже если это одно намерение slots
    if intents == ["slots"]:
        slots = _extract_slots_in_order(question)
        return len(slots) >= 2 or any(t in (question or "").lower() for t in ("паспорт", "краткий план", "план вкр"))

    return False


def _build_subquestion_for_slot(question: str, slot: str) -> str:
    # Нормализуем словесные числительные → цифры
    q_lower = _normalize_word_numbers((question or "").lower())

    if slot == "chapter":
        # "глава 2"
        m = re.search(r'глав[а-яё]*\s*(?:№\s*)?(\d+(?:\.\d+)*)', q_lower)
        if m:
            return f"Опиши главу {m.group(1)}: тема, ключевые тезисы, что рассматривается, выводы (если есть)."

        # "во 2-й главе", "2-я глава", "2 глава", "первая глава" (→ "1 глава" после нормализации)
        m = re.search(r'\b(\d+(?:\.\d+)*)\s*[--–—]?\s*(?:й|я|е|го|му)?\s*глав[а-яё]*', q_lower)
        if m:
            return f"Опиши главу {m.group(1)}: тема, ключевые тезисы, что рассматривается, выводы (если есть)."

        m = re.search(r'(?:раздел|подраздел|пункт|параграф)\s*(?:№\s*)?(\d+(?:\.\d+)*)', q_lower)
        if m:
            return f"Расскажи про раздел {m.group(1)}: тема, ключевые тезисы, что рассматривается."

        return "Опиши структуру документа по главам и подглавам (по заголовкам/оглавлению)."

    if slot == "table":
        nums = extract_numbers_from_question(question, "table")
        if nums:
            return f"Расскажи про таблицу {nums[0]}: что в ней, какие значения, основные выводы."
        return "Какие таблицы есть в работе? Перечисли таблицы и подписи."

    if slot == "figure":
        nums = extract_numbers_from_question(question, "figure")
        if nums:
            return f"Расскажи про рисунок {nums[0]}: подпись, значения/данные и выводы."
        return "Какие рисунки/диаграммы есть в работе? Перечисли рисунки и подписи."

    cfg = _SLOTS.get(slot) or {}
    return cfg.get("subq") or question


def _build_multi_system_prompt() -> str:
    # намеренно коротко: базовые правила + формат
    return (
        "Ты помощник по анализу ВКР.\n"
        "Отвечай СТРОГО на основе предоставленного контекста.\n"
        "Запрещено выдумывать. Если по пункту нет данных — пиши: «Не указано в документе».\n"
        "Ответ оформи как нумерованный список в порядке, заданном пользователем (по смыслу).\n"
    )

def _merge_slot_contexts(slot_blocks: List[Dict[str, str]]) -> str:
    # лёгкое ограничение, чтобы не раздувать контекст
    parts = []
    total_limit = 24000
    per_slot_limit = 7000
    total = 0

    for b in slot_blocks:
        ctx = b.get("context") or ""
        if len(ctx) > per_slot_limit:
            ctx = ctx[:per_slot_limit] + "\n\n[... контекст сокращён ...]"
        block = (
            f"=== ПУНКТ: {b.get('title','')} ===\n"
            f"Под-вопрос: {b.get('subq','')}\n"
            f"Контекст:\n{ctx}\n"
        )
        if total + len(block) > total_limit:
            break
        parts.append(block)
        total += len(block)

    return "\n\n---\n\n".join(parts)


async def _search_figure_in_text(owner_id: int, doc_id: int, figure_num: str) -> str:
    """
    Ищет текстовое описание рисунка в документе.
    Используется когда:
    1) Vision API не смог проанализировать изображение
    2) Изображение не найдено/это текстовая схема
    3) Vision API дал "не тот" рисунок
    """
    try:
        from .db import get_conn
        con = get_conn()
        cur = con.cursor()

        result_text = ""

        # Стратегия 1: пробуем связать номер рисунка с section_path / упоминанием
        section_num = figure_num

        cur.execute("""
            SELECT text
            FROM chunks
            WHERE owner_id = ? AND doc_id = ?
              AND (section_path LIKE ? OR text LIKE ?)
            ORDER BY id ASC
            LIMIT 20
        """, (owner_id, doc_id, f'%{section_num}%', f'%{section_num}%'))

        for row in cur.fetchall():
            chunk_text = row["text"]
            if not chunk_text:
                continue
            if len(chunk_text) <= 50:
                continue

            # НЕ выкидываем всё, где есть "Рис." — иногда там сразу идёт пояснение.
            # Пропускаем только совсем короткие строки-подписи.
            low = chunk_text.strip().lower()
            is_caption_like = low.startswith(("рис.", "рисунок")) and len(chunk_text.strip()) < 120
            if not is_caption_like:
                result_text += chunk_text + "\n\n"

        # Стратегия 2: найдём подпись "Рис X" и заберём следующие чанки
        if len(result_text) < 500:
            search_patterns = [
                f'%Рис%{figure_num}%',
                f'%рис%{figure_num}%',
                f'%Рисунок%{figure_num}%',
                f'%рисунок%{figure_num}%',
            ]

            caption_chunk_id = None
            for pattern in search_patterns:
                cur.execute("""
                    SELECT id, text
                    FROM chunks
                    WHERE owner_id = ? AND doc_id = ?
                      AND text LIKE ?
                    ORDER BY id ASC
                    LIMIT 1
                """, (owner_id, doc_id, pattern))
                row = cur.fetchone()
                if row:
                    caption_chunk_id = row["id"]
                    break

            if caption_chunk_id is not None:
                cur.execute("""
                    SELECT text
                    FROM chunks
                    WHERE owner_id = ? AND doc_id = ?
                      AND id > ?
                    ORDER BY id ASC
                    LIMIT 15
                """, (owner_id, doc_id, caption_chunk_id))

                for row in cur.fetchall():
                    chunk_text = row["text"]
                    if chunk_text and len(chunk_text) > 30 and chunk_text not in result_text:
                        result_text += chunk_text + "\n\n"

        # Стратегия 3: тематический fallback (оставляем как есть, но только если всё ещё мало)
        if len(result_text) < 500:
            cur.execute("""
                SELECT text
                FROM chunks
                WHERE owner_id = ? AND doc_id = ?
                  AND (
                    text LIKE '%метод%анализ%' OR
                    text LIKE '%ABC%анализ%' OR
                    text LIKE '%вертикальн%анализ%' OR
                    text LIKE '%горизонтальн%анализ%' OR
                    text LIKE '%структурн%анализ%'
                  )
                ORDER BY id ASC
                LIMIT 10
            """, (owner_id, doc_id))

            for row in cur.fetchall():
                chunk_text = row["text"]
                if chunk_text and len(chunk_text) > 50 and chunk_text not in result_text:
                    result_text += chunk_text + "\n\n"

        con.close()

        if len(result_text) > 8000:
            result_text = result_text[:8000] + "\n\n[... текст сокращён ...]"

        if result_text:
            logger.info(f"Найдено текстовое описание рисунка {figure_num}: {len(result_text)} символов")

        return result_text

    except Exception as e:
        logger.warning(f"Ошибка поиска текстового описания рисунка: {e}")
        return ""

# ==================== ПОЛУЧЕНИЕ КОНТЕКСТА ====================

async def get_context_for_question(
    doc_id: int,
    owner_id: int,
    question: str,
    question_type: str,
) -> str:
    """
    Получает релевантный контекст в зависимости от типа вопроса.
    """

    q_lower = (question or "").lower()

    # 1) TABLE или CALC + упоминание таблицы
    is_table_question = (
        question_type == QuestionType.TABLE or
        (question_type == QuestionType.CALC and any(w in q_lower for w in ['таблиц', 'табл']))
    )

    if is_table_question:
        table_nums = extract_numbers_from_question(question, "table")
        logger.info(f"[TABLE] is_table_question=True table_nums={table_nums} doc_id={doc_id} owner_id={owner_id}")

        if table_nums:
            # get_table_context_for_numbers возвращает список snippet-dicts
            rag_snippets = await asyncio.to_thread(
                get_table_context_for_numbers,
                owner_id, doc_id, table_nums
            )
            rag_text = build_context(rag_snippets) if rag_snippets else ""
            logger.info(f"[TABLE] RAG snippets: {len(rag_snippets) if rag_snippets else 0}, text_len={len(rag_text)}")

            table_data_text = ""
            try:
                from .db import get_conn
                con = get_conn()
                cur = con.cursor()

                for table_num in table_nums:
                    search_patterns = [
                        f'%Таблица {table_num}%',
                        f'%таблица {table_num}%',
                        f'%Таблица{table_num}%',
                        f'%Табл. {table_num}%',
                        f'%табл. {table_num}%',
                        f'%Table {table_num}%',
                    ]

                    for pattern in search_patterns:
                        # Ищем чанки типа table/table_row ИЛИ содержащие упоминание таблицы
                        cur.execute("""
                            SELECT text, element_type, section_path
                            FROM chunks
                            WHERE owner_id = ? AND doc_id = ?
                              AND (
                                element_type IN ('table', 'table_row') OR
                                section_path LIKE ? OR
                                text LIKE ?
                              )
                            ORDER BY
                                CASE WHEN element_type IN ('table', 'table_row') THEN 0 ELSE 1 END,
                                id ASC
                            LIMIT 30
                        """, (owner_id, doc_id, pattern, pattern))

                        rows = cur.fetchall()
                        logger.info(f"[TABLE] pattern={pattern} found={len(rows)}")

                        if rows:
                            for row in rows:
                                chunk_text = row['text']
                                if chunk_text and len(chunk_text) > 30:
                                    table_data_text += chunk_text + "\n\n"

                            if table_data_text:
                                break

                    if table_data_text:
                        break

                    # Fallback: поиск table_row чанков с attrs содержащими номер
                    if not table_data_text:
                        like_attrs1 = f'%"caption_num": "{table_num}"%'
                        like_attrs2 = f'%"label": "{table_num}"%'
                        cur.execute("""
                            SELECT text, section_path
                            FROM chunks
                            WHERE owner_id = ? AND doc_id = ?
                              AND element_type IN ('table', 'table_row')
                              AND (attrs LIKE ? OR attrs LIKE ?)
                            ORDER BY id ASC
                            LIMIT 50
                        """, (owner_id, doc_id, like_attrs1, like_attrs2))

                        rows = cur.fetchall()
                        logger.info(f"[TABLE] attrs search found={len(rows)}")
                        for row in rows:
                            chunk_text = row['text']
                            if chunk_text and len(chunk_text) > 20:
                                table_data_text += chunk_text + "\n\n"

                con.close()
            except Exception as e:
                logger.warning(f"[TABLE] Ошибка поиска таблицы: {e}")

            # Собираем итоговый контекст
            combined = ""
            if table_data_text:
                combined += f"ДАННЫЕ ТАБЛИЦЫ:\n{table_data_text}\n\n"
            if rag_text:
                combined += f"КОНТЕКСТ ИЗ ДОКУМЕНТА:\n{rag_text}\n\n"
            if combined:
                combined += "ВАЖНО: Используй данные из таблицы для точного ответа!"
                return combined

            # Если ничего не нашли по номеру — fallback в RAG
            logger.warning(f"[TABLE] Ничего не найдено для table_nums={table_nums}")
            return rag_text if rag_text else ""

    # 2) CALC без явной таблицы, но номер таблицы указан
    if question_type == QuestionType.CALC:
        table_nums = extract_numbers_from_question(question, "table")
        if table_nums:
            snippets = await asyncio.to_thread(
                get_table_context_for_numbers,
                owner_id, doc_id, table_nums
            )
            return build_context(snippets) if snippets else ""

    # 3) MIXED
    if question_type == QuestionType.MIXED:
        table_nums = extract_numbers_from_question(question, "table")
        fig_nums = extract_numbers_from_question(question, "figure")

        combined_parts = []

        if table_nums:
            table_data_text = ""
            try:
                from .db import get_conn
                con = get_conn()
                cur = con.cursor()

                for table_num in table_nums:
                    search_patterns = [
                        f'%Таблица {table_num}%',
                        f'%таблица {table_num}%',
                        f'%Таблица{table_num}%',
                        f'%Табл. {table_num}%',
                        f'%табл. {table_num}%',
                    ]

                    for pattern in search_patterns:
                        cur.execute("""
                            SELECT text, element_type, section_path
                            FROM chunks
                            WHERE owner_id = ? AND doc_id = ?
                              AND (
                                element_type IN ('table', 'table_row') OR
                                section_path LIKE ? OR
                                text LIKE ?
                              )
                            ORDER BY
                                CASE WHEN element_type IN ('table', 'table_row') THEN 0 ELSE 1 END,
                                id ASC
                            LIMIT 30
                        """, (owner_id, doc_id, pattern, pattern))

                        rows = cur.fetchall()
                        if rows:
                            for row in rows:
                                chunk_text = row['text']
                                if chunk_text and len(chunk_text) > 30:
                                    table_data_text += chunk_text + "\n\n"
                            if table_data_text:
                                break

                    if table_data_text:
                        break

                con.close()

                if table_data_text:
                    combined_parts.append(f"ДАННЫЕ ТАБЛИЦЫ {table_nums[0]}:\n{table_data_text}")

            except Exception as e:
                logger.warning(f"Ошибка загрузки таблицы для MIXED: {e}")

        if fig_nums:
            try:
                vision_result = await asyncio.to_thread(
                    analyze_figure_with_vision,
                    owner_id, doc_id, fig_nums[0], question
                )
                if vision_result:
                    combined_parts.append(f"ВИЗУАЛЬНЫЙ АНАЛИЗ РИСУНКА {fig_nums[0]} (Vision API):\n{vision_result}")
            except Exception as e:
                logger.warning(f"Vision API failed для MIXED: {e}")

        text_context = await asyncio.to_thread(
            retrieve, owner_id, doc_id, question, top_k=5
        )
        text_context_str = build_context(text_context) if text_context else ""

        if combined_parts:
            combined_context = "\n\n---\n\n".join(combined_parts)
            if text_context_str:
                combined_context += f"\n\nДОПОЛНИТЕЛЬНЫЙ КОНТЕКСТ:\n{text_context_str}"
            return combined_context

        return text_context_str

    # 4) FIGURE
    if question_type == QuestionType.FIGURE:
        fig_nums = extract_numbers_from_question(question, "figure")
        if fig_nums:
            text_context = await asyncio.to_thread(
                get_figure_context_for_numbers,
                owner_id, doc_id, fig_nums
            )

            chart_data_text = ""
            try:
                from .db import get_figures_for_doc
                figures = await asyncio.to_thread(get_figures_for_doc, doc_id)

                logger.info(f"Найдено {len(figures)} рисунков в БД для doc_id={doc_id}")

                for fig in figures:
                    fig_label = str(fig.get('label') or fig.get('figure_label') or fig.get('num') or '')
                    fig_num_from_db = fig_label.strip()

                    for requested_num in fig_nums:
                        if (
                            requested_num == fig_num_from_db or
                            fig_num_from_db.endswith(requested_num) or
                            requested_num in fig_num_from_db
                        ):
                            attrs = fig.get('attrs') or {}
                            if isinstance(attrs, str):
                                try:
                                    attrs = json.loads(attrs)
                                except Exception:
                                    attrs = {}

                            chart_data = attrs.get('chart_data')
                            if chart_data:
                                try:
                                    from .chart_extractor import get_chart_data_as_text
                                    chart_data_text = get_chart_data_as_text(chart_data)
                                except ImportError:
                                    chart_data_text = _format_chart_data_simple(chart_data)
                            break

                    if chart_data_text:
                        break

            except Exception as e:
                logger.warning(f"Не удалось получить chart_data из БД: {e}")

            if chart_data_text:
                return f"""
ОПИСАНИЕ ИЗ ДОКУМЕНТА:
{text_context}

ТОЧНЫЕ ДАННЫЕ ДИАГРАММЫ (извлечены из DOCX):
{chart_data_text}

ВАЖНО: Используй ТОЛЬКО эти точные данные при ответе!
"""

            try:
                vision_result = await asyncio.to_thread(
                    analyze_figure_with_vision,
                    owner_id, doc_id, fig_nums[0], question
                )

                vision_is_valid = True
                if vision_result:
                    vision_lower = vision_result.lower()
                    critical_failures = [
                        'изображение полностью чёрное',
                        'изображение пустое',
                        'не удалось загрузить',
                        'изображение не загружено',
                        'контент отсутствует',
                        'изображение не отображается',
                    ]
                    if any(sign in vision_lower[:200] for sign in critical_failures):
                        vision_is_valid = False

                if vision_result and vision_is_valid:
                    figure_text_context = await _search_figure_in_text(owner_id, doc_id, fig_nums[0])

                    combined_context = f"""
ОПИСАНИЕ ИЗ ДОКУМЕНТА:
{text_context}

ВИЗУАЛЬНЫЙ АНАЛИЗ РИСУНКА (Vision API):
{vision_result}
"""
                    if figure_text_context and len(figure_text_context) > 200:
                        combined_context += f"""

ТЕКСТОВОЕ ОПИСАНИЕ РИСУНКА {fig_nums[0]} (из документа):
{figure_text_context}

КРИТИЧЕСКИ ВАЖНО:
1) Визуальный анализ может не соответствовать подписи.
2) Если есть конфликт, используй ТЕКСТОВОЕ ОПИСАНИЕ как основной источник.
"""
                    return combined_context

                figure_text_context = await _search_figure_in_text(owner_id, doc_id, fig_nums[0])
                if figure_text_context:
                    return f"""
ОПИСАНИЕ ИЗ ДОКУМЕНТА:
{text_context}

ТЕКСТОВОЕ ОПИСАНИЕ РИСУНКА {fig_nums[0]} (из документа):
{figure_text_context}

ВАЖНО: Визуальный анализ рисунка недоступен. Отвечай на основе текстового описания из документа.
"""

            except Exception as e:
                logger.warning(f"Vision API failed: {e}")

            return text_context

    # 5) SLOTS
    if question_type == QuestionType.SLOTS:
        slots_context = ""
        q_lower = (question or "").lower()

        slot_map = {
            "goal": ["цель", "целью"],
            "tasks": ["задач"],
            "object": ["объект"],
            "subject": ["предмет"],
            "methods": ["метод", "методолог"],
            "results": ["результат", "итог"],
            "conclusions": ["вывод", "заключен"],
        }

        requested_slots = []
        for slot, keys in slot_map.items():
            if any(k in q_lower for k in keys):
                requested_slots.append(slot)

        if not requested_slots and any(t in q_lower for t in ["краткий план", "план вкр", "паспорт"]):
            requested_slots = list(slot_map.keys())

        try:
            from .db import get_conn, get_document_ai_meta

            meta = None
            try:
                meta = get_document_ai_meta(doc_id)
            except Exception as e:
                logger.warning(f"get_document_ai_meta failed: {e}")
                meta = None

            if meta:
                parts = []
                if ("goal" in requested_slots) and meta.get("goal"):
                    parts.append(f"ЦЕЛЬ (meta): {meta['goal']}")
                if ("object" in requested_slots) and meta.get("object"):
                    parts.append(f"ОБЪЕКТ (meta): {meta['object']}")
                if ("subject" in requested_slots) and meta.get("subject"):
                    parts.append(f"ПРЕДМЕТ (meta): {meta['subject']}")
                if ("tasks" in requested_slots) and meta.get("tasks"):
                    if isinstance(meta["tasks"], list) and meta["tasks"]:
                        tasks_formatted = "\n".join([f"  {i+1}. {t}" for i, t in enumerate(meta["tasks"])])
                        parts.append(f"ЗАДАЧИ (meta):\n{tasks_formatted}")
                if ("methods" in requested_slots) and meta.get("methods"):
                    parts.append(f"МЕТОДЫ (meta): {meta['methods']}")
                if meta.get("relevance"):
                    parts.append(f"АКТУАЛЬНОСТЬ (meta): {meta['relevance']}")
                if meta.get("hypothesis"):
                    parts.append(f"ГИПОТЕЗА (meta): {meta['hypothesis']}")
                if parts:
                    slots_context += "=== METADATA (document_meta) ===\n" + "\n\n".join(parts) + "\n\n"

            slot_filter_terms = None
            if len(requested_slots) == 1:
                slot_filter_terms = slot_map.get(requested_slots[0], None)

            con = get_conn()
            cur = con.cursor()

            if any(s in requested_slots for s in ["goal", "tasks", "object", "subject", "methods"]):
                cur.execute("""
                    SELECT text, section_path, element_type
                    FROM chunks
                    WHERE owner_id = ? AND doc_id = ?
                      AND (
                        section_path LIKE '%ВВЕД%' OR section_path LIKE '%введ%' OR
                        text LIKE '%ВВЕДЕНИЕ%' OR text LIKE '%Введение%' OR
                        section_path LIKE '%АННОТАЦ%' OR text LIKE '%Аннотац%'
                      )
                    ORDER BY id ASC
                    LIMIT 200
                """, (owner_id, doc_id))
                intro_rows = cur.fetchall()
                if intro_rows:
                    texts = []
                    for r in intro_rows:
                        t = r["text"]
                        if not t:
                            continue
                        if slot_filter_terms and not any(term in t.lower() for term in slot_filter_terms):
                            continue
                        texts.append(t)
                    intro_text = "\n\n".join(texts)
                    if intro_text:
                        slots_context += "=== ФРАГМЕНТЫ: ВВЕДЕНИЕ/АННОТАЦИЯ ===\n" + intro_text + "\n\n"

            if any(s in requested_slots for s in ["results", "conclusions"]):
                cur.execute("""
                    SELECT text, section_path, element_type
                    FROM chunks
                    WHERE owner_id = ? AND doc_id = ?
                      AND (
                        section_path LIKE '%ЗАКЛЮЧ%' OR section_path LIKE '%заключ%' OR
                        text LIKE '%ЗАКЛЮЧЕНИЕ%' OR text LIKE '%Заключение%' OR
                        text LIKE '%Выводы%' OR section_path LIKE '%вывод%'
                      )
                    ORDER BY id ASC
                    LIMIT 220
                """, (owner_id, doc_id))
                concl_rows = cur.fetchall()
                if concl_rows:
                    texts = []
                    for r in concl_rows:
                        t = r["text"]
                        if not t:
                            continue
                        if slot_filter_terms and not any(term in t.lower() for term in slot_filter_terms):
                            continue
                        texts.append(t)
                    concl_text = "\n\n".join(texts)
                    if concl_text:
                        slots_context += "=== ФРАГМЕНТЫ: ЗАКЛЮЧЕНИЕ/ВЫВОДЫ ===\n" + concl_text + "\n\n"

            con.close()

            slot_queries = []
            if "goal" in requested_slots:
                slot_queries.append("цель исследования")
            if "tasks" in requested_slots:
                slot_queries.append("задачи исследования")
            if "object" in requested_slots:
                slot_queries.append("объект исследования")
            if "subject" in requested_slots:
                slot_queries.append("предмет исследования")
            if "methods" in requested_slots:
                slot_queries.append("методы исследования")
            if "results" in requested_slots:
                slot_queries.append("результаты исследования")
            if "conclusions" in requested_slots:
                slot_queries.append("выводы заключение")

            rag_parts = []
            for sq in slot_queries[:7]:
                snippets = await asyncio.to_thread(retrieve, owner_id, doc_id, sq, top_k=5)
                rag_text = build_context(snippets) if snippets else ""
                if rag_text and len(rag_text) > 80:
                    rag_parts.append(f"[RAG: {sq}]\n{rag_text}")

            if rag_parts:
                slots_context += "=== ДОПОЛНИТЕЛЬНО (RAG ПО СЛОТАМ) ===\n" + "\n\n---\n\n".join(rag_parts) + "\n\n"

            if slots_context.strip():
                return f"""ИНФОРМАЦИЯ ДЛЯ ЗАПРОСА (СЛОТЫ):

{slots_context}

ВАЖНО: Отвечай строго по пунктам. Если чего-то нет в контексте — пиши 'В документе не указано'."""

        except Exception as e:
            logger.warning(f"Ошибка поиска SLOTS: {e}")

    # 6) CHAPTER — выделенная логика для конкретной главы/раздела
    if question_type == QuestionType.CHAPTER:
        q_lower = _normalize_word_numbers((question or "").lower())

        # Извлекаем номера глав/разделов
        chapter_nums = []
        chapter_nums += re.findall(r'глав[а-яё]*\s*(?:№\s*)?(\d+(?:\.\d+)*)', q_lower)
        chapter_nums += re.findall(r'\b(\d+(?:\.\d+)*)\s*[--–—]?\s*(?:й|я|е|го|му)?\s*глав[а-яё]*', q_lower)
        chapter_nums += re.findall(r'(?:раздел|подраздел|пункт|параграф)[а-яё]*\s*(?:№\s*)?(\d+(?:\.\d+)*)', q_lower)
        chapter_nums += re.findall(r'\b(\d+(?:\.\d+)+)\b', question or "")  # числа вида 2.1, 2.1.3

        all_nums = sorted(set(chapter_nums), key=len, reverse=True)
        logger.info("[CHAPTER] question='%s', all_nums=%s", question[:60], all_nums)

        if not all_nums:
            # Нет номера — fallback в SECTION
            question_type = QuestionType.SECTION
        else:
            chapter_context = ""
            _CHAPTER_CTX_LIMIT = 14000

            try:
                from .db import get_conn

                con = get_conn()
                cur = con.cursor()

                for num in all_nums:
                    if len(chapter_context) >= _CHAPTER_CTX_LIMIT:
                        break

                    # Стратегия 1: поиск heading-чанков по номеру главы
                    heading_patterns = []
                    if '.' not in str(num):
                        heading_patterns += [
                            f'%ГЛАВА {num}%', f'%Глава {num}%', f'%глава {num}%',
                            f'%ГЛАВА{num}%', f'%Глава{num}%',
                            f'%{num}.%', f'%{num} %',
                        ]
                    else:
                        heading_patterns += [
                            f'%{num}%',
                            f'%{num}.%',
                        ]

                    found_section_paths = []

                    # Ищем заголовки heading-типа
                    for pat in heading_patterns:
                        if found_section_paths:
                            break
                        cur.execute("""
                            SELECT text, section_path
                            FROM chunks
                            WHERE owner_id = ? AND doc_id = ? AND element_type = 'heading'
                              AND (text LIKE ? OR section_path LIKE ?)
                            ORDER BY id ASC
                            LIMIT 20
                        """, (owner_id, doc_id, pat, pat))
                        for row in cur.fetchall():
                            sp = row["section_path"]
                            if sp and sp not in found_section_paths:
                                found_section_paths.append(sp)

                    # Стратегия 2: если heading не нашлись, ищем по section_path в любых чанках
                    if not found_section_paths:
                        section_like_patterns = []
                        if '.' not in str(num):
                            section_like_patterns = [
                                f'Глава {num}%', f'ГЛАВА {num}%', f'глава {num}%',
                                f'{num} %', f'{num}.%',
                            ]
                        else:
                            section_like_patterns = [f'%{num}%']

                        for pat in section_like_patterns:
                            if found_section_paths:
                                break
                            cur.execute("""
                                SELECT DISTINCT section_path
                                FROM chunks
                                WHERE owner_id = ? AND doc_id = ?
                                  AND section_path LIKE ?
                                ORDER BY id ASC
                                LIMIT 10
                            """, (owner_id, doc_id, pat))
                            for row in cur.fetchall():
                                sp = row["section_path"]
                                if sp and sp not in found_section_paths:
                                    found_section_paths.append(sp)

                    logger.info("[CHAPTER] num=%s, found_section_paths=%s", num, found_section_paths[:5])

                    # Собираем контент по найденным section_path
                    for sp in found_section_paths:
                        if len(chapter_context) >= _CHAPTER_CTX_LIMIT:
                            break
                        cur.execute("""
                            SELECT text
                            FROM chunks
                            WHERE owner_id = ? AND doc_id = ? AND section_path LIKE ?
                            ORDER BY id ASC
                            LIMIT 200
                        """, (owner_id, doc_id, f'{sp}%'))

                        for crow in cur.fetchall():
                            ctext = crow["text"]
                            if ctext and len(ctext) > 30 and ctext not in chapter_context:
                                chapter_context += ctext + "\n\n"
                                if len(chapter_context) >= _CHAPTER_CTX_LIMIT:
                                    chapter_context += "\n[... контекст сокращён ...]\n"
                                    break

                    # Стратегия 3: если вообще ничего не нашли — пробуем text LIKE
                    if not chapter_context.strip():
                        text_search_patterns = []
                        if '.' not in str(num):
                            text_search_patterns = [
                                f'%Глава {num}%', f'%ГЛАВА {num}%',
                                f'%глава {num}%',
                            ]
                        else:
                            text_search_patterns = [f'%{num}%']

                        for pat in text_search_patterns:
                            if chapter_context.strip():
                                break
                            cur.execute("""
                                SELECT text, section_path
                                FROM chunks
                                WHERE owner_id = ? AND doc_id = ?
                                  AND text LIKE ?
                                ORDER BY id ASC
                                LIMIT 30
                            """, (owner_id, doc_id, pat))
                            for crow in cur.fetchall():
                                ctext = crow["text"]
                                if ctext and len(ctext) > 50 and ctext not in chapter_context:
                                    chapter_context += ctext + "\n\n"
                                    if len(chapter_context) >= _CHAPTER_CTX_LIMIT:
                                        break

                con.close()

            except Exception as e:
                logger.warning(f"[CHAPTER] Ошибка поиска главы: {e}", exc_info=True)

            if chapter_context.strip():
                rag_context = await asyncio.to_thread(
                    retrieve, owner_id, doc_id, question, top_k=5
                )
                rag_text = build_context(rag_context) if rag_context else ""

                full_ctx = f"""СОДЕРЖАНИЕ ГЛАВЫ/РАЗДЕЛА:

{chapter_context}

ДОПОЛНИТЕЛЬНЫЙ КОНТЕКСТ:
{rag_text}

ВАЖНО: Опиши содержание главы/раздела на основе этих данных. Если информации недостаточно — укажи это."""
                logger.info("[CHAPTER] Final context: %d chars", len(full_ctx))
                return full_ctx

            logger.warning("[CHAPTER] Ничего не нашли для nums=%s, fallback в SECTION", all_nums)
            # Если CHAPTER не нашёл данные — попробуем SECTION
            question_type = QuestionType.SECTION

    # 7) SECTION
    if question_type == QuestionType.SECTION:
        section_context = ""
        # Нормализуем словесные числительные → цифры (единая функция)
        q_lower = _normalize_word_numbers((question or "").lower())

        # 6.1) Извлекаем номера (поддержка "глава 2", "во 2-й главе", "первая глава" → "1 глава")
        chapter_nums = []
        chapter_nums += re.findall(r'глав[а-яё]*\s*(?:№\s*)?(\d+)', q_lower)
        chapter_nums += re.findall(r'\b(\d+)\s*[--–—]?\s*(?:й|я|е|го|му)?\s*глав[а-яё]*', q_lower)

        section_nums = re.findall(r'\b(\d+(?:\.\d+)+)\b', question or "")
        named_nums = re.findall(r'(?:раздел|подраздел|пункт|параграф)\s*(?:№\s*)?(\d+(?:\.\d+)*)', q_lower)

        all_nums = sorted(set(chapter_nums + section_nums + named_nums), key=len, reverse=True)

        logger.info(
            "[SECTION] question='%s', all_nums=%s, chapter_nums=%s",
            question[:60], all_nums, chapter_nums,
        )

        asks_structure = any(kw in q_lower for kw in ['структур', 'содержани', 'оглавлен', 'все главы', 'все подглав', 'план'])
        asks_conclusion = 'заключен' in q_lower or 'вывод' in q_lower

        try:
            from .db import get_conn

            if asks_structure or 'подглав' in q_lower:
                con = get_conn()
                cur = con.cursor()
                cur.execute("""
                    SELECT text
                    FROM chunks
                    WHERE owner_id = ? AND doc_id = ? AND element_type = 'heading'
                    ORDER BY id ASC
                    LIMIT 400
                """, (owner_id, doc_id))
                headings = cur.fetchall()
                con.close()

                if headings:
                    heading_texts = [row["text"] for row in headings if row["text"]]
                    if heading_texts:
                        section_context += "СТРУКТУРА ДОКУМЕНТА (ЗАГОЛОВКИ):\n\n" + "\n".join(heading_texts) + "\n\n"

            # Лимит на размер section_context, чтобы не перегружать API
            _SECTION_CTX_LIMIT = 12000

            if all_nums:
                con = get_conn()
                cur = con.cursor()

                for num in all_nums:
                    patterns = []

                    if re.search(r'\bглав[а-яё]*\b', q_lower) and '.' not in str(num):
                        patterns += [
                            f'%ГЛАВА {num}.%', f'%ГЛАВА {num} %',
                            f'%Глава {num}.%', f'%Глава {num} %',
                            f'%глава {num}.%', f'%глава {num} %',
                        ]
                    else:
                        patterns += [
                            f'{num} %',
                            f'{num}. %',
                            f'{num}.%',
                            f'%> {num} %',
                            f'%> {num}.%',
                            f'%{num} %',
                        ]

                    text_clause = " OR ".join(["text LIKE ?"] * len(patterns))
                    path_clause = " OR ".join(["section_path LIKE ?"] * len(patterns))

                    sql = f"""
                        SELECT text, section_path
                        FROM chunks
                        WHERE owner_id = ? AND doc_id = ? AND element_type = 'heading'
                          AND ( ({text_clause}) OR ({path_clause}) )
                        ORDER BY id ASC
                        LIMIT 20
                    """
                    cur.execute(sql, (owner_id, doc_id, *patterns, *patterns))
                    hrows = cur.fetchall()

                    found_section_paths = [row["section_path"] for row in hrows if row["section_path"]]
                    logger.info(
                        "[SECTION] num=%s, heading_rows=%d, section_paths=%s",
                        num, len(hrows), found_section_paths[:5],
                    )

                    for sp in found_section_paths:
                        if len(section_context) >= _SECTION_CTX_LIMIT:
                            logger.info("[SECTION] Context limit reached (%d chars), stopping", len(section_context))
                            break

                        cur.execute("""
                            SELECT text
                            FROM chunks
                            WHERE owner_id = ? AND doc_id = ? AND section_path LIKE ?
                            ORDER BY id ASC
                            LIMIT 200
                        """, (owner_id, doc_id, f'{sp}%'))

                        for crow in cur.fetchall():
                            ctext = crow["text"]
                            if ctext and len(ctext) > 50 and ctext not in section_context:
                                section_context += ctext + "\n\n"
                                if len(section_context) >= _SECTION_CTX_LIMIT:
                                    section_context += "\n[... контекст сокращён ...]\n"
                                    break

                con.close()
                logger.info("[SECTION] Collected section_context: %d chars", len(section_context))

            if asks_conclusion:
                con = get_conn()
                cur = con.cursor()
                cur.execute("""
                    SELECT text
                    FROM chunks
                    WHERE owner_id = ? AND doc_id = ?
                      AND (
                        text LIKE '%ЗАКЛЮЧЕНИЕ%' OR
                        text LIKE '%Заключение%' OR
                        section_path LIKE '%заключен%'
                      )
                    ORDER BY id ASC
                    LIMIT 80
                """, (owner_id, doc_id))
                rows = cur.fetchall()
                con.close()

                for row in rows:
                    t = row["text"]
                    if t and t not in section_context:
                        section_context += t + "\n\n"

            if section_context.strip():
                logger.info(f"[SECTION] Собран контекст: {len(section_context)} символов")

                rag_context = await asyncio.to_thread(
                    retrieve, owner_id, doc_id, question, top_k=5
                )
                rag_text = build_context(rag_context) if rag_context else ""

                full_ctx = f"""ИНФОРМАЦИЯ ПО ЗАПРОСУ:

{section_context}

ДОПОЛНИТЕЛЬНЫЙ КОНТЕКСТ:
{rag_text}

ВАЖНО: Отвечай на основе представленных данных. Если информации недостаточно — укажи это."""
                logger.info("[SECTION] Final context for LLM: %d chars", len(full_ctx))
                return full_ctx

            logger.warning("[SECTION] Специфичный поиск не дал результата (all_nums=%s), fallback в RAG coverage", all_nums)

        except Exception as e:
            logger.warning(f"Ошибка поиска разделов: {e}", exc_info=True)

    # 8) Общий RAG coverage fallback
    logger.info("[FALLBACK] Using RAG coverage fallback for question='%s', type=%s", question[:60], question_type)
    try:
        result = await asyncio.to_thread(
            retrieve_coverage,
            owner_id, doc_id, question,
            per_item_k=3,
            backfill_k=5,
        )
        if result and result.get('snippets'):
            ctx = build_context_coverage(result['snippets'])
            logger.info("[FALLBACK] retrieve_coverage returned %d chars", len(ctx) if ctx else 0)
            return ctx
    except Exception as e:
        logger.warning(f"retrieve_coverage failed: {e}, fallback to basic retrieve")

    snippets = await asyncio.to_thread(
        retrieve, owner_id, doc_id, question, top_k=10
    )
    ctx = build_context(snippets) if snippets else ""
    logger.info("[FALLBACK] basic retrieve returned %d chars, snippets=%d", len(ctx) if ctx else 0, len(snippets) if snippets else 0)
    return ctx

# ==================== ПОСТРОЕНИЕ ПРОМПТА ====================

def build_system_prompt(question_type: str) -> str:
    """
    Создаёт системный промпт в зависимости от типа вопроса.
    """
    
    base = (
        "Ты помощник по анализу выпускных квалификационных работ (ВКР).\n"
        "Отвечай СТРОГО на основе предоставленного контекста из документа.\n"
        "Не используй общие знания, догадки или типовые определения.\n"
        "Если в контексте нет информации для ответа - честно скажи об этом.\n"
        "\n"
        "ВАЖНО: ВСЕГДА завершай ответ полностью. Не обрывай на полуслове.\n"
        "Если ответ получается длинным - сократи его, но ОБЯЗАТЕЛЬНО напиши вывод.\n"
    )
    
    if question_type == QuestionType.CALC:
        return base + (
            "\nТип вопроса: ВЫЧИСЛИТЕЛЬНЫЙ.\n"
            "- Извлекай числа из таблиц и выполняй точные вычисления.\n"
            "- Показывай формулы и промежуточные результаты.\n"
            "- Не придумывай значения, которых нет в таблице.\n"
            "- В конце ОБЯЗАТЕЛЬНО напиши краткий вывод.\n"
        )
    
    elif question_type == QuestionType.TABLE:
        return base + (
            "\nТип вопроса: ТАБЛИЦА.\n"
            "- Анализируй структуру и данные таблицы.\n"
            "- Ссылайся на конкретные строки и столбцы.\n"
            "- Выделяй ключевые значения и тренды.\n"
            "- В конце ОБЯЗАТЕЛЬНО напиши краткий вывод.\n"
        )
    
    elif question_type == QuestionType.FIGURE:
        return base + (
            "\nТип вопроса: РИСУНОК/ГРАФИК.\n"
            "- Описывай визуальные элементы графика.\n"
            "- Указывай тренды, максимумы, минимумы.\n"
            "- Ссылайся на подпись и легенду рисунка.\n"
            "- В конце ОБЯЗАТЕЛЬНО напиши краткий вывод.\n"
        )
    
    elif question_type == QuestionType.SLOTS:
        return base + (
            "\nТип вопроса: ПАСПОРТ ВКР / СЛОТЫ.\n"
            "Дай ответ СТРОГО по пунктам:\n"
            "1) Цель\n"
            "2) Задачи\n"
            "3) Объект\n"
            "4) Предмет\n"
            "5) Методы\n"
            "6) Результаты\n"
            "7) Выводы\n"
            "Правила:\n"
            "- Заполняй каждый пункт только тем, что есть в контексте.\n"
            "- Если пункта нет — пиши: 'В документе не указано'.\n"
            "- Не подменяй цель задачами и наоборот.\n"
            "- Для задач: если это список — сохрани нумерацию.\n"
            "- В конце отдельный 'Краткий вывод' (1-2 предложения).\n"
        )
    
    elif question_type == QuestionType.CHAPTER:
        return base + (
            "\nТип вопроса: КОНКРЕТНАЯ ГЛАВА/РАЗДЕЛ.\n"
            "ПРАВИЛА:\n"
            "- Дай содержательное описание главы/раздела на основе контекста:\n"
            "  1) тема и назначение главы/раздела,\n"
            "  2) ключевые положения, определения, понятия,\n"
            "  3) что рассматривается по подпунктам (кратко по каждому),\n"
            "  4) выводы по главе (если есть в тексте).\n"
            "- Не навязывай пункты 'цель/объект/предмет/задачи/методы', если пользователь об этом не спрашивал.\n"
            "- Если информации в контексте нет — честно скажи: 'В документе не указано'.\n"
            "- В конце: краткий вывод.\n"
        )

    elif question_type == QuestionType.SECTION:
        return base + (
            "\nТип вопроса: РАЗДЕЛ/ГЛАВА/СТРУКТУРА.\n"
            "ПРАВИЛА:\n"
            "- Отвечай на конкретный запрос пользователя (описать главу, перечислить подглавы, раскрыть содержание раздела).\n"
            "- Если пользователь просит 'опиши главу N' — дай краткое содержание главы:\n"
            "  1) тема и назначение главы,\n"
            "  2) ключевые положения и определения,\n"
            "  3) что рассматривается по подпунктам,\n"
            "  4) выводы по главе (если есть в тексте).\n"
            "- Если пользователь просит 'структуру/оглавление/план' — перечисли заголовки и подзаголовки.\n"
            "- Не навязывай пункты 'цель/объект/предмет/задачи/методы', если пользователь об этом не спрашивал.\n"
            "- Если информации в контексте нет — честно скажи: 'В документе не указано'.\n"
            "- В конце: краткий вывод.\n"
        )


    else:  # SEMANTIC или SECTION
        return base + (
            "\nТип вопроса: СМЫСЛОВОЙ.\n"
            "ПРАВИЛА ОТВЕТА:\n"
            "- Отвечай ТОЛЬКО на основе контекста из документа.\n"
            "- Приводи прямые цитаты в кавычках «...».\n"
            "- НЕ используй ссылки [S1], [S2] — они не видны пользователю.\n"
            "- Если информации нет в контексте — честно скажи: 'В документе не указано'.\n"
            "- НЕ выдумывай 'предполагаемые формулировки' — только факты из документа.\n"
            "- Структурируй ответ: пункты с конкретными данными.\n"
            "- В конце — краткий вывод из 1-2 предложений.\n"
        )


# ==================== ГЛАВНЫЙ ПАЙПЛАЙН ====================

class UnifiedPipeline:
    """
    Единый пайплайн для обработки ВСЕХ типов вопросов.
    """
    
    def __init__(self):
        self._cache = {}  # Простой кэш ответов
        self._cache_size = 100  # Максимум элементов в кэше
    
    def _get_cache_key(self, doc_id: int, question: str) -> str:
        """Создаёт ключ для кэша"""
        import hashlib
        q_hash = hashlib.md5(question.encode()).hexdigest()[:8]
        return f"{doc_id}:{q_hash}"
    
    async def answer(
        self,
        owner_id: int,
        doc_id: int,
        question: str,
        *,
        use_cache: bool = True,
        stream: bool = False,
    ) -> str:
        """
        Главная функция: отвечает на вопрос по документу.
        """

        # 1) Проверяем кэш
        if use_cache:
            cache_key = self._get_cache_key(doc_id, question)
            if cache_key in self._cache:
                logger.info(f"Cache HIT for: {question[:50]}...")
                return self._cache[cache_key]

        # 2) Multi-intent: декомпозируем на под-вопросы, LLM вызываем ОДИН раз
        if _looks_like_multi_slot(question):
            intents = _extract_intents_in_order(question)
            logger.info(f"Multi-intent question detected: intents={intents}")

            slot_blocks: List[Dict[str, str]] = []

            # 2.1) Если в вопросе есть "slots" — раскладываем на конкретные слоты (goal/tasks/...)
            if "slots" in intents:
                slots = _extract_slots_in_order(question)

                # если человек написал “паспорт/краткий план”, но не перечислил — берём стандартный набор
                if not slots and any(t in question.lower() for t in ("паспорт", "краткий план", "план вкр")):
                    slots = ["goal", "tasks", "object", "subject", "methods", "results", "conclusions"]

                for slot in slots:
                    subq = _build_subquestion_for_slot(question, slot)

                    # КЛЮЧЕВОЙ ФИКС:
                    # слоты ВСЕГДА обрабатываем через QuestionType.SLOTS,
                    # чтобы get_context_for_question подтягивал введение/аннотацию/заключение и meta.
                    sub_context = await get_context_for_question(doc_id, owner_id, subq, QuestionType.SLOTS)

                    slot_blocks.append({
                        "slot": slot,
                        "title": (_SLOTS.get(slot) or {}).get("title", slot),
                        "subq": subq,
                        "context": sub_context or "",
                    })

            # 2.2) Остальные намерения: chapter/table/figure/gost/calc (в порядке появления)
            for intent in intents:
                if intent == "slots":
                    continue

                if intent == "chapter":
                    subq = _build_subquestion_for_slot(question, "chapter")
                    # Используем CHAPTER для конкретных глав (не SECTION!)
                    sub_context = await get_context_for_question(doc_id, owner_id, subq, QuestionType.CHAPTER)
                    slot_blocks.append({
                        "slot": "chapter",
                        "title": "Глава/раздел",
                        "subq": subq,
                        "context": sub_context or "",
                    })

                elif intent == "table":
                    subq = _build_subquestion_for_slot(question, "table")
                    sub_context = await get_context_for_question(doc_id, owner_id, subq, QuestionType.TABLE)
                    slot_blocks.append({
                        "slot": "table",
                        "title": "Таблица",
                        "subq": subq,
                        "context": sub_context or "",
                    })

                elif intent == "figure":
                    subq = _build_subquestion_for_slot(question, "figure")
                    sub_context = await get_context_for_question(doc_id, owner_id, subq, QuestionType.FIGURE)
                    slot_blocks.append({
                        "slot": "figure",
                        "title": "Рисунок/график",
                        "subq": subq,
                        "context": sub_context or "",
                    })

                elif intent == "gost":
                    subq = (
                        "Какие требования к оформлению (ГОСТ/шрифт/поля/интервал) указаны в документе? "
                        "Выпиши только то, что явно написано."
                    )
                    sub_context = await get_context_for_question(doc_id, owner_id, subq, QuestionType.GOST)
                    slot_blocks.append({
                        "slot": "gost",
                        "title": "Оформление/ГОСТ",
                        "subq": subq,
                        "context": sub_context or "",
                    })

                elif intent == "calc":
                    subq = question  # калькуляцию лучше не перефразировать
                    sub_context = await get_context_for_question(doc_id, owner_id, subq, QuestionType.CALC)
                    slot_blocks.append({
                        "slot": "calc",
                        "title": "Расчёт",
                        "subq": subq,
                        "context": sub_context or "",
                    })

            merged_context = _merge_slot_contexts(slot_blocks)

            if not merged_context or len(merged_context) < 100:
                return (
                    "В документе не найдено информации для ответа на этот вопрос.\n"
                    "Попробуйте:\n"
                    "- Уточнить формулировку\n"
                    "- Указать номер главы/раздела\n"
                    "- Спросить про другой аспект работы"
                )

            system_prompt = _build_multi_system_prompt()

            if stream and chat_with_gpt_stream:
                return self._answer_stream(system_prompt, question, merged_context)

            answer = await self._answer_sync(system_prompt, question, merged_context)

            if not answer or not answer.strip():
                logger.warning(
                    "[PIPELINE] LLM returned empty after retries (multi-intent), question='%s'",
                    question[:80],
                )
                answer = (
                    "Не удалось получить ответ от языковой модели.\n"
                    "Возможные причины:\n"
                    "- Временные проблемы с сервисом генерации ответов\n"
                    "- Слишком большой объём контекста\n\n"
                    "Попробуйте повторить запрос через несколько секунд."
                )

            if use_cache and answer:
                cache_key = self._get_cache_key(doc_id, question)
                self._update_cache(cache_key, answer)
            return answer

        # 3) Обычный режим (single-intent)
        question_type = detect_question_type(question)
        logger.info(f"[PIPELINE] Single-intent: question='{question[:60]}', type={question_type}")

        context = await get_context_for_question(doc_id, owner_id, question, question_type)

        ctx_len = len(context) if context else 0
        logger.info(f"[PIPELINE] Context retrieved: {ctx_len} chars for type={question_type}")

        if not context or len(context) < 100:
            return (
                "В документе не найдено информации для ответа на этот вопрос.\n"
                "Попробуйте:\n"
                "- Уточнить формулировку\n"
                "- Указать номер главы/раздела\n"
                "- Спросить про другой аспект работы"
            )

        system_prompt = build_system_prompt(question_type)

        if stream and chat_with_gpt_stream:
            return self._answer_stream(system_prompt, question, context)

        answer = await self._answer_sync(system_prompt, question, context)

        if not answer or not answer.strip():
            logger.warning(
                "[PIPELINE] LLM returned empty after retries, returning fallback for question='%s'",
                question[:80],
            )
            answer = (
                "Не удалось получить ответ от языковой модели.\n"
                "Возможные причины:\n"
                "- Временные проблемы с сервисом генерации ответов\n"
                "- Слишком большой объём контекста\n\n"
                "Попробуйте повторить запрос через несколько секунд."
            )

        if use_cache and answer:
            cache_key = self._get_cache_key(doc_id, question)
            self._update_cache(cache_key, answer)
        return answer



    
    async def _answer_sync(self, system: str, question: str, context) -> str:
        """Синхронный вызов LLM (поддерживает мультимодальность)"""
        try:
            # Проверяем, мультимодальный ли контекст
            images = []
            text_context = context

            if isinstance(context, dict) and context.get("type") == "multimodal":
                # Мультимодальный режим: есть картинки
                text_context = context.get("text", "")
                images = context.get("images", [])
                logger.info(f"Multimodal request with {len(images)} image(s)")
            elif isinstance(context, str):
                # Обычный текстовый режим
                text_context = context
            else:
                # Странный формат - пытаемся извлечь текст
                text_context = str(context)

            ctx_len = len(text_context) if isinstance(text_context, str) else 0
            logger.info(
                "[PIPELINE] _answer_sync: question='%s', context_len=%d, images=%d",
                question[:80], ctx_len, len(images),
            )

            # Формируем сообщения
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Контекст из документа:\n{text_context}\n\nВопрос: {question}"}
            ]

            # Вызываем API (с картинками или без) с ретраем при пустом ответе
            max_retries = 2
            answer = ""
            for attempt in range(1, max_retries + 1):
                if images:
                    answer = await asyncio.to_thread(
                        chat_with_gpt_multimodal,
                        messages=messages,
                        images=images,
                        temperature=0.2,
                        max_tokens=4000,
                    )
                else:
                    answer = await asyncio.to_thread(
                        chat_with_gpt,
                        messages=messages,
                        temperature=0.3,
                        max_tokens=4000,
                    )

                if answer and answer.strip():
                    break

                logger.warning(
                    "[PIPELINE] LLM returned empty on attempt %d/%d for question='%s'",
                    attempt, max_retries, question[:80],
                )
                if attempt < max_retries:
                    await asyncio.sleep(1.5)

            logger.info(
                "[PIPELINE] _answer_sync result: answer_len=%d, empty=%s",
                len(answer) if answer else 0, not bool(answer),
            )
            return answer
        except Exception as e:
            logger.error(f"LLM call failed: {e}", exc_info=True)
            return f"Ошибка при обращении к модели: {e}"
    
    async def _answer_stream(self, system: str, question: str, context: str):
        """Стриминговый вызов LLM (генератор)"""
        try:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {question}"}
            ]
            
            # Генератор чанков
            async for chunk in chat_with_gpt_stream(
                messages=messages,
                temperature=0.3,
                max_tokens=4000,
            ):
                yield chunk
        except Exception as e:
            logger.error(f"LLM stream failed: {e}")
            yield f"Ошибка: {e}"
    
    def _update_cache(self, key: str, value: str):
        """Обновляет кэш (с ограничением размера)"""
        if len(self._cache) >= self._cache_size:
            # Удаляем самый старый элемент (FIFO)
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        
        self._cache[key] = value


# ==================== ГЛОБАЛЬНЫЙ ЭКЗЕМПЛЯР ====================

# Создаём единственный экземпляр для всего приложения
pipeline = UnifiedPipeline()

# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

async def answer_question(
    user_id: int,
    question: str,
    *,
    stream: bool = False,
) -> str:
    """
    Удобная обёртка для использования в bot.py
    
    Пример использования:
        answer = await answer_question(user_id=123, question="Что такое ВКР?")
    """
    # Получаем активный документ пользователя
    doc_id = await asyncio.to_thread(get_user_active_doc, user_id)
    
    if not doc_id:
        return (
            "У вас нет активного документа.\n"
            "Загрузите файл ВКР командой /upload или отправьте его сюда."
        )
    
    # Вызываем unified pipeline
    return await pipeline.answer(
        owner_id=user_id,
        doc_id=doc_id,
        question=question,
        stream=stream,
    )


# ==================== ЭКСПОРТ ====================

__all__ = [
    'UnifiedPipeline',
    'pipeline',
    'answer_question',
    'detect_question_type',
    'QuestionType',
]

def _format_chart_data_simple(chart_data: dict) -> str:
    """Простое форматирование chart_data без chart_extractor"""
    if not chart_data:
        return ""
    
    lines = []
    chart_type = chart_data.get('type', 'диаграмма')
    lines.append(f"Тип: {chart_type}")
    
    if chart_data.get('title'):
        lines.append(f"Заголовок: {chart_data['title']}")
    
    lines.append("\nДанные:")
    
    for series in chart_data.get('series', []):
        if series.get('name'):
            lines.append(f"\nСерия: {series['name']}")
        
        categories = series.get('categories') or chart_data.get('categories', [])
        values = series.get('values', [])
        
        for cat, val in zip(categories, values):
            if isinstance(val, float) and val < 1:
                lines.append(f"  - {cat}: {val*100:.1f}%")
            else:
                lines.append(f"  - {cat}: {val}")
    
    return '\n'.join(lines)