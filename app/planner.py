# app/planner.py
"""
Планировщик подзадач для длинных пользовательских запросов по ВКР.

Идея:
- Если запрос простой — используем старую логику (bot.py решит через is_big_complex_query).
- Если запрос длинный/намешанный — разбиваем его на подзадачи:
    * теория (ПБУ, МСФО, рентабельность, резервы и т.п.)
    * предприятие (описание, оргструктура)
    * анализ финансового состояния / структуры активов
    * таблицы (по номерам: 2.2, 3.1, 4 и т.д.)
    * рисунки (по разделу/номеру: 2.3 и т.п.)
    * проводки (типовые проводки, проводки по данным таблиц)

Сейчас всё делаем на правилах (регулярки + ключевые слова),
без прямых вызовов LLM.

Добавлено:
- Более гибкий разбор таблиц:
  * поддержка форм вида "Таблица. 3.1", "таблица № 3.1"
  * дополнительно ищем "голые" номера вида 3.1 / 3.3, если рядом есть
    слова вроде "таблица", "запасы", "активы", "показатели" и т.п.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Iterable, Tuple
import re


# ==========================
#  Типы задач и структура
# ==========================

class TaskType(str, Enum):
    """Тип подзадачи, чтобы дальше в bot.py понять,
    какой сценарий/пайплайн вызывать.
    """
    THEORY = "theory"                  # Теория: ПБУ, МСФО, рентабельность, запасы, резервы и т.п.
    ENTERPRISE = "enterprise"          # Кратко о предприятии, оргструктура
    ENTERPRISE_FINANCE = "enterprise_finance"  # Структура активов, фин. состояние, прибыль/убыток
    TABLE = "table"                    # Анализ таблицы по номеру/обозначению
    FIGURES = "figures"                # Анализ рисунков/графиков по пункту/разделу
    POSTINGS = "postings"              # Проводки (теория + по данным ВКР)
    RAW = "raw"                        # Запасной вариант: сырая задача, если не удалось классифицировать


@dataclass
class Task:
    """Одна подзадача, выделенная из длинного запроса."""
    type: TaskType
    title: str                          # Короткое имя задачи (для заголовка в ответе)
    description: str                    # Человекочитаемое описание, что нужно сделать
    # Доп. поля для конкретных типов:
    table_ref: Optional[str] = None     # Например "2.2" или "3.1"
    figure_ref: Optional[str] = None    # Например "2.3" или "рисунок 2.3"
    topics: List[str] = field(default_factory=list)  # Для теории: список тем
    # Сервисное:
    original_text_span: Optional[str] = None  # Часть исходного запроса, на которую опирались
    priority: int = 0                         # Можно использовать для сортировки


# ==========================
#  Утилиты
# ==========================

WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Чуть-чуть чистим текст: убираем лишние пробелы."""
    return WHITESPACE_RE.sub(" ", text).strip()


# ==========================
#  Детект "большого" запроса
# ==========================

def is_big_complex_query(text: str) -> bool:
    """Очень простой эвристический детектор "монстр-запроса".

    ЭТО ТО, ЧТО НУЖНО БУДЕТ ВЫЗВАТЬ В bot.py,
    чтобы решить: включать ли режим планировщика.
    """
    clean = normalize_text(text)
    length = len(clean)

    # Минимальный триггер по длине
    if length > 800:
        return True

    # Кол-во "тяжёлых" ключевых слов
    keywords = [
        "таблица", "табл.", "рисунок", "рис.", "показател", "рентабельн",
        "обязательств", "резерв", "обесцен", "оборотн", "актив",
        "запас", "пбу", "п. б. у", "мсфо", "связанн", "проводк",
        "финансовое состояние", "структура активов", "предприят"
    ]
    hits = sum(1 for kw in keywords if kw.lower() in clean.lower())

    # Если и текст не короткий, и много триггеров – считаем запрос сложным
    if length > 400 and hits >= 4:
        return True

    return False


# ==========================
#  Парсинг подзадач
# ==========================

# 1) Основной паттерн для таблиц: "таблица 2.2", "табл. 3.1", "таблица. 3.1", "таблица № 3.1"
TABLE_RE = re.compile(
    r"(таблиц[аеы]?|табл\.)\s*[\.\,:;№-]*\s*([0-9]+(?:\.[0-9]+)*)",
    flags=re.IGNORECASE
)

# 2) Паттерн для "голых" номеров вида "3.1", "3.3" и т.п.
BARE_TABLE_NUM_RE = re.compile(
    r"\b([0-9]+(?:\.[0-9]+)+)\b"
)

FIGURE_RE = re.compile(
    r"(рисунк[аеы]?|рис\.)\s*([0-9]+(?:\.[0-9]+)*)",
    flags=re.IGNORECASE
)

SECTION_RE = re.compile(
    r"пункт[е]?\s*([0-9]+(?:\.[0-9]+)*)",
    flags=re.IGNORECASE
)


def extract_table_refs(text: str) -> List[str]:
    """Находит упоминания таблиц по паттернам:
    'таблица 2.2', 'табл. 3.1', 'таблица. 3.1' и т.п.
    Возвращает уникальные ссылки.
    """
    refs: List[str] = []
    for m in TABLE_RE.finditer(text):
        ref = m.group(2)
        if ref not in refs:
            refs.append(ref)
    return refs


def extract_additional_table_refs_from_bare_numbers(
    text: str,
    already: List[str],
    exclude: List[str],
    window: int = 40,
) -> List[str]:
    """Дополнительно ищем номера вида '3.1', '3.3' и т.п., даже если рядом явно не написано 'таблица'.

    Логика:
    - ищем все вхождения паттерна N.N (минимум одна точка);
    - вокруг каждого матча берём контекст +-window символов;
    - если в контексте присутствует одно из ключевых слов:
        'таблица', 'табл', 'запас', 'актив', 'показател'
      и при этом номер ещё не в already/exclude,
      считаем это ссылкой на таблицу.
    - exclude используется, чтобы не путать, например, '2.3' из 'пункт 2.3' с таблицей.
    """
    lower = text.lower()
    extra: List[str] = []

    for m in BARE_TABLE_NUM_RE.finditer(text):
        ref = m.group(1)
        if ref in already or ref in extra or ref in exclude:
            continue

        start, end = m.span(1)
        left = max(0, start - window)
        right = min(len(text), end + window)
        ctx = lower[left:right]

        # Слова, которые часто сопровождают номера таблиц в речи студентов
        context_keywords = [
            "таблиц", "таблица", "табл.",
            "запас", "запасы",
            "актив", "активов",
            "показател", "динамик", "темп"
        ]
        if any(kw in ctx for kw in context_keywords):
            extra.append(ref)

    return extra


def extract_figure_refs(text: str) -> List[str]:
    """Находит упоминания рисунков по паттернам:
    'рисунок 2.3', 'рис. 3.1'."""
    refs: List[str] = []
    for m in FIGURE_RE.finditer(text):
        ref = m.group(2)
        if ref not in refs:
            refs.append(ref)
    return refs


def detect_theory_topics(text: str) -> List[str]:
    """Выделяем теоретические темы, которые пользователь просит объяснить.
    Это будет использовано для Task(type=THEORY).
    """
    lower = text.lower()
    topics: List[str] = []

    def add(name: str):
        if name not in topics:
            topics.append(name)

    if "операцион" in lower:
        add("операционные и неоперационные расходы")
    if "неоперацион" in lower:
        add("операционные и неоперационные расходы")
    if "пбу" in lower or "п. б. у" in lower:
        add("ПБУ")
    if "мсф" in lower:  # МСФО
        add("МСФО")
    if "глав" in lower or "глап" in lower:
        add("ГЛАВ / российские стандарты")
    if "проводк" in lower:
        add("типовые бухгалтерские проводки")
    if "рентабельн" in lower:
        add("показатели рентабельности")
    if "оборотн" in lower and "актив" in lower:
        add("оборотные активы")
    if "запас" in lower:
        add("запасы товаров / материалов")
    if "связанн" in lower and "сторон" in lower:
        add("связанные стороны")
    if "резерв" in lower:
        add("резервы (сомнительные долги, обесценение и др.)")
    if "обесцен" in lower:
        add("обесценение внеоборотных активов и финансовых инструментов")
    if "текущ" in lower and "обязательств" in lower:
        add("текущие обязательства")
    if "биржев" in lower and "стоим" in lower:
        add("биржевая стоимость и её отличие от балансовой")

    return topics


def detect_enterprise_flags(text: str) -> Tuple[bool, bool]:
    """Определяем, просит ли пользователь:
    - кратко рассказать о предприятии,
    - проанализировать структуру активов / фин. состояние.
    """
    lower = text.lower()

    want_overview = False
    want_finance = False

    if "предприяти" in lower or "организаци" in lower:
        want_overview = True
    if "оргструктур" in lower or ("организационн" in lower and "структур" in lower):
        want_overview = True

    finance_keywords = [
        "структур", "активов", "финансовое состояние",
        "финансового состояния", "убытк", "прибыл", "много денег"
    ]
    if any(kw in lower for kw in finance_keywords):
        want_finance = True

    return want_overview, want_finance


def detect_postings_requested(text: str) -> bool:
    """Понимаем, просит ли пользователь про проводки по данным ВКР."""
    lower = text.lower()
    if "проводк" in lower:
        return True
    return False


# ==========================
#  Основная функция планировщика
# ==========================

def plan_tasks_from_user_query(
    text: str,
    max_tasks: Optional[int] = None
) -> List[Task]:
    """Главная функция планировщика.

    STRICT-режим (новая логика):
    - Планировщик НЕ должен создавать задачи, которые провоцируют ответ "из головы"
      (теория, типовые определения, проводки и т.п.).
    - Такие запросы превращаем в RAW-задачу "найти в тексте ВКР", чтобы дальше retrieval
      либо нашёл опору, либо честно отказал ("в документе не найдено").
    """
    clean = normalize_text(text)
    lower = clean.lower()
    tasks: List[Task] = []

    # 1. Теория (СТРОГО: только если это есть в тексте ВКР)
    theory_topics = detect_theory_topics(clean)
    if theory_topics:
        tasks.append(
            Task(
                type=TaskType.RAW,
                title="Теория (только из текста ВКР)",
                description=(
                    "Найти в тексте ВКР фрагменты по следующим темам и пересказать строго по ним, "
                    "без добавления общих определений: "
                    + ", ".join(theory_topics)
                ),
                original_text_span=text,
                priority=10,
            )
        )

    # 2. Предприятие / его структура
    want_overview, want_finance = detect_enterprise_flags(clean)
    if want_overview:
        tasks.append(
            Task(
                type=TaskType.ENTERPRISE,
                title="Кратко о предприятии и оргструктуре",
                description=(
                    "Кратко описать предприятие, его профиль деятельности "
                    "и организационную структуру ТОЛЬКО на основе текста ВКР."
                ),
                original_text_span=text,
                priority=9,
            )
        )
    if want_finance:
        tasks.append(
            Task(
                type=TaskType.ENTERPRISE_FINANCE,
                title="Финансовое состояние и структура активов",
                description=(
                    "Проанализировать структуру активов и общее финансовое "
                    "состояние предприятия ТОЛЬКО на основе данных ВКР: динамику, прибыль/убыток, "
                    "наличие проблем или устойчивость."
                ),
                original_text_span=text,
                priority=8,
            )
        )

    # 3. Рисунки / разделы (сначала собираем, чтобы потом не путать с таблицами)
    figure_refs = extract_figure_refs(clean)

    section_refs: List[str] = []
    for m in SECTION_RE.finditer(clean):
        sec = m.group(1)
        if sec not in section_refs:
            section_refs.append(sec)

    # 4. Таблицы: сначала явные "таблица 2.2", затем дополнительные "3.1/3.3" по контексту
    table_refs = extract_table_refs(clean)

    exclude_for_tables = set(figure_refs + section_refs)

    extra_table_refs = extract_additional_table_refs_from_bare_numbers(
        clean,
        already=table_refs,
        exclude=list(exclude_for_tables),
    )

    all_table_refs: List[str] = []
    for ref in table_refs + extra_table_refs:
        if ref not in all_table_refs:
            all_table_refs.append(ref)

    for ref in all_table_refs:
        tasks.append(
            Task(
                type=TaskType.TABLE,
                title=f"Анализ таблицы {ref}",
                description=(
                    f"Проанализировать таблицу {ref} ТОЛЬКО по данным ВКР: структуру, динамику, "
                    f"темпы роста/сокращения и выводы."
                ),
                table_ref=ref,
                original_text_span=text,
                priority=7,
            )
        )

    # 5. Рисунки / графики
    for ref in figure_refs:
        tasks.append(
            Task(
                type=TaskType.FIGURES,
                title=f"Анализ рисунков {ref}",
                description=(
                    f"Найти рисунок(и) {ref} и проанализировать ТОЛЬКО по данным ВКР: "
                    f"что показано, как меняются показатели, какие выводы сделаны в тексте."
                ),
                figure_ref=ref,
                original_text_span=text,
                priority=6,
            )
        )

    if "рисунк" in lower or "рис." in lower:
        for sec in section_refs:
            tasks.append(
                Task(
                    type=TaskType.FIGURES,
                    title=f"Анализ рисунков в пункте {sec}",
                    description=(
                        f"Проанализировать рисунки/графики в пункте {sec} ТОЛЬКО по данным ВКР: "
                        f"описать динамику, тренды и выводы, которые есть в тексте."
                    ),
                    figure_ref=sec,
                    original_text_span=text,
                    priority=6,
                )
            )

    # 6. Проводки (СТРОГО: только если это есть в тексте/таблицах ВКР)
    if detect_postings_requested(clean):
        tasks.append(
            Task(
                type=TaskType.RAW,
                title="Проводки (только если есть в ВКР)",
                description=(
                    "Найти в тексте/таблицах ВКР упоминания бухгалтерских проводок или примеры операций "
                    "и пересказать строго по найденным фрагментам. Если в ВКР этого нет — сообщить, что не найдено."
                ),
                original_text_span=text,
                priority=5,
            )
        )

    # Если вообще ничего не распознали — создадим одну RAW-задачу
    if not tasks:
        tasks.append(
            Task(
                type=TaskType.RAW,
                title="Общий вопрос по ВКР",
                description=(
                    "Ответить на общий вопрос пользователя по ВКР "
                    "ТОЛЬКО на основе предоставленного текста и данных."
                ),
                original_text_span=text,
                priority=0,
            )
        )

    tasks.sort(key=lambda t: t.priority, reverse=True)

    if max_tasks is not None and max_tasks > 0:
        tasks = tasks[:max_tasks]

    return tasks

# ==========================
#  Разбиение задач на батчи
# ==========================

def batch_tasks(tasks: Iterable[Task], batch_size: int = 3) -> List[List[Task]]:
    """Разбить список задач на батчи по batch_size штук.
    Это поможет в bot.py обрабатывать по 2–3 подзадачи за один ответ.
    """
    batch: List[Task] = []
    result: List[List[Task]] = []
    for t in tasks:
        batch.append(t)
        if len(batch) >= batch_size:
            result.append(batch)
            batch = []
    if batch:
        result.append(batch)
    return result
