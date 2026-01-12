# app/document_semantic_planner.py
"""
Высокоуровневый «семантический планировщик» вопросов по документу.

Цель модуля — в одном месте решать, О ЧЁМ именно спрашивает пользователь
и К КАКИМ кускам документа (главы / таблицы / рисунки) относится его вопрос.

И дополнительно — давать высокоуровневый ответ для "репетиторских" запросов
по введению и главам (актуальность, объект/предмет, цель, задачи, гипотеза, выводы),
если удалось собрать нормальный контекст.
"""

from __future__ import annotations
from .db import get_document_base_meta
import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Literal
from .db import get_document_sections

logger = logging.getLogger(__name__)

# ------------------------------------
# Импортируем готовые хелперы из intents (если они есть)
# ------------------------------------
try:  # штатный путь
    from .intents import (  # type: ignore
        extract_table_numbers,
        extract_figure_numbers,
        extract_section_hints,
    )
except Exception:  # фолбэк: простые парсеры внутри этого модуля
    logger.exception("document_semantic_planner: fallback to local parsers")

    TABLE_NUM_RE = re.compile(
        r"(?i)\b(таблица|табл\.|table)\s*([A-Za-zА-Яа-я]?\s*\d+(?:[.,]\d+)*)"
    )
    FIG_NUM_RE = re.compile(
        r"(?i)\b(рисунок|рис\.|figure|fig\.)\s*([A-Za-zА-Яа-я]?\s*\d+(?:[.,]\d+)*)"
    )
    SECTION_HINT_RE = re.compile(
        r"(?i)\b(глава|раздел|пункт|подраздел|§|section|chapter|clause)\s*([A-Za-zА-Яа-я]?\s*\d+(?:[.,]\d+)*)"
    )

    def _normalize_num(val: str) -> str:
        # убираем пробелы и запятые → точки, нормализуем буква+цифры
        s = (val or "").replace(" ", "").replace(",", ".")
        s = re.sub(r"([A-Za-zА-Яа-я])[.-]?(?=\d)", r"\1", s)
        return s.strip()

    def extract_table_numbers(text: str) -> List[str]:  # type: ignore
        out: List[str] = []
        for m in TABLE_NUM_RE.findall(text or ""):
            num = _normalize_num(m[1] or "")
            if num:
                out.append(num)
        # уникализируем, сохраняя порядок
        seen = set()
        uniq: List[str] = []
        for v in out:
            if v not in seen:
                seen.add(v)
                uniq.append(v)
        return uniq

    def extract_figure_numbers(text: str) -> List[str]:  # type: ignore
        out: List[str] = []
        for m in FIG_NUM_RE.findall(text or ""):
            num = _normalize_num(m[1] or "")
            if num:
                out.append(num)
        seen = set()
        uniq: List[str] = []
        for v in out:
            if v not in seen:
                seen.add(v)
                uniq.append(v)
        return uniq

    def extract_section_hints(text: str) -> List[str]:  # type: ignore
        out: List[str] = []
        for m in SECTION_HINT_RE.findall(text or ""):
            num = _normalize_num(m[1] or "")
            if num:
                out.append(num)
        seen = set()
        uniq: List[str] = []
        for v in out:
            if v not in seen:
                seen.add(v)
                uniq.append(v)
        return uniq


# ------------------------------------
# Базовые структуры данных
# ------------------------------------

SemanticMode = Literal[
    "generic",          # обычный вопрос «как есть»
    "tutor_overview",   # объяснение как репетитор / конспект по главам
    "compare_objects",  # сравнение таблиц / рисунков / разделов
]


@dataclass
class DocumentObjectRef:
    """
    Ссылка на объект документа, к которому относится вопрос:
      - раздел/глава (SECTION)
      - таблица (TABLE)
      - рисунок (FIGURE)
    """
    kind: Literal["section", "table", "figure"]
    key: str                     # '1', '2.3', 'A1', 'intro' и т.п.
    label: str                   # человекочитаемое: 'глава 2', 'таблица 2.3', 'введение'
    source_span: Optional[str] = None  # фрагмент вопроса, из которого это извлекли


@dataclass
class SemanticSlots:
    """
    Набор «смысловых слотов» — о чём именно просит пользователь.
    Слоты подобраны под типичные ВКР/курсовые.
    """
    relevance: bool = False     # актуальность темы
    obj: bool = False           # объект исследования
    subj: bool = False          # предмет исследования
    goal: bool = False          # цель исследования
    tasks: bool = False         # задачи исследования
    hypothesis: bool = False    # гипотеза
    chapter_conclusions: bool = False  # выводы по главам / разделам

    def any_slot_requested(self) -> bool:
        return any(
            [
                self.relevance,
                self.obj,
                self.subj,
                self.goal,
                self.tasks,
                self.hypothesis,
                self.chapter_conclusions,
            ]
        )


@dataclass
class SemanticPlan:
    """
    Итоговый «план понимания» вопроса:
      - какой это режим (mode),
      - к каким объектам документа относится,
      - какие смысловые слоты нужно заполнить.
    """
    mode: SemanticMode = "generic"
    objects: List[DocumentObjectRef] = field(default_factory=list)
    slots: SemanticSlots = field(default_factory=SemanticSlots)
    original_question: str = ""
    normalized_question: str = ""  # можно использовать для переформулировки


# ------------------------------------
# Детект режимов и слотов
# ------------------------------------

_TUTOR_RE = re.compile(
    r"(?i)\b(как\s+репетитор|как\s+препод(ователь)?|объясни\s+простым\s+языком|"
    r"объясни\s+как\s+для\s+студента|объясни\s+по\s+простому)\b"
)

_COMPARE_RE = re.compile(
    r"(?i)\b(сравн(и|ить)|сопостав(ь|ить)|отличи(я|е)|различи(я|е))\b"
)

_INTRO_RE = re.compile(r"(?i)\bвведен(ие|ии|ия)?\b")


def _detect_mode(text: str) -> SemanticMode:
    t = (text or "").strip()
    if not t:
        return "generic"
    if _TUTOR_RE.search(t):
        return "tutor_overview"
    if _COMPARE_RE.search(t):
        return "compare_objects"
    return "generic"


def _detect_slots(text: str) -> SemanticSlots:
    """
    Простейший rule-based детектор смысловых слотов.
    Не завязан на конкретный документ, работает по ключевым словам.
    """
    t = (text or "").lower()

    slots = SemanticSlots()

    # актуальность
    if re.search(r"актуал(ьн|изир)", t):
        slots.relevance = True

    # --- НОРМАЛИЗАЦИЯ ДЛЯ КЕЙСА "объектом и предметом исследования" ---
    # Если в вопросе одновременно упоминаются объект+предмет+исследование — это ВСЕГДА два слота.
    has_obj_root = bool(re.search(r"\bобъект\w*\b", t))
    has_subj_root = bool(re.search(r"\bпредмет\w*\b", t))
    has_research_root = bool(re.search(r"\bисслед\w*\b", t))

    if has_obj_root and has_subj_root and has_research_root:
        slots.obj = True
        slots.subj = True

    # объект
    if not slots.obj:
        if (
            re.search(r"\bобъект\w*\s+исслед\w*\b", t)
            or re.search(r"\bобъект\w*\s+работ\w*\b", t)
            or "кто объект" in t
            or "объект работы" in t
            or "объект и предмет" in t
            or "объектом и предметом" in t
        ):
            slots.obj = True

    # предмет
    if not slots.subj:
        if (
            re.search(r"\bпредмет\w*\s+исслед\w*\b", t)
            or re.search(r"\bпредмет\w*\s+работ\w*\b", t)
            or "что является предмет" in t
            or "предмет работы" in t
            or "объект и предмет" in t
            or "объектом и предметом" in t
        ):
            slots.subj = True

    # цель
    if (
        re.search(r"\bцель\w*\s+(исслед\w*|работ\w*)\b", t)
        or "цель работы" in t
        or "как сформулирована цель" in t
        or "как сформулированы цель" in t
        or "цель и задачи" in t
    ):
        slots.goal = True

    # задачи
    if (
        re.search(r"\bзадач\w*\s+(исслед\w*|работ\w*)\b", t)
        or "какие задачи" in t
        or "как сформулированы задачи" in t
        or "цель и задачи" in t
    ):
        slots.tasks = True

    # гипотеза
    if re.search(r"\bгипотез\w*\b", t) or "какая гипотеза" in t:
        slots.hypothesis = True

    # выводы по главам / разделам
    if re.search(r"вывод(ы)?\s+по\s+(глав|раздел|параграф|част)", t) or "главные выводы" in t:
        slots.chapter_conclusions = True

    # мягкий фолбэк для репетиторского режима
    if _TUTOR_RE.search(t) and (" глава" in t or "главе" in t or _INTRO_RE.search(t)):
        if not slots.goal and not slots.tasks and not slots.chapter_conclusions:
            slots.goal = True
            slots.tasks = True
            slots.chapter_conclusions = True

    return slots

# ------------------------------------
# Извлечение ссылок на объекты документа
# ------------------------------------
def _extract_section_objects(text: str) -> List[DocumentObjectRef]:
    out: List[DocumentObjectRef] = []

    txt = text or ""
    seen_keys = set()

    # 0) "в главах 1 и 2" / "в главах 1–3"
    multi_re1 = re.compile(r"(?i)(?:\bв\s+)?главах?\s+(\d+)\s*(?:[-–]\s*(\d+)|и\s+(\d+))")

    # 0b) "в 1 и 2 главе" / "в 1–3 главе"
    multi_re2 = re.compile(r"(?i)\bв\s+(\d+)\s*(?:[-–]\s*(\d+)|и\s+(\d+))\s+глав")

    for rgx in (multi_re1, multi_re2):
        for m in rgx.finditer(txt):
            first = m.group(1)
            second = m.group(2) or m.group(3)

            nums: List[str] = []
            if first:
                nums.append(first.strip())
            if second:
                try:
                    a = int(first)
                    b = int(second)
                    if b >= a:
                        nums.extend(str(x) for x in range(a + 1, b + 1))
                    else:
                        nums.append(second.strip())
                except Exception:
                    nums.append(second.strip())

            for n in nums:
                key = n.strip()
                if not key or key in seen_keys:
                    continue
                seen_keys.add(key)
                out.append(DocumentObjectRef(kind="section", key=key, label=f"глава {key}"))

    # 1) Явные подсказки вида «глава 1», «раздел 2.3», «пункт 1.2»
    hints: List[str] = []
    try:
        hints = extract_section_hints(txt) or []  # type: ignore[arg-type]
    except Exception:
        logger.exception("document_semantic_planner: extract_section_hints failed")
        hints = []

    for h in hints:
        key = h.strip()
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        out.append(
            DocumentObjectRef(
                kind="section",
                key=key,
                label=f"раздел/глава {key}",
            )
        )

    # 2) Введение — отдельный флаг, часто без номера
    if _INTRO_RE.search(txt):
        if "intro" not in seen_keys:
            out.insert(
                0,
                DocumentObjectRef(
                    kind="section",
                    key="intro",
                    label="введение",
                ),
            )
            seen_keys.add("intro")

    return out


def _extract_table_objects(text: str) -> List[DocumentObjectRef]:
    out: List[DocumentObjectRef] = []
    try:
        nums = extract_table_numbers(text) or []  # type: ignore[arg-type]
    except Exception:
        logger.exception("document_semantic_planner: extract_table_numbers failed")
        nums = []

    seen = set()
    for n in nums:
        key = (n or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(
            DocumentObjectRef(
                kind="table",
                key=key,
                label=f"таблица {key}",
            )
        )
    return out


def _extract_figure_objects(text: str) -> List[DocumentObjectRef]:
    out: List[DocumentObjectRef] = []
    try:
        nums = extract_figure_numbers(text) or []  # type: ignore[arg-type]
    except Exception:
        logger.exception("document_semantic_planner: extract_figure_numbers failed")
        nums = []

    seen = set()
    for n in nums:
        key = (n or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(
            DocumentObjectRef(
                kind="figure",
                key=key,
                label=f"рисунок {key}",
            )
        )
    return out


def extract_document_objects(text: str) -> List[DocumentObjectRef]:
    """
    Универсальный хелпер: по тексту вопроса извлекает ссылки на разделы/главы,
    таблицы и рисунки в формате единых DocumentObjectRef.

    Порядок важен:
      - сначала введение/главы (общая структура),
      - затем таблицы,
      - затем рисунки.
    """
    sections = _extract_section_objects(text)
    tables = _extract_table_objects(text)
    figures = _extract_figure_objects(text)

    objects: List[DocumentObjectRef] = []
    objects.extend(sections)
    objects.extend(tables)
    objects.extend(figures)

    return objects


# ------------------------------------
# Публичное API модуля — план
# ------------------------------------

def build_semantic_plan(question: str) -> SemanticPlan:
    q = (question or "").strip()

    q_norm = re.sub(r"\s+", " ", q).strip()

    plan = SemanticPlan(
        mode=_detect_mode(q_norm),
        objects=extract_document_objects(q_norm),
        slots=_detect_slots(q_norm),
        original_question=q,
        normalized_question=q_norm,
    )

    # Если спрашивают учебные слоты, но разделы не распознаны — добавляем введение.
    if plan.slots.any_slot_requested() and not any(o.kind == "section" for o in plan.objects):
        plan.objects.insert(
            0,
            DocumentObjectRef(kind="section", key="intro", label="введение"),
        )

    q_lower = q.lower()

    want_ch1 = any(p in q_lower for p in ("1 глава", "первая глава", "глава 1"))
    want_ch2 = any(p in q_lower for p in ("2 глава", "вторая глава", "глава 2"))

    # Если попросили "выводы по главам" — по умолчанию глава 1 + глава 2
    if plan.slots.chapter_conclusions:
        want_ch1, want_ch2 = True, True

    def _ensure_section_object(key: str, label: str) -> None:
        if not any(o.kind == "section" and o.key == key for o in plan.objects):
            plan.objects.append(DocumentObjectRef(kind="section", key=key, label=label))

    # ✅ ВАЖНО: это ДОЛЖНО быть снаружи функции, а не внутри неё
    if want_ch1:
        _ensure_section_object("1", "глава 1")
    if want_ch2:
        _ensure_section_object("2", "глава 2")


    logger.info(
        "semantic_plan: mode=%s, objects=%d, slots=%s",
        plan.mode,
        len(plan.objects),
        {
            "relevance": plan.slots.relevance,
            "obj": plan.slots.obj,
            "subj": plan.slots.subj,
            "goal": plan.slots.goal,
            "tasks": plan.slots.tasks,
            "hypothesis": plan.slots.hypothesis,
            "chapter_conclusions": plan.slots.chapter_conclusions,
        },
    )

    return plan


# ------------------------------------
# Высокоуровневый ответчик по плану
# ------------------------------------

from .retrieval import (
    get_section_context_for_hints,
    get_table_context_for_numbers,
    get_figure_context_for_numbers,
)

try:
    # та же функция, что используется в bot.py для обычных ответов
    from .polza_client import chat_with_gpt  # type: ignore
except Exception:
    chat_with_gpt = None  # type: ignore


def _slots_to_instruction(slots: SemanticSlots) -> str:
    parts: list[str] = []

    if slots.relevance:
        parts.append("в чём актуальность темы")
    if slots.obj:
        parts.append("кто или что является объектом исследования")
    if slots.subj:
        parts.append("что является предметом исследования")
    if slots.goal:
        parts.append("как сформулирована цель исследования")
    if slots.tasks:
        parts.append("какие задачи исследования выделены")
    if slots.hypothesis:
        parts.append("какая выдвинута гипотеза")
    if slots.chapter_conclusions:
        parts.append("какие основные выводы по указанным главам/разделам")

    if not parts:
        return ""

    return "Кратко раскрой " + "; ".join(parts) + "."


def _collect_section_hints(plan: SemanticPlan) -> list[str]:
    return [o.key for o in plan.objects if o.kind == "section"]


def _collect_table_nums(plan: SemanticPlan) -> list[str]:
    return [o.key for o in plan.objects if o.kind == "table"]


def _collect_figure_nums(plan: SemanticPlan) -> list[str]:
    return [o.key for o in plan.objects if o.kind == "figure"]


async def answer_semantic_query(
    uid: int,
    doc_id: int,
    q_text: str,
    plan: SemanticPlan,
) -> Optional[str]:
    """
    Репетиторский структурный ответ:
    - GPT по зонам: intro/chapter_1/chapter_2 (по необходимости)
    - Строгое извлечение паспорта (объект/предмет/цель/задачи/гипотеза/актуальность)
    """

    has_slots = plan.slots.any_slot_requested()
    has_sections = any(o.kind == "section" for o in plan.objects)

    if not (has_sections and (plan.mode == "tutor_overview" or has_slots)):
        return None

    if chat_with_gpt is None:
        logger.warning("answer_semantic_query: chat_with_gpt is not available")
        return None

    try:
        from .retrieval import get_area_text
    except Exception:
        logger.exception("answer_semantic_query: cannot import get_area_text")
        return None

    # --- Определяем, какие зоны реально нужны ---
    want_intro = any(o.kind == "section" and o.key == "intro" for o in plan.objects) or has_slots
    want_ch1 = any(o.kind == "section" and o.key == "1" for o in plan.objects) or plan.slots.chapter_conclusions
    want_ch2 = any(o.kind == "section" and o.key == "2" for o in plan.objects) or plan.slots.chapter_conclusions

    intro_text = get_area_text(uid, doc_id, role="intro", max_chars=20000) if want_intro else ""
    ch1_text = get_area_text(uid, doc_id, role="chapter_1", max_chars=20000) if want_ch1 else ""
    ch2_text = get_area_text(uid, doc_id, role="chapter_2", max_chars=20000) if want_ch2 else ""

    if not any([intro_text, ch1_text, ch2_text]):
        logger.info("answer_semantic_query: no texts found even with fallback; fallback to legacy pipeline")
        return None

    strict_system = (
        "Ты репетитор по дипломным работам (ВКР).\n"
        "Тебе дан ТОЛЬКО текст ВКР.\n"
        "Правила:\n"
        "1) Отвечай ТОЛЬКО по данному тексту.\n"
        "2) Запрещено додумывать типовые формулировки.\n"
        "3) Если формулировка не найдена явно — пиши: «в тексте ВКР не найдено».\n"
        "4) Для полей паспорта (объект/предмет/цель/задачи/гипотеза/актуальность) "
        "ОБЯЗАТЕЛЬНО приводи короткую цитату (5–25 слов) в кавычках."
    )

    def _sanitize_passport_block(text: str) -> str:
        if not (text or "").strip():
            return ""

        fields = ["Актуальность", "Объект", "Предмет", "Цель", "Задачи", "Гипотеза"]
        out_lines: list[str] = []
        for line in (text or "").splitlines():
            s = line.strip()
            if not s:
                continue
            if s.startswith("## "):
                out_lines.append(s)
                continue

            m = re.match(r"^\*\*(.+?)\*\*:\s*(.*)$", s)
            if not m:
                out_lines.append(s)
                continue

            name = m.group(1).strip()
            val = (m.group(2) or "").strip()

            if name in fields:
                low = val.lower()
                has_not_found = "не найден" in low
                has_quote = ("«" in val and "»" in val) or ('"' in val)
                if has_not_found:
                    out_lines.append(f"**{name}:** в тексте ВКР не найдено")
                elif has_quote:
                    out_lines.append(f"**{name}:** {val}")
                else:
                    out_lines.append(f"**{name}:** в тексте ВКР не найдено")
            else:
                out_lines.append(s)

        return "\n".join(out_lines).strip()

    # --- Поля введения формируем по реально запрошенным слотам ---
    slot_to_field = [
        ("relevance", "Актуальность"),
        ("obj", "Объект"),
        ("subj", "Предмет"),
        ("goal", "Цель"),
        ("tasks", "Задачи"),
        ("hypothesis", "Гипотеза"),
    ]

    requested_fields: list[str] = []
    if plan.slots.relevance:
        requested_fields.append("Актуальность")
    if plan.slots.obj:
        requested_fields.append("Объект")
    if plan.slots.subj:
        requested_fields.append("Предмет")
    if plan.slots.goal:
        requested_fields.append("Цель")
    if plan.slots.tasks:
        requested_fields.append("Задачи")
    if plan.slots.hypothesis:
        requested_fields.append("Гипотеза")

    # Если пользователь не просил конкретные слоты (tutor_overview), то делаем полный паспорт
    if not requested_fields and plan.mode == "tutor_overview":
        requested_fields = ["Актуальность", "Объект", "Предмет", "Цель", "Задачи", "Гипотеза"]

    def _intro_format_block(fields: list[str]) -> str:
        # строгое требование формата под существующий санитайзер
        lines = ["## Введение"]
        for f in fields:
            lines.append(f"**{f}:** <формулировка> (цитата: «...»)")
        return "\n".join(lines)

    intro_answer = ""
    if intro_text and requested_fields:
        intro_user = (
            f"Вопрос студента:\n{q_text}\n\n"
            "Составь ТОЛЬКО блок 'Введение' по тексту введения.\n"
            "Заполни ТОЛЬКО перечисленные поля. "
            "Если нет явной формулировки — «в тексте ВКР не найдено».\n"
            "К каждому заполненному полю добавь короткую цитату из текста в кавычках.\n\n"
            "Формат строго такой:\n"
            + _intro_format_block(requested_fields)
            + "\n"
        )
        try:
            intro_answer = chat_with_gpt(
                [
                    {"role": "system", "content": strict_system},
                    {"role": "assistant", "content": "[Текст введения]\n" + intro_text},
                    {"role": "user", "content": intro_user},
                ],
                temperature=0.0,
                max_tokens=700,
            ).strip()
            intro_answer = _sanitize_passport_block(intro_answer)
        except Exception:
            logger.exception("answer_semantic_query: intro gpt call failed")
            intro_answer = ""

    ch_system = (
        "Ты репетитор по ВКР.\n"
        "Отвечай ТОЛЬКО по данному тексту главы.\n"
        "Запрещено добавлять внешние факты.\n"
        "Если в тексте главы нет выводов/идей — так и напиши: «в тексте ВКР не найдено»."
    )

    ch1_answer = ""
    if ch1_text:
        ch1_user = (
            f"Вопрос студента:\n{q_text}\n\n"
            "Составь ТОЛЬКО блок 'Глава 1' по тексту главы 1.\n"
            "Нужно:\n"
            "- Главные идеи (3–7 буллетов)\n"
            "- Выводы по главе (если они явно есть в тексте; иначе «в тексте ВКР не найдено»)\n\n"
            "Формат строго такой:\n"
            "## Глава 1\n"
            "**Главные идеи:**\n"
            "- ...\n"
            "**Выводы по главе:** ...\n"
        )
        try:
            ch1_answer = chat_with_gpt(
                [
                    {"role": "system", "content": ch_system},
                    {"role": "assistant", "content": "[Текст главы 1]\n" + ch1_text},
                    {"role": "user", "content": ch1_user},
                ],
                temperature=0.1,
                max_tokens=900,
            ).strip()
        except Exception:
            logger.exception("answer_semantic_query: chapter1 gpt call failed")
            ch1_answer = ""

    ch2_answer = ""
    if ch2_text:
        ch2_user = (
            f"Вопрос студента:\n{q_text}\n\n"
            "Составь ТОЛЬКО блок 'Глава 2' по тексту главы 2.\n"
            "Нужно:\n"
            "- Главные идеи (3–7 буллетов)\n"
            "- Выводы по главе (если они явно есть в тексте; иначе «в тексте ВКР не найдено»)\n\n"
            "Формат строго такой:\n"
            "## Глава 2\n"
            "**Главные идеи:**\n"
            "- ...\n"
            "**Выводы по главе:** ...\n"
        )
        try:
            ch2_answer = chat_with_gpt(
                [
                    {"role": "system", "content": ch_system},
                    {"role": "assistant", "content": "[Текст главы 2]\n" + ch2_text},
                    {"role": "user", "content": ch2_user},
                ],
                temperature=0.1,
                max_tokens=900,
            ).strip()
        except Exception:
            logger.exception("answer_semantic_query: chapter2 gpt call failed")
            ch2_answer = ""

    parts = [p for p in [intro_answer, ch1_answer, ch2_answer] if (p or "").strip()]
    if not parts:
        return None

    final = "\n\n".join(parts).strip()

    # --- Новое правило "короткий ответ допустим", если вопрос был паспортный/точечный ---
    passport_only = (
        has_slots
        and not plan.slots.chapter_conclusions
        and not want_ch1
        and not want_ch2
    )

    if passport_only:
        # достаточно, чтобы в финале были хотя бы запрошенные поля (или "не найдено")
        need_any = any(f"**{f}:" in final for f in requested_fields) if requested_fields else False
        if need_any and len(final) >= 120:
            return final
        # иначе — пусть падает в legacy/RAG
        logger.info("answer_semantic_query: passport_only too weak -> fallback legacy (len=%s)", len(final))
        return None

    # для “репетиторского” режима с главами оставляем более высокий порог
    if len(final) < 400:
        logger.info("answer_semantic_query: final too short -> fallback legacy (len=%s)", len(final))
        return None

    must_have = []
    if want_intro:
        must_have.append("## Введение")
    if want_ch1:
        must_have.append("## Глава 1")
    if want_ch2:
        must_have.append("## Глава 2")

    if must_have and any(k not in final for k in must_have):
        try:
            heal_user = (
                "Ты не дописал структуру. "
                "Дополни ТОЛЬКО отсутствующие блоки/поля. Не повторяй то, что уже есть.\n"
                "Правила: никаких домыслов, если нет формулировки в тексте — «в тексте ВКР не найдено».\n\n"
                "Вот текущий ответ:\n" + final
            )

            heal_ctx = ""
            if intro_text:
                heal_ctx += "[Текст введения]\n" + intro_text + "\n\n"
            if ch1_text:
                heal_ctx += "[Текст главы 1]\n" + ch1_text + "\n\n"
            if ch2_text:
                heal_ctx += "[Текст главы 2]\n" + ch2_text + "\n\n"

            extra = chat_with_gpt(
                [
                    {"role": "system", "content": strict_system},
                    {"role": "assistant", "content": heal_ctx.strip()[:20000]},
                    {"role": "user", "content": heal_user},
                ],
                temperature=0.0,
                max_tokens=900,
            ).strip()

            if extra:
                extra = _sanitize_passport_block(extra)
                final = final.rstrip() + "\n\n" + extra
        except Exception:
            logger.exception("answer_semantic_query: self-heal failed")

    return final
