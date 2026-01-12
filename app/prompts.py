# app/bot.py
from __future__ import annotations
from config import Cfg

from typing import Any, Dict, List, Optional, Callable

from .retrieval import (
    retrieve,
    retrieve_coverage,
    build_context,
    build_context_coverage,
    describe_figures_by_numbers,
    _extract_figure_numbers_from_query,  # используем внутренний хелпер
)
from .prompts import SYS_ANSWER, PROMPT_RULES_MD


# ---------------------- chart_matrix → текстовая таблица ----------------------


def _chart_matrix_to_text_table(chart_matrix: Dict[str, Any]) -> str:
    """
    Преобразует chart_matrix в текстовую табличку формата:

        Категория | Серия 1 | Серия 2
        A         | 10      | 20
        B         | 30      | 40

    или, если серия одна:

        Категория | Значение [unit]
        A         | 10
        B         | 30

    Это всё попадает в КОНТЕКСТ для модели (не пользователю напрямую),
    чтобы она могла надёжно ссылаться на конкретные числовые значения.
    """
    if not isinstance(chart_matrix, dict):
        return ""

    categories = chart_matrix.get("categories") or []
    series_list = chart_matrix.get("series") or []
    if not categories or not series_list:
        return ""

    categories = [str(c) for c in categories]
    multi = len(series_list) > 1

    lines: List[str] = []

    if not multi:
        s = series_list[0] or {}
        unit = (s.get("unit") or chart_matrix.get("unit") or "").strip()
        # заголовок
        head = "Категория | Значение"
        if unit:
            head += f" ({unit})"
        lines.append(head)

        vals = s.get("values") or s.get("data") or []
        for idx, cat in enumerate(categories):
            if idx < len(vals):
                v = vals[idx]
            else:
                v = ""
            lines.append(f"{cat} | {v}")
    else:
        # несколько серий: Категория | Серия 1 | Серия 2 | ...
        header_cells = ["Категория"]
        series_names: List[str] = []
        for i, s in enumerate(series_list, start=1):
            if not isinstance(s, dict):
                series_names.append(f"Серия {i}")
                continue
            name = (s.get("name") or s.get("series_name") or "").strip()
            if not name:
                name = f"Серия {i}"
            series_names.append(name)
        header_cells.extend(series_names)
        lines.append(" | ".join(header_cells))

        # строки: по категориям, в каждой ячейке значение соответствующей серии
        for idx, cat in enumerate(categories):
            row_vals: List[str] = []
            for s in series_list:
                if not isinstance(s, dict):
                    row_vals.append("")
                    continue
                vals = s.get("values") or s.get("data") or []
                if idx < len(vals):
                    row_vals.append(str(vals[idx]))
                else:
                    row_vals.append("")
            lines.append(" | ".join([cat] + row_vals))

    return "\n".join(lines).strip()


def _figures_tables_block_from_query(
    owner_id: int,
    doc_id: int,
    question: str,
    *,
    lang: str = "ru",
) -> str:
    """
    Если в вопросе явно упомянуты «Рисунок N» / «Рис. N» / «Figure N» — собираем
    дополнительный блок контекста с табличными данными диаграмм:

      — заголовок рисунка;
      — values_str (краткое перечисление значений);
      — табличное представление chart_matrix (если есть).

    Этот блок ДОБАВЛЯЕТСЯ к обычному RAG-контексту.
    """
    nums = _extract_figure_numbers_from_query(question)
    if not nums:
        return ""

    cards = describe_figures_by_numbers(
        owner_id,
        doc_id,
        nums,
        sample_chunks=2,
        use_vision=False,   # для текстового описания диаграмм достаточно табличных данных
        lang=lang,
    )
    if not cards:
        return ""

    parts: List[str] = []
    parts.append("ДОПОЛНИТЕЛЬНЫЕ ДАННЫЕ ПО УКАЗАННЫМ РИСУНКАМ (ТАБЛИЦЫ ЗНАЧЕНИЙ ДИАГРАММ):")

    for card in cards:
        display = card.get("display") or f"Рисунок {card.get('num')}"
        parts.append(f"\n{display}:")

        # values_str — аккуратное перечисление значений («— Категория: 25%» и т.п.)
        values_str = (card.get("values_str") or "").strip()
        if values_str:
            parts.append("Сводка значений диаграммы:")
            parts.append(values_str)

        # chart_matrix → markdown-подобная таблица
        chart_matrix = card.get("chart_matrix")
        table_txt = _chart_matrix_to_text_table(chart_matrix) if chart_matrix else ""
        if table_txt:
            parts.append("Табличные данные диаграммы (chart_matrix):")
            parts.append(table_txt)

    return "\n".join(p for p in parts if p.strip()).strip()


# ---------------------------- Контекст для LLM ----------------------------


def build_rag_context_with_figures(
    owner_id: int,
    doc_id: int,
    question: str,
    *,
    use_coverage: bool = True,
    per_item_k: int = 2,
    backfill_k: int = 4,
) -> Dict[str, Any]:
    """
    Строит RAG-контекст. Если ничего не найдено — context будет пустым.
    """
    question_norm = (question or "").strip()

    base_ctx = ""
    raw: Dict[str, Any] = {}

    if use_coverage:
        cov = retrieve_coverage(
            owner_id,
            doc_id,
            question_norm,
            per_item_k=per_item_k,
            backfill_k=backfill_k,
        ) or {}
        snippets = cov.get("snippets") or []
        base_ctx = build_context_coverage(
            snippets,
            items_count=len(cov.get("items") or []) or None,
        ) if snippets else ""
        raw = cov
    else:
        snips = retrieve(owner_id, doc_id, question_norm, top_k=max(10, backfill_k * 2)) or []
        base_ctx = build_context(snips) if snips else ""
        raw = {"snippets": snips}

    fig_block = _figures_tables_block_from_query(owner_id, doc_id, question_norm) or ""

    if fig_block.strip():
        full_ctx = f"{base_ctx}\n\n---\n{fig_block}" if base_ctx.strip() else fig_block
    else:
        full_ctx = base_ctx

    return {
        "context": (full_ctx or "").strip(),
        "raw": raw,
        "fig_block": fig_block,
    }


# ---------------------------- Высокоуровневый бот ----------------------------

LLMClient = Callable[[str, str, str], str]
# ожидается интерфейс вида: llm_client(system_prompt, user_question, context) -> answer


def call_llm_default(
    llm_client: LLMClient,
    system_prompt: str,
    question: str,
    context: str,
) -> str:
    """
    Тонкая обёртка над внешним LLM-клиентом.
    Здесь можно подставить ваш polza/openai клиент.
    """
    # В минимальном варианте можно просто склеить всё в один промпт.
    # Реальная интеграция зависит от вашего клиента.
    # Пример (псевдокод):
    #
    # return llm_client(
    #     system=system_prompt,
    #     user=f"Вопрос:\n{question}\n\nКонтекст:\n{context}",
    # )
    return llm_client(system_prompt, question, context)


def answer_question(
    owner_id: int,
    doc_id: int,
    question: str,
    *,
    llm_client: LLMClient,
    use_coverage: bool = True,
    lang: str = "ru",
) -> str:
    """
    Строгий режим: отвечаем ТОЛЬКО по RAG-контексту из документа.
    Если контекст не найден/пустой/слишком слабый — не вызываем LLM, возвращаем "не найдено".
    """
    rag = build_rag_context_with_figures(
        owner_id,
        doc_id,
        question,
        use_coverage=use_coverage,
    )

    context = (rag.get("context") or "").strip()

    # пороги (настройки можно хранить в Cfg, чтобы синхронизировать с bot.py)
    min_ctx_chars = int(getattr(Cfg, "MIN_GROUNDED_CTX_CHARS", 260))
    min_score = float(getattr(Cfg, "RETRIEVE_MIN_SCORE", 0.24))

    # вытащим метрики, если build_rag_context_with_figures их уже кладёт (не обязательно)
    max_score = None
    strong_hits = None
    try:
        # варианты ключей на будущее: rag["quality"] или rag["raw"]
        quality = rag.get("quality") or {}
        if isinstance(quality, dict):
            if quality.get("max_score") is not None:
                max_score = float(quality.get("max_score"))
            if quality.get("strong_hits") is not None:
                strong_hits = int(quality.get("strong_hits"))
        if max_score is None:
            raw = rag.get("raw") or {}
            if isinstance(raw, dict) and raw.get("max_score") is not None:
                max_score = float(raw.get("max_score"))
            if isinstance(raw, dict) and raw.get("strong_hits") is not None:
                strong_hits = int(raw.get("strong_hits"))
    except Exception:
        max_score = None
        strong_hits = None

    def _refusal(reason: str) -> str:
        base = (
            "В загруженном документе не найдено фрагментов, которые позволяют ответить на этот вопрос.\n"
            "Похоже, нужный текст не распарсился/не проиндексировался или сформулирован иначе.\n"
            "Уточните запрос (глава/раздел/страница) или пришлите фрагмент текста."
        )
        if bool(getattr(Cfg, "DEBUG", False)):
            return base + f"\n(debug: {reason})"
        return base

    # ✅ ГЛАВНЫЙ ГАРД: нет контекста
    if not context:
        return _refusal("empty_context")

    # ✅ ГАРД ПО ИНФОРМАТИВНОСТИ: слишком короткий контекст часто = нерелевантный обрывок
    if len(context) < min_ctx_chars:
        return _refusal(f"context_too_short<{min_ctx_chars}")

    # ✅ (опционально) ГАРД ПО КАЧЕСТВУ: если метрики есть и они слабые — не зовём LLM
    # strong_hits = 0 или max_score ниже порога → считаем, что опоры нет
    if strong_hits is not None and int(strong_hits) <= 0:
        return _refusal("no_strong_hits")
    if max_score is not None and float(max_score) < min_score:
        return _refusal(f"max_score<{min_score}")

    # Системный промпт: строгая приставка + ваши правила.
    strict_prefix = (
        "ВАЖНО: Отвечай ТОЛЬКО на основе переданного CONTEXT из документа.\n"
        "Запрещено использовать общие знания, догадки или типовые определения.\n"
        "Если в CONTEXT нет ответа — выдай отказ: «В документе не найдено. Уточните запрос/пришлите фрагмент».\n"
        "Не приписывай документу того, чего в нём нет.\n"
    )
    system_prompt = strict_prefix + SYS_ANSWER + "\n\n" + PROMPT_RULES_MD

    answer = call_llm_default(
        llm_client=llm_client,
        system_prompt=system_prompt,
        question=question,
        context=context,
    )
    return answer