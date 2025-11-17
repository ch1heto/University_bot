# app/bot.py
from __future__ import annotations

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
    Строит RAG-контекст для вопроса пользователя, а также доклеивает
    дополнительные табличные данные по рисункам (если пользователь
    явно просит «рисунок N»).

    Возвращает:
      {
        "context": <строка для system/user-контекста>,
        "raw": <полная структура из retrieval (snippets/coverage)>,
        "fig_block": <дополнительный блок по рисункам или "" >
      }
    """
    question_norm = (question or "").strip()

    if use_coverage:
        cov = retrieve_coverage(
            owner_id,
            doc_id,
            question_norm,
            per_item_k=per_item_k,
            backfill_k=backfill_k,
        )
        snippets = cov.get("snippets") or []
        base_ctx = build_context_coverage(
            snippets,
            items_count=len(cov.get("items") or []) or None,
        )
        raw = cov
    else:
        snips = retrieve(owner_id, doc_id, question_norm, top_k=max(10, backfill_k * 2))
        base_ctx = build_context(snips)
        raw = {"snippets": snips}

    fig_block = _figures_tables_block_from_query(owner_id, doc_id, question_norm)

    if fig_block:
        full_ctx = f"{base_ctx}\n\n---\n{fig_block}" if base_ctx.strip() else fig_block
    else:
        full_ctx = base_ctx

    return {
        "context": full_ctx.strip(),
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
    Главная точка входа для «бота»:

    1) Строит RAG-контекст (coverage/обычный).
    2) Подмешивает табличные данные диаграмм (chart_matrix/values_str) для запросов
       вида «опиши рисунок N», «что показывает рис. 2.3» и т.п.
    3) Вызывает LLM с системной подсказкой + правилами.

    Возвращает текст ответа модели.
    """
    rag = build_rag_context_with_figures(
        owner_id,
        doc_id,
        question,
        use_coverage=use_coverage,
    )

    context = rag["context"]

    # Системный промпт: общие правила + работа с таблицами/диаграммами.
    system_prompt = SYS_ANSWER + "\n\n" + PROMPT_RULES_MD

    # Здесь вызываем конкретный LLM-клиент
    answer = call_llm_default(
        llm_client=llm_client,
        system_prompt=system_prompt,
        question=question,
        context=context,
    )
    return answer
