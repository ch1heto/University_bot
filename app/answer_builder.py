# app/answer_builder.py
from __future__ import annotations

import re
import json
from typing import Dict, List, Any, Optional

try:
    # Основной генератор ответа (STRICT / EXPLAIN / EXPAND) — уже есть в проекте
    from .ace import ace_once
except Exception:
    # Мягкий фолбэк, если ace не прогружен (не должен срабатывать в нормальной сборке)
    def ace_once(question: str, ctx: str, pass_score: int = 85) -> str:
        return f"{question}\n\n[fallback]\n{ctx[:1000]}"

# Опциональные аналитические «подсказки» по таблицам (если модуль есть)
try:
    from .analytics import analyze_table_by_num  # type: ignore
except Exception:
    analyze_table_by_num = None  # type: ignore


# ----------------------------- НОРМАЛИЗАЦИЯ ТЕКСТА -----------------------------

_NUM_TOKEN = re.compile(r"(?<!\d)(\d{1,3}(?:[ \u00A0]\d{3})+|\d+)([.,]\d+)?\s*(%?)")

def _normalize_numbers(s: str) -> str:
    """
    Нормализация чисел:
      - 12 345 → 12345 (убираем пробелы тысяч);
      - 1,25 → 1.25 (десятичная точка);
      - '  %' → '%'.
    """
    def repl(m: re.Match) -> str:
        int_part = (m.group(1) or "").replace(" ", "").replace("\u00A0", "")
        frac = (m.group(2) or "").replace(",", ".")
        pct = (m.group(3) or "")
        return f"{int_part}{frac}{pct}"

    s = _NUM_TOKEN.sub(repl, s or "")
    # унификация длинных тире/минусов
    s = s.replace("–", "—").replace("--", "—")
    return s

def _shorten(s: str, limit: int = 300) -> str:
    s = (s or "").strip()
    return s if len(s) <= limit else (s[:limit - 1].rstrip() + "…")

def _md_list(arr: List[str], max_show: int, more: Optional[int]) -> str:
    out = []
    for x in (arr or [])[:max_show]:
        out.append(f"- {_normalize_numbers(x)}")
    if more and more > 0:
        out.append(f"… и ещё {more}")
    return "\n".join(out)


# ----------------------------- ПРАВИЛА ДЛЯ МОДЕЛИ -----------------------------

_DEFAULT_RULES = (
    "1) Ответь одним сообщением, закрой все подпункты вопроса.\n"
    "2) Заголовки таблиц: если есть номер → «Таблица N — Название»; если номера нет — только название.\n"
    "3) Не выводи служебные метки и размеры (никаких [Таблица], «ряд 1», «(6×7)»).\n"
    "4) В списках покажи не более 25 строк, затем «… и ещё M», если есть.\n"
    "5) Не придумывай факты вне блока Facts; если данных нет — скажи честно.\n"
)


# ----------------------------- СБОРКА БЛОКА ФАКТОВ -----------------------------

def _cards_for_tables(
    table_describe: List[Dict[str, Any]],
    *,
    owner_id: Optional[int] = None,
    doc_id: Optional[int] = None,
    lang: str = "ru",
    insights_top_k: int = 3,
) -> List[Dict[str, Any]]:
    """
    Приводим карточки таблиц к компактному формату; при наличии analytics и идентификаторов
    документа аккуратно добавляем вычисленные инсайты (короткий текст + небольшие топы),
    не нарушая принципа «только из фактов».
    """
    cards: List[Dict[str, Any]] = []

    for c in (table_describe or []):
        card = {
            "num": c.get("num"),
            "display": _normalize_numbers(c.get("display") or ""),
            "where": c.get("where") or {},
            "highlights": [_normalize_numbers(h) for h in (c.get("highlights") or [])][:2],
        }

        # Мягкая интеграция аналитики — только если есть owner_id/doc_id и корректная функция.
        num = (c.get("num") or "").strip() if isinstance(c.get("num"), str) else None
        if analyze_table_by_num and owner_id is not None and doc_id is not None and num:
            try:
                res = analyze_table_by_num(owner_id, doc_id, num, top_k=max(1, insights_top_k), lang=lang)  # type: ignore
                if isinstance(res, dict) and res.get("ok"):
                    insight: Dict[str, Any] = {}

                    # Короткий текст отчёта (ужимаем)
                    text = res.get("text") or ""
                    if text:
                        insight["text"] = _normalize_numbers(_shorten(str(text), 420))

                    # Топы по числовым колонкам (до 2 колонок, каждый топ — до insights_top_k строк)
                    tops: List[Dict[str, Any]] = []
                    for col in (res.get("numeric_summary") or [])[:2]:
                        col_name = str(col.get("col") or "").strip() or "Колонка"
                        col_top = col.get("top") or []
                        small = []
                        for item in col_top[:insights_top_k]:
                            row_lbl = _normalize_numbers(str(item.get("row") or ""))
                            val = item.get("value")
                            # Представим число аккуратно (как есть)
                            try:
                                small.append({"row": row_lbl, "value": float(val) if val is not None else None})
                            except Exception:
                                small.append({"row": row_lbl, "value": val})
                        if small:
                            tops.append({"col": _normalize_numbers(col_name), "top": small})
                    if tops:
                        insight["tops"] = tops

                    if insight:
                        card["insights"] = insight
            except Exception:
                # Мягкая деградация — без инсайтов
                pass

        cards.append(card)
    return cards


def facts_to_prompt(
    facts: Dict[str, Any],
    *,
    rules: Optional[str] = None,
    owner_id: Optional[int] = None,
    doc_id: Optional[int] = None,
    lang: str = "ru",
) -> str:
    """
    Превращает dict `facts` (как собирается в bot._gather_facts) в устойчивый markdown-блок
    для модели: [Facts] ... [Rules] ...
    """
    parts: List[str] = []

    # ----- Таблицы -----
    tables = (facts or {}).get("tables") or {}
    if tables:
        block: List[str] = []
        if "count" in tables:
            block.append(f"count: {int(tables.get('count') or 0)}")
        if tables.get("list"):
            block.append("list:\n" + _md_list(tables["list"], 25, tables.get("more", 0)))
        if tables.get("describe"):
            cards = _cards_for_tables(
                tables.get("describe") or [],
                owner_id=owner_id,
                doc_id=doc_id,
                lang=lang,
            )
            # JSON-представление карточек — компактное, но человекочитаемое
            block.append("describe:\n" + json.dumps(cards, ensure_ascii=False, indent=2))
        parts.append("- Tables:\n  " + "\n  ".join(block))

    # ----- Рисунки -----
    figures = (facts or {}).get("figures") or {}
    if figures:
        block = [f"count: {int(figures.get('count') or 0)}"]
        if figures.get("list"):
            block.append("list:\n" + _md_list(figures["list"], 25, figures.get("more", 0)))
        desc_lines = [str(x).strip() for x in (figures.get("describe_lines") or []) if str(x).strip()]
        if desc_lines:
            block.append("describe:\n" + "\n".join([f"- {_normalize_numbers(x)}" for x in desc_lines[:25]]))
        parts.append("- Figures:\n  " + "\n  ".join(block))

    # ----- Источники -----
    sources = (facts or {}).get("sources") or {}
    if sources:
        block = [f"count: {int(sources.get('count') or 0)}"]
        if sources.get("list"):
            block.append("list:\n" + _md_list(sources["list"], 25, sources.get("more", 0)))
        parts.append("- Sources:\n  " + "\n  ".join(block))

    # ----- Практическая часть -----
    if "practical_present" in (facts or {}):
        parts.append(f"- PracticalPartPresent: {bool(facts.get('practical_present'))}")

    # ----- Краткое содержание -----
    if (facts or {}).get("summary_text"):
        st = str(facts["summary_text"])
        parts.append("- Summary:\n  " + _normalize_numbers(_shorten(st, 1200)).replace("\n", "\n  "))

    # ----- Вербатим-цитаты (шинглы) -----
    if (facts or {}).get("verbatim_hits"):
        hits_md = []
        for h in facts["verbatim_hits"]:
            page = h.get("page")
            sec = (h.get("section_path") or "").strip()
            page_str = (str(page) if page is not None else "?")
            where = f'в разделе «{sec}», стр. {page_str}' if sec else f'на стр. {page_str}'
            snippet = _normalize_numbers(h.get("snippet") or "")
            hits_md.append(f"- Match {where}: «{snippet}»")
        parts.append("- Citations:\n  " + "\n  ".join(hits_md))

    # ----- Общий контекст -----
    if (facts or {}).get("general_ctx"):
        ctx = str(facts.get("general_ctx") or "")
        ctx = _normalize_numbers(_shorten(ctx, 1500))
        parts.append("- Context:\n  " + ctx.replace("\n", "\n  "))

    facts_md = "[Facts]\n" + "\n".join(parts) + "\n\n[Rules]\n" + (rules or _DEFAULT_RULES)
    return facts_md


# ----------------------------- ПУБЛИЧНОЕ API -----------------------------

def generate_answer(
    question: str,
    facts: Dict[str, Any],
    *,

    language: str = "ru",
    pass_score: int = 85,
    rules_override: Optional[str] = None,
) -> str:
    """
    Универсальный билдер финального ответа:
      1) нормализует числа/символы в тексте фактов;
      2) собирает устойчивый блок промпта (Facts + Rules);
      3) вызывает строгий агент ace_once (внутри поддерживает STRICT/EXPLAIN/EXPAND);
      4) возвращает финальный текст.

    Совместим с форматом facts из bot._gather_facts.
    """
    q = (question or "").strip()
    if not q:
        return "Вопрос пустой. Сформулируйте, пожалуйста, что именно требуется разобрать по ВКР."

    # Пытаемся аккуратно достать owner_id/doc_id из facts (несколько вариантов ключей)
    owner_id = (
        facts.get("owner_id")
        or facts.get("_owner_id")
        or (facts.get("meta") or {}).get("owner_id")
    )
    doc_id = (
        facts.get("doc_id")
        or facts.get("_doc_id")
        or (facts.get("meta") or {}).get("doc_id")
    )
    try:
        owner_id = int(owner_id) if owner_id is not None else None
    except Exception:
        owner_id = None
    try:
        doc_id = int(doc_id) if doc_id is not None else None
    except Exception:
        doc_id = None

    ctx = facts_to_prompt(
        facts,
        rules=rules_override,
        owner_id=owner_id,
        doc_id=doc_id,
        lang=language,
    )
    try:
        reply = ace_once(q, ctx, pass_score=pass_score)
        return (reply or "").strip()
    except Exception:
        # Мягкая деградация: вернём «сшитый» фолбэк без ACE
        fallback = [
            "Не удалось сгенерировать ответ строгим агентом. Ниже — краткий конспект найденных фактов.",
            "",
            ctx[:4000]
        ]
        return "\n".join(fallback)


def debug_digest(facts: Dict[str, Any]) -> str:
    """
    Короткий диагностический дайджест по фактам — удобно логировать/показывать в /diag.
    """
    parts = []
    tbl = (facts or {}).get("tables") or {}
    fig = (facts or {}).get("figures") or {}
    src = (facts or {}).get("sources") or {}

    parts.append(f"Таблиц: {int(tbl.get('count') or 0)} (показано: {len(tbl.get('list') or [])})")
    parts.append(f"Рисунков: {int(fig.get('count') or 0)} (показано: {len(fig.get('list') or [])})")
    parts.append(f"Источников: {int(src.get('count') or 0)} (показано: {len(src.get('list') or [])})")

    if facts.get("practical_present") is not None:
        parts.append(f"Практическая часть: {'да' if facts.get('practical_present') else 'нет'}")

    if facts.get("summary_text"):
        parts.append("Есть краткое содержание (summary_text).")

    if facts.get("general_ctx"):
        parts.append("Есть общий контекст (general_ctx).")

    if facts.get("verbatim_hits"):
        parts.append(f"Есть точные совпадения (verbatim_hits): {len(facts.get('verbatim_hits') or [])}")

    return "\n".join(parts)
