# app/document_calc_agent.py
# -*- coding: utf-8 -*-
"""
Document Calculation Agent (offline checks + optional LLM narration)

Задача модуля:
- По запросам вида "Реалистичны ли расчёты?", "Проверь расчёты/формулы/итоги/проценты"
  извлечь из БД документа потенциально "расчётные" фрагменты (таблицы и абзацы),
  выполнить базовые детерминированные проверки (суммы/итого по таблицам, простые арифм. равенства),
  и (опционально) отдать LLM короткий, строгий отчёт по найденным несостыковкам/проверкам.

Интеграция:
- В bot.py (respond_with_answer) можно вызывать:
    from app.document_calc_agent import answer_calc_question
    agent_ans = await answer_calc_question(uid, doc_id, q_text, chat_with_gpt=chat_with_gpt)
    if agent_ans:
        await _send(m, agent_ans); return
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

# chat_with_gpt ожидается совместимым с вашим (messages, temperature=?, max_tokens=?)->str
ChatFn = Callable[..., str]


# -----------------------------
# Intent detection
# -----------------------------

_CALC_TRIGGERS = (
    "реалистичн", "расчёт", "расчет", "провер", "корректн", "правильн", "ошибк",
    "формул", "итог", "сумм", "процент", "доля", "баланс", "увяз", "сходим",
    "арифмет", "пересчит", "проверь",
)


def is_calc_question(q: str) -> bool:
    """Сильно похоже на расчётный запрос."""
    t = (q or "").strip().lower()
    if not t:
        return False
    return any(k in t for k in _CALC_TRIGGERS)


# -----------------------------
# DB access helpers
# -----------------------------

def _get_conn():
    # В вашем проекте get_conn лежит в app.db
    from app.db import get_conn  # type: ignore
    return get_conn()


def _chunks_table_columns(cur) -> set:
    try:
        cur.execute("PRAGMA table_info(chunks)")
        return {row[1] for row in cur.fetchall()}
    except Exception:
        return set()


def _sections_table_columns(cur) -> set:
    try:
        cur.execute("PRAGMA table_info(document_sections)")
        return {row[1] for row in cur.fetchall()}
    except Exception:
        return set()


def _safe_json_loads(s: Any) -> Dict[str, Any]:
    if not s or not isinstance(s, str):
        return {}
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}

def _cfg_calc_agent_use_llm() -> bool:
    """
    Если bot.py всегда прокидывает chat_with_gpt, то это позволит
    отключать LLM-ветку без правок bot.py.
    По умолчанию: False.
    """
    try:
        from app.config import Cfg  # type: ignore
        return bool(getattr(Cfg, "CALC_AGENT_USE_LLM", False))
    except Exception:
        return False


def _wants_llm_narration(question: str) -> bool:
    """
    Явный запрос на “красивый” нарратив через LLM.
    В остальных случаях лучше детерминированный отчёт.
    """
    q = (question or "").lower()
    triggers = ("сформулируй красиво", "красивый ответ", "сделай вывод", "написать вывод", "объясни", "объяснение")
    return any(t in q for t in triggers)


def _ref_for_user(c: "CheckResult", show_chunk_id: bool = False) -> str:
    parts = []
    if c.section_path:
        parts.append(c.section_path)
    if show_chunk_id and c.chunk_id is not None:
        parts.append(f"chunk_id={c.chunk_id}")
    return f" (источник: {', '.join(parts)})" if parts else ""


def _format_checks_pretty(checks: Sequence["CheckResult"], show_chunk_id: bool = False) -> str:
    """
    Человекочитаемый детерминированный отчёт.
    Без мусора, без LLM, но с понятным выводом и конкретикой.
    """
    bad = [x for x in checks if not x.ok]
    good = [x for x in checks if x.ok]

    lines: List[str] = []

    if bad:
        lines.append("Вывод: обнаружены арифметические несоответствия. Рекомендую перепроверить указанные формулы/ячейки.")
        lines.append("")
        lines.append("Что не сходится:")
        for x in bad[:12]:
            lines.append(f"- {x.message}{_ref_for_user(x, show_chunk_id=show_chunk_id)}")
        if good:
            lines.append("")
            lines.append("Что удалось подтвердить пересчётом:")
            for x in good[:8]:
                lines.append(f"- {x.message}{_ref_for_user(x, show_chunk_id=show_chunk_id)}")
        return "\n".join(lines).strip()

    if good:
        lines.append("Вывод: явных арифметических ошибок по найденным фрагментам не обнаружено (в пределах допуска округления).")
        lines.append("")
        lines.append("Что удалось подтвердить пересчётом:")
        for x in good[:12]:
            lines.append(f"- {x.message}{_ref_for_user(x, show_chunk_id=show_chunk_id)}")
        return "\n".join(lines).strip()

    return ""

# -----------------------------
# Numeric parsing
# -----------------------------

_NUM_TOKEN_RE = re.compile(
    r"""
    (?P<sign>[-−])?
    (?P<num>
        (?:
            \d{1,3}(?:[ \u00A0]\d{3})+   # 1 234 567
            |
            \d+(?:[.,]\d+)?              # 1234 or 12,34
        )
    )
    \s*
    (?P<unit>тыс\.?|тысяч|млн\.?|миллион(?:а|ов)?|млрд\.?|миллиард(?:а|ов)?)?
    """,
    re.VERBOSE | re.IGNORECASE,
)

_PERCENT_RE = re.compile(r"(?P<p>\d+(?:[.,]\d+)?)\s*%")

# Простейший паттерн: числа + операторы + число
_FORMULA_LINE_RE = re.compile(r"(?i)(\d[0-9 \u00A0.,]*)\s*([+\-−*/×x])\s*(\d[0-9 \u00A0.,]*)")
# A = B (просто равенство)
_EQUALITY_RE = re.compile(r"(?i)(\d[0-9 \u00A0.,]*)\s*=\s*(\d[0-9 \u00A0.,]*)")
# NEW: полноценное равенство с оператором: A - B = C
_FORMULA_EQUALITY_RE = re.compile(
    r"(?i)"
    r"(\d[0-9 \u00A0.,]*)\s*([+\-−*/×x])\s*(\d[0-9 \u00A0.,]*)\s*=\s*(\d[0-9 \u00A0.,]*)"
)


def _norm_num_str(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("\u00A0", " ")
    s = re.sub(r"[ ]+", " ", s)
    s = s.replace("−", "-")
    return s


def _parse_number(token: str, unit: Optional[str] = None, sign: Optional[str] = None) -> Optional[float]:
    if not token:
        return None

    t = _norm_num_str(token)
    t = t.replace(" ", "")

    # decimal comma -> dot
    if t.count(",") == 1 and t.count(".") == 0:
        t = t.replace(",", ".")
    # many commas -> thousand separators
    if t.count(",") > 1 and t.count(".") == 0:
        t = t.replace(",", "")
    # many dots -> thousand separators
    if t.count(".") > 1 and t.count(",") == 0:
        t = t.replace(".", "")

    try:
        val = float(t)
    except Exception:
        return None

    if sign and sign.strip() in ("-", "−"):
        val = -val

    u = (unit or "").lower().strip()
    mult = 1.0
    if u:
        if u.startswith("тыс") or "тысяч" in u:
            mult = 1_000.0
        elif u.startswith("млн") or "миллион" in u:
            mult = 1_000_000.0
        elif u.startswith("млрд") or "миллиард" in u:
            mult = 1_000_000_000.0

    return val * mult


def _approx_equal(a: float, b: float, rel: float = 1e-3, abs_tol: float = 1e-6) -> bool:
    # относительная погрешность: 0.1% по умолчанию
    return abs(a - b) <= max(abs_tol, rel * max(1.0, abs(a), abs(b)))


# -----------------------------
# Evidence model
# -----------------------------

@dataclass
class EvidenceChunk:
    chunk_id: int
    element_type: str
    section_path: str
    text: str
    attrs: Dict[str, Any]


@dataclass
class CheckResult:
    kind: str  # "table_total" | "formula" | "percent" | ...
    ok: bool
    message: str
    chunk_id: Optional[int] = None
    section_path: Optional[str] = None


# -----------------------------
# Retrieval of calc-related chunks from DB
# -----------------------------

def _select_calc_candidate_chunks(owner_id: int, doc_id: int, limit: int = 140) -> List[EvidenceChunk]:
    """
    Извлекаем "подозрительные" чанки из таблицы chunks:
    - таблицы (element_type='table') + attrs.is_table
    - абзацы с числами, процентами или знаками арифметики/равенства
    """
    con = _get_conn()
    cur = con.cursor()
    cols = _chunks_table_columns(cur)

    # Required minimal columns in chunks: id, text
    if "id" not in cols or "text" not in cols:
        con.close()
        return []

    has_element_type = "element_type" in cols
    has_attrs = "attrs" in cols
    has_section_path = "section_path" in cols

    patterns_like = [
        "%=%", "%+%", "%-%", "%×%", "%x%", "%*%", "%/%",
        "%итог%", "%итого%", "%сумм%", "%баланс%", "%процент%", "%доля%",
        "%руб%", "%тыс%", "%млн%", "%млрд%",
        "%выруч%", "%доход%", "%расход%", "%затрат%", "%прибыл%", "%маржин%", "%себестоим%",
        "%0%", "%1%", "%2%", "%3%", "%4%", "%5%", "%6%", "%7%", "%8%", "%9%",
    ]

    where_parts = ["owner_id=? AND doc_id=?"]
    params: List[Any] = [owner_id, doc_id]

    if has_element_type and has_attrs:
        where_parts.append(
            "("
            "element_type IN ('table','paragraph','text') "
            "OR attrs LIKE '%\"is_table\"%' "
            ")"
        )
    elif has_element_type:
        where_parts.append("element_type IN ('table','paragraph','text')")

    like_sql = " OR ".join(["text LIKE ?"] * len(patterns_like))
    where_parts.append(f"({like_sql})")
    params.extend(patterns_like)

    select_cols = ["id", "text"]
    if has_element_type:
        select_cols.append("element_type")
    if has_section_path:
        select_cols.append("section_path")
    if has_attrs:
        select_cols.append("attrs")

    sql = f"""
        SELECT {", ".join(select_cols)}
        FROM chunks
        WHERE {" AND ".join(where_parts)}
        ORDER BY id ASC
        LIMIT ?
    """
    params.append(int(limit))

    cur.execute(sql, tuple(params))
    rows = list(cur.fetchall())

    # NEW: добор по '=' (часто формулы без ключевых слов)
    if len(rows) < int(limit):
        try:
            sql2 = f"""
                SELECT {", ".join(select_cols)}
                FROM chunks
                WHERE owner_id=? AND doc_id=? AND text LIKE '%=%'
                ORDER BY id ASC
                LIMIT ?
            """
            cur.execute(sql2, (owner_id, doc_id, int(limit)))
            rows2 = cur.fetchall()

            ids = set()
            for r in rows:
                try:
                    ids.add(int(r["id"]))
                except Exception:
                    ids.add(int(r[0]))

            for r in rows2:
                try:
                    rid = int(r["id"])
                except Exception:
                    rid = int(r[0])
                if rid not in ids:
                    rows.append(r)
                    ids.add(rid)
        except Exception:
            pass

    con.close()

    out: List[EvidenceChunk] = []
    for r in rows:
        try:
            cid = int(r["id"])
        except Exception:
            cid = int(r[0])

        try:
            txt = r["text"]
        except Exception:
            txt = r[1] if len(r) > 1 else ""

        txt = (txt or "").strip()
        if not txt:
            continue

        et = ""
        sp = ""
        attrs_s = None

        if has_element_type:
            try:
                et = (r["element_type"] or "").strip()
            except Exception:
                pass
        if has_section_path:
            try:
                sp = (r["section_path"] or "").strip()
            except Exception:
                pass
        if has_attrs:
            try:
                attrs_s = r["attrs"]
            except Exception:
                attrs_s = None

        attrs = _safe_json_loads(attrs_s)

        low = txt.lower()
        if not (
            _PERCENT_RE.search(txt)
            or _FORMULA_EQUALITY_RE.search(txt)
            or _FORMULA_LINE_RE.search(txt)
            or _EQUALITY_RE.search(txt)
            or ("итог" in low or "сумм" in low or "баланс" in low)
            or any(k in low for k in ("выруч", "доход", "расход", "затрат", "прибыл"))
            or any(ch.isdigit() for ch in txt)
        ):
            continue

        if not et:
            et = "table" if attrs.get("is_table") else "paragraph"

        out.append(EvidenceChunk(chunk_id=cid, element_type=et, section_path=sp, text=txt, attrs=attrs))

    return out


def _select_calc_candidate_sections(owner_id: int, doc_id: int, limit: int = 220) -> List[EvidenceChunk]:
    """
    Fallback: если чанков мало/нет — берём document_sections.
    """
    con = _get_conn()
    cur = con.cursor()
    cols = _sections_table_columns(cur)

    if "id" not in cols or "text" not in cols:
        con.close()
        return []

    has_element_type = "element_type" in cols
    has_section_path = "section_path" in cols
    has_attrs = "attrs" in cols

    select_cols = ["id", "text"]
    if has_element_type:
        select_cols.append("element_type")
    if has_section_path:
        select_cols.append("section_path")
    if has_attrs:
        select_cols.append("attrs")

    where_parts = []
    params: List[Any] = []

    if "owner_id" in cols:
        where_parts.append("owner_id=?")
        params.append(owner_id)
    if "doc_id" in cols:
        where_parts.append("doc_id=?")
        params.append(doc_id)

    if not where_parts:
        con.close()
        return []

    sql = f"""
        SELECT {", ".join(select_cols)}
        FROM document_sections
        WHERE {" AND ".join(where_parts)}
        ORDER BY id ASC
        LIMIT ?
    """
    params.append(int(limit))

    cur.execute(sql, tuple(params))
    rows = cur.fetchall()
    con.close()

    out: List[EvidenceChunk] = []
    for r in rows:
        try:
            sid = int(r["id"])
        except Exception:
            sid = int(r[0])

        try:
            txt = r["text"]
        except Exception:
            txt = r[1] if len(r) > 1 else ""

        txt = (txt or "").strip()
        if not txt:
            continue

        et = ""
        sp = ""
        attrs_s = None

        if has_element_type:
            try:
                et = (r["element_type"] or "").strip()
            except Exception:
                pass
        if has_section_path:
            try:
                sp = (r["section_path"] or "").strip()
            except Exception:
                pass
        if has_attrs:
            try:
                attrs_s = r["attrs"]
            except Exception:
                attrs_s = None

        attrs = _safe_json_loads(attrs_s)

        low = txt.lower()
        if not (
            _PERCENT_RE.search(txt)
            or _FORMULA_EQUALITY_RE.search(txt)
            or _FORMULA_LINE_RE.search(txt)
            or _EQUALITY_RE.search(txt)
            or any(k in low for k in ("итог", "итого", "сумм", "баланс", "выруч", "доход", "расход", "затрат", "прибыл"))
            or any(ch.isdigit() for ch in txt)
        ):
            continue

        if not et:
            et = "table" if attrs.get("is_table") else "paragraph"

        out.append(EvidenceChunk(chunk_id=sid, element_type=et, section_path=sp, text=txt, attrs=attrs))

    return out


# -----------------------------
# Checks
# -----------------------------

def _check_simple_formulas(chunks: Sequence[EvidenceChunk], max_checks: int = 30) -> List[CheckResult]:
    res: List[CheckResult] = []

    def _emit(ok: bool, msg: str, c: EvidenceChunk):
        res.append(CheckResult(
            kind="formula",
            ok=ok,
            message=msg,
            chunk_id=c.chunk_id,
            section_path=c.section_path or None,
        ))

    for c in chunks:
        if len(res) >= max_checks:
            break

        text = c.text or ""

        # 0) A op B = C
        for m in _FORMULA_EQUALITY_RE.finditer(text):
            a = _parse_number(m.group(1), None, None)
            op = (m.group(2) or "").strip()
            b = _parse_number(m.group(3), None, None)
            given = _parse_number(m.group(4), None, None)
            if a is None or b is None or given is None:
                continue

            calc: Optional[float] = None
            if op == "+":
                calc = a + b
            elif op in ("-", "−"):
                calc = a - b
            elif op in ("*", "×", "x", "X"):
                calc = a * b
            elif op == "/":
                if abs(b) > 1e-12:
                    calc = a / b

            if calc is None:
                continue

            snippet = (m.group(0) or "").strip()
            ok = _approx_equal(calc, given, rel=2e-3)

            if ok:
                _emit(True, f"Проверка выражения корректна: {snippet}", c)
            else:
                _emit(False, f"Возможная ошибка: {snippet}. Пересчёт даёт {calc:g} вместо {given:g}.", c)

            if len(res) >= max_checks:
                break

        if len(res) >= max_checks:
            break

        # 1) простое равенство A = B
        for m in _EQUALITY_RE.finditer(text):
            left = _parse_number(m.group(1), unit=None, sign=None)
            right = _parse_number(m.group(2), unit=None, sign=None)
            if left is None or right is None:
                continue

            ok = _approx_equal(left, right, rel=1e-3)
            snippet = (m.group(0) or "").strip()
            if ok:
                _emit(True, f"Проверка равенства выглядит корректно: {snippet}", c)
            else:
                _emit(False, f"Возможная ошибка в равенстве: {snippet} (лево={left:g}, право={right:g})", c)

            if len(res) >= max_checks:
                break

        if len(res) >= max_checks:
            break

        # 2) выражение без итога — добавляем только если ещё нет ошибок (уменьшаем шум)
        has_bad = any((not r.ok) for r in res)
        if not has_bad:
            for m in _FORMULA_LINE_RE.finditer(text):
                expr = f"{(m.group(1) or '').strip()} {(m.group(2) or '').strip()} {(m.group(3) or '').strip()}"
                _emit(True, f"Найдена формула/выражение (без явного итога для сверки): {expr}", c)
                if len(res) >= max_checks:
                    break

    return res


def _check_revenue_cost_profit(chunks: Sequence[EvidenceChunk], max_checks: int = 20) -> List[CheckResult]:
    res: List[CheckResult] = []

    def _pick_near(text: str, pat: str) -> Optional[float]:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if not m:
            return None
        tail = text[m.end(): m.end() + 160]
        mnum = _NUM_TOKEN_RE.search(tail)
        if not mnum:
            return None
        return _parse_number(mnum.group("num"), unit=mnum.group("unit"), sign=mnum.group("sign"))

    ordered = sorted(list(chunks), key=lambda x: x.chunk_id)

    # окно из 3 чанков: текущий + 2 следующих
    for i in range(len(ordered)):
        if len(res) >= max_checks:
            break

        window = ordered[i:i + 3]
        text = "\n".join((c.text or "") for c in window if (c.text or "").strip())
        low = text.lower()

        if not any(k in low for k in ("выруч", "доход", "расход", "затрат", "прибыл")):
            continue

        rev = _pick_near(text, r"\b(выручк[аеиу]|доход[ыа]?)\b")
        cost = _pick_near(text, r"\b(расход[ыа]?|затрат[ыа]?|издержк[иа])\b")
        prof = _pick_near(text, r"\b(прибыл[ьи])\b")

        if rev is None or cost is None or prof is None:
            continue

        calc = rev - cost
        ok = _approx_equal(calc, prof, rel=2e-3)
        anchor = window[0]

        if ok:
            res.append(CheckResult(
                kind="formula",
                ok=True,
                message=f"Выручка–расходы сходятся с прибылью: {rev:g} - {cost:g} ≈ {prof:g}.",
                chunk_id=anchor.chunk_id,
                section_path=anchor.section_path or None,
            ))
        else:
            res.append(CheckResult(
                kind="formula",
                ok=False,
                message=f"Возможная ошибка: выручка–расходы≠прибыль. {rev:g} - {cost:g} = {calc:g}, но указано {prof:g}.",
                chunk_id=anchor.chunk_id,
                section_path=anchor.section_path or None,
            ))

    return res


def _parse_table_grid_from_attrs(attrs: Dict[str, Any]) -> Optional[List[List[str]]]:
    grid = attrs.get("table_grid")
    if isinstance(grid, list) and all(isinstance(r, list) for r in grid):
        out: List[List[str]] = []
        for r in grid:
            out.append([str(x) if x is not None else "" for x in r])
        return out

    tsv = attrs.get("table_tsv")
    if isinstance(tsv, str) and tsv.strip():
        rows = [ln.split("\t") for ln in tsv.splitlines()]
        return [[(c or "").strip() for c in r] for r in rows if any((c or "").strip() for c in r)]

    return None


def _is_total_row(row: List[str]) -> bool:
    joined = " ".join((c or "") for c in row).lower()
    return any(k in joined for k in ("итого", "итог", "всего", "сумма", "итогов"))


def _cell_to_number(cell: str) -> Optional[float]:
    if not cell:
        return None
    if "%" in cell:
        return None
    m = _NUM_TOKEN_RE.search(cell)
    if not m:
        return None
    return _parse_number(m.group("num"), unit=m.group("unit"), sign=m.group("sign"))


def _check_table_totals(chunks: Sequence[EvidenceChunk], max_tables: int = 10) -> List[CheckResult]:
    """
    Проверяем "итого" по числовым столбцам внутри каждой таблицы.
    Работает только если в attrs лежит table_grid/table_tsv.
    """
    res: List[CheckResult] = []
    tables = [c for c in chunks if (c.element_type or "").lower() == "table" or c.attrs.get("is_table")]
    tables = tables[:max_tables]

    for c in tables:
        grid = _parse_table_grid_from_attrs(c.attrs) or None
        if not grid or len(grid) < 3:
            continue

        total_rows_idx = [i for i, r in enumerate(grid) if _is_total_row(r)]
        if not total_rows_idx:
            continue

        ti = total_rows_idx[-1]
        total_row = grid[ti]
        data_rows = grid[:ti]
        if len(data_rows) < 2:
            continue

        n_cols = max(len(r) for r in grid)
        data_rows = [r + [""] * (n_cols - len(r)) for r in data_rows]
        total_row = total_row + [""] * (n_cols - len(total_row))

        numeric_cols: List[int] = []
        for j in range(n_cols):
            vals = [_cell_to_number(r[j]) for r in data_rows]
            cnt = sum(v is not None for v in vals)
            if cnt >= max(2, int(0.5 * len(data_rows))):
                numeric_cols.append(j)

        if not numeric_cols:
            continue

        checked_any = False
        for j in numeric_cols:
            total_val = _cell_to_number(total_row[j])
            if total_val is None:
                continue

            vals = [_cell_to_number(r[j]) for r in data_rows]
            vals_f = [v for v in vals if v is not None]
            if len(vals_f) < 2:
                continue

            s = float(sum(vals_f))
            ok = _approx_equal(s, total_val, rel=3e-3)
            checked_any = True

            col_name = ""
            if len(grid) >= 1:
                col_name = (grid[0][j] if j < len(grid[0]) else "") or ""
                col_name = col_name.strip()

            hint = f" (столбец: {col_name})" if col_name else f" (столбец #{j+1})"
            if ok:
                res.append(CheckResult(
                    kind="table_total",
                    ok=True,
                    message=f"Таблица: итого сходится{hint}: сумма={s:g}, итого={total_val:g}.",
                    chunk_id=c.chunk_id,
                    section_path=c.section_path or None,
                ))
            else:
                res.append(CheckResult(
                    kind="table_total",
                    ok=False,
                    message=f"Таблица: возможная ошибка итога{hint}: сумма={s:g}, а в 'итого'={total_val:g}.",
                    chunk_id=c.chunk_id,
                    section_path=c.section_path or None,
                ))

            if len(res) >= 30:
                break

        if checked_any and len(res) >= 30:
            break

    return res


def _check_percent_sanity(chunks: Sequence[EvidenceChunk], max_checks: int = 20) -> List[CheckResult]:
    """
    Мягкая проверка процентов:
    - если в одном фрагменте несколько процентов, проверим, не превышает ли сумма сильно 100% (для распределений).
    """
    res: List[CheckResult] = []
    for c in chunks:
        if len(res) >= max_checks:
            break

        ps: List[float] = []
        for m in _PERCENT_RE.finditer(c.text or ""):
            p = m.group("p").replace(",", ".")
            try:
                ps.append(float(p))
            except Exception:
                pass

        if len(ps) < 3:
            continue

        s = sum(ps)
        low = (c.text or "").lower()
        looks_distribution = any(k in low for k in ("структур", "удельн", "доля", "распредел", "в %", "процент"))
        if looks_distribution and s > 110.0:
            res.append(CheckResult(
                kind="percent",
                ok=False,
                message=f"Сумма процентов в одном фрагменте выглядит завышенной: {s:.2f}%. Проверьте базу/округление/двойной счёт.",
                chunk_id=c.chunk_id,
                section_path=c.section_path or None,
            ))
        elif looks_distribution and 90.0 <= s <= 110.0:
            res.append(CheckResult(
                kind="percent",
                ok=True,
                message=f"Сумма процентов в одном фрагменте выглядит правдоподобной: {s:.2f}% (возможны округления).",
                chunk_id=c.chunk_id,
                section_path=c.section_path or None,
            ))

    return res


# -----------------------------
# Agent orchestration
# -----------------------------

def _format_checks_for_user(checks: Sequence[CheckResult], max_items: int = 18) -> str:
    if not checks:
        return ""

    bad = [c for c in checks if not c.ok]
    good = [c for c in checks if c.ok]

    lines: List[str] = []
    if bad:
        lines.append("Найдены потенциальные несостыковки в расчётах:")
        for c in bad[:max_items]:
            ref = ""
            if c.chunk_id is not None:
                ref = f" (chunk_id={c.chunk_id})"
            if c.section_path:
                ref += f" [{c.section_path}]"
            lines.append(f"- {c.message}{ref}")

    if good and len(lines) < max_items + 2:
        lines.append("")
        lines.append("Что удалось подтвердить пересчётом:")
        for c in good[:max(0, max_items - len(bad))]:
            ref = ""
            if c.chunk_id is not None:
                ref = f" (chunk_id={c.chunk_id})"
            if c.section_path:
                ref += f" [{c.section_path}]"
            lines.append(f"- {c.message}{ref}")

    return "\n".join(lines).strip()


def _build_llm_messages(question: str, evidence: Sequence[EvidenceChunk], checks: Sequence[CheckResult]) -> List[Dict[str, str]]:
    ev_sorted = list(evidence)

    def score(c: EvidenceChunk) -> int:
        t = c.text or ""
        return (
            20 * int((c.element_type or "").lower() == "table")
            + 5 * sum(ch.isdigit() for ch in t)
            + 3 * len(_PERCENT_RE.findall(t))
            + 2 * int("=" in t)
        )

    ev_sorted.sort(key=score, reverse=True)
    ev_sorted = ev_sorted[:18]

    ctx_parts: List[str] = []
    for c in ev_sorted:
        sp = c.section_path or ""
        head = f"[chunk_id={c.chunk_id} type={c.element_type}{' path='+sp if sp else ''}]"
        body = (c.text or "").strip()
        if len(body) > 1800:
            body = body[:1800] + "\n...[обрезано]..."

        if (c.element_type or "").lower() == "table":
            tsv = c.attrs.get("table_tsv")
            if isinstance(tsv, str) and tsv.strip():
                body = "[table_tsv]\n" + tsv.strip()

        ctx_parts.append(head + "\n" + body)

    checks_text = _format_checks_for_user(checks, max_items=25) or "Детерминированные проверки не дали однозначных несостыковок."
    system = (
        "Ты — ассистент-аудитор расчётов внутри ВКР/отчёта.\n"
        "Тебе дан только контекст из документа (чанки и таблицы) и результаты машинной проверки.\n"
        "Правила:\n"
        "1) Отвечай строго по контексту, без внешних знаний.\n"
        "2) Если данных недостаточно, так и скажи, предложив, что именно проверить/где искать.\n"
        "3) Если указываешь на проблему — приведи короткую цитату или ссылку на chunk_id.\n"
        "4) Не придумывай цифры, которых нет в контексте.\n"
        "5) Сначала: краткий вывод, затем: список конкретных проверок/замечаний.\n"
    )
    user = (
        f"Вопрос пользователя: {question}\n\n"
        f"Результаты детерминированных проверок:\n{checks_text}\n\n"
        "Контекст из документа ниже. Сформируй ответ:\n"
        "- короткий вывод (1–3 предложения)\n"
        "- затем 'Проверки и замечания' (буллеты)\n"
        "- если несостыковок нет, напиши, что именно удалось проверить и что осталось непроверенным."
    )

    return [
        {"role": "system", "content": system},
        {"role": "assistant", "content": "[КОНТЕКСТ ИЗ ДОКУМЕНТА]\n\n" + "\n\n---\n\n".join(ctx_parts)},
        {"role": "user", "content": user},
    ]


async def answer_calc_question(
    uid: int,
    doc_id: int,
    question: str,
    *,
    chat_with_gpt: Optional[ChatFn] = None,
    max_chunks: int = 140,
) -> Optional[str]:
    """
    Возвращает текст ответа, если вопрос "про расчёты", иначе None.

    Если chat_with_gpt передан — отдаёт LLM-наратив поверх проверок и цитат.
    Если нет — возвращает строгий детерминированный отчёт.
    """
    if not is_calc_question(question):
        return None

    # 1) candidates from chunks
    evidence = await asyncio.to_thread(_select_calc_candidate_chunks, uid, doc_id, max_chunks)

    # 1b) fallback from document_sections if мало/нет кандидатов
    if not evidence or len(evidence) < 5:
        sec_ev = await asyncio.to_thread(
            _select_calc_candidate_sections,
            uid,
            doc_id,
            max(220, max_chunks),
        )
        if sec_ev:
            # более стабильный ключ: тип + section_path + нормализованный текст
            def _norm_key(x: EvidenceChunk) -> tuple:
                txt = (x.text or "").strip()
                txt = re.sub(r"\s+", " ", txt)  # схлопываем пробелы
                return ((x.element_type or "").lower(), (x.section_path or "").strip(), txt[:300])

            seen = {_norm_key(e) for e in evidence}
            for e in sec_ev:
                k = _norm_key(e)
                if k not in seen:
                    evidence.append(e)
                    seen.add(k)


    if not evidence:
        return (
            "По текущему индексу документа не удалось найти фрагменты с расчётами/формулами/таблицами "
            "(числовых данных недостаточно для проверки). "
            "Если расчёты есть в документе, проверьте, что они корректно проиндексированы (особенно таблицы)."
        )

    # 2) checks
    checks: List[CheckResult] = []
    checks.extend(_check_table_totals(evidence, max_tables=10))
    checks.extend(_check_simple_formulas(evidence, max_checks=30))
    checks.extend(_check_revenue_cost_profit(evidence, max_checks=20))
    checks.extend(_check_percent_sanity(evidence, max_checks=20))

    allow_llm = (
        chat_with_gpt is not None
        and (_cfg_calc_agent_use_llm() or _wants_llm_narration(question))
    )

    pretty = _format_checks_pretty(checks, show_chunk_id=False)

    if not allow_llm:
        if pretty:
            return pretty
        # fallback: если проверок нет/они неинформативны
        raw = _format_checks_for_user(checks, max_items=18)
        if raw:
            return raw
        return (
            "Я нашёл в документе числовые фрагменты/таблицы и попытался проверить итоги/простые выражения, "
            "но не получил однозначных несостыковок автоматическими правилами. "
            "Если нужно, уточните: какие именно таблицы/разделы/показатели проверять "
            "(например: 'таблица 3, итого по столбцу X' или 'в абзаце про выручку/расходы/прибыль')."
        )


    # 4) LLM narration over evidence+checks
    messages = _build_llm_messages(question, evidence, checks)
    try:
        ans = await asyncio.to_thread(chat_with_gpt, messages, temperature=0.0, max_tokens=1200)
        ans = (ans or "").strip()
        if ans:
            # анти-галлюцинация: если LLM заявляет “ошибки/не сходится”,
            # а детерминированных ошибок нет — возвращаем deterministic pretty
            bad_cnt = sum(1 for c in checks if not c.ok)
            norm = ans.lower()
            claims_error = any(w in norm for w in ("ошиб", "не сход", "несоответ", "некоррект", "нереалист"))
            if bad_cnt == 0 and claims_error:
                return pretty or _format_checks_for_user(checks, max_items=18) or ans
            return ans
    except Exception:
        pass


    return pretty or _format_checks_for_user(checks, max_items=18) or (
        "Не удалось сформировать ответ: LLM недоступен, а автоматические проверки не дали результата."
    )
