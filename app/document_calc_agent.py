from __future__ import annotations

import asyncio
import json
import re
import csv
from io import StringIO
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

_POSTING_CTX_RE = re.compile(
    r"(?i)\b(проводк|корреспонденц|дебет|кредит|\bдт\b|\bкт\b|счет(а|ов)?|сч\.|субсчет|субсч\.|п/п)\b"
)

def _is_posting_context(text: str) -> bool:
    """
    Возвращает True, если фрагмент похож на бухгалтерские проводки/корреспонденцию счетов.
    В таких местах выражения вида 90.1-44 — это НЕ арифметика, а номера счетов.
    """
    return bool(_POSTING_CTX_RE.search(text or ""))

def _looks_like_account_code(raw: str) -> bool:
    """
    True, если токен похож на номер счета/субсчета или технический идентификатор:
    62.02, 90.01, 10.3, 4.1.1, и т.п.
    """
    s = (raw or "").strip().replace(" ", "").replace(",", ".")
    if not s:
        return False
    # много точек или формат "NN.NN" / "NN.N" / "N.NN"
    if re.fullmatch(r"\d{1,2}\.\d{1,2}(?:\.\d{1,2})*", s):
        return True
    # счета могут встречаться как "99", "44", "51" и т.п. — но это опасно:
    # не будем отсеивать одиночные целые всегда, только когда контекст проводочный.
    return False

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
        return bool(getattr(Cfg, "CALC_AGENT_USE_LLM", True))
    except Exception:
        return False


def _wants_llm_narration(question: str) -> bool:
    """
    Явный запрос на “красивый” нарратив через LLM.
    В остальных случаях лучше детерминированный отчёт.
    """
    q = (question or "").lower()
    triggers = (
        "сформулируй красиво", "красивый ответ", "сделай вывод", "написать вывод", "объясни", "объяснение",
        "реалистич", "проверь расч", "проверка расч", "правильн ли расч", "корректн ли расч"
    )

    return any(t in q for t in triggers)

_TRUNC_END_RE = re.compile(
    r"(\bнапример\b\s*[:,-]?\s*$|\($|,\s*$|:\s*$|—\s*$)",
    flags=re.IGNORECASE,
)

def _looks_truncated_for_user(text: str) -> bool:
    """
    Эвристика 'ответ оборван/обрезан по лимиту'.
    Нужна для UX: не отдавать пользователю полусловные ответы.
    """
    s = (text or "").strip()
    if not s:
        return True

    # если слишком короткий, но при этом содержит заголовки структуры — похоже на обрыв
    if len(s) < 220 and ("проверки и замечания" in s.lower() or "что проверить" in s.lower()):
        return True

    # частые признаки обрыва: заканчивается на "например" / "например," / "например:" или на открывающую скобку
    tail = s[-60:].lower()
    if re.search(r"(например|например,|например:)\s*$", tail):
        return True
    if re.search(r"[\(\[\{]\s*$", s):
        return True

    # если нет финальной пунктуации и текст достаточно длинный — подозрительно
    if len(s) > 450 and not re.search(r"[.!?…]\s*$", s):
        return True

    # если последняя строка явно обрублена (очень короткая и без знака конца)
    last_line = s.splitlines()[-1].strip()
    if len(last_line) < 25 and not re.search(r"[.!?…]\s*$", last_line):
        return True

    return False


_ROW_MARK_RE = re.compile(r"\[row\s+\d+\]", re.IGNORECASE)

def _strip_row_markers(text: str) -> str:
    """
    Удаляет технические маркеры вида [row N] из любого текста.
    Нужна как финальная защита UX (и для LLM-ветки тоже).
    """
    if not text:
        return ""
    # убираем все вхождения [row N]
    out = _ROW_MARK_RE.sub("", text)
    # нормализуем пробелы вокруг удалённого
    out = re.sub(r"[ \t]+\n", "\n", out)
    out = re.sub(r"\n[ \t]+", "\n", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out.strip()

_ROW_SUFFIX_RE = re.compile(r"\s*\[row\s+\d+\]\s*$", flags=re.IGNORECASE)

def _clean_section_path_for_user(sp: str) -> str:
    sp = (sp or "").strip()
    sp = _ROW_SUFFIX_RE.sub("", sp)
    sp = re.sub(r"\s+", " ", sp).strip()
    return sp

def _ref_for_user(c: "CheckResult", show_chunk_id: bool = False) -> str:
    # show_chunk_id оставлен только ради обратной совместимости, но сознательно игнорируется
    _ = show_chunk_id

    parts = []
    sp = _clean_section_path_for_user(c.section_path or "")
    if sp:
        if len(sp) > 180:
            sp = "…" + sp[-180:]
        parts.append(sp)
    return f" (источник: {', '.join(parts)})" if parts else ""


def extract_numbers(text: str) -> List[float]:
    out: List[float] = []
    for m in _NUM_TOKEN_RE.finditer(text or ""):
        val = _parse_number(m.group("num"), unit=m.group("unit"), sign=m.group("sign"))
        if val is not None:
            out.append(val)
    return out

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

@dataclass
class SampleContext:
    total_n: Optional[int]
    groups: Dict[str, int]          # например {"отцы": 20, "матери": 20}
    notes: List[str]               # короткие источники/подсказки


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

                # антишум: если это “годы/нумерация/счета” без денежного/итогового контекста — пропускаем
        has_money_context = any(k in low for k in ("руб", "тыс", "млн", "млрд", "выруч", "доход", "расход", "затрат", "прибыл", "итог", "итого", "сумм"))
        has_ops = bool(_FORMULA_EQUALITY_RE.search(txt) or _FORMULA_LINE_RE.search(txt) or _EQUALITY_RE.search(txt))

        # NEW: проценты/структуры считаем валидным "расчётным" контекстом, не выкидываем как "малые индексы"
        has_percent = bool(_PERCENT_RE.search(txt))
        has_distribution_context = any(k in low for k in ("структур", "удельн", "доля", "распредел", "состав", "соотнош", "в %", "процент", "опрос", "респонд"))

        if not has_money_context and not has_ops and not has_percent and not has_distribution_context:
            nums = extract_numbers(txt)
            # если все числа похожи на годы/индексы/коды — не берём
            if nums and all(_looks_like_year(n) or _looks_like_small_index(n) for n in nums):
                continue

                # антишум для процентов: длинные текстовые чанки часто смешивают несколько диаграмм/оснований
        if et in ("paragraph", "text") and len(txt) > 2000 and has_percent and not has_ops:
            # оставим компактный контекст, чтобы проверки не суммировали "полглавы"
            txt = txt[:1200] + "\n…\n" + txt[-400:]


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

        has_money_context = any(k in low for k in ("руб", "тыс", "млн", "млрд", "выруч", "доход", "расход", "затрат", "прибыл", "итог", "итого", "сумм"))
        has_ops = bool(_FORMULA_EQUALITY_RE.search(txt) or _FORMULA_LINE_RE.search(txt) or _EQUALITY_RE.search(txt))

        if not has_money_context and not has_ops:
            nums = extract_numbers(txt)
            if nums and all(_looks_like_year(n) or _looks_like_small_index(n) for n in nums):
                continue

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

        # NEW: проводки/корреспонденция счетов — не арифметика, пропускаем
        if _is_posting_context(text):
            continue

        # 0) A op B = C
        for m in _FORMULA_EQUALITY_RE.finditer(text):
            a_raw = (m.group(1) or "").strip()
            b_raw = (m.group(3) or "").strip()
            g_raw = (m.group(4) or "").strip()

            # NEW: номера счетов/субсчетов/разделов — не считаем
            if _looks_like_account_code(a_raw) or _looks_like_account_code(b_raw) or _looks_like_account_code(g_raw):
                continue

            if _looks_like_non_math_identifier(a_raw) or _looks_like_non_math_identifier(b_raw) or _looks_like_non_math_identifier(g_raw):
                continue

            a = _parse_number(a_raw, None, None)
            op = (m.group(2) or "").strip()
            b = _parse_number(b_raw, None, None)
            given = _parse_number(g_raw, None, None)
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
                _emit(False, f"Арифметика не сходится: {snippet}. Пересчёт даёт {calc:g} вместо {given:g}.", c)

            if len(res) >= max_checks:
                break

        if len(res) >= max_checks:
            break

        # 1) простое равенство A = B
        for m in _EQUALITY_RE.finditer(text):
            l_raw = (m.group(1) or "").strip()
            r_raw = (m.group(2) or "").strip()

            if _looks_like_account_code(l_raw) or _looks_like_account_code(r_raw):
                continue

            if _looks_like_non_math_identifier(l_raw) or _looks_like_non_math_identifier(r_raw):
                continue

            left = _parse_number(l_raw, unit=None, sign=None)
            right = _parse_number(r_raw, unit=None, sign=None)
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

        # 2) выражение без итога — максимально осторожно
        has_bad = any((not r.ok) for r in res)
        if not has_bad:
            for m in _FORMULA_LINE_RE.finditer(text):
                a_raw = (m.group(1) or "").strip()
                b_raw = (m.group(3) or "").strip()

                if _looks_like_account_code(a_raw) or _looks_like_account_code(b_raw):
                    continue

                if _looks_like_non_math_identifier(a_raw) or _looks_like_non_math_identifier(b_raw):
                    continue

                a = _parse_number(a_raw, None, None)
                b = _parse_number(b_raw, None, None)
                if a is None or b is None:
                    continue

                if _looks_like_year(a) and _looks_like_year(b):
                    continue

                low = (text or "").lower()
                has_money_context = any(k in low for k in ("руб", "тыс", "млн", "млрд", "выруч", "расход", "затрат", "прибыл", "итог", "сумм"))
                if not has_money_context and max(abs(a), abs(b)) < 10_000:
                    continue

                expr = f"{a_raw} {(m.group(2) or '').strip()} {b_raw}"
                _emit(True, f"Найдено арифметическое выражение (без явного итога для сверки): {expr}", c)

                if len(res) >= max_checks:
                    break

    return res

@dataclass
class _AnchorFact:
    label_norm: str
    label_raw: str
    year: int
    value: float
    kind: str                 # money/percent/count/ratio/unknown
    scale: float              # 1 / 1e3 / 1e6 / 1e9 (если удалось определить), иначе 1
    value_base: float         # value * scale
    chunk_id: int
    section_path: str
    ocr_quality: str          # "", "high", "mid", "low"


def _parse_scale_from_text(s: str) -> float:
    """
    Улавливаем масштаб: тыс/млн/млрд.
    Очень частый кейс ВКР: "тыс. руб.", "млн руб.".
    """
    low = (s or "").lower()
    if "млрд" in low:
        return 1e9
    if "млн" in low:
        return 1e6
    if "тыс" in low:
        return 1e3
    return 1.0


def _extract_year_from_cell(s: str) -> List[int]:
    ys: List[int] = []
    for m in re.finditer(r"(19|20)\d{2}", s or ""):
        try:
            ys.append(int(m.group(0)))
        except Exception:
            pass
    return ys


def _normalize_label(s: str) -> str:
    """
    Нормализуем "имя показателя" из первой колонки:
    - lower, убираем пунктуацию
    - режем на токены
    - выбрасываем частые стоп-слова
    - оставляем компактный ключ (до 8 токенов)
    """
    txt = (s or "").lower().strip()
    txt = re.sub(r"[\(\)\[\]\{\}]", " ", txt)
    txt = re.sub(r"[^\w\s\-]", " ", txt, flags=re.UNICODE)
    txt = re.sub(r"\s+", " ", txt).strip()

    if not txt:
        return ""

    stop = {
        "и", "в", "во", "на", "по", "за", "от", "до", "к", "с", "со", "при",
        "показатель", "показатели", "итого", "итог", "всего",
        "тыс", "млн", "млрд", "руб", "рублей", "р.", "ед", "ед.", "шт", "шт.", "чел", "чел.",
        "год", "годы", "гг", "г.", "гг.",
        "значение", "уровень", "динамика",
    }

    toks = [t for t in txt.split(" ") if t and t not in stop]
    # выкинуть одиночные дефисы/мусор
    toks = [t for t in toks if t not in ("-", "—", "–")]
    # ограничим длину ключа (чтобы не было огромных)
    toks = toks[:8]
    return " ".join(toks).strip()


def _is_plausible_label(label_norm: str) -> bool:
    if not label_norm:
        return False
    # слишком короткие/общие
    if label_norm in ("итого", "всего"):
        return False
    # если только цифры/коды
    if re.fullmatch(r"[\d\s\-]+", label_norm):
        return False
    return True


def _infer_table_scale_and_kind(header: List[str], chunk: EvidenceChunk) -> tuple[float, str]:
    """
    Пытаемся определить масштаб и общий kind таблицы.
    kind берём по "самому сильному" индикатору.
    """
    header_text = " ".join([str(x or "") for x in header])
    sp = (chunk.section_path or "")
    cap = (chunk.text or "")
    joined = f"{header_text} {sp} {cap}"

    scale = _parse_scale_from_text(joined)

    # kind: если в заголовках/подписях явный % -> percent, руб/тыс/млн -> money
    low = joined.lower()
    if "%" in low or "процент" in low or "темп" in low or "рост" in low or "прирост" in low:
        kind = "percent"
    elif any(k in low for k in ("руб", "тыс", "млн", "млрд", "выруч", "доход", "расход", "затрат", "прибыл", "себестоим", "стоим")):
        kind = "money"
    elif any(k in low for k in ("шт", "чел", "кол-во", "количество", "единиц", "ед.")):
        kind = "count"
    else:
        kind = "unknown"

    return scale, kind


def _extract_anchor_facts_from_table(
    chunk: EvidenceChunk,
    *,
    max_rows: int = 30,
) -> List[_AnchorFact]:
    """
    Извлекаем "якорные факты" вида (label, year, value) из таблицы.
    Основа для кросс-табличной согласованности.
    """
    grid = _parse_table_grid_from_attrs(chunk.attrs) or None
    if not grid or len(grid) < 3:
        return []

    header_idx, data_start = _header_row_and_data_start(grid)
    n_cols = max(len(r) for r in grid)

    header = [str(x or "") for x in grid[header_idx]]
    header = header + [""] * (n_cols - len(header))

    # карта: year -> col
    year_to_col: Dict[int, int] = {}
    for j in range(n_cols):
        ys = _extract_year_from_cell(header[j])
        for y in ys:
            if y not in year_to_col:
                year_to_col[y] = j

    if len(year_to_col) < 1:
        # иногда годы в 2-й строке заголовка
        if header_idx == 0 and len(grid) > 1:
            header2 = [str(x or "") for x in grid[1]]
            header2 = header2 + [""] * (n_cols - len(header2))
            for j in range(n_cols):
                ys = _extract_year_from_cell(header2[j])
                for y in ys:
                    if y not in year_to_col:
                        year_to_col[y] = j

    if len(year_to_col) < 1:
        return []

    years = sorted(year_to_col.keys())
    scale_table, kind_table = _infer_table_scale_and_kind(header, chunk)
    ocr_quality = (chunk.attrs or {}).get("ocr_quality") or ""

    facts: List[_AnchorFact] = []
    rows = grid[data_start : min(len(grid), data_start + max_rows)]

    for row in rows:
        # label — первая ячейка
        label_raw = str(row[0] if len(row) > 0 else "").strip()
        label_norm = _normalize_label(label_raw)

        if not _is_plausible_label(label_norm):
            continue

        # иногда первая колонка пустая, а метка во второй
        if not label_norm and len(row) > 1:
            label_raw2 = str(row[1] or "").strip()
            label_norm2 = _normalize_label(label_raw2)
            if _is_plausible_label(label_norm2):
                label_raw = label_raw2
                label_norm = label_norm2

        if not _is_plausible_label(label_norm):
            continue

        for y in years:
            j = year_to_col[y]
            cell = str(row[j] if j < len(row) else "").strip()
            if not cell:
                continue
            v = _cell_to_number(cell)
            if v is None:
                continue

            # kind уточняем по конкретной колонке (если там %, то percent)
            kind = _infer_kind_from_header(header[j]) if header and j < len(header) else kind_table
            if kind == "unknown":
                kind = kind_table

            scale = scale_table
            v_base = float(v) * float(scale)

            facts.append(_AnchorFact(
                label_norm=label_norm,
                label_raw=label_raw,
                year=int(y),
                value=float(v),
                kind=kind,
                scale=float(scale),
                value_base=v_base,
                chunk_id=int(chunk.chunk_id),
                section_path=str(chunk.section_path or ""),
                ocr_quality=str(ocr_quality or ""),
            ))

    return facts


def _cluster_median(values: List[float]) -> float:
    if not values:
        return 0.0
    vs = sorted(values)
    return vs[len(vs) // 2]


def _scale_mismatch_hint(a: float, b: float) -> Optional[str]:
    """
    Если значения отличаются примерно в 1000x или 1e6x — типичный mismatch масштаба.
    """
    aa = abs(a)
    bb = abs(b)
    if aa < 1e-12 or bb < 1e-12:
        return None
    r = max(aa, bb) / max(1e-12, min(aa, bb))
    # грубые зоны: 900..1100 и 9e5..1.1e6
    if 900.0 <= r <= 1100.0:
        return "Возможна смена масштаба (тыс. ↔ руб.)."
    if 9e5 <= r <= 1.1e6:
        return "Возможна смена масштаба (млн ↔ руб.) или ошибка единиц."
    return None

def _cross_table_close(expected: float, given: float, *, max_scale: float, ocr_quality: str = "") -> bool:
    """
    Более строгая близость для межтабличных сравнений.
    Логика:
    - rel меньше, чем в обычных money-checks
    - abs_tol зависит от max_scale (тыс/млн/млрд), т.к. округления там крупнее
    """
    rel = 0.005  # 0.5%
    abs_tol = 2000.0  # базовый допуск в рублях

    # если хотя бы одна таблица в тыс/млн/млрд, позволяем абсолютную погрешность кратно масштабу
    # тыс -> ~2000 руб, млн -> ~2 000 000 руб, млрд -> ~2 000 000 000 руб
    if max_scale >= 1e3:
        abs_tol = max(abs_tol, 2.0 * float(max_scale))

    q = (ocr_quality or "").lower()
    if q == "low":
        rel *= 1.6
        abs_tol *= 2.0
    elif q == "mid":
        rel *= 1.3
        abs_tol *= 1.6

    try:
        return _approx_equal(expected, given, rel=rel, abs_tol=abs_tol)  # type: ignore
    except Exception:
        diff = abs(expected - given)
        return diff <= abs_tol or diff <= rel * max(1.0, abs(expected), abs(given))


def _check_cross_table_consistency(
    chunks: Sequence[EvidenceChunk],
    *,
    max_tables: int = 10,
    max_checks: int = 18,
) -> List[CheckResult]:
    """
    Этап 3.1: кросс-табличная согласованность.
    Ищем повторяющиеся (label, year) с разными значениями в разных таблицах/разделах.
    """
    res: List[CheckResult] = []

    tables = [c for c in chunks if (c.element_type or "").lower() == "table" or c.attrs.get("is_table")]
    tables = tables[:max_tables]

    facts: List[_AnchorFact] = []
    for t in tables:
        if len(facts) > 500:
            break
        facts.extend(_extract_anchor_facts_from_table(t, max_rows=30))

    if not facts:
        return res

    # группируем по (label_norm, year, kind)
    by_key: Dict[tuple, List[_AnchorFact]] = {}
    for f in facts:
        key = (f.label_norm, f.year, f.kind)
        by_key.setdefault(key, []).append(f)

    # только те ключи, где есть минимум 2 источника (разные chunk_id)
    candidates = []
    for key, items in by_key.items():
        uniq = {it.chunk_id for it in items}
        if len(uniq) >= 2:
            candidates.append((key, items))

    # ранжируем кандидаты: больше источников и больше разброс
    def _spread_score(items: List[_AnchorFact]) -> float:
        vals = [it.value_base for it in items]
        if not vals:
            return 0.0
        return (max(vals) - min(vals)) / max(1.0, abs(_cluster_median(vals)))

    candidates.sort(key=lambda kv: (len({i.chunk_id for i in kv[1]}), _spread_score(kv[1])), reverse=True)

    for (label_norm, year, kind), items in candidates:
        if len(res) >= max_checks:
            break

        # берём по одному факту на источник (таблицу)
        picked: Dict[int, _AnchorFact] = {}
        for it in items:
            if it.chunk_id not in picked:
                picked[it.chunk_id] = it
        group = list(picked.values())

        if len(group) < 2:
            continue

        vals = [g.value_base for g in group]
        med = _cluster_median(vals)

        # если OCR-качество низкое хотя бы у одного — делаем вывод мягче
        worst_ocr = "high"
        for g in group:
            q = (g.ocr_quality or "").lower()
            if q == "low":
                worst_ocr = "low"
                break
            if q == "mid" and worst_ocr != "low":
                worst_ocr = "mid"

        # проверяем отклонения от медианы
        max_scale = max(float(g.scale or 1.0) for g in group)

        bad_items = []
        for g in group:
            if kind == "money":
                close = _cross_table_close(med, g.value_base, max_scale=max_scale, ocr_quality=worst_ocr)
            else:
                # для не-money оставим старое поведение (мягче)
                close = _adaptive_close(med, g.value_base, kind="unknown", ocr_quality=worst_ocr)

            if not close:
                bad_items.append(g)


        if not bad_items:
            continue

        # если плохих слишком много, может быть неоднозначная метка — всё равно покажем, но осторожно
        # сформируем 1 сообщение на группу, с 2-3 примерами
        label_show = group[0].label_raw or label_norm
        label_show = re.sub(r"\s+", " ", label_show).strip()
        if len(label_show) > 60:
            label_show = label_show[:60].rstrip() + "…"

        examples = sorted(group, key=lambda x: abs(x.value_base - med), reverse=True)[:3]
        ex_txt = []
        for e in examples:
            v_disp = e.value
            # покажем и масштаб, если не 1
            scale_note = ""
            if abs(e.scale - 1.0) > 1e-9:
                if e.scale == 1e3:
                    scale_note = " (в тыс.)"
                elif e.scale == 1e6:
                    scale_note = " (в млн.)"
                elif e.scale == 1e9:
                    scale_note = " (в млрд.)"
            sp = e.section_path or ""
            if sp:
                sp = re.sub(r"\s+", " ", sp).strip()
                if len(sp) > 80:
                    sp = sp[:80].rstrip() + "…"
            ex_txt.append(f"{v_disp:g}{scale_note} — {sp}")

        hint = _scale_mismatch_hint(max(vals), min(vals))
        note = ""
        if worst_ocr in ("low", "mid"):
            note = " Возможна погрешность распознавания таблицы (OCR)."

        msg = (
            f"Кросс-проверка: показатель «{label_show}» за {year} год встречается с разными значениями в документе. "
            f"Примеры: " + "; ".join(ex_txt[:3]) + "."
        )
        if hint:
            msg += " " + hint
        msg += note

        # chunk_id привяжем к первому конфликтному источнику (это нужно внутр. системе, но не показывается пользователю в pretty)
        res.append(CheckResult(
            kind="cross_table",
            ok=False,
            message=msg,
            chunk_id=group[0].chunk_id,
            section_path=group[0].section_path or None,
        ))

    return res


def _check_trend_outliers(
    chunks: Sequence[EvidenceChunk],
    *,
    max_tables: int = 10,
    max_checks: int = 12,
) -> List[CheckResult]:
    """
    Этап 3.2: sanity по временным рядам и выбросам (мягко).
    Ищем:
    - скачки на порядок (ratio > 30x) между соседними годами
    - подозрительные отрицательные значения для денег/количества (warning)
    """
    res: List[CheckResult] = []

    tables = [c for c in chunks if (c.element_type or "").lower() == "table" or c.attrs.get("is_table")]
    tables = tables[:max_tables]

    for t in tables:
        if len(res) >= max_checks:
            break

        grid = _parse_table_grid_from_attrs(t.attrs) or None
        if not grid or len(grid) < 4:
            continue

        header_idx, data_start = _header_row_and_data_start(grid)
        n_cols = max(len(r) for r in grid)
        header = [str(x or "") for x in grid[header_idx]]
        header = header + [""] * (n_cols - len(header))

        # year->col
        year_to_col: Dict[int, int] = {}
        for j in range(n_cols):
            ys = _extract_year_from_cell(header[j])
            for y in ys:
                if y not in year_to_col:
                    year_to_col[y] = j

        years = sorted(year_to_col.keys())
        if len(years) < 2:
            continue

        scale_table, kind_table = _infer_table_scale_and_kind(header, t)
        ocr_quality = (t.attrs or {}).get("ocr_quality") or ""

        rows = grid[data_start : min(len(grid), data_start + 30)]
        for row in rows:
            if len(res) >= max_checks:
                break

            label_raw = str(row[0] if len(row) > 0 else "").strip()
            label_norm = _normalize_label(label_raw)
            if not _is_plausible_label(label_norm):
                continue

            # соберём ряд
            series: List[tuple[int, float]] = []
            for y in years:
                j = year_to_col[y]
                cell = str(row[j] if j < len(row) else "").strip()
                if not cell:
                    continue
                v = _cell_to_number(cell)
                if v is None:
                    continue
                kind = _infer_kind_from_header(header[j]) if header and j < len(header) else kind_table
                if kind == "unknown":
                    kind = kind_table
                vb = float(v) * float(scale_table)
                series.append((y, vb))

            if len(series) < 2:
                continue

            series.sort(key=lambda x: x[0])
            # проверка скачков
            for i in range(1, len(series)):
                y0, v0 = series[i - 1]
                y1, v1 = series[i]
                if abs(v0) < 1e-12:
                    continue
                ratio = abs(v1) / max(1e-12, abs(v0))
                if ratio > 30.0:
                    label_show = re.sub(r"\s+", " ", label_raw).strip()
                    if len(label_show) > 60:
                        label_show = label_show[:60].rstrip() + "…"
                    note = ""
                    if (ocr_quality or "").lower() in ("low", "mid"):
                        note = " (возможна погрешность OCR)"
                    res.append(CheckResult(
                        kind="trend",
                        ok=True,
                        message=(
                            f"Динамика: у показателя «{label_show}» наблюдается резкий скачок между {y0} и {y1}: "
                            f"{v0:g} → {v1:g} (≈{ratio:.1f}×). Проверьте единицы/масштаб/ввод.{note}"
                        ),
                        chunk_id=t.chunk_id,
                        section_path=t.section_path or None,
                    ))
                    break

            # проверка отрицательных значений (мягко; не всегда ошибка)
            if kind_table in ("money", "count") or any(_infer_kind_from_header(h) in ("money", "count") for h in header):
                negatives = [(y, v) for (y, v) in series if v < -1e-9]
                if negatives and len(res) < max_checks:
                    label_show = re.sub(r"\s+", " ", label_raw).strip()
                    if len(label_show) > 60:
                        label_show = label_show[:60].rstrip() + "…"
                    ys = ", ".join(str(y) for y, _ in negatives[:3])
                    res.append(CheckResult(
                        kind="trend",
                        ok=True,
                        message=(
                            f"Сanity: у показателя «{label_show}» есть отрицательные значения ({ys}). "
                            f"Если это не допускается методикой расчёта, проверьте формулы/источник."
                        ),
                        chunk_id=t.chunk_id,
                        section_path=t.section_path or None,
                    ))

    return res

def _check_revenue_cost_profit(chunks: Sequence[EvidenceChunk], max_checks: int = 24) -> List[CheckResult]:
    res: List[CheckResult] = []

    PROF_PAT = r"\b(прибыл[ьи]|финансов(?:ый|ого)\s+результат|результат\s+от\s+продаж|валов(?:ая|ой)\s+прибыл)\b"

    def _pick_near(text: str, pat: str) -> Optional[float]:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if not m:
            return None
        tail = text[m.end(): m.end() + 220]
        mnum = _NUM_TOKEN_RE.search(tail)
        if not mnum:
            return None
        return _parse_number(mnum.group("num"), unit=mnum.group("unit"), sign=mnum.group("sign"))

    def _try_from_table(chunk: EvidenceChunk) -> Optional[tuple]:
        grid = _parse_table_grid_from_attrs(chunk.attrs) if chunk.attrs else None
        if not grid or len(grid) < 2:
            return None

        # NEW: если таблица похожа на проводки — не интерпретируем как экономику
        joined_all = " ".join(" ".join(map(str, r)) for r in grid).lower()
        if _is_posting_context(joined_all):
            return None

        norm = []
        for r in grid:
            norm.append([str(x).strip() for x in r])

        def row_value_by_keywords(keywords: tuple[str, ...]) -> Optional[float]:
            for r in norm:
                joined = " ".join(r).lower()
                if any(k in joined for k in keywords):
                    best: Optional[float] = None
                    for cell in reversed(r):
                        v = _cell_to_number(cell)
                        if v is not None:
                            best = v
                            break
                    if best is not None:
                        return best
            return None

        rev = row_value_by_keywords(("выруч", "доход"))
        cost = row_value_by_keywords(("расход", "затрат", "издерж"))
        prof = row_value_by_keywords(("прибыл", "финансов", "результат", "валов"))

        if rev is None or cost is None or prof is None:
            return None

        return (rev, cost, prof)

    ordered = sorted(list(chunks), key=lambda x: x.chunk_id)

    for i in range(len(ordered)):
        if len(res) >= max_checks:
            break

        window = ordered[i:i + 3]
        text = "\n".join((c.text or "") for c in window if (c.text or "").strip())
        low = text.lower()

        # NEW: окно похоже на проводки — пропускаем
        if _is_posting_context(text):
            continue

        tab_triplet = None
        anchor = window[0]
        for w in window:
            if (w.element_type or "").lower() == "table" or (w.attrs or {}).get("is_table"):
                tab_triplet = _try_from_table(w)
                if tab_triplet:
                    anchor = w
                    break

        if tab_triplet:
            rev, cost, prof = tab_triplet
        else:
            if not any(k in low for k in ("выруч", "доход", "расход", "затрат", "издерж", "прибыл", "финансов", "результат")):
                continue

            rev = _pick_near(text, r"\b(выручк[аеиу]|доход[ыа]?)\b")
            cost = _pick_near(text, r"\b(расход[ыа]?|затрат[ыа]?|издержк[иа])\b")
            prof = _pick_near(text, PROF_PAT)

            if rev is None or cost is None or prof is None:
                continue

        calc = rev - cost
        ok = _approx_equal(calc, prof, rel=2e-3)

        if ok:
            res.append(CheckResult(
                kind="formula",
                ok=True,
                message=f"Выручка − расходы сходятся с прибылью: {rev:g} - {cost:g} ≈ {prof:g}.",
                chunk_id=anchor.chunk_id,
                section_path=anchor.section_path or None,
            ))
        else:
            res.append(CheckResult(
                kind="formula",
                ok=False,
                message=f"Выручка − расходы не сходятся с прибылью: {rev:g} - {cost:g} = {calc:g}, но указано {prof:g}.",
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

def _check_delta_and_growth_columns(
    chunks: Sequence[EvidenceChunk],
    *,
    max_tables: int = 10,
    max_checks: int = 24,
) -> List[CheckResult]:
    """
    Универсальная проверка типовых аналитических таблиц ВКР:
    - "Отклонение / изменение" (Δ): cur - prev
    - "% / темп роста / темп прироста":
        вариант A: (cur/prev)*100
        вариант B: (cur/prev - 1)*100

    Работает только если table_grid/table_tsv есть в attrs (или был добавлен enrichment'ом OCR).
    """
    res: List[CheckResult] = []

    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip().lower())

    def _has_any(h: str, keys: Sequence[str]) -> bool:
        hh = _norm(h)
        return any(k in hh for k in keys)

    def _extract_years(s: str) -> List[int]:
        ys = []
        for m in re.finditer(r"(19|20)\d{2}", s or ""):
            try:
                ys.append(int(m.group(0)))
            except Exception:
                pass
        return ys

    def _cell_to_percent(cell: str) -> Optional[float]:
        """
        Пытаемся извлечь число-процент:
        - "12,3%" -> 12.3
        - "12.3" (в колонке % бывает без знака) -> 12.3
        """
        if not cell:
            return None
        t = (cell or "").strip()
        # явный процент
        m = _PERCENT_RE.search(t)
        if m:
            try:
                return float(m.group("p").replace(",", "."))
            except Exception:
                return None
        # иначе попробуем как число (но только если там нет явной валюты/единиц)
        m2 = _NUM_TOKEN_RE.search(t)
        if not m2:
            return None
        try:
            v = _parse_number(m2.group("num"), unit=None, sign=m2.group("sign"))
            return float(v) if v is not None else None
        except Exception:
            return None

    def _best_pct_variant(pairs: List[tuple]) -> Optional[str]:
        """
        pairs: [(prev, cur, given_pct), ...]
        Возвращает "A" или "B", где:
          A = cur/prev*100
          B = (cur/prev - 1)*100
        """
        if len(pairs) < 3:
            return None

        errs_a = []
        errs_b = []
        for prev, cur, given in pairs:
            if prev is None or cur is None or given is None:
                continue
            if abs(prev) < 1e-12:
                continue
            a = (cur / prev) * 100.0
            b = (cur / prev - 1.0) * 100.0
            errs_a.append(abs(a - given))
            errs_b.append(abs(b - given))

        if len(errs_a) < 3 or len(errs_b) < 3:
            return None

        # медиана абсолютной ошибки (устойчиво к выбросам)
        errs_a.sort()
        errs_b.sort()
        ma = errs_a[len(errs_a) // 2]
        mb = errs_b[len(errs_b) // 2]
        return "A" if ma <= mb else "B"

    tables = [c for c in chunks if (c.element_type or "").lower() == "table" or c.attrs.get("is_table")]
    tables = tables[:max_tables]

    for c in tables:
        if len(res) >= max_checks:
            break

        grid = _parse_table_grid_from_attrs(c.attrs) or None
        if not grid or len(grid) < 3:
            continue

        # -----------------
        # Определяем header-строки
        # -----------------
        # Часто: 2 header-строки, где годы во 2-й
        header_idx = 0
        data_start = 1

        row0 = [str(x or "") for x in grid[0]]
        row1 = [str(x or "") for x in (grid[1] if len(grid) > 1 else [])]

        years0 = []
        for cell in row0:
            years0 += _extract_years(cell)
        years1 = []
        for cell in row1:
            years1 += _extract_years(cell)

        if len(set(years0)) < 2 and len(set(years1)) >= 2:
            header_idx = 1
            data_start = 2

        header = [str(x or "") for x in grid[header_idx]]
        n_cols = max(len(r) for r in grid)
        header = header + [""] * (n_cols - len(header))

        # -----------------
        # Ищем колонки-«годы»
        # -----------------
        year_cols: Dict[int, int] = {}
        for j in range(n_cols):
            ys = _extract_years(header[j])
            if ys:
                # если несколько лет в заголовке — берём все, но это редкость
                for y in ys:
                    if y not in year_cols:
                        year_cols[y] = j

        years_sorted = sorted(year_cols.keys())
        if len(years_sorted) < 2:
            continue  # нечего сравнивать

        # берём "последнюю пару лет" как дефолт
        prev_year = years_sorted[-2]
        cur_year = years_sorted[-1]
        prev_col = year_cols[prev_year]
        cur_col = year_cols[cur_year]

        # -----------------
        # Кандидаты колонок: delta и pct
        # -----------------
        delta_cols: List[int] = []
        pct_cols: List[int] = []

        for j in range(n_cols):
            h = header[j]
            if not h:
                continue

            if _has_any(h, ("откл", "отклон", "измен", "разниц", "дельт", "∆")):
                delta_cols.append(j)
                continue

            if _has_any(h, ("%", "процент", "темп", "рост", "прирост")):
                pct_cols.append(j)
                continue

            # иногда пишут "2023-2022" как отдельный заголовок без слов
            if re.search(r"(19|20)\d{2}\s*[-–—]\s*(19|20)\d{2}", h):
                delta_cols.append(j)

        # если ни delta, ни pct — не шумим
        if not delta_cols and not pct_cols:
            continue

        # -----------------
        # Собираем строки данных
        # -----------------
        ok_delta = 0
        bad_delta = 0
        ok_pct = 0
        bad_pct = 0

        pct_pairs_for_choice: List[tuple] = []
        parsed_rows = 0

        for i in range(data_start, len(grid)):
            if len(res) >= max_checks:
                break

            row = [str(x or "") for x in grid[i]]
            row = row + [""] * (n_cols - len(row))

            # пропустим пустые строки
            if not any((cell or "").strip() for cell in row):
                continue

            prev_val = _cell_to_number(row[prev_col])
            cur_val = _cell_to_number(row[cur_col])

            if prev_val is None or cur_val is None:
                continue

            parsed_rows += 1
            row_label = (row[0] or "").strip()
            if row_label:
                row_label = re.sub(r"\s+", " ", row_label)
                if len(row_label) > 60:
                    row_label = row_label[:60].rstrip() + "…"

            # -------- delta --------
            for dj in delta_cols:
                given_delta = _cell_to_number(row[dj])
                if given_delta is None:
                    continue

                expected = cur_val - prev_val
                # допускаем округление: для денег обычно ±1..±2 в зависимости от масштаба
                abs_tol = 1.0 if max(abs(cur_val), abs(prev_val)) >= 50 else 0.5
                ok = _approx_equal(expected, given_delta, rel=0.01, abs_tol=abs_tol)

                col_name = (header[dj] or "").strip()
                col_hint = f" (колонка: {col_name})" if col_name else ""

                if ok:
                    ok_delta += 1
                else:
                    bad_delta += 1
                    who = f" строка «{row_label}»" if row_label else f" строка #{i+1}"
                    res.append(CheckResult(
                        kind="table_delta",
                        ok=False,
                        message=(
                            f"Таблица: отклонение не сходится{col_hint}:"
                            f"{who}: {cur_year}-{prev_year} = {cur_val:g}-{prev_val:g} = {expected:g},"
                            f" но указано {given_delta:g}."
                        ),
                        chunk_id=c.chunk_id,
                        section_path=c.section_path or None,
                    ))
                    if len(res) >= max_checks:
                        break

            # -------- pct --------
            for pj in pct_cols:
                given_pct = _cell_to_percent(row[pj])
                if given_pct is None:
                    continue
                if abs(prev_val) < 1e-12:
                    continue

                pct_pairs_for_choice.append((prev_val, cur_val, given_pct))

        # если данных мало — не шумим
        if parsed_rows < 3:
            continue

        # Выбираем вариант формулы для процентов (A или B)
        pct_variant = _best_pct_variant(pct_pairs_for_choice)

        if pct_variant and pct_cols:
            # второй проход по строкам, чтобы репортить конкретные расхождения (но экономно)
            for i in range(data_start, len(grid)):
                if len(res) >= max_checks:
                    break

                row = [str(x or "") for x in grid[i]]
                row = row + [""] * (n_cols - len(row))
                if not any((cell or "").strip() for cell in row):
                    continue

                prev_val = _cell_to_number(row[prev_col])
                cur_val = _cell_to_number(row[cur_col])
                if prev_val is None or cur_val is None or abs(prev_val) < 1e-12:
                    continue

                row_label = (row[0] or "").strip()
                if row_label:
                    row_label = re.sub(r"\s+", " ", row_label)
                    if len(row_label) > 60:
                        row_label = row_label[:60].rstrip() + "…"

                expected_pct = (cur_val / prev_val) * 100.0 if pct_variant == "A" else (cur_val / prev_val - 1.0) * 100.0

                for pj in pct_cols:
                    given_pct = _cell_to_percent(row[pj])
                    if given_pct is None:
                        continue

                    # допускаем округления в п.п.
                    ok = abs(expected_pct - given_pct) <= 1.2  # 1.2 процентного пункта
                    if ok:
                        ok_pct += 1
                    else:
                        bad_pct += 1
                        col_name = (header[pj] or "").strip()
                        col_hint = f" (колонка: {col_name})" if col_name else ""
                        who = f" строка «{row_label}»" if row_label else f" строка #{i+1}"

                        formula_hint = "cur/prev*100" if pct_variant == "A" else "(cur/prev-1)*100"
                        res.append(CheckResult(
                            kind="table_percent",
                            ok=False,
                            message=(
                                f"Таблица: процент/темп не сходится{col_hint}:"
                                f"{who}: ожидается {expected_pct:.2f}% ({formula_hint}), но указано {given_pct:.2f}%."
                            ),
                            chunk_id=c.chunk_id,
                            section_path=c.section_path or None,
                        ))
                        if len(res) >= max_checks:
                            break

        # если ни одной ошибки не нашли, но подтверждений много — добавим 1 “ok” для качества отчёта
        if len(res) < max_checks:
            # delta: если было много валидных и нет ошибок
            if delta_cols and bad_delta == 0 and ok_delta >= 3:
                res.append(CheckResult(
                    kind="table_delta",
                    ok=True,
                    message=f"Таблица: отклонения/разницы по годам выглядят согласованными (проверено строк: {ok_delta}).",
                    chunk_id=c.chunk_id,
                    section_path=c.section_path or None,
                ))
            # pct: если было много валидных и нет ошибок
            if pct_cols and bad_pct == 0 and ok_pct >= 3:
                res.append(CheckResult(
                    kind="table_percent",
                    ok=True,
                    message=f"Таблица: проценты/темпы выглядят согласованными (проверено значений: {ok_pct}).",
                    chunk_id=c.chunk_id,
                    section_path=c.section_path or None,
                ))

    return res

@dataclass
class _InferredRule:
    target_col: int
    a_col: int
    b_col: int
    op: str                 # "add", "sub_ab", "sub_ba", "mul", "div_ab", "div_ba", "pct_ab", "pct_ba", "pctchg_ab", "pctchg_ba"
    support: int
    total: int
    support_ratio: float
    median_abs_err: float


def _adaptive_close(expected: float, given: float, *, kind: str = "unknown", ocr_quality: str = "") -> bool:
    """
    Универсальная проверка близости с адаптивными допусками.
    kind: money/percent/count/ratio/unknown
    ocr_quality: "high"/"mid"/"low" (если low — допускаем больше ошибок распознавания)
    """
    if expected is None or given is None:
        return False

    # базовые допуски
    rel = 0.02
    abs_tol = 0.8

    if kind == "percent":
        rel = 0.03
        abs_tol = 1.2  # 1.2 п.п.
    elif kind == "count":
        rel = 0.01
        abs_tol = 1.0
    elif kind == "money":
        rel = 0.02
        # деньги часто округляют сильнее, особенно в тыс/млн
        mag = max(abs(expected), abs(given))
        if mag >= 1_000_000:
            abs_tol = 2000.0
        elif mag >= 100_000:
            abs_tol = 200.0
        elif mag >= 10_000:
            abs_tol = 20.0
        elif mag >= 1000:
            abs_tol = 5.0
        else:
            abs_tol = 1.0
    elif kind == "ratio":
        rel = 0.03
        abs_tol = 0.02

    # если OCR низкого качества — расширяем допуск
    if (ocr_quality or "").lower() == "low":
        rel *= 1.8
        abs_tol *= 2.5
    elif (ocr_quality or "").lower() == "mid":
        rel *= 1.3
        abs_tol *= 1.6

    # используем вашу существующую approx (если есть), иначе простой вариант
    try:
        return _approx_equal(expected, given, rel=rel, abs_tol=abs_tol)  # type: ignore
    except Exception:
        diff = abs(expected - given)
        return diff <= abs_tol or diff <= rel * max(1.0, abs(expected), abs(given))


def _infer_kind_from_header(h: str) -> str:
    hh = (h or "").lower()
    if any(k in hh for k in ("%", "процент", "темп", "рост", "прирост")):
        return "percent"
    if any(k in hh for k in ("шт", "ед", "чел", "человек", "кол-во", "количество", "единиц")):
        return "count"
    if any(k in hh for k in ("руб", "тыс", "млн", "млрд", "стоим", "выруч", "доход", "расход", "затрат", "прибыл", "себестоим")):
        return "money"
    if any(k in hh for k in ("коэфф", "показат", "удельн", "на 1", "на 1 руб", "на рубль", "рентаб")):
        return "ratio"
    return "unknown"


def _header_row_and_data_start(grid: List[List[str]]) -> tuple[int, int]:
    """
    Пытаемся определить, где header и где начинаются данные.
    Логика мягкая: если во 2-й строке больше "лет", чем в 1-й — header_idx=1.
    """
    def _years_in_row(r: List[str]) -> int:
        cnt = 0
        for cell in r:
            if re.search(r"(19|20)\d{2}", cell or ""):
                cnt += 1
        return cnt

    if not grid or len(grid) < 2:
        return (0, 1)
    r0 = [str(x or "") for x in grid[0]]
    r1 = [str(x or "") for x in grid[1]]
    y0 = _years_in_row(r0)
    y1 = _years_in_row(r1)
    if y1 >= 2 and y1 > y0:
        return (1, 2)
    return (0, 1)


def _numeric_columns(grid: List[List[str]], data_start: int, header: List[str]) -> List[int]:
    """
    Возвращает индексы "числовых" колонок по частоте парсинга чисел в первых строках данных.
    """
    if not grid or len(grid) <= data_start:
        return []

    n_cols = max(len(r) for r in grid)
    sample_rows = grid[data_start : min(len(grid), data_start + 12)]

    numeric_cols: List[int] = []
    for j in range(n_cols):
        parsed = 0
        seen = 0
        for row in sample_rows:
            cell = str(row[j] if j < len(row) else "")
            cell = (cell or "").strip()
            if not cell:
                continue
            seen += 1
            v = _cell_to_number(cell)
            if v is not None:
                parsed += 1

        # колонка числовая, если хотя бы 3 значения и парсинг >= 60%
        if seen >= 3 and parsed / max(1, seen) >= 0.6:
            numeric_cols.append(j)

    # часто 0-я колонка — текстовая метка (исключим, если header не числовой)
    if 0 in numeric_cols:
        h0 = (header[0] or "").lower() if header else ""
        if not re.search(r"\d", h0) and not any(k in h0 for k in ("год", "year")):
            # если первая колонка больше похожа на метки, убираем
            numeric_cols.remove(0)

    return numeric_cols


def _year_columns_from_header(header: List[str]) -> set[int]:
    out: set[int] = set()
    for j, h in enumerate(header or []):
        if re.search(r"(19|20)\d{2}", h or ""):
            out.add(j)
    return out


def _op_predict(op: str, a: float, b: float) -> Optional[float]:
    try:
        if op == "add":
            return a + b
        if op == "sub_ab":
            return a - b
        if op == "sub_ba":
            return b - a
        if op == "mul":
            return a * b
        if op == "div_ab":
            if abs(b) < 1e-12:
                return None
            return a / b
        if op == "div_ba":
            if abs(a) < 1e-12:
                return None
            return b / a
        if op == "pct_ab":
            if abs(b) < 1e-12:
                return None
            return (a / b) * 100.0
        if op == "pct_ba":
            if abs(a) < 1e-12:
                return None
            return (b / a) * 100.0
        if op == "pctchg_ab":
            if abs(b) < 1e-12:
                return None
            return (a / b - 1.0) * 100.0
        if op == "pctchg_ba":
            if abs(a) < 1e-12:
                return None
            return (b / a - 1.0) * 100.0
    except Exception:
        return None
    return None


def _median(xs: List[float]) -> float:
    if not xs:
        return 0.0
    xs2 = sorted(xs)
    return xs2[len(xs2) // 2]


def _infer_rules_for_table(
    grid: List[List[str]],
    *,
    ocr_quality: str = "",
    min_support_ratio: float = 0.70,
    min_support_rows: int = 5,
) -> List[_InferredRule]:
    """
    Индукция формул по таблице:
    Для каждой "целевой" числовой колонки пытаемся объяснить её через 2 другие колонки простыми операциями.
    Возвращает список правил, которые имеют высокий support.
    """
    if not grid or len(grid) < 4:
        return []

    header_idx, data_start = _header_row_and_data_start(grid)
    n_cols = max(len(r) for r in grid)

    header = [str(x or "") for x in grid[header_idx]]
    header = header + [""] * (n_cols - len(header))

    year_cols = _year_columns_from_header(header)
    num_cols = _numeric_columns(grid, data_start, header)

    # исключаем годовые колонки из кандидатов формул (они чаще "база", но редко "результат")
    # при этом годовые колонки могут быть A/B, но как target почти всегда нет смысла.
    target_candidates = [j for j in num_cols if j not in year_cols]
    if len(target_candidates) < 2:
        return []

    ops = ["add", "sub_ab", "sub_ba", "mul", "div_ab", "div_ba", "pct_ab", "pct_ba", "pctchg_ab", "pctchg_ba"]

    rules: List[_InferredRule] = []
    data_rows = grid[data_start:]

    # предварительно: соберём распарсенные значения для ускорения
    parsed: List[List[Optional[float]]] = []
    for row in data_rows:
        prow: List[Optional[float]] = []
        for j in range(n_cols):
            cell = str(row[j] if j < len(row) else "")
            prow.append(_cell_to_number(cell))
        parsed.append(prow)

    for tcol in target_candidates:
        t_kind = _infer_kind_from_header(header[tcol])

        best: Optional[_InferredRule] = None

        # ограничим перебор: только по числовым колонкам
        source_cols = [j for j in num_cols if j != tcol]
        if len(source_cols) < 2:
            continue

        for ai in range(len(source_cols)):
            a_col = source_cols[ai]
            for bi in range(ai + 1, len(source_cols)):
                b_col = source_cols[bi]

                for op in ops:
                    # если целевая колонка НЕ процентная, не пробуем pct-операции (шум)
                    if t_kind != "percent" and op.startswith("pct"):
                        continue

                    # если целевая колонка процентная, то add/mul часто шумят — но оставим div/pct
                    # (оставим add/sub как fallback, но они редко победят по support)
                    ok = 0
                    total = 0
                    abs_errs: List[float] = []

                    for r in range(len(parsed)):
                        a = parsed[r][a_col]
                        b = parsed[r][b_col]
                        t = parsed[r][tcol]
                        if a is None or b is None or t is None:
                            continue

                        pred = _op_predict(op, a, b)
                        if pred is None:
                            continue

                        total += 1
                        if _adaptive_close(pred, t, kind=t_kind, ocr_quality=ocr_quality):
                            ok += 1
                        else:
                            abs_errs.append(abs(pred - t))

                    if total < min_support_rows:
                        continue

                    ratio = ok / max(1, total)
                    if ratio < min_support_ratio:
                        continue

                    med_abs = _median(abs_errs) if abs_errs else 0.0
                    cand = _InferredRule(
                        target_col=tcol,
                        a_col=a_col,
                        b_col=b_col,
                        op=op,
                        support=ok,
                        total=total,
                        support_ratio=ratio,
                        median_abs_err=med_abs,
                    )

                    # выбираем лучшее по support_ratio, затем по support, затем по median_abs_err
                    if best is None:
                        best = cand
                    else:
                        if cand.support_ratio > best.support_ratio + 1e-9:
                            best = cand
                        elif abs(cand.support_ratio - best.support_ratio) < 1e-9 and cand.support > best.support:
                            best = cand
                        elif abs(cand.support_ratio - best.support_ratio) < 1e-9 and cand.support == best.support and cand.median_abs_err < best.median_abs_err:
                            best = cand

        if best is not None:
            rules.append(best)

    return rules


def _op_human(op: str, a_name: str, b_name: str) -> str:
    a_name = a_name or f"col{a_name}"
    b_name = b_name or f"col{b_name}"
    if op == "add":
        return f"{a_name} + {b_name}"
    if op == "sub_ab":
        return f"{a_name} - {b_name}"
    if op == "sub_ba":
        return f"{b_name} - {a_name}"
    if op == "mul":
        return f"{a_name} × {b_name}"
    if op == "div_ab":
        return f"{a_name} / {b_name}"
    if op == "div_ba":
        return f"{b_name} / {a_name}"
    if op == "pct_ab":
        return f"({a_name} / {b_name}) × 100"
    if op == "pct_ba":
        return f"({b_name} / {a_name}) × 100"
    if op == "pctchg_ab":
        return f"({a_name} / {b_name} − 1) × 100"
    if op == "pctchg_ba":
        return f"({b_name} / {a_name} − 1) × 100"
    return f"{a_name} ? {b_name}"


def _check_inferred_table_formulas(
    chunks: Sequence[EvidenceChunk],
    *,
    max_tables: int = 8,
    max_checks: int = 28,
) -> List[CheckResult]:
    """
    Этап 2: универсальная индукция формул внутри таблиц.
    Находит устойчивые зависимости между колонками и репортит строки-исключения.

    Важно: НЕ пытается "понять экономику", только внутреннюю согласованность таблицы.
    """
    res: List[CheckResult] = []
    tables = [c for c in chunks if (c.element_type or "").lower() == "table" or c.attrs.get("is_table")]
    tables = tables[:max_tables]

    for c in tables:
        if len(res) >= max_checks:
            break

        grid = _parse_table_grid_from_attrs(c.attrs) or None
        if not grid or len(grid) < 4:
            continue

        ocr_quality = (c.attrs or {}).get("ocr_quality") or ""
        # если OCR совсем плох — не строим "жёсткую" индукцию (слишком риск ложных ошибок)
        if (ocr_quality or "").lower() == "low":
            # можно всё равно попытаться, но только для поиска крупных расхождений — пропустим в быстром варианте
            pass

        header_idx, data_start = _header_row_and_data_start(grid)
        n_cols = max(len(r) for r in grid)
        header = [str(x or "") for x in grid[header_idx]]
        header = header + [""] * (n_cols - len(header))

        rules = _infer_rules_for_table(grid, ocr_quality=(ocr_quality or ""), min_support_ratio=0.72, min_support_rows=5)
        if not rules:
            continue

        # применяем правила, репортим исключения
        data_rows = grid[data_start:]
        parsed: List[List[Optional[float]]] = []
        for row in data_rows:
            prow: List[Optional[float]] = []
            for j in range(n_cols):
                cell = str(row[j] if j < len(row) else "")
                prow.append(_cell_to_number(cell))
            parsed.append(prow)

        # ограничим количество ошибок с одной таблицы, чтобы не заспамить
        table_bad = 0
        table_ok_hits = 0

        for rule in rules:
            if len(res) >= max_checks:
                break

            tcol = rule.target_col
            a_col = rule.a_col
            b_col = rule.b_col
            op = rule.op
            t_kind = _infer_kind_from_header(header[tcol])

            t_name = (header[tcol] or f"col{tcol}").strip()
            a_name = (header[a_col] or f"col{a_col}").strip()
            b_name = (header[b_col] or f"col{b_col}").strip()
            formula = _op_human(op, a_name, b_name)

            # пройдём по строкам и найдём расхождения
            for r in range(len(parsed)):
                if len(res) >= max_checks:
                    break
                if table_bad >= 8:  # лимит "плохих" на таблицу
                    break

                a = parsed[r][a_col]
                b = parsed[r][b_col]
                t = parsed[r][tcol]
                if a is None or b is None or t is None:
                    continue

                pred = _op_predict(op, a, b)
                if pred is None:
                    continue

                if _adaptive_close(pred, t, kind=t_kind, ocr_quality=ocr_quality):
                    table_ok_hits += 1
                    continue

                # метка строки — первый столбец (если есть)
                row_label = ""
                try:
                    row0 = str(data_rows[r][0] if len(data_rows[r]) > 0 else "")
                    row_label = re.sub(r"\s+", " ", (row0 or "").strip())
                    if len(row_label) > 60:
                        row_label = row_label[:60].rstrip() + "…"
                except Exception:
                    row_label = ""

                who = f"строка «{row_label}»" if row_label else f"строка #{(data_start + r + 1)}"
                note = ""
                if (ocr_quality or "").lower() in ("low", "mid"):
                    note = " (возможна погрешность распознавания таблицы)"

                res.append(CheckResult(
                    kind="table_infer",
                    ok=False,
                    message=(
                        f"Таблица: в колонке «{t_name}» не сходится вычисление по устойчивой формуле: {formula}. "
                        f"{who}: ожидается {pred:.4g}, но указано {t:.4g}.{note}"
                    ),
                    chunk_id=c.chunk_id,
                    section_path=c.section_path or None,
                ))
                table_bad += 1

            # если по правилу не нашли ошибок и правило сильное — добавим 1 подтверждение (без спама)
            if len(res) < max_checks and table_bad == 0 and rule.support >= 6 and rule.support_ratio >= 0.8:
                res.append(CheckResult(
                    kind="table_infer",
                    ok=True,
                    message=(
                        f"Таблица: обнаружена устойчиво согласованная связь: «{t_name}» ≈ {formula} "
                        f"(подтверждено по строкам: {rule.support}/{rule.total})."
                    ),
                    chunk_id=c.chunk_id,
                    section_path=c.section_path or None,
                ))

        # если вообще не нашли ошибок, но много попаданий — добавим общий ok
        if len(res) < max_checks and table_bad == 0 and table_ok_hits >= 8:
            res.append(CheckResult(
                kind="table_infer",
                ok=True,
                message="Таблица: внутритабличные вычисления в целом выглядят согласованными по выявленным зависимостям.",
                chunk_id=c.chunk_id,
                section_path=c.section_path or None,
            ))

    return res

def _check_percent_sanity(chunks: Sequence[EvidenceChunk], max_checks: int = 20) -> List[CheckResult]:
    """
    Проверка процентов для ВКР (безопаснее):
    1) Если у чанка есть table_grid -> проверяем суммы процентов ПО СТРОКАМ таблицы.
       Это убирает ложные 200/300/700% когда в таблице несколько групп (например "Отцы"/"Матери").
    2) Для текстовых чанков -> режем на строки/предложения и проверяем локально (не весь chunk целиком).
    3) Не считаем "проценты изменений" (на X%, увеличилось на..., снизилось на...) как распределение.
    4) Если видно признаки нескольких оснований/групп в одной строке -> не объявляем арифметическую ошибку,
       максимум мягкая заметка.
    """
    res: List[CheckResult] = []

    def _extract_percents(text: str) -> List[float]:
        ps: List[float] = []
        for m in _PERCENT_RE.finditer(text or ""):
            p = m.group("p").replace(",", ".")
            try:
                ps.append(float(p))
            except Exception:
                pass
        return ps

    def _is_distribution_context(text: str, section_path: str = "") -> bool:
        low = (text or "").lower()
        sp = (section_path or "").lower()
        keys = (
            "структур", "удельн", "доля", "распредел", "состав", "в %", "процент",
            "соотнош", "част", "категор", "группа", "респонд", "опрос",
        )
        ctx = low + " " + sp
        return any(k in ctx for k in keys)

    def _looks_like_change_percent(text: str) -> bool:
        """
        Проценты-изменения НЕ суммируем как структуру:
        "увеличилось на 5%", "снизилось на 3%", "на 10% больше", "по сравнению ... на X%"
        """
        low = (text or "").lower()
        change_keys = (
            "увелич", "сниз", "возрос", "уменьш", "на ", "по сравнению", "больше", "меньше",
            "темп роста", "темп прироста", "прирост", "измен",
        )
        if not any(k in low for k in change_keys):
            return False
        # типичные конструкции "на 5%" / "на 5,2%"
        if re.search(r"\bна\s+\d+(?:[.,]\d+)?\s*%\b", low):
            return True
        # "увеличилось ... %", "снизилось ... %"
        if re.search(r"(увелич|сниз|возрос|уменьш)\w*\s+.*?\d+(?:[.,]\d+)?\s*%", low):
            return True
        return False

    def _has_multi_base_markers(text: str) -> bool:
        """
        Маркеры, что в одном фрагменте смешаны несколько групп/оснований.
        """
        low = (text or "").lower()
        return any(k in low for k in ("отцы", "матери", "всего", "итого", "среди", "группа", "категор", "по ", "в то время как"))

    def _append(kind_ok: bool, msg: str, c: EvidenceChunk, *, section_path_override: Optional[str] = None) -> None:
        nonlocal res
        if len(res) >= max_checks:
            return
        res.append(CheckResult(
            kind="percent",
            ok=kind_ok,
            message=msg,
            chunk_id=c.chunk_id,
            section_path=(section_path_override if section_path_override is not None else (c.section_path or None)),
        ))


    def _eval_percent_sum(ps: List[float], *, is_table_row: bool, c: EvidenceChunk, row_idx: Optional[int] = None) -> None:
        if len(res) >= max_checks:
            return

        # ловим невозможные значения
        bad_values = [p for p in ps if p < -0.5 or p > 100.5]
        if bad_values:
            _append(
                False,
                f"В распределительном фрагменте встречаются подозрительные проценты: "
                f"{', '.join(f'{x:.2f}%' for x in bad_values)}. Проверьте источник/ввод.",
                c,
            )
            return


        s = float(sum(ps))

        tol_ok_low = 98.0
        tol_ok_high = 102.0
        if len(ps) >= 6:
            tol_ok_low = 97.0
            tol_ok_high = 103.0

        sp_has_row = bool(re.search(r"\[row\s+\d+\]", (c.section_path or ""), flags=re.IGNORECASE))
        sp2 = c.section_path or None
        if is_table_row and row_idx is not None and not sp_has_row:
            sp2 = (c.section_path or "").rstrip() + f" [row {row_idx}]"

        if s > 110.0:
            _append(
                False,
                f"Сумма процентов в распределении выглядит завышенной: {s:.2f}%. Вероятен двойной счёт/ошибка основания/смешение групп.",
                c,
                section_path_override=sp2,
            )
        elif tol_ok_low <= s <= tol_ok_high:
            _append(
                True,
                f"Сумма процентов в распределении выглядит согласованной: {s:.2f}% (возможны округления).",
                c,
                section_path_override=sp2,
            )
        elif s < 80.0:
            _append(
                True,
                f"Сумма процентов заметно меньше 100%: {s:.2f}%. Возможно, перечислены не все категории/группы или приведён фрагмент структуры.",
                c,
                section_path_override=sp2,
            )
        else:
            _append(
                True,
                f"Сумма процентов отклоняется от 100%: {s:.2f}%. Проверьте округление/полноту перечня категорий.",
                c,
                section_path_override=sp2,
            )


    # --- 1) Сначала таблицы: проверяем ПО СТРОКАМ ---
    for c in chunks:
        if len(res) >= max_checks:
            break

        grid = (c.attrs or {}).get("table_grid")
        if not grid or not isinstance(grid, list) or len(grid) < 3:
            continue

        # контекст распределения смотрим по заголовку/секции
        header_txt = " | ".join(str(x or "") for x in (grid[0] or []))
        looks_distribution_table = _is_distribution_context(header_txt, c.section_path or "") or _is_distribution_context(c.text or "", c.section_path or "")
        if not looks_distribution_table:
            continue

        # строки данных (после header)
        for r_i, row in enumerate(grid[1:], start=1):
            if len(res) >= max_checks:
                break
            row_txt = " | ".join(str(x or "") for x in row)
            ps = _extract_percents(row_txt)
            if len(ps) < 2:
                continue

            # если строка похожа на "проценты изменения" (редко в таблицах, но бывает) — пропускаем
            if _looks_like_change_percent(row_txt):
                continue

            _eval_percent_sum(ps, is_table_row=True, c=c, row_idx=r_i)

    # --- 2) Затем текст: проверяем локально (строки/предложения), а не весь chunk ---
    for c in chunks:
        if len(res) >= max_checks:
            break

        # таблицы уже обработали, чтобы не дублировать
        if (c.attrs or {}).get("table_grid"):
            continue

        text = (c.text or "").strip()
        if not text:
            continue

        # если вообще нет распределительного контекста — не шумим
        if not _is_distribution_context(text, c.section_path or ""):
            continue

        # режем на строки, потом на предложения
        lines = [ln.strip() for ln in re.split(r"[\n\r]+", text) if ln.strip()]
        if not lines:
            continue

        for ln in lines:
            if len(res) >= max_checks:
                break

            if not _is_distribution_context(ln, c.section_path or ""):
                continue

            # проценты-изменения не трогаем
            if _looks_like_change_percent(ln):
                continue

            ps = _extract_percents(ln)
            if len(ps) < 2:
                continue

            # Если в строке слишком много процентов — почти наверняка несколько разных распределений/рисунков
            if len(ps) > 6:
                continue

            # Если явно похоже на смешение групп/оснований — не делаем "красную" арифметическую ошибку
            if _has_multi_base_markers(ln) and len(ps) >= 4:
                s = float(sum(ps))
                if s > 110.0 and len(res) < max_checks:
                    _append(True,
                            f"В одном фрагменте много процентов и похоже на несколько оснований/групп; сумма {s:.2f}% может быть нормальной. Уточните, относятся ли проценты к одной структуре.",
                            c)
                continue

            # обычная проверка по одной строке/фразе
            _eval_percent_sum(ps, is_table_row=False, c=c, row_idx=None)

    return res

# -----------------------------
# Agent orchestration
# -----------------------------

def _format_checks_for_user(checks: Sequence[CheckResult], max_items: int = 18) -> str:
    if not checks:
        return ""

    def _norm_space(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip())

    def _strip_row_suffix(s: str) -> str:
        s = (s or "").strip()
        # убираем маркер row в любом месте строки
        s = re.sub(r"\s*\[row\s+\d+\]\s*", " ", s, flags=re.IGNORECASE)
        # нормализуем пробелы
        s = re.sub(r"\s+", " ", s).strip()
        return s


    def _norm_msg(s: str) -> str:
        s = _norm_space(s)
        s = re.sub(r"\s*\[row\s+\d+\]\s*$", "", s, flags=re.IGNORECASE).strip()
        return s

    def _dedup(items: List[CheckResult]) -> List[CheckResult]:
        seen = set()
        out: List[CheckResult] = []
        for c in items:
            k = ((c.kind or "").strip().lower(), _norm_msg(c.message or ""), _strip_row_suffix(c.section_path or ""))
            if k in seen:
                continue
            seen.add(k)
            out.append(c)
        return out

    bad = [c for c in checks if not c.ok]
    good_all = [c for c in checks if c.ok]

    # NEW: дедуп (чтобы не было повторов одинаковых строк)
    bad = _dedup(bad)
    good_all = _dedup(good_all)

    bad_non_percent = [c for c in bad if (c.kind or "") != "percent"]
    bad_percent = [c for c in bad if (c.kind or "") == "percent"]

    # если есть реальные ошибки — не засоряем “good” выражениями без итога
    if bad:
        good = [c for c in good_all if "без явного итога" not in (c.message or "").lower()]
    else:
        good = [c for c in good_all if "без явного итога" not in (c.message or "").lower()]

    def _ref(c: CheckResult) -> str:
        sp = (c.section_path or "").strip()
        if not sp:
            return ""
        # убираем технический хвост [row N] для красивого ответа
        sp = re.sub(r"\s*\[row\s+\d+\]\s*", " ", sp, flags=re.IGNORECASE)
        sp = re.sub(r"\s+", " ", sp).strip()
        sp = re.sub(r"\s+", " ", sp)
        if len(sp) > 180:
            sp = "…" + sp[-180:]
        return f" (источник: {sp})"


    lines: List[str] = []

    if bad_non_percent:
        lines.append("Найдены потенциальные несостыковки в расчётах:")
        for c in (bad_non_percent + bad_percent)[:max_items]:
            lines.append(f"- {c.message}{_ref(c)}")
    elif bad_percent:
        lines.append("Найдены потенциально спорные фрагменты с процентами (возможны разные основания/группы):")
        for c in bad_percent[:max_items]:
            lines.append(f"- {c.message}{_ref(c)}")

    if good and len(lines) < max_items + 2:
        lines.append("")
        lines.append("Что удалось подтвердить пересчётом:")
        take_n = max(0, max_items - min(len(bad), max_items))
        for c in good[:take_n]:
            lines.append(f"- {c.message}{_ref(c)}")

    return "\n".join(lines).strip()


def _wants_llm_narration(question: str) -> bool:
    """
    Включаем LLM только если пользователь явно просит "объяснить красиво/с выводами".
    """
    q = (question or "").lower()
    triggers = (
        "сформулируй красиво", "красивый ответ", "объясни подробно", "подробный вывод",
        "сделай вывод", "напиши вывод", "развернутый анализ",
    )
    return any(t in q for t in triggers)


def _looks_like_year(x: float) -> bool:
    try:
        xi = int(round(float(x)))
    except Exception:
        return False
    return 1900 <= xi <= 2100


def _looks_like_small_index(x: float) -> bool:
    # типичные индексы/номера строк/пунктов: 0..200
    try:
        xf = float(x)
    except Exception:
        return False
    return abs(xf) <= 200.0


def _looks_like_non_math_identifier(raw: str) -> bool:
    """
    Отсекаем “62.02”, “2.3”, “4.1.1”, “2023” и т.п., когда это похоже на идентификатор,
    а не на расчётную величину.
    """
    s = (raw or "").strip()
    if not s:
        return True

    # явные бухгалтерские счета/разделы: много точек
    if re.fullmatch(r"\d+(?:\.\d+){1,}", s):
        return True

    # годы как “операнды” (особенно в выражениях)
    try:
        v = float(s.replace(" ", "").replace(",", "."))
        if _looks_like_year(v):
            return True
    except Exception:
        pass

    return False


def _format_checks_pretty(checks: Sequence[CheckResult], *, max_bad: int = 8, max_good: int = 4) -> str:
    """
    User-facing формат без chunk_id, с краткими источниками.
    Важно: если "плохие" пункты только из percent-check — делаем мягкий вывод, не "арифметические несоответствия".
    """
    if not checks:
        return ""

    def _norm_space(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip())

    def _strip_row_suffix(s: str) -> str:
        s = (s or "").strip()
        # убираем маркер row в любом месте строки
        s = re.sub(r"\s*\[row\s+\d+\]\s*", " ", s, flags=re.IGNORECASE)
        # нормализуем пробелы
        s = re.sub(r"\s+", " ", s).strip()
        return s


    def _norm_msg(s: str) -> str:
        # нормализуем message и убираем хвост row, если он вдруг есть
        s = _norm_space(s)
        s = re.sub(r"\s*\[row\s+\d+\]\s*$", "", s, flags=re.IGNORECASE).strip()
        return s

    def _dedup(items: List[CheckResult]) -> List[CheckResult]:
        seen = set()
        out: List[CheckResult] = []
        for c in items:
            k = ((c.kind or "").strip().lower(), _norm_msg(c.message or ""), _strip_row_suffix(c.section_path or ""))
            if k in seen:
                continue
            seen.add(k)
            out.append(c)
        return out

    bad = [c for c in checks if not c.ok]
    good = [c for c in checks if c.ok]

    # убираем “формулы без итога” из good — это почти всегда шум
    good = [c for c in good if "без явного итога" not in (c.message or "").lower()]

    # NEW: дедуп по смыслу (убираем дубли из-за [row N] и повторов evidence)
    bad = _dedup(bad)
    good = _dedup(good)

    def src(c: CheckResult) -> str:
        sp = (c.section_path or "").strip()
        if not sp:
            return ""
        sp = _strip_row_suffix(sp)  # убираем [row N]
        sp = _norm_space(sp)
        if len(sp) > 220:
            sp = "…" + sp[-220:]
        return f" (источник: {sp})"


    # классификация плохих
    bad_non_percent = [c for c in bad if (c.kind or "") != "percent"]
    bad_percent = [c for c in bad if (c.kind or "") == "percent"]

    lines: List[str] = []

    if bad_non_percent:
        lines.append("Вывод: обнаружены арифметические несоответствия. Рекомендую перепроверить указанные формулы/ячейки.")
        lines.append("")
        lines.append("Что не сходится:")
        for c in (bad_non_percent + bad_percent)[:max_bad]:
            lines.append(f"- {c.message}{src(c)}")
    elif bad_percent:
        lines.append("Вывод: обнаружены потенциально спорные фрагменты с процентами (возможны разные основания/группы).")
        lines.append("Это не обязательно ошибка в расчётах, но стоит перепроверить, что проценты относятся к одной структуре.")
        lines.append("")
        lines.append("Что требует внимания:")
        for c in bad_percent[:max_bad]:
            lines.append(f"- {c.message}{src(c)}")
    else:
        lines.append("Вывод: явных арифметических несоответствий автоматическая проверка не обнаружила.")
        lines.append("Это не гарантирует корректность всех формул, но базовые проверки не выявили ошибок.")

    if good:
        lines.append("")
        lines.append("Что удалось подтвердить пересчётом:")
        for c in good[:max_good]:
            lines.append(f"- {c.message}{src(c)}")

    out = "\n".join(lines).strip()
    # железобетонный UX-контракт: row не должен появляться нигде
    out = re.sub(r"\[row\s+\d+\]", "", out, flags=re.IGNORECASE)
    out = re.sub(r"\s{2,}", " ", out).strip()
    return out



def _build_llm_messages(
    question: str,
    evidence: Sequence[EvidenceChunk],
    checks: Sequence["CheckResult"],
) -> List[Dict[str, str]]:
    """
    LLM-нарратив поверх evidence+checks.
    Важно: НЕ отправляем LLM технические метки вида [row N].
    Важно: НЕ раздуваем prompt (иначе ответ будет обрываться по лимиту).
    """

    # --- лимиты prompt (ключевое) ---
    MAX_PROMPT_CHARS = 14000   # общий бюджет на user.content
    MAX_CHUNKS = 14            # максимум фрагментов контекста
    MAX_TXT = 700              # максимум символов на фрагмент текста
    MAX_TABLE_ROWS = 12        # максимум строк из table_grid
    MAX_TABLE_COLS = 8         # максимум колонок из table_grid

    def _short_path(sp: str) -> str:
        sp = _clean_section_path_for_user(sp)  # должен убирать [row N]
        sp = re.sub(r"\s+", " ", (sp or "")).strip()
        if len(sp) > 220:
            sp = "…" + sp[-220:]
        return sp

    def _clip(s: str, n: int) -> str:
        s = (s or "").strip()
        if len(s) <= n:
            return s
        return s[:n].rstrip() + "…"

    def _table_preview_from_grid(e: EvidenceChunk) -> str:
        grid = (e.attrs or {}).get("table_grid")
        if not isinstance(grid, list) or not grid:
            return ""
        rows = []
        for r in grid[:MAX_TABLE_ROWS]:
            if not isinstance(r, list):
                continue
            cells = [str(x or "").strip() for x in r[:MAX_TABLE_COLS]]
            rows.append(" | ".join(cells))
        out = "\n".join(rows).strip()
        if not out:
            return ""
        return _clip(out, MAX_TXT)

    # 1) Сначала проверки: все BAD + немного OK
    bad_checks = [c for c in checks if not getattr(c, "ok", False)]
    ok_checks = [c for c in checks if getattr(c, "ok", False)]

    checks_lines: List[str] = []
    for c in (bad_checks[:18] + ok_checks[:6]):
        kind = (c.kind or "").strip()
        ok = "OK" if getattr(c, "ok", False) else "BAD"
        msg = (c.message or "").strip()
        sp = _short_path(c.section_path or "")
        if sp:
            checks_lines.append(f"- [{ok}] ({kind}) {msg} | источник: {sp}")
        else:
            checks_lines.append(f"- [{ok}] ({kind}) {msg}")

    checks_text = "\n".join(checks_lines) if checks_lines else "(нет)"

    # 2) Приоритизация evidence по BAD-источникам + по маркерам выборки
    bad_sources = set(_short_path(c.section_path or "") for c in bad_checks if (c.section_path or "").strip())

    def _is_ctx_chunk(e: EvidenceChunk) -> bool:
        low = (e.text or "").lower()
        sp = _short_path(e.section_path or "")
        if any(k in low for k in ("выборк", "n=", "n =", "респонд", "отцов", "матер")):
            return True
        if sp and sp in bad_sources:
            return True
        return False

    prioritized = [e for e in evidence if _is_ctx_chunk(e)]
    # rest: берём остальные через ключи (без зависимости от __eq__)
    def _ekey(e: EvidenceChunk) -> tuple:
        txt = re.sub(r"\s+", " ", (e.text or "")).strip()
        return ((e.element_type or "").lower(), _short_path(e.section_path or ""), txt[:240])

    pkeys = { _ekey(e) for e in prioritized }
    rest = [e for e in evidence if _ekey(e) not in pkeys]

    # дедуп уже на выбранных
    chosen_raw = (prioritized + rest)
    seen = set()
    chosen: List[EvidenceChunk] = []
    for e in chosen_raw:
        k = _ekey(e)
        if k in seen:
            continue
        seen.add(k)
        chosen.append(e)
        if len(chosen) >= MAX_CHUNKS:
            break

    system = (
        "Ты — аналитик, который проверяет реалистичность расчётов в тексте ВКР.\n"
        "Правила:\n"
        "1) Опирайся строго на контекст и на детерминированные проверки.\n"
        "2) НЕ используй и НЕ выводи технические метки вида [row N].\n"
        "3) Не пиши длинные цитаты. Если нужна опора — 1 короткий фрагмент (до 15–20 слов) и источник.\n"
        "4) Если находишь проблему — объясни, в чём именно несостыковка (арифметика/знаменатель/масштаб/логика групп).\n"
        "5) Структура ответа:\n"
        "   - Краткий вывод (1–3 предложения)\n"
        "   - Затем: 'Проверки и замечания' (буллеты)\n"
        "   - Затем: 'Что проверить в документе' (2–5 конкретных шагов)\n"
    )

    # 3) Формируем контекст с жёстким бюджетом
    ctx_parts: List[str] = []
    for e in chosen:
        sp = _short_path(e.section_path or "")
        et = (e.element_type or "").strip()
        txt = (e.text or "").strip()

        # таблицы: если есть grid — добавляем превью
        if (et or "").lower() == "table":
            preview = _table_preview_from_grid(e)
            if preview:
                txt = "Таблица (фрагмент):\n" + preview

        if not txt:
            continue
        txt = _clip(txt, MAX_TXT)

        header = f"Источник: {sp}" if sp else "Источник: (неизвестно)"
        if et:
            header += f" | тип: {et}"
        part = header + "\n" + txt

        # проверяем общий бюджет
        if ctx_parts:
            candidate = "\n\n---\n\n".join(ctx_parts + [part])
        else:
            candidate = part

        # user message ещё включает checks_text и обвязку — оставим запас
        if len(candidate) > (MAX_PROMPT_CHARS - 3500):
            break

        ctx_parts.append(part)

    user = (
        f"Вопрос пользователя: {question}\n\n"
        f"Результаты детерминированных проверок:\n{checks_text}\n\n"
        "[КОНТЕКСТ ИЗ ДОКУМЕНТА]\n"
        + ("\n\n---\n\n".join(ctx_parts) if ctx_parts else "(контекст не найден)")
        + "\n\nСформируй ответ строго по структуре из system."
    )

    # финальный предохранитель: если всё равно раздулось — режем контекст хвостом
    if len(user) > MAX_PROMPT_CHARS:
        user = user[:MAX_PROMPT_CHARS].rstrip() + "…"

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


async def _enrich_tables_with_ocr(
    owner_id: int,
    doc_id: int,
    evidence: List[EvidenceChunk],
    *,
    max_tables: int = 3,
    max_rows: int = 60,
    max_cols: int = 16,
) -> List[EvidenceChunk]:
    """
    В calc-режиме: если таблица была вставлена как изображение и не дала table_grid/table_tsv,
    пытаемся восстановить её через app.vision_tables_ocr.ocr_table_section().

    Безопасность:
    - обрабатываем максимум max_tables таблиц на один вопрос
    - если OCR/Vision недоступны или провалились — просто возвращаем evidence как есть
    """
    try:
        from app.vision_tables_ocr import ocr_table_section  # type: ignore
    except Exception:
        return evidence

    def _base_section_path(sp: str) -> str:
        # убираем хвост " [row N]" если он есть
        return re.sub(r"\s*\[row\s+\d+\]\s*$", "", (sp or "").strip(), flags=re.IGNORECASE)

    def _looks_like_table_chunk(c: EvidenceChunk) -> bool:
        et = (c.element_type or "").lower().strip()
        if et in ("table", "table_row"):
            return True
        if c.attrs.get("is_table"):
            return True
        low_sp = (c.section_path or "").lower()
        low_tx = (c.text or "").lower()
        return ("таблица" in low_sp) or ("таблица" in low_tx) or ("[таблица]" in low_sp)

    def _has_grid(c: EvidenceChunk) -> bool:
        if not c.attrs:
            return False
        g = c.attrs.get("table_grid")
        if isinstance(g, list) and g:
            return True
        tsv = c.attrs.get("table_tsv")
        return isinstance(tsv, str) and bool(tsv.strip())

    def _csv_to_grid(csv_text: str) -> Optional[List[List[str]]]:
        if not csv_text or not csv_text.strip():
            return None
        try:
            f = StringIO(csv_text)
            reader = csv.reader(f)
            rows = []
            for r in reader:
                rr = [str(x or "").strip() for x in r]
                if any(x for x in rr):
                    rows.append(rr)
            if not rows:
                return None
            # ограничим размер
            rows = rows[:max_rows]
            rows = [r[:max_cols] for r in rows]
            return rows
        except Exception:
            return None

    def _grid_to_tsv(grid: List[List[str]]) -> str:
        out_lines = []
        for r in grid:
            rr = [str(x or "").strip() for x in r[:max_cols]]
            out_lines.append("\t".join(rr))
        return "\n".join(out_lines)

    updated = evidence[:]  # копия списка
    touched = 0
    seen_sections = set()

    for idx, c in enumerate(updated):
        if touched >= max_tables:
            break
        if _has_grid(c):
            continue
        if not _looks_like_table_chunk(c):
            continue

        base_sp = _base_section_path(c.section_path or "")
        if not base_sp or base_sp in seen_sections:
            continue
        seen_sections.add(base_sp)

        # OCR/vision-вызываем в отдельном треде (функция синхронная)
        try:
            card = await asyncio.to_thread(ocr_table_section, owner_id, doc_id, base_sp, lang="ru", prefer_markdown=True)
        except Exception:
            continue

        csv_text = (card or {}).get("csv") or ""
        grid = _csv_to_grid(csv_text)
        if not grid:
            # если CSV пуст, можно попробовать markdown (но это сложнее и шумнее) — пропускаем
            continue

                # --- OCR quality scoring (простая эвристика) ---
        total_cells = 0
        nonempty_cells = 0
        numeric_cells = 0

        for rr in grid[:max_rows]:
            for cell in rr[:max_cols]:
                total_cells += 1
                t = (str(cell or "")).strip()
                if t:
                    nonempty_cells += 1
                    if _cell_to_number(t) is not None:
                        numeric_cells += 1

        nonempty_ratio = nonempty_cells / max(1, total_cells)
        numeric_ratio = numeric_cells / max(1, nonempty_cells)

        # пороги грубые, но рабочие
        if nonempty_ratio < 0.25 or numeric_ratio < 0.20:
            ocr_quality = "low"
        elif nonempty_ratio < 0.45 or numeric_ratio < 0.35:
            ocr_quality = "mid"
        else:
            ocr_quality = "high"


        # добавляем в attrs того чанка, на котором “сидит” таблица
        new_attrs = dict(c.attrs or {})
        new_attrs["table_grid"] = grid
        new_attrs["table_tsv"] = _grid_to_tsv(grid)
        new_attrs["ocr_table"] = True
        new_attrs["ocr_quality"] = ocr_quality
        new_attrs["ocr_stats"] = {
            "nonempty_ratio": round(nonempty_ratio, 3),
            "numeric_ratio": round(numeric_ratio, 3),
            "cells": int(total_cells),
        }

        note = (card or {}).get("note") or ""
        if note:
            new_attrs["ocr_note"] = str(note)

        updated[idx] = EvidenceChunk(
            chunk_id=c.chunk_id,
            element_type=c.element_type,
            section_path=c.section_path,
            text=c.text,
            attrs=new_attrs,
        )
        touched += 1

    return updated

_SAMPLE_N_RE = re.compile(r"(?i)\bN\s*=\s*(\d{1,4})\b")
_SAMPLE_WORD_RE = re.compile(r"(?i)\b(выборк[аи]|n)\s*(?:составля(ет|ла)|=|—|:)?\s*(\d{1,4})\b")
_GROUP_RE = re.compile(
    r"(?i)\b(\d{1,4})\s*(?:чел\.?|человек|респондентов|испытуемых|участников|родител(?:ей|и))\b"
)
_GROUP_NAMED_RE = re.compile(
    r"(?i)\b(\d{1,4})\s*(матер(?:ей|и)?|отц(?:ов|ы)?|женщин|мужчин)\b"
)

def _canonical_group_name(raw: str) -> Optional[str]:
    s = (raw or "").strip().lower()
    if not s:
        return None
    if "мат" in s or "женщ" in s:
        return "матери"
    if "отц" in s or "муж" in s:
        return "отцы"
    return None

def _extract_sample_context(evidence: Sequence[EvidenceChunk]) -> SampleContext:
    """
    Этап 4 (extraction):
    Пытаемся извлечь объём выборки N и размеры подгрупп из текста/заголовков.
    Работает “эвристически”, но закрывает 80% ВКР формата: N=40, 20 матерей и 20 отцов, и т.п.
    """
    total_n: Optional[int] = None
    groups: Dict[str, int] = {}
    notes: List[str] = []

    def _add_note(c: EvidenceChunk, msg: str) -> None:
        sp = (c.section_path or "").strip()
        sp = re.sub(r"\s+", " ", sp)
        if len(sp) > 140:
            sp = "…" + sp[-140:]
        notes.append(f"{msg} (источник: {sp})" if sp else msg)

    # 1) ищем N=...
    for c in evidence:
        t = c.text or ""
        m = _SAMPLE_N_RE.search(t)
        if m:
            try:
                n = int(m.group(1))
                if 2 <= n <= 5000:
                    total_n = total_n or n
                    _add_note(c, f"Найдено N={n}")
            except Exception:
                pass

    # 2) ищем "выборка ... 40"
    if total_n is None:
        for c in evidence:
            t = c.text or ""
            m = _SAMPLE_WORD_RE.search(t)
            if m:
                try:
                    n = int(m.group(3))
                    if 2 <= n <= 5000:
                        total_n = n
                        _add_note(c, f"Найден объём выборки {n}")
                        break
                except Exception:
                    pass

    # 3) ищем именованные подгруппы: "20 матерей", "20 отцов" и т.п.
    for c in evidence:
        t = c.text or ""
        for m in _GROUP_NAMED_RE.finditer(t):
            try:
                n = int(m.group(1))
                g = _canonical_group_name(m.group(2))
                if g and 2 <= n <= 5000:
                    if g not in groups:
                        groups[g] = n
                        _add_note(c, f"Найдена подгруппа: {g}={n}")
            except Exception:
                pass

    # 4) если есть две подгруппы, и total_n не найден — попробуем восстановить как сумму
    if total_n is None and groups:
        s = sum(groups.values())
        if 2 <= s <= 5000:
            total_n = s
            notes.append(f"Общий N восстановлен как сумма подгрупп: {s}")

    return SampleContext(total_n=total_n, groups=groups, notes=notes)


def _cell_percents(s: str) -> List[float]:
    ps: List[float] = []
    for m in _PERCENT_RE.finditer(s or ""):
        p = m.group("p").replace(",", ".")
        try:
            ps.append(float(p))
        except Exception:
            pass
    return ps

def _iter_table_percent_rows(c: EvidenceChunk):
    """
    Генератор по строкам table_grid, где есть проценты.
    Возвращает (row_idx, row_label, percents, header_labels)
    row_idx — индекс строки в grid (0..)
    """
    grid = (c.attrs or {}).get("table_grid")
    if not isinstance(grid, list) or not grid:
        return
    header = grid[0] if isinstance(grid[0], list) else []
    header_labels = [str(x or "").strip() for x in header]

    for i, row in enumerate(grid[1:], start=1):
        if not isinstance(row, list) or not row:
            continue
        row_label = str(row[0] or "").strip()
        row_text = " ".join(str(x or "") for x in row)
        ps = _cell_percents(row_text)
        if len(ps) >= 2:
            yield i, row_label, ps, header_labels


def _choose_denominator_for_row(row_label: str, c: EvidenceChunk, ctx: SampleContext) -> Optional[int]:
    """
    Выбираем N для строки:
    - если строка/контекст явно про "отцы/матери/мужчины/женщины" и ctx.groups содержит это — берём подгруппу
    - иначе берём общий ctx.total_n
    """
    label = (row_label or "").lower()
    sp = (c.section_path or "").lower()
    tx = (c.text or "").lower()
    combined = f"{label} {sp} {tx}"

    for k, n in (ctx.groups or {}).items():
        if k in combined:
            return n

    # доп. эвристика: отцы/матери без нормализации
    if "отц" in combined and "отцы" in (ctx.groups or {}):
        return ctx.groups["отцы"]
    if "матер" in combined and "матери" in (ctx.groups or {}):
        return ctx.groups["матери"]

    return ctx.total_n


def _check_percent_denominator_consistency(
    evidence: Sequence[EvidenceChunk],
    ctx: SampleContext,
    *,
    max_checks: int = 18,
) -> List[CheckResult]:
    """
    Этап 4A:
    Ловим проценты, которые математически невозможны как доли людей при заданном N
    (например, N=20 => шаг 5%, поэтому 1% и 74% невозможны как 'процент респондентов').

    Важно: формулируем как "если это доля людей", чтобы не ругаться на проценты от баллов.
    """
    res: List[CheckResult] = []
    if not ctx or not (ctx.total_n or ctx.groups):
        return res

    for c in evidence:
        if len(res) >= max_checks:
            break

        # работаем в первую очередь по таблицам
        grid = (c.attrs or {}).get("table_grid")
        if not isinstance(grid, list) or len(grid) < 2:
            continue

        low_ctx = ((c.text or "") + " " + (c.section_path or "")).lower()
        ambiguous_base = any(k in low_ctx for k in (
            "балл", "баллов", "сумма баллов", "средн", "среднее значение", "процентная выраженность"
        ))

        for row_idx, row_label, ps, _hdr in _iter_table_percent_rows(c):
            if len(res) >= max_checks:
                break

            denom = _choose_denominator_for_row(row_label, c, ctx)
            if not denom or denom < 2 or denom > 5000:
                continue

            step = 100.0 / float(denom)
            # для очень больших N шаг слишком мал, проверка теряет смысл
            if step < 0.25:
                continue

            # допускаем чуть больше погрешности, чем step*0.3, но не меньше 0.7%
            tol = max(0.35, step * 0.15)

            bad_ps: List[float] = []
            for p in ps:
                if p < -0.5 or p > 100.5:
                    bad_ps.append(p)
                    continue
                k = int(round(p / step))
                recon = k * step
                if abs(p - recon) > tol:
                    bad_ps.append(p)

            if bad_ps:
                base_hint = (
                    "Если проценты считаются как доли респондентов,"
                    if not ambiguous_base
                    else "Если проценты интерпретируются как доли респондентов (а не доли баллов/выборов),"
                )
                msg = (
                    f"{base_hint} при N={denom} шаг составляет {step:.2f}%, поэтому значения "
                    f"{', '.join(f'{x:.2f}%' for x in bad_ps)} выглядят математически невозможными/сомнительными. "
                    f"Проверьте знаменатель (N) и метод подсчёта."
                )
                res.append(CheckResult(
                    kind="denominator",
                    ok=False,
                    message=msg,
                    chunk_id=c.chunk_id,
                    section_path=c.section_path or None,
                ))

    return res


def _find_text_total_distribution(evidence: Sequence[EvidenceChunk], want_k: int) -> Optional[tuple]:
    """
    Ищем в текстовых чанках распределение из want_k процентов (например, 3 числа 29/46/25),
    похожее на "по выборке/в целом/всего".
    Возвращаем (percents_list, section_path).
    """
    keys = ("в целом", "по выборке", "всего", "общ", "итог", "в сумме", "суммар")
    for c in evidence:
        txt = (c.text or "")
        low = txt.lower()
        if not any(k in low for k in keys):
            continue

        ps = _cell_percents(txt)
        if len(ps) == want_k:
            return ps, (c.section_path or "")
    return None


def _check_subgroup_to_total_consistency(
    evidence: Sequence[EvidenceChunk],
    ctx: SampleContext,
    *,
    max_checks: int = 10,
) -> List[CheckResult]:
    """
    Этап 4B:
    Если есть подгруппы (например, отцы=20, матери=20) и таблица с распределениями по подгруппам,
    пересчитываем общий итог (N=40) и сверяем с текстовым итогом (если он есть).
    """
    res: List[CheckResult] = []
    if not ctx or not ctx.total_n or not ctx.groups:
        return res

    total_n = ctx.total_n
    if total_n < 2 or total_n > 5000:
        return res

    # по каждой таблице ищем распределение по 2+ подгруппам
    for c in evidence:
        if len(res) >= max_checks:
            break

        grid = (c.attrs or {}).get("table_grid")
        if not isinstance(grid, list) or len(grid) < 3:
            continue

        # собираем строки-подгруппы
        group_rows = []
        for row_idx, row_label, ps, hdr in _iter_table_percent_rows(c):
            denom = _choose_denominator_for_row(row_label, c, ctx)
            if denom and denom in ctx.groups.values():
                group_rows.append((row_label, denom, ps, hdr))

        if len(group_rows) < 2:
            continue

        # берём минимальную длину распределения (чтобы не ломаться на мусорных процентах)
        k = min(len(gr[2]) for gr in group_rows)
        if k < 2:
            continue

        # пересчитываем counts по каждой подгруппе
        total_counts = [0] * k
        ok_row_count = 0

        for row_label, denom, ps, _hdr in group_rows:
            step = 100.0 / float(denom)
            tol = max(0.35, step * 0.15)

            counts = []
            row_ok = True
            for p in ps[:k]:
                kk = int(round(p / step))
                recon = kk * step
                if abs(p - recon) > tol:
                    row_ok = False
                    break
                counts.append(kk)

            if not row_ok:
                continue

            ok_row_count += 1
            for i in range(k):
                total_counts[i] += counts[i]

        if ok_row_count < 2:
            continue

        # переводим в проценты по total_n
        total_ps = [100.0 * cnt / float(total_n) for cnt in total_counts]

        # пытаемся найти текстовый итог (29/46/25) или иной общий результат
        found = _find_text_total_distribution(evidence, want_k=k)
        if not found:
            continue

        stated_ps, stated_sp = found

        # сравнение с допуском округления ~1.5%
                # сравнение с допуском округления
        diffs = [abs(a - b) for a, b in zip(total_ps, stated_ps)]

        # допуск привязываем к шагу общей выборки:
        # N=40 => шаг 2.5%. "29%" не является округлением от 27.5% (ожидали бы 28% или 30%).
        step_total = 100.0 / float(total_n)
        tol_total = max(0.6, step_total * 0.4)  # для N=40 это 1.0%

        # доп. критерий: кратность заявленных процентов шагу 100/N (если это доли людей)
        def _is_multiple_of_step(p: float, step: float, tol: float) -> bool:
            k = int(round(p / step))
            recon = k * step
            return abs(p - recon) <= tol

        stated_not_multiple = [p for p in stated_ps if not _is_multiple_of_step(p, step_total, tol_total)]

        if any(d > tol_total for d in diffs) or stated_not_multiple:
            extra = ""
            if stated_not_multiple:
                extra = (
                    f" Дополнительно: при N={total_n} шаг составляет {step_total:.2f}%, поэтому значения "
                    f"{', '.join(f'{x:.1f}%' for x in stated_not_multiple)} некратны шагу и не выглядят долями людей."
                )

            msg = (
                f"Есть несоответствие между распределением по подгруппам и общим итогом по выборке. "
                f"По подгруппам (N={total_n}) получается примерно: "
                f"{', '.join(f'{x:.1f}%' for x in total_ps)}, "
                f"а в тексте указано: {', '.join(f'{x:.1f}%' for x in stated_ps)}."
                f"{extra} Проверьте базу (N) и пересчёт."
            )
            res.append(CheckResult(
                kind="subgroup_total",
                ok=False,
                message=msg,
                chunk_id=c.chunk_id,
                section_path=(stated_sp or c.section_path or None),
            ))


    return res


async def answer_calc_question(
    uid: int,
    doc_id: int,
    question: str,
    *,
    chat_with_gpt: Optional[ChatFn] = None,
    max_chunks: int = 140,
) -> Optional[str]:
    if not is_calc_question(question):
        return None

    # 1) candidates from chunks
    evidence = await asyncio.to_thread(_select_calc_candidate_chunks, uid, doc_id, max_chunks)

    # 1b) fallback from document_sections if мало/нет кандидатов
    if not evidence or len(evidence) < 5:
        sec_ev = await asyncio.to_thread(_select_calc_candidate_sections, uid, doc_id, max(220, max_chunks))
        if sec_ev:
            def _norm_key(x: EvidenceChunk) -> tuple:
                txt = (x.text or "").strip()
                txt = re.sub(r"\s+", " ", txt)
                # ключ устойчивее: тип + путь + префикс текста
                return ((x.element_type or "").lower(), (x.section_path or "").strip(), txt[:400])

            seen = {_norm_key(e) for e in evidence}
            for e in sec_ev:
                k = _norm_key(e)
                if k not in seen:
                    evidence.append(e)
                    seen.add(k)

    if not evidence:
        return (
            "Не нашёл в документе расчётных фрагментов (таблиц/формул/числовых абзацев), которые можно проверить. "
            "Проверьте, что таблицы корректно проиндексированы (особенно если они как изображение), либо уточните, "
            "в каком разделе/таблице находятся расчёты."
        )

    # 1c) enrichment: если таблицы могли быть картинками — пробуем OCR/Vision (только в calc-режиме)
    evidence = await _enrich_tables_with_ocr(uid, doc_id, evidence, max_tables=3)

    # 2) checks (сначала самые “сильные” и менее шумные)
    checks: List[CheckResult] = []
    checks.extend(_check_table_totals(evidence, max_tables=12))
    checks.extend(_check_delta_and_growth_columns(evidence, max_tables=10, max_checks=24))
    checks.extend(_check_inferred_table_formulas(evidence, max_tables=8, max_checks=28))

    # Этап 3: межтабличные противоречия и sanity по динамике
    checks.extend(_check_cross_table_consistency(evidence, max_tables=10, max_checks=18))
    checks.extend(_check_trend_outliers(evidence, max_tables=10, max_checks=12))

    # Этап 4: знаменатель (N) и согласование подгрупп
    sample_ctx = _extract_sample_context(evidence)
    checks.extend(_check_percent_denominator_consistency(evidence, sample_ctx, max_checks=18))
    checks.extend(_check_subgroup_to_total_consistency(evidence, sample_ctx, max_checks=10))

    checks.extend(_check_revenue_cost_profit(evidence, max_checks=24))
    checks.extend(_check_simple_formulas(evidence, max_checks=30))
    checks.extend(_check_percent_sanity(evidence, max_checks=10))


    # 3) decide LLM
    allow_llm = (
        chat_with_gpt is not None
        and (_cfg_calc_agent_use_llm() or _wants_llm_narration(question))
    )

    # “красивый” детерминированный отчёт (без chunk_id, с источниками)
    pretty = _format_checks_pretty(checks, max_bad=8, max_good=4)

    # 4) no LLM -> pretty deterministic, fallback to raw only if pretty пуст
    if not allow_llm:
        if pretty:
            return pretty
        raw = _format_checks_for_user(checks, max_items=18)
        if raw:
            return raw
        return (
            "Я нашёл числовые фрагменты, но не смог уверенно подтвердить несоответствия автоматическими правилами. "
            "Уточните, какие именно показатели проверить (например: “таблица X: выручка/расходы/прибыль” или “итого по столбцу”)."
        )

    # 5) LLM narration over evidence+checks
    messages = _build_llm_messages(question, evidence, checks)
    try:
        ans = await asyncio.to_thread(chat_with_gpt, messages, temperature=0.0, max_tokens=1800)
        ans = _strip_row_markers((ans or "").strip())

        if ans:
            if _looks_truncated_for_user(ans):
                # ретрай 1 раз: уменьшаем evidence (чтобы ответ не обрубало)
                ev2 = list(evidence)[:max(6, min(10, len(evidence)))]
                messages2 = _build_llm_messages(question, ev2, checks)
                ans2 = await asyncio.to_thread(chat_with_gpt, messages2, temperature=0.0, max_tokens=1800)
                ans2 = _strip_row_markers((ans2 or "").strip())
                if ans2 and not _looks_truncated_for_user(ans2):
                    ans = ans2
                else:
                    # если и второй раз обрыв — только тогда pretty
                    return _strip_row_markers(pretty or ans)

            bad_cnt = sum(1 for c in checks if not c.ok)
            norm = ans.lower()
            claims_error = any(w in norm for w in ("ошиб", "не сход", "несоответ", "некоррект", "нереалист"))
            if bad_cnt == 0 and claims_error:
                return _strip_row_markers(pretty or ans)

            return ans

    except Exception:
        pass


    final = pretty or _format_checks_for_user(checks, max_items=18) or (
    "Не удалось сформировать ответ: LLM недоступен, а автоматические проверки не дали результата."
    )
    return _strip_row_markers(final)

