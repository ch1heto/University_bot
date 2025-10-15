from __future__ import annotations
import re
from typing import List, Dict, Any, Optional, Tuple

from .db import get_conn
from .retrieval import retrieve, build_context

# ---------------------------- utils ----------------------------

_WHITESPACE_RE = re.compile(r"\s+")
_NUM_RE = re.compile(r"\b\d+(?:[.,]\d+)?%?\b")
_TOKEN_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё_+-]+", re.UNICODE)

def _norm_text(s: str | None) -> str:
    if not s:
        return ""
    s = s.replace("\xa0", " ")
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return s

def _extract_numbers(s: str) -> set[str]:
    return set(_NUM_RE.findall(s or ""))

def _table_has(con, table: str) -> bool:
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
    return cur.fetchone() is not None

def _fts_available(con) -> bool:
    return _table_has(con, "chunks_fts")

def _sanitize_like(q: str) -> str:
    # экранирование для LIKE
    return (q or "").replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

# ---------- безопасная сборка FTS5-запроса и SQL-литерала ----------

def _fts_escape_token(tok: str) -> str:
    # берём только разрешённые символы (в т.ч. кириллица), экранируем двойные кавычки
    m = _TOKEN_RE.findall(tok or "")
    t = "".join(m)
    return t.replace('"', '""')

def _make_fts_query(raw: str) -> str:
    """
    Строим FTS5 строку запроса: "слово1"* OR "слово2"* ...
    (префиксный поиск по каждому токену).
    """
    toks = [_fts_escape_token(t) for t in re.split(r"\s+", _norm_text(raw)) if t.strip()]
    toks = [t for t in toks if t]
    if not toks:
        return ""
    parts = [f'"{t}"*' for t in toks[:10]]  # ограничим до 10 токенов
    return " OR ".join(parts)

def _sql_literal(s: str) -> str:
    # SQL-литерал: одинарные кавычки удваиваем
    return "'" + (s or "").replace("'", "''") + "'"


# ------------------------- core: lex search -------------------------

def lex_search(owner_id: int, doc_id: int, query: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Лексический поиск по документу:
    - если есть FTS5 (chunks_fts) — используем MATCH (строка-запрос как SQL-литерал),
    - иначе fallback на LIKE.
    Возвращает [{id, page, section_path, text, score, source}], где source ∈ {"fts","like"}.
    score: для bm25 — значение ранга (чем МЕНЬШЕ, тем лучше), без bm25 — 1.0; для LIKE — эвристика.
    """
    q = _norm_text(query)
    if not q:
        return []

    con = get_conn()
    rows: List[Dict[str, Any]] = []

    if _fts_available(con):
        fts_q = _make_fts_query(q)
        if not fts_q:
            con.close()
            return []
        fts_lit = _sql_literal(fts_q)
        cur = con.cursor()
        # пробуем с bm25 (может отсутствовать в сборке)
        try:
            sql = f"""
            SELECT c.id, c.page, c.section_path, c.text, bm25(chunks_fts) AS rank
            FROM chunks_fts f
            JOIN chunks c ON c.id = f.rowid
            WHERE c.owner_id = ? AND c.doc_id = ? AND f.chunks_fts MATCH {fts_lit}
            ORDER BY rank ASC
            LIMIT ?
            """
            cur.execute(sql, (owner_id, doc_id, limit))
            for r in cur.fetchall():
                rows.append({
                    "id": r["id"],
                    "page": r["page"],
                    "section_path": r["section_path"],
                    "text": r["text"],
                    "score": float(r["rank"]),
                    "source": "fts"
                })
        except Exception:
            # без bm25 — хотя бы фильтруем по FTS
            sql = f"""
            SELECT c.id, c.page, c.section_path, c.text
            FROM chunks_fts f
            JOIN chunks c ON c.id = f.rowid
            WHERE c.owner_id = ? AND c.doc_id = ? AND f.chunks_fts MATCH {fts_lit}
            LIMIT ?
            """
            cur.execute(sql, (owner_id, doc_id, limit))
            for r in cur.fetchall():
                rows.append({
                    "id": r["id"],
                    "page": r["page"],
                    "section_path": r["section_path"],
                    "text": r["text"],
                    "score": 1.0,   # нейтральный скор для гибрида
                    "source": "fts"
                })
        con.close()
        return rows

    # --- fallback: LIKE (без FTS) ---
    like_q = "%" + _sanitize_like(q) + "%"
    cur = con.cursor()
    cur.execute(
        """
        SELECT id, page, section_path, text
        FROM chunks
        WHERE owner_id=? AND doc_id=? AND (text LIKE ? ESCAPE '\\' OR section_path LIKE ? ESCAPE '\\')
        LIMIT ?
        """,
        (owner_id, doc_id, like_q, like_q, limit)
    )
    for r in cur.fetchall():
        t = r["text"] or ""
        # грубая эвристика: больше общих чисел и точных токенов — лучше
        q_nums = _extract_numbers(q)
        c_nums = _extract_numbers(t)
        inter = len(q_nums & c_nums)
        toks = set(_norm_text(q).lower().split())
        hits = sum(1 for tok in toks if tok and tok in (t.lower()))
        score = max(0.0, 10.0 - 2.0 * inter - 0.5 * hits)  # «меньше — лучше»
        rows.append({
            "id": r["id"],
            "page": r["page"],
            "section_path": r["section_path"],
            "text": r["text"],
            "score": score,
            "source": "like"
        })
    con.close()
    rows.sort(key=lambda x: x["score"])  # лучше сверху
    return rows[:limit]


# ----------------------- hybrid: semantic + lex -----------------------

def hybrid_search(owner_id: int,
                  doc_id: int,
                  query: str,
                  sem_top_k: int = 6,
                  lex_top_k: int = 10,
                  merge_top_k: int = 8) -> List[Dict[str, Any]]:
    """
    Объединяем семантический RAG (retrieve) и лексический поиск:
    - берём top семантических и лексических хитов,
    - мерджим по chunk id,
    - пересчитываем скор: sem_score + бонусы за лекс.совпадение/числа,
    - возвращаем top-k.
    """
    sem_hits = retrieve(owner_id, doc_id, query, top_k=sem_top_k)  # [{id,page,section_path,text,score}]
    lex_hits = lex_search(owner_id, doc_id, query, limit=lex_top_k)

    lex_by_id: Dict[int, Dict[str, Any]] = {int(h["id"]): h for h in lex_hits if h.get("id") is not None}

    q_nums = _extract_numbers(query)
    out: List[Tuple[int, Dict[str, Any], float]] = []
    seen_ids = set()

    # семантика — базовый скор + бонусы, если подтверждается лексикой
    for h in sem_hits:
        cid = int(h["id"])
        base = float(h["score"])
        bonus = 0.0

        l = lex_by_id.get(cid)
        if l:
            bonus += 0.12  # лексическое подтверждение
            c_nums = _extract_numbers(l.get("text") or "")
            if q_nums:
                inter = len(q_nums & c_nums)
                if inter:
                    bonus += min(0.02 * inter, 0.08)

        out.append((cid, h, base + bonus))
        seen_ids.add(cid)

    # добавляем чисто лексические (которых не было в семантических)
    for l in lex_hits:
        cid = int(l["id"])
        if cid in seen_ids:
            continue
        try:
            lex_norm = 1.0 / (1.0 + float(l["score"]))  # чем меньше score, тем выше вклад
        except Exception:
            lex_norm = 0.3
        sem_stub = {
            "id": cid,
            "page": l.get("page"),
            "section_path": l.get("section_path"),
            "text": l.get("text") or "",
            "score": 0.0
        }
        out.append((cid, sem_stub, 0.15 * lex_norm))
        seen_ids.add(cid)

    out.sort(key=lambda x: -x[2])
    merged: List[Dict[str, Any]] = []
    for _, h, sc in out[:merge_top_k]:
        merged.append({
            "id": h["id"],
            "page": h["page"],
            "section_path": h["section_path"],
            "text": h["text"],
            "score": float(sc)
        })
    return merged


# ----------------------- helpers for bot usage -----------------------

def best_context(owner_id: int, doc_id: int, query: str, max_chars: int = 6000) -> str:
    """
    Строит контекст, используя гибридный поиск (лучше покрытие формулировок/таблиц).
    """
    hits = hybrid_search(owner_id, doc_id, query, sem_top_k=6, lex_top_k=12, merge_top_k=8)
    if not hits:
        return ""
    return build_context(hits, max_chars=max_chars)

def quick_find(owner_id: int, doc_id: int, query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Быстрый лексический поиск без семантики (например, для UI-подсказок).
    """
    return lex_search(owner_id, doc_id, query, limit=limit)
