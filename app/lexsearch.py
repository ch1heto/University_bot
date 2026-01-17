from __future__ import annotations
import re
import sqlite3
from typing import List, Dict, Any, Optional, Tuple

from .db import get_conn
from .retrieval import retrieve, build_context

# ---------------------------- utils ----------------------------

_WHITESPACE_RE = re.compile(r"\s+")
_NUM_RE = re.compile(r"\b\d+(?:[.,]\d+)?%?\b")
_TOKEN_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё_+-]+", re.UNICODE)

# NEW: детерминированный парсер области "глава/раздел/пункт 2.2", "chapter 2.2", "sec. 2.2"
_SECTION_SCOPE_RE = re.compile(
    r"(?i)\b(?:глава|раздел|пункт|подраздел|секц(?:ия)?|sec(?:tion)?\.?|chapter)\s*"
    r"(?:№\s*|no\.?\s*|номер\s*)?"
    r"([A-Za-zА-Яа-я]?\s*\d+(?:[.,]\d+)*)"
)

def _parse_section_scope(q: str) -> Optional[str]:
    """
    Возвращает нормализованный префикс секции ('2.2' или 'A.1') если в вопросе есть
    явная ссылка на главу/раздел/пункт. Иначе — None.
    """
    m = _SECTION_SCOPE_RE.search(q or "")
    if not m:
        return None
    raw = (m.group(1) or "").strip().replace(" ", "")
    return raw.replace(",", ".")

# ID/область
RE_CHAPTER = re.compile(
    r"(?i)\b(?:(?:в|по)\s+)?(?:глав[аеы]|раздел|пункт|подраздел|секц(?:ия)?|chapter|section|sec\.?)\s*"
    r"([A-Za-zА-Яа-я]?\s*\d+(?:[.,]\d+)*)"
)
RE_FIG = re.compile(r"(?i)\b(?:рис(?:\.|унок)?|figure|fig\.?)\s*(?:№\s*)?([A-Za-zА-Яа-я]?\s*\d+(?:[.,]\d+)*)")
RE_TBL = re.compile(r"(?i)\b(?:табл(?:ица)?|table)\s*(?:№\s*)?([A-Za-zА-Яа-я]?\s*\d+(?:[.,]\d+)*)")

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

def _table_info(con, table: str) -> set[str]:
    cur = con.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cur.fetchall()}

def _fts_available(con) -> bool:
    return _table_has(con, "chunks_fts")

def _sanitize_like(q: str) -> str:
    # экранирование для LIKE
    return (q or "").replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

def _has_col(con, table: str, col: str) -> bool:
    return col in _table_info(con, table)

# ---------- безопасная сборка FTS5-запроса ----------

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

# ---------------------------- ID parsing ----------------------------

def _norm_label(s: str | None) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", "", str(s)).replace(",", ".").strip()

def parse_query_ids(q: str) -> Dict[str, Any]:
    """
    Парсит запрос пользователя на предмет:
    - area: глава/раздел N  -> {'area_num': '1.2'}
    - target: рисунок/таблица N -> {'target_kind': 'figure'|'table', 'target_num': '2.1'}
    """
    s = _norm_text(q)
    out: Dict[str, Any] = {}

    m_area = RE_CHAPTER.search(s)
    if m_area:
        out["area_num"] = _norm_label(m_area.group(1))

    m_fig = RE_FIG.search(s)
    m_tbl = RE_TBL.search(s)
    if m_fig:
        out["target_kind"] = "figure"
        out["target_num"] = _norm_label(m_fig.group(1))
    elif m_tbl:
        out["target_kind"] = "table"
        out["target_num"] = _norm_label(m_tbl.group(1))

    return out

# ---------------------------- suggestions (Levenshtein) ----------------------------

def _levenshtein(a: str, b: str) -> int:
    a, b = a or "", b or ""
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    prev = list(range(m + 1))
    cur = [0] * (m + 1)
    for i in range(1, n + 1):
        cur[0] = i
        ca = a[i - 1]
        for j in range(1, m + 1):
            cb = b[j - 1]
            cost = 0 if ca == cb else 1
            cur[j] = min(cur[j-1] + 1, prev[j] + 1, prev[j-1] + cost)
        prev, cur = cur, prev
    return prev[m]

# ---------------------------- document_sections helpers ----------------------------

def _load_sections(owner_id: int, doc_id: int, *, kind: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Пытаемся читать из document_sections; если таблицы нет — мягкий фолбэк на chunks.
    kind: 'figure' | 'table' | 'heading' | None (все)
    """
    con = get_conn()
    cur = con.cursor()
    use_doc_sections = _table_has(con, "document_sections")

    rows = []
    if use_doc_sections:
        if kind:
            cur.execute(
                "SELECT ord, title, level, page, section_path, element_type, text, attrs "
                "FROM document_sections WHERE doc_id=? AND element_type=? ORDER BY ord ASC",
                (doc_id, kind),
            )
        else:
            cur.execute(
                "SELECT ord, title, level, page, section_path, element_type, text, attrs "
                "FROM document_sections WHERE doc_id=? ORDER BY ord ASC",
                (doc_id,),
            )
        rows = cur.fetchall() or []
    else:
        # Фолбэк: собираем «секции» прямо из chunks
        cols = _table_info(con, "chunks")
        has_attrs = "attrs" in cols
        has_et   = "element_type" in cols
        sel_attrs = "attrs" if has_attrs else "NULL AS attrs"
        sel_et    = "element_type" if has_et else "'' AS element_type"

        if kind and has_et:
            cur.execute(
                f"""
                SELECT id AS ord, '' AS title, 0 AS level, page, section_path, {sel_et} AS element_type, text, {sel_attrs} AS attrs
                FROM chunks
                WHERE owner_id=? AND doc_id=? AND element_type=?
                ORDER BY id ASC
                """,
                (owner_id, doc_id, kind),
            )
        else:
            cur.execute(
                f"""
                SELECT id AS ord, '' AS title, 0 AS level, page, section_path, {sel_et} AS element_type, text, {sel_attrs} AS attrs
                FROM chunks
                WHERE owner_id=? AND doc_id=?
                ORDER BY id ASC
                """,
                (owner_id, doc_id),
            )
        rows = cur.fetchall() or []

    con.close()

    out: List[Dict[str, Any]] = []
    for r in rows:
        try:
            import json
            attrs = json.loads(r["attrs"]) if "attrs" in r.keys() and r["attrs"] else {}
        except Exception:
            attrs = {}
        out.append({
            "ord": r["ord"],
            "title": r["title"] or "",
            "level": r["level"],
            "page": r["page"],
            "section_path": r["section_path"] or "",
            "element_type": (r["element_type"] or "").lower(),
            "text": r["text"] or "",
            "attrs": attrs,
        })
    return out


def _find_area_prefix(owner_id: int, doc_id: int, area_num: str) -> Optional[str]:
    """
    Ищем префикс области. Если заголовок не нашли — пробуем по chunks.section_path LIKE 'N%'.
    В крайнем случае возвращаем сам номер как префикс.
    """
    if not area_num:
        return None

    sections = _load_sections(owner_id, doc_id, kind="heading")
    pat = re.compile(rf"(^|\D){re.escape(area_num)}(\D|$)")
    for s in sections:
        t = f"{s.get('title','')} || {s.get('section_path','')}"
        if pat.search(t):
            return s.get("section_path") or s.get("title") or area_num

    # Фолбэк по chunks
    con = get_conn()
    cur = con.cursor()
    try:
        cur.execute(
            "SELECT section_path FROM chunks WHERE owner_id=? AND doc_id=? AND section_path LIKE ? || '%' ORDER BY id ASC LIMIT 1",
            (owner_id, doc_id, area_num),
        )
        row = cur.fetchone()
        if row and row["section_path"]:
            return row["section_path"]
    finally:
        con.close()
    return area_num

def _find_target_by_id(owner_id: int, doc_id: int, kind: str, num: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Сначала ищем по attrs.caption_num/label в document_sections;
    если таблицы нет — _load_sections() вернёт данные из chunks (фолбэк).
    """
    want = _norm_label(num)
    secs = _load_sections(owner_id, doc_id, kind=kind)

    hits: List[Dict[str, Any]] = []
    cands: List[Tuple[int, Dict[str, Any]]] = []

    for s in secs:
        a = s.get("attrs") or {}
        lab = _norm_label(a.get("caption_num") or a.get("label") or "")
        if lab:
            if lab == want:
                hits.append(s)
            else:
                cands.append((_levenshtein(want, lab), s))

    # если точного попадания по attrs нет — пробуем по подписи в section_path/text
    if not hits:
        needle_ru = rf"(?i)\b{'рисунок' if kind=='figure' else 'таблица'}\s+{re.escape(want)}\b"
        needle_en = rf"(?i)\b{'figure' if kind=='figure' else 'table'}\s+{re.escape(want)}\b"
        rgx_ru = re.compile(needle_ru)
        rgx_en = re.compile(needle_en)
        for s in secs:
            sp = s.get("section_path") or ""
            tx = s.get("text") or ""
            if rgx_ru.search(sp) or rgx_ru.search(tx) or rgx_en.search(sp) or rgx_en.search(tx):
                hits.append(s)

    cands.sort(key=lambda x: x[0])
    sugg = [{
        "label": _norm_label((s.get("attrs") or {}).get("caption_num") or (s.get("attrs") or {}).get("label") or ""),
        "title": s.get("title") or "",
        "section_path": s.get("section_path") or "",
        "dist": int(d),
    } for d, s in cands[:5]]

    return hits, sugg

# ------------------------- core: lex search (с фильтрами) -------------------------

def lex_search(owner_id: int,
               doc_id: int,
               query: str,
               limit: int = 20,
               *,
               section_prefix: Optional[str] = None,
               element_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Лексический поиск по документу:
    - если есть FTS5 (chunks_fts) — используем MATCH,
    - иначе fallback на LIKE.
    Фильтры:
      section_prefix: ограничить section_path LIKE '{prefix}%'
      element_types:  ограничить по типам (если колонка есть): ['paragraph','table','figure','heading','page','table_row']
    Возвращает [{id, page, section_path, text, score, source}].
    """
    q = _norm_text(query)
    if not q:
        return []

    con = get_conn()
    rows: List[Dict[str, Any]] = []

    # Какая схема у chunks
    chunks_cols = _table_info(con, "chunks")
    has_elem_col = ("element_type" in chunks_cols)

    # WHERE кусочки
    extra_where = []
    params: List[Any] = []

    if section_prefix:
        extra_where.append("c.section_path LIKE ? ESCAPE '\\'")
        params.append(_sanitize_like(section_prefix) + "%")

    if element_types and has_elem_col:
        placeholders = ",".join("?" for _ in element_types)
        extra_where.append(f"c.element_type IN ({placeholders})")
        params.extend([et for et in element_types])

    extra_sql = ""
    if extra_where:
        extra_sql = " AND " + " AND ".join(extra_where)

    if _fts_available(con):
        fts_q = _make_fts_query(q)
        if not fts_q:
            con.close()
            return []
        cur = con.cursor()
        try:
            sql = f"""
            SELECT c.id, c.page, c.section_path, c.text, bm25(chunks_fts) AS rank
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.rowid
            WHERE c.owner_id = ? AND c.doc_id = ? AND chunks_fts MATCH ? {extra_sql}
            ORDER BY rank ASC, c.id ASC
            LIMIT ?
            """
            cur.execute(sql, (owner_id, doc_id, fts_q, *params, limit))
            for r in cur.fetchall():
                rows.append({
                    "id": r["id"],
                    "page": r["page"],
                    "section_path": r["section_path"],
                    "text": r["text"],
                    "score": float(r["rank"]),
                    "source": "fts"
                })
        except sqlite3.OperationalError:
            sql = f"""
            SELECT c.id, c.page, c.section_path, c.text
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.rowid
            WHERE c.owner_id = ? AND c.doc_id = ? AND chunks_fts MATCH ? {extra_sql}
            ORDER BY c.id ASC
            LIMIT ?
            """
            cur.execute(sql, (owner_id, doc_id, fts_q, *params, limit))
            for r in cur.fetchall():
                rows.append({
                    "id": r["id"],
                    "page": r["page"],
                    "section_path": r["section_path"],
                    "text": r["text"],
                    "score": 1.0,
                    "source": "fts"
                })
        con.close()
        return rows

    # --- fallback: LIKE (без FTS) ---
    like_q = "%" + _sanitize_like(q) + "%"
    cur = con.cursor()
    base = f"""
        SELECT id, page, section_path, text
        FROM chunks c
        WHERE c.owner_id=? AND c.doc_id=? AND (c.text LIKE ? ESCAPE '\\' OR c.section_path LIKE ? ESCAPE '\\')
    """
    tail = ""
    if extra_where:
        tail = " AND " + " AND ".join(extra_where)
    sql = base + tail + " LIMIT ?"
    cur.execute(sql, (owner_id, doc_id, like_q, like_q, *params, limit))
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

# ----------------------- hybrid: semantic + lex (с областью) -----------------------

def hybrid_search(owner_id: int,
                  doc_id: int,
                  query: str,
                  sem_top_k: int = 6,
                  lex_top_k: int = 10,
                  merge_top_k: int = 8,
                  *,
                  section_prefix: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Объединяем семантический RAG (retrieve) и лексический поиск.
    Если задан section_prefix — семантику оставить как есть (retrieve не умеет фильтры),
    а лексический — ограничить областью; затем смержить.
    """
    try:
        sem_hits = retrieve(owner_id, doc_id, query, top_k=sem_top_k)  # [{id,page,section_path,text,score}]
    except Exception:
        sem_hits = []

    # Если задана область — остаются только те сем-хиты, которые попадают в неё
    if section_prefix:
        pfx = section_prefix
        sem_hits = [h for h in sem_hits if str(h.get("section_path","")).startswith(pfx)]

    lex_hits = lex_search(owner_id, doc_id, query, limit=lex_top_k, section_prefix=section_prefix)

    lex_by_id: Dict[int, Dict[str, Any]] = {int(h["id"]): h for h in lex_hits if h.get("id") is not None}

    q_nums = _extract_numbers(query)
    out: List[Tuple[int, Dict[str, Any], float]] = []
    seen_ids = set()

    # семантика — базовый скор + бонусы, если подтверждается лексикой
    for h in sem_hits:
        try:
            cid = int(h["id"])
        except Exception:
            continue
        base = float(h.get("score", 0.0))
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

# ----------------------- ID-aware orchestration -----------------------

def id_aware_search(owner_id: int,
                    doc_id: int,
                    query: str,
                    *,
                    max_ctx_chars: int = 6000) -> Dict[str, Any]:
    """
    Главная точка: понимает ID, область, собирает контекст.
    """
    meta = parse_query_ids(query)
    area_prefix: Optional[str] = None

    # сначала пробуем area_num из parse_query_ids, затем fallback на детерминированный парсер _parse_section_scope
    area_num = meta.get("area_num") or _parse_section_scope(query)
    if area_num:
        area_prefix = _find_area_prefix(owner_id, doc_id, area_num)

        # Приоритет: если явно указан ID рисунка/таблицы — находим его
    tkind = meta.get("target_kind")
    tnum = meta.get("target_num")
    if tkind and tnum:
        hits, sugg = _find_target_by_id(owner_id, doc_id, tkind, tnum)
        if hits:
            sec = hits[0]
            attrs = sec.get("attrs") or {}
            scope = attrs.get("section_scope") or sec.get("section_path") or ""

            # Якорный чанк той же секции (если есть) + локальный контекст вокруг него
            anchor_hits: List[Dict[str, Any]] = []
            neighbor_hits: List[Dict[str, Any]] = []
            heading_hits: List[Dict[str, Any]] = []
            mention_hits: List[Dict[str, Any]] = []

            con = get_conn()
            try:
                cur = con.cursor()
                cur.execute(
                    "SELECT id, page, section_path, text FROM chunks WHERE owner_id=? AND doc_id=? AND section_path=? LIMIT 1",
                    (owner_id, doc_id, sec.get("section_path") or "")
                )
                row = cur.fetchone()
                anchor_id: Optional[int] = None
                if row:
                    anchor_id = int(row["id"])
                    anchor_hits.append({
                        "id": row["id"],
                        "page": row["page"],
                        "section_path": row["section_path"],
                        "text": row["text"] or "",
                        "score": 1.0,
                    })

                if anchor_id is not None:
                    neighbor_hits = _gather_neighbors(con, owner_id, doc_id, anchor_id, before=3, after=3)
                    heading_hits = _nearest_heading(con, owner_id, doc_id, anchor_id)

                # упоминания "Рисунок/Таблица N.M" по всему документу (пара абзацев)
                mention_hits = _gather_mentions(con, owner_id, doc_id, tnum, tkind, limit=4)
            finally:
                con.close()

            area_hits = hybrid_search(
                owner_id, doc_id, query, sem_top_k=6, lex_top_k=12, merge_top_k=10,
                section_prefix=scope
            )

            # дедупликация в порядке значимости: якорь → заголовок → соседи → упоминания → область
            by_id: Dict[int, Dict[str, Any]] = {}
            for lst in (anchor_hits, heading_hits, neighbor_hits, mention_hits, area_hits):
                for h in lst:
                    by_id.setdefault(int(h["id"]), h)

            final_hits = list(by_id.values())
            ctx = build_context(final_hits, max_chars=max_ctx_chars)

            return {
                "found": True,
                "context": ctx,
                "kind": tkind,
                "label": tnum,
                "area_prefix": scope,
                "suggestions": None,
            }

        # --- НОВОЕ: фолбэк по текстовым упоминаниям, если структурный ID не нашли ---
        # Пытаемся хотя бы привязаться к абзацам, где встречается "Таблица/Рисунок N.M"
        con = get_conn()
        try:
            mention_hits = _gather_mentions(con, owner_id, doc_id, tnum, tkind, limit=1)
            if mention_hits:
                anchor = mention_hits[0]
                anchor_id = int(anchor["id"])

                neighbor_hits = _gather_neighbors(con, owner_id, doc_id, anchor_id, before=3, after=3)
                heading_hits = _nearest_heading(con, owner_id, doc_id, anchor_id)

                # область — по section_path этого упоминания (если есть)
                scope = anchor.get("section_path") or area_prefix or ""

                area_hits = hybrid_search(
                    owner_id,
                    doc_id,
                    query,
                    sem_top_k=6,
                    lex_top_k=12,
                    merge_top_k=10,
                    section_prefix=scope or None,
                )

                by_id: Dict[int, Dict[str, Any]] = {}
                for lst in ([anchor], heading_hits, neighbor_hits, area_hits):
                    for h in lst:
                        by_id.setdefault(int(h["id"]), h)

                final_hits = list(by_id.values())
                ctx = build_context(final_hits, max_chars=max_ctx_chars)

                return {
                    "found": True,
                    "context": ctx,
                    "kind": tkind,
                    "label": tnum,
                    "area_prefix": scope or None,
                    "suggestions": None,
                }
        finally:
            con.close()

        # Если ни ID, ни текстовых упоминаний — честно говорим, что не нашли
        return {
            "found": False,
            "context": "",
            "kind": tkind,
            "label": tnum,
            "area_prefix": area_prefix,
            "suggestions": sugg or [],
        }



    # Если область задана, но без ID — ограничим поиск ей
        # Если область задана, но без ID — сначала пробуем ограничиться ей,
    # а при полном отсутствии хитов делаем обычный поиск по всему документу.
    if area_prefix:
        # --- НОВОЕ: прямой контекст по всей области (глава/раздел), если такая секция реально есть ---
        con = get_conn()
        try:
            cur = con.cursor()
            cur.execute(
                """
                SELECT id, page, section_path, text
                FROM chunks
                WHERE owner_id=? AND doc_id=? AND section_path LIKE ? || '%'
                ORDER BY id ASC
                """,
                (owner_id, doc_id, area_prefix),
            )
            rows = cur.fetchall() or []
        finally:
            con.close()

        if rows:
            area_rows: List[Dict[str, Any]] = []
            for r in rows:
                area_rows.append({
                    "id": r["id"],
                    "page": r["page"],
                    "section_path": r["section_path"],
                    "text": r["text"] or "",
                    "score": 1.0,
                })
            ctx = build_context(area_rows, max_chars=max_ctx_chars)
            return {
                "found": True,
                "context": ctx,
                "kind": None,
                "label": None,
                "area_prefix": area_prefix,
                "suggestions": None,
            }

        # --- СТАРОЕ поведение: гибридный поиск с ограничением по области + фолбэк на весь документ ---
        hits = hybrid_search(
            owner_id,
            doc_id,
            query,
            sem_top_k=6,
            lex_top_k=12,
            merge_top_k=8,
            section_prefix=area_prefix,
        )

        if not hits:
            # Fallback: игнорируем подсказку области и ищем по всему документу
            hits = hybrid_search(owner_id, doc_id, query, sem_top_k=6, lex_top_k=12, merge_top_k=8)
            if not hits:
                return {
                    "found": False,
                    "context": "",
                    "kind": None,
                    "label": None,
                    "area_prefix": area_prefix,
                    "suggestions": [],
                }
            ctx = build_context(hits, max_chars=max_ctx_chars)
            return {
                "found": True,
                "context": ctx,
                "kind": None,
                "label": None,
                # область не сработала, контекст общий
                "area_prefix": None,
                "suggestions": None,
            }

        ctx = build_context(hits, max_chars=max_ctx_chars)
        return {
            "found": True,
            "context": ctx,
            "kind": None,
            "label": None,
            "area_prefix": area_prefix,
            "suggestions": None,
        }



    # Обычный путь — без области/ID
    hits = hybrid_search(owner_id, doc_id, query, sem_top_k=6, lex_top_k=12, merge_top_k=8)
    if not hits:
        return {"found": False, "context": "", "kind": None, "label": None, "area_prefix": None, "suggestions": []}
    ctx = build_context(hits, max_chars=max_ctx_chars)
    return {"found": True, "context": ctx, "kind": None, "label": None, "area_prefix": None, "suggestions": None}

# ----------------------- helpers for bot usage -----------------------

def best_context(owner_id: int, doc_id: int, query: str, max_chars: int = 6000) -> str:
    """
    ID-aware: если в запросе указан рисунок/таблица/область — отдаём локальный контекст.
    Иначе — гибридный контекст по всему документу.

    Строгий режим: если контекст слишком слабый/шумный — возвращаем пустую строку.
    ВАЖНО: короткие, но фактологичные фрагменты НЕ режем (это повышает универсальность).
    """
    res = id_aware_search(owner_id, doc_id, query, max_ctx_chars=max_chars) or {}
    ctx = (res.get("context") or "").strip()

    if not ctx:
        return ""

    def _looks_facty(text: str) -> bool:
        t = text or ""
        tl = t.lower()

        # цифры, проценты, годы, формулы, диапазоны
        if re.search(r"\d", t):
            return True
        if "%" in t or "±" in t or "≤" in t or "≥" in t:
            return True

        # латиница/аббревиатуры/технологии/методики часто пишутся латиницей
        if re.search(r"[A-Za-z]{2,}", t):
            return True

        # частые маркеры терминов/кратких фактов: двоеточия, перечисления, кавычки
        if ":" in t or "—" in t or "–" in t:
            return True

        # “жирные” слова (универсально): методика/шкала/опросник/модель/алгоритм/фреймворк/платформа/ПО
        if re.search(r"\b(методик|шкал|опросник|тест|модель|алгоритм|фреймворк|библиотек|платформ|программ|по\b)\b", tl):
            return True

        return False

    words = [w for w in re.split(r"\s+", ctx) if w]

    # было: <200 или <30 слов -> пусто
    # стало: короткое разрешаем, если оно "фактологичное"
    if (len(ctx) < 200 or len(words) < 30) and not _looks_facty(ctx):
        return ""

    # фильтр “обрывки строк” смягчаем, иначе реально полезные списки/определения режутся
    lines = [ln.strip() for ln in ctx.splitlines() if ln.strip()]
    if lines and len(lines) >= 8:
        short_ratio = sum(1 for ln in lines if len(ln) < 25) / max(1, len(lines))
        if short_ratio > 0.75 and not _looks_facty(ctx):
            return ""

    return ctx[:max_chars]


def quick_find(owner_id: int, doc_id: int, query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Быстрый лексический поиск без семантики (например, для UI-подсказок).
    """
    return lex_search(owner_id, doc_id, query, limit=limit)

# --- neighbors/mentions/heading helpers (NEW) ---

def _gather_neighbors(con, owner_id: int, doc_id: int, anchor_id: int, *, before: int = 3, after: int = 3) -> List[Dict[str, Any]]:
    """Берём N чанков до и после якорного id по порядку вставки."""
    cur = con.cursor()
    cur.execute(
        """
        SELECT id, page, section_path, text
        FROM chunks
        WHERE owner_id=? AND doc_id=? AND id BETWEEN ? AND ?
        ORDER BY id ASC
        """,
        (owner_id, doc_id, max(1, anchor_id - max(1, before)), anchor_id + max(1, after)),
    )
    rows = cur.fetchall() or []
    hits: List[Dict[str, Any]] = []
    for r in rows:
        hits.append({
            "id": r["id"],
            "page": r["page"],
            "section_path": r["section_path"],
            "text": r["text"] or "",
            "score": 0.86,  # около-якорная важность
        })
    return hits

def _gather_mentions(con, owner_id: int, doc_id: int, label: str, kind: str, *, limit: int = 4) -> List[Dict[str, Any]]:
    """
    Находим места в тексте, где встречается 'Рисунок N'/'Figure N' или 'Таблица N'/'Table N'.
    kind: 'figure' | 'table'
    """
    lab = (label or "").replace(",", ".")
    ru_word = "рисунок" if kind == "figure" else "таблица"
    en_word = "figure" if kind == "figure" else "table"
    rx = re.compile(rf"(?i)\b({ru_word}|{en_word})\s+{re.escape(lab)}\b")

    cur = con.cursor()
    cur.execute(
        """
        SELECT id, page, section_path, text
        FROM chunks
        WHERE owner_id=? AND doc_id=? AND (
            text LIKE '%Рисун%' ESCAPE '\\' OR text LIKE '%Figure%' ESCAPE '\\'
            OR text LIKE '%Таблиц%' ESCAPE '\\' OR text LIKE '%Table%' ESCAPE '\\'
        )
        ORDER BY id ASC
        """,
        (owner_id, doc_id),
    )
    rows = cur.fetchall() or []
    out: List[Dict[str, Any]] = []
    for r in rows:
        t = r["text"] or ""
        if rx.search(t):
            out.append({
                "id": r["id"],
                "page": r["page"],
                "section_path": r["section_path"],
                "text": t,
                "score": 0.84,
            })
            if len(out) >= limit:
                break
    return out

def _nearest_heading(con, owner_id: int, doc_id: int, around_id: int) -> List[Dict[str, Any]]:
    """
    Если в chunks есть element_type — подтягиваем ближайший заголовок перед якорём.
    Иначе возвращаем пусто.
    """
    cur = con.cursor()
    cur.execute("PRAGMA table_info(chunks)")
    cols = {row[1] for row in cur.fetchall()}
    if "element_type" not in cols:
        return []
    cur.execute(
        """
        SELECT id, page, section_path, text
        FROM chunks
        WHERE owner_id=? AND doc_id=? AND element_type='heading' AND id<=?
        ORDER BY id DESC LIMIT 1
        """,
        (owner_id, doc_id, around_id),
    )
    r = cur.fetchone()
    if not r:
        return []
    return [{
        "id": r["id"],
        "page": r["page"],
        "section_path": r["section_path"],
        "text": r["text"] or "",
        "score": 0.90,  # глава важнее соседей
    }]
