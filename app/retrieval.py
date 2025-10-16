# app/retrieval.py
import re
import json
import numpy as np
from typing import Optional, List, Dict, Tuple

from .db import get_conn
from .polza_client import embeddings  # только эмбеддинги; чат не используется здесь

# Кэш векторов на процесс: (owner_id, doc_id) -> {"mat": np.ndarray [N,D], "meta": list[dict]}
_DOC_CACHE: dict[tuple[int, int], dict] = {}

# ---------------------------
# Утилиты схемы
# ---------------------------

def _table_has_columns(con, table: str, cols: List[str]) -> bool:
    cur = con.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    have = {row[1] for row in cur.fetchall()}
    return all(c in have for c in cols)

# ---------------------------
# Загрузка матрицы документа
# ---------------------------

def _load_doc(owner_id: int, doc_id: int) -> dict:
    """
    Грузит из SQLite все чанки документа, формирует нормированную матрицу эмбеддингов
    и метаданные. Результат кэшируется в памяти процесса.
    """
    key = (owner_id, doc_id)
    if key in _DOC_CACHE:
        return _DOC_CACHE[key]

    con = get_conn()
    cur = con.cursor()

    # Пытаемся взять расширенные поля (element_type, attrs), если они есть
    has_ext = _table_has_columns(con, "chunks", ["element_type", "attrs"])
    if has_ext:
        cur.execute(
            "SELECT id, page, section_path, text, element_type, attrs, embedding "
            "FROM chunks WHERE owner_id=? AND doc_id=?",
            (owner_id, doc_id),
        )
    else:
        cur.execute(
            "SELECT id, page, section_path, text, embedding "
            "FROM chunks WHERE owner_id=? AND doc_id=?",
            (owner_id, doc_id),
        )
    rows = cur.fetchall()
    con.close()

    if not rows:
        pack = {"mat": np.zeros((0, 1), np.float32), "meta": []}
        _DOC_CACHE[key] = pack
        return pack

    meta, vecs = [], []
    for r in rows:
        if has_ext:
            et = (r["element_type"] or "").lower() if "element_type" in r.keys() else ""
            attrs_raw = r["attrs"] if "attrs" in r.keys() else None
            try:
                attrs = json.loads(attrs_raw) if attrs_raw else {}
            except Exception:
                attrs = {}
            meta.append(
                {
                    "id": r["id"],
                    "page": r["page"],
                    "section_path": r["section_path"],
                    "text": r["text"],
                    "element_type": et,
                    "attrs": attrs,
                }
            )
            vecs.append(np.frombuffer(r["embedding"], dtype=np.float32))
        else:
            meta.append(
                {
                    "id": r["id"],
                    "page": r["page"],
                    "section_path": r["section_path"],
                    "text": r["text"],
                    # полей нет в схеме — оставим пустыми
                    "element_type": "",
                    "attrs": {},
                }
            )
            vecs.append(np.frombuffer(r["embedding"], dtype=np.float32))

    mat = np.vstack(vecs)  # [N, D]
    # Нормируем по L2 для косинуса
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
    mat = (mat / norms).astype(np.float32, copy=False)

    pack = {"mat": mat, "meta": meta}
    _DOC_CACHE[key] = pack
    return pack

# ---------------------------
# Вспомогалки сигналы/классы
# ---------------------------

_NUM_RE = re.compile(r"\b\d+(?:[.,]\d+)?%?\b")

def _extract_numbers(s: str) -> set[str]:
    return set(_NUM_RE.findall(s or ""))

def _classify_by_prefix(text: str) -> str:
    """Классификация по префиксам из indexing.py, если нет element_type в БД."""
    t = (text or "").lower()
    if t.startswith("[таблица]"):
        return "table"
    if t.startswith("[рисунок]"):
        return "figure"
    if t.startswith("[заголовок]"):
        return "heading"
    if t.startswith("[страница]"):
        return "page"
    return "text"

def _chunk_type(meta: Dict) -> str:
    """
    Читаем element_type из БД. Если отсутствует —:
    1) по названию раздела распознаём 'reference' (Источники/Список литературы/Библиография/References);
    2) иначе по префиксу текста.
    """
    et = (meta.get("element_type") or "").lower()
    if et:
        return et
    sec = (meta.get("section_path") or "").lower()
    if ("источник" in sec) or ("литератур" in sec) or ("библиограф" in sec) or ("reference" in sec):
        return "reference"
    return _classify_by_prefix(meta.get("text") or "")

def _query_signals(q: str) -> Dict[str, bool]:
    ql = (q or "").lower()
    return {
        "ask_table":   bool(re.search(r"\bтабл|таблица\b", ql)),
        "ask_figure":  bool(re.search(r"\bрис\.?|рисунок\b", ql)),
        "ask_heading": bool(re.search(r"\bглава\b|\bраздел\b|\bвведение\b|\bзаключение\b", ql)),
        "ask_sources": bool(re.search(r"\bисточник(?:и|ов)?\b|\bсписок\s+литературы\b|\bбиблиограф", ql) or
                            re.search(r"\breferences?\b", ql)),
    }

# ---------------------------
# Основной RAG-поиск
# ---------------------------

def _embed_query(query: str) -> Optional[np.ndarray]:
    """Безопасно получаем эмбеддинг вопроса. В случае ошибки вернём None."""
    try:
        vec = embeddings([query])[0]
        q = np.asarray(vec, dtype=np.float32)
        q /= (np.linalg.norm(q) + 1e-9)
        return q
    except Exception:
        return None

def retrieve(owner_id: int, doc_id: int, query: str, top_k: int = 8) -> List[Dict]:
    """
    Возвращает top-k чанков по косинусному сходству с вопросом (с лёгким переранжированием).
    Каждый элемент: {id, page, section_path, text, score}.
    """
    pack = _load_doc(owner_id, doc_id)
    if pack["mat"].shape[0] == 0:
        return []

    q = _embed_query(query)
    if q is None:
        return []

    sims = pack["mat"] @ q  # косинус
    N = sims.shape[0]
    # Возьмём побольше кандидатов для переранжа
    prelim_k = max(top_k * 3, top_k, 16)
    prelim_k = min(prelim_k, N)
    if prelim_k <= 0:
        return []

    # Быстрый выбор top-prelim_k без полного сортирования
    idx = np.argpartition(-sims, range(prelim_k))[:prelim_k]
    idx = idx[np.argsort(-sims[idx])]  # фикс: argsort латиницей

    # Переранж: тип чанка/совпадения номеров/чисел
    sig = _query_signals(query)
    q_nums = _extract_numbers(query)
    tab_pat = _mk_table_pattern(query)
    fig_pat = _mk_figure_pattern(query)
    tab_rgx = re.compile(tab_pat, re.IGNORECASE) if tab_pat else None
    fig_rgx = re.compile(fig_pat, re.IGNORECASE) if fig_pat else None

    rescored: List[Tuple[int, float]] = []
    for i in idx:
        m = pack["meta"][int(i)]
        text = (m["text"] or "")
        score = float(sims[i])

        ctype = _chunk_type(m)

        # Тематические бусты по типу чанка
        if sig["ask_table"] and ctype in {"table", "table_row"}:
            score += 0.10
        if sig["ask_figure"] and ctype == "figure":
            score += 0.08
        if sig["ask_heading"] and ctype == "heading":
            score += 0.05
        if sig["ask_sources"] and ctype == "reference":
            score += 0.25  # заметный буст для списка литературы

        # Совпадение конкретного номера таблицы/рисунка
        if tab_rgx and tab_rgx.search(text):
            score += 0.22
        if fig_rgx and fig_rgx.search(text):
            score += 0.22

        # Совпадение чисел (до +0.10 суммарно)
        if q_nums:
            c_nums = _extract_numbers(text)
            inter = len(q_nums & c_nums)
            if inter:
                score += min(0.02 * inter, 0.10)

        # Лёгкий штраф для «страниц», если не спрашивали именно про разделы/структуру
        if ctype == "page" and not (sig["ask_table"] or sig["ask_figure"] or sig["ask_heading"] or sig["ask_sources"]):
            score -= 0.02

        rescored.append((int(i), score))

    # Финальная сортировка по новым скорингам
    rescored.sort(key=lambda x: -x[1])
    best = rescored[:top_k]

    out: List[Dict] = []
    for i, sc in best:
        m = pack["meta"][int(i)]
        out.append(
            {
                "id": m["id"],
                "page": m["page"],
                "section_path": m["section_path"],
                "text": (m["text"] or "").strip(),
                "score": float(sc),
            }
        )
    return out


def build_context(snippets: List[Dict], max_chars: int = 6000) -> str:
    """
    Склеивает фрагменты ТОЛЬКО текстом без каких-либо ссылочных пометок
    (никаких страниц/разделов в квадратных скобках). Блоки разделяются пустой строкой.
    """
    parts: List[str] = []
    total = 0
    for s in snippets:
        block = (s.get("text") or "").strip()
        if not block:
            continue
        # Урезаем последний блок, чтобы не превысить max_chars
        if total + len(block) > max_chars:
            remaining = max_chars - total
            if remaining <= 0:
                break
            block = block[:remaining]
        parts.append(block)
        total += len(block)
        if total >= max_chars:
            break
    return "\n\n".join(parts)


def invalidate_cache(owner_id: int, doc_id: int):
    """Сбрасывает кэш матрицы для документа (вызывать после переиндексации)."""
    _DOC_CACHE.pop((owner_id, doc_id), None)

# ---------------------------
# Keyword-fallback (опционально)
# ---------------------------

def _mk_table_pattern(q: str) -> Optional[str]:
    """
    Пытаемся вытащить номер таблицы N.M из вопроса.
    Поддерживаем формы: 'таблица 3.1', 'табл. 3,1', нечувствительно к регистру.
    Возвращаем паттерн regex или None.
    """
    m = re.search(r"табл(?:ица)?\.?\s*(\d+)\s*[.,]\s*(\d+)", q, re.IGNORECASE)
    if not m:
        return None
    num = f"{m.group(1)}[.,]{m.group(2)}"
    return rf"(табл(?:ица)?\.?\s*{num})"


def _mk_figure_pattern(q: str) -> Optional[str]:
    """
    Поиск 'рисунок 2.4' / 'рис. 2,4'.
    """
    m = re.search(r"рис(?:унок)?\.?\s*(\d+)\s*[.,]\s*(\d+)", q, re.IGNORECASE)
    if not m:
        return None
    num = f"{m.group(1)}[.,]{m.group(2)}"
    return rf"(рис(?:унок)?\.?\s*{num})"


def keyword_find(owner_id: int, doc_id: int, pattern: str, max_hits: int = 3) -> List[Dict]:
    """
    Ищем прямые вхождения паттерна regex в текстах чанков.
    Возвращаем: [{page, section_path, snippet}]
    """
    rgx = re.compile(pattern, re.IGNORECASE)
    con = get_conn()
    cur = con.cursor()
    cur.execute(
        "SELECT page, section_path, text FROM chunks WHERE owner_id=? AND doc_id=?",
        (owner_id, doc_id),
    )
    rows = cur.fetchall()
    con.close()

    hits: List[Dict] = []
    for r in rows:
        t = (r["text"] or "")
        m = rgx.search(t)
        if m:
            s = max(m.start() - 120, 0)
            e = min(m.end() + 120, len(t))
            hits.append(
                {
                    "page": r["page"],
                    "section_path": r["section_path"],
                    "snippet": t[s:e].strip(),
                }
            )
            if len(hits) >= max_hits:
                break
    return hits
