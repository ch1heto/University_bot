import re
import numpy as np
from typing import Optional, List, Dict, Tuple
from .db import get_conn
from .polza_client import embeddings

# Кэш векторов на процесс: (owner_id, doc_id) -> {"mat": np.ndarray [N,D], "meta": list[dict]}
_DOC_CACHE: dict[tuple[int, int], dict] = {}

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
        meta.append(
            {
                "id": r["id"],
                "page": r["page"],
                "section_path": r["section_path"],
                "text": r["text"],
            }
        )
        # эмбеддинг хранится как BLOB(float32[])
        vecs.append(np.frombuffer(r["embedding"], dtype=np.float32))

    mat = np.vstack(vecs)  # [N, D]
    # Нормируем по L2 для косинуса
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
    mat = mat / norms

    pack = {"mat": mat.astype(np.float32, copy=False), "meta": meta}
    _DOC_CACHE[key] = pack
    return pack


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
    Возвращает top-k чанков по косинусному сходству с вопросом.
    Каждый элемент: {id, page, section_path, text, score}.
    """
    pack = _load_doc(owner_id, doc_id)
    if pack["mat"].shape[0] == 0:
        return []

    q = _embed_query(query)
    if q is None:
        return []

    sims = pack["mat"] @ q  # косинус
    k = min(top_k, sims.shape[0])
    if k <= 0:
        return []

    # Быстрый выбор top-k без полного сортирования
    idx = np.argpartition(-sims, range(k))[:k]
    idx = idx[np.argsort(-sims[idx])]

    out: List[Dict] = []
    for i in idx:
        m = pack["meta"][int(i)]
        out.append(
            {
                "id": m["id"],
                "page": m["page"],
                "section_path": m["section_path"],
                "text": (m["text"] or "").strip(),
                "score": float(sims[i]),
            }
        )
    return out


def build_context(snippets: List[Dict], max_chars: int = 6000) -> str:
    """
    Склеивает фрагменты в вид:
    [Источник N] ...текст... (стр. X • Раздел)
    """
    parts, total = [], 0
    for i, s in enumerate(snippets, 1):
        block = f"[Источник {i}] {s['text']}  (стр. {s['page']} • {s['section_path']})"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
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
    # варианты 'таблица 3.1', 'табл. 3,1'
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


# Экспортируем паттерн-хелперы, чтобы дергать их в bot.py перед обычным RAG
# Пример использования:
#   pat = _mk_table_pattern(question) or _mk_figure_pattern(question)
#   if pat:
#       hits = keyword_find(uid, doc_id, pat)
#       if hits: ... ответить короткой справкой про страницы/фрагменты
__all__ = [
    "retrieve",
    "build_context",
    "invalidate_cache",
    "_mk_table_pattern",
    "_mk_figure_pattern",
    "keyword_find",
]
