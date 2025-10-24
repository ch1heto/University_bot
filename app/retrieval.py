# app/retrieval.py
import re
import json
import numpy as np
from typing import Optional, List, Dict, Tuple, Any

from .db import get_conn
from .polza_client import embeddings, vision_describe  # эмбеддинги + vision

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
            emb = r["embedding"]
            vecs.append(np.frombuffer(emb, dtype=np.float32))
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
            emb = r["embedding"]
            vecs.append(np.frombuffer(emb, dtype=np.float32))

    # Сшиваем и нормируем L2
    mat = np.vstack(vecs).astype(np.float32, copy=False)  # [N, D]
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
    mat = (mat / norms).astype(np.float32, copy=False)

    pack = {"mat": mat, "meta": meta}
    _DOC_CACHE[key] = pack
    return pack

# ---------------------------
# Вспомогалки сигналы/классы
# ---------------------------

_NUM_RE = re.compile(r"\b\d[\d\s.,]*%?\b")

_CLEAN_MARKERS_RE = re.compile(
    r"^\s*\[(?:таблица|рисунок|заголовок|страница)\]\s*|"
    r"\s*\[\s*row\s*\d+\s*\]\s*|"
    r"\s*\(\d+\s*[×x]\s*\d+\)\s*$",
    re.IGNORECASE
)

def _clean_for_ctx(s: str) -> str:
    if not s:
        return ""
    t = _CLEAN_MARKERS_RE.sub("", s)
    t = t.replace("\u00A0", " ")                # NBSP -> пробел
    t = t.replace("–", "-").replace("—", "-")   # нормализуем тире
    t = re.sub(r"\s+\|\s+", " — ", t)           # "a | b | c" -> "a — b — c"
    t = re.sub(r"\s+", " ", t).strip()
    return t

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
        "ask_table":   bool(re.search(r"\b(табл|таблица|table)\b", ql)),
        "ask_figure":  bool(re.search(r"\b(рис\.?|рисунок|figure|fig\.?)\b", ql)),
        "ask_heading": bool(re.search(r"\b(глава|раздел|введение|заключение|heading|chapter|section)\b", ql)),
        "ask_sources": bool(re.search(r"\bисточник(?:и|ов)?\b|\bсписок\s+литературы\b|\bбиблиограф", ql) or
                            re.search(r"\breferences?\b|bibliograph\w*", ql)),
    }

# ---------------------------
# Базовые эмбеддинги
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

# ---------------------------
# Внутренний скорер/ранжирование (переиспользуется)
# ---------------------------

def _mk_table_pattern(q: str) -> Optional[str]:
    """
    Пытаемся вытащить номер таблицы из вопроса.
    Поддерживаем формы:
      - 'таблица 3.1', 'табл. 3,1'
      - 'table 2.4'
      - допускаем '№'
      - допускаем буквенный префикс 'А.1' / 'П1.2'
    Возвращаем паттерн regex или None.
    """
    ql = (q or "")
    m = re.search(r"(табл(?:ица)?|table)\s*(?:№\s*)?([A-Za-zА-Яа-я]?\s*\d+(?:[.,]\d+)*)", ql, re.IGNORECASE)
    if not m:
        return None
    raw = m.group(2).strip()
    raw = re.sub(r"\s+", " ", raw).replace(" ", "")
    # Буквенный префикс?
    if re.match(r"^[A-Za-zА-Яа-я]\d", raw):
        letter = raw[0]
        rest = raw[1:]
        # экранируем, а затем превращаем '\.' в '[.,]'
        rest_esc = re.escape(rest)
        rest_pat = re.sub(r"\\\.", "[.,]", rest_esc)
        pat_num = rf"{re.escape(letter)}\.?\s*{rest_pat}"
    else:
        pat_num = re.sub(r"\\\.", "[.,]", re.escape(raw))
    return rf"(?:табл(?:ица)?|table)\s*\.?\s*(?:№\s*)?{pat_num}"

def _mk_figure_pattern(q: str) -> Optional[str]:
    """
    Поиск 'рисунок 2.4' / 'рис. 2,4' / 'figure 3' / 'fig. 1' / 'рис. А.1'.
    Возвращаем паттерн regex или None.
    """
    ql = (q or "")
    m = re.search(r"(рис(?:унок)?|fig(?:ure)?\.?)\s*(?:№\s*)?([A-Za-zА-Яа-я]?\s*\d+(?:[.,]\d+)*)", ql, re.IGNORECASE)
    if not m:
        return None
    raw = m.group(2).strip()
    raw = re.sub(r"\s+", " ", raw).replace(" ", "")
    if re.match(r"^[A-Za-zА-Яа-я]\d", raw):
        letter = raw[0]
        rest = raw[1:]
        rest_esc = re.escape(rest)
        rest_pat = re.sub(r"\\\.", "[.,]", rest_esc)
        pat_num = rf"{re.escape(letter)}\.?\s*{rest_pat}"
    else:
        pat_num = re.sub(r"\\\.", "[.,]", re.escape(raw))
    return rf"(?:рис(?:унок)?|fig(?:ure)?\.?)\s*\.?\s*(?:№\s*)?{pat_num}"

def _score_and_rank(pack: dict, query: str, *, prelim_k: int = 48) -> List[Tuple[int, float]]:
    """
    Возвращает список индексов чанков и их скорингов, отсортированный по убыванию.
    Используем в retrieve() и coverage-вариантах.
    """
    if pack["mat"].shape[0] == 0:
        return []

    q = _embed_query(query)
    if q is None:
        return []

    sims = pack["mat"] @ q  # косинус
    N = sims.shape[0]
    if N == 0:
        return []

    prelim_k = max(min(prelim_k, N), 1)
    part_idx = np.argpartition(sims, -prelim_k)[-prelim_k:]
    idx = part_idx[np.argsort(-sims[part_idx])]

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
            score += 0.12
        if sig["ask_figure"] and ctype == "figure":
            score += 0.14
        if sig["ask_heading"] and ctype == "heading":
            score += 0.06
        if sig["ask_sources"] and ctype == "reference":
            score += 0.35
        if (not sig["ask_sources"]) and ctype == "reference":
            score -= 0.12

        # Совпадение конкретного номера таблицы/рисунка
        if tab_rgx and tab_rgx.search(text):
            score += 0.24
        if fig_rgx and fig_rgx.search(text):
            score += 0.26

        # Совпадение чисел (до +0.12 суммарно)
        if q_nums:
            c_nums = _extract_numbers(text)
            inter = len(q_nums & c_nums)
            if inter:
                score += min(0.025 * inter, 0.12)

        # Лёгкий штраф для «страниц», если не спрашивали про структуру
        if ctype == "page" and not (sig["ask_table"] or sig["ask_figure"] or sig["ask_heading"] or sig["ask_sources"]):
            score -= 0.03

        # Пустые/очень короткие текстовые чанки — мягкий штраф
        if len((text or "").strip()) < 30:
            score -= 0.04

        rescored.append((int(i), score))

    rescored.sort(key=lambda x: -x[1])
    return rescored

# ---------------------------
# Основной RAG-поиск (backward-compatible)
# ---------------------------

def retrieve(owner_id: int, doc_id: int, query: str, top_k: int = 8) -> List[Dict]:
    """
    Возвращает top-k чанков по косинусному сходству с вопросом (с лёгким переранжированием).
    Каждый элемент: {id, page, section_path, text, score}.
    """
    pack = _load_doc(owner_id, doc_id)
    rescored = _score_and_rank(pack, query, prelim_k=max(top_k * 3, 16))
    if not rescored:
        return []

    # Фильтр источников если их явно не просили
    sig = _query_signals(query)
    filtered: List[Tuple[int, float]] = []
    for i, sc in rescored:
        if len(filtered) >= top_k:
            break
        if (not sig["ask_sources"]) and _chunk_type(pack["meta"][i]) == "reference":
            continue
        filtered.append((i, sc))
    best = (filtered or rescored)[:top_k]

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
    Склеиваем контекст, предварительно очищая служебные метки и нормализуя пробелы/тире.
    Страницы/разделы не вставляем — только чистый текст.
    """
    parts: List[str] = []
    total = 0
    for s in snippets:
        raw = (s.get("text") or "")
        block = _clean_for_ctx(raw)
        if not block:
            continue
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

# =======================================================================
#                НОВОЕ: «РИСУНКИ» — СЧЁТЧИК, СПИСОК, КАРТОЧКИ
# =======================================================================

# «Рисунок 3.2 — ...», «Рисунок 5: ...», «Figure 2 — ...»
_FIG_TITLE_RE = re.compile(
    r"(?i)\b(?:рис(?:унок)?|figure)\s+([A-Za-zА-Яа-я]?\s*\d+(?:[.,]\d+)*)\b(?:\s*[—\-–:]\s*(.+))?"
)

def _shorten(s: str, limit: int = 120) -> str:
    s = (s or "").strip()
    if len(s) <= limit:
        return s
    return s[:limit - 1].rstrip() + "…"

def _last_segment(name: str) -> str:
    """Берём «хвост» из длинных путей section_path."""
    s = (name or "").strip()
    if "/" in s:
        s = s.split("/")[-1].strip()
    s = re.sub(r"^\[\s*рисунок\s*\]\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*[-–—]\s*", " — ", s)
    s = re.sub(r"\s+", " ", s).strip(" .")
    return s

def _parse_figure_title(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Извлекаем номер и хвост подписи из строки вида 'Рисунок 2.3 — Название'.
    Возвращает (num, title).
    """
    t = (text or "").strip()
    m = _FIG_TITLE_RE.search(t)
    if not m:
        return (None, None)
    raw_num = (m.group(1) or "").strip()
    raw_num = raw_num.replace(" ", "")
    num = raw_num.replace(",", ".") or None
    title = (m.group(2) or "").strip() or None
    return (num, title)

def _distinct_figure_basenames(uid: int, doc_id: int) -> List[str]:
    """
    Собираем уникальные имена «рисунков» по section_path (совместимо со старыми индексами,
    где использовался префикс '[Рисунок]' в тексте).
    """
    con = get_conn()
    cur = con.cursor()
    if _table_has_columns(con, "chunks", ["element_type"]):
        cur.execute(
            """
            SELECT DISTINCT section_path
            FROM chunks
            WHERE owner_id=? AND doc_id=? AND element_type='figure'
            ORDER BY section_path ASC
            """,
            (uid, doc_id),
        )
    else:
        cur.execute(
            """
            SELECT DISTINCT section_path
            FROM chunks
            WHERE owner_id=? AND doc_id=? AND text LIKE '[Рисунок]%%'
            ORDER BY section_path ASC
            """,
            (uid, doc_id),
        )
    rows = cur.fetchall()
    con.close()
    return [r["section_path"] for r in rows if r["section_path"]]

def count_figures(uid: int, doc_id: int) -> int:
    """Сколько всего «рисунков» в документе (по уникальным section_path)."""
    return len(_distinct_figure_basenames(uid, doc_id))

def list_figures(uid: int, doc_id: int, limit: int = 25) -> Dict[str, object]:
    """
    Возвращает короткий список «рисунков»:
      { "count": N, "list": [ 'Рисунок 2.1 — Схема …', ... ], "more": M }
    """
    bases = _distinct_figure_basenames(uid, doc_id)
    items: List[str] = []
    for base in bases:
        # пробуем извлечь номер/хвост из хвоста section_path
        tail = _last_segment(base)
        num, title = _parse_figure_title(tail)
        if num and title:
            items.append(f"Рисунок {num} — {_shorten(title)}")
        elif num:
            items.append(f"Рисунок {num}")
        elif tail:
            items.append(_shorten(tail))
        else:
            items.append("Рисунок")
    items = [x for x in items if x]
    return {
        "count": len(bases),
        "list": items[:limit],
        "more": max(0, len(items) - limit),
    }

def _collect_images_for_section(cur, uid: int, doc_id: int, section_path: str) -> List[str]:
    """Подтягиваем images из attrs по всем чанкам данной секции (на случай, если не в первом)."""
    images: List[str] = []
    try:
        cur.execute(
            """
            SELECT attrs FROM chunks
            WHERE owner_id=? AND doc_id=? AND section_path=? AND attrs IS NOT NULL
            ORDER BY id ASC LIMIT 10
            """,
            (uid, doc_id, section_path),
        )
        rows = cur.fetchall() or []
        for rr in rows:
            try:
                obj = json.loads(rr["attrs"] or "{}")
                imgs = obj.get("images") or []
                for p in imgs:
                    if p and p not in images:
                        images.append(p)
            except Exception:
                continue
    except Exception:
        pass
    return images

def describe_figures_by_numbers(
    uid: int,
    doc_id: int,
    numbers: List[str],
    sample_chunks: int = 2,
    *,
    use_vision: bool = True,
    lang: str = "ru",
    vision_first_image_only: bool = True,
) -> List[Dict[str, Any]]:
    """
    Возвращает карточки по явным номерам «рисунков».

    Формат одной карточки:
      {
        "num": "2.3",
        "display": "Рисунок 2.3 — Название",
        "where": {"page": 7, "section_path": ".../Рисунок 2.3 — Название"},
        "images": ["/uploads/.."],
        "highlights": ["краткая строка контекста", ...],
        "vision": {"description": "...", "tags": [...] }  # если use_vision и есть изображение
      }
    """
    if not numbers:
        return []

    # нормализуем N,M → N.M
    want = sorted({str(n).replace(",", ".").strip() for n in numbers if n and str(n).strip()})
    con = get_conn()
    cur = con.cursor()
    cards: List[Dict[str, Any]] = []

    has_ext = _table_has_columns(con, "chunks", ["element_type", "attrs"])

    for num in want:
        # 1) ищем по element_type='figure' и секции с нужным номером
        found = None
        if has_ext:
            cur.execute(
                """
                SELECT page, section_path, text, attrs
                FROM chunks
                WHERE owner_id=? AND doc_id=? AND element_type='figure'
                  AND (section_path LIKE ? COLLATE NOCASE
                       OR text LIKE ? COLLATE NOCASE)
                ORDER BY id ASC LIMIT 1
                """,
                (uid, doc_id, f"%Рисунок {num}%", f"%Рисунок {num}%"),
            )
            found = cur.fetchone()
        # 2) fallback для старых индексов — по тексту
        if not found:
            sel = "SELECT page, section_path, text" + (", attrs" if has_ext else "")
            cur.execute(
                f"""
                {sel}
                FROM chunks
                WHERE owner_id=? AND doc_id=? AND (text LIKE '[Рисунок]%%' OR section_path LIKE '%%Рисунок %%')
                  AND (section_path LIKE ? COLLATE NOCASE OR text LIKE ? COLLATE NOCASE)
                ORDER BY id ASC LIMIT 1
                """,
                (uid, doc_id, f"%Рисунок {num}%", f"%Рисунок {num}%"),
            )
            found = cur.fetchone()

        if not found:
            # не нашли — пропускаем номер
            continue

        page = found["page"]
        sec = found["section_path"]
        tail = _last_segment(sec)
        n_p, title_p = _parse_figure_title(tail)
        display = f"Рисунок {n_p or num}" + (f" — {title_p}" if title_p else "")

        # 2a) извлекаем images из attrs найденного чанка + по всей секции
        images: List[str] = []
        if has_ext and "attrs" in found.keys() and found["attrs"]:
            try:
                obj = json.loads(found["attrs"] or "{}")
                imgs = obj.get("images") or []
                for p in imgs:
                    if p and p not in images:
                        images.append(p)
            except Exception:
                pass
        # добираем изображения из других чанков этой же секции (если есть)
        images_extra = _collect_images_for_section(cur, uid, doc_id, sec)
        for p in images_extra:
            if p not in images:
                images.append(p)

        # 3) собираем 1–2 ближайших текстовых кусочка как «highlights»
        cur.execute(
            """
            SELECT text FROM chunks
            WHERE owner_id=? AND doc_id=? AND section_path=?
            ORDER BY id ASC LIMIT ?
            """,
            (uid, doc_id, sec, max(1, sample_chunks)),
        )
        rows = cur.fetchall() or []
        highlights: List[str] = []
        for rr in rows:
            t = (rr["text"] or "").strip()
            # чистим возможный префикс "[Рисунок] ..."`
            t = re.sub(r"^\[\s*Рисунок\s*\]\s*", "", t, flags=re.IGNORECASE)
            if t:
                highlights.append(_shorten(t, 200))
        # дополнительно, если пусто — берём любой соседний чанк из той же страницы
        if not highlights:
            cur.execute(
                """
                SELECT text FROM chunks
                WHERE owner_id=? AND doc_id=? AND page=?
                ORDER BY id ASC LIMIT 1
                """,
                (uid, doc_id, page),
            )
            rr = cur.fetchone()
            if rr and rr["text"]:
                highlights.append(_shorten(rr["text"], 200))

        card: Dict[str, Any] = {
            "num": num,
            "display": display,
            "where": {"page": page, "section_path": sec},
            "images": images,
            "highlights": highlights[:2],
        }

        # 4) vision-описание (опционально)
        if use_vision and images:
            try:
                img_for_vision = images[0] if vision_first_image_only else images
                vision = vision_describe(img_for_vision, lang=lang)
                card["vision"] = vision
            except Exception:
                # мягкая деградация
                card["vision"] = {"description": "описание изображения недоступно.", "tags": []}

        cards.append(card)

    con.close()
    return cards

# =======================================================================
#        НОВОЕ: Таблицы — поиск/раскрытие и «все значения»
# =======================================================================

# «Таблица 2.1 — ...», «Table A.1 — ...»
_TABLE_TITLE_RE = re.compile(
    r"(?i)\b(?:табл(?:ица)?|table)\s+([A-Za-zА-Яа-я]?\s*\d+(?:[.,]\d+)*)\b(?:\s*[—\-–:]\s*(.+))?"
)
_ROW_TAG_RE = re.compile(r"\[\s*row\s+(\d+)\s*\]", re.IGNORECASE)

def _num_norm(s: str | None) -> str:
    s = (s or "").strip()
    s = s.replace(" ", "").replace(",", ".")
    return s

def _table_base_from_section(section_path: str) -> str:
    """Убираем хвост ' [row k]' если присутствует."""
    if not section_path:
        return ""
    pos = section_path.lower().find(" [row ")
    return section_path if pos < 0 else section_path[:pos]

def _parse_table_title(text: str) -> Tuple[Optional[str], Optional[str]]:
    t = (text or "").strip()
    m = _TABLE_TITLE_RE.search(t)
    if not m:
        return (None, None)
    num = _num_norm(m.group(1))
    title = (m.group(2) or "").strip() or None
    return (num or None, title)

def _extract_table_numbers_from_query(q: str) -> List[str]:
    out: List[str] = []
    for m in re.finditer(r"(?:табл(?:ица)?|table)\s*(?:№\s*)?([A-Za-zА-Яа-я]?\s*\d+(?:[.,]\d+)*)", q or "", re.IGNORECASE):
        out.append(_num_norm(m.group(1)))
    return [x for x in out if x]

def _extract_figure_numbers_from_query(q: str) -> List[str]:
    out: List[str] = []
    for m in re.finditer(r"(?:рис(?:унок)?|fig(?:ure)?\.?)\s*(?:№\s*)?([A-Za-zА-Яа-я]?\s*\d+(?:[.,]\d+)*)", q or "", re.IGNORECASE):
        out.append(_num_norm(m.group(1)))
    return [x for x in out if x]

_ALL_VALUES_HINT = re.compile(
    r"(все|всю|полные|полный|целиком|полностью)\s+(значени|строк|столбц|данн|таблиц)|all\s+values|entire\s+table|full\s+table",
    re.IGNORECASE
)

def _wants_all_values(ask: str) -> bool:
    return bool(_ALL_VALUES_HINT.search(ask or ""))

def _find_table_bases_by_number(pack: dict, num: str) -> List[str]:
    """
    Ищем секции таблиц по номеру:
      - по attrs.caption_num/label,
      - по вхождению 'Таблица N' в section_path/text.
    Возвращаем список базовых section_path (без [row k]).
    """
    bases: List[str] = []
    needle = _num_norm(num)
    for m in pack["meta"]:
        et = _chunk_type(m)
        if et not in {"table", "table_row"}:
            continue
        attrs = m.get("attrs") or {}
        cand = _num_norm(str(attrs.get("caption_num") or attrs.get("label") or ""))
        if cand and cand == needle:
            base = _table_base_from_section(m.get("section_path") or "")
            if base and base not in bases:
                bases.append(base)
            continue
        # fallback по тексту/пути
        sp = (m.get("section_path") or "")
        tx = (m.get("text") or "")
        if re.search(rf"(?i)\bтаблица\s+{re.escape(needle)}\b", sp) or re.search(rf"(?i)\bтаблица\s+{re.escape(needle)}\b", tx):
            base = _table_base_from_section(sp)
            if base and base not in bases:
                bases.append(base)
    return bases

def _row_index(section_path: str) -> int:
    m = _ROW_TAG_RE.search(section_path or "")
    try:
        return int(m.group(1)) if m else 10**9
    except Exception:
        return 10**9

def _gather_table_chunks_for_base(pack: dict, base: str, *, include_header: bool = True, row_limit: Optional[int] = None) -> List[Dict]:
    """
    Собираем чанк заголовка таблицы (если есть) и все её строки (table_row) по base.
    Возвращаем элементы в формате retrieve(): {id, page, section_path, text, score}
    """
    rows: List[Dict] = []
    header: Optional[Dict] = None

    for m in pack["meta"]:
        sp = (m.get("section_path") or "")
        if not sp:
            continue
        if _table_base_from_section(sp) != base:
            continue
        et = _chunk_type(m)
        item = {
            "id": m["id"],
            "page": m["page"],
            "section_path": sp,
            "text": (m.get("text") or "").strip(),
            "score": 0.9 if et == "table" else 0.85
        }
        if et == "table":
            if include_header and not header:
                header = item
        elif et == "table_row" or "[row " in sp.lower():
            rows.append(item)

    rows.sort(key=lambda x: _row_index(x.get("section_path") or ""))

    if row_limit is not None and row_limit >= 0:
        rows = rows[:row_limit]

    out: List[Dict] = []
    if include_header and header:
        out.append(header)
    out.extend(rows)
    return out

def _inject_special_sources_for_item(pack: dict, ask: str, used_ids: set[int]) -> List[Dict]:
    """
    Если подпункт явно ссылается на «Таблица N» / «Рисунок N», добавляем соответствующие
    источники (включая все строки таблицы при «дай все значения»).
    """
    added: List[Dict] = []

    # Таблицы
    table_nums = _extract_table_numbers_from_query(ask)
    want_all = _wants_all_values(ask)
    for num in table_nums:
        bases = _find_table_bases_by_number(pack, num)
        for base in bases:
            chunks = _gather_table_chunks_for_base(pack, base, include_header=True, row_limit=(None if want_all else 8))
            for ch in chunks:
                if ch["id"] in used_ids:
                    continue
                ch["for_item"] = None  # будет выставлено выше по месту использования
                added.append(ch)
                used_ids.add(ch["id"])

    # Рисунки: берём все чанки этой секции (обычно один-три), чтобы было описание + подпись
    fig_nums = _extract_figure_numbers_from_query(ask)
    if fig_nums:
        for num in fig_nums:
            for m in pack["meta"]:
                if _chunk_type(m) != "figure":
                    continue
                sp = m.get("section_path") or ""
                tx = m.get("text") or ""
                # распознаём номер из tail'а
                tail = _last_segment(sp)
                n_p, _ = _parse_figure_title(tail)
                if _num_norm(n_p) != _num_norm(num) and not re.search(rf"(?i)\bрисунок\s+{re.escape(_num_norm(num))}\b", tx):
                    continue
                # добавляем все чанки этой секции (на случай, если текст разбит)
                sec = sp
                for mm in pack["meta"]:
                    if (mm.get("section_path") or "") == sec:
                        ch = {
                            "id": mm["id"],
                            "page": mm["page"],
                            "section_path": mm["section_path"],
                            "text": (mm.get("text") or "").strip(),
                            "score": 0.88
                        }
                        if ch["id"] in used_ids:
                            continue
                        ch["for_item"] = None
                        added.append(ch)
                        used_ids.add(ch["id"])
    return added

# =======================================================================
#        НОВОЕ: Coverage-aware извлечение под многопунктные вопросы
#               с интеграцией таблиц/рисунков
# =======================================================================

# Эвристики распознавания подпунктов (совпадают с ace-планировщиком по смыслу)
_ENUM_LINE = re.compile(r"(?m)^\s*(?:\d+[).\)]|[-—•])\s+")
_ENUM_INLINE = re.compile(r"(?:\s|^)\(?\d{1,2}[)\.]\s+")

def plan_subitems(question: str, *, min_items: int = 2, max_items: int = 12) -> List[str]:
    """
    Пытаемся вытащить подпункты из длинного вопроса:
      - маркеры в начале строк (1. … / • …);
      - inline-формат «1) 2) 3) …»;
      - разделение по ';' как крайний случай.
    Возвращаем список коротких формулировок.
    """
    q = (question or "").strip()
    items: List[str] = []

    # 1) построчно с маркерами
    if _ENUM_LINE.search(q):
        for line in q.splitlines():
            if _ENUM_LINE.match(line):
                items.append(re.sub(r"^\s*(?:\d+[).\)]|[-—•])\s+", "", line).strip())

    # 2) inline «1) 2) 3)»
    if not items and _ENUM_INLINE.search(q):
        buf, cur = [], None
        for token in re.finditer(r"\(?\d{1,2}[)\.]\s+|[^()]+", q):
            s = token.group(0)
            if re.fullmatch(r"\(?\d{1,2}[)\.]\s+", s):
                if cur:
                    items.append(cur.strip())
                cur = ""
            else:
                cur = (cur or "") + s
        if cur:
            items.append(cur.strip())

    # 3) по ';'
    if not items and q.count(";") >= 2:
        items = [p.strip() for p in q.split(";") if p.strip()]

    # чистка
    items = [re.sub(r"\s+", " ", it).strip(" .;—-") for it in items if it and len(it.strip()) >= 2]
    if len(items) < min_items:
        return []
    return items[:max_items]

def retrieve_for_items(
    owner_id: int,
    doc_id: int,
    items: List[str],
    *,
    per_item_k: int = 2,
    prelim_factor: int = 4,
    avoid_refs_when_not_asked: bool = True,
    overall_backfill_from_query: Optional[str] = None,
    backfill_k: int = 4,
) -> Dict[str, List[Dict]]:
    """
    Для каждого подпункта возвращаем до per_item_k релевантных чанков.
    Опционально добираем несколько общих чанков для «связности».
    Возвращает словарь id->список чанков (каждый чанк как в retrieve()).
    """
    if not items:
        return {}

    pack = _load_doc(owner_id, doc_id)
    by_id: Dict[str, List[Dict]] = {}
    used_ids: set[int] = set()

    for idx, ask in enumerate(items, start=1):
        # 0) Сначала — специальные источники: таблицы/рисунки, если явно упомянуты
        special = _inject_special_sources_for_item(pack, ask, used_ids)
        for sp in special:
            sp["for_item"] = str(idx)
        picked: List[Dict] = list(special)

        # 1) Затем обычный семантический поиск
        rescored = _score_and_rank(pack, ask, prelim_k=max(per_item_k * prelim_factor, 16))
        for i, sc in rescored:
            m = pack["meta"][int(i)]
            if m["id"] in used_ids:
                continue
            if avoid_refs_when_not_asked and _chunk_type(m) == "reference":
                # пропускаем источники, если подпункт явно не про них
                if not re.search(r"\b(источник|литератур|reference|bibliograph)\w*\b", ask.lower()):
                    continue
            picked.append(
                {
                    "id": m["id"],
                    "page": m["page"],
                    "section_path": m["section_path"],
                    "text": (m["text"] or "").strip(),
                    "score": float(sc),
                    "for_item": str(idx),
                }
            )
            used_ids.add(m["id"])
            # ВНИМАНИЕ: не обрезаем по per_item_k здесь — пусть bucket содержит всё полезное.
            # Итоговое сечение пойдёт на этапе merge в retrieve_coverage.

        by_id[str(idx)] = picked

    # Дополнительная «склейка» общим контекстом из исходного вопроса — backfill
    if overall_backfill_from_query:
        rescored_q = _score_and_rank(pack, overall_backfill_from_query, prelim_k=max(backfill_k * 3, 16))
        extra = []
        for i, sc in rescored_q:
            m = pack["meta"][int(i)]
            if m["id"] in used_ids:
                continue
            # мягко избегаем reference
            if _chunk_type(m) == "reference":
                continue
            extra.append(
                {
                    "id": m["id"],
                    "page": m["page"],
                    "section_path": m["section_path"],
                    "text": (m["text"] or "").strip(),
                    "score": float(sc),
                    "for_item": None,
                }
            )
            used_ids.add(m["id"])
            if len(extra) >= backfill_k:
                break
        if extra:
            by_id["_backfill"] = extra

    return by_id

def retrieve_coverage(
    owner_id: int,
    doc_id: int,
    question: str,
    subitems: Optional[List[str]] = None,
    *,
    per_item_k: int = 2,
    prelim_factor: int = 4,
    backfill_k: int = 4,
) -> Dict[str, Any]:
    """
    Coverage-aware выборка под многопунктный вопрос.
    Если subitems не переданы — попытаемся извлечь их из вопроса.
    Возвращает:
      {
        "items": ["…", "…", ...],                 # подпункты (может быть пусто)
        "by_item": {"1":[...], "2":[...], ...},   # чанки по подпунктам (как в retrieve), поле for_item у чанка
        "snippets": [...],                        # все чанки в одном списке, упорядоченные round-robin (+добавки для «все значения»)
      }
    """
    items = list(subitems or plan_subitems(question)) if (subitems or question) else []
    by_item = {}
    if items:
        by_item = retrieve_for_items(
            owner_id,
            doc_id,
            items,
            per_item_k=per_item_k,
            prelim_factor=prelim_factor,
            overall_backfill_from_query=question,
            backfill_k=backfill_k,
        )
        # round-robin порядок: 1-й из каждого подпункта, затем 2-й
        buckets = [by_item.get(str(i + 1), []) for i in range(len(items))]
        merged: List[Dict] = []
        for r in range(per_item_k):
            for b in buckets:
                if r < len(b):
                    merged.append(b[r])

        # Для подпунктов вида «дай все значения» — добавляем ОСТАЛЬНЫЕ строчки их bucket
        for i, ask in enumerate(items, start=1):
            if _wants_all_values(ask):
                bucket = by_item.get(str(i), [])
                # уже добавлены первые per_item_k, доклеим остаток
                if len(bucket) > per_item_k:
                    merged.extend(bucket[per_item_k:])

        # добавим backfill в конце
        if by_item.get("_backfill"):
            merged.extend(by_item["_backfill"])
        return {"items": items, "by_item": by_item, "snippets": merged}

    # Если подпункты не распознаны — обычный retrieve с небольшим расширением k
    base = retrieve(owner_id, doc_id, question, top_k=max(10, backfill_k * 2))
    return {"items": [], "by_item": {"_single": base}, "snippets": base}

def build_context_coverage(
    snippets: List[Dict],
    * ,
    items_count: Optional[int] = None,
    base_chars: int = 6000,
    per_item_bonus: int = 900,
    hard_limit: int = 18000,
) -> str:
    """
    Адаптивная сборка контекста под многопунктные вопросы:
      - итоговый лимит = base_chars + per_item_bonus*(items_count-1), но не больше hard_limit;
      - тексты склеиваются в round-robin порядке по подпунктам (если есть for_item),
        чтобы каждый подпункт появился в начале контекста хотя бы раз.
      - удаляются дубликаты (по id/section_path/тексту).
    """
    if not snippets:
        return ""

    # Считаем количество подпунктов из меток for_item, если явно не задано
    if items_count is None:
        items_count = len({s.get("for_item") for s in snippets if s.get("for_item")}) or 1

    max_chars = min(hard_limit, int(base_chars + max(0, items_count - 1) * per_item_bonus))

    # Группировка по подпунктам
    groups: Dict[str, List[Dict]] = {}
    extras: List[Dict] = []
    for s in snippets:
        fid = s.get("for_item")
        if fid:
            groups.setdefault(str(fid), []).append(s)
        else:
            extras.append(s)

    # round-robin
    ordered: List[Dict] = []
    if groups:
        # выровняем порядок ключей групп по возрастанию
        keys = sorted(groups.keys(), key=lambda x: (len(x), x))
        round_idx = 0
        more = True
        while more:
            more = False
            for k in keys:
                bucket = groups.get(k, [])
                if round_idx < len(bucket):
                    ordered.append(bucket[round_idx])
                    more = True
            round_idx += 1
        # extras в конце
        ordered.extend(extras)
    else:
        ordered = snippets[:]

    # дедуп по id/section_path/тексту
    seen_ids: set[int] = set()
    seen_keys: set[str] = set()
    parts: List[str] = []
    total = 0
    for s in ordered:
        sid = s.get("id")
        spath = (s.get("section_path") or "").strip()
        raw = (s.get("text") or "")
        block = _clean_for_ctx(raw)
        if not block:
            continue

        k = f"{sid or 0}|{spath}|{hash(block)}"
        if sid and sid in seen_ids:
            continue
        if k in seen_keys:
            continue

        if total + len(block) > max_chars:
            remaining = max_chars - total
            if remaining <= 0:
                break
            block = block[:remaining]

        parts.append(block)
        total += len(block)
        if sid:
            seen_ids.add(sid)
        seen_keys.add(k)

        if total >= max_chars:
            break

    return "\n\n".join(parts)
