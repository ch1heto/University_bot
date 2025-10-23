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
    if N == 0:
        return []

    # Возьмём побольше кандидатов для переранжа
    prelim_k = max(top_k * 3, top_k, 16)
    prelim_k = min(prelim_k, N)
    if prelim_k <= 0:
        return []

    # Быстрый выбор top-prelim_k без полного сортирования
    part_idx = np.argpartition(sims, -prelim_k)[-prelim_k:]
    idx = part_idx[np.argsort(-sims[part_idx])]

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
            score += 0.12
        if sig["ask_figure"] and ctype == "figure":
            score += 0.14
        if sig["ask_heading"] and ctype == "heading":
            score += 0.06
        if sig["ask_sources"] and ctype == "reference":
            score += 0.35
        # если источники не просили — пусть уходят вниз, но не исчезают полностью
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

    # Финальная сортировка по новым скорингам
    rescored.sort(key=lambda x: -x[1])

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
            # чистим возможный префикс "[Рисунок] ..."
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
