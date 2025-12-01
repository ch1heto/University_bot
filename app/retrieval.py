# app/retrieval.py
import re
import json
import numpy as np
from typing import Optional, List, Dict, Tuple, Any

from .db import get_conn, get_figures_for_doc
from .polza_client import embeddings, vision_describe  # эмбеддинги + vision

# Опциональные хелперы по номерам таблиц/рисунков и «все значения»
try:
    from .intents import extract_table_numbers, extract_figure_numbers, TABLE_ALL_VALUES_RE  # type: ignore
except Exception:
    extract_table_numbers = None      # type: ignore
    extract_figure_numbers = None     # type: ignore
    TABLE_ALL_VALUES_RE = re.compile(  # type: ignore
        r"(?i)\b(все|всю|целиком|полностью|полная)\b.*\b(таблиц\w*|таблица|значени\w*|данн\w*|строк\w*|колон\w*)\b"
        r"|(?:\ball\b.*\b(table|values|rows|columns)\b|\bfull\s+(table|values)\b|\bentire\s+table\b)"
    )

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

    meta: List[Dict[str, Any]] = []
    vecs: List[np.ndarray] = []
    dim: Optional[int] = None

    for r in rows:
        emb = r["embedding"]
        if emb is None:
            # Пропускаем пустые эмбеддинги
            continue

        v = np.frombuffer(emb, dtype=np.float32)
        # Защита от старых/битых эмбеддингов с другим размером
        if dim is None:
            dim = v.size
        elif v.size != dim:
            continue

        if has_ext:
            et = (r["element_type"] or "").lower()
            attrs_raw = r["attrs"]
            try:
                attrs = json.loads(attrs_raw) if attrs_raw else {}
            except Exception:
                attrs = {}
        else:
            et = ""
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
        vecs.append(v)


    if not vecs:
        pack = {"mat": np.zeros((0, 1), np.float32), "meta": []}
        _DOC_CACHE[key] = pack
        return pack

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
        "ask_figure":  bool(
            re.search(r"\b(рис\.?|рисунк\w*|figure|fig\.?|диаграм\w*|diagram)\b", ql)
            or re.search(r"\bна\s+рисунк\w*\b", ql)
        ),
        "ask_heading": bool(re.search(r"\b(глава|раздел|введение|заключение|heading|chapter|section)\b", ql)),
        "ask_sources": bool(re.search(r"\bисточник(?:и|ов)?\b|\bсписок\s+литературы\b|\bбиблиограф", ql) or
                            re.search(r"\breferences?\b|bibliograph\w*", ql)),
    }

# ---------------------------
# Базовые эмбеддинги
# ---------------------------

def _embed_query(query: str) -> Optional[np.ndarray]:
    """
    Безопасно получаем эмбеддинг вопроса.
    В случае ошибки или пустой строки вернём None.
    """
    q_str = (query or "").strip()
    if not q_str:
        return None
    try:
        vec = embeddings([q_str])[0]
        q = np.asarray(vec, dtype=np.float32)
        norm = float(np.linalg.norm(q))
        if not np.isfinite(norm) or norm == 0.0:
            return None
        q /= norm
        return q.astype(np.float32, copy=False)
    except Exception:
        return None

# ---------------------------
# Регексы-помощники для ID-aware
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
    m = re.search(r"(табл(?:ица)?|table)\s*(?:№\s*|no\.?\s*|номер\s*)?([A-Za-zА-Яа-я]?\s*\d+(?:[.,]\d+)*)", ql, re.IGNORECASE)
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
    return rf"(?:табл(?:ица)?|table)\s*\.?\s*(?:№\s*|no\.?\s*|номер\s*)?{pat_num}"

def _mk_figure_pattern(q: str) -> Optional[str]:
    """
    Поиск 'рисунок 2.4' / 'рис. 2,4' / 'figure 3' / 'fig. 1' / 'рис. А.1'.
    Возвращаем паттерн regex или None.
    """
    ql = (q or "")
    m = re.search(r"(рис(?:унок)?|fig(?:ure)?\.?)\s*(?:№\s*|no\.?\s*|номер\s*)?([A-Za-zА-Яа-я]?\s*\d+(?:[.,]\d+)*)", ql, re.IGNORECASE)
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
    return rf"(?:рис(?:унок)?|fig(?:ure)?\.?)\s*\.?\s*(?:№\s*|no\.?\s*|номер\s*)?{pat_num}"

# ---------------------------
# Внутренний скорер/ранжирование (ID-aware + область)
# ---------------------------

def _score_and_rank(
    pack: dict,
    query: str,
    *,
    prelim_k: int = 48,
    section_prefix: Optional[str] = None,
    element_types: Optional[List[str]] = None,
) -> List[Tuple[int, float]]:
    """
    Возвращает список индексов чанков и их скорингов, отсортированный по убыванию.
    Используем в retrieve() и coverage-вариантах.
    Можно ограничить кандидатов областью документа (section_prefix) и/или типами чанков.
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

    # --- кандидаты по области/типам ---
    cand_idx = list(range(N))
    if section_prefix or element_types:
        spfx = section_prefix or ""
        want_types = {t.lower() for t in (element_types or [])}
        cand_idx = []
        for i, m in enumerate(pack["meta"]):
            ok = True
            if spfx and not str(m.get("section_path") or "").startswith(spfx):
                ok = False
            if ok and want_types and (_chunk_type(m) not in want_types):
                ok = False
            if ok:
                cand_idx.append(i)
        if not cand_idx:
            return []

    cand_idx = np.asarray(cand_idx, dtype=np.int64)
    sims_sub = sims[cand_idx]

    prelim_k = max(min(prelim_k, sims_sub.shape[0]), 1)
    part_idx_local = np.argpartition(sims_sub, -prelim_k)[-prelim_k:]
    order_local = part_idx_local[np.argsort(-sims_sub[part_idx_local])]
    idx = cand_idx[order_local]

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
        score = float(sims[int(i)])

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

        # Пустые/очень короткие ТЕКСТОВЫЕ чанки — мягкий штраф
        # (табличные/рисунки не режем, у них строки часто короткие по природе)
        if ctype in {"text", "heading", "page"} and len((text or "").strip()) < 30:
            score -= 0.04

        rescored.append((int(i), score))

    rescored.sort(key=lambda x: -x[1])
    return rescored

# ---------------------------
# Основной RAG-поиск (ID-aware + область)
# ---------------------------

def retrieve(
    owner_id: int,
    doc_id: int,
    query: str,
    top_k: int = 8,
    *,
    section_prefix: Optional[str] = None,
    element_types: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Возвращает top-k чанков по косинусному сходству с вопросом (с лёгким переранжированием).
    Можно ограничивать областью (section_prefix) и/или типами чанков (element_types).
    Каждый элемент: {id, page, section_path, text, score}.

    NEW:
      - перед обычным семантическим поиском пробуем подтянуть «специальные» источники
        для явных ссылок вида «Таблица 2.2», «все значения таблицы 3.1», «Рисунок 6» —
        через _inject_special_sources_for_item();
      - все источники (спец + обычные + синтетические по рисункам) дедуплицируем по id.
    """

    # 🔧 Авто-увеличение top_k для длинных и многочастных запросов,
    # чтобы модель видела больше контекста при сложных вопросах.
    q_norm = (query or "").strip()
    if q_norm:
        long_query = len(q_norm) > 200
        many_parts = (q_norm.count(" и ") + q_norm.count(",") + q_norm.count(";")) > 3
        multi_questions = q_norm.count("?") > 1

        if long_query or many_parts or multi_questions:
            # Сложный / составной запрос — поднимаем верхнюю границу
            top_k = max(top_k, 12)
        elif len(q_norm) > 100:
            # Умеренно длинный запрос — немного расширяем выборку
            top_k = max(top_k, 10)

    pack = _load_doc(owner_id, doc_id)

    # Специальные источники: таблицы/рисунки, если явно упомянуты в вопросе
    used_ids: set[int] = set()
    special: List[Dict[str, Any]] = _inject_special_sources_for_item(
        pack,
        query,
        used_ids,
        doc_id=doc_id,
    ) or []  # <-- если функция вернула None, подставляем пустой список

    # for_item для одиночного вопроса нам не нужен
    for sp in special:
        sp["for_item"] = None


    rescored = _score_and_rank(
        pack,
        query,
        prelim_k=max(top_k * 3, 16),
        section_prefix=section_prefix,
        element_types=element_types,
    )

    # Если ни семантики, ни спец-источников не нашли — выходим
    if not rescored and not special:
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

    # 1) сначала добавляем спец-фрагменты (таблицы/рисунки, «все значения таблицы» и т.п.)
    for sp in special:
        if sp["id"] in used_ids:
            continue
        out.append(sp)
        used_ids.add(sp["id"])

    # 2) если вопрос ссылается на «Рисунок N» — добавляем синтетические сниппеты с анализом рисунков
    fig_snips = _figure_context_snippets_for_query(doc_id, query, max_items=2)
    for fs in fig_snips:
        fid = int(fs.get("id", 0))
        if fid in used_ids:
            continue
        out.append(fs)
        used_ids.add(fid)

    # 3) затем добираем обычные чанки до top_k
    for i, sc in best:
        if len(out) >= top_k:
            break
        m = pack["meta"][int(i)]
        mid = m["id"]
        if mid in used_ids:
            continue
        out.append(
            {
                "id": mid,
                "page": m["page"],
                "section_path": m["section_path"],
                "text": (m["text"] or "").strip(),
                "score": float(sc),
            }
        )
        used_ids.add(mid)

    return out


# --- NEW: chart_matrix/chart_data → rows/values_str (для диаграмм в DOCX)
def _chart_rows_from_attrs(attrs: str | dict | None) -> list[dict] | None:
    """
    Достаём rows из attrs и приводим к каноническому виду:
      {
        "label": str,        # подпись категории (возможен префикс серии)
        "value": float|Any,  # числовое значение (если есть)
        "unit": str|None,    # "%", "шт", и т.п.
        "value_raw": str|None,   # как в исходном XML/таблице
        "series_name": str|None,
        "category": str|None,
      }

    Поддерживаем:
      - нормализованный OOXML-вид: attrs["chart_matrix"] =
            {"categories":[...],
             "series":[{"name":..., "values":[...], "unit":"%"/None, ...}],
             "unit": "...", "meta": {...}}
      - новый формат: attrs["chart_data"] = [ {...}, {...} ]
      - старый формат: attrs["chart_data"] = [ {"label": ..., "value": ...}, ... ]
      - dict вида {"categories":[...], "series":[{"values":[...], "unit":"%"}]}
    """
    # attrs может быть и строкой JSON, и уже dict'ом
    if isinstance(attrs, str):
        try:
            a = json.loads(attrs or "{}")
        except Exception:
            return None
    elif isinstance(attrs, dict):
        a = attrs
    else:
        return None

    rows_norm: list[dict] = []

    # ---- случай 0: нормализованный chart_matrix ----
    cm = a.get("chart_matrix")
    if isinstance(cm, dict):
        cats = cm.get("categories") or []
        series_list = cm.get("series") or []
        if isinstance(cats, list) and isinstance(series_list, list) and cats and series_list:
            cats = [str(c) for c in cats]
            multi = len(series_list) > 1
            default_unit = cm.get("unit")
            for s in series_list:
                if not isinstance(s, dict):
                    continue
                s_name = (s.get("name") or s.get("series_name") or "").strip() or None
                vals = s.get("values") or s.get("data") or []
                unit = s.get("unit") or default_unit
                for idx, v in enumerate(vals):
                    cat = cats[idx] if idx < len(cats) else f"Категория {idx + 1}"
                    label = f"{s_name}: {cat}" if (multi and s_name) else cat
                    rows_norm.append(
                        {
                            "label": label,
                            "value": v,
                            "unit": unit,
                            "value_raw": None,
                            "series_name": s_name,
                            "category": cat,
                        }
                    )
        if rows_norm:
            return rows_norm

    # ---- случай 1: уже список (новый/старый формат chart_data) ----
    raw = (
        a.get("chart_data")
        or (a.get("chart") or {}).get("data")
        or a.get("data")
        or a.get("series")
    )

    if isinstance(raw, list) and raw:
        for item in raw:
            if isinstance(item, dict):
                label = (
                    item.get("label")
                    or item.get("category")
                    or item.get("name")
                    or ""
                )
                series_name = item.get("series_name") or item.get("series") or None
                category = item.get("category") or item.get("cat") or None

                value = item.get("value")
                value_raw = item.get("value_raw")
                unit = item.get("unit")

                if value_raw is None and isinstance(value, str):
                    value_raw = value

                rows_norm.append(
                    {
                        "label": str(label).strip(),
                        "value": value,
                        "unit": unit,
                        "value_raw": value_raw,
                        "series_name": series_name,
                        "category": category,
                    }
                )
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                rows_norm.append(
                    {
                        "label": str(item[0]),
                        "value": item[1],
                        "unit": None,
                        "value_raw": None,
                        "series_name": None,
                        "category": None,
                    }
                )

        return rows_norm or None

    # ---- случай 2: dict с categories/series ----
    if isinstance(raw, dict) and raw.get("categories") and raw.get("series"):
        cats = list(raw.get("categories") or [])
        s0 = (raw.get("series") or [{}])[0] or {}
        vals = list(s0.get("values") or s0.get("data") or [])
        unit = s0.get("unit")
        series_name = s0.get("name") or s0.get("series_name")

        for i in range(min(len(cats), len(vals))):
            rows_norm.append(
                {
                    "label": str(cats[i]),
                    "value": vals[i],
                    "unit": unit,
                    "value_raw": None,
                    "series_name": series_name,
                    "category": str(cats[i]),
                }
            )

        return rows_norm or None

    return None


def _format_chart_values(rows: list[dict]) -> str:
    """
    Формирует человекочитаемые строки значений диаграммы из нормализованных rows.
    """
    def _fmt_percent(v: float) -> str:
        if abs(v - round(v)) < 0.05:
            return f"{int(round(v))}%"
        s = f"{v:.2f}".rstrip("0").rstrip(".")
        return f"{s}%"

    def _fmt_number(v: float) -> str:
        return f"{v:.6g}"

    lines: list[str] = []

    for r in rows or []:
        lab = str(
            r.get("label")
            or r.get("name")
            or r.get("category")
            or ""
        ).strip()

        raw = r.get("value_raw")
        if isinstance(raw, (int, float)):
            raw = str(raw)
        raw = (raw or "").strip()

        unit = r.get("unit")
        value = r.get("value")
        num_val: float | None = None
        if isinstance(value, (int, float)):
            num_val = float(value)

        vstr = ""
        if raw:
            if unit == "%" and "%" not in raw and isinstance(num_val, float):
                vstr = _fmt_percent(num_val)
            else:
                vstr = raw
        elif isinstance(num_val, float):
            if unit == "%":
                vstr = _fmt_percent(num_val)
            else:
                vstr = _fmt_number(num_val)
        else:
            other = r.get("value")
            vstr = str(other) if other is not None else ""

        if not lab and not vstr:
            continue

        unit_suffix = ""
        if isinstance(unit, str) and unit.strip() and unit.strip() != "%":
            unit_suffix = f" {unit.strip()}"

        if lab and vstr:
            line = f"— {lab}: {vstr}{unit_suffix}"
        elif lab:
            line = f"— {lab}"
        else:
            line = f"— {vstr}{unit_suffix}"

        lines.append(line.strip())

    return "\n".join(lines)


def retrieve_in_area(owner_id: int, doc_id: int, query: str, section_prefix: str, top_k: int = 8) -> List[Dict]:
    """Семантический поиск, жёстко ограниченный заданной областью документа."""
    return retrieve(owner_id, doc_id, query, top_k=top_k, section_prefix=section_prefix)

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
# Поддерживаем: «Рисунок 3.2», «Рис. 3.2», «Рис.3.2.», «Figure 2», «Fig. 2»
_FIG_TITLE_RE = re.compile(
    r"(?i)\b(?:рис(?:\.|унок)?|fig(?:\.|ure)?)\s*([A-Za-zА-Яа-я]?\s*\d+(?:[.,]\d+)*\.?)\s*(?:[—\-–:\u2013\u2014]\s*(.+))?"
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
    t = (text or "").strip()
    m = _FIG_TITLE_RE.search(t)
    if not m:
        return (None, None)
    raw_num = (m.group(1) or "").strip().replace(" ", "")
    num = _num_norm(raw_num)
    title = (m.group(2) or "").strip() or None
    return (num or None, title)

def count_figures(uid: int, doc_id: int) -> int:
    """
    Сколько всего «рисунков» в документе.
    Теперь считаем по таблице figures, а не по chunks/regex.
    """
    figs = get_figures_for_doc(doc_id)
    return len(figs or [])


def list_figures(uid: int, doc_id: int, limit: int = 25) -> Dict[str, object]:
    """
    Короткий список «рисунков» на основе таблицы figures:
      { "count": N, "list": [ 'Рисунок 2.1 — Схема …', ... ], "more": M }
    """
    figs = get_figures_for_doc(doc_id) or []

    items: List[str] = []
    for f in figs:
        num = _num_norm(str(f.get("label") or f.get("figure_label") or f.get("number") or ""))
        caption = (f.get("caption") or "").strip()
        if num and caption:
            items.append(f"Рисунок {num} — {_shorten(caption)}")
        elif num:
            items.append(f"Рисунок {num}")
        elif caption:
            items.append(_shorten(caption))
        else:
            items.append("Рисунок")

    items = [x for x in items if x]
    count = len(items)
    return {
        "count": count,
        "list": items[:limit],
        "more": max(0, count - limit),
    }


def get_figure_record(
    doc_id: int,
    *,
    figure_label: Optional[str] = None,
    figure_id: Optional[int] = None,
    ensure_analysis: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Возвращает одну запись о рисунке из таблицы figures + результат анализа (vision_analyzer).

    Можно передать либо figure_id, либо номер рисунка (figure_label, напр. '2.3').
    Если ensure_analysis=True — при отсутствии анализа он будет посчитан и сохранён
    через figures.analyze_figure_for_db(..).
    """
    if figure_id is None and not figure_label:
        return None

    figs = get_figures_for_doc(doc_id)
    row: Optional[Dict[str, Any]] = None

    if figure_id is not None:
        for f in figs:
            if int(f.get("figure_id")) == int(figure_id):
                row = f
                break
    else:
        # ищем по номеру рисунка (label), с нормализацией '2.3'/'2,3'
        wanted = _num_norm(str(figure_label))
        for f in figs:
            lab = _num_norm(str(f.get("label") or f.get("figure_label") or ""))
            if lab and lab == wanted:
                row = f
                break

    if not row:
        return None

    analysis: Optional[Dict[str, Any]] = None
    if ensure_analysis:
        try:
            # ленивый импорт, чтобы избежать циклов
            from .figures import analyze_figure_for_db
            analysis = analyze_figure_for_db(
                doc_id=doc_id,
                figure_id=int(row["figure_id"]),
                intent="describe-with-numbers",
            )
        except Exception:
            analysis = None

    return {
        "figure": row,
        "analysis": analysis,
    }

def _figure_values_str_from_row(fig_row: Dict[str, Any]) -> Optional[str]:
    """
    Достаём values_str для диаграммы из attrs фигуры (таблица figures):
      - attrs.chart_data
      - или attrs.chart_matrix / data / series.
    Используем общие _chart_rows_from_attrs + _format_chart_values.
    """
    attrs = fig_row.get("attrs")
    if isinstance(attrs, str):
        try:
            attrs_obj = json.loads(attrs or "{}")
        except Exception:
            attrs_obj = {}
    elif isinstance(attrs, dict):
        attrs_obj = attrs
    else:
        attrs_obj = {}

    if not isinstance(attrs_obj, dict):
        return None

    rows = _chart_rows_from_attrs(attrs_obj)
    if not rows:
        return None
    return _format_chart_values(rows)

def _build_figure_context_text(rec: Dict[str, Any]) -> str:
    """
    Собираем человекочитаемый текст для включения в RAG-контекст из записи figure+analysis.
    Дополнительно подмешиваем values_str для диаграммы из attrs.chart_data/chart_matrix.
    """
    fig = rec.get("figure") or {}
    analysis = rec.get("analysis") or {}

    num = _num_norm(str(fig.get("label") or fig.get("figure_label") or "")) or None
    caption = (fig.get("caption") or "").strip()
    base_text = (analysis.get("text") or "").strip()

    prefix = f"Рисунок {num}" if num else "Рисунок"

    if not base_text and caption:
        base_text = caption
    if not base_text:
        return ""

    # Если caption есть и не входит в основное описание — добавим его в заголовок
    if caption and caption not in base_text:
        head = f"{prefix} — {caption}"
        if not base_text.startswith(head):
            text = f"{head}. {base_text}"
        else:
            text = base_text
    else:
        if not base_text.lower().startswith(prefix.lower()):
            text = f"{prefix}: {base_text}"
        else:
            text = base_text

    # Подмешиваем структурные значения диаграммы (если есть в attrs фигуры)
    vals_str = _figure_values_str_from_row(fig)
    if vals_str:
        # чтобы не задублировать, только если этих строк ещё нет в тексте
        if vals_str not in text:
            text = f"{text}\n\n{vals_str}".strip()

    meta = (analysis.get("meta") or {}) if isinstance(analysis, dict) else {}
    caveat = (meta.get("caveat") or "").strip()
    if caveat and caveat not in text:
        text = f"{text} {caveat}".strip()

    return text


def _figure_context_snippets_for_query(doc_id: int, query: str, max_items: int = 2) -> List[Dict[str, Any]]:
    """
    Если вопрос явно ссылается на «Рисунок N», подмешиваем в контекст 1–2
    синтетических сниппета с кратким описанием соответствующих рисунков.
    """
    nums = extract_figure_numbers(query) if extract_figure_numbers else []  # type: ignore
    if not nums:
        return []

    out: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for raw in nums:
        num = _num_norm(raw)
        if not num or num in seen:
            continue
        seen.add(num)

        rec = get_figure_record(doc_id, figure_label=num, ensure_analysis=True)
        if not rec or not rec.get("analysis"):
            continue

        text = _build_figure_context_text(rec)
        if not text:
            continue

        fig = rec["figure"]
        fid = int(fig.get("figure_id"))
        synthetic_id = -int(1_000_000 + fid)

        out.append(
            {
                "id": synthetic_id,
                "page": fig.get("page"),
                "section_path": (fig.get("section") or ""),
                "text": text,
                "score": 0.99,
            }
        )
        if len(out) >= max_items:
            break

    return out

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
    Строит RAG-карточки по номерам рисунков.

    ЕДИНЫЙ путь:
      1) нормализуем номера ('6', '2.3', 'Рисунок 2.3' → '6' / '2.3');
      2) по каждому номеру берём запись из таблицы figures через get_figure_record();
      3) из figures:
          - page / section,
          - image_path,
          - attrs (chart_data / chart_matrix → values_str);
      4) соседний текст поднимаем из chunks по section_path или page;
      5) (опционально) vision-описание по image_path.
    """
    if not numbers:
        return []

    # нормализуем и дедупим номера
    want = sorted(
        {
            _num_norm(str(n))
            for n in numbers
            if n and str(n).strip()
        }
    )
    want = [n for n in want if n]

    con = get_conn()
    cur = con.cursor()
    cards: List[Dict[str, Any]] = []

    for num in want:
        # 1) достаём фигуру + анализ из таблицы figures / figure_analysis
        rec = get_figure_record(doc_id, figure_label=num, ensure_analysis=use_vision)
        if not rec or not rec.get("figure"):
            continue

        fig = rec["figure"]
        analysis = rec.get("analysis") or {}

        # номер / подпись
        label = _num_norm(str(fig.get("label") or fig.get("figure_label") or num))
        caption = (fig.get("caption") or "").strip()

        display = f"Рисунок {label}"
        if caption:
            display += f" — {caption}"

        page = fig.get("page")
        sec = fig.get("section") or ""

        # 2) путь(и) до картинки — из таблицы figures (image_path)
        images: List[str] = []
        img = fig.get("image_path")
        if img:
            images.append(img)

        # 3) 1–2 соседних текстовых фрагмента
        highlights: List[str] = []

        if sec:
            cur.execute(
                """
                SELECT text FROM chunks
                WHERE owner_id=? AND doc_id=? AND section_path=?
                ORDER BY id ASC LIMIT ?
                """,
                (uid, doc_id, sec, max(1, sample_chunks)),
            )
            rows = cur.fetchall() or []
            for rr in rows:
                t = (rr["text"] or "").strip()
                if not t:
                    continue
                t = re.sub(r"^\[\s*Рисунок\s*\]\s*", "", t, flags=re.IGNORECASE)
                highlights.append(_shorten(t, 200))

        if not highlights and page is not None:
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

        # 4) chart_matrix/values_str — из attrs фигуры
                # 4) chart_matrix/values_str — из attrs фигуры
        chart_matrix = None
        try:
            attrs = fig.get("attrs")
            if isinstance(attrs, str):
                attrs_obj = json.loads(attrs or "{}") or {}
            elif isinstance(attrs, dict):
                attrs_obj = attrs
            else:
                attrs_obj = {}
        except Exception:
            attrs_obj = {}

        if isinstance(attrs_obj, dict) and attrs_obj.get("chart_matrix") is not None:
            chart_matrix = attrs_obj.get("chart_matrix")

        # значения, привязанные к САМОЙ диаграмме (chart_data/chart_matrix)
        values_str = _figure_values_str_from_row(fig)

        # NEW: если своих чисел у рисунка нет, но подпись/текст вокруг
        # явно ссылается на таблицу-источник — подтягиваем значения из таблицы.
        if not values_str:
            # берём подпись + соседний текст, чтобы искать фразы "по данным таблицы 2.1"
            hint_text_parts: List[str] = []
            if caption:
                hint_text_parts.append(caption)
            hint_text_parts.extend(h for h in (highlights or []) if h)
            hint_text = " ".join(hint_text_parts)

            src_table_num = _extract_table_source_from_text(hint_text)
            if src_table_num:
                try:
                    table_vals = _table_values_for_figure_from_doc(
                        uid,
                        doc_id,
                        src_table_num,
                        max_rows=12,
                    )
                except Exception:
                    table_vals = None

                if table_vals:
                    values_str = (
                        f"Значения диаграммы получены по данным таблицы {src_table_num}:\n"
                        f"{table_vals}"
                    )


        # 5) vision-описание (опционально)
                # 5) vision-описание (опционально) + нормализованный формат
        vision_desc: str = ""
        vision_raw_text: str = ""

        if use_vision and images:
            try:
                img_for_vision = images[0] if vision_first_image_only else images
                vp = vision_describe(img_for_vision, lang=lang)
            except Exception:
                vp = {
                    "description": "описание изображения недоступно.",
                    "tags": [],
                }

            # vp может быть строкой или словарём — аккуратно нормализуем
            if isinstance(vp, str):
                vision_desc = vp.strip()
            elif isinstance(vp, dict):
                vision_desc = (vp.get("description") or "").strip()
                # главное: не потерять сырой текст / OCR
                vision_raw_text = (vp.get("raw_text") or vp.get("text") or "").strip()

        # дополняем текстом из анализа фигуры, если он есть
        if isinstance(analysis, dict):
            ana_text = (analysis.get("text") or analysis.get("raw_text") or "").strip()
            if ana_text:
                if vision_raw_text:
                    vision_raw_text = f"{vision_raw_text}\n{ana_text}"
                else:
                    vision_raw_text = ana_text

        vision_payload = None
        if vision_desc or vision_raw_text:
            vision_payload = {
                "description": vision_desc,
                "raw_text": vision_raw_text,
            }

        card: Dict[str, Any] = {
            "num": label,
            "display": display,
            "where": {"page": page, "section_path": sec},
            "images": images,
            "highlights": highlights[:2],
        }

        if chart_matrix is not None:
            card["chart_matrix"] = chart_matrix
        if values_str:
            card["values_str"] = values_str
        if vision_payload is not None:
            card["vision"] = vision_payload
        if analysis:
            card["analysis"] = analysis


        cards.append(card)

    con.close()
    return cards

# =======================================================================
#        НОВОЕ: Таблицы — поиск/раскрытие и «все значения»
# =======================================================================

# «Таблица 2.1 — ...», «Table A.1 — ...»
_TABLE_TITLE_RE = re.compile(
    r"(?i)\b(?:табл(?:ица)?|table)\s+([A-Za-zА-Яа-я]?\s*\d+(?:[.,]\d+)*)\b(?:\s*[—\-–:]\s*(.+))?",
)
_ROW_TAG_RE = re.compile(r"\[\s*row\s+(\d+)\s*\]", re.IGNORECASE)

def _num_norm(s: str | None) -> str:
    s = (s or "").strip()
    s = s.replace(" ", "").replace(",", ".")
    s = re.sub(r"[.)]+$", "", s)  # «3.1.» -> «3.1», «(3.1)» -> «3.1»
    return s

# сильные формулировки связи с таблицей:
# "по данным таблицы 2.1", "по данным табл. 2.1",
# "на основании данных таблицы 2.1", "согласно таблице 2.1" и т.п.
_TABLE_SOURCE_RE = re.compile(
    r"(?i)"
    r"(?:по\s+данн\w+\s+табл(?:иц\w*)?\s+"
    r"|на\s+основани[иия]\s+данн\w+\с+табл(?:иц\w*)?\s+"
    r"|согласно\s+табл(?:иц\w*)?\s+)"
    r"([A-Za-zА-Яа-я]?\s*\d+(?:[.,]\d+)*)"
)

def _extract_table_source_from_text(text: str) -> Optional[str]:
    """
    Ищем в подписи/рядом с рисунком сильную фразу вида
    «по данным таблицы 2.1 / табл. 2.1 / согласно таблице 2.1».
    Возвращаем нормализованный номер таблицы или None.
    """
    if not text:
        return None
    m = _TABLE_SOURCE_RE.search(text)
    if not m:
        return None
    return _num_norm(m.group(1))


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

# --- Fallback-реализации, если extract_* не приехали из .intents ---

if extract_table_numbers is None:  # type: ignore
    def extract_table_numbers(q: str) -> List[str]:  # type: ignore
        out: List[str] = []
        for m in re.finditer(
            r"(?:табл(?:ица)?|table)\s*(?:№\s*|no\.?\s*|номер\s*)?([A-Za-zА-Яа-я]?\s*\d+(?:[.,]\d+)*)",
            q or "",
            re.IGNORECASE,
        ):
            out.append(_num_norm(m.group(1)))
        return [x for x in out if x]

if extract_figure_numbers is None:  # type: ignore
    def extract_figure_numbers(q: str) -> List[str]:  # type: ignore
        out: List[str] = []
        for m in re.finditer(
            r"(?:рис(?:унок)?|fig(?:ure)?\.?)\s*(?:№\s*|no\.?\s*|номер\s*)?([A-Za-zА-Яа-я]?\s*\d+(?:[.,]\d+)*)",
            q or "",
            re.IGNORECASE,
        ):
            out.append(_num_norm(m.group(1)))
        return [x for x in out if x]

def _wants_all_values(ask: str) -> bool:
    """
    Хинт «все значения / всю таблицу»:
      - либо через TABLE_ALL_VALUES_RE из .intents,
      - либо через локальный fallback (он уже зашит в TABLE_ALL_VALUES_RE выше).
    """
    return bool(TABLE_ALL_VALUES_RE.search(ask or ""))

def _find_table_bases_by_number(pack: dict, num: str) -> List[str]:
    """
    Ищем секции таблиц по номеру:
      - по attrs.caption_num/label,
      - по вхождению 'Таблица/Табл./Table N' в section_path/text
        с терпимостью к '2.1' / '2,1' / '2.1.'.
      - ЕСЛИ не нашли — резервный режим: по порядковому номеру таблицы (attrs.order_index),
        чтобы уметь находить даже «кривые» таблицы без нормальной подписи.
    Возвращаем список базовых section_path (без [row k]).
    """
    bases: List[str] = []
    needle = _num_norm(num)
    if not needle:
        return bases

    # Допускаем варианты числа: 2.1 / 2,1 / 2.1.
    core = re.escape(needle).replace(r"\.", r"[.,]")
    num_pat = rf"{core}\s*[.)]?"          # '2.1', '2,1', '2.1.' и т.п.
    ru_pat = re.compile(rf"(?i)\bтабл(?:ица)?\s+{num_pat}")
    en_pat = re.compile(rf"(?i)\btable\s+{num_pat}")

    # --- 1) Обычный путь: по caption_num/label и вхождениям "Таблица N" в тексте/пути ---
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

        # fallback по тексту/пути (RU/EN, 'Таблица' и 'Табл.')
        sp = (m.get("section_path") or "")
        tx = (m.get("text") or "")
        if ru_pat.search(sp) or ru_pat.search(tx) or en_pat.search(sp) or en_pat.search(tx):
            base = _table_base_from_section(sp)
            if base and base not in bases:
                bases.append(base)

    # Если уже что-то нашли — резервный режим не нужен
    if bases:
        return bases

    # --- 2) Резервный режим: поиск по порядковому номеру таблицы (attrs.order_index) ---
    # Работает, когда пользователь спрашивает "таблица 6", а подпись в документе кривая
    # или вообще отсутствует, но при парсинге мы всё равно присвоили order_index=6.
    try:
        # Берём целую часть номера (для '6', '6.1' → 6)
        ordinal = int(float(needle.split(".")[0]))
    except Exception:
        return bases

    # Собираем все таблицы с order_index и их базовые пути
    candidates: List[Tuple[int, str]] = []
    for m in pack["meta"]:
        et = _chunk_type(m)
        if et not in {"table", "table_row"}:
            continue

        attrs = m.get("attrs") or {}
        oi = attrs.get("order_index")
        if isinstance(oi, str):
            try:
                oi = int(oi)
            except Exception:
                oi = None
        if not isinstance(oi, int):
            continue

        base = _table_base_from_section(m.get("section_path") or "")
        if not base:
            continue
        candidates.append((oi, base))

    if not candidates:
        return bases

    # Для каждого base оставляем минимальный order_index (заголовочный чанк таблицы)
    best_by_base: Dict[str, int] = {}
    for oi, base in candidates:
        prev = best_by_base.get(base)
        if prev is None or oi < prev:
            best_by_base[base] = oi

    # Сортируем по порядку появления в документе
    sorted_bases = sorted(best_by_base.items(), key=lambda x: x[1])

    # Ищем таблицу с нужным порядковым номером
    for oi, base in sorted_bases:
        if oi == ordinal and base not in bases:
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

def _table_values_for_figure_from_doc(
    owner_id: int,
    doc_id: int,
    table_num: str,
    *,
    max_rows: int = 12,
) -> Optional[str]:
    """
    Для рисунка, построенного «по данным таблицы N», поднимаем текстовые
    значения этой таблицы из chunks и превращаем их в один текстовый блок.

    Используется ТОЛЬКО когда у самого рисунка нет своих чисел.
    """
    pack = _load_doc(owner_id, doc_id)
    bases = _find_table_bases_by_number(pack, table_num)
    if not bases:
        return None

    parts: List[str] = []
    seen: set[str] = set()

    # обычно первая база — нужная таблица; остальные игнорируем
    base = bases[0]
    chunks = _gather_table_chunks_for_base(
        pack,
        base,
        include_header=True,
        row_limit=max_rows,
    )

    for ch in chunks:
        t = _clean_for_ctx(ch.get("text") or "")
        if not t:
            continue
        if t in seen:
            continue
        seen.add(t)
        parts.append(t)

    if not parts:
        return None

    return "\n".join(parts)


def _inject_special_sources_for_item(pack: dict, ask: str, used_ids: set[int], *, doc_id: int) -> List[Dict]:
    """
    Если подпункт явно ссылается на «Таблица N» / «Рисунок N», добавляем соответствующие
    источники (включая все строки таблицы при «дай все значения» и краткие описания рисунков).
    """

    added: List[Dict] = []

    # Таблицы
    table_nums = extract_table_numbers(ask) if extract_table_numbers else []  # type: ignore
    want_all = _wants_all_values(ask)
    for num in table_nums:
        bases = _find_table_bases_by_number(pack, num)
        for base in bases:
            chunks = _gather_table_chunks_for_base(
                pack,
                base,
                include_header=True,
                row_limit=(None if want_all else 8),
            )
            for ch in chunks:
                if ch["id"] in used_ids:
                    continue
                ch["for_item"] = None  # будет выставлено выше по месту использования
                added.append(ch)
                used_ids.add(ch["id"])

    # Рисунки: работаем только через таблицу figures (без regex-поиска по chunks)
    fig_nums = extract_figure_numbers(ask) if extract_figure_numbers else []  # type: ignore

    if fig_nums:
        for raw_num in fig_nums:
            num = _num_norm(raw_num)
            if not num:
                continue

            try:
                rec = get_figure_record(doc_id, figure_label=num, ensure_analysis=True)
            except Exception:
                rec = None
            if not rec or not rec.get("figure"):
                continue

            fig = rec["figure"]
            text = _build_figure_context_text(rec)
            if not text:
                continue

            fid = int(fig.get("figure_id"))
            synthetic_id = -int(10_000_000 + fid)
            if synthetic_id in used_ids:
                continue

            # основной сниппет — фигура + анализ + values_str
            added.append(
                {
                    "id": synthetic_id,
                    "page": fig.get("page"),
                    "section_path": (fig.get("section") or ""),
                    "text": text,
                    "score": 0.96,
                    "for_item": None,
                }
            )
            used_ids.add(synthetic_id)

            # при возможности добавим 1 небольшой текстовый фрагмент из той же секции
            sec = fig.get("section") or ""
            if sec:
                for m in pack["meta"]:
                    if (m.get("section_path") or "") != sec:
                        continue
                    if m["id"] in used_ids:
                        continue
                    t = (m.get("text") or "").strip()
                    if not t:
                        continue
                    added.append(
                        {
                            "id": m["id"],
                            "page": m["page"],
                            "section_path": m["section_path"],
                            "text": _shorten(t, 200),
                            "score": 0.9,
                            "for_item": None,
                        }
                    )
                    used_ids.add(m["id"])
                    break  # одного соседнего чанка достаточно

        return added

    # Если ни таблиц, ни рисунков по вопросу не нашли — возвращаем пустой список
    return added


# =======================================================================
#   Публичные хелперы: точечный контекст по таблицам, рисункам и главам
# =======================================================================

def get_table_context_for_numbers(
    owner_id: int,
    doc_id: int,
    numbers: List[str],
    *,
    include_all_values: bool = False,
    rows_limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Возвращает список чанков по номерам таблиц:
      - заголовок таблицы (если есть),
      - строки (table_row) в порядке следования.

    Параметры:
      include_all_values=True  → игнорируем rows_limit и отдаём все строки;
      rows_limit=N             → ограничиваем количество строк (если include_all_values=False).
    """
    if not numbers:
        return []

    pack = _load_doc(owner_id, doc_id)
    used_ids: set[int] = set()
    snippets: List[Dict[str, Any]] = []

    for raw in numbers:
        num = _num_norm(str(raw))
        if not num:
            continue

        bases = _find_table_bases_by_number(pack, num)
        for base in bases:
            chunks = _gather_table_chunks_for_base(
                pack,
                base,
                include_header=True,
                row_limit=None if include_all_values else rows_limit,
            )
            for ch in chunks:
                cid = int(ch["id"])
                if cid in used_ids:
                    continue
                snippets.append(ch)
                used_ids.add(cid)

    return snippets


def get_figure_context_for_numbers(
    owner_id: int,
    doc_id: int,
    numbers: List[str],
    *,
    use_vision: bool = True,
    lang: str = "ru",
) -> List[Dict[str, Any]]:
    """
    Возвращает RAG-фрагменты по номерам рисунков:
      - page/section_path,
      - текстовое описание (анализ + подпись + values_str для диаграмм),
      - при необходимости vision-описание и пути к изображениям.

    Формат элементов максимально близок к элементам retrieve():
      {id, page, section_path, text, score}
    """
    if not numbers:
        return []

    cards = describe_figures_by_numbers(
        uid=owner_id,
        doc_id=doc_id,
        numbers=numbers,
        sample_chunks=2,
        use_vision=use_vision,
        lang=lang,
    )

    snippets: List[Dict[str, Any]] = []
    used_ids: set[int] = set()

    for card in cards:
        num = card.get("num")
        where = card.get("where") or {}
        page = where.get("page")
        sec = where.get("section_path") or ""

        # строим текст так же, как в _build_figure_context_text
        analysis = card.get("analysis") or {}
        fig_stub = {
            "label": num,
            "figure_label": num,
            "caption": "",
            "attrs": {},
            "page": page,
            "section": sec,
        }
        rec = {"figure": fig_stub, "analysis": analysis}
        text = _build_figure_context_text(rec)
        if not text:
            text = ""

        # NEW: подмешиваем values_str из карточки (сюда могут попасть
        # либо собственные числа диаграммы, либо значения, подтянутые из таблицы)
        values_str = card.get("values_str")
        if values_str and values_str not in text:
            text = f"{text}\n\n{values_str}".strip()

        if not text:
            continue

        # synthetic_id — чтобы не конфликтовать с id чанков
        synthetic_id = -int(20_000_000 + hash(str(num)))
        if synthetic_id in used_ids:
            continue

        snippets.append(
            {
                "id": synthetic_id,
                "page": page,
                "section_path": sec,
                "text": text,
                "score": 0.97,
            }
        )
        used_ids.add(synthetic_id)

    return snippets


def get_section_context_for_hints(
    owner_id: int,
    doc_id: int,
    section_hints: List[str],
    *,
    per_section_k: int = 3,
) -> List[Dict[str, Any]]:
    """
    Возвращает контекст по подсказкам разделов/глав:
      section_hints = ['1', '2.3', ...] — как из extract_section_hints().

    Для каждого hint берём несколько чанков, у которых section_path
    начинается с этой «головы» (например, '1', '1.1', '1.2', ...).
    """
    if not section_hints:
        return []

    pack = _load_doc(owner_id, doc_id)
    snippets: List[Dict[str, Any]] = []
    used_ids: set[int] = set()

    for h in section_hints:
        head = str(h or "").strip()
        if not head:
            continue

        # Простая фильтрация по префиксу section_path
        candidates: List[Tuple[int, Dict[str, Any]]] = []
        for m in pack["meta"]:
            sp = str(m.get("section_path") or "")
            # section_path в формате "1", "1/1.1", "[заголовок] 1.1" и т.п. —
            # используем грубую проверку по вхождению головы
            if not sp:
                continue
            if not (sp.startswith(head) or f"/{head}" in sp or f" {head}" in sp):
                continue
            candidates.append(
                (
                    int(m["id"]),
                    {
                        "id": m["id"],
                        "page": m["page"],
                        "section_path": sp,
                        "text": (m.get("text") or "").strip(),
                        "score": 0.9,
                    },
                )
            )

        # берём первые per_section_k по порядку появления id
        candidates.sort(key=lambda t: t[0])
        taken = 0
        for cid, item in candidates:
            if taken >= per_section_k:
                break
            if cid in used_ids:
                continue
            snippets.append(item)
            used_ids.add(cid)
            taken += 1

    return snippets


# =======================================================================
#        НОВОЕ: Coverage-aware извлечение под многопунктные вопросы
#               с интеграцией таблиц/рисунков
# =======================================================================

# Эвристики распознавания подпунктов
_ENUM_LINE = re.compile(r"(?m)^\s*(?:\d+[).\)]|[-—•])\s+")

_ENUM_INLINE = re.compile(r"(?:\s|^)\(?\d{1,2}[)\.]\s+")

def plan_subitems(question: str, *, min_items: int = 2, max_items: int = 12) -> List[str]:
    """
    Пытаемся вытащить подпункты из длинного вопроса.
    """
    q = (question or "").strip()
    items: List[str] = []

    # 1) построчно с маркерами
    if _ENUM_LINE.search(q):
        for line in q.splitlines():
            if _ENUM_LINE.match(line):
                items.append(
                    re.sub(r"^\s*(?:\d+[).\)]|[-—•])\s+", "", line).strip()
                )

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
        special = _inject_special_sources_for_item(pack, ask, used_ids, doc_id=doc_id)
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

        by_id[str(idx)] = picked

    # Дополнительная «склейка» общим контекстом из исходного вопроса — backfill
    if overall_backfill_from_query:
        rescored_q = _score_and_rank(pack, overall_backfill_from_query, prelim_k=max(backfill_k * 3, 16))
        extra = []
        for i, sc in rescored_q:
            m = pack["meta"][int(i)]
            if m["id"] in used_ids:
                continue
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
    subitems: Optional[List[Any]] = None,
    *,
    per_item_k: int = 2,
    prelim_factor: int = 4,
    backfill_k: int = 4,
) -> Dict[str, Any]:
    """
    Coverage-aware выборка под многопунктный вопрос.

    Возвращает:
      {
        "items": [{"id": 1, "ask": "..."}, ...],
        "by_item": {"1":[...], "2":[...], ...},
        "by": {"1":[idx, ...], ...},          # карта: подпункт -> индексы в snippets (для bot.py)
        "snippets": [...],
      }

    subitems может быть:
      - списком строк: ["опиши рисунок 2.2", "таблицу 2.1", "главу 1"];
      - списком словарей из intents.detect_intents():
        [{"id": 1, "ask": "...", ...}, ...].
    """
    # Если subitems не передан, пытаемся сами распланировать подпункты
    if subitems is None:
        items_list: List[str] = list(plan_subitems(question)) if question else []
        items_norm = [{"id": i + 1, "ask": it} for i, it in enumerate(items_list)]
    else:
        # subitems передан явно: может быть list[str] или list[dict]
        if subitems and isinstance(subitems[0], dict):
            # Формат из intents.detect_intents(): уже есть id и ask
            items_norm = []
            items_list = []
            for it in subitems:
                ask = (it.get("ask") or "").strip()
                if not ask:
                    continue
                # если id уже есть — используем его, иначе пронумеруем позже
                items_norm.append({"id": it.get("id"), "ask": ask, **{k: v for k, v in it.items() if k not in {"id", "ask"}}})
                items_list.append(ask)
            # Пронумеруем те элементы, у кого id отсутствует
            next_id = 1
            for it in items_norm:
                if it.get("id") is None:
                    it["id"] = next_id
                    next_id += 1
        else:
            # Старый формат: список строк
            items_list = [str(it or "").strip() for it in subitems if str(it or "").strip()]
            items_norm = [{"id": i + 1, "ask": it} for i, it in enumerate(items_list)]

    if not items_list:
        # Если подпункты не распознаны — обычный retrieve с небольшим расширением k
        base = retrieve(owner_id, doc_id, question, top_k=max(10, backfill_k * 2))
        by_indices = {"_single": list(range(len(base)))}
        return {"items": [], "by_item": {"_single": base}, "by": by_indices, "snippets": base}

    # ниже оставляем существующую логику как есть


    if items_list:
        by_item = retrieve_for_items(
            owner_id,
            doc_id,
            items_list,
            per_item_k=per_item_k,
            prelim_factor=prelim_factor,
            overall_backfill_from_query=question,
            backfill_k=backfill_k,
        )

        # round-robin порядок: 1-й из каждого подпункта, затем 2-й и т.д.
        buckets = [by_item.get(str(i + 1), []) for i in range(len(items_list))]
        merged: List[Dict] = []
        for r in range(per_item_k):
            for b in buckets:
                if r < len(b):
                    merged.append(b[r])

        # Для подпунктов вида «дай все значения» — добавляем ОСТАЛЬНЫЕ строчки их bucket
        for i, ask in enumerate(items_list, start=1):
            if _wants_all_values(ask):
                bucket = by_item.get(str(i), [])
                if len(bucket) > per_item_k:
                    merged.extend(bucket[per_item_k:])

        # добавим backfill в конце
        if by_item.get("_backfill"):
            merged.extend(by_item["_backfill"])

        # Построим карту индексов для bot.py: { "1": [indices in merged], ... }
        by_indices: Dict[str, List[int]] = {}
        for idx, sn in enumerate(merged):
            fid = sn.get("for_item")
            if fid:
                by_indices.setdefault(str(fid), []).append(idx)

        return {"items": items_norm, "by_item": by_item, "by": by_indices, "snippets": merged}

    # Если подпункты не распознаны — обычный retrieve с небольшим расширением k
    base = retrieve(owner_id, doc_id, question, top_k=max(10, backfill_k * 2))
    # Единый «by» для удобства: вся выдача принадлежит "_single"
    by_indices = {"_single": list(range(len(base)))}
    return {"items": [], "by_item": {"_single": base}, "by": by_indices, "snippets": base}

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
