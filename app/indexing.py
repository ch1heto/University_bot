# app/indexing.py
import json
import numpy as np
from typing import List, Dict, Any, Iterable
from .db import get_conn
from .polza_client import embeddings
from .chunking import split_into_chunks


# ---------- helpers ----------

def _norm(s: str | None) -> str:
    return (s or "").strip()

def _prefix(section: Dict[str, Any]) -> str:
    """
    Короткий префикс контекста для каждого чанка (повышает качество поиска).
    Примеры:
      [Заголовок] Введение
      [Таблица] стр.3 Таблица 2.1 — Состав выборки
      [Текст] стр.5 Глава 2. Методика исследования
    """
    et = (section.get("element_type") or "").lower()
    title = _norm(section.get("title")) or "Документ"
    page = section.get("page")
    pfx_parts = []
    if et == "heading":
        pfx_parts.append("[Заголовок]")
    elif et == "table":
        pfx_parts.append("[Таблица]")
    elif et == "figure":
        pfx_parts.append("[Рисунок]")
    elif et == "page":
        pfx_parts.append("[Страница]")
    elif et == "reference":
        pfx_parts.append("[Источник]")  # Новый префикс для источников
    else:
        pfx_parts.append("[Текст]")

    if page:
        pfx_parts.append(f"стр.{page}")
    pfx_parts.append(title)
    return " ".join(pfx_parts)

def _yield_chunks_for_section(
    section: Dict[str, Any],
    *,
    max_table_rows: int = 500
) -> Iterable[tuple[str, Dict[str, Any], str, dict]]:
    """
    Генерирует чанки для одной секции.
    Возвращает кортежи: (text, meta, element_type, attrs_dict)

    - Таблицы -> одна строка = один чанк (element_type='table_row', attrs: row_index).
    - Заголовки -> отдельный маленький чанк (element_type='heading').
    - Источники — каждый источник как отдельный чанк (element_type='reference').
    - Остальной текст -> split_into_chunks(...) (element_type='paragraph' или исходный для page/figure).
    """
    et = (section.get("element_type") or "").lower()
    title = _norm(section.get("title")) or "Документ"
    section_path = _norm(section.get("section_path")) or title
    page = section.get("page")
    base_attrs = dict(section.get("attrs") or {})
    text = section.get("text") or ""
    prefix = _prefix(section)

    # Заголовок — отдельный небольшой чанк
    if et == "heading":
        head_txt = f"{prefix}"
        yield head_txt, {"page": page, "section_path": section_path}, "heading", base_attrs
        return

    # Источник — отдельный чанк
    if et == "reference":
        ref_title = f"{prefix} — {title}"
        yield ref_title, {"page": page, "section_path": section_path}, "reference", base_attrs
        return

    # Таблица — построчно
    if et == "table":
        lines = [ln.strip() for ln in (text or "").splitlines() if ln and ln.strip()]
        if not lines:
            # Индексируем факт существования пустой таблицы
            yield f"{prefix}\n(пустая таблица)", {"page": page, "section_path": section_path}, "table", base_attrs
            return
        for i, row in enumerate(lines[:max_table_rows], 1):
            attrs = dict(base_attrs)
            attrs["row_index"] = i
            yield (
                f"{prefix} | ряд {i}\n{row}",
                {"page": page, "section_path": f"{section_path} [row {i}]"},
                "table_row",
                attrs,
            )
        return

    # Фигуры / страницы / обычные абзацы
    base = text if isinstance(text, str) else str(text)
    for ch in split_into_chunks(base):
        if not ch.strip():
            continue
        out_et = et if et in {"page", "figure", "paragraph"} else ("paragraph" if et == "" else et)
        yield f"{prefix}\n{ch}", {"page": page, "section_path": section_path}, out_et, base_attrs


def _batched(items: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(items), n):
        yield items[i:i + n]


def _chunks_table_has(con, cols: List[str]) -> bool:
    cur = con.cursor()
    cur.execute("PRAGMA table_info(chunks)")
    have = {row[1] for row in cur.fetchall()}
    return all(c in have for c in cols)


# ---------- API ----------

def index_document(
    owner_id: int,
    doc_id: int,
    sections: List[Dict[str, Any]],
    *,
    batch_size: int = 128
) -> None:
    """
    Индексирует документ:
    - строит чанки с богатыми префиксами;
    - таблицы индексируются построчно (element_type='table_row', attrs.row_index);
    - эмбеддинги считаются пакетами (batch_size);
    - пишет в таблицу chunks: (doc_id, owner_id, page, section_path, text, [element_type, attrs], embedding).
      element_type/attrs пишутся только если соответствующие колонки есть в схеме БД.
    """
    rows_text: List[str] = []
    rows_meta: List[Dict[str, Any]] = []
    rows_type: List[str] = []
    rows_attrs: List[dict] = []

    for s in sections or []:
        for txt, meta, etype, attrs in _yield_chunks_for_section(s):
            if not txt.strip():
                continue
            rows_text.append(txt)
            rows_meta.append(meta)
            rows_type.append(etype)
            rows_attrs.append(attrs or {})

    if not rows_text:
        return

    con = get_conn()
    cur = con.cursor()
    has_extended_cols = _chunks_table_has(con, ["element_type", "attrs"])

    # Считаем эмбеддинги батчами
    idx = 0
    for batch in _batched(rows_text, batch_size):
        vecs = embeddings(batch)  # list[list[float]]
        if not vecs or len(vecs) != len(batch):
            raise RuntimeError("embeddings() вернул неожиданный размер результата.")
        for j, vec in enumerate(vecs):
            k = idx + j
            meta = rows_meta[k]
            blob = np.asarray(vec, dtype=np.float32).tobytes()

            if has_extended_cols:
                cur.execute(
                    "INSERT INTO chunks(doc_id, owner_id, page, section_path, text, element_type, attrs, embedding) "
                    "VALUES(?,?,?,?,?,?,?,?)",
                    (
                        doc_id,
                        owner_id,
                        meta.get("page"),
                        meta.get("section_path"),
                        batch[j],
                        rows_type[k],
                        json.dumps(rows_attrs[k], ensure_ascii=False),
                        blob,
                    ),
                )
            else:
                cur.execute(
                    "INSERT INTO chunks(doc_id, owner_id, page, section_path, text, embedding) "
                    "VALUES(?,?,?,?,?,?)",
                    (doc_id, owner_id, meta.get("page"), meta.get("section_path"), batch[j], blob),
                )
        idx += len(batch)

    con.commit()
    con.close()
