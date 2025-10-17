# app/summarizer.py
from __future__ import annotations
import re
import json
import base64
import mimetypes
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from .db import get_conn
from .polza_client import chat_with_gpt  # одна и та же Chat API умеет и vision через image_url

# ----------------------- СИСТЕМНЫЙ ПРОМТ ДЛЯ РЕЗЮМЕ -----------------------

SYS_SUMMARY = (
    "Ты русскоязычный ассистент по ВКР. Тебе даны фрагменты дипломной работы (контекст). "
    "Сделай краткое, но содержательное резюме работы БЕЗ выдумываний, опираясь только на эти фрагменты.\n\n"
    "Формат:\n"
    "— 2–3 предложения с сутью работы (тема, цель, объект/предмет, общий подход).\n"
    "— 4–7 пунктов: задачи, данные/методы, ключевые результаты/числа, вклад.\n"
    "— (если есть) 2–4 пункта по таблицам/рисункам, что в них показано.\n"
    "— В конце перечисли источники вида [Источник N], которые ты использовал.\n\n"
    "Правила:\n"
    "- Никаких фактов вне контекста; если не хватает данных — скажи об этом кратко.\n"
    "- Не придумывай номера страниц/таблиц/рисунков; не используй внешние знания.\n"
    "- Пиши чётко и по делу."
)

# ----------------------- МАЛЫЕ УТИЛИТЫ ДЛЯ БД -----------------------

def _has_columns(con, table: str, cols: List[str]) -> bool:
    cur = con.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    have = {row[1] for row in cur.fetchall()}
    return all(c in have for c in cols)

def _fetchall(con, sql: str, params: Tuple[Any, ...]) -> List[Dict[str, Any]]:
    cur = con.cursor()
    cur.execute(sql, params)
    rows = cur.fetchall()
    return [dict(r) for r in rows]

def _fetchone(con, sql: str, params: Tuple[Any, ...]) -> Optional[Dict[str, Any]]:
    cur = con.cursor()
    cur.execute(sql, params)
    row = cur.fetchone()
    return (dict(row) if row else None)

def _clean(s: str) -> str:
    return (s or "").replace("\xa0", " ").strip()

# ----------------------- СБОР «ПОЛНОГО» ТЕКСТА -----------------------

def _collect_full_text(owner_id: int, doc_id: int) -> str:
    """Собирает весь текст документа в исходном порядке."""
    con = get_conn()
    cur = con.cursor()
    cur.execute(
        "SELECT text FROM chunks WHERE owner_id=? AND doc_id=? ORDER BY page ASC, id ASC",
        (owner_id, doc_id)
    )
    rows = cur.fetchall()
    con.close()
    return "\n\n".join([_clean(row["text"]) for row in rows if row["text"]])

def make_full_context(owner_id: int, doc_id: int, max_chars: int = 6000) -> str:
    """Формирует полный контекст (обрезаем до max_chars)."""
    return _collect_full_text(owner_id, doc_id)[:max_chars]

# ----------------------- СЕМАНТИЧЕСКАЯ ВЫБОРКА ДЛЯ ОБЗОРА -----------------------

# Интент на «краткое содержание»
_SUMMARY_HINT_RE = re.compile(
    r"\b(суть|кратко|основн|главн|summary|overview|итог|выводы?|резюме|реферат)\w*\b",
    re.IGNORECASE
)

def is_summary_intent(text: str) -> bool:
    return bool(_SUMMARY_HINT_RE.search(text or ""))

# Ключевые маркеры по содержанию
_KEY_HINTS = [
    "Цель", "Задач", "Объект", "Предмет", "Гипотез",
    "Метод", "Методик", "Материал", "Данные", "Вывод", "Заключен", "Введен"
]
_KEY_RE = re.compile("|".join([fr"{h}" for h in _KEY_HINTS]), re.IGNORECASE)

def _collect_headings(owner_id: int, doc_id: int, limit: int = 50) -> List[Dict[str, Any]]:
    """Заголовки (новые индексы: element_type='heading', старые: префикс)."""
    con = get_conn()
    has_type = _has_columns(con, "chunks", ["element_type"])
    if has_type:
        rows = _fetchall(
            con,
            "SELECT id, page, section_path, text FROM chunks "
            "WHERE owner_id=? AND doc_id=? AND element_type='heading' "
            "ORDER BY page ASC, id ASC LIMIT ?",
            (owner_id, doc_id, limit)
        )
    else:
        rows = _fetchall(
            con,
            "SELECT id, page, section_path, text FROM chunks "
            "WHERE owner_id=? AND doc_id=? AND text LIKE '[Заголовок]%' "
            "ORDER BY page ASC, id ASC LIMIT ?",
            (owner_id, doc_id, limit)
        )
    con.close()
    return rows

def _collect_keylines(owner_id: int, doc_id: int, limit: int = 30) -> List[Dict[str, Any]]:
    """Строки с целями/задачами/методами/выводами и т.п. (быстрые эвристики)."""
    con = get_conn()
    rows = _fetchall(
        con,
        "SELECT id, page, section_path, text FROM chunks "
        "WHERE owner_id=? AND doc_id=? "
        "ORDER BY page ASC, id ASC",
        (owner_id, doc_id)
    )
    con.close()
    out = []
    for r in rows:
        t = (r.get("text") or "")
        if _KEY_RE.search(t):
            out.append(r)
            if len(out) >= limit:
                break
    return out

def _collect_tables(owner_id: int, doc_id: int, max_tables: int = 8, sample_rows: int = 2) -> List[Dict[str, Any]]:
    """Список таблиц + первые строки (совместимость со старыми индексами)."""
    con = get_conn()
    has_type = _has_columns(con, "chunks", ["element_type"])

    if has_type:
        tables = _fetchall(
            con,
            "SELECT DISTINCT "
            "CASE WHEN instr(section_path, ' [row ')>0 "
            "     THEN substr(section_path, 1, instr(section_path,' [row ')-1) "
            "     ELSE section_path END AS base_name "
            "FROM chunks "
            "WHERE owner_id=? AND doc_id=? AND element_type IN ('table','table_row')",
            (owner_id, doc_id)
        )
    else:
        tables = _fetchall(
            con,
            "SELECT DISTINCT "
            "CASE WHEN instr(section_path, ' [row ')>0 "
            "     THEN substr(section_path, 1, instr(section_path,' [row ')-1) "
            "     ELSE section_path END AS base_name "
            "FROM chunks "
            "WHERE owner_id=? AND doc_id=? AND text LIKE '[Таблица]%'",
            (owner_id, doc_id)
        )

    base_names = [t["base_name"] for t in tables if t.get("base_name")]
    base_names = sorted(set(base_names), key=lambda s: s.lower())[:max_tables]

    result: List[Dict[str, Any]] = []
    conn2 = get_conn()
    cur = conn2.cursor()
    for name in base_names:
        samples: List[str] = []
        for r in range(1, sample_rows + 1):
            cur.execute(
                "SELECT text, page FROM chunks WHERE owner_id=? AND doc_id=? AND section_path=? "
                "ORDER BY id ASC LIMIT 1",
                (owner_id, doc_id, f"{name} [row {r}]")
            )
            row = cur.fetchone()
            if row and row["text"]:
                samples.append(_clean(row["text"]))
        result.append({"name": name, "samples": samples})
    conn2.close()
    return result

def _collect_figures(owner_id: int, doc_id: int, max_figs: int = 8) -> List[Dict[str, Any]]:
    """Список «рисунков» (совместимость со старыми индексами)."""
    con = get_conn()
    has_type = _has_columns(con, "chunks", ["element_type"])
    if has_type:
        rows = _fetchall(
            con,
            "SELECT DISTINCT page, section_path, text, attrs FROM chunks "
            "WHERE owner_id=? AND doc_id=? AND element_type='figure' "
            "ORDER BY id ASC LIMIT ?",
            (owner_id, doc_id, max_figs)
        )
    else:
        rows = _fetchall(
            con,
            "SELECT DISTINCT page, section_path, text, NULL as attrs FROM chunks "
            "WHERE owner_id=? AND doc_id=? AND text LIKE '[Рисунок]%' "
            "ORDER BY id ASC LIMIT ?",
            (owner_id, doc_id, max_figs)
        )
    con.close()
    return rows

def _select_first_paragraphs(owner_id: int, doc_id: int, limit: int = 12) -> List[Dict[str, Any]]:
    """Ранние абзацы как общий фолбэк контекста."""
    con = get_conn()
    has_type = _has_columns(con, "chunks", ["element_type"])
    if has_type:
        rows = _fetchall(
            con,
            "SELECT page, section_path, text FROM chunks "
            "WHERE owner_id=? AND doc_id=? AND (element_type IS NULL OR element_type IN ('paragraph','text','page')) "
            "ORDER BY page ASC, id ASC LIMIT ?",
            (owner_id, doc_id, limit)
        )
    else:
        rows = _fetchall(
            con,
            "SELECT page, section_path, text FROM chunks "
            "WHERE owner_id=? AND doc_id=? "
            "ORDER BY page ASC, id ASC LIMIT ?",
            (owner_id, doc_id, limit)
        )
    con.close()
    # уберём пустые
    return [r for r in rows if (r.get("text") or "").strip()][:limit]

def _build_context_block(rows: List[Dict[str, Any]], max_chars: int = 6000) -> str:
    """
    Формирует контекст вида:
    [Источник N] ...текст...
    (стр. X • Раздел)
    """
    parts: List[str] = []
    total = 0
    i = 1
    for r in rows:
        page = r.get("page")
        sec = (r.get("section_path") or "").strip()
        text = _clean(r.get("text") or "")
        if not text:
            continue
        block = f"[Источник {i}] {text}"
        loc = []
        if page is not None:
            loc.append(f"стр. {page}")
        if sec:
            loc.append(sec)
        if loc:
            block += f"  ({' • '.join(loc)})"
        # лимит
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
        i += 1
    return "\n\n".join(parts)

def overview_context(owner_id: int, doc_id: int, max_chars: int = 6000) -> str:
    """
    «Умный» контекст для резюме: заголовки, цели/задачи/методы/выводы,
    краткие сведения о таблицах и рисунках + немного первых абзацев.
    Всегда с маркерами [Источник N] для удобного цитирования в ответе.
    """
    rows: List[Dict[str, Any]] = []

    # 1) Заголовки верхних уровней
    rows += _collect_headings(owner_id, doc_id, limit=20)

    # 2) Ключевые строки (цели/задачи/методы/выводы…)
    rows += _collect_keylines(owner_id, doc_id, limit=30)

    # 3) Таблицы: склеим имя секции + первую строку
    for t in _collect_tables(owner_id, doc_id, max_tables=8, sample_rows=2):
        name = t["name"]
        samples = [s for s in (t.get("samples") or []) if s]
        if not samples:
            continue
        first = samples[0].splitlines()[0]
        rows.append({
            "page": None,
            "section_path": name,
            "text": f"{name}: {first}"
        })

    # 4) Рисунки: возьмём подпись целиком (section_path или text)
    for f in _collect_figures(owner_id, doc_id, max_figs=8):
        cap = (f.get("text") or f.get("section_path") or "").strip()
        if not cap:
            continue
        rows.append({
            "page": f.get("page"),
            "section_path": f.get("section_path"),
            "text": cap
        })

    # 5) Общий фолбэк (первые абзацы)
    rows += _select_first_paragraphs(owner_id, doc_id, limit=12)

    if not rows:
        return ""

    return _build_context_block(rows, max_chars=max_chars)

# ----------------------- РЕЗЮМЕ ДОКУМЕНТА -----------------------

def summarize_document(owner_id: int, doc_id: int) -> str:
    """
    Делаем краткое резюме дипломной работы:
    — строим «умный» контекст overview_context,
    — отправляем в модель с SYS_SUMMARY,
    — возвращаем ответ.
    """
    ctx = overview_context(owner_id, doc_id, max_chars=6000)
    if not ctx.strip():
        # крайний случай — шлём просто первые ~6000 символов
        ctx = make_full_context(owner_id, doc_id, max_chars=6000)

    if not ctx.strip():
        return ("Не удалось собрать достаточно контекста для резюме. "
                "Проверьте, что в работе есть текстовые разделы (цели/задачи/методы/выводы).")

    reply = chat_with_gpt(
        [
            {"role": "system", "content": SYS_SUMMARY},
            {"role": "assistant", "content": f"Контекст:\n{ctx}"},
            {"role": "user", "content": "Сформулируй краткое резюме работы по приведённым фрагментам."}
        ],
        temperature=0.2,
    )
    return (reply or "").strip()

# ======================================================================
#                  VISION: ОПИСАНИЕ РИСУНКОВ ПО НОМЕРАМ
# ======================================================================

# Регекс для «Рисунок 2.1 — ...» / «Рис. 2,1 ...» / «Figure 3»
_FIG_CAP_RE = re.compile(
    r"(?i)\b(?:рис(?:\.|унок)?|figure|fig\.?)\s*(?:№\s*)?(\d+(?:[.,]\d+)*)(?:\s*[—\-–:]\s*(.+))?"
)

def _norm_num(n: Optional[str]) -> Optional[str]:
    if not n:
        return None
    return n.replace(",", ".").strip()

def _parse_figure_title_line(s: str) -> Tuple[Optional[str], Optional[str]]:
    m = _FIG_CAP_RE.search(s or "")
    if not m:
        return (None, None)
    return (_norm_num(m.group(1)), (m.group(2) or "").strip() or None)

def _guess_mime(path: str) -> str:
    mt, _ = mimetypes.guess_type(path)
    return mt or "image/png"

def _file_to_data_url(path: str) -> Optional[str]:
    """Локальный файл -> data URL (base64) для image_url."""
    try:
        p = Path(path)
        if not p.exists() or not p.is_file():
            return None
        b = p.read_bytes()
        b64 = base64.b64encode(b).decode("ascii")
        mime = _guess_mime(path)
        return f"data:{mime};base64,{b64}"
    except Exception:
        return None

def _figure_row_by_number(owner_id: int, doc_id: int, num: str) -> Optional[Dict[str, Any]]:
    """Ищем первый чанк рисунка по номеру (новый и старый индексы)."""
    con = get_conn()
    want = _norm_num(num) or ""
    row = None
    if _has_columns(con, "chunks", ["element_type", "attrs"]):
        row = _fetchone(
            con,
            "SELECT id, page, section_path, text, attrs FROM chunks "
            "WHERE owner_id=? AND doc_id=? AND element_type='figure' "
            "AND (section_path LIKE ? COLLATE NOCASE OR text LIKE ? COLLATE NOCASE) "
            "ORDER BY id ASC LIMIT 1",
            (owner_id, doc_id, f"%Рисунок {want}%", f"%Рисунок {want}%")
        )
        if not row:
            # Попробуем англ. подпись
            row = _fetchone(
                con,
                "SELECT id, page, section_path, text, attrs FROM chunks "
                "WHERE owner_id=? AND doc_id=? AND element_type='figure' "
                "AND (section_path LIKE ? COLLATE NOCASE OR text LIKE ? COLLATE NOCASE) "
                "ORDER BY id ASC LIMIT 1",
                (owner_id, doc_id, f"%Figure {want}%", f"%Figure {want}%")
            )
    else:
        row = _fetchone(
            con,
            "SELECT id, page, section_path, text, NULL AS attrs FROM chunks "
            "WHERE owner_id=? AND doc_id=? AND text LIKE '[Рисунок]%' "
            "AND (section_path LIKE ? COLLATE NOCASE OR text LIKE ? COLLATE NOCASE) "
            "ORDER BY id ASC LIMIT 1",
            (owner_id, doc_id, f"%Рисунок {want}%", f"%Рисунок {want}%")
        )
    con.close()
    return row

def _extract_images_from_attrs(attrs_raw: Any) -> List[str]:
    """Извлекаем список путей изображений из JSON-attrs."""
    try:
        if not attrs_raw:
            return []
        if isinstance(attrs_raw, dict):
            obj = attrs_raw
        else:
            obj = json.loads(attrs_raw)
        imgs = obj.get("images") or []
        out: List[str] = []
        for p in imgs:
            if not p:
                continue
            sp = str(p)
            if Path(sp).exists():
                out.append(sp)
        return out
    except Exception:
        return []

def _collect_images_for_section(owner_id: int, doc_id: int, section_path: str) -> List[str]:
    """
    Подтягиваем images из attrs по всем чанкам данной секции.
    Нужен для случаев, когда картинки лежат не в первом чанке figure.
    """
    con = get_conn()
    cur = con.cursor()
    try:
        cur.execute(
            """
            SELECT attrs FROM chunks
            WHERE owner_id=? AND doc_id=? AND section_path=? AND attrs IS NOT NULL
            ORDER BY id ASC LIMIT 12
            """,
            (owner_id, doc_id, section_path),
        )
        rows = cur.fetchall() or []
    finally:
        con.close()

    images: List[str] = []
    for rr in rows:
        try:
            obj = json.loads(rr["attrs"] or "{}")
        except Exception:
            obj = {}
        for p in (obj.get("images") or []):
            if p and p not in images and Path(str(p)).exists():
                images.append(str(p))
    return images

def _vision_describe(paths: List[str], caption_hint: Optional[str]) -> str:
    """
    1–2 предложения описания изображения(й) через chat_with_gpt с image_url (data URL).
    Если передано несколько картинок, используем первую-вторую.
    """
    chosen = [p for p in paths if p][:2]
    image_contents = []
    for p in chosen:
        data_url = _file_to_data_url(p)
        if data_url:
            image_contents.append({"type": "image_url", "image_url": {"url": data_url}})

    if not image_contents:
        return ""

    user_text = (
        "Опиши содержимое изображения кратко и академично (1–2 предложения), "
        "без выдумок и оценочных суждений. Если на схеме есть подписи/оси — упомяни их родовые названия, "
        "но не переписывай длинные тексты целиком. Избегай слов «на фото видно» — сразу пиши, что изображено."
    )
    if caption_hint:
        user_text += f"\nКонтекст подписи: «{caption_hint}»."

    messages = [
        {"role": "system", "content": "Ты ассистент по дипломным работам. Пиши по-русски, академично, кратко."},
        {"role": "user", "content": [{"type": "text", "text": user_text}] + image_contents}
    ]
    try:
        resp = chat_with_gpt(messages, temperature=0.1, max_tokens=180)
        return (resp or "").strip()
    except Exception:
        return ""

def _format_figure_sentence(num: Optional[str], vision_text: str, tail: Optional[str]) -> str:
    """
    Финальный формат строки ответа. Мини-очистка вводных конструкций.
    """
    n = num or "—"
    vt = (vision_text or "").strip()
    vt = re.sub(r"(?i)^(на\s+изображении|на\s+рисунке|на\s+фото)\s+(представлен[оы]?|показан[оы]?|изображен[оы]?)\s*", "", vt)
    vt = vt.rstrip(".").strip()

    if vt:
        base = f"На этом рисунке (Рисунок {n}) изображено {vt}."
    elif tail:
        base = f"На этом рисунке (Рисунок {n}) изображено: {tail.strip()}."
    else:
        base = f"На этом рисунке (Рисунок {n}) представлено содержание, соответствующее подписи автора."
    return base

def describe_figures(owner_id: int, doc_id: int, numbers: List[str]) -> str:
    """
    Принимает список номеров рисунков ['2.1', '3', ...] и возвращает по каждому
    короткое академичное описание (vision + подпись, если картинка не извлечена).
    """
    if not numbers:
        return "Не указаны номера рисунков. Например: 2.1, 3.2, 5."

    lines: List[str] = []
    for raw in numbers:
        num = _norm_num(raw)
        row = _figure_row_by_number(owner_id, doc_id, num or "")
        if not row:
            lines.append(f"Рисунок {num or raw}: не найден в документе.")
            continue

        # Подпись-намёк (из section_path или text)
        sec = row.get("section_path") or ""
        txt = row.get("text") or ""
        n_sec, tail_sec = _parse_figure_title_line(sec)
        n_txt, tail_txt = _parse_figure_title_line(txt)
        tail = tail_sec or tail_txt

        # Картинки: из attrs найденного чанка + по всей секции (на случай, если картинки лежат в других чанках)
        img_paths = _extract_images_from_attrs(row.get("attrs"))
        if sec:
            extra = _collect_images_for_section(owner_id, doc_id, sec)
            for p in extra:
                if p not in img_paths:
                    img_paths.append(p)

        # Описание через vision (если есть что отправить)
        vision_text = _vision_describe(img_paths, tail)

        # Ответная строка
        out = _format_figure_sentence(num or n_sec or n_txt, vision_text, tail)
        if not img_paths and not vision_text:
            out += " (Картинка не была извлечена из файла; описание дано по подписи/контексту.)"

        lines.append(out)

    return "\n".join(lines)

# Обёртка для одного номера
def describe_figure(owner_id: int, doc_id: int, number: str) -> str:
    return describe_figures(owner_id, doc_id, [number])
