# app/bot.py
import re
import os
import html
import json
import logging
import asyncio
import time
import math
from decimal import Decimal 
logger = logging.getLogger(__name__)
from typing import Iterable, AsyncIterable, Optional, List, Tuple
from .docs_handlers import register_docs_handlers
from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import Command
from aiogram.exceptions import TelegramBadRequest
from aiogram.enums import ChatAction
from aiogram.types import FSInputFile, InputMediaPhoto

from .ooxml_lite import (
    build_index as oox_build_index,
    figure_lookup as oox_fig_lookup,
    table_lookup as oox_tbl_lookup,
)

# ---------- answer builder: пытаемся взять стримовую версию, фолбэк на нестримовую ----------
try:
    from .answer_builder import generate_answer, generate_answer_stream  # type: ignore
except Exception:
    from .answer_builder import generate_answer  # type: ignore
    generate_answer_stream = None  # стрима нет — будем фолбэкать

from .config import Cfg, ProcessingState
from .db import (
    ensure_user, get_conn,
    set_document_indexer_version, get_document_indexer_version,
    CURRENT_INDEXER_VERSION,
    update_document_meta, delete_document_chunks,
    set_user_active_doc, get_user_active_doc,
    # ↓ новое для FSM/очереди
    enqueue_pending_query, dequeue_all_pending_queries,
    get_processing_state, start_downloading,
)
from .parsing import parse_docx, parse_doc, save_upload
from .indexing import index_document
from .retrieval import (
    retrieve, build_context, invalidate_cache,
    retrieve_coverage, build_context_coverage,
    describe_figures_by_numbers,
)
from .intents import detect_intents

# ---------- polza client: пробуем стрим, фолбэк на обычный чат ----------
try:
    from .polza_client import (
        probe_embedding_dim,
        chat_with_gpt,
        chat_with_gpt_stream,
        vision_extract_values,
        vision_extract_table_values,      # ← НОВОЕ: спец-функция для таблиц-картинок
        # NEW: мультимодальные обёртки (текст + картинки)
        chat_with_gpt_multimodal,
        chat_with_gpt_stream_multimodal,
    )  # type: ignore

    # NEW: прямой индекс рисунков из файла
    from .figures import (
        index_document as fig_index_document,
        load_index   as fig_load_index,
        find_figure  as fig_find,
        figure_display_name,
    )
except Exception:
    from .polza_client import probe_embedding_dim, chat_with_gpt  # type: ignore
    chat_with_gpt_stream = None
    vision_extract_values = None  # фолбэк: если нет функции, не падаем

    # ← НОВОЕ: если спец-функции для таблиц-картинок нет, просто отключаем её
    vision_extract_table_values = None  # type: ignore

    # NEW: мягкие фолбэки
    chat_with_gpt_multimodal = None  # type: ignore
    chat_with_gpt_stream_multimodal = None  # type: ignore


    # безопасные заглушки для figures, чтобы остальной код не падал
    fig_index_document = None       # type: ignore
    fig_load_index = None           # type: ignore

    def fig_find(*args, **kwargs):  # type: ignore
        return []

    def figure_display_name(rec):   # type: ignore
        rec = rec or {}
        return str(
            rec.get("title")
            or rec.get("caption")
            or rec.get("num")
            or "Рисунок"
        )




# НОВОЕ: оркестратор приёма/обогащения (OCR таблиц-картинок, нормализация чисел)
from .ingest_orchestrator import enrich_sections, ingest_document
# НОВОЕ: аналитика таблиц
from .analytics import analyze_table_by_num

# утилиты
from .utils import safe_filename, sha256_bytes, split_for_telegram, infer_doc_kind

# гибридный контекст: семантика + FTS/LIKE
from .lexsearch import best_context

# сразу под текущими import’ами
from .paywall_stub import setup_paywall

# где у вас создаются объекты бота и диспетчера:
bot = Bot(Cfg.TG_TOKEN)
dp = Dispatcher()
# добавьте эту строку (один раз):
setup_paywall(dp, bot)

register_docs_handlers(dp)

# --------------------- ПАРАМЕТРЫ СТРИМИНГА (с дефолтами) ---------------------

STREAM_ENABLED: bool = getattr(Cfg, "STREAM_ENABLED", True)
STREAM_EDIT_INTERVAL_MS: int = getattr(Cfg, "STREAM_EDIT_INTERVAL_MS", 900)
STREAM_MIN_CHARS: int = getattr(Cfg, "STREAM_MIN_CHARS", 120)
STREAM_MODE: str = getattr(Cfg, "STREAM_MODE", "edit")
TG_MAX_CHARS: int = getattr(Cfg, "TG_MAX_CHARS", 3900)
FIG_MEDIA_LIMIT: int = getattr(Cfg, "FIG_MEDIA_LIMIT", 12)

TG_SPLIT_TARGET: int = getattr(Cfg, "TG_SPLIT_TARGET", 2000)
TG_SPLIT_MAX_PARTS: int = getattr(Cfg, "TG_SPLIT_MAX_PARTS", 6)

# ↓ НОВОЕ: пауза между кусками при нестримовой отправке
MULTIPART_SLEEP_MS: int = getattr(Cfg, "MULTIPART_SLEEP_MS", 200)

_SPLIT_ANCHOR_RE = re.compile(
    r"(?m)^(?:### .+|## .+|\*\*[^\n]+?\*\*|\d+[).] .+|- .+)$"
)  # предпочитаемые границы (заголовки/списки)
STREAM_HEAD_START_MS: int = getattr(Cfg, "STREAM_HEAD_START_MS", 250)        # первый апдейт быстрее
FINAL_MAX_TOKENS: int = getattr(Cfg, "FINAL_MAX_TOKENS", 5000)
TYPE_INDICATION_EVERY_MS: int = getattr(Cfg, "TYPE_INDICATION_EVERY_MS", 2000)
# NEW: строгий режим для рисунков — не отдаём числа без надёжного источника
FIG_STRICT: bool = getattr(Cfg, "FIG_STRICT", True)
# ↓ новое: управление многошаговой подачей
MULTI_STEP_SEND_ENABLED: bool = getattr(Cfg, "MULTI_STEP_SEND_ENABLED", True)
MULTI_STEP_MIN_ITEMS: int = getattr(Cfg, "MULTI_STEP_MIN_ITEMS", 2)
MULTI_STEP_MAX_ITEMS: int = getattr(Cfg, "MULTI_STEP_MAX_ITEMS", 8)
MULTI_STEP_FINAL_MERGE: bool = getattr(Cfg, "MULTI_STEP_FINAL_MERGE", True)
MULTI_STEP_PAUSE_MS: int = getattr(Cfg, "MULTI_STEP_PAUSE_MS", 120)  # м/у блоками
MULTI_PASS_SCORE: int = getattr(Cfg, "MULTI_PASS_SCORE", 85)         # порог критика в ace
# минимальная длина вопроса, при которой вообще есть смысл городить подпункты
MULTI_STEP_MIN_QUESTION_LEN: int = getattr(Cfg, "MULTI_STEP_MIN_QUESTION_LEN", 200)


# --------------------- форматирование и отправка ---------------------
# Markdown → HTML (минимально-необходимое: заголовки, **bold**, *italic*, `code`)
_MD_H_RE       = re.compile(r"(?m)^\s{0,3}#{1,6}\s+(.+?)\s*$")
_MD_BOLD_RE    = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_MD_BOLD2_RE   = re.compile(r"__(.+?)__", re.DOTALL)
_MD_ITALIC_RE  = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", re.DOTALL)
_MD_ITALIC2_RE = re.compile(r"(?<!_)_(?!_)(.+?)(?<!_)_(?!_)", re.DOTALL)
_MD_CODE_RE    = re.compile(r"`([^`]+)`")

def _to_html(text: str) -> str:
    if not text:
        return ""
    original = text

    # 0) временно заменим кодовые спаны плейсхолдерами
    code_buf = []
    def _stash(m):
        code_buf.append(m.group(1))
        return f"@@CODE{len(code_buf)-1}@@"

    txt = _MD_CODE_RE.sub(_stash, original)
    txt = html.escape(txt)

    # 1) заголовки/жирный/курсив
    txt = _MD_H_RE.sub(r"<b>\1</b>", txt)
    txt = _MD_BOLD_RE.sub(r"<b>\1</b>", txt)
    txt = _MD_BOLD2_RE.sub(r"<b>\1</b>", txt)
    txt = _MD_ITALIC_RE.sub(r"<i>\1</i>", txt)
    txt = _MD_ITALIC2_RE.sub(r"<i>\1</i>", txt)

    # 2) зачистка «висячих» ** — уже безопасно (в коде их нет)
    txt = re.sub(r"(?<!\*)\*\*(?!\*)", "", txt)

    # 3) вернуть кодовые спаны, экранировав их контент
    def _restore(m):
        i = int(m.group(1))
        return f"<code>{html.escape(code_buf[i])}</code>"
    txt = re.sub(r"@@CODE(\d+)@@", _restore, txt)

    return txt


# -------- Приветствие --------
_GREET_RE = re.compile(
    r"(?i)\b(привет|здравств|добрый\s*(день|вечер|утро)|hello|hi|hey|хай|салют|ку)\b"
)

def _is_greeting(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    # короткие приветствия или фразы, где встречается ключевое слово
    return bool(_GREET_RE.search(t))


def _split_multipart(text: str,
                     *,
                     target: int = TG_SPLIT_TARGET,
                     max_parts: int = TG_SPLIT_MAX_PARTS,  # параметр оставлен для совместимости, НЕ используется
                     hard: int = TG_MAX_CHARS) -> list[str]:
    s = text or ""
    if not s:
        return []
    parts: list[str] = []
    rest = s

    # режем по «красивым» границам столько раз, сколько нужно
    while len(rest) > target:
        cut = -1
        for m in _SPLIT_ANCHOR_RE.finditer(rest[: min(len(rest), hard)]):
            if m.start() < target:
                cut = m.start()
        if cut <= 0:
            cut = _smart_cut_point(rest, min(hard, target))
        parts.append(rest[:cut].rstrip())
        rest = rest[cut:].lstrip()

    # финальный хвост и сверхжёсткое разбиение по лимиту Telegram
    while rest:
        parts.append(rest[:hard])
        rest = rest[hard:]
    return parts


async def _send(m: types.Message, text: str):
    """Бережно отправляем длинный текст частями в HTML-режиме (нестримовый фолбэк)."""
    chunks = _split_multipart(text or "")
    logger.info(
        "SEND: %d chunk(s) to chat_id=%s (message_id=%s), total_len=%d",
        len(chunks),
        m.chat.id,
        getattr(m, "message_id", None),
        len(text or ""),
    )
    for i, chunk in enumerate(chunks):
        # небольшая пауза между сообщениями, чтобы не заспамить чат
        if i > 0 and MULTIPART_SLEEP_MS > 0:
            await asyncio.sleep(MULTIPART_SLEEP_MS / 1000)

        try:
            await m.answer(
                _to_html(chunk),
                parse_mode="HTML",
                disable_web_page_preview=True,
            )
            logger.debug(
                "SEND: chunk %d/%d sent, len=%d",
                i + 1,
                len(chunks),
                len(chunk),
            )
        except Exception:
            logger.exception(
                "SEND: failed to send chunk %d/%d (len=%d)",
                i + 1,
                len(chunks),
                len(chunk),
            )


# ---- Verbosity helpers ----
def _detect_verbosity(text: str) -> str:
    t = (text or "").lower()
    detailed = re.search(r"\b(подробн|детал|развёрнут|развернут|разбор|explain in detail|detailed)\b", t)
    brief    = re.search(r"\b(кратк|в\s*двух\s*слов|коротк|выжимк|summary|brief)\b", t)
    if detailed:
        return "detailed"
    if brief:
        return "brief"
    # эвристика: очень длинное сообщение — скорее подробный ответ
    if len(t) > 600:
        return "detailed"
    return "normal"


def _verbosity_addendum(verbosity: str, what: str = "ответ") -> str:
    """
    Небольшая приписка к промпту в зависимости от требуемой детализации.
    `what` — что именно нужно описывать: 'ответ', 'описания рисунков' и т.п.
    """
    what = (what or "ответ").strip()

    if verbosity in ("short", "brief"):
        # пример: "Ответь кратко (по описанию рисунков)."
        return f" Ответь кратко (по {what})."

    if verbosity == "detailed":
        # пример: "Дай развёрнутое, подробное описание рисунков."
        return f" Дай развёрнутое, подробное {what}."

    # default — без дополнительных указаний
    return ""

# --------------------- STREAM: вспомогалки ---------------------

def _now_ms() -> int:
    return int(time.time() * 1000)

async def _typing_loop(chat_id: int, stop_event: asyncio.Event):
    try:
        while not stop_event.is_set():
            await bot.send_chat_action(chat_id, ChatAction.TYPING)
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=TYPE_INDICATION_EVERY_MS / 1000)
            except asyncio.TimeoutError:
                # просто продолжаем цикл, чтобы периодически слать "typing"
                pass
    except Exception:
        # глушим любые нетипичные ошибки, чтобы не ронять стрим
        pass



def _section_context(owner_id: int, doc_id: int, sec: str, *, max_chars: int = 9000) -> str:
    # 1) генерим варианты записи номера
    base = (sec or "").strip()
    variants = {
        base,
        base.replace(" ", ""),
        base.replace(" ", "").replace(",", "."),
        base.replace(" ", "").replace(".", ","),
    }
    prefixes = ["", "Раздел ", "Пункт ", "Глава ", "Подраздел "]
    patterns = [f"%{v}%" for v in variants]
    patterns += [f"%{p}{base}%" for p in prefixes]
    # уберём дубли и ограничим разумным числом плейсхолдеров
    patterns = list(dict.fromkeys(patterns))[:8]

    con = get_conn()
    cur = con.cursor()

    rows = []
    if patterns:
        placeholders = " OR ".join(["section_path LIKE ?"] * len(patterns))
        cur.execute(
            f"""
            SELECT page, section_path, text
            FROM chunks
            WHERE owner_id=? AND doc_id=? AND ({placeholders})
            ORDER BY page ASC, id ASC
            """,
            (owner_id, doc_id, *patterns),
        )
        rows = cur.fetchall() or []

    # 2) фолбэк: найдём heading с номером и возьмём его секцию
    if not rows:
        has_et = _table_has_columns(con, "chunks", ["element_type"])
        if has_et:
            cur.execute(
                """
                SELECT section_path
                FROM chunks
                WHERE owner_id=? AND doc_id=?
                AND (element_type='heading' OR element_type IS NULL)
                AND (section_path LIKE ? OR text LIKE ?)
                ORDER BY page ASC, id ASC LIMIT 1
                """,
                (owner_id, doc_id, f"%{base}%", f"%{base}%"),
            )
        else:
            # старая схема — без условия по element_type
            cur.execute(
                """
                SELECT section_path
                FROM chunks
                WHERE owner_id=? AND doc_id=?
                AND (section_path LIKE ? OR text LIKE ?)
                ORDER BY page ASC, id ASC LIMIT 1
                """,
                (owner_id, doc_id, f"%{base}%", f"%{base}%"),
            )
        h = cur.fetchone()
        if h and h["section_path"]:
            cur.execute(
                """
                SELECT page, section_path, text
                FROM chunks
                WHERE owner_id=? AND doc_id=? AND section_path=?
                ORDER BY page ASC, id ASC
                """,
                (owner_id, doc_id, h["section_path"]),
            )
            rows = cur.fetchall() or []

    con.close()
    if not rows:
        return ""

    parts, total = [], 0
    header_inserted = False
    for r in rows:
        secpath = (r["section_path"] or "").strip()
        t = (r["text"] or "").strip()
        if not t:
            continue
        chunk = (f"[{secpath}]\n{t}") if not header_inserted else t
        header_inserted = True
        if total + len(chunk) > max_chars:
            parts.append(chunk[: max_chars - total])
            break
        parts.append(chunk)
        total += len(chunk)
    return "\n\n".join(parts)


def _ensure_iterable(stream_obj) -> Iterable[str]:
    """Нормализуем в (a)синхронный итератор строк; поддерживаем случай, когда прилетела корутина."""
    import inspect

    # Если это корутина — обернём в async-генератор, который сначала её await-ит,
    # а потом уже итерируется по реальному стриму.
    if inspect.iscoroutine(stream_obj):
        async def _await_then_iter():
            real = await stream_obj
            if hasattr(real, "__aiter__"):
                async for chunk in real:
                    yield chunk
            else:
                for chunk in real:
                    yield chunk
        return _await_then_iter()

    if hasattr(stream_obj, "__aiter__"):
        async def _drain_to_queue(q: asyncio.Queue):
            try:
                async for chunk in stream_obj:  # type: ignore
                    await q.put(chunk or "")
            except Exception:
                pass
            finally:
                await q.put(None)

        queue: asyncio.Queue = asyncio.Queue()

        async def _producer():
            await _drain_to_queue(queue)

        asyncio.create_task(_producer())

        async def _async_iter():
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
        return _async_iter()

    return stream_obj

async def _iterate_chunks(stream_obj) -> AsyncIterable[str]:
    """Единый асинхронный источник чанков (умеет работать и с sync-, и с async-итераторами)."""
    if hasattr(stream_obj, "__aiter__"):
        async for ch in stream_obj:
            if ch:
                yield str(ch)
        return
    for ch in stream_obj:
        if ch:
            yield str(ch)

def _smart_cut_point(s: str, limit: int) -> int:
    """Ищем «красивое» место разреза <= limit (по переносу/точке/пробелу)."""
    if len(s) <= limit:
        return len(s)
    cut = s.rfind("\n", 0, limit)
    if cut == -1:
        cut = s.rfind(". ", 0, limit)
    if cut == -1:
        cut = s.rfind(" ", 0, limit)
    if cut == -1:
        cut = limit
    return max(1, cut)

# --- [VISION] helpers: выбрать картинки и привести пары значений ---
def _pick_images_from_hits(hits: list[dict], limit: int = 3) -> list[str]:
    acc: list[str] = []
    for h in hits or []:
        attrs = (h.get("attrs") or {})

        # 1) стандартный путь: attrs.images
        for p in (attrs.get("images") or []):
            if p and os.path.exists(p) and p not in acc:
                acc.append(p)
            if len(acc) >= limit:
                return acc

        # 2) фолбэк: иногда путь лежит прямо в самом хите
        for p in (
            h.get("image_path"),
            h.get("image"),
        ):
            if p and os.path.exists(p) and p not in acc:
                acc.append(p)
            if len(acc) >= limit:
                return acc

    return acc


def _pairs_to_bullets(pairs: list[dict]) -> str:
    """
    Аккуратно форматируем пары (label, value, unit):
    - 0.25 при unit='%’ → 25%;
    - числа округляем до целых или 2 знаков;
    - убираем хвосты вида 0.42000000000000004.
    """
    def _fmt(value, unit: str) -> str:
        unit = (unit or "").strip()
        sval = ""
        v_num: float | None = None

        # пробуем привести к числу
        if isinstance(value, (int, float, Decimal)):
            v_num = float(value)
        else:
            try:
                v_num = float(str(value).replace(",", "."))
            except Exception:
                sval = str(value) if value is not None else ""

        if v_num is not None:
            # эвристика: доли с unit='%' → проценты
            if unit and "%" in unit and 0.0 <= v_num <= 1.2:
                v_num *= 100.0

            if abs(v_num - round(v_num)) < 0.05:
                sval = str(int(round(v_num)))
            else:
                sval = f"{v_num:.2f}".rstrip("0").rstrip(".")

        # добавляем единицы измерения
        if unit:
            if "%" in unit and not sval.endswith("%"):
                sval += "%"
            else:
                sval += f" {unit}"
        return sval

    lines: list[str] = []
    for r in (pairs or []):
        lab = (str(r.get("label") or "")).strip()
        unit = (str(r.get("unit") or "")).strip()
        raw_val = r.get("value")
        val = _fmt(raw_val, unit)

        if not lab and not val:
            continue
        if lab and val:
            lines.append(f"— {lab}: {val}")
        elif lab:
            lines.append(f"— {lab}")
        else:
            lines.append(f"— {val}")
    return "\n".join(lines)


async def _stream_to_telegram(m: types.Message, stream, head_text: str = "⌛️ Печатаю ответ…") -> None:
    logger.info(
        "STREAM: start for chat_id=%s message_id=%s",
        m.chat.id,
        getattr(m, "message_id", None),
    )
    current_text = ""
    sent_parts = 0
    initial = await m.answer(
        _to_html(head_text),
        parse_mode="HTML",
        disable_web_page_preview=True,
    )
    last_edit_at = _now_ms() - STREAM_HEAD_START_MS
    stop_typer = asyncio.Event()
    typer_task = asyncio.create_task(_typing_loop(m.chat.id, stop_event=stop_typer))

    # 🔧 новое: после первой части в multi больше не редактируем initial
    freeze_initial = False

    try:
        async for delta in _iterate_chunks(_ensure_iterable(stream)):
            current_text += delta

            # 3.a) мульти-режим: сбрасываем порциями
            if STREAM_MODE == "multi" and len(current_text) >= TG_SPLIT_TARGET:
                cut = -1
                for mm in _SPLIT_ANCHOR_RE.finditer(current_text[: min(len(current_text), TG_MAX_CHARS)]):
                    if mm.start() < TG_SPLIT_TARGET:
                        cut = mm.start()
                if cut <= 0:
                    cut = _smart_cut_point(current_text, min(TG_MAX_CHARS, TG_SPLIT_TARGET))

                part = current_text[:cut].rstrip()
                try:
                    if sent_parts == 0:
                        await initial.edit_text(
                            _to_html(part),
                            parse_mode="HTML",
                            disable_web_page_preview=True,
                        )
                        freeze_initial = True  # <- больше не трогаем initial
                    else:
                        await m.answer(
                            _to_html(part),
                            parse_mode="HTML",
                            disable_web_page_preview=True,
                        )
                except TelegramBadRequest:
                    await m.answer(
                        _to_html(part),
                        parse_mode="HTML",
                        disable_web_page_preview=True,
                    )

                sent_parts += 1  # ### ДОБАВЛЕНО: считаем отправленные части
                current_text = current_text[cut:].lstrip()
                last_edit_at = _now_ms()
                continue

            # 3.b) защита от лимита
            if len(current_text) >= TG_MAX_CHARS:
                cut = _smart_cut_point(current_text, TG_MAX_CHARS)
                final_part = current_text[:cut]

                if STREAM_MODE == "multi" and (freeze_initial or sent_parts > 0):
                    # 🔧 в multi не редактируем initial после 1-й части
                    await m.answer(
                        _to_html(final_part),
                        parse_mode="HTML",
                        disable_web_page_preview=True,
                    )
                else:
                    try:
                        await initial.edit_text(
                            _to_html(final_part),
                            parse_mode="HTML",
                            disable_web_page_preview=True,
                        )
                    except TelegramBadRequest:
                        await m.answer(
                            _to_html(final_part),
                            parse_mode="HTML",
                            disable_web_page_preview=True,
                        )

                sent_parts += 1  # ### ДОБАВЛЕНО: эта часть тоже считается отправленной
                current_text = current_text[cut:].lstrip()
                # 🔧 новый плейсхолдер нужен только в edit-режиме
                if STREAM_MODE == "edit":
                    initial = await m.answer(
                        _to_html("…"),
                        parse_mode="HTML",
                        disable_web_page_preview=True,
                    )
                last_edit_at = _now_ms()
                continue

            # 3.c) периодические правки — 🔧 ТОЛЬКО в режиме edit
            now = _now_ms()
            if (
                STREAM_MODE == "edit"
                and (now - last_edit_at) >= STREAM_EDIT_INTERVAL_MS
                and len(current_text) >= STREAM_MIN_CHARS
            ):
                try:
                    await initial.edit_text(
                        _to_html(current_text),
                        parse_mode="HTML",
                        disable_web_page_preview=True,
                    )
                    last_edit_at = now
                except TelegramBadRequest:
                    pass

        # финальный хвост
        if current_text:
            logger.info(
                "STREAM: finishing with tail, len=%d, sent_parts=%d",
                len(current_text),
                sent_parts,
            )
            # аккуратно режем остаток так же, как в нестримовом режиме
            tail_parts = _split_multipart(current_text or "")

            if STREAM_MODE == "multi" and sent_parts > 0:
                # в multi-режиме после первой части всё остальное — отдельными сообщениями
                for part in tail_parts:
                    html_part = _to_html(part)
                    try:  # ### ДОБАВЛЕНО: не даём исключению убить весь хвост
                        await m.answer(
                            html_part,
                            parse_mode="HTML",
                            disable_web_page_preview=True,
                        )
                    except TelegramBadRequest:
                        # если HTML/длина сломали разметку — шлём как есть
                        await m.answer(part)
            else:
                # в edit-режиме (или когда ещё не было частей) первый кусок пытаемся
                # положить в initial, а остальное — отдельными сообщениями
                first = True
                for part in tail_parts:
                    html_part = _to_html(part)
                    if first:
                        try:
                            await initial.edit_text(
                                html_part,
                                parse_mode="HTML",
                                disable_web_page_preview=True,
                            )
                        except TelegramBadRequest:
                            # если редактирование не удалось — шлём отдельным сообщением
                            try:
                                await m.answer(
                                    html_part,
                                    parse_mode="HTML",
                                    disable_web_page_preview=True,
                                )
                            except TelegramBadRequest:
                                await m.answer(part)
                        first = False
                    else:
                        try:
                            await m.answer(
                                html_part,
                                parse_mode="HTML",
                                disable_web_page_preview=True,
                            )
                        except TelegramBadRequest:
                            await m.answer(part)

    except Exception:
        logger.exception(
            "STREAM: unexpected error for chat_id=%s",
            m.chat.id,
        )
    finally:
        stop_typer.set()
        try:
            await typer_task
        except Exception:
            pass
        logger.info(
            "STREAM: stop for chat_id=%s message_id=%s",
            m.chat.id,
            getattr(m, "message_id", None),
        )


def _plan_subtasks_via_gpt(question: str, max_items: int = 8) -> list[dict]:
    """
    Планировщик подпунктов без ACE.
    Берёт исходный вопрос и просит GPT разбить его на 2–N подпунктов.
    Возвращает список dict: {"id": int, "ask": str}.
    """
    question = (question or "").strip()
    if not question:
        return []

    if "chat_with_gpt" not in globals() or chat_with_gpt is None:
        return []

    system_prompt = (
        "Ты помогаешь студенту с дипломом. Получив сложный или многочастный вопрос, "
        "разбей его на несколько более простых подпунктов, которые можно последовательно разобрать. "
        "Верни ТОЛЬКО JSON-массив без текста вокруг, формата:\n"
        "[{\"id\": 1, \"ask\": \"...\"}, {\"id\": 2, \"ask\": \"...\"}, ...].\n"
        "Не добавляй пояснений, комментариев и текста вне JSON."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    try:
        raw = chat_with_gpt(messages, temperature=0.0, max_tokens=400) or ""
    except Exception as e:
        logging.exception("plan_subtasks_via_gpt failed: %s", e)
        return []

    raw = raw.strip()

    # Пытаемся выдернуть JSON-массив
    data = None
    try:
        data = json.loads(raw)
    except Exception:
        m = re.search(r"\[[\s\S]*\]", raw)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except Exception:
            return []

    if not isinstance(data, list):
        return []

    items: list[dict] = []
    for i, it in enumerate(data, start=1):
        if isinstance(it, str):
            ask = it.strip()
            if not ask:
                continue
            items.append({"id": i, "ask": ask})
        elif isinstance(it, dict):
            ask = str(it.get("ask") or it.get("question") or it.get("text") or "").strip()
            if not ask:
                continue
            iid = it.get("id") or i
            items.append({"id": iid, "ask": ask})
        if len(items) >= max_items:
            break

    return items


def _answer_subpoint_via_gpt(
    ask: str,
    ctx_text: str,
    base_question: str,
    *,
    verbosity: str = "normal",
) -> str:
    """
    Генерация ответа по одному подпункту через GPT (без ACE).
    """
    ask = (ask or "").strip()
    if not ask:
        return ""

    if "chat_with_gpt" not in globals() or chat_with_gpt is None:
        return ""

    ctx = (ctx_text or "").strip()

    system_prompt = (
        "Ты репетитор по дипломным работам. Тебе дали фрагмент текста диплома "
        "и один подпункт вопроса.\n"
        "Отвечай ТОЛЬКО по этому фрагменту. Не придумывай фактов, терминов и предметную область, "
        "которых в тексте нет (например, не упоминай продажи, клиентов, выручку, маркетинг и т.п., "
        "если этих слов нет во фрагменте). Если информации недостаточно для уверенного ответа, "
        "честно скажи об этом и прямо напиши, что в этом фрагменте об этом не сказано.\n"
        "Не добавляй разделы вида «чего не хватает»."
    )


    if ctx:
        assistant_ctx = f"[Фрагмент диплома]\n{ctx}"
    else:
        assistant_ctx = "[Фрагмент по этому подпункту не найден в тексте документа]"

    user_prompt = (
        f"Исходный общий вопрос пользователя:\n{base_question}\n\n"
        f"Текущий подпункт (подвопрос): {ask}\n\n"
        "Ответь только по этому подпункту, опираясь на переданный фрагмент диплома."
        f"{_verbosity_addendum(verbosity)}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": assistant_ctx},
        {"role": "user", "content": user_prompt},
    ]

    try:
        ans = chat_with_gpt(messages, temperature=0.2, max_tokens=FINAL_MAX_TOKENS) or ""
    except Exception as e:
        logging.exception("answer_subpoint_via_gpt failed: %s", e)
        return ""

    return ans.strip()


def _merge_subanswers_via_gpt(
    base_question: str,
    items: list[dict],
    subanswers: list[str],
    *,
    verbosity: str = "normal",
) -> str:
    """
    Финальный сводный ответ по всем подпунктам без ACE.
    """
    if not subanswers:
        return ""

    if "chat_with_gpt" not in globals() or chat_with_gpt is None:
        return ""

    blocks: list[str] = []
    for i, ans in enumerate(subanswers, start=1):
        it = items[i - 1] if i - 1 < len(items) else {}
        ask = (isinstance(it, dict) and (it.get("ask") or "")) or ""
        ask = str(ask).strip()
        header = f"[Подпункт {i}" + (f": {ask}]" if ask else "]")
        blocks.append(f"{header}\n{ans}")

    ctx = "\n\n".join(blocks)

    system_prompt = (
        "Ты репетитор по дипломным работам. Ниже собраны ответы по отдельным подпунктам "
        "одного большого вопроса. Твоя задача — сделать один связный общий ответ.\n"
        "Не повторяй дословно все подпункты, а аккуратно их объединяй. "
        "Не добавляй новых фактов, терминов и предметную область, которых нет в подпунктах "
        "(например, не придумывай продажи, клиентов, выручку, маркетинг и т.п., если этого нет "
        "в самих подпунктах).\n"
        "Не пиши разделы вида «чего не хватает»."
    )


    user_prompt = (
        f"Исходный общий вопрос пользователя:\n{base_question}\n\n"
        "На него уже есть ответы по подпунктам (см. ниже). "
        "Собери из них один цельный ответ для пользователя."
        f"{_verbosity_addendum(verbosity)}\n\n"
        "[Ответы по подпунктам]\n"
        f"{ctx}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        merged = chat_with_gpt(messages, temperature=0.2, max_tokens=FINAL_MAX_TOKENS) or ""
    except Exception as e:
        logging.exception("merge_subanswers_via_gpt failed: %s", e)
        return ""

    return merged.strip()


async def _run_multistep_answer(
    m: types.Message,
    uid: int,
    doc_id: int,
    q_text: str,
    *,
    discovered_items: list[dict] | None = None,
) -> bool:
    """
    Многошаговый ответ без ACE:
    1) План подпунктов — через _plan_subtasks_via_gpt или coverage.
    2) По каждому подпункту — отдельный вызов GPT с жёстким контекстом.
    3) (опц.) финальный merge через _merge_subanswers_via_gpt.
    """
    if not MULTI_STEP_SEND_ENABLED:
        return False

    # GPT обязателен для этого режима
    if "chat_with_gpt" not in globals() or chat_with_gpt is None:
        return False

    verbosity = _detect_verbosity(q_text)

    # 1) план из coverage/discovered_items или строим через GPT
    items = (discovered_items or [])
    if not items:
        items = _plan_subtasks_via_gpt(q_text, max_items=MULTI_STEP_MAX_ITEMS)

    # нормализация: поддерживаем и dict, и str
    norm_items: list[dict] = []
    for idx, it in enumerate(items, start=1):
        if isinstance(it, str):
            ask = it.strip()
            if ask:
                norm_items.append({"id": idx, "ask": ask})
        elif isinstance(it, dict):
            ask = (it.get("ask") or it.get("text") or it.get("q") or "").strip()
            if ask:
                norm_items.append({"id": it.get("id") or idx, "ask": ask})

    items = [it for it in norm_items if (it.get("ask") or "").strip()]
    if len(items) < MULTI_STEP_MIN_ITEMS:
        return False

    # отсечём хвост по лимиту
    items = items[:MULTI_STEP_MAX_ITEMS]

    # краткий анонс
    preview = "\n".join([f"{i+1}) {(it['ask'] or '').strip()}" for i, it in enumerate(items)])
    await _send(
        m,
        f"Вопрос многочастный. Отвечаю по подпунктам ({len(items)} шт.):\n\n{preview}",
    )

    subanswers: list[str] = []

    # coverage-aware раздача контекстов: разложим выжимки по подпунктам
    cov = None
    try:
        cov = retrieve_coverage(owner_id=uid, doc_id=doc_id, question=q_text)
    except Exception:
        cov = None
    cov_map = (cov or {}).get("by_item") or {}

    # по очереди: A → send, B → send, ...
    for i, it in enumerate(items, start=1):
        ask = (it.get("ask") or "").strip()
        if not ask:
            continue

        # контекст для конкретного подпункта
        ctx_text = ""
        try:
            # если есть coverage-бакет — собираем контекст прямо из чанков подпункта
            bucket = cov_map.get(str(it.get("id") or i)) or []
            if bucket:
                ctx_text = build_context_coverage(bucket, items_count=1)
        except Exception:
            ctx_text = ""

        # фолбэки по контексту
        if not ctx_text:
            ctx_text = best_context(uid, doc_id, ask, max_chars=6000) or ""
        if not ctx_text:
            hits = retrieve(uid, doc_id, ask, top_k=8)
            if hits:
                ctx_text = build_context(hits)
        if not ctx_text:
            ctx_text = _first_chunks_context(uid, doc_id, n=12, max_chars=6000)

        # генерация по подпункту через GPT (без ACE)
        try:
            part = _answer_subpoint_via_gpt(
                ask=ask,
                ctx_text=ctx_text,
                base_question=q_text,
                verbosity=verbosity,
            )
        except Exception as e:
            logging.exception("answer_subpoint_via_gpt failed: %s", e)
            part = ""

        # отправка блока
        header = f"**{i}. {ask}**\n\n"
        await _send(m, header + (part or "Не удалось сгенерировать ответ по этому подпункту."))
        subanswers.append(f"{header}{part}")

        # микропаузa, чтобы не упереться в rate/чаты
        await asyncio.sleep(MULTI_STEP_PAUSE_MS / 1000)

    # (опционально) финальный сводный блок
    if MULTI_STEP_FINAL_MERGE and subanswers:
        try:
            merged = _merge_subanswers_via_gpt(
                base_question=q_text,
                items=items,
                subanswers=subanswers,
                verbosity=verbosity,
            ).strip()
            if merged:
                await _send(m, "**Итоговый сводный ответ**\n\n" + merged)
        except Exception as e:
            logging.exception("merge_subanswers_via_gpt failed: %s", e)

    return True

def _should_use_multistep(q_text: str, discovered_items: list[dict] | None) -> bool:
    """
    Простая эвристика: включаем многошаговый режим, только если:
      — мультиответ вообще разрешён конфигом;
      — есть подпункты из coverage/general_subitems;
      — подпунктов не меньше MULTI_STEP_MIN_ITEMS;
      — вопрос достаточно длинный (чтобы стоило городить подпункты).
    """
    if not MULTI_STEP_SEND_ENABLED:
        return False

    if not discovered_items:
        return False

    if len(discovered_items) < MULTI_STEP_MIN_ITEMS:
        return False

    if len((q_text or "").strip()) < MULTI_STEP_MIN_QUESTION_LEN:
        return False

    return True


# summarizer (мягкий импорт)
try:
    from .summarizer import is_summary_intent, overview_context  # могут отсутствовать — есть фолбэки ниже
except Exception:
    def is_summary_intent(text: str) -> bool:
        return bool(re.search(
            r"\b(суть|кратко|основн|главн|summary|overview|итог|вывод)\w*\b",
            text or "",
            re.IGNORECASE,
        ))

    def overview_context(owner_id: int, doc_id: int, max_chars: int = 6000) -> str:
        con = get_conn()
        cur = con.cursor()
        cur.execute(
            """
            SELECT page, section_path, text
            FROM chunks
            WHERE owner_id=? AND doc_id=?
              AND (text LIKE '[Заголовок]%%'
                   OR text LIKE '%%Цель%%'
                   OR text LIKE '%%Задач%%'
                   OR text LIKE '%%Введен%%'
                   OR text LIKE '%%Заключен%%'
                   OR text LIKE '%%Вывод%%')
            ORDER BY page ASC, id ASC
            LIMIT 14
            """,
            (owner_id, doc_id),
        )
        rows = cur.fetchall()
        con.close()
        if not rows:
            return ""
        parts, total = [], 0
        for r in rows:
            block = f"{(r['text'] or '').strip()}"
            if total + len(block) > max_chars:
                break
            parts.append(block)
            total += len(block)
        return "\n\n".join(parts)

# NEW: точные значения для одного рисунка (через vision-анализ в summarizer.py)
try:
    from .summarizer import extract_figure_value as summ_extract_figure_value  # type: ignore
except Exception:
    # если summarizer или сама функция недоступны — просто не используем фолбэк по картинкам
    summ_extract_figure_value = None  # type: ignore

# vision-описание рисунков (мягкий импорт; если нет — отвечаем текстовым фолбэком)
try:
    from .summarizer import describe_figures as vision_describe_figures
except Exception:
    def vision_describe_figures(owner_id: int, doc_id: int, numbers: list[str]) -> str:
        if not numbers:
            return "Не указаны номера рисунков."
        return "Описания рисунков недоступны (vision-модуль не подключён)."


# NEW: точечный анализ одной картинки (связный текст + числа)
try:
    from .vision_analyzer import analyze_figure as va_analyze_figure  # type: ignore
except Exception:
    va_analyze_figure = None  # type: ignore

# ГОСТ-валидатор (мягкий импорт)
try:
    from .validators_gost import validate_gost, render_report
except Exception:
    validate_gost = None
    render_report = None


# «Активный документ» в памяти процесса
ACTIVE_DOC: dict[int, int] = {}  # user_id -> doc_id
# NEW: короткая «память» последнего упомянутого объекта пользователем
LAST_REF: dict[int, dict] = {}   # {uid: {"figure_nums": list[str], "area": "3.2"}}
FIG_INDEX: dict[int, dict] = {}
OOXML_INDEX: dict[int, dict] = {}
# NEW: кеш для распознанных «таблиц-картинок» (doc_id, num) -> текстовый блок
_OCR_TABLE_CACHE: dict[tuple[int, str], str] = {}

# NEW: ждём ли от пользователя «да/нет» на предложение ответа от [модель]
MODEL_EXTRA_PENDING: dict[int, dict] = {}   # {uid: {"question": str}}

# NEW: для подстановки номера раздела из вопроса и для анафоры «этот пункт/рисунок»
# NEW: для подстановки номера раздела из вопроса и для анафоры «этот пункт/рисунок»
_SECTION_NUM_RE = re.compile(
    r"(?i)\b(?:глава\w*|раздел\w*|пункт\w*|подраздел\w*|sec(?:tion)?\.?|chapter)"
    r"\s*(?:№\s*)?((?:[A-Za-zА-Яа-я](?=[\.\d]))?\s*\d+(?:[.,]\d+)*)"
)
_ANAPH_HINT_RE = re.compile(r"(?i)\b(этот|эта|это|данн\w+|про него|про неё|про нее)\b")

# считаем "уточняющим" любой запрос, где есть глагол + слово "подробнее" в одной фразе
_FOLLOWUP_MORE_RE = re.compile(
    r"(?i)\b(опиши|распиши|объясни|расскажи)\b.*\bподробнее\b|\bподробнее\b.*\b(опиши|распиши|объясни|расскажи)\b"
)

def _expand_with_last_referent(uid: int, text: str) -> str:
    """
    Подставляем последний объект (таблица/рисунок/пункт) для реплик вида:
      - «опиши подробнее»
      - «расскажи про него»
      - «опиши её подробнее»
    чтобы они превратились, например, в
      - «опиши подробнее (имеется в виду таблица 4)».
    """
    t = (text or "").strip()
    if not t:
        return text

    # если уже явно указана таблица/рисунок/пункт — ничего не меняем
    if _TABLE_NUM_IN_TEXT_RE.search(t) or FIG_NUM_RE.search(t) or _SECTION_NUM_RE.search(t):
        return text

    # нет ни анафоры («этот/про неё»), ни короткого фоллоу-апа «опиши подробнее» — выходим
    if not (_ANAPH_HINT_RE.search(t) or _FOLLOWUP_MORE_RE.search(t)):
        return text

    last = LAST_REF.get(uid) or {}

    # 1) приоритет — последняя таблица
    tables = last.get("table_nums") or []
    if tables:
        num = str(tables[0])
        return f"{text} (имеется в виду таблица {num})"

    # 2) затем — последний рисунок
    figs = last.get("figure_nums") or []
    if figs:
        num = str(figs[0])
        return f"{text} (имеется в виду рисунок {num})"

    # 3) затем — последний пункт/раздел
    area = (last.get("area") or "").strip()
    if area:
        if not re.search(r"(?i)\b(глава|раздел|пункт|подраздел)\b", t):
            return f"{text} (имеется в виду пункт {area})"
        return f"{text} ({area})"

    return text


# ------------------------ Гардрейлы ------------------------

_BANNED_PATTERNS = [
    r"jail ?break|system\s*prompt|developer\s*mode|dan\b|ignore (all|previous) (rules|instructions)",
    r"\bвзлом|хаки?|кейген|кряк|социальн(ая|ые) инженерия",
    r"\bвирус|вредонос|эксплойт|ботнет|ddos\b",
    r"\bоружи|взрывчат|бомб|наркот|порно|эротик|18\+",
    r"\bпаспорт|снилс|инн\b.*(сгенер|поддел)",
    r"\bобой(д|ти)\b.*антиплаг|антиплагиат|antiplagiat",
    r"sql.?инъек|инъекци(я|и) sql",
]

def safety_check(text: str) -> str | None:
    t = (text or "").lower()
    for p in _BANNED_PATTERNS:
        if re.search(p, t, flags=re.IGNORECASE):
            return ("Запрос нарушает правила безопасности "
                    "(взлом/вредонос/обход ограничений/NSFW/личные данные).")
    return None

_ALLOWED_HINT_WORDS = [
    "вкр", "диплом", "курсов", "методолог", "литератур", "литобзор",
    "гипотез", "цель", "задач", "введение", "заключен", "обзор",
    "оформлен", "гост", "таблиц", "рисунк", "антиплаг", "плагиат",
    "презентац", "защиту", "опрос", "анкета", "методы", "статистик",
]

def topical_check(text: str) -> str | None:
    """
    Мягкое тематическое ограничение — используем ТОЛЬКО как подсказку,
    даже если нет активного документа.
    """
    t = (text or "").lower()
    if not any(w in t for w in _ALLOWED_HINT_WORDS):
        return ("Подсказка: сильнее всего отвечаю по содержанию ВКР (главы, таблицы, рисунки, выводы). "
                "Если пришлёте файл диплома — смогу объяснять прямо по вашему тексту.")
    return None

# --------------------- БД / утилиты ---------------------

def _table_has_columns(con, table: str, cols: list[str]) -> bool:
    cur = con.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    have = {row[1] for row in cur.fetchall()}
    return all(c in have for c in cols)


def _current_embedding_profile() -> str:
    """
    Текущий профиль эмбеддингов, который пишем в layout_profile.
    Если в конфиге ничего не задано — используем 'default'.
    """
    return getattr(Cfg, "EMBEDDING_PROFILE", "default")


# --------------------- Таблицы: парсинг/нормализация ---------------------

_TABLE_ANY = re.compile(r"\bтаблиц\w*|\bтабл\.\b|\bтаблица\w*|(?:^|\s)table(s)?\b", re.IGNORECASE)
# Поддерживаем: 2.1, 3, A.1, А.1, П1.2
_TABLE_TITLE_RE = re.compile(r"(?i)\bтаблица\s+(\d+(?:[.,]\d+)*|[a-zа-я]\.?\s*\d+(?:[.,]\d+)*)\b(?:\s*[—\-–]\s*(.+))?")
_COUNT_HINT = re.compile(r"\bсколько\b|how many", re.IGNORECASE)
_WHICH_HINT = re.compile(r"\bкаки(е|х)\b|\bсписок\b|\bперечисл\w*\b|\bназов\w*\b", re.IGNORECASE)

# НОВОЕ: поддерживаем "таблица 6", "табл. 6", "table 6.1" и т.п.
_TABLE_NUM_IN_TEXT_RE = re.compile(
    r"(?i)\b(?:таблиц[а-я]*|табл\.?|table)\s*([A-Za-zА-Яа-я]?\s*\d+(?:[.,]\d+)*)"
)


def _extract_table_nums(text: str) -> list[str]:
    """Достаём все номера таблиц из фразы пользователя."""
    nums: list[str] = []
    for m in _TABLE_NUM_IN_TEXT_RE.finditer(text or ""):
        raw = (m.group(1) or "").strip()
        #  " 4 , 1 " -> "4.1"
        norm = raw.replace(" ", "").replace(",", ".")
        if norm:
            nums.append(norm)
    return nums

def _is_pure_table_request(text: str) -> bool:
    """
    Эвристика: запрос ТОЛЬКО про конкретные таблицы
    (например: "опиши таблицу 4", "что показывает таблица 2.3"),
    без рисунков, разделов и общих вопросов.
    """
    t = (text or "").strip()
    if not t:
        return False

    # нет слова "таблица" — точно не наш случай
    if not _TABLE_ANY.search(t):
        return False

    # нет номера после "таблицы" — тоже не чистый запрос
    if not _TABLE_NUM_IN_TEXT_RE.search(t):
        return False

    # если одновременно спрашивают про рисунки или разделы — это уже смешанный вопрос
    if FIG_NUM_RE.search(t) or _SECTION_NUM_RE.search(t):
        return False

    return True

def _plural_tables(n: int) -> str:
    n_abs = abs(n) % 100
    n1 = n_abs % 10
    if 11 <= n_abs <= 14:
        return "таблиц"
    if n1 == 1:
        return "таблица"
    if 2 <= n1 <= 4:
        return "таблицы"
    return "таблиц"

def _strip_table_prefix(s: str) -> str:
    return re.sub(r"^\[\s*таблица\s*\]\s*", "", s or "", flags=re.IGNORECASE)

def _last_segment(name: str) -> str:
    s = (name or "").strip()
    if "/" in s:
        s = s.split("/")[-1].strip()
    s = _strip_table_prefix(s)
    s = re.sub(r"\s*[-–—]\s*", " — ", s)
    s = re.sub(r"\s+", " ", s).strip(" .")
    return s

def _parse_table_title(text: str) -> tuple[str | None, str | None]:
    t = (text or "").strip()
    m = _TABLE_TITLE_RE.search(t)
    if not m:
        return (None, None)
    num = (m.group(1) or "").strip() or None
    title = (m.group(2) or "").strip() or None
    return (num, title)

def _table_num_variants(num: str) -> list[str]:
    """
    Делаем несколько вариантов написания номера таблицы:
    6.1 ↔ 6,1; убираем пробелы.
    Нужны, чтобы находить таблицу даже если в вопросе точка,
    а в тексте запятая (или наоборот).
    """
    raw = (str(num) or "").strip()
    if not raw:
        return []
    base = raw.replace(" ", "")
    dot = base.replace(",", ".")
    comma = base.replace(".", ",")
    variants = {base, dot, comma}
    return [v for v in variants if v]


def _shorten(s: str, limit: int = 120) -> str:
    """
    Аккуратно обрезаем строку до limit символов с троеточием.
    Стараемся резать по границе слова.
    """
    s = (s or "").strip()
    if len(s) <= limit:
        return s

    # пытаемся резать по пробелу, чтобы не рубить слово пополам
    cut = s.rfind(" ", 0, limit)
    # если пробел слишком рано (или вообще не найден) — режем ровно по лимиту
    if cut < max(10, limit // 2):
        cut = limit

    return s[:cut].rstrip(" .,;:–-") + "…"




# -------- Таблицы: подсчёт и список (совместимо со старыми БД) --------

def _distinct_table_basenames(uid: int, doc_id: int) -> list[str]:
    """
    Собираем «базовые» имена таблиц (section_path без хвоста ' [row …]').
    Работает и с новыми индексами (table_row) и со старыми.

    НОВОЕ:
    - если есть колонка attrs и парсер проставил is_table=true,
      считаем «живыми» только такие таблицы;
    - старые документы без attrs / без is_table продолжают учитываться как раньше.
    """
    con = get_conn()
    cur = con.cursor()

    has_et    = _table_has_columns(con, "chunks", ["element_type"])
    has_attrs = _table_has_columns(con, "chunks", ["attrs"])

    # сначала пробуем опереться на типы
    if has_et:
        if has_attrs:
            # только "настоящие" DOCX-таблицы (или старые записи без attrs)
            cur.execute(
                """
                SELECT DISTINCT
                    CASE
                        WHEN instr(section_path, ' [row ')>0
                            THEN substr(section_path, 1, instr(section_path,' [row ')-1)
                        ELSE section_path
                    END AS base_name
                FROM chunks
                WHERE doc_id=? AND owner_id=?
                  AND element_type IN ('table','table_row')
                  AND (attrs IS NULL OR attrs LIKE '%"is_table": true%')
                """,
                (doc_id, uid),
            )
        else:
            # очень старый индекс без attrs — поведение как раньше
            cur.execute(
                """
                SELECT DISTINCT
                    CASE
                        WHEN instr(section_path, ' [row ')>0
                            THEN substr(section_path, 1, instr(section_path,' [row ')-1)
                        ELSE section_path
                    END AS base_name
                FROM chunks
                WHERE doc_id=? AND owner_id=? AND element_type IN ('table','table_row')
                """,
                (doc_id, uid),
            )
    else:
        # очень старый индекс — эвристика по тексту/section_path
        cur.execute(
            """
            SELECT DISTINCT
                CASE
                    WHEN instr(section_path, ' [row ')>0
                        THEN substr(section_path, 1, instr(section_path,' [row ')-1)
                    ELSE section_path
                END AS base_name
            FROM chunks
            WHERE doc_id=? AND owner_id=? AND (
                  lower(section_path) LIKE '%таблица%'
               OR lower(text)        LIKE '%таблица%'
               OR section_path LIKE 'Таблица %' COLLATE NOCASE
               OR text        LIKE '[Таблица]%' COLLATE NOCASE
               OR lower(section_path) LIKE '%table %'
               OR lower(text)        LIKE '%table %'
            )
            """,
            (doc_id, uid),
        )

    base_items = [r["base_name"] for r in cur.fetchall() if r["base_name"]]
    con.close()
    base_items = sorted(set(base_items), key=lambda s: s.lower())
    return base_items

def _count_tables(uid: int, doc_id: int) -> int:
    return len(_distinct_table_basenames(uid, doc_id))

def _compose_display_from_attrs(attrs_json: str | None, base: str, first_row_text: str | None) -> str:
    """
    Правила отображения:
      1) есть caption_num → 'Таблица N — tail/header/firstrow' (всегда с описанием, если есть).
      2) нет номера → показываем только описание: caption_tail/ header_preview/ first_row.
      3) фолбэк: парсим номер и хвост из base ('Таблица N — ...') и показываем с номером.
      4) если ничего не вышло — берём короткий base без служебных слов.
    """
    num = None
    tail = None
    header_preview = None
    if attrs_json:
        try:
            a = json.loads(attrs_json or "{}")
            num = a.get("caption_num") or a.get("label")
            tail = a.get("caption_tail") or a.get("title")
            header_preview = a.get("header_preview")
        except Exception:
            pass

    if num:
        num = str(num).replace(",", ".").strip()
        tail_like = (tail or header_preview or first_row_text or "").strip()
        return f"Таблица {num}" + (f" — {_shorten(tail_like, 160)}" if tail_like else "")

    # без номера в attrs — пробуем распарсить из base и показать С номером
    num_b, title_b = _parse_table_title(_last_segment(base))
    if num_b:
        text_tail = title_b or first_row_text or header_preview
        return f"Таблица {num_b}" + (f" — {_shorten(text_tail, 160)}" if text_tail else "")

    # без номера — только описание
    if tail:
        return _shorten(str(tail), 160)
    if header_preview:
        return _shorten(str(header_preview), 160)
    if first_row_text:
        return _shorten(first_row_text, 160)

    s = _last_segment(base)
    s = re.sub(r"(?i)^\s*таблица\s+\d+(?:\.\d+)*\s*", "", s).strip(" —–-")
    return _shorten(s or "Таблица", 160)


# ------------------------------ Источники ------------------------------

_SOURCES_HINT = re.compile(
    r"\b(источник(?:и|ов)?|список\s+литературы|список\s+источников|библиограф\w*|references?|bibliograph\w*)\b",
    re.IGNORECASE
)
_REF_LINE_RE = re.compile(r"^\s*(?:\[\d+\]|\d+[.)])\s+.+", re.MULTILINE)

def _count_sources(uid: int, doc_id: int) -> int:
    """
    Подсчёт источников:
      1) если в БД есть element_type='reference' — используем его;
      2) иначе собираем любые непустые абзацы внутри секций «источники/литература/…»
         (без требования, чтобы строка начиналась с номера).
    """
    con = get_conn()
    cur = con.cursor()
    has_type = _table_has_columns(con, "chunks", ["element_type"])

    total = 0
    if has_type:
        cur.execute(
            "SELECT COUNT(*) AS c FROM chunks WHERE owner_id=? AND doc_id=? AND element_type='reference'",
            (uid, doc_id),
        )
        row = cur.fetchone()
        total = int(row["c"] or 0)

    if total == 0:
        items = set()
        cur.execute(
            """
            SELECT element_type, section_path, text
            FROM chunks
            WHERE owner_id=? AND doc_id=?
            ORDER BY page ASC, id ASC
            """,
            (uid, doc_id),
        )
        raw_rows = cur.fetchall()
        for r in raw_rows:
            sec = (r["section_path"] or "").lower()
            if not any(k in sec for k in ("источник", "литератур", "библиограф", "reference", "bibliograph")):
                continue
            et = (r["element_type"] or "").lower()
            if et in ("heading", "table", "figure", "table_row"):
                continue
            t = (r["text"] or "").strip()
            if not t:
                continue
            k = re.sub(r"\s+", " ", t).strip().rstrip(".").lower()
            if len(k) >= 5:
                items.add(k)
        total = len(items)

    con.close()
    return total


# --------- быстрый ответ: наличие практической части ---------
_PRACTICAL_Q = re.compile(r"(есть ли|наличие|присутствует ли|имеется ли).{0,40}практическ", re.IGNORECASE)

def _has_practical_part(uid: int, doc_id: int) -> bool:
    con = get_conn()
    cur = con.cursor()
    cur.execute(
        """
        SELECT 1
        FROM chunks
        WHERE owner_id=? AND doc_id=? AND (
            lower(section_path) LIKE '%практическ%' OR
            lower(text)         LIKE '%практическ%'
        )
        LIMIT 1
        """,
        (uid, doc_id),
    )
    row = cur.fetchone()
    con.close()
    return row is not None


# ------------- ГОСТ-интент и проверка -------------

_GOST_HINT = re.compile(r"\b(гост|оформлени|шрифт|межстроч|кегл|выравнивани|поля|оформить)\w*\b", re.IGNORECASE)

async def _maybe_run_gost(m: types.Message, uid: int, doc_id: int, text: str) -> bool:
    """Если похоже, что просят проверку оформления — запускаем валидатор ГОСТ. Возвращаем True, если ответили."""
    if not validate_gost or not render_report:
        return False
    if not _GOST_HINT.search(text or ""):
        return False

    con = get_conn()
    cur = con.cursor()
    cur.execute("SELECT path FROM documents WHERE id=? AND owner_id=?", (doc_id, uid))
    row = cur.fetchone()
    con.close()
    if not row:
        return False

    path = row["path"]
    try:
        sections = _parse_by_ext(path)
    except Exception:
        return False

    report = validate_gost(sections)
    text_rep = render_report(report, max_issues=25)
    await _send(m, text_rep)
    return True

def _cap(s: str, limit: int = 950) -> str:
    """Обрезаем caption для media (у TG лимит ~1024 символа)."""
    s = (s or "").strip()
    if len(s) <= limit:
        return s
    return s[:limit - 1].rstrip() + "…"

def _safe_fs_input(path: str) -> FSInputFile | None:
    try:
        p = os.path.abspath(path or "")
        if not os.path.isfile(p):
            return None
        return FSInputFile(p)
    except Exception:
        return None

def _media_groups_from_cards(cards: list[dict], *, per_group: int = 10, per_figure: int = 4) -> list[list[InputMediaPhoto]]:
    """
    Собираем InputMediaPhoto из карточек describe_figures_by_numbers.
    Не больше FIG_MEDIA_LIMIT всего, per_group — ограничение Telegram (10).
    """
    media: list[InputMediaPhoto] = []
    total = 0
    for c in cards or []:
        disp = c.get("display") or f"Рисунок {c.get('num') or ''}".strip()
        imgs = (c.get("images") or [])[:per_figure]
        if not imgs:
            continue
        cap = _cap(disp)
        first = True
        for img in imgs:
            if total >= FIG_MEDIA_LIMIT:
                break
            fh = _safe_fs_input(img)
            if not fh:
                continue
            # caption ставим только на первое фото рисунка (TG best-practice)
            media.append(InputMediaPhoto(media=fh, caption=cap if first else None))
            total += 1
            first = False
        if total >= FIG_MEDIA_LIMIT:
            break

    # разбиваем по 10 элементов на группу
    groups: list[list[InputMediaPhoto]] = []
    for i in range(0, len(media), per_group):
        groups.append(media[i:i + per_group])
    return groups

async def _send_media_from_cards(m: types.Message, cards: list[dict]) -> bool:
    """
    Пробуем отправить медиагруппы по карточкам. Возвращает True, если что-то отправили.
    """
    groups = _media_groups_from_cards(cards)
    sent_any = False
    for g in groups:
        if not g:
            continue
        try:
            await m.answer_media_group(g)
            sent_any = True
        except TelegramBadRequest:
            # если медиа-группа не зашла (например, одно фото) — отправим поштучно
            for item in g:
                try:
                    await m.answer_photo(item.media, caption=item.caption)
                    sent_any = True
                except Exception:
                    pass
        except Exception:
            pass
    return sent_any

# ------------------------------ helpers ------------------------------

def _parse_by_ext(path: str) -> list[dict]:
    fname = (os.path.basename(path) or "").lower()
    if fname.endswith(".docx"):
        return parse_docx(path)
    if fname.endswith(".doc"):
        return parse_doc(path)
    raise RuntimeError("Поддерживаю только .doc и .docx.")

def _first_chunks_context(owner_id: int, doc_id: int, n: int = 10, max_chars: int = 6000) -> str:
    con = get_conn()
    cur = con.cursor()
    cur.execute(
        "SELECT page, section_path, text FROM chunks "
        "WHERE owner_id=? AND doc_id=? "
        "ORDER BY page ASC, id ASC LIMIT ?",
        (owner_id, doc_id, n)
    )
    rows = cur.fetchall()
    con.close()
    if not rows:
        return ""
    parts, total = [], 0
    for r in rows:
        block = f"{(r['text'] or '').strip()}"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)


def _ooxml_get_index(doc_id: int) -> dict | None:
    """Возвращает OOXML-индекс из памяти или с диска. Сначала runtime/indexes/<doc_id>.json,
    затем фолбэк — ищем json с совпадающим meta.file (путь к исходному файлу)."""
    idx = OOXML_INDEX.get(doc_id)
    if idx:
        return idx

    p = os.path.join("runtime", "indexes", f"{doc_id}.json")
    if os.path.isfile(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                idx = json.load(f)
            OOXML_INDEX[doc_id] = idx
            return idx
        except Exception:
            pass

    # фолбэк: подобрать индекс по совпадению пути файла
    try:
        con = get_conn()
        cur = con.cursor()
        cur.execute("SELECT path FROM documents WHERE id=?", (doc_id,))
        row = cur.fetchone()
        con.close()
        doc_path = os.path.abspath(row["path"]) if row else None
        if doc_path:
            idx_dir = os.path.join("runtime", "indexes")
            if os.path.isdir(idx_dir):
                for name in os.listdir(idx_dir):
                    if not name.endswith(".json"):
                        continue
                    try:
                        with open(os.path.join(idx_dir, name), "r", encoding="utf-8") as f:
                            cand = json.load(f)
                        if (cand.get("meta") or {}).get("file") == doc_path:
                            OOXML_INDEX[doc_id] = cand
                            return cand
                    except Exception:
                        continue
    except Exception:
        pass
    return None


# Небольшой хелпер: понять, что подпись больше похожа на «таблицу», а не на обычный рисунок
_TABLE_CAPTION_HINT_RE = re.compile(
    r"(?i)\bтаблиц\w*|\bтабл\.\b|\bтаблица\w*|(?:^|\s)table(s)?\b"
)

def _looks_like_table_caption(rec: dict) -> bool:
    """
    Эвристика: запись похожа на таблицу, если:
      — явно указаны тип/вид 'table/таблица' ИЛИ
      — в caption/title/name есть слово «таблица»/«table».
    Используется только для отбора кандидатов в _ooxml_find_table_image.
    """
    if not isinstance(rec, dict):
        return False

    kind = str(rec.get("kind") or rec.get("type") or "").lower()
    if "table" in kind or "таблиц" in kind:
        return True

    cap = (rec.get("caption") or rec.get("title") or rec.get("name") or "").strip()
    if not cap:
        return False

    return bool(_TABLE_CAPTION_HINT_RE.search(cap))


def _ooxml_find_figure_by_label(idx: dict, num_str: str) -> dict | None:
    """
    Ищем запись о рисунке по номеру вида '2.3' из подписи.
    Сравниваем именно текст в caption/title, а не только целую часть.
    """
    target = _num_norm_fig(num_str)
    if not target:
        return None
    figs = (idx or {}).get("figures") or []
    for f in figs:
        cap = (f.get("caption") or f.get("title") or "").strip()
        m = _FIG_TITLE_RE.search(cap)
        if not m:
            continue
        cap_num = _num_norm_fig(m.group(2))
        if cap_num == target:
            return f
    return None


# NEW: поиск «таблицы-картинки» по подписи "Таблица N ..."
# NEW: поиск «таблицы-картинки» по подписи "Таблица N ..."
def _ooxml_find_table_image(idx: dict, num: str) -> dict | None:
    """
    Ищем запись, соответствующую таблице {num}, которая вставлена картинкой.
    Смотрим и в figures, и в tables (если там есть image_path).

    ВАЖНО:
      — из коллекции tables берём всё;
      — из figures берём ТОЛЬКО записи, подпись которых выглядит как таблица
        (_looks_like_table_caption), чтобы не перепутать с «Рисунок 6».

    Приоритет:
      1) точное совпадение по числовым полям (caption_num/label/num/n)
         ТОЛЬКО среди «табличных» кандидатов;
      2) совпадение по подписи 'Таблица {num} ...'.
    """
    if not idx:
        return None

    target = str(num).replace(" ", "").replace(",", ".")

    def _iter_candidates():
        """
        Даём (kind, rec):
          kind == 'tables'  → всегда считаем таблицей;
          kind == 'figures' → используем только если подпись похожа на таблицу.
        """
        for kind in ("figures", "tables"):
            coll = (idx.get(kind) or [])
            if isinstance(coll, dict):
                coll = list(coll.values())
            if not isinstance(coll, list):
                continue
            for rec in coll:
                if not isinstance(rec, dict):
                    continue
                # всё из tables берём без условий,
                # а из figures — только «табличные» подписи
                if kind == "figures" and not _looks_like_table_caption(rec):
                    continue
                yield kind, rec

    # 1) Сначала пробуем по числовым полям (caption_num/label/num/n)
    for _kind, rec in _iter_candidates():
        for fld in ("caption_num", "label", "num", "n"):
            raw = rec.get(fld)
            if not raw:
                continue
            cand = str(raw).replace(" ", "").replace(",", ".")
            if cand == target:
                return rec

    # 2) Фолбэк — по тексту подписи
    for _kind, rec in _iter_candidates():
        cap = (rec.get("caption") or rec.get("title") or rec.get("name") or "").strip()
        if not cap:
            continue
        # "Таблица 6", "таблица 6 – ...", "Таблица 6. ..."
        m = re.search(r"(?i)\bтаблиц\w*\s+(\d+(?:[.,]\d+)*)", cap)
        if not m:
            continue
        cap_num = (m.group(1) or "").replace(" ", "").replace(",", ".")
        if cap_num == target:
            return rec

    return None


# NEW: OCR-фолбэк для таблиц, вставленных картинкой
def _ocr_table_block_from_image(uid: int, doc_id: int, num: str) -> str | None:
    key = (doc_id, str(num))
    cached = _OCR_TABLE_CACHE.get(key)
    if cached is not None:
        logging.info("TAB[img] cache hit for doc=%s, table=%s", doc_id, num)
        return cached

    # 1) сначала пробуем OOXML-индекс
    idx = _ooxml_get_index(doc_id)
    if not idx:
        logging.info("TAB[img] no OOXML index for doc=%s", doc_id)
    rec = _ooxml_find_table_image(idx, num) if idx else None

    img_path: str | None = None
    if rec:
        # 1) стандартные поля
        img_path = rec.get("image_path") or rec.get("image")

        # 2) иногда ooxml_lite кладёт список картинок
        if not img_path:
            imgs = rec.get("images") or rec.get("imgs") or []
            if isinstance(imgs, (list, tuple)) and imgs:
                img_path = imgs[0]

        logging.info(
            "TAB[img] OOXML candidate for table %s: image_path=%r",
            num,
            img_path,
        )


    # 2) если из OOXML ничего не получилось — пробуем достать картинку через retrieve(...)
    if not img_path:
        try:
            logging.info("TAB[img] fallback retrieve() for table %s", num)
            hits = retrieve(uid, doc_id, f"Таблица {num}", top_k=6)
        except Exception as e:
            logging.exception("TAB[img] retrieve() failed for table %s: %s", num, e)
            hits = []

        if hits:
            paths = _pick_images_from_hits(hits, limit=1)
            logging.info("TAB[img] retrieve() returned image paths=%r", paths)
            if paths:
                img_path = paths[0]

    if not img_path or not os.path.isfile(img_path):
        logging.info(
            "TAB[img] no image found on disk for table %s (img_path=%r)",
            num,
            img_path,
        )
        return None

    # общий список всех найденных пар значений
    all_pairs: list[dict] = []
    # дополнительные текстовые куски с изображения (в т.ч. «Примечание»)
    extra_text_parts: list[str] = []

    def _add_pairs(pairs: list[dict] | None) -> None:
        """Аккуратно добавляем пары без грубых дублей по (label, value, unit)."""
        nonlocal all_pairs
        if not pairs:
            return
        seen = {(str(p.get("label") or "").strip(),
                 str(p.get("value") or "").strip(),
                 str(p.get("unit") or "").strip())
                for p in all_pairs}
        for p in pairs:
            key_p = (
                str(p.get("label") or "").strip(),
                str(p.get("value") or "").strip(),
                str(p.get("unit") or "").strip(),
            )
            if key_p in seen:
                continue
            seen.add(key_p)
            all_pairs.append(p)

    # 3.a) спец-функция по таблицам-картинкам
    if vision_extract_table_values is not None:
        try:
            pairs1 = vision_extract_table_values(img_path, lang="ru") or []
            _add_pairs(pairs1)
        except Exception as e:
            logging.exception(
                "ocr_table_block_from_image: vision_extract_table_values failed for %s (table %s): %s",
                img_path,
                num,
                e,
            )

    # 3.b) общий vision-анализ: и пары, и текст (часто содержит «Примечание»)
    if va_analyze_figure is not None:
        try:
            try:
                res = va_analyze_figure(
                    img_path,
                    caption_hint=f"Таблица {num}",
                    lang="ru",
                )
            except TypeError:
                res = va_analyze_figure(img_path, lang="ru")  # type: ignore

            if isinstance(res, dict):
                pairs2 = res.get("data") or []
                _add_pairs(pairs2)
                txt2 = (res.get("text") or "").strip()
                if txt2:
                    extra_text_parts.append(txt2)
            else:
                txt2 = (str(res) or "").strip()
                if txt2:
                    extra_text_parts.append(txt2)
        except Exception as e:
            logging.exception(
                "ocr_table_block_from_image: va_analyze_figure failed for %s (table %s): %s",
                img_path,
                num,
                e,
            )

    # 3.c) общий extractor пар label/value
    if vision_extract_values is not None:
        try:
            pairs3 = vision_extract_values(img_path, lang="ru") or []
            _add_pairs(pairs3)
        except Exception as e:
            logging.exception(
                "ocr_table_block_from_image: vision_extract_values failed for %s (table %s): %s",
                img_path,
                num,
                e,
            )



    values_block = _pairs_to_bullets(all_pairs) if all_pairs else ""
    values_block = (values_block or "").strip()
    extra_text = "\n".join(
        t.strip() for t in extra_text_parts if t and t.strip()
    )

    if not values_block and not extra_text:
        return None

    lines: list[str] = [f"Таблица {num} (распознана по изображению):"]
    if values_block:
        lines.append(values_block)
    if extra_text:
        lines.append("")  # пустая строка
        lines.append("[Текст с изображения]")
        lines.append(extra_text)

    out = "\n".join(lines).strip()
    _OCR_TABLE_CACHE[key] = out
    return out


# ---------- verbatim fallback по цитате (шинглы + LIKE/NOCASE) ----------

def _normalize_for_like(s: str) -> str:
    s = (s or "")
    s = s.replace("\u00A0", " ")  # NBSP -> пробел
    s = s.replace("«", '"').replace("»", '"').replace("“", '"').replace("”", '"')
    s = s.replace("’", "'").replace("‘", "'")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _make_shingles(s: str, min_len: int = 30, max_len: int = 90, step: int = 25) -> list[str]:
    s = _normalize_for_like(s)
    if not s:
        return []
    if len(s) <= max_len:
        return [s]
    out = []
    i = 0
    while i < len(s):
        chunk = s[i:i + max_len]
        if len(chunk) >= min_len:
            out.append(chunk)
        i += step
    return out[:6]

def verbatim_find(owner_id: int, doc_id: int, q_text: str, max_hits: int = 3) -> list[dict]:
    shingles = _make_shingles(q_text)
    if not shingles:
        return []
    con = get_conn()
    cur = con.cursor()
    hits: list[dict] = []
    for sh in shingles:
        pattern = f"%{sh}%"
        cur.execute(
            """
            SELECT page, section_path, text FROM chunks
            WHERE owner_id=? AND doc_id=? AND
                  replace(text, char(160), ' ') LIKE ? COLLATE NOCASE
            ORDER BY page ASC, id ASC
            LIMIT ?
            """,
            (owner_id, doc_id, pattern, max_hits - len(hits)),
        )
        for r in cur.fetchall():
            t = (r["text"] or "")
            t_norm = _normalize_for_like(t)
            pos = t_norm.lower().find(_normalize_for_like(sh).lower())
            if pos >= 0:
                s = max(pos - 120, 0)
                e = min(pos + len(sh) + 120, len(t_norm))
                hits.append({
                    "page": r["page"],
                    "section_path": r["section_path"],
                    "snippet": t_norm[s:e].strip(),
                })
            if len(hits) >= max_hits:
                con.close()
                return hits
    con.close()
    return hits


# ------------------------------ /start ------------------------------

@dp.message(Command("start"))
async def start(m: types.Message):
    ensure_user(str(m.from_user.id))
    await _send(m,
        "Привет! Я репетитор по твоей ВКР. Пришли файл ВКР — я проиндексирую и буду объяснять содержание: главы простым языком, смысл таблиц/рисунков, конспекты к защите. Можешь прикрепить вопрос к файлу или написать его отдельным сообщением."
    )


# ------------------------------ /diag ------------------------------

@dp.message(Command("diag"))
async def cmd_diag(m: types.Message):
    uid = ensure_user(str(m.from_user.id))
    doc_id = ACTIVE_DOC.get(uid) or get_user_active_doc(uid)
    if not doc_id:
        await _send(m, "Активного документа нет. Пришлите файл ВКР сначала.")
        return

    # базовые метрики из БД
    con = get_conn()
    cur = con.cursor()
    cur.execute("SELECT path FROM documents WHERE id=? AND owner_id=?", (doc_id, uid))
    row = cur.fetchone()
    path = row["path"] if row else "—"

    cur.execute("SELECT COUNT(*) AS c FROM chunks WHERE owner_id=? AND doc_id=?", (uid, doc_id))
    chunks_cnt = int(cur.fetchone()["c"])

    con.close()

    tables_cnt = _count_tables(uid, doc_id)
    figures_cnt = _list_figures_db(uid, doc_id, limit=999999)["count"]
    # NEW: если в БД 0 — возьмём число рисунков из OOXML-индекса
    if figures_cnt == 0:
        idx_oox = _ooxml_get_index(doc_id)
        if idx_oox:
            figures_cnt = len(idx_oox.get("figures", []))
    sources_cnt = _count_sources(uid, doc_id)

    indexer_ver = get_document_indexer_version(doc_id) or 0

    txt = (
        f"Диагностика документа #{doc_id}\n"
        f"— Путь: {path}\n"
        f"— Чанков: {chunks_cnt}\n"
        f"— Таблиц: {tables_cnt}\n"
        f"— Рисунков: {figures_cnt}\n"
        f"— Источников: {sources_cnt}\n"
        f"— Версия индексатора: {indexer_ver} (текущая {CURRENT_INDEXER_VERSION})\n"
    )
    await _send(m, txt)


# ------------------------------ /reindex ------------------------------

@dp.message(Command("reindex"))
async def cmd_reindex(m: types.Message):
    uid = ensure_user(str(m.from_user.id))
    doc_id = ACTIVE_DOC.get(uid) or get_user_active_doc(uid)
    if not doc_id:
        await _send(m, "Активного документа нет. Пришлите файл сначала.")
        return

    con = get_conn()
    cur = con.cursor()
    cur.execute("SELECT path FROM documents WHERE id=? AND owner_id=?", (doc_id, uid))
    row = cur.fetchone()
    con.close()

    if not row:
        await _send(m, "Не смог найти путь к файлу. Загрузите документ заново.")
        return

    path = row["path"]
    try:
        sections = _parse_by_ext(path)
        # обогащаем секции перед индексом
        sections = enrich_sections(sections, doc_kind=os.path.splitext(path)[1].lower().strip("."))
        delete_document_chunks(doc_id, uid)
        index_document(uid, doc_id, sections)
        invalidate_cache(uid, doc_id)
        set_document_indexer_version(doc_id, CURRENT_INDEXER_VERSION)
        update_document_meta(doc_id, layout_profile=_current_embedding_profile())
        await _send(m, f"Документ #{doc_id} переиндексирован.")
    except Exception as e:
        logging.exception("reindex failed: %s", e)
        await _send(m, f"Не удалось переиндексировать документ: {e}")



# ---------- Рисунки: вспомогательные функции (локальные, без зависимостей от retrieval.py) ----------

_FIG_TITLE_RE = re.compile(
    r"(?i)\b(рис(?:\.|унок)?|схем(?:а|ы)?|картин(?:ка|ки)?|figure|fig\.?|picture|pic\.?)"
    r"\s*(?:№\s*)?(\d+(?:[.,]\d+)*)\b(?:\s*[—\-–:\u2013\u2014]\s*(.+))?"
)

# Включать извлечение числовых значений с картинок по умолчанию
FIG_VALUES_DEFAULT: bool = getattr(Cfg, "FIG_VALUES_DEFAULT", True)

def _compose_figure_display(attrs_json: str | None, section_path: str, title_text: str | None) -> str:
    """Делаем красивый заголовок рисунка по приоритетам."""
    num = None
    tail = None
    if attrs_json:
        try:
            a = json.loads(attrs_json or "{}")
            num  = str(a.get("caption_num") or a.get("label") or "").strip()
            tail = str(a.get("caption_tail") or a.get("title") or "").strip()
        except Exception:
            pass

    if not num or not num.strip():
        cand = title_text or section_path or ""
        m = _FIG_TITLE_RE.search(cand)
        if m:
            num = (m.group(2) or "").replace(",", ".").strip()
            if not tail:
                tail = (m.group(3) or "").strip()

    if num:
        return f"Рисунок {num}" + (f" — {_shorten(tail, 160)}" if tail else "")
    base = title_text or _last_segment(section_path or "")
    base = re.sub(
    r"(?i)^\s*(рис(?:\.|унок)?|схем(?:а|ы)?|картин(?:ка|ки)?|figure|fig\.?|picture|pic\.?)\s*",
        "", base
    ).strip(" —–-")
    return _shorten(base or "Рисунок", 160)

# ---------- NEW: точные значения из DOCX-графиков (chart_data) ----------

def _fetch_figure_row_by_num(uid: int, doc_id: int, num: str):
    """
    Возвращает строку chunks для рисунка с указанным номером (если найдена),
    ориентируясь ТОЛЬКО на caption_num (номер рисунка в подписи).
    По label больше не ищем, чтобы не путать номер рисунка с подписями серий/категорий.
    """
    con = get_conn()
    cur = con.cursor()
    like1 = f'%\"caption_num\": \"{num}\"%'
    row = None

    # 1) по номеру в attrs (caption_num)
    try:
        cur.execute(
            """
            SELECT id, page, section_path, attrs, text
            FROM chunks
            WHERE owner_id=? AND doc_id=? AND element_type='figure'
              AND attrs LIKE ?
            ORDER BY id ASC LIMIT 1
            """,
            (uid, doc_id, like1),
        )
        row = cur.fetchone()
    except Exception:
        row = None

    # 2) фолбэк — по section_path
    if not row:
        try:
            cur.execute(
                """
                SELECT id, page, section_path, attrs, text
                FROM chunks
                WHERE owner_id=? AND doc_id=? AND element_type='figure'
                  AND section_path LIKE ? COLLATE NOCASE
                ORDER BY id ASC LIMIT 1
                """,
                (uid, doc_id, f'%Рисунок {num}%'),
            )
            row = cur.fetchone()
        except Exception:
            row = None

    con.close()
    return row

def _figure_following_paragraphs(
    uid: int,
    doc_id: int,
    fig_row,
    max_paragraphs: int = 3,
    max_chars: int = 1500,
) -> list[str]:
    """
    Берём 2–3 абзаца текста ПОСЛЕ рисунка в том же section_path.
    Останавливаемся, как только встретили следующий heading/table/figure.
    """
    if not fig_row:
        return []

    base_id = fig_row["id"]
    sec = fig_row["section_path"] or ""

    con = get_conn()
    cur = con.cursor()

    has_et = _table_has_columns(con, "chunks", ["element_type"])

    if has_et:
        cur.execute(
            """
            SELECT text, element_type
            FROM chunks
            WHERE owner_id=? AND doc_id=? AND id>? AND section_path=?
            ORDER BY id ASC
            LIMIT 20
            """,
            (uid, doc_id, base_id, sec),
        )
    else:
        cur.execute(
            """
            SELECT text, NULL AS element_type
            FROM chunks
            WHERE owner_id=? AND doc_id=? AND id>? AND section_path=?
            ORDER BY id ASC
            LIMIT 20
            """,
            (uid, doc_id, base_id, sec),
        )

    rows = cur.fetchall() or []
    con.close()

    paras: list[str] = []
    total = 0

    for r in rows:
        et = (r["element_type"] or "").lower() if "element_type" in r.keys() else ""
        if et in ("heading", "table", "figure", "table_row"):
            # дошли до следующего структурного блока — дальше уже не «про этот рисунок»
            break
        t = (r["text"] or "").strip()
        if not t:
            continue

        paras.append(t)
        total += len(t)
        if len(paras) >= max_paragraphs or total >= max_chars:
            break

    return paras


def _parse_chart_data(attrs_json: str | None) -> tuple[list | None, str | None, dict]:
    """
    Извлекает данные графика из разных возможных схем attrs.
    Возвращает (data_rows, chart_type, attrs_dict), где data_rows — список словарей
    вида {"label": ..., "value": ..., "unit": ...}.
    """
    try:
        a = json.loads(attrs_json or "{}")

        # самые частые варианты размещения данных
        raw = (a.get("chart_data")
            or (a.get("chart") or {}).get("data")
            or a.get("data")
            or a.get("series"))
        ctype = (a.get("chart_type")
                or (a.get("chart") or {}).get("type")
                or a.get("type"))


        # Уже нормализованный список [{label, value, unit?}]
        if isinstance(raw, list) and raw:
            return raw, ctype, a

        # Распространённая форма: {"categories":[...], "series":[{"name":..., "values":[...], "unit":"%"}]}
        if isinstance(raw, dict) and raw.get("categories") and raw.get("series"):
            cats = list(raw.get("categories") or [])
            s0   = (raw.get("series") or [{}])[0] or {}
            vals = list(s0.get("values") or s0.get("data") or [])
            unit = s0.get("unit")
            rows = []
            for i in range(min(len(cats), len(vals))):
                rows.append({
                    "label": str(cats[i]),
                    "value": vals[i],
                    "unit": unit
                })
            if rows:
                return rows, (ctype or s0.get("type") or "chart"), a
    except Exception:
        pass
    return None, None, {}

def _extract_raw_values_from_attrs(attrs_json: str | None) -> dict | None:
    """
    Пытаемся вытащить структурированные данные диаграммы ИЗ OOXML-атрибутов.

    Возвращает dict вида:
    {
      "categories": [...],
      "series": [
        {"name": "Низкий уровень", "unit": "%", "values": [65, 60, ...]},
        ...
      ]
    }
    или None, если такой структуры нет.

    ВАЖНО:
    - поддерживаем формат из нового ooxml_lite (chart.cats + chart.series);
    - если chart_data — список строк [{label, value, unit, series_name}],
      собираем полноценные серии, а не схлопываем всё в одну.
    """
    if not attrs_json:
        return None
    try:
        a = json.loads(attrs_json or "{}")
    except Exception:
        return None

    # 1) Типичный формат OOXML-индекса: {"chart_data": {"categories": [...], "series": [...]}},
    #    либо {"chart": {...}}, либо {"data": {...}}.
    for key in ("chart_data", "chart", "data"):
        raw = a.get(key)
        if isinstance(raw, dict) and raw.get("categories") and raw.get("series"):
            cats = list(raw.get("categories") or [])
            series_out: list[dict] = []
            for s in (raw.get("series") or []):
                if not isinstance(s, dict):
                    continue
                name = s.get("name")
                unit = s.get("unit")
                vals = list(s.get("values") or s.get("data") or [])
                series_out.append({
                    "name": name,
                    "unit": unit,
                    "values": vals,
                })
            if cats and series_out:
                return {"categories": cats, "series": series_out}

    # 1а) Новый формат из ooxml_lite: "chart": {"cats": [...], "series": [...]}
    chart = a.get("chart")
    if isinstance(chart, dict) and chart.get("series"):
        cats = chart.get("cats") or chart.get("categories") or []
        cats = [str(c) for c in cats]
        series_out: list[dict] = []
        for s in (chart.get("series") or []):
            if not isinstance(s, dict):
                continue
            name = s.get("name")
            unit = s.get("unit")
            # в ooxml_lite мы уже кладём числовые значения в s["values"]
            vals = list(s.get("values") or s.get("data") or s.get("vals") or [])
            series_out.append({
                "name": name,
                "unit": unit,
                "values": vals,
            })
        if cats and series_out:
            return {"categories": cats, "series": series_out}

    # 2) chart_data как СПИСОК строк [{label, value, unit, series_name, ...}]
    raw_rows = a.get("chart_data")
    if isinstance(raw_rows, list) and raw_rows:
        # 2.1. Категории — уникальные label'ы в порядке первого появления
        categories: list[str] = []
        cat_index: dict[str, int] = {}
        for r in raw_rows:
            label = str(
                r.get("label")
                or r.get("name")
                or r.get("category")
                or ""
            ).strip()
            if label not in cat_index:
                cat_index[label] = len(categories)
                categories.append(label)

        if not categories:
            return None

        # 2.2. Группируем по series_name (если есть), иначе всё в одну серию None
        series_map: dict[str | None, dict] = {}
        any_named_series = False

        for r in raw_rows:
            # имя серии
            raw_sname = (
                r.get("series_name")
                or r.get("series")
                or r.get("name")
            )
            sname = str(raw_sname).strip() if raw_sname is not None else None
            if sname:
                any_named_series = True
                key = sname
            else:
                key = None  # безымянная серия

            # категория → индекс
            label = str(
                r.get("label")
                or r.get("name")
                or r.get("category")
                or ""
            ).strip()
            idx = cat_index.get(label)
            if idx is None:
                continue

            # значение и unit
            v = r.get("value")
            if v is None:
                v = r.get("y") or r.get("x") or r.get("v") or r.get("count")

            unit = r.get("unit")
            unit_str = str(unit).strip() if isinstance(unit, str) else None

            if key not in series_map:
                series_map[key] = {
                    "name": key,
                    "unit": unit_str,
                    "values": [None] * len(categories),
                }

            # если в серии unit ещё не проставлен, а тут есть — запоминаем
            if unit_str and not series_map[key].get("unit"):
                series_map[key]["unit"] = unit_str

            series_map[key]["values"][idx] = v

        series_out = list(series_map.values())

        # если имён серий нет вообще — всё равно вернём одну серию,
        # чтобы _format_exact_values мог её красиво оформить
        if not series_out and raw_rows:
            vals: list = []
            unit: str | None = None
            for r in raw_rows:
                vv = r.get("value")
                if vv is None:
                    vv = r.get("y") or r.get("x") or r.get("v") or r.get("count")
                vals.append(vv)
                if isinstance(r.get("unit"), str):
                    unit = r.get("unit")
            if vals:
                series_out = [{
                    "name": None,
                    "unit": unit,
                    "values": vals,
                }]

        if categories and series_out:
            return {
                "categories": categories,
                "series": series_out,
            }

    return None



def _format_exact_values(raw_values: dict) -> str:
    """
    Форматируем raw_values, аккуратно обращаясь с долями:
    если unit содержит '%' и ВСЕ значения лежат в [0 .. 1.2],
    считаем их долями и выводим как проценты (0.7 → 70 %).
    """
    if not raw_values:
        return ""

    cats = list(raw_values.get("categories") or [])
    series = list(raw_values.get("series") or [])

    if not cats or not series:
        return ""

    lines: list[str] = ["Точные значения (как в документе):", ""]
    n = len(cats)

    for s in series:
        name = (s.get("name") or "").strip()
        unit = (s.get("unit") or "").strip()
        vals = list(s.get("values") or [])

        # эвристика «это доли в процентах»
        numeric_vals: list[float] = []
        for v in vals:
            try:
                numeric_vals.append(float(str(v).replace(",", ".")))
            except Exception:
                # текст/пусто — игнорим для эвристики
                pass

        has_percent_unit = bool(unit) and "%" in unit
        is_share_like = bool(
            has_percent_unit
            and numeric_vals
            and all(0.0 <= x <= 1.2 for x in numeric_vals)
        )

        header = name or "Серия"
        # если единицы НЕ проценты — покажем их в заголовке
        if unit and "%" not in unit:
            header = f"{header} ({unit})"
        lines.append(f"{header}:")

        for i in range(min(n, len(vals))):
            label = str(cats[i]).strip() or str(i + 1)
            raw_v = vals[i]

            v_num: float | None = None
            sval = ""

            if isinstance(raw_v, (int, float, Decimal)):
                v_num = float(raw_v)
            else:
                try:
                    v_num = float(str(raw_v).replace(",", "."))
                except Exception:
                    sval = str(raw_v) if raw_v is not None else ""

            if v_num is not None:
                if is_share_like:
                    v_num *= 100.0  # 0.7 → 70.0

                if abs(v_num - round(v_num)) < 0.05:
                    sval = str(int(round(v_num)))
                else:
                    sval = f"{v_num:.2f}".rstrip("0").rstrip(".")

            # суффикс единиц
            unit_suffix = ""
            if has_percent_unit:
                # хотим «70%», без пробела
                if not sval.endswith("%"):
                    unit_suffix = "%"
            elif unit:
                unit_suffix = f" {unit}"

            line = f"— {label}: {sval}{unit_suffix}".strip()
            if line:
                lines.append(line)

        lines.append("")  # пустая строка между сериями

    return "\n".join(l for l in lines if l.strip())



def _format_chart_values(chart_data: list) -> str:
    """
    Форматируем chart_data БЕЗ нормировки сумм и без «подгонки» к 100%.

    Логика:
      - если unit содержит '%' и все значения в [0..1.2],
        трактуем их как доли (0.8 → 80) и домножаем на 100;
      - дальше просто печатаем: «— label: 80%».
    """
    rows = chart_data or []
    if not rows:
        return "Нет данных для вывода."

    # соберём данные для евристики «это доли в процентах»
    numeric_vals: list[float] = []
    has_percent_unit = False
    for r in rows:
        unit = r.get("unit")
        if isinstance(unit, str) and "%" in unit:
            has_percent_unit = True

        v = r.get("value")
        if v is None:
            v = r.get("y") or r.get("x") or r.get("v") or r.get("count")
        try:
            numeric_vals.append(float(str(v).replace(",", ".")))
        except Exception:
            # если хоть один не приводится к числу — просто не применяем евристику
            pass

    is_share_like = bool(
        has_percent_unit
        and numeric_vals
        and all(0.0 <= x <= 1.2 for x in numeric_vals)
    )

    lines: list[str] = []
    for r in rows:
        label = (str(r.get("label") or r.get("name") or r.get("category") or "")).strip()

        raw_v = r.get("value")
        if raw_v is None:
            raw_v = r.get("y") or r.get("x") or r.get("v") or r.get("count")

        unit = r.get("unit")

        v_num: float | None = None
        sval = ""

        # пробуем привести к числу
        if isinstance(raw_v, (int, float, Decimal)):
            v_num = float(raw_v)
        else:
            try:
                v_num = float(str(raw_v).replace(",", "."))
            except Exception:
                # это строка/текст — оставим как есть
                sval = str(raw_v) if raw_v is not None else ""

        # числовое значение → форматируем
        if v_num is not None:
            if is_share_like:
                v_num *= 100.0  # 0.8 → 80.0

            if abs(v_num - round(v_num)) < 0.05:
                sval = str(int(round(v_num)))
            else:
                sval = f"{v_num:.2f}".rstrip("0").rstrip(".")

        # добавляем единицы измерения
        unit_suffix = ""
        if isinstance(unit, str) and unit.strip():
            u = unit.strip()
            # если это проценты и в строке ещё нет '%', добавим без пробела
            if "%" in u and not sval.endswith("%"):
                unit_suffix = "%"
            else:
                unit_suffix = f" {u}"

        text = (f"— {label}: {sval}{unit_suffix}").strip()
        if text:
            lines.append(text)

    return "\n".join(lines) if lines else "Нет данных для вывода."

# --- небольшая косметика для процентов из OOXML-графиков ---

# двоеточие + пробелы, сразу перед ';' или концом строки
_EMPTY_PERCENT_RE = re.compile(r"(:\s*)(?=;|$)")

def _fill_empty_percents(text: str) -> str:
    """
    'label:' или 'label: ;' → 'label: 0%' перед ';' или концом строки.
    Работает и для кусочков вида '…; 3:' и '…; 3: ;'.
    """
    return _EMPTY_PERCENT_RE.sub(lambda m: m.group(1) + "0%", text)


def _list_figures_db(uid: int, doc_id: int, limit: int = 25) -> dict:
    """Собираем список рисунков из БД (совместимо со старыми индексами)."""
    con = get_conn()
    cur = con.cursor()
    has_type = _table_has_columns(con, "chunks", ["element_type", "attrs"])

    if has_type:
        cur.execute(
            "SELECT DISTINCT section_path, attrs, text FROM chunks "
            "WHERE owner_id=? AND doc_id=? AND element_type='figure' "
            "ORDER BY id ASC",
            (uid, doc_id),
        )
    else:
        # старые индексы — колонки attrs может не быть, не выбираем её
        cur.execute(
            "SELECT DISTINCT section_path, text FROM chunks "
            "WHERE owner_id=? AND doc_id=? AND (text LIKE '[Рисунок]%' OR lower(section_path) LIKE '%рисунок%') "
            "ORDER BY id ASC",
            (uid, doc_id),
        )
    rows = cur.fetchall() or []
    con.close()

    items: list[str] = []
    for r in rows:
        section_path = r["section_path"] or ""
        attrs_json = r["attrs"] if ("attrs" in r.keys()) else None  # в else её просто нет — ок
        txt = r["text"] or None
        disp = _compose_figure_display(attrs_json, section_path, txt)
        items.append(disp)

    seen = set()
    uniq = []
    for it in items:
        k = it.strip().lower()
        if k and k not in seen:
            seen.add(k)
            uniq.append(it)

    total = len(uniq)
    return {
        "count": total,
        "list": uniq[:limit],
        "more": max(0, total - limit),
    }



# -------- Ранний обработчик вопросов вида «рисунок 2.1», «рис. 3», «figure 1.2» --------

FIG_NUM_RE = re.compile(
    r"(?i)\b(?:рис\w*|схем\w*|картин\w*|диаграм\w*|гистограм\w*|diagram|chart|figure|fig\.?|picture|pic\.?)"
    r"\s*(?:№\s*|no\.?\s*|номер\s*)?([A-Za-zА-Яа-я]?\s*[\d.,\s]+(?:\s*(?:и|and)\s*[\d.,\s]+)*)"
)

# новый хинт для режима «извлечь значения»
_VALUES_HINT = re.compile(r"(?i)\b(значени[яе]|цифр[аы]|процент[а-я]*|values?|numbers?)\b")
_SPLIT_FIG_LIST_RE = re.compile(r"\s*(?:,|;|\band\b|и)\s*", re.IGNORECASE)

def _extract_fig_nums(text: str) -> list[str]:
    nums: list[str] = []
    for mm in FIG_NUM_RE.finditer(text or ""):
        seg = (mm.group(1) or "").strip()
        # разделители: запятая, точка с запятой, "и/and"
        parts = _SPLIT_FIG_LIST_RE.split(seg)
        for p in parts:
            p = p.strip()
            if not p:
                continue
            nums.append(p)
    return nums

_ALL_FIGS_HINT = re.compile(r"(?i)\b(все\s+рисунк\w*|все\s+схем\w*|все\s+картин\w*|all\s+pictures?|all\s+figs?)\b")

def _num_norm_fig(s: str | None) -> str:
    s = (s or "").strip()
    s = s.replace("\u00A0", " ")   # NBSP -> пробел
    s = s.replace(" ", "")
    s = s.replace(",", ".")        # 4,1 -> 4.1
    s = re.sub(r"[.:;)\]]+$", "", s)  # срез хвостовой пунктуации: "4." -> "4"
    return s


def _is_pure_figure_request(text: str) -> bool:
    """
    Эвристика: запрос ТОЛЬКО про рисунки (один или несколько номеров),
    без таблиц, разделов и общих вопросов.

    Используем, чтобы:
    — уйти в единый figure-пайплайн;
    — не запускать потом общий RAG-пайплайн, который дублирует ответы
      и может выдавать «данного рисунка нет в работе».
    """
    t = (text or "").strip()
    if not t:
        return False

    # «все рисунки» — отдельная ветка (_ALL_FIGS_HINT)
    if _ALL_FIGS_HINT.search(t):
        return False

    # нет упоминания рисунков — не наш случай
    if not FIG_NUM_RE.search(t):
        return False

    # если явно упоминают таблицы или разделы/главы — это уже смешанный запрос
    if _TABLE_ANY.search(t) or _SECTION_NUM_RE.search(t):
        return False

    return True


def _is_pure_section_request(text: str, intents: dict | None = None) -> bool:
    """
    Эвристика: «чистый» запрос про главу/раздел/пункт:
      — есть ссылка на главу/раздел/пункт;
      — нет одновременного запроса про таблицы, рисунки или список источников.

    Всё, что смешанное (глава + таблицы/рисунки/источники), должно идти
    в общий мультийнтентный пайплайн, а не в отдельный секционный ответ.
    """
    t = (text or "").strip()
    if not t:
        return False

    # нет указания пункта/главы — не наш случай
    if not _SECTION_NUM_RE.search(t):
        return False

    # если в тексте явно спрашивают про таблицы/рисунки/источники — это уже микс
    if _TABLE_ANY.search(t) or FIG_NUM_RE.search(t) or _SOURCES_HINT.search(t):
        return False

    # если detect_intents уже увидел таблицы/рисунки/источники — тоже не чистый раздел
    if intents:
        if (
            intents.get("tables", {}).get("want")
            or intents.get("figures", {}).get("want")
            or intents.get("sources", {}).get("want")
        ):
            return False

    return True


def _build_figure_records(uid: int, doc_id: int, nums: list[str]) -> list[dict]:
    """
    Единая "сборка" информации о рисунках:
    — номер и красивый display;
    — пути к картинкам;
    — точные числовые значения (из chart_data);
    — подпись и текст рядом;
    — vision-описание (если есть и не «описание не распознано»).
    """
    if not nums:
        return []

    # заранее тянем карточки из retrieval, чтобы получить images/текст рядом/vision
    try:
        cards = describe_figures_by_numbers(
            uid,
            doc_id,
            nums,
            sample_chunks=2,
            use_vision=True,
            lang="ru",
            vision_first_image_only=True,
        ) or []
    except Exception:
        cards = []

    cards_by_norm: dict[str, dict] = {}
    for c in cards:
        key = _num_norm_fig(str(c.get("num") or ""))
        if key and key not in cards_by_norm:
            cards_by_norm[key] = c

    idx_oox = _ooxml_get_index(doc_id)
    fig_idx = FIG_INDEX.get(doc_id)

    # ключ: нормализованный номер рисунка → record
    records_by_num: dict[str, dict] = {}

    for orig in nums:
        norm = _num_norm_fig(orig)
        if not norm:
            continue

        # если этот номер уже собран — не создаём дубль
        if norm in records_by_num:
            continue

        card = cards_by_norm.get(norm)
        rec: dict = {
            "owner_id": uid,      # нужен для вызова summarizer.extract_figure_value
            "doc_id": doc_id,     # идентификатор документа
            "num": norm,
            "orig": orig,
            "display": None,
            "images": [],
            # ЕДИНЫЙ источник истины по числам
            "raw_values": None,        # структурированные данные из OOXML
            "values_text": None,       # уже отформатированный блок «Точные значения»
            "values_source": None,     # "ooxml" | "summary" | "vision" | ...
            # для обратной совместимости со старым кодом
            "values": None,
            "near_text": [],
            "caption": None,
            "vision_desc": None,
        }


        # --- 1) данные из RAG-карточек ---
        if card:
            rec["display"] = card.get("display") or rec["display"]
            rec["images"] = [p for p in (card.get("images") or []) if p]
            rec["near_text"] = [
                (h or "").strip()
                for h in (card.get("highlights") or [])
                if (h or "").strip()
            ]
            vis = (card.get("vision") or {}).get("description") or ""
            vis_clean = vis.strip()
            low = vis_clean.lower()
            # отбрасываем заглушки вида «содержимое изображения (описание не распознано)»
            if vis_clean and "описание не распознано" not in low and "содержимое изображения" not in low:
                rec["vision_desc"] = vis_clean

        # --- 2) OOXML-индекс: подпись и image_path ---
        if idx_oox:
            oox_rec = _ooxml_find_figure_by_label(idx_oox, norm) or _ooxml_find_figure_by_label(idx_oox, orig)
            if oox_rec:
                cap = (oox_rec.get("caption") or "").strip()
                if cap:
                    rec["caption"] = cap
                if not rec["display"]:
                    label = oox_rec.get("n") or norm
                    rec["display"] = f"Рисунок {label}" + (f" — {cap}" if cap else "")
                path = oox_rec.get("image_path")
                if path and path not in rec["images"]:
                    rec["images"].append(path)

                # --- 3) локальный индекс figures.py: путь к картинке + подпись ---
        if fig_idx:
            try:
                recs = fig_find(fig_idx, number=orig) or fig_find(fig_idx, number=norm) or []
            except Exception:
                recs = []
            for r in recs:
                if not rec["display"]:
                    rec["display"] = figure_display_name(r)
                ap = r.get("abs_path")
                if ap and ap not in rec["images"]:
                    rec["images"].append(ap)
                cap_text = r.get("caption") or r.get("title")
                if cap_text and not rec["caption"]:
                    rec["caption"] = cap_text
                if not rec["near_text"] and cap_text:
                    rec["near_text"].append(cap_text)

        # --- 4) chart_data из attrs (ТОЧНЫЕ числовые значения из OOXML) ---
        row = _fetch_figure_row_by_num(uid, doc_id, orig)
        if not row and norm != orig:
            row = _fetch_figure_row_by_num(uid, doc_id, norm)

        if row:
            attrs_json = row["attrs"] if ("attrs" in row.keys()) else None

            # 4.a) пробуем достать структурированные raw_values из OOXML
            raw = _extract_raw_values_from_attrs(attrs_json)
            if raw:
                rec["raw_values"] = raw
                rec["values_text"] = _format_exact_values(raw)
                rec["values_source"] = "ooxml"
                rec["values"] = rec["values_text"]  # для старого кода

            # фолбэк: если raw_values не получилось собрать, но старый
            # _parse_chart_data вернул плоский список — аккуратно
            # оборачиваем его в одну серию без доп. магии
            if not rec.get("raw_values"):
                cd, _ctype, _attrs = _parse_chart_data(attrs_json)
                if cd:
                    categories = [
                        str(r.get("label") or r.get("name") or r.get("category") or "")
                        for r in cd
                    ]
                    values = []
                    for r in cd:
                        v = r.get("value")
                        if v is None:
                            v = r.get("y") or r.get("x") or r.get("v") or r.get("count")
                        values.append(v)
                    rec["raw_values"] = {
                        "categories": categories,
                        "series": [{
                            "name": None,
                            "unit": (cd[0].get("unit") if cd and isinstance(cd[0].get("unit"), str) else None),
                            "values": values,
                        }],
                    }
                    rec["values_text"] = _format_exact_values(rec["raw_values"])
                    rec["values_source"] = "ooxml"
                    rec["values"] = rec["values_text"]

            # 4.b) display — как и раньше
            if not rec["display"]:
                title_text = row["text"] if ("text" in row.keys()) else None
                rec["display"] = _compose_figure_display(
                    attrs_json,
                    row["section_path"],
                    title_text,
                )

            # 4.c) ТЕКСТ ПОСЛЕ РИСУНКА: 2–3 абзаца прямо из диплома
            try:
                follow = _figure_following_paragraphs(uid, doc_id, row, max_paragraphs=3, max_chars=1500)
                if follow:
                    rec["near_text"] = follow
            except Exception:
                # не ломаем обработку рисунка, если что-то пошло не так
                pass


        if not rec["display"]:
            rec["display"] = f"Рисунок {norm}"

        records_by_num[norm] = rec

    return list(records_by_num.values())


def _fig_values_text_from_records(
    records: list[dict],
    *,
    need_values: bool,
) -> str:
    """
    Собираем текстовый блок с точными значениями по рисункам.
    Приоритет:
      1) rec.raw_values / rec.values_text (OOXML);
      2) oox_fig_lookup (готовый текст).
    """
    lines: list[str] = []

    for rec in records:
        # 1) приоритет — сырые OOXML-данные
        raw = rec.get("raw_values")
        values_text = (rec.get("values_text") or rec.get("values") or "").strip()

        if raw and not values_text:
            values_text = _format_exact_values(raw)
            rec["values_text"] = values_text
            rec["values"] = values_text
            rec["values_source"] = rec.get("values_source") or "ooxml"

        # 2) ФОЛБЭК: OOXML-индекс figure_lookup (готовый текст)
        if not values_text:
            try:
                doc_id = rec.get("doc_id")
                num = rec.get("orig") or rec.get("num")
                idx = _ooxml_get_index(doc_id) if doc_id else None
                body = ""

                if idx and "oox_fig_lookup" in globals() and num:
                    oox_res = oox_fig_lookup(idx, str(num))
                    if isinstance(oox_res, str):
                        body = oox_res.strip()
                    elif isinstance(oox_res, dict):
                        body = (
                            (oox_res.get("values_text")
                             or oox_res.get("text")
                             or "")
                        ).strip()

                if body:
                    values_text = body
                    rec["values_text"] = body
                    rec["values"] = body
                    rec["values_source"] = rec.get("values_source") or "ooxml_text"
            except Exception:
                pass

        # если после всех попыток чисел нет — пропускаем этот рисунок
        if not values_text:
            continue

        disp = rec.get("display") or f"Рисунок {rec.get('num') or ''}".strip()
        # для OOXML-данных заголовок однозначный
        if rec.get("values_source") == "ooxml":
            title = f"**{disp} — точные значения (как в документе)**"
        elif rec.get("values_source") in {"summary", "vision"}:
            title = f"**{disp} — значения, распознанные по изображению (возможны неточности)**"
        else:
            title = f"**{disp} — значения**"

        lines.append(f"{title}\n\n{values_text}")

    if lines:
        return "\n\n".join(lines)

    if need_values:
        return (
            "По указанным рисункам не удалось автоматически извлечь точные числовые данные "
            "(нет структурированных OOXML-данных и распознавания по картинкам). "
            "Могу дать только текстовое описание."
        )

    return ""


async def _send_fig_values_from_records(
    m: types.Message,
    records: list[dict],
    *,
    need_values: bool,
) -> None:
    """
    Обратная совместимость: если нужно отдельно отправить только числа.
    В основном сценарии теперь используем _fig_values_text_from_records
    и склеиваем с описанием.
    """
    text = _fig_values_text_from_records(records, need_values=need_values)
    if text:
        await _send(m, text)


async def _explain_figures_with_gpt(
    m: types.Message,
    records: list[dict],
    question: str,
    *,
    verbosity: str,
    need_values: bool,
    values_prefix: str = "",
) -> None:
    """
    Финальный шаг: GPT даёт связное текстовое объяснение по всем рисункам сразу,
    используя подписи, текст рядом и уже извлечённые числовые данные.

    Если передан values_prefix, то он добавляется в начало ответа:
    сначала блок «точные значения», затем интерпретация.
    """
    if not (chat_with_gpt or chat_with_gpt_stream):
        return

    if not records:
        return

    ctx_blocks: list[str] = []
    for rec in records:
        disp = rec.get("display") or f"Рисунок {rec.get('num') or ''}".strip()
        parts: list[str] = [disp]
        if rec.get("caption"):
            parts.append(f"Подпись: {rec['caption']}")
        if rec.get("near_text"):
            parts.append("Текст рядом: " + " ".join(rec["near_text"][:2]))
        if rec.get("vision_desc"):
            parts.append("Описание по картинке: " + rec["vision_desc"])

        values_text = (rec.get("values_text") or rec.get("values") or "").strip()
        if values_text:
            parts.append("Точные значения (как в документе):\n" + values_text)

        # при желании можно подать и сырую структуру, чтобы LLM понимала серии/категории
        if rec.get("raw_values"):
            try:
                parts.append("Сырые данные диаграммы (JSON):\n" + json.dumps(rec["raw_values"], ensure_ascii=False))
            except Exception:
                pass

        ctx_blocks.append("\n".join(parts))


    ctx = "\n\n---\n\n".join(ctx_blocks)
    if not ctx.strip():
        return

    focus = (
        "с акцентом на точные числовые значения и их интерпретацию"
        if need_values
        else "подробно поясняя смысл и выводы по рисункам"
    )

    system_prompt = (
        "Ты репетитор по дипломным работам. В ЭТОМ вызове ты анализируешь только рисунки "
        "(диаграммы, графики) из диплома. "
        "Тебе уже даны подписи, ближайший текст и точные числовые данные диаграмм из документа. "
        "Используй эти данные как есть: не придумывай новые числа, не пересчитывай проценты "
        "и не пытайся нормировать суммы до 100%. Не ссылайся на номера страниц.\n"
        "Не придумывай предметную область и термины (например, продажи, клиенты, выручка, "
        "маркетинг и т.п.), если они не упомянуты в подписях или тексте рядом с рисунком."
    )


    user_prompt = (
        f"Вопрос пользователя: {question}\n\n"
        "Числа по каждому рисунку уже перечислены выше в блоках «Точные значения (как в документе)» "
        "и/или в JSON-структуре. НЕ переписывай эти числа и НЕ изменяй проценты. "
        "Твоя задача — только смысловая интерпретация: что показывают рисунки, какие уровни выше/ниже, "
        "какие тенденции видны, какие выводы можно сделать.\n"
        "Не придумывай, к какой предметной области относятся данные (например, продажи, клиенты, рынок, "
        "маркетинг и т.п.), если это явно не указано в подписях или тексте рядом. Если по рисункам "
        "непонятно, к чему относятся показатели, прямо напиши, что предметная область в тексте не указана.\n\n"
        f"Сконцентрируйся только на указанных рисунках, опиши их содержание и сделай интерпретацию {focus}.\n"
        f"{_verbosity_addendum(verbosity, 'описания рисунков')}"
    )


    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": f"[Собранная информация о рисунках]\n{ctx}"},
        {"role": "user", "content": user_prompt},
    ]

    try:
        ans = chat_with_gpt(messages, temperature=0.2, max_tokens=FINAL_MAX_TOKENS)
    except Exception as e:
        logging.exception("figure explanation failed: %s", e)
        ans = ""

    ans = (ans or "").strip()
    prefix = (values_prefix or "").strip()

    if prefix and ans:
        final = prefix + "\n\n" + ans
    elif prefix:
        final = prefix
    else:
        final = ans

    if final:
        await _send(m, _strip_unwanted_sections(final))

async def _answer_figure_query(
    m: types.Message, uid: int, doc_id: int, text: str, *, verbosity: str = "normal"
) -> bool:
    """
    Новый единый сценарий:
    1) всегда вытаскиваем максимум по рисункам (_build_figure_records);
    2) собираем общий блок с точными значениями (если есть);
    3) даём одно связное пояснение через GPT, в которое подмешан блок значений.

    Поведение не зависит от того, спросили ли «опиши рисунок 2.3»
    или «дай точные значения по рисунку 2.3» — меняется только акцент
    в текстовом объяснении.
    """
    # 0) нужен ли особый акцент на числах
    need_values = bool(_VALUES_HINT.search(text or ""))

    # 1) вытаскиваем номера рисунков из текста
    raw_list = _extract_fig_nums(text or "")
    seen: set[str] = set()
    nums: list[str] = []
    for token in raw_list:
        n = _num_norm_fig(token)
        if n and n not in seen:
            seen.add(n)
            nums.append(token)   # сохраняем исходный вид номера

    if not nums:
        return False

    # 2) собираем единую структуру по всем рисункам
    records = _build_figure_records(uid, doc_id, nums)
    if not records:
        # 🔧 Раньше здесь сразу шёл ответ «Указанные рисунки в работе не найдены.»,
        # из-за чего мы не доходили до общего RAG-пайплайна и не могли,
        # например, ответить по таблицам или общему контексту.
        # Теперь просто говорим вызывающему коду «я не обработал этот запрос».
        return False

    # 4) собираем общий блок с точными значениями (без отправки)
    values_block = _fig_values_text_from_records(records, need_values=need_values)

    # 5) один текст: «значения + описание»
    await _explain_figures_with_gpt(
        m,
        records,
        text,
        verbosity=verbosity,
        need_values=need_values,
        values_prefix=values_block,
    )

    # 6) обновляем «последний упомянутый рисунок» для анафорических вопросов
    try:
        LAST_REF.setdefault(uid, {})["figure_nums"] = [r["num"] for r in records]
    except Exception:
        pass

    return True

def _ooxml_table_block(uid: int, doc_id: int, num: str) -> str | None:
    """
    1) Пытаемся взять сырые данные таблицы из OOXML-индекса через oox_tbl_lookup.
    2) Если там ничего не нашли — падаем в обычные chunks (table/table_row) и
       собираем текст таблицы по строкам. Это защищает от глюков OOXML-парсера.
    """

    # маленький внутренний хелпер: красиво форматируем rows → текстовую таблицу
    def _format_oox_rows(res: dict) -> str:
        rows = res.get("rows") or []
        lines: list[str] = []
        for row in rows:
            if not isinstance(row, (list, tuple)):
                continue
            cells = [(str(c) if c is not None else "").strip() for c in row]
            # пустые хвостовые ячейки убираем
            while cells and cells[-1] == "":
                cells.pop()
            lines.append(" | ".join(cells))
        return "\n".join(lines).strip()

    # --- 1. OOXML ---
    idx = _ooxml_get_index(doc_id)
    if idx and "oox_tbl_lookup" in globals():
        try:
            res = oox_tbl_lookup(idx, str(num))
        except Exception:
            res = None

        if res is not None:
            if isinstance(res, str):
                body = res.strip()
            elif isinstance(res, dict) and "rows" in res:
                # НОРМАЛЬНОЕ человекочитаемое представление таблицы
                body = _format_oox_rows(res)
            else:
                # запасной вариант: если структура неожиданная — просто сериализуем
                try:
                    body = json.dumps(res, ensure_ascii=False, indent=2)
                except Exception:
                    body = str(res)

            body = (body or "").strip()
            if body:
                # меняем подпись, т.к. теперь это не «сырой JSON», а нормальная таблица
                return f"Таблица {num} (как в документе):\n{body}"

    # --- 2. Фолбэк: chunks из БД ---
    con = get_conn()
    cur = con.cursor()

    has_et   = _table_has_columns(con, "chunks", ["element_type"])
    has_attr = _table_has_columns(con, "chunks", ["attrs"])

    rows = []

    try:
        if has_attr and has_et:
            like1 = f'%\"caption_num\": \"{num}\"%'
            like2 = f'%\"label\": \"{num}\"%'
            cur.execute(
                """
                SELECT section_path, text
                FROM chunks
                WHERE owner_id=? AND doc_id=?
                  AND element_type IN ('table','table_row')
                  AND (attrs LIKE ? OR attrs LIKE ?)
                ORDER BY id ASC
                """,
                (uid, doc_id, like1, like2),
            )
            rows = cur.fetchall() or []

        if not rows:
            # фолбэк по section_path / тексту, работает даже на старых индексах
            variants = _table_num_variants(num)
            sec_patterns = []
            txt_patterns = []

            for v in variants:
                # "Таблица 6", "таблица 6", "табл. 6"
                sec_patterns.append(f"%Таблица {v}%")
                sec_patterns.append(f"%таблица {v}%")
                sec_patterns.append(f"%табл. {v}%")
                # текстовый вариант типа "[Таблица] 6" и "[табл.] 6"
                txt_patterns.append(f"[Таблица]%{v}%")
                txt_patterns.append(f"[табл.]%{v}%")

            if not sec_patterns and not txt_patterns:
                rows = []
            else:
                conds_sec = " OR ".join(["section_path LIKE ? COLLATE NOCASE"] * len(sec_patterns)) if sec_patterns else "0"
                conds_txt = " OR ".join(["text LIKE ? COLLATE NOCASE"] * len(txt_patterns)) if txt_patterns else "0"

                sql = f"""
                    SELECT section_path, text
                    FROM chunks
                    WHERE owner_id=? AND doc_id=?
                      AND ( ({conds_sec}) OR ({conds_txt}) )
                    ORDER BY page ASC, id ASC
                """

                params = [uid, doc_id] + sec_patterns + txt_patterns
                cur.execute(sql, params)
                rows = cur.fetchall() or []
    finally:
        con.close()

    if not rows:
        return None

    sec = (rows[0]["section_path"] or "").strip()
    lines = []
    if sec:
        lines.append(f"[{sec}]")
    for r in rows:
        t = (r["text"] or "").strip()
        if t:
            lines.append(t)

    body = "\n".join(lines).strip()
    if not body:
        return None

    return f"Таблица {num} (как в документе, по строкам таблицы):\n{body}"


def _table_related_context(
    uid: int,
    doc_id: int,
    num: str,
    *,
    max_chars: int = 4000,
) -> str:
    """
    Ищем дополнительный текст, который связан с таблицей `num`.

    1) Сначала ищем прямые упоминания «таблица N» (кроме самих ячеек таблицы).
    2) Если таких фрагментов нет – пытаемся найти абзац(ы) вида «Примечание …»
       сразу после этой таблицы.
    3) Если и это не сработало – делаем семантический поиск по всему документу.
    """
    con = get_conn()
    cur = con.cursor()
    has_et = _table_has_columns(con, "chunks", ["element_type"])

    variants = _table_num_variants(num)
    txt_patterns: list[str] = []
    for v in variants:
        txt_patterns.append(f"%Таблица {v}%")
        txt_patterns.append(f"%таблица {v}%")
        txt_patterns.append(f"%табл. {v}%")

    if not txt_patterns:
        cur.close()
        con.close()
        return ""

    conds_txt = " OR ".join(["text LIKE ? COLLATE NOCASE"] * len(txt_patterns))

    if has_et:
        cur.execute(
            f"""
            SELECT page, section_path, text, element_type
            FROM chunks
            WHERE owner_id=? AND doc_id=?
              AND ({conds_txt})
            ORDER BY page ASC, id ASC
            """,
            (uid, doc_id, *txt_patterns),
        )
    else:
        cur.execute(
            f"""
            SELECT page, section_path, text
            FROM chunks
            WHERE owner_id=? AND doc_id=?
              AND ({conds_txt})
            ORDER BY page ASC, id ASC
            """,
            (uid, doc_id, *txt_patterns),
        )

    rows = cur.fetchall() or []
    con.close()

    parts: list[str] = []
    total = 0

    for r in rows:
        et = ""
        if "element_type" in r.keys():
            et = (r["element_type"] or "").lower()
        if et in ("table", "table_row"):
            continue

        t = (r["text"] or "").strip()
        if not t:
            continue

        if total + len(t) > max_chars:
            parts.append(t[: max_chars - total])
            break

        parts.append(t)
        total += len(t)

    extra = "\n\n".join(parts).strip()
    if extra:
        return extra

    # НОВОЕ: если прямых упоминаний «таблица N» нет,
    # попробуем подобрать «Примечание …» сразу после этой таблицы.
    try:
        con = get_conn()
        cur = con.cursor()
        has_et = _table_has_columns(con, "chunks", ["element_type"])
        has_attr = _table_has_columns(con, "chunks", ["attrs"])

        base_row = None
        if has_attr and has_et:
            like1 = f'%\"caption_num\": \"{num}\"%'
            like2 = f'%\"label\": \"{num}\"%'
            cur.execute(
                """
                SELECT id, page, section_path, element_type, text
                FROM chunks
                WHERE owner_id=? AND doc_id=?
                  AND element_type IN ('table','table_row')
                  AND (attrs LIKE ? OR attrs LIKE ?)
                ORDER BY id ASC LIMIT 1
                """,
                (uid, doc_id, like1, like2),
            )
            base_row = cur.fetchone()

        if not base_row:
            cur.execute(
                """
                SELECT id, page, section_path, element_type, text
                FROM chunks
                WHERE owner_id=? AND doc_id=?
                  AND element_type IN ('table','table_row')
                  AND section_path LIKE ? COLLATE NOCASE
                ORDER BY id ASC LIMIT 1
                """,
                (uid, doc_id, f'%Таблица {num}%'),
            )
            base_row = cur.fetchone()

        note_text = ""
        if base_row:
            base_id = base_row["id"]
            page = base_row["page"]

            cur.execute(
                """
                SELECT text, element_type
                FROM chunks
                WHERE owner_id=? AND doc_id=? AND id>? AND page=?
                ORDER BY id ASC LIMIT 10
                """,
                (uid, doc_id, base_id, page),
            )
            note_parts: list[str] = []
            started = False
            for r in cur.fetchall() or []:
                et = (r["element_type"] or "").lower() if "element_type" in r.keys() else ""
                if et in ("heading", "table", "figure", "table_row"):
                    # дальше уже другая структура
                    break
                t = (r["text"] or "").strip()
                if not t:
                    continue
                low = t.lower()
                if low.startswith("примечание"):
                    started = True
                    note_parts.append(t)
                    continue
                if started:
                    # цепляем хвост примечания, если оно в несколько абзацев
                    note_parts.append(t)

            if note_parts:
                note_text = "\n".join(note_parts).strip()

        con.close()

        if note_text:
            return note_text
    except Exception:
        # не ломаем пайплайн, если что-то пошло не так
        pass

    # второй проход: семантический поиск по всему документу
    query = f"подробное текстовое пояснение, анализ и выводы по данным таблицы {num}"
    try:
        ctx = best_context(
            uid,
            doc_id,
            query,
            max_chars=max_chars,
        ) or ""
    except Exception:
        ctx = ""

    return (ctx or "").strip()




async def _answer_table_query(
    m: types.Message,
    uid: int,
    doc_id: int,
    text: str,
    *,
    verbosity: str = "normal",
    mode: str = "normal",
) -> bool:
    """
    Спец-путь для запросов вида:
      - "опиши таблицу 4"
      - "что показывает таблица 2.3"
      - "сделай выводы по таблице 4"
      - и фоллоу-апа "опиши подробнее" по этой же таблице (mode=\"more\").
    """
    nums = _extract_table_nums(text)
    if not nums:
        return False

    # запоминаем последнюю(ие) таблицу(ы) для фраз типа «опиши подробнее»
    try:
        LAST_REF.setdefault(uid, {})["table_nums"] = [
            n.replace(" ", "").replace(",", ".") for n in nums
        ]
    except Exception:
        pass

    blocks: list[str] = []
    missing: list[str] = []

    for n in nums:
        # 1) обычный путь: таблица из OOXML
        blk = _ooxml_table_block(uid, doc_id, n)
        if blk:
            blocks.append(blk)
            continue

        # 1a) НОВОЕ: попробуем найти "таблицу-рисунок" как диаграмму/рисунок
        fig_records = _build_figure_records(uid, doc_id, [n])
        logging.info(
            "TAB[query] table %s as figure: %d records",
            n,
            len(fig_records) if fig_records else 0,
        )
        if fig_records:
            values_text = _fig_values_text_from_records(fig_records, need_values=True)
            if values_text:
                blocks.append(
                    f"Таблица {n} (в документе оформлена как диаграмма/рисунок):\n"
                    f"{values_text}"
                )
                try:
                    LAST_REF.setdefault(uid, {})["figure_nums"] = [r["num"] for r in fig_records]
                except Exception:
                    pass
                continue

        # 2) fallback: настоящая OCR по картинке (если никаких chart_data нет)
        ocr_blk = _ocr_table_block_from_image(uid, doc_id, n)
        if ocr_blk:
            blocks.append(ocr_blk)
        else:
            missing.append(n)

    if not blocks:
        # ничего не смогли собрать ни из OOXML, ни из OCR/диаграмм
        bad = ", ".join(missing or nums)
        await _send(
            m,
            f"Таблица {bad} в документе не найдена. "
            "Проверь, правильно ли указан номер."
        )
        return True  # считаем запрос обработанным, дальше по пайплайну не идём

    # если часть таблиц не найдена — явно предупреждаем об этом в ответе
    if missing:
        blocks.append(
            "⚠️ По следующим таблицам данных в документе не найдено: "
            + ", ".join(missing)
        )

    ctx_tables = "\n\n---\n\n".join(blocks)

    # Блок, который ВСЕГДА пойдёт в финальный ответ пользователю,
    # чтобы он видел все значения таблицы ровно в том виде, как мы её распознали.
    raw_values_text = ""
    if ctx_tables:
        raw_values_text = (
            "**Все значения таблиц (как в документе)**\n\n"
            f"{ctx_tables}"
        )

    # Дополнительный текст по таблицам (для обычного режима и "подробнее")
    # В обычном ответе берём поменьше символов, в "подробнее" — побольше.
    extra_ctx_parts: list[str] = []

    for n in nums:
        extra = _table_related_context(
            uid,
            doc_id,
            n,
            max_chars=4000 if mode == "more" else 2000,
        )
        if extra:
            extra_ctx_parts.append(
                f"[Дополнительный текст по таблице {n}]\n{extra}"
            )

    extra_ctx = "\n\n---\n\n".join(extra_ctx_parts).strip()

    # Если это запрос «подробнее», но в самой работе НЕТ доп. текста про эту таблицу,
    # мы сохраняем сырые данные таблиц и предлагаем расширенный ответ от [модель],
    # который будет опираться на ЭТИ данные.
    if mode == "more" and not extra_ctx:
        nums_str = ", ".join(nums)
        MODEL_EXTRA_PENDING[uid] = {
            "kind": "table_more",
            # сам вопрос пользователя (чаще всего «опиши подробнее (таблица N)»)
            "question": text,
            # сырые данные таблиц из OOXML/картинки — чтобы [модель] их видела
            "ctx_tables": ctx_tables,
            "nums": nums,
            # нужен, чтобы потом ещё раз сходить в документ за контекстом
            "doc_id": doc_id,
        }
        await _send(
            m,
            "В самой работе нет дополнительного текста, который подробно объясняет эту таблицу. "
            "Могу дополнительно, как [модель], подробно пояснить её, опираясь на сами данные таблицы "
            "и общие теоретические знания по теме (без ссылок на текст ВКР). "
            "Если нужно — напиши «да», если не нужно — «нет»."
        )
        return True


    # Общий контекст для GPT: сырые данные таблиц +, при наличии, доп. текст
    full_ctx = ctx_tables
    if extra_ctx:
        full_ctx += "\n\n[Дополнительный текст из работы про эти таблицы]\n" + extra_ctx

    system_prompt = (
        "Ты репетитор по дипломным работам. Ниже даны таблицы, распарсенные прямо из документа.\n"
        "Отвечай СТРОГО по этим данным:\n"
        "— не придумывай новые строки, столбцы и значения;\n"
        "— не добавляй факты, которых нет в таблицах;\n"
        "— не придумывай предметную область и термины (например, продажи, клиенты, выручка, рынок, "
        "маркетинг и т.п.), если они не встречаются в заголовках, подписях или строках таблиц;\n"
        "— не ссылаться на страницы, только описывай содержание.\n"
        "Если в вопросе указан номер таблицы, но такой таблицы нет в переданном контексте — "
        "напиши, что по этому номеру в контексте данных нет.\n\n"
        "Структура ответа ДОЛЖНА быть такой:\n"
        "1) Раздел «Структура таблицы» — коротко объясни, что по строкам и что по столбцам.\n"
        "2) Раздел «Все значения» — выпиши ВСЕ числовые значения таблицы БЕЗ ПРОПУСКОВ.\n"
        "   Для каждой строки таблицы (например: «Отцы», «Матери», «Общее»)\n"
        "   напиши одну строку вида:\n"
        "   «Отцы: 31,55; 26,85; 27,1; …; 51,1» — значения идут строго по порядку столбцов.\n"
        "   Нельзя объединять строки и нельзя выбрасывать какие-либо числа.\n"
        "3) Раздел «Выводы» — сделай аккуратную интерпретацию и выводы на основе этих данных.\n"
        "Если в контексте есть абзац, начинающийся с «Примечание», обязательно приведи его "
        "отдельным подпунктом «Примечание» и не сокращай текст."
    )


    # В режиме "подробнее" прямо говорим, что нужен более развёрнутый разбор
    if mode == "more":

        user_prompt = (
            f"Вопрос пользователя: {text}\n\n"
            "Ниже структура таблиц в машинно-читаемом виде и дополнительный текст из работы. "
            "Сделай БОЛЕЕ ПОДРОБНЫЙ разбор по этой таблице.\n\n"
            "Обязательно:\n"
            "— строго следуй структуре ответа из системной инструкции "
            "(«Структура таблицы» → «Все значения» → «Выводы»);\n"
            "— в разделе «Все значения» перечисли ВСЕ показатели и их значения, без сокращений "
            "и пропусков (можно в виде списков по группам/строкам/столбцам);\n"
            "— в разделе «Выводы» аккуратно интерпретируй различия и тенденции, не придумывая новых данных."
            f"{_verbosity_addendum('detailed', 'подробного описания таблицы')}\n\n"
            "[Таблицы и связанный текст из документа]\n"
            f"{full_ctx}"
        )

    else:
        user_prompt = (
            f"Вопрос пользователя: {text}\n\n"
            "Ниже структура таблиц в машинно-читаемом виде. "
            "Ответ оформи в три явных раздела: «Структура таблицы», «Все значения», «Выводы».\n"
            "Сначала в разделе «Структура таблицы» простыми словами объясни, что по строкам и что по столбцам.\n"
            "Затем в разделе «Все значения» выпиши все числовые значения таблицы в текстовом виде "
            "(можно списками по группам/строкам/столбцам), не сокращая и не выбрасывая числа.\n"
            "В конце, в разделе «Выводы», сделай аккуратные выводы: какие значения выше/ниже, "
            "какие различия заметны, какие тенденции можно отметить.\n"
            "Не придумывай никаких фактов, которых нет в данных таблиц."
            f"{_verbosity_addendum(verbosity, 'описания таблицы')}\n\n"
            "[Таблицы из документа]\n"
            f"{full_ctx}"
        )


    try:
        answer = chat_with_gpt(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=FINAL_MAX_TOKENS,
        )
    except Exception as e:
        logging.exception("table explanation failed: %s", e)
        await _send(
            m,
            "Не получилось сгенерировать объяснение по таблице — произошла техническая ошибка. "
            f"Подробности для логов: {e}"
        )
        return True  # запрос обработан, не проваливаемся в общий RAG

    answer = (answer or "").strip()

    # --- НОВОЕ: если модель дала пустой или совсем короткий ответ,
    # пробуем более простой fallback-промпт только по значениям таблицы.
    if not answer or len(answer) < 60:
        if raw_values_text:
            fb_system = (
                "Ты репетитор по дипломным работам. Ниже дано текстовое представление таблицы "
                "из диплома (все её значения). "
                "По этим данным опиши простыми словами, что показывает таблица, "
                "какие значения выше/ниже и какие 2–3 вывода можно сделать. "
                "Не придумывай новых чисел и не пересчитывай проценты. "
                "Не придумывай предметную область и термины (например, продажи, клиенты, выручка, "
                "маркетинг и т.п.), если они не указаны в самой таблице."
            )

            fb_user = (
                f"Таблица из диплома:\n{ctx_tables}\n\n"
                f"Вопрос пользователя: {text}\n\n"
                "Сформулируй понятное человеку описание и выводы по этой таблице."
            )
            try:
                fb_answer = chat_with_gpt(
                    [
                        {"role": "system", "content": fb_system},
                        {"role": "user",   "content": fb_user},
                    ],
                    temperature=0.3,
                    max_tokens=FINAL_MAX_TOKENS,
                )
            except Exception as e:
                logging.exception("table fallback explanation failed: %s", e)
                fb_answer = ""

            fb_answer = (fb_answer or "").strip()
            if fb_answer:
                answer = fb_answer
    # --- /НОВОЕ ---

    # Если модель в итоге так и не дала осмысленный текст — показываем только значения
    if not answer:
        if raw_values_text:
            await _send(
                m,
                raw_values_text
                + "\n\n"
                + "Модель не смогла сгенерировать осмысленное текстовое описание таблицы. "
                  "Вот сами значения таблицы. Если нужно пояснение — попробуй переформулировать вопрос."
            )
        else:
            await _send(
                m,
                "Модель не вернула осмысленный текст по таблице. "
                "Попробуй переформулировать вопрос или задать его ещё раз."
            )
        return True  # тоже не падаем в общий пайплайн

    # Нормальный кейс: сначала ВСЕ значения, потом человеческое объяснение
    final_answer = _strip_unwanted_sections(answer)
    if raw_values_text:
        final_answer = raw_values_text + "\n\n\n" + final_answer

    await _send(m, final_answer)
    return True



# -------------------------- САМОВОССТАНОВЛЕНИЕ ИНДЕКСА --------------------------

def _count_et(con, uid: int, doc_id: int, et: str) -> int:
    cur = con.cursor()
    if _table_has_columns(con, "chunks", ["element_type"]):
        cur.execute(
            "SELECT COUNT(*) AS c FROM chunks WHERE owner_id=? AND doc_id=? AND element_type=?",
            (uid, doc_id, et),
        )
        row = cur.fetchone()
        return int(row["c"] or 0)
    return 0

def _need_self_heal(uid: int, doc_id: int, need_refs: bool, need_figs: bool) -> tuple[bool, int, int]:
    """
    Самовосстановление теперь завязано на наличие МЕДИАДАННЫХ:
      — есть ли где-то в attrs список images;
      — есть ли chart_data (данные диаграмм), независимо от element_type.
    Если хотя бы одно найдено — считаем, что «фигуры есть» (fc=1).
    Фолбэк для старых БД без attrs: считаем element_type='figure'.
    """
    con = get_conn()
    rc = _count_et(con, uid, doc_id, "reference") if need_refs else 1

    # по умолчанию «фигуры присутствуют», если они не нужны
    fc = 1
    if need_figs:
        media_found = False
        try:
            cur = con.cursor()
            # есть ли колонка attrs — проверяем напрямую медиаданные
            if _table_has_columns(con, "chunks", ["attrs"]):
                cur.execute(
                    "SELECT attrs FROM chunks WHERE owner_id=? AND doc_id=? AND attrs IS NOT NULL",
                    (uid, doc_id),
                )
                rows = cur.fetchall() or []
                for r in rows:
                    attrs_json = r["attrs"] or None
                    if not attrs_json:
                        continue
                    # быстрый чек на images
                    try:
                        a = json.loads(attrs_json)
                        imgs = a.get("images") or []
                        if isinstance(imgs, list) and any(imgs):
                            media_found = True
                            break
                    except Exception:
                        pass
                    # аккуратно проверим chart_data (используем существующий парсер)
                    try:
                        cd, _, _ = _parse_chart_data(attrs_json)  # returns (rows|None, type|None, attrs_dict)
                        if cd:
                            media_found = True
                            break
                    except Exception:
                        # парсер не обязателен для решения — просто идём дальше
                        pass
            else:
                # очень старый индекс без attrs — фолбэк к figure-чанкам
                media_found = (_count_et(con, uid, doc_id, "figure") > 0)
        except Exception:
            # защитный фолбэк: считаем по figure-чанкам
            media_found = (_count_et(con, uid, doc_id, "figure") > 0)

        fc = 1 if media_found else 0

    con.close()
    return (rc == 0 or fc == 0, rc, fc)


def _reindex_with_sections(uid: int, doc_id: int, sections: list[dict]) -> None:
    delete_document_chunks(doc_id, uid)
    index_document(uid, doc_id, sections)
    invalidate_cache(uid, doc_id)
    set_document_indexer_version(doc_id, CURRENT_INDEXER_VERSION)
    update_document_meta(doc_id, layout_profile=_current_embedding_profile())


async def _ensure_modalities_indexed(m: types.Message, uid: int, doc_id: int, intents: dict):
    """Если документ старый и нет reference/figure — тихо перепарсим новым парсером и переиндексируем."""
    need_refs = bool(intents.get("sources", {}).get("want"))

    need_figs = bool(intents.get("figures", {}).get("want"))
    if not (need_refs or need_figs):
        return

    should, have_refs, have_figs = _need_self_heal(uid, doc_id, need_refs, need_figs)
    if not should:
        return

    con = get_conn()
    cur = con.cursor()
    cur.execute("SELECT path FROM documents WHERE id=? AND owner_id=?", (doc_id, uid))
    row = cur.fetchone()
    con.close()
    if not row:
        return

    path = row["path"]
    try:
        sections = _parse_by_ext(path)
        sections = enrich_sections(sections, doc_kind=os.path.splitext(path)[1].lower().strip("."))
    except Exception as e:
        logging.exception("re-parse/enrich failed: %s", e)
        return

    new_refs = sum(1 for s in sections if (s.get("element_type") == "reference"))
    new_figs = sum(1 for s in sections if (s.get("element_type") == "figure"))

    do_reindex = False
    if need_refs and have_refs == 0 and new_refs > 0:
        do_reindex = True
    if need_figs and have_figs == 0 and new_figs > 0:
        do_reindex = True

    if do_reindex:
        try:
            _reindex_with_sections(uid, doc_id, sections)
            await _send(m, "Обновил индекс документа: добавлены распознанные рисунки/источники (включая OOXML-диаграммы).")
        except Exception as e:
            logging.exception("self-heal reindex failed: %s", e)


# -------------------------- Сбор фактов --------------------------

def _gather_facts(uid: int, doc_id: int, intents: dict) -> dict:
    """
    Собираем ТОЛЬКО факты из БД/индекса, без генерации текста.
    """
    facts: dict[str, object] = {"doc_id": doc_id, "owner_id": uid}
    # флаг «точные числа как в документе»
    exact = bool(intents.get("exact_numbers"))
    # если явно просят конкретную таблицу(ы) — всегда работаем в режиме ТОЧНЫХ чисел
    if intents.get("tables", {}).get("describe"):
        exact = True
    facts["exact_numbers"] = exact


    # ----- Таблицы -----
    if intents["tables"]["want"]:
        total_tables = _count_tables(uid, doc_id)
        basenames = _distinct_table_basenames(uid, doc_id)

        con = get_conn()
        cur = con.cursor()
        items: list[str] = []
        for base in basenames:
            cur.execute(
                """
                SELECT attrs FROM chunks
                WHERE owner_id=? AND doc_id=? AND element_type='table_row'
                  AND section_path LIKE ? || ' [row %'
                ORDER BY id ASC LIMIT 1
                """,
                (uid, doc_id, base),
            )
            r = cur.fetchone()
            attrs_json = r["attrs"] if r else None

            cur.execute(
                """
                SELECT text FROM chunks
                WHERE owner_id=? AND doc_id=? AND element_type='table_row'
                  AND section_path LIKE ? || ' [row %'
                ORDER BY id ASC LIMIT 2
                """,
                (uid, doc_id, base),
            )
            rows = cur.fetchall() or []
            first_row_text = None
            for rr in rows:
                cand = (rr["text"] or "").split("\n")[0]
                cand = " — ".join([c.strip() for c in cand.split(" | ") if c.strip()])
                if cand:
                    first_row_text = cand
                    break

            title = _compose_display_from_attrs(attrs_json, base, first_row_text)
            title = _strip_table_prefix(title)
            items.append(title)

        con.close()
        t_limit = int(intents.get("tables", {}).get("limit", 10))
        facts["tables"] = {
            "count": total_tables,
            "list": items[:t_limit],
            "more": max(0, len(items) - t_limit),
            "describe": [],
        }

        # Авто-описание для общего запроса про таблицы
        desc_cards = []
        if not intents.get("tables", {}).get("describe"):
            # возьмём первые 3–5 таблиц из списка
            bases = _distinct_table_basenames(uid, doc_id)[:min(5, t_limit)]
            con = get_conn()
            cur = con.cursor()
            for base in bases:
                # attrs + первые 1–2 строки
                cur.execute("""
                    SELECT page, section_path, attrs FROM chunks
                    WHERE owner_id=? AND doc_id=? AND element_type IN ('table','table_row')
                    AND (section_path=? OR section_path LIKE ? || ' [row %')
                    ORDER BY id ASC LIMIT 1
                """, (uid, doc_id, base, base))
                row = cur.fetchone()
                if not row:
                    continue

                cur.execute("""
                    SELECT text FROM chunks
                    WHERE owner_id=? AND doc_id=? AND element_type='table_row'
                    AND (section_path=? OR section_path LIKE ? || ' [row %')
                    ORDER BY id ASC LIMIT 2
                """, (uid, doc_id, row["section_path"], row["section_path"]))
                rows = cur.fetchall() or []
                highlights = []
                for r in rows:
                    first = (r["text"] or "").split("\n")[0]
                    if first:
                        highlights.append(" — ".join([c.strip() for c in first.split(" | ") if c.strip()]))

                attrs_json = row["attrs"] if row else None
                display = _compose_display_from_attrs(attrs_json, row["section_path"], highlights[0] if highlights else None)
                display = _strip_table_prefix(display)

                # попробуем вытащить номер для stats
                num, _ = _parse_table_title(display)
                stats = None
                if num:
                    try:
                        stats = analyze_table_by_num(uid, doc_id, num, max_series=6)
                    except Exception:
                        stats = None

                desc_cards.append({
                    "num": num,
                    "display": display,
                    "where": {"page": row["page"], "section_path": row["section_path"]},
                    "highlights": highlights,
                    "stats": stats,
                })
            con.close()

        # Общий список таблиц с кратким описанием
        facts["tables"]["describe"] = desc_cards

        # describe по конкретным номерам + точные расчеты (перезаписываем describe, если запрос конкретный)
        if intents.get("tables", {}).get("describe"):
            desc_cards = []
            con = get_conn()
            cur = con.cursor()
            for num in intents["tables"]["describe"]:

                like1 = f'%\"caption_num\": \"{num}\"%'
                like2 = f'%\"label\": \"{num}\"%'
                cur.execute(
                    """
                    SELECT page, section_path, attrs FROM chunks
                    WHERE owner_id=? AND doc_id=? AND element_type IN ('table','table_row')
                      AND (attrs LIKE ? OR attrs LIKE ?)
                    ORDER BY id ASC LIMIT 1
                    """,
                    (uid, doc_id, like1, like2),
                )
                row = cur.fetchone()

                if not row:
                    cur.execute(
                        """
                        SELECT page, section_path, attrs FROM chunks
                        WHERE owner_id=? AND doc_id=? AND element_type IN ('table','table_row')
                          AND section_path LIKE ? COLLATE NOCASE
                        ORDER BY id ASC LIMIT 1
                        """,
                        (uid, doc_id, f'%Таблица {num}%'),
                    )
                    row = cur.fetchone()

                if not row:
                    continue

                attrs_json = row["attrs"] if row else None
                # 1–2 первых строки как highlights
                cur.execute(
                    """
                    SELECT text FROM chunks
                    WHERE owner_id=? AND doc_id=? AND element_type='table_row'
                      AND (section_path=? OR section_path LIKE ? || ' [row %')
                    ORDER BY id ASC LIMIT 2
                    """,
                    (uid, doc_id, row["section_path"], row["section_path"]),
                )
                rows = cur.fetchall()
                highlights = []
                for r in rows or []:
                    first_line = (r["text"] or "").split("\n")[0]
                    if first_line:
                        highlights.append(" — ".join([c.strip() for c in first_line.split(" | ") if c.strip()]))

                base = row["section_path"]
                first_row_text = highlights[0] if highlights else None
                display = _compose_display_from_attrs(attrs_json, base, first_row_text)
                display = _strip_table_prefix(display)

                # НОВОЕ: точная аналитика по таблице
                stats = None
                try:
                    stats = analyze_table_by_num(uid, doc_id, num, max_series=6)
                except Exception:
                    stats = None

                desc_cards.append({
                    "num": num,
                    "display": display,
                    "where": {"page": row["page"], "section_path": row["section_path"]},
                    "highlights": highlights,
                    "stats": stats,
                })
            con.close()

            facts["tables"]["describe"] = desc_cards

            # запомним эти номера таблиц как «последние упомянутые»
            try:
                LAST_REF.setdefault(uid, {})["table_nums"] = [
                    str(c["num"]) for c in desc_cards if c.get("num")
                ]
            except Exception:
                pass

    # ----- Рисунки -----
    if intents["figures"]["want"]:
        f_limit = int(intents.get("figures", {}).get("limit", 10))
        lst = _list_figures_db(uid, doc_id, limit=f_limit)
        figs_block = {
            "count": int(lst.get("count") or 0),
            "list": list(lst.get("list") or []),
            "more": int(lst.get("more") or 0),
            "describe_lines": [],
            "describe_cards": [],
        }

        nums = list(intents.get("figures", {}).get("describe") or [])
        if nums:
            try:
                # ⚙️ Берём карточки ТОЛЬКО по указанным номерам (5 и 6 → только 5 и 6)
                cards = describe_figures_by_numbers(
                    uid,
                    doc_id,
                    nums,
                    sample_chunks=2,
                    use_vision=True,
                    lang="ru",
                    vision_first_image_only=True,
                ) or []
                logging.info(
                    "FIG: получено %d рисунков для номеров %s",
                    len(cards),
                    ", ".join(map(str, nums)),
                )

                if not cards:
                    figs_block["describe_lines"] = ["Данного рисунка нет в работе."]
                    figs_block["describe_cards"] = []
                else:
                    # Основное для answer_builder: полный набор describe_cards
                    figs_block["describe_cards"] = cards

                    # Чтобы в [Figures]/list не попадали лишние рисунки,
                    # если запрос был только про конкретные номера.
                    figs_block["list"] = [
                        (c.get("display")
                         or f"Рисунок {c.get('num') or ''}".strip())
                        for c in cards
                    ]
                    figs_block["count"] = len(figs_block["list"])
                    figs_block["more"] = 0

                    # Для обратной совместимости — короткие describe_lines
                    lines = []
                    for c in cards:
                        disp = c.get("display") or "Рисунок"
                        vis = (c.get("vision") or {}).get("description", "") or ""
                        vis_clean = vis.strip()
                        low_vis = vis_clean.lower()
                        if ("описание не распознано" in low_vis
                                or "содержимое изображения" in low_vis):
                            vis_clean = ""
                        hint = "; ".join([h for h in (c.get("highlights") or []) if h])
                        if vis_clean:
                            lines.append(f"{disp}: {vis_clean}")
                        elif hint:
                            lines.append(f"{disp}: {hint}")
                        else:
                            lines.append(disp)
                    figs_block["describe_lines"] = lines[:25]
            except Exception as e:
                figs_block["describe_lines"] = [f"Не удалось описать рисунки: {e}"]
                figs_block["describe_cards"] = []

        facts["figures"] = figs_block


    # ----- Источники -----
    if intents["sources"]["want"]:
        con = get_conn()
        cur = con.cursor()
        has_type = _table_has_columns(con, "chunks", ["element_type", "attrs"])
        items: list[str] = []

        if has_type:
            cur.execute(
                "SELECT text FROM chunks WHERE owner_id=? AND doc_id=? AND element_type='reference' ORDER BY id ASC",
                (uid, doc_id),
            )
            items = [(r["text"] or "").strip() for r in cur.fetchall()]

        if not any(items):
            cur.execute(
                """
                SELECT element_type, section_path, text
                FROM chunks
                WHERE owner_id=? AND doc_id=?
                ORDER BY page ASC, id ASC
                """,
                (uid, doc_id),
            )
            raw = []
            for r in cur.fetchall():
                sec = (r["section_path"] or "").lower()
                if not any(k in sec for k in ("источник", "литератур", "библиограф", "reference", "bibliograph")):
                    continue
                et = (r["element_type"] or "").lower()
                if et in ("heading", "table", "figure", "table_row"):
                    continue
                t = (r["text"] or "").strip()
                if t:
                    raw.append(t)

            seen = set()
            items = []
            for t in raw:
                k = re.sub(r"\s+", " ", t).strip().rstrip(".").lower()
                if len(k) < 5 or k in seen:
                    continue
                seen.add(k)
                items.append(t)

        con.close()

        s_limit = int(intents.get("sources", {}).get("limit", 25))
        facts["sources"] = {
            "count": len(items),
            "list": items[:s_limit],
            "more": max(0, len(items) - s_limit),
        }


    # ----- Практическая часть -----
    if intents.get("practical"):
        facts["practical_present"] = _has_practical_part(uid, doc_id)

    # ----- Summary -----
    if intents.get("summary"):
        s = overview_context(uid, doc_id, max_chars=6000) or _first_chunks_context(uid, doc_id, n=12, max_chars=6000)
        if s:
            facts["summary_text"] = s

    # ----- Общий контекст / цитаты -----
    # app/bot.py (_gather_facts: общий контекст / цитаты)
    if intents.get("general_question"):
        question_text = intents.get("general_question") or ""

        vb = verbatim_find(uid, doc_id, question_text, max_hits=3)

        cov = retrieve_coverage(
            owner_id=uid,
            doc_id=doc_id,
            question=question_text,
        )
        ctx = ""
        if cov and cov.get("snippets"):
            ctx = build_context_coverage(
                cov["snippets"],
                items_count=len(cov.get("items") or []) or None,
            )

        if not ctx:
            ctx = best_context(uid, doc_id, question_text, max_chars=6000)
        if not ctx:
            hits = retrieve(uid, doc_id, question_text, top_k=12)
            if hits:
                ctx = build_context(hits)
        if not ctx:
            ctx = _first_chunks_context(uid, doc_id, n=12, max_chars=6000)

        if ctx:
            facts["general_ctx"] = ctx
        if vb:
            facts["verbatim_hits"] = vb
        if cov and cov.get("items"):
            facts["coverage"] = {"items": cov["items"]}
            facts["general_subitems"] = [
                {"id": i + 1, "ask": s} if isinstance(s, str) else s
                for i, s in enumerate(cov["items"])
            ]

        # --- [VISION] второй проход: числа из диаграмм/картинок (подмешиваем в контекст) ---
        try:
            vision_block = ""
            if getattr(Cfg, "vision_active", lambda: False)():
                # 1) берём топ-хиты специально для картинок
                hits_v = retrieve(uid, doc_id, question_text, top_k=10) or []

                # 1а) если в хитах есть chart_data (DOCX-диаграммы) — используем точные числа, без vision
                chart_lines: list[str] = []
                for h in hits_v:
                    attrs = (h.get("attrs") or {})
                    cd = attrs.get("chart_data")
                    if cd:
                        # переиспользуем уже написанный парсер: упакуем в attrs-json
                        try:
                            cd_list, _, _ = _parse_chart_data(json.dumps({"chart_data": cd}))
                        except Exception:
                            cd_list = None
                        if cd_list:
                            chart_lines.append(_format_chart_values(cd_list))

                if chart_lines:
                    vision_block = "\n".join(chart_lines[:3])
                else:
                    # 2) иначе — отправляем 1–3 картинки в новый vision_analyzer
                    img_paths = _pick_images_from_hits(
                        hits_v,
                        limit=getattr(Cfg, "VISION_MAX_IMAGES_PER_REQUEST", 3),
                    )
                    if img_paths and va_analyze_figure:
                        chunks: list[str] = []
                        hint = question_text[:300]
                        for p in img_paths:
                            try:
                                res = va_analyze_figure(p, caption_hint=hint, lang="ru")
                            except Exception:
                                continue

                            text_block = ""
                            if isinstance(res, dict):
                                pairs = res.get("data") or []
                                text_block = (res.get("text") or "").strip() or _pairs_to_bullets(pairs)
                            else:
                                text_block = (str(res) or "").strip()

                            if text_block:
                                chunks.append("[Text on image]\n" + text_block)

                        if chunks:
                            vision_block = "\n\n".join(chunks)
                        elif FIG_STRICT:
                            vision_block = "[No precise data]"

            if vision_block:
                prev = facts.get("general_ctx") or ""
                glue = ("\n\n" if prev else "")
                facts["general_ctx"] = (prev + glue + vision_block)
        except Exception:
            # не ломаем основной ответ, если vision дал сбой
            pass

        # --- [/VISION] ---


    # логируем маленький срез фактов (без огромных текстов)
    log_snapshot = dict(facts)
    if "general_ctx" in log_snapshot and isinstance(log_snapshot["general_ctx"], str):
        log_snapshot["general_ctx"] = log_snapshot["general_ctx"][:300] + "…" if len(log_snapshot["general_ctx"]) > 300 else log_snapshot["general_ctx"]
    if "summary_text" in log_snapshot and isinstance(log_snapshot["summary_text"], str):
        log_snapshot["summary_text"] = log_snapshot["summary_text"][:300] + "…" if len(log_snapshot["summary_text"]) > 300 else log_snapshot["summary_text"]
    logging.debug("FACTS: %s", json.dumps(log_snapshot, ensure_ascii=False))
    return facts


def _strip_unwanted_sections(s: str) -> str:
    """
    Аккуратно убираем служебные хвосты вроде заголовков
    «Чего не хватает: …», но не допускаем, чтобы ответ
    обнулился полностью — в этом случае возвращаем исходный текст.
    """
    if not s:
        return s

    original = s

    # вырезаем заголовок + абзац(ы) до следующего пустого разрыва
    pat = re.compile(r"(?mis)^\s*(?:чего|что)\s+не\s+хватает\s*:.*?(?:\n\s*\n|\Z)")
    s = pat.sub("", s)
    # отдельные строки-метки
    s = re.sub(r"(?mi)^\s*не\s+хватает\s*:.*$", "", s)

    s = s.strip()
    # если после зачистки всё исчезло — лучше вернуть исходный ответ,
    # чем отдать пользователю пустоту
    return s or original.strip()


# ------------------------------ FULLREAD: модель читает весь файл ------------------------------

def _full_document_text(owner_id: int, doc_id: int, *, limit_chars: int | None = None) -> str:
    """Склеиваем ВЕСЬ текст из chunks (page ASC, id ASC)."""
    con = get_conn()
    cur = con.cursor()
    cur.execute(
        "SELECT text FROM chunks WHERE owner_id=? AND doc_id=? ORDER BY page ASC, id ASC",
        (owner_id, doc_id),
    )
    rows = cur.fetchall() or []
    con.close()

    parts = []
    total = 0
    for r in rows:
        t = (r["text"] or "").strip()
        if not t:
            continue
        if limit_chars is not None and total + len(t) > limit_chars:
            remaining = max(0, limit_chars - total)
            if remaining > 0:
                parts.append(t[:remaining])
                total += remaining
            break
        parts.append(t)
        total += len(t)
    return "\n\n".join(parts)

def _fullread_try_answer(uid: int, doc_id: int, q_text: str) -> str | None:
    """
    DIRECT: отдаём модели целиком весь текст документа как единый контекст.
    Если документ слишком большой — возвращаем None (уйдём в иной режим).
    """
    if getattr(Cfg, "FULLREAD_MODE", "off") != "direct":
        return None

    _limit = int(getattr(Cfg, "DIRECT_MAX_CHARS", 80000))
    full_text = _full_document_text(uid, doc_id, limit_chars=_limit + 1)
    if not full_text.strip():
        return None

    if len(full_text) > _limit:
        return None

    system_prompt = (
        "Ты ассистент по дипломным работам. Тебе дан ПОЛНЫЙ текст ВКР/документа.\n"
        "Отвечай строго по этому тексту, без внешних фактов. Не добавляй разделов вида "
        "«Чего не хватает» и не проси дополнительные данные.\n"
        "Если вопрос про таблицы/рисунки — используй подписи и ближайший текст; не придумывай номера/значения.\n"
        "Если запрошенного рисунка/таблицы нет в тексте — ответь: «данного рисунка нет в работе».\n"
        "Если объект есть, но он в плохом качестве/нечитаем — ответь: «Рисунок плохого качества, не могу проанализировать», "
        "и добавь краткую подпись/контекст из текста. Цитируй коротко, без ссылок на страницы."
    )

    verbosity = _detect_verbosity(q_text)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": f"[Документ — полный текст]\n{full_text}"},
        {"role": "user", "content": f"{q_text}\n\n{_verbosity_addendum(verbosity)}"},
    ]

    try:
        answer = chat_with_gpt(messages, temperature=0.2, max_tokens=FINAL_MAX_TOKENS)
        return (answer or "").strip() or None
    except Exception as e:
        logging.exception("fullread direct failed: %s", e)
        return None


def _fullread_collect_sections(uid: int, doc_id: int, *, max_sections: int = 800) -> List[str]:
    """
    Секции для итеративного режима: собираем блоки текста по section_path в порядке следования.
    """
    con = get_conn()
    cur = con.cursor()
    cur.execute(
        "SELECT section_path, text FROM chunks WHERE owner_id=? AND doc_id=? ORDER BY page ASC, id ASC",
        (uid, doc_id)
    )
    rows = cur.fetchall() or []
    con.close()

    out: List[str] = []
    cur_sec = None
    buf: List[str] = []

    def _flush():
        if buf:
            text = "\n".join([t for t in buf if t.strip()]).strip()
            if text:
                title = f"[{cur_sec}]" if cur_sec else ""
                out.append(f"{title}\n{text}" if title else text)
        buf.clear()


    for r in rows:
        sec = r["section_path"] or ""
        t = (r["text"] or "").strip()
        if not t:
            continue
        if cur_sec is None:
            cur_sec = sec
        if sec != cur_sec:
            _flush()
            cur_sec = sec
        buf.append(t)
        if len(out) >= max_sections:
            break
    _flush()
    return out[:max_sections]


def _group_for_steps(sections: Iterable[str], per_step_chars: int, max_steps: int) -> List[str]:
    """Группируем секции в батчи по символам (для map-шага)."""
    batches: List[str] = []
    cur: List[str] = []
    cur_len = 0
    for s in sections:
        if cur_len + len(s) + 1 > per_step_chars and cur:
            batches.append("\n\n".join(cur))
            cur, cur_len = [], 0
            if len(batches) >= max_steps:
                break
        cur.append(s)
        cur_len += len(s) + 1
    if cur and len(batches) < max_steps:
        batches.append("\n\n".join(cur))
    return batches[:max_steps]

def _map_extract(uid: int, doc_id: int, question: str, chunk_text: str, *, map_tokens: int) -> str:
    """Один map-вызов: извлекаем только релевантные факты/цитаты из фрагмента."""
    sys_map = (
        "Ты ассистент-экстрактор. Тебе дан фрагмент диплома и вопрос. "
        "Извлеки ТОЛЬКО факты и мини-цитаты, относящиеся к вопросу. "
        "Если встречаются таблицы — включай их названия и 1–2 ключевые строки с числами "
        "(сохраняй порядок и значения). Формат: буллеты."
    )
    return chat_with_gpt(
        [
            {"role": "system", "content": sys_map},
            {"role": "assistant", "content": f"[Фрагмент документа]\n{chunk_text}"},
            {"role": "user", "content": f"Вопрос: {question}\nСделай короткую выжимку (буллеты)."},
        ],
        temperature=0.1,
        max_tokens=max(120, int(map_tokens)),
    )

def _iterative_fullread_build_messages(uid: int, doc_id: int, question: str) -> Tuple[Optional[list], Optional[str]]:
    """
    Собираем map-выжимки синхронно, возвращаем reduce-сообщения для стрима
    ИЛИ итоговый ответ (если что-то пошло не так).
    """
    per_step = int(getattr(Cfg, "FULLREAD_STEP_CHARS", 14000))
    max_steps = int(getattr(Cfg, "FULLREAD_MAX_STEPS", 2))
    map_tokens = int(getattr(Cfg, "DIGEST_TOKENS_PER_SECTION", 300))

    sections = _fullread_collect_sections(uid, doc_id)
    if not sections:
        return None, "Не удалось прочитать документ секциями."

    batches = _group_for_steps(sections, per_step_chars=per_step, max_steps=max_steps)
    if not batches:
        return None, "Не удалось сформировать шаги чтения документа."

    digests: List[str] = []
    for b in batches:
        try:
            digests.append(_map_extract(uid, doc_id, question, b, map_tokens=map_tokens))
        except Exception as e:
            logging.exception("map extract failed: %s", e)
            digests.append(b[:800])

    # вместо техничных [MAP 1] используем более нейтральные метки
    joined = "\n\n".join([f"[Фрагмент {i+1}]\n{d}" for i, d in enumerate(digests)])
    ctx = joined[: int(getattr(Cfg, "FULLREAD_CONTEXT_CHARS", 9000))]

    sys_reduce = (
        "Ты репетитор по ВКР. Ниже — короткие факты из разных частей документа. "
        "Собери из них связный ответ на вопрос. Не выдумывай новых цифр/таблиц и не добавляй разделов "
        "про «чего не хватает». Отвечай только по имеющимся данным. "
        "Не придумывай предметную область и термины (например, продажи, клиенты, выручка, маркетинг и т.п.), "
        "если они не присутствуют в фактах/цитатах.\n"
        "Если запрошенного рисунка/таблицы нет в тексте — сформулируй кратко: «данного рисунка нет в работе». "
        "Если объект есть, но он нечитаем, дай: «Рисунок плохого качества, не могу проанализировать», "
        "и добавь подпись/контекст из текста. "
        "В своём ответе не ссылайся на технические метки вроде «фрагмент 1» и не используй слово «выжимка»."
    )



    verbosity = _detect_verbosity(question)
    messages = [
        {"role": "system", "content": sys_reduce},
        {"role": "assistant", "content": f"Сводные факты из документа:\n{ctx}"},
        {"role": "user", "content": f"{question}\n\n{_verbosity_addendum(verbosity)}"},
    ]
    return messages, None


# ------------------------------ загрузка файла ------------------------------

@dp.message(F.document)
async def handle_doc(m: types.Message):
    uid = ensure_user(str(m.from_user.id))
    doc = m.document

    # 0) FSM: фиксируем, что начали скачивание
    start_downloading(uid)
    await _send(m, Cfg.MSG_ACK_DOWNLOADING)

    # 1) скачиваем файл целиком (без обрезки)
    from io import BytesIO

    file = await bot.get_file(doc.file_id)
    buf = BytesIO()
    await bot.download_file(file.file_path, destination=buf)
    buf.seek(0)
    data = buf.read()  # здесь все байты файла как есть

    # 2) сохраняем на диск (единственный источник правды для оркестратора)
    filename = safe_filename(f"{m.from_user.id}_{doc.file_name}")
    path = save_upload(data, filename, Cfg.UPLOAD_DIR)
    await _send(m, Cfg.MSG_ACK_INDEXING)

    # 3) обёртка индексатора под сигнатуру оркестратора (замыкаем uid)
    def _indexer_fn(doc_id: int, file_path: str, kind: str) -> dict:
        sections = _parse_by_ext(file_path)
        sections = enrich_sections(sections, doc_kind=os.path.splitext(file_path)[1].lower().strip("."))
        # sanity-check на «пустые» файлы
        if sum(len(s.get("text") or "") for s in sections) < 500 and not any(
            s.get("element_type") in ("table", "table_row", "figure") for s in sections
        ):
            raise RuntimeError("Похоже, файл не содержит «живого» текста/структур.")
        # индексация «как раньше»
        delete_document_chunks(doc_id, uid)
        index_document(uid, doc_id, sections)
        invalidate_cache(uid, doc_id)
        update_document_meta(doc_id, layout_profile=_current_embedding_profile())
        return {"sections_count": len(sections)}

    # 4) запускаем оркестратор (он сам: идемпотентность, INDEXING, READY/IDLE, версия индексатора)
    try:
        result = ingest_document(
            user_id=uid,
            file_path=path,
            kind=infer_doc_kind(doc.file_name),
            file_uid=getattr(doc, "file_unique_id", None),
            content_sha256=sha256_bytes(data),
            indexer_fn=_indexer_fn,
        )
    except Exception as e:
        logging.exception("ingest failed: %s", e)
        await _send(m, Cfg.MSG_INDEX_FAILED + f" Подробности: {e}")
        return

    doc_id = int(result["doc_id"])
    ACTIVE_DOC[uid] = doc_id
    set_user_active_doc(uid, doc_id)

    # NEW: построить индекс рисунков из исходного файла и закэшировать (старый путь)
    try:
        if fig_index_document is not None:
            FIG_INDEX[doc_id] = fig_index_document(path)
    except Exception as e:
        logging.exception("figures indexing failed: %s", e)


        # NEW: построить ЕДИНЫЙ OOXML-индекс (главы/рисунки/таблицы/источники) без LibreOffice
    # NEW: построить ЕДИНЫЙ OOXML-индекс (главы/рисунки/таблицы/источники) без LibreOffice
    try:
        idx_oox = oox_build_index(path)
        OOXML_INDEX[doc_id] = idx_oox
        #persist под ID документа из БД — чтобы _ooxml_get_index работал после рестарта процесса
        try:
            os.makedirs(os.path.join("runtime", "indexes"), exist_ok=True)
            with open(os.path.join("runtime", "indexes", f"{doc_id}.json"), "w", encoding="utf-8") as f:
                json.dump(idx_oox, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    except Exception as e:
        logging.exception("ooxml build_index failed: %s", e)


    # 5) READY: сообщаем и обрабатываем ...
    await _send(
        m,
        (f"Этот файл уже был загружен как документ #{doc_id}. " if result.get("reused") else "") + Cfg.MSG_READY
    )

    caption = (m.caption or "").strip()
    if caption:
        await respond_with_answer(m, uid, doc_id, caption)

    # авто-дренаж очереди ожидания (без дублей одинаковых вопросов)
    try:
        queued = dequeue_all_pending_queries(uid)
        for item in queued:
            q = (item.get("text") or "").strip()
            if not q:
                continue
            # если вопрос из очереди совпадает с подписью к файлу — не отвечаем второй раз
            if caption and q.strip() == caption:
                continue
            await respond_with_answer(m, uid, doc_id, q)
            await asyncio.sleep(0)  # не блокируем цикл
    except Exception as e:
        logging.exception("drain pending queue failed: %s", e)



# ------------------------------ основной ответчик ------------------------------

async def _answer_with_model_extra(m: types.Message, uid: int, base_question: str) -> None:
    """
    Ответ без привязки к конкретному документу — общий совет от [модель].

    Используется, когда в тексте работы не нашлось фактов по вопросу
    и пользователь подтвердил, что хочет такой ответ.
    """
    if not (chat_with_gpt or chat_with_gpt_stream):
        await _send(
            m,
            "Сейчас могу отвечать только по тексту документа, режим [модель] недоступен."
        )
        return

    base_question = (base_question or "").strip()
    if not base_question:
        await _send(m, "Не удалось восстановить исходный вопрос. Сформулируй его ещё раз, пожалуйста.")
        return

    system_prompt = (
        "Ты помощник по учёбе. В ЭТОМ ответе ты не опираешься на текст диплома пользователя, "
        "а используешь только свои общие знания и здравый смысл. "
        "Сразу в начале ответа укажи тег '[модель] ' и дальше отвечай простым, понятным языком."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": base_question},
    ]

    try:
        if STREAM_ENABLED and chat_with_gpt_stream is not None:
            stream = chat_with_gpt_stream(messages, temperature=0.3, max_tokens=FINAL_MAX_TOKENS)  # type: ignore
            await _stream_to_telegram(m, stream)
            return

        answer = chat_with_gpt(messages, temperature=0.3, max_tokens=FINAL_MAX_TOKENS)
    except Exception as e:
        logging.exception("model-extra answer failed: %s", e)
        await _send(
            m,
            "Не получилось получить дополнительный ответ от [модель]. Попробуй переформулировать вопрос."
        )
        return

    answer = (answer or "").strip()
    if not answer:
        await _send(
            m,
            "Не получилось получить дополнительный ответ от [модель]. Попробуй переформулировать вопрос."
        )
        return

    if not answer.startswith("[модель]"):
        answer = "[модель] " + answer

    await _send(m, answer)

async def _answer_with_model_extra_table(
    m: types.Message,
    uid: int,
    doc_id: int,
    base_question: str,
    ctx_tables: str,
    nums: list[str],
) -> None:
    """
    Расширенный ответ от [модель] по таблице(таблицам):
    модель видит сырые данные таблиц из OOXML и может на них опираться,
    добавляя общую теорию, но НЕ меняя сами числа.
    """
    if not (chat_with_gpt or chat_with_gpt_stream):
        await _send(
            m,
            "Сейчас могу отвечать только по тексту документа, режим [модель] недоступен."
        )
        return

    ctx_tables = (ctx_tables or "").strip()
    if not ctx_tables:
        # на всякий случай — фолбэк в общий режим
        await _answer_with_model_extra(m, uid, base_question)
        return

    nums = [str(n).strip() for n in (nums or []) if str(n).strip()]
    nums_str = ", ".join(nums) if nums else "этим таблицам"

    base_question = (base_question or "").strip()
    if not base_question:
        base_question = f"Подробно объясни и интерпретируй данные по таблице(таблицам) {nums_str}."

    system_prompt = (
        "Ты помощник по учёбе. В ЭТОМ ответе ты опираешься на данные таблиц из диплома пользователя "
        "(они переданы ниже в машинно-читаемом виде). "
        "Используй эти числа как источник истины: не меняй их и не придумывай другие значения. "
        "При этом можешь дополнять интерпретацию общими теоретическими сведениями по теме. "
        "Сразу в начале ответа укажи тег '[модель] '."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": f"[Данные таблиц из диплома]\n{ctx_tables}"},
        {"role": "user", "content": base_question},
    ]

    try:
        if STREAM_ENABLED and chat_with_gpt_stream is not None:
            stream = chat_with_gpt_stream(
                messages,
                temperature=0.3,
                max_tokens=FINAL_MAX_TOKENS,
            )  # type: ignore
            await _stream_to_telegram(m, stream)
            return

        answer = chat_with_gpt(
            messages,
            temperature=0.3,
            max_tokens=FINAL_MAX_TOKENS,
        )
    except Exception as e:
        logging.exception("model-extra-table answer failed: %s", e)
        await _send(
            m,
            "Не получилось получить расширенный ответ по таблице. Попробуй переформулировать вопрос."
        )
        return

    answer = (answer or "").strip()
    if not answer:
        await _send(
            m,
            "Не получилось получить расширенный ответ по таблице. Попробуй переформулировать вопрос."
        )
        return

    if not answer.startswith("[модель]"):
        answer = "[модель] " + answer

    await _send(m, answer)

async def respond_with_answer(m: types.Message, uid: int, doc_id: int, q_text: str):
    q_text = (q_text or "").strip()
    orig_q_text = q_text  # запомним исходную формулировку до подстановок
    logger.info(
        "ANSWER: new question (uid=%s, doc_id=%s, len=%d): %r",
        uid,
        doc_id,
        len(q_text or ""),
        q_text,
    )
    if not q_text:
        logger.warning(
            "ANSWER: empty question from uid=%s, doc_id=%s",
            uid,
            doc_id,
        )
        await _send(m, "Вопрос пустой. Напишите, что именно вас интересует по ВКР.")
        return

    viol = safety_check(q_text)
    if viol:
        logger.warning(
            "ANSWER: safety_check blocked question (uid=%s, doc_id=%s): %s",
            uid,
            doc_id,
            viol,
        )
        await _send(m, viol + " Задайте корректный вопрос по ВКР.")
        return

    logger.debug(
        "ANSWER: before GOST check (uid=%s, doc_id=%s)",
        uid,
        doc_id,
    )
    if await _maybe_run_gost(m, uid, doc_id, q_text):
        logger.info(
            "ANSWER: handled by GOST validator (uid=%s, doc_id=%s)",
            uid,
            doc_id,
        )
        return

    # для коротких реплик вида «опиши подробнее», «расскажи про него»
    q_text = _expand_with_last_referent(uid, q_text)

        # Примеры: "опиши таблицу 4", "что показывает таблица 2.3", "сделай выводы по таблице 4"
    if _is_pure_table_request(q_text):
        verbosity = _detect_verbosity(q_text)
        base_text = (orig_q_text or "")
        mode = "more" if _FOLLOWUP_MORE_RE.search(base_text) else "normal"
        logger.info(
            "ANSWER: pure table request detected (uid=%s, doc_id=%s, mode=%s)",
            uid,
            doc_id,
            mode,
        )
        handled = await _answer_table_query(
            m, uid, doc_id, q_text, verbosity=verbosity, mode=mode
        )
        logger.info(
            "ANSWER: _answer_table_query finished (uid=%s, doc_id=%s, handled=%s)",
            uid,
            doc_id,
            handled,
        )
        if handled:
            return
        else:
            logger.info(
                "ANSWER: table pipeline did not handle request, falling back to general pipeline "
                "(uid=%s, doc_id=%s)",
                uid,
                doc_id,
            )


        # если _answer_table_query не смог ответить (таблица не нашлась в OOXML/картинке),


    # быстрый путь для запросов про рисунки
    if _is_pure_figure_request(q_text):
        verbosity = _detect_verbosity(q_text)
        logger.info(
            "ANSWER: pure figure request detected (uid=%s, doc_id=%s)",
            uid,
            doc_id,
        )
        handled = await _answer_figure_query(
            m,
            uid,
            doc_id,
            q_text,
            verbosity=verbosity,
        )
        logger.info(
            "ANSWER: _answer_figure_query finished (uid=%s, doc_id=%s, handled=%s)",
            uid,
            doc_id,
            handled,
        )
        if handled:
            return

    # Если одновременно упоминаются главы/таблицы — даём это
    # обработать мультийнтентному пайплайну ниже.
    if (
        _ALL_FIGS_HINT.search(q_text or "")
        and not _SECTION_NUM_RE.search(q_text or "")
        and not _TABLE_ANY.search(q_text or "")
    ):
        meta = _list_figures_db(uid, doc_id, limit=999999)
        total = int(meta["count"])
        if total == 0:
            await _send(m, "В работе не найдено ни одного рисунка.")
            return
        # партиями по 8–12 номеров
        nums = []
        for disp in meta["list"]:
            # из "Рисунок 2.1 — ..." вытащим "2.1" (если есть)
            mnum = re.search(r"(?i)\bрисунок\s+([A-Za-zА-Яа-я]?\s*\d+(?:[.,]\d+)*)\b", disp)
            if mnum:
                nums.append(mnum.group(1).replace(" ", "").replace(",", "."))
        batch = nums[:8] or nums[:12]
        # карточки (используем только для текста; картинки в чат больше не шлём)
        cards = []
        try:
            cards = describe_figures_by_numbers(uid, doc_id, batch, sample_chunks=1, use_vision=False, lang="ru") or []
        except Exception:
            cards = []

        # затем — связный текст по каждому рисунку: prefer vision_analyzer
        lines = []

        if va_analyze_figure and cards:
            for c in cards:
                disp = c.get("display") or f"Рисунок {c.get('num') or ''}".strip()
                imgs = c.get("images") or []
                hint = (c.get("highlights") or [None])[0]
                if not imgs:
                    continue
                try:
                    res = va_analyze_figure(imgs[0], caption_hint=hint, lang="ru")
                    if isinstance(res, dict):
                        text_block = (res.get("text") or "").strip() or _pairs_to_bullets(res.get("data") or [])
                    else:
                        text_block = (str(res) or "").strip()
                except Exception:
                    text_block = ""
                if not text_block:
                    # фолбэк — старый summarizer больше не используем
                    text_block = ""
                if text_block:
                    # это текст по изображению (OCR/описание)
                    lines.append(f"[Text on image] **{disp}**\n\n{text_block}")
                else:
                    # строгий режим — явно говорим, что точных данных нет
                    if FIG_STRICT:
                        lines.append(f"[No precise data] **{disp}**")

        suffix = (f"\n\nПоказана первая партия из {len(batch)} / {total}." if total > len(batch) else "")
        if lines:
            await _send(m, "\n\n".join(lines) + suffix)
        else:
            # финальный фолбэк, если анализатор недоступен
            await _send(m, "Не удалось описать рисунки." + suffix)
        return


    # NEW: если в вопросе явно указан раздел/пункт — запоминаем его как последний
    m_area = _SECTION_NUM_RE.search(q_text)
    if m_area:
        try:
            area = (m_area.group(1) or "").replace(" ", "").replace(",", ".")
            LAST_REF.setdefault(uid, {})["area"] = area
        except Exception:
            pass

        # --- Определяем интенты заранее
    intents = detect_intents(q_text)

    # Чистый запрос про конкретные рисунки (нет секций/таблиц/источников/общего вопроса)
    pure_figs = intents["figures"]["want"] and not (
        intents["tables"]["want"] or intents["sources"]["want"] or
        intents.get("summary") or intents.get("general_question") or
        _SECTION_NUM_RE.search(q_text)
    )

    # NEW: явная обработка «по пункту/разделу/главе X.Y» (но только для ЧИСТЫХ запросов)
    m_sec = _SECTION_NUM_RE.search(q_text)
    sec = None
    if m_sec:
        raw_sec = (m_sec.group(1) or "").strip()
        raw_sec = re.sub(r"^[A-Za-zА-Яа-я]\s+(?=\d)", "", raw_sec)
        sec = raw_sec.replace(" ", "").replace(",", ".")

    # Строгий секционный ответ — только если запрос не смешанный
    if sec and _is_pure_section_request(q_text, intents):
        verbosity = _detect_verbosity(q_text)
        ctx = _section_context(uid, doc_id, sec, max_chars=9000)
        if ctx:
            base_sys = (
                "Ты репетитор по ВКР. Ниже — контекст ТОЛЬКО по одному пункту/главе диплома.\n"
                "Отвечай строго по этому тексту: не добавляй внешних фактов, не придумывай новых положений "
                "и не пересказывай то, чего в фрагменте нет. Если информации недостаточно, честно напиши, "
                "что данных в этом пункте не хватает для полного ответа."
            )
            if verbosity == "brief":
                sys_prompt = base_sys + " Нужна КРАТКАЯ выжимка."
                user_prompt = (
                    f"Вопрос пользователя: {q_text}\n\n"
                    f"Сделай краткую выжимку по пункту {sec}. {_verbosity_addendum('brief')}"
                )
            elif verbosity == "detailed":
                sys_prompt = base_sys + " Нужен ПОДРОБНЫЙ разбор."
                user_prompt = (
                    f"Вопрос пользователя: {q_text}\n\n"
                    f"Сделай подробный разбор по пункту {sec}. {_verbosity_addendum('detailed')}"
                )
            else:
                sys_prompt = base_sys + " Ответь по делу, без лишних рассуждений."
                user_prompt = (
                    f"Вопрос пользователя: {q_text}\n\n"
                    f"Ответь по пункту {sec}. {_verbosity_addendum('normal')}"
                )

            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "assistant", "content": f"[Контекст по пункту {sec}]\n{ctx}"},
                {"role": "user", "content": user_prompt},
            ]

            if STREAM_ENABLED and chat_with_gpt_stream is not None:
                try:
                    stream = chat_with_gpt_stream(messages, temperature=0.2, max_tokens=FINAL_MAX_TOKENS)  # type: ignore
                    await _stream_to_telegram(m, stream)
                    return
                except Exception as e:
                    logging.exception("section summary stream failed: %s", e)
            try:
                ans = chat_with_gpt(messages, temperature=0.2, max_tokens=FINAL_MAX_TOKENS)
                if ans:
                    await _send(m, _strip_unwanted_sections(ans))
                    return
            except Exception as e:
                logging.exception("section summary non-stream failed: %s", e)
                # не возвращаемся — пусть пойдёт обычный пайплайн ниже, если что-то сломалось
        else:
            await _send(m, f"Пункт {sec} не найден в индексе документа.")
            return



    # ====== FULLREAD: auto ======
    fr_mode = getattr(Cfg, "FULLREAD_MODE", "off")
    # FULLREAD(auto) включаем только для общих вопросов по содержанию,
    # чтобы не перебивать спец-логики по таблицам/рисункам/источникам.
    if (
        fr_mode == "auto"
        and intents.get("general_question")
        and not intents["tables"]["want"]
        and not intents["figures"]["want"]
        and not intents["sources"]["want"]
    ):
        logger.info(
            "ANSWER: FULLREAD(auto) mode, uid=%s, doc_id=%s",
            uid,
            doc_id,
        )
        _limit = int(getattr(Cfg, "DIRECT_MAX_CHARS", 80000))
        full_text = _full_document_text(uid, doc_id, limit_chars=_limit + 1)
        full_len = len(full_text or "")
        logger.debug(
            "ANSWER: FULLREAD(auto) full_text_len=%d (limit=%d)",
            full_len,
            _limit,
        )

        # 1) вообще пустой текст → честно падаем в обычный RAG-пайплайн ниже
        if not full_text.strip():
            logger.warning(
                "ANSWER: FULLREAD(auto) got empty full_text, falling back to RAG (uid=%s, doc_id=%s)",
                uid,
                doc_id,
            )
        # 2) документ целиком влезает в лимит → прямой FULLREAD
        elif full_len <= _limit:
            system_prompt = (
                "Ты ассистент по дипломным работам. Тебе дан ПОЛНЫЙ текст ВКР/документа.\n"
                "Отвечай строго по этому тексту, без внешних фактов. Не добавляй разделов вида "
                "«Чего не хватает» и не проси дополнительные данные.\n"
                "Если вопрос про таблицы/рисунки — используй подписи и ближайший текст; не придумывай номера/значения. "
                "Не придумывай также предметную область и термины (например, продажи, клиенты, выручка, маркетинг и т.п.), "
                "если они прямо не указаны в тексте.\n"
                "Если запрошенного рисунка/таблицы нет в тексте — ответь: «данного рисунка нет в работе».\n"
                "Если объект есть, но он в плохом качестве/нечитаем — ответь: «Рисунок плохого качества, не могу проанализировать», "
                "и добавь краткую подпись/контекст из текста. Цитируй коротко, без ссылок на страницы."
            )

            verbosity = _detect_verbosity(q_text)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "assistant", "content": f"[Документ — полный текст]\n{full_text}"},
                {"role": "user", "content": f"{q_text}\n\n{_verbosity_addendum(verbosity)}"},
            ]

            if STREAM_ENABLED and chat_with_gpt_stream is not None:
                try:
                    stream = chat_with_gpt_stream(
                        messages,
                        temperature=0.2,
                        max_tokens=FINAL_MAX_TOKENS,
                    )  # type: ignore
                    await _stream_to_telegram(m, stream)
                    return
                except Exception as e:
                    logging.exception("auto fullread stream failed: %s", e)

            try:
                ans = chat_with_gpt(
                    messages,
                    temperature=0.2,
                    max_tokens=FINAL_MAX_TOKENS,
                )
                if ans:
                    await _send(m, _strip_unwanted_sections(ans))
                    return
            except Exception as e:
                logging.exception("auto fullread non-stream failed: %s", e)
        # 3) документ длинный → итеративное чтение (map→reduce)
        else:
            # документ большой → итеративное чтение (map→reduce)
            messages, err = _iterative_fullread_build_messages(uid, doc_id, q_text)
            if messages:
                if STREAM_ENABLED and chat_with_gpt_stream is not None:
                    try:
                        stream = chat_with_gpt_stream(
                            messages,
                            temperature=0.2,
                            max_tokens=FINAL_MAX_TOKENS,
                        )  # type: ignore
                        await _stream_to_telegram(m, stream)
                        return
                    except Exception as e:
                        logging.exception("auto iterative stream failed: %s", e)
                try:
                    ans = chat_with_gpt(messages, temperature=0.2, max_tokens=FINAL_MAX_TOKENS)
                    if ans:
                        await _send(m, _strip_unwanted_sections(ans))
                        return
                except Exception as e:
                    logging.exception("auto iterative non-stream failed: %s", e)
            elif err:
                # В auto-режиме не рвём основной пайплайн, а тихо логируем
                logging.warning(
                    "ANSWER: FULLREAD(auto) iterative build failed, "
                    "falling back to RAG (uid=%s, doc_id=%s): %s",
                    uid,
                    doc_id,
                    err,
                )
                # без return — ниже спокойно отработает обычный RAG-ответ

    # ====== FULLREAD: iterative/digest ======
    if fr_mode in {"iterative", "digest"} and not pure_figs:
        messages, err = _iterative_fullread_build_messages(uid, doc_id, q_text)
        if messages:
            if STREAM_ENABLED and chat_with_gpt_stream is not None:
                try:
                    stream = chat_with_gpt_stream(messages, temperature=0.2, max_tokens=FINAL_MAX_TOKENS)  # type: ignore
                    await _stream_to_telegram(m, stream)
                    return
                except Exception as e:
                    logging.exception("iterative fullread stream failed: %s", e)
            try:
                ans = chat_with_gpt(messages, temperature=0.2, max_tokens=FINAL_MAX_TOKENS)
                if ans:
                    await _send(m, _strip_unwanted_sections(ans))
                    return
            except Exception as e:
                logging.exception("iterative fullread non-stream failed: %s", e)
        else:
            if err:
                await _send(m, err)
                return
        # если что-то не вышло — проваливаемся в стандартный режим ниже

    # ====== Стандартный мульти-интент пайплайн (RAG) ======
    await _ensure_modalities_indexed(m, uid, doc_id, intents)
    facts = _gather_facts(uid, doc_id, intents)
    logger.info(
        "ANSWER: RAG facts gathered (uid=%s, doc_id=%s, keys=%s)",
        uid,
        doc_id,
        list(facts.keys()) if isinstance(facts, dict) else type(facts),
    )

    # NEW: если по общему вопросу ничего не нашлось именно в тексте работы —
    # сначала спрашиваем, можно ли ответить в общем виде как [модель].
    if intents.get("general_question") and not facts.get("general_ctx") and not facts.get("summary_text"):
        MODEL_EXTRA_PENDING[uid] = {
            "kind": "generic",
            "question": intents["general_question"] or q_text,
        }
        await _send(
            m,
            "По этому вопросу я не нашёл явной информации в самом тексте работы. "
            "Могу ответить в общем виде как [модель] (это уже не будет опираться на документ). "
            "Напиши «да» или «нет»."
        )
        return

    # ↓ НОВОЕ: если есть план подпунктов — включаем многошаговую подачу
        # ↓ НОВОЕ: если есть план подпунктов — включаем многошаговую подачу,
    # но только когда это реально оправдано (есть подпункты и вопрос не слишком короткий).
    discovered_items: list[dict] | None = None
    if isinstance(facts, dict):
        discovered_items = (
            (facts.get("coverage") or {}).get("items")
            or facts.get("general_subitems")
        )

    if _should_use_multistep(q_text, discovered_items):
        try:
            handled = await _run_multistep_answer(
                m,
                uid,
                doc_id,
                q_text,
                discovered_items=discovered_items,  # отправит A→B→… и вернёт True
            )
            if handled:
                return
        except Exception as e:
            logging.exception("multistep pipeline failed, fallback to normal: %s", e)
    # если мультишаг не подошёл — ниже идём по обычному пайплайну



        # обычный путь + явная инструкция по вербозности
    verbosity = _detect_verbosity(q_text)
    SAFE_RULES = (
        "Отвечай строго по приведённым фактам и цитатам из контекста. "
        "Если данных нет — так и скажи, без домыслов. Не придумывай номера/значения "
        "и не придумывай предметную область и термины (например, продажи, клиенты, выручка, "
        "маркетинг и т.п.), если их нет в тексте."
    )

    enriched_q = f"{SAFE_RULES}\n\n{q_text}\n\n{_verbosity_addendum(verbosity)}"

    # если хочется обновлять «последний упомянутый рисунок» — возьми из текста запроса
    figs_in_q = [_num_norm_fig(n) for n in _extract_fig_nums(q_text)]
    if figs_in_q:
        LAST_REF.setdefault(uid, {})["figure_nums"] = figs_in_q

        # NEW: прямой мультимодальный ответ, если есть релевантные картинки из документа
    # (не ломает старую логику: если не получилось/нет картинок — идём в generate_answer)
    try:
        if intents.get("general_question") and getattr(Cfg, "vision_active", lambda: False)():
            # подтянем релевантные чанк-хиты и выберем 1–3 файла-изображения
            hits_v = retrieve(uid, doc_id, intents["general_question"], top_k=10) or []
            img_paths = _pick_images_from_hits(hits_v, limit=getattr(Cfg, "VISION_MAX_IMAGES_PER_REQUEST", 3))
            if img_paths and (chat_with_gpt_stream_multimodal or chat_with_gpt_multimodal):
                # контекст из RAG, если он есть
                ctx = (facts.get("general_ctx") or "").strip() if isinstance(facts, dict) else ""
                mm_system = (
                    "Ты репетитор по ВКР. У тебя есть вопрос, краткий текстовый контекст и сами изображения "
                    "(фото/сканы/диаграммы) из документа. Отвечай по делу, используя изображения напрямую. "
                    "Не придумывай значения и номера, пиши только то, что видно или есть в тексте. "
                    "Не придумывай также предметную область и термины (например, продажи, клиенты, выручка, "
                    "маркетинг и т.п.), если они не указаны в тексте или подписях к изображениям."
                )

                mm_prompt = (f"{q_text}\n\nКонтекст из документа:\n{ctx}" if ctx else q_text)

                if STREAM_ENABLED and chat_with_gpt_stream_multimodal is not None:
                    stream = chat_with_gpt_stream_multimodal(
                        mm_prompt,
                        image_paths=img_paths,
                        system=mm_system,
                        temperature=0.2,
                        max_tokens=FINAL_MAX_TOKENS,
                    )
                    await _stream_to_telegram(m, stream)
                    return
                elif chat_with_gpt_multimodal is not None:
                    ans_mm = chat_with_gpt_multimodal(
                        mm_prompt,
                        image_paths=img_paths,
                        system=mm_system,
                        temperature=0.2,
                        max_tokens=FINAL_MAX_TOKENS,
                    )
                    ans_mm = (ans_mm or "").strip()
                    if ans_mm:
                        await _send(m, _strip_unwanted_sections(ans_mm))
                        return
    except Exception as e:
        logging.exception("multimodal answer path failed, falling back: %s", e)

    # --- старый путь RAG → генерация ответа по фактам (answer_builder) ---

    # 1) пробуем стримовую версию, если она есть
    if STREAM_ENABLED and generate_answer_stream is not None:
        try:
            stream = generate_answer_stream(
                enriched_q,
                facts,
                language=intents.get("language", "ru"),
            )
            await _stream_to_telegram(m, stream)
            return
        except Exception:
            logging.exception("generate_answer_stream failed, fallback to sync")

    # 2) нестримовый фолбэк
    try:
        answer = generate_answer(
            enriched_q,
            facts,
            language=intents.get("language", "ru"),
        )
    except Exception as e:
        logging.exception("generate_answer failed: %s", e)
        answer = ""

    # 3) страховка от пустого ответа: всегда что-то показываем
    answer = (answer or "").strip()
    if not answer:
        answer = (
            "Не удалось получить содержательный ответ из текста работы. "
            "Попробуй переформулировать вопрос или уточнить, какой раздел, таблицу или рисунок тебя интересует."
        )

    await _send(m, _strip_unwanted_sections(answer))


# ------------------------------ эмбеддинг-профиль ------------------------------

def _current_embedding_profile() -> str:
    dim = probe_embedding_dim(None)
    if dim:
        return f"emb={Cfg.POLZA_EMB}|dim={dim}"
    return f"emb={Cfg.POLZA_EMB}"

def _needs_reindex_by_embeddings(con, doc_id: int) -> bool:
    """
    Проверяем, не пора ли переиндексировать документ из-за смены embedding-модели
    или размерности эмбеддингов.

    В layout_profile храним строку вида:
      "emb=polza-emb-v1|dim=768"
    """
    if not _table_has_columns(con, "documents", ["layout_profile"]):
        # старые базы без layout_profile — лучше переиндексировать
        return True

    cur = con.cursor()
    cur.execute("SELECT layout_profile FROM documents WHERE id=?", (doc_id,))
    row = cur.fetchone()
    stored = (row["layout_profile"] or "") if row else ""
    if not stored:
        # профиля нет — тоже повод переиндексировать
        return True

    cur_model = Cfg.POLZA_EMB.strip().lower()
    stored_model = ""
    stored_dim: int | None = None

    for part in stored.split("|"):
        part = (part or "").strip().lower()
        if part.startswith("emb="):
            # "emb=polza-emb-v1" → "polza-emb-v1"
            stored_model = part[4:]
        elif part.startswith("dim="):
            # "dim=768" → 768
            try:
                stored_dim = int(part[4:])
            except ValueError:
                stored_dim = None

    # если embedding-модель поменялась — точно переиндексировать
    if stored_model and stored_model != cur_model:
        return True

    # сверяем размерность эмбеддингов, если она известна
    try:
        cur_dim = probe_embedding_dim(None)
    except Exception:
        cur_dim = None

    if cur_dim and stored_dim and cur_dim != stored_dim:
        return True

    # всё совпало — можно не трогать документ
    return False


# ------------------------------ обычный текст ------------------------------

@dp.message(F.text & ~F.via_bot & ~F.text.startswith("/"))
async def qa(m: types.Message):
    uid = ensure_user(str(m.from_user.id))
    doc_id = ACTIVE_DOC.get(uid)

    if not doc_id:
        persisted = get_user_active_doc(uid)
        if persisted:
            ACTIVE_DOC[uid] = persisted
            doc_id = persisted

    text = (m.text or "").strip()

    # NEW: если ждём от пользователя ответа «да/нет» про [модель] — обрабатываем его отдельно
    pending = MODEL_EXTRA_PENDING.get(uid)
    if pending:
        low = text.lower()
        if low in ("да", "д", "ага", "ок", "хорошо", "yes", "y"):
            info = MODEL_EXTRA_PENDING.pop(uid, None) or {}
            kind = (info.get("kind") or "generic").lower()

            if kind == "table_more":
                # doc_id могли сохранить в момент вопроса «опиши подробнее»
                doc_id_for_pending = info.get("doc_id") or doc_id

                # если почему-то doc_id не удалось восстановить — падаем в общий [модель]
                if not doc_id_for_pending:
                    await _answer_with_model_extra(
                        m,
                        uid,
                        info.get("question") or "",
                    )
                    return

                await _answer_with_model_extra_table(
                    m,
                    uid,
                    doc_id_for_pending,
                    info.get("question") or "",
                    info.get("ctx_tables") or "",
                    info.get("nums") or [],
                )
            else:
                await _answer_with_model_extra(
                    m,
                    uid,
                    info.get("question") or "",
                )
            return
        # любая другая реплика сбрасывает ожидание и идёт по обычному пути
        MODEL_EXTRA_PENDING.pop(uid, None)

    # 👋 РАННИЙ ответ на приветствие, без постановки в очередь
    if _is_greeting(text):
        greet = getattr(
            Cfg, "MSG_GREET",
            "Привет! Я репетитор по твоей ВКР. Пришли файл ВКР (.doc/.docx) — и я помогу по содержанию."
        )
        await _send(m, greet)
        return

    if not doc_id:
        # сохраняем вопрос, чтобы ответить после индексации первого файла
        if text:
            enqueue_pending_query(uid, text, meta={"source": "chat", "reason": "no_active_doc"})
        await _send(m, Cfg.MSG_NEED_FILE_QUEUED)
        return

    await respond_with_answer(m, uid, doc_id, text)
