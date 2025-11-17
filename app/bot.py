# app/bot.py
import re
import os
import html
import json
import logging
import asyncio
import time
import math
from typing import Iterable, AsyncIterable, Optional, List, Tuple

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

# ↓ добавили мягкий импорт по-подпунктной генерации из ace
try:
    from .ace import plan_subtasks, answer_subpoint, _merge_subanswers as merge_subanswers  # type: ignore
except Exception:
    try:
        # бэкап: если в ace функции экспортированы с подчёркиванием
        from .ace import _plan_subtasks as plan_subtasks, _answer_subpoint as answer_subpoint, _merge_subanswers as merge_subanswers  # type: ignore
    except Exception:
        plan_subtasks = None   # type: ignore
        answer_subpoint = None # type: ignore
        merge_subanswers = None # type: ignore

# ---------- polza client: пробуем стрим, фолбэк на обычный чат ----------
try:
    from .polza_client import (
        probe_embedding_dim,
        chat_with_gpt,
        chat_with_gpt_stream,
        vision_extract_values,
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


# --------------------- ПАРАМЕТРЫ СТРИМИНГА (с дефолтами) ---------------------

STREAM_ENABLED: bool = getattr(Cfg, "STREAM_ENABLED", True)
STREAM_EDIT_INTERVAL_MS: int = getattr(Cfg, "STREAM_EDIT_INTERVAL_MS", 900)  # как часто редактировать сообщение
STREAM_MIN_CHARS: int = getattr(Cfg, "STREAM_MIN_CHARS", 120)               # мин. приращение между апдейтами
STREAM_MODE: str = getattr(Cfg, "STREAM_MODE", "edit")                       # "edit" | "multi"
TG_MAX_CHARS: int = getattr(Cfg, "TG_MAX_CHARS", 3900)
FIG_MEDIA_LIMIT: int = getattr(Cfg, "FIG_MEDIA_LIMIT", 12)

# ↓ Новое: управляем «много сообщений» даже когда не упираемся в 4096
TG_SPLIT_TARGET: int = getattr(Cfg, "TG_SPLIT_TARGET", 1600)   # целевой размер части
TG_SPLIT_MAX_PARTS: int = getattr(Cfg, "TG_SPLIT_MAX_PARTS", 3)  # не больше 3 сообщений
_SPLIT_ANCHOR_RE = re.compile(
    r"(?m)^(?:### .+|## .+|\*\*[^\n]+?\*\*|\d+[).] .+|- .+)$"
)  # предпочитаемые границы (заголовки/списки)
STREAM_HEAD_START_MS: int = getattr(Cfg, "STREAM_HEAD_START_MS", 250)        # первый апдейт быстрее
FINAL_MAX_TOKENS: int = getattr(Cfg, "FINAL_MAX_TOKENS", 1600)
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

# --------------------- форматирование и отправка ---------------------

# Markdown → HTML (минимально-необходимое: **bold**, __bold__, *italic*, _italic_, `code`)
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
    for chunk in _split_multipart(text or ""):
        await m.answer(_to_html(chunk), parse_mode="HTML", disable_web_page_preview=True)


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

    if verbosity == "short":
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
        for p in (attrs.get("images") or []):
            if p and os.path.exists(p) and p not in acc:
                acc.append(p)
            if len(acc) >= limit:
                return acc
    return acc

def _pairs_to_bullets(pairs: list[dict]) -> str:
    lines = []
    for r in (pairs or []):
        lab = (str(r.get("label") or "")).strip()
        val = (str(r.get("value") or "")).strip()
        unit = (str(r.get("unit") or "")).strip()
        if lab or val:
            lines.append(f"— {lab}: {val}" + (f" {unit}" if unit else ""))
    return "\n".join(lines)

async def _stream_to_telegram(m: types.Message, stream, head_text: str = "⌛️ Печатаю ответ…") -> None:
    current_text = ""
    sent_parts = 0
    initial = await m.answer(_to_html(head_text), parse_mode="HTML", disable_web_page_preview=True)
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
                        await initial.edit_text(_to_html(part), parse_mode="HTML", disable_web_page_preview=True)
                        freeze_initial = True  # <- больше не трогаем initial
                    else:
                        await m.answer(_to_html(part), parse_mode="HTML", disable_web_page_preview=True)
                except TelegramBadRequest:
                    await m.answer(_to_html(part), parse_mode="HTML", disable_web_page_preview=True)

                sent_parts += 1
                current_text = current_text[cut:].lstrip()
                last_edit_at = _now_ms()
                continue

            # 3.b) защита от лимита
            if len(current_text) >= TG_MAX_CHARS:
                cut = _smart_cut_point(current_text, TG_MAX_CHARS)
                final_part = current_text[:cut]

                if STREAM_MODE == "multi" and (freeze_initial or sent_parts > 0):
                    # 🔧 в multi не редактируем initial после 1-й части
                    await m.answer(_to_html(final_part), parse_mode="HTML", disable_web_page_preview=True)
                else:
                    try:
                        await initial.edit_text(_to_html(final_part), parse_mode="HTML", disable_web_page_preview=True)
                    except TelegramBadRequest:
                        await m.answer(_to_html(final_part), parse_mode="HTML", disable_web_page_preview=True)

                current_text = current_text[cut:].lstrip()
                # 🔧 новый плейсхолдер нужен только в edit-режиме
                if STREAM_MODE == "edit":
                    initial = await m.answer(_to_html("…"), parse_mode="HTML", disable_web_page_preview=True)
                last_edit_at = _now_ms()
                continue

            # 3.c) периодические правки — 🔧 ТОЛЬКО в режиме edit
            now = _now_ms()
            if STREAM_MODE == "edit" and (now - last_edit_at) >= STREAM_EDIT_INTERVAL_MS and len(current_text) >= STREAM_MIN_CHARS:
                try:
                    await initial.edit_text(_to_html(current_text), parse_mode="HTML", disable_web_page_preview=True)
                    last_edit_at = now
                except TelegramBadRequest:
                    pass

        # финальный хвост
        if current_text:
            try:
                if STREAM_MODE == "multi" and sent_parts > 0:
                    await m.answer(_to_html(current_text), parse_mode="HTML", disable_web_page_preview=True)
                else:
                    await initial.edit_text(_to_html(current_text), parse_mode="HTML", disable_web_page_preview=True)
            except TelegramBadRequest:
                await m.answer(_to_html(current_text), parse_mode="HTML", disable_web_page_preview=True)

    finally:
        stop_typer.set()
        try:
            await typer_task
        except Exception:
            pass


async def _run_multistep_answer(
    m: types.Message,
    uid: int,
    doc_id: int,
    q_text: str,
    *,
    discovered_items: list[dict] | None = None,
) -> bool:
    """
    Генерируем: план → по каждому подпункту отдельный ответ → (опц.) финальный merge.
    Возвращает True, если путь обработан и ничего дальше делать не нужно.
    """
    if not MULTI_STEP_SEND_ENABLED:
        return False
    if not (plan_subtasks and answer_subpoint and merge_subanswers):
        # нет необходимых функций из ace — выходим
        return False

    # план из coverage или строим планерoм
            # план из coverage или строим планером
    items = (discovered_items or [])
    if not items:
        try:
            items = plan_subtasks(q_text) or []
        except Exception:
            items = []

    # нормализация: поддерживаем и dict, и str
    norm_items: list[dict] = []
    for idx, it in enumerate(items, start=1):
        if isinstance(it, str):
            norm_items.append({"id": idx, "ask": it.strip()})
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
    await _send(m, f"Вопрос многочастный. Отвечаю по подпунктам ({len(items)} шт.):\n\n{preview}")

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
        # контекст для конкретного подпункта
        ctx_text = ""
        try:
            # если есть coverage-бакет — собираем контекст прямо из чанков подпункта
            bucket = cov_map.get(str(it.get("id") or i)) or []
            if bucket:
                ctx_text = build_context_coverage(bucket, items_count=1)
        except Exception:
            ctx_text = ""
        # 2) фолбэки
        if not ctx_text:
            ctx_text = best_context(uid, doc_id, ask, max_chars=6000) or ""
        if not ctx_text:
            hits = retrieve(uid, doc_id, ask, top_k=8)
            if hits:
                ctx_text = build_context(hits)
        if not ctx_text:
            ctx_text = _first_chunks_context(uid, doc_id, n=12, max_chars=6000)

        # генерация по подпункту (кастомная подсказка в ace + критика/правка)
        try:
            part = answer_subpoint(ask, ctx_text, MULTI_PASS_SCORE).strip()
        except Exception as e:
            logging.exception("answer_subpoint failed: %s", e)
            part = ""

        # отправка блока
        header = f"**{i}. {ask}**\n\n"
        await _send(m, header + (part or "Не удалось сгенерировать ответ по этому подпункту."))
        subanswers.append(f"{header}{part}")

        # микропаузa, чтобы не упереться в rate/чаты
        await asyncio.sleep(MULTI_STEP_PAUSE_MS / 1000)

    # (опционально) финальный сводный блок
    if MULTI_STEP_FINAL_MERGE:
        try:
            merged = merge_subanswers(q_text, items, subanswers).strip()
            if merged:
                await _send(m, "**Итоговый сводный ответ**\n\n" + merged)
        except Exception as e:
            logging.exception("merge_subanswers failed: %s", e)

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

# NEW: для подстановки номера раздела из вопроса и для анафоры «этот пункт/рисунок»
_SECTION_NUM_RE = re.compile(
    r"(?i)\b(?:глава\w*|раздел\w*|пункт\w*|подраздел\w*|sec(?:tion)?\.?|chapter)"
    r"\s*(?:№\s*)?((?:[A-Za-zА-Яа-я](?=[\.\d]))?\s*\d+(?:[.,]\d+)*)"
)
_ANAPH_HINT_RE = re.compile(r"(?i)\b(этот|эта|это|данн\w+|про него|про неё|про нее)\b")


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



# --------------------- Таблицы: парсинг/нормализация ---------------------

_TABLE_ANY = re.compile(r"\bтаблиц\w*|\bтабл\.\b|\bтаблица\w*|(?:^|\s)table(s)?\b", re.IGNORECASE)
# Поддерживаем: 2.1, 3, A.1, А.1, П1.2
_TABLE_TITLE_RE = re.compile(r"(?i)\bтаблица\s+(\d+(?:[.,]\d+)*|[a-zа-я]\.?\s*\d+(?:[.,]\d+)*)\b(?:\s*[—\-–]\s*(.+))?")
_COUNT_HINT = re.compile(r"\bсколько\b|how many", re.IGNORECASE)
_WHICH_HINT = re.compile(r"\bкаки(е|х)\b|\bсписок\b|\bперечисл\w*\b|\bназов\w*\b", re.IGNORECASE)

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

def _shorten(s: str, limit: int = 120) -> str:
    # Ничего не режем — возвращаем как есть.
    return (s or "").strip()



# -------- Таблицы: подсчёт и список (совместимо со старыми БД) --------

def _distinct_table_basenames(uid: int, doc_id: int) -> list[str]:
    """
    Собираем «базовые» имена таблиц (section_path без хвоста ' [row …]').
    Работает и с новыми индексами (table_row) и со старыми.
    """
    con = get_conn()
    cur = con.cursor()

    # сначала пробуем опереться на типы
    if _table_has_columns(con, "chunks", ["element_type"]):
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
        # очень старый индекс — эвристика
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
    желательно ту, где в attrs лежит caption_num/label.
    """
    con = get_conn()
    cur = con.cursor()
    like1 = f'%\"caption_num\": \"{num}\"%'
    like2 = f'%\"label\": \"{num}\"%'
    row = None

    # 1) по номеру в attrs
    try:
        cur.execute(
            """
            SELECT page, section_path, attrs, text
            FROM chunks
            WHERE owner_id=? AND doc_id=? AND element_type='figure'
              AND (attrs LIKE ? OR attrs LIKE ?)
            ORDER BY id ASC LIMIT 1
            """,
            (uid, doc_id, like1, like2),
        )
        row = cur.fetchone()
    except Exception:
        row = None

    # 2) фолбэк — по section_path
    if not row:
        try:
            cur.execute(
                """
                SELECT page, section_path, attrs, text
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



def _format_chart_values(chart_data: list) -> str:
    rows = chart_data or []

    # Соберём числа и метки
    labels, nums, units = [], [], []
    all_numeric = True
    for r in rows:
        labels.append((str(r.get("label") or r.get("name") or r.get("category") or "")).strip())
        val = r.get("value")
        if val is None:
            val = r.get("y") or r.get("x") or r.get("v") or r.get("count")
        units.append(r.get("unit") or "")
        try:
            nums.append(float(str(val).replace(",", ".")))
        except Exception:
            all_numeric = False
            break

    # Эвристики "это проценты":
    #  - единицы содержат '%' ИЛИ
    #  - все значения в [0..1.2] и сумма ≈ 1 (доли) ИЛИ
    #  - все значения в [0..100] и сумма ≈ 100 (почти проценты)
    if all_numeric and rows:
        total = sum(nums)
        unit_has_percent = any(isinstance(u, str) and "%" in u for u in units)
        looks_fraction = all(0 <= v <= 1.2 for v in nums) and 0.98 <= total <= 1.02
        looks_percent  = all(0 <= v <= 100 for v in nums) and 99 <= total <= 101

        if unit_has_percent or looks_fraction or looks_percent:
            base = [v * 100 for v in nums] if looks_fraction else nums[:]
            # Округляем так, чтобы сумма была ровно 100 (метод наибольших остатков)
            floors = [int(math.floor(x)) for x in base]
            need = int(round(100 - sum(floors)))
            remainders = [x - f for x, f in zip(base, floors)]
            order = sorted(range(len(base)), key=lambda i: remainders[i], reverse=True)
            for i in order[:max(0, abs(need))]:
                floors[i] += 1 if need > 0 else -1
            return "\n".join([f"— {labels[i]}: {floors[i]}%" for i in range(len(floors))])

    # Фолбэк: как было
    lines = []
    for i, r in enumerate(rows):
        label = labels[i] if i < len(labels) else (str(r.get("label") or r.get("name") or r.get("category") or "")).strip()
        val = r.get("value")
        if val is None:
            val = r.get("y") or r.get("x") or r.get("v") or r.get("count")
        unit = r.get("unit")
        unit_s = f" {unit}" if isinstance(unit, str) and unit.strip() else ""
        if label or val is not None:
            lines.append(f"— {label}: {val}{unit_s}".strip())
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

        # --- 4) chart_data из attrs (точные числовые значения) ---
        row = _fetch_figure_row_by_num(uid, doc_id, orig)
        if not row and norm != orig:
            row = _fetch_figure_row_by_num(uid, doc_id, norm)
        if row:
            attrs_json = row["attrs"] if ("attrs" in row.keys()) else None
            cd, _ctype, _attrs = _parse_chart_data(attrs_json)
            if cd:
                rec["values"] = _fill_empty_percents(_format_chart_values(cd))
            if not rec["display"]:
                title_text = row["text"] if ("text" in row.keys()) else None
                rec["display"] = _compose_figure_display(
                    attrs_json,
                    row["section_path"],
                    title_text,
                )

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
    Собираем текстовый блок с точными значениями по рисункам
    (без отправки в Telegram). Используется как префикс к
    основному описанию.
    """
    lines: list[str] = []

    for rec in records:
        # 1) chart_data уже мог быть проставлен в _build_figure_records
        values = (rec.get("values") or "").strip()

        # 2) ФОЛБЭК №1: если chart_data нет, пробуем через summarizer.extract_figure_value
        if not values and summ_extract_figure_value is not None:
            owner_id = rec.get("owner_id")
            doc_id = rec.get("doc_id")
            num = rec.get("orig") or rec.get("num")

            if owner_id and doc_id and num:
                try:
                    raw = summ_extract_figure_value(owner_id, doc_id, str(num)) or ""
                    raw = raw.strip()
                except Exception:
                    raw = ""

                # ожидаемый формат из summarizer.extract_figure_value:
                # "**Рисунок N.** Заголовок\n<табличка/CSV/...>"
                if raw.startswith("**Рисунок"):
                    raw_lines = raw.splitlines()
                    body = "\n".join(raw_lines[1:]).strip() if len(raw_lines) >= 2 else ""
                    if body:
                        values = body
                        rec["values"] = body  # чтобы GPT-описание тоже видело эти числа

        # 3) ФОЛБЭК №2: если нет ни chart_data, ни summarizer — пробуем OOXML-индекс
        if not values:
            try:
                doc_id = rec.get("doc_id")
                num = rec.get("orig") or rec.get("num")
                idx = _ooxml_get_index(doc_id) if doc_id else None
                body = ""

                if idx and "oox_fig_lookup" in globals() and num:
                    oox_res = oox_fig_lookup(idx, str(num))

                    # Подстрой под свою реализацию figure_lookup:
                    # здесь предполагаем, что вернётся либо строка, либо dict.
                    if isinstance(oox_res, str):
                        body = oox_res.strip()
                    elif isinstance(oox_res, dict):
                        body = (
                            (oox_res.get("values_text")
                             or oox_res.get("text")
                             or "")
                        ).strip()

                if body:
                    values = body
                    rec["values"] = body
            except Exception:
                # не ломаем весь ответ, если с OOXML что-то пошло не так
                pass

        # 4) если после всех попыток чисел нет — пропускаем этот рисунок
        if not values:
            continue

        disp = rec.get("display") or f"Рисунок {rec.get('num') or ''}".strip()
        lines.append(f"**{disp} — точные значения**\n\n{values}")

    if lines:
        return "\n\n".join(lines)

    if need_values:
        return (
            "По указанным рисункам не удалось автоматически извлечь точные числовые данные "
            "(ни из chart_data, ни через анализ изображения/OOXML). "
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
        if rec.get("values"):
            parts.append("Точные значения (как в документе):\n" + rec["values"])
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
        "Ты репетитор по дипломным работам. У тебя есть информация о рисунках диплома "
        "(подписи, текст рядом, распознанные данные диаграмм). "
        "Объясни студенту, что показывают рисунки, какие тенденции видны и какие выводы можно сделать. "
        "Не придумывай новые числа и не ссылайся на номера страниц."
    )
    user_prompt = (
        f"Вопрос пользователя: {question}\n\n"
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
    2) всегда стараемся отправить сами картинки;
    3) собираем общий блок с точными значениями (если есть);
    4) даём одно связное пояснение через GPT, в которое подмешан блок значений.

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
        await _send(m, "Указанные рисунки в работе не найдены.")
        return True

    # 3) сначала сами изображения (по всем найденным)
    try:
        cards_for_media = [
            {
                "num": r["num"],
                "display": r.get("display") or f"Рисунок {r['num']}",
                "images": r.get("images") or [],
            }
            for r in records
        ]
        await _send_media_from_cards(m, cards_for_media)
    except Exception:
        # если что-то не получилось — не срываем дальнейшие шаги
        pass

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

        # запишем даже если список пустой — генератор ответа это учтёт
        facts["tables"]["describe"] = desc_cards
        # describe по конкретным номерам + точные расчеты
        desc_cards = []
        if intents["tables"]["describe"]:
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

    # ----- Рисунки -----
    if intents["figures"]["want"]:
        f_limit = int(intents.get("figures", {}).get("limit", 10))
        lst = _list_figures_db(uid, doc_id, limit=f_limit)
        figs_block = {
            "count": int(lst.get("count") or 0),
            "list": list(lst.get("list") or []),
            "more": int(lst.get("more") or 0),
            "describe_lines": [],
        }

        if intents["figures"]["describe"]:
            try:
                cards = describe_figures_by_numbers(
                    uid, doc_id, intents["figures"]["describe"],
                    sample_chunks=2, use_vision=True, lang="ru"
                )
                if not cards:
                    figs_block["describe_lines"] = ["Данного рисунка нет в работе."]
                else:
                    lines = []
                    for c in cards:
                        disp = c.get("display") or "Рисунок"
                        vis  = (c.get("vision") or {}).get("description", "") or ""
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
        vb = verbatim_find(uid, doc_id, intents["general_question"], max_hits=3)

        cov = retrieve_coverage(
            owner_id=uid,
            doc_id=doc_id,
            question=intents["general_question"],
        )
        ctx = ""
        if cov and cov.get("snippets"):
            ctx = build_context_coverage(
                cov["snippets"],
                items_count=len(cov.get("items") or []) or None,
            )

        if not ctx:
            ctx = best_context(uid, doc_id, intents["general_question"], max_chars=6000)
        if not ctx:
            hits = retrieve(uid, doc_id, intents["general_question"], top_k=12)
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
            if Cfg.vision_active():
                # 1) берём топ-хиты специально для картинок
                hits_v = retrieve(uid, doc_id, intents["general_question"], top_k=10) or []

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
                    # 2) иначе — отправляем 1–3 картинки в vision_extract_values
                    img_paths = _pick_images_from_hits(hits_v, limit=getattr(Cfg, "VISION_MAX_IMAGES_PER_REQUEST", 3))
                    if img_paths and vision_extract_values:
                        hint = (hits_v[0].get("text") or "")[:300]
                        res = vision_extract_values(img_paths, caption_hint=hint, lang="ru")
                        rows = (res or {}).get("data") or []
                        if rows:
                            vision_block = "\n".join(
                                ["[Text on image]"] +
                                _pairs_to_bullets(rows).splitlines()
                            )
                        elif FIG_STRICT:
                            # нет надёжных чисел с картинки — явно помечаем
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
    """Удаляем разделы 'Чего не хватает'/'Не хватает' и подобные хвосты."""
    if not s:
        return s
    # вырезаем заголовок + абзац(ы) до следующего пустого разрыва
    pat = re.compile(r"(?mis)^\s*(?:чего|что)\s+не\s+хватает\s*:.*?(?:\n\s*\n|\Z)")
    s = pat.sub("", s)
    # отдельные строки-метки
    s = re.sub(r"(?mi)^\s*не\s+хватает\s*:.*$", "", s)
    return s.strip()


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


    if STREAM_ENABLED and chat_with_gpt_stream is not None:
        return ("__STREAM__", json.dumps(messages, ensure_ascii=False))

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

    joined = "\n\n".join([f"[MAP {i+1}]\n{d}" for i, d in enumerate(digests)])
    ctx = joined[: int(getattr(Cfg, "FULLREAD_CONTEXT_CHARS", 9000))]

    sys_reduce = (
        "Ты репетитор по ВКР. Ниже — короткие факты из разных частей документа (map-выжимки). "
        "Собери из них связный ответ на вопрос. Не выдумывай новых цифр/таблиц и не добавляй разделов "
        "про «чего не хватает». Отвечай только по имеющимся данным. Если запрошенного рисунка/таблицы "
        "нет в тексте — сформулируй кратко: «данного рисунка нет в работе». Если объект есть, но он "
        "нечитабелен, дай: «Рисунок плохого качества, не могу проанализировать», и добавь подпись/контекст из текста."
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

    # 1) скачиваем файл
    file = await bot.get_file(doc.file_id)
    stream = await bot.download_file(file.file_path)
    try:
        data = stream.read()
    finally:
        try:
            stream.close()
        except Exception:
            pass


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

async def respond_with_answer(m: types.Message, uid: int, doc_id: int, q_text: str):
    q_text = (q_text or "").strip()
    logging.debug(f"Получен запрос от пользователя: {q_text}")
    if not q_text:
        await _send(m, "Вопрос пустой. Напишите, что именно вас интересует по ВКР.")
        return

    viol = safety_check(q_text)
    if viol:
        await _send(m, viol + " Задайте корректный вопрос по ВКР.")
        return

    if await _maybe_run_gost(m, uid, doc_id, q_text):
        return


    # РАНО в respond_with_answer, до detect_intents:
    if _ALL_FIGS_HINT.search(q_text or ""):
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
        # карточки + сначала отправим фотографии пользователю
        cards = []
        try:
            cards = describe_figures_by_numbers(uid, doc_id, batch, sample_chunks=1, use_vision=False, lang="ru") or []
        except Exception:
            cards = []
        await _send_media_from_cards(m, cards)

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
                    # фолбэк — старый summarizer
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
            txt = vision_describe_figures(uid, doc_id, batch)
            await _send(m, (txt or "Не удалось описать рисунки.") + suffix)
        return


    # NEW: если в вопросе явно указан раздел/пункт — запоминаем его как последний
    m_area = _SECTION_NUM_RE.search(q_text)
    if m_area:
        try:
            area = (m_area.group(1) or "").replace(" ", "").replace(",", ".")
            LAST_REF.setdefault(uid, {})["area"] = area
        except Exception:
            pass

    # NEW: если вопрос расплывчатый «про этот ...», подставим последний референт
    def _expand_with_last_referent(uid: int, text: str) -> str:
        if not _ANAPH_HINT_RE.search(text or ""):
            return text
        last = LAST_REF.get(uid) or {}
        # приоритет — последний рисунок
        figs = last.get("figure_nums") or []
        if figs:
            return f"{text} (имеется в виду рисунок {figs[0]})"
        area = (last.get("area") or "").strip()
        if area:
            # если нет слова «пункт/раздел», добавим
            if not re.search(r"(?i)\b(глава|раздел|пункт|подраздел)\b", text):
                return f"{text} (имеется в виду пункт {area})"
            return f"{text} ({area})"
        return text
    q_text = _expand_with_last_referent(uid, q_text)

    # NEW: быстрый детерминированный путь для «поясни рисунок 2.1/3.4 …»
    # --- Определяем интенты заранее
    intents = detect_intents(q_text)
    verbosity = _detect_verbosity(q_text)

    # Чистый запрос про рисунки (нет секций/таблиц/общего обсуждения)
    pure_figs = intents["figures"]["want"] and not (
        intents["tables"]["want"] or intents["sources"]["want"] or
        intents.get("summary") or intents.get("general_question") or
        _SECTION_NUM_RE.search(q_text)
    )

    if intents["figures"]["want"]:
        try:
            await _ensure_modalities_indexed(m, uid, doc_id, intents)  # если figure==0, тихо переиндексирует
        except Exception:
            pass

    if pure_figs:
        if await _answer_figure_query(m, uid, doc_id, q_text, verbosity=verbosity):
            return
    else:
        if intents["figures"]["want"]:
            # сначала кратко ответим по рисункам, затем продолжим общий пайплайн
            await _answer_figure_query(m, uid, doc_id, q_text, verbosity=verbosity)



    # NEW: явная обработка «по пункту/разделу/главе X.Y»
        # NEW: явная обработка «по пункту/разделу/главе X.Y» (с защитой от «залипаний»)
    m_sec = _SECTION_NUM_RE.search(q_text)
    sec = None
    if m_sec:
        raw_sec = (m_sec.group(1) or "").strip()
        raw_sec = re.sub(r"^[A-Za-zА-Яа-я]\s+(?=\d)", "", raw_sec)
        sec = raw_sec.replace(" ", "").replace(",", ".")

    # ВСЕГДА первым делом пробуем строгий секционный ответ, если номер найден
    if sec:
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

    # Если sec найден, но запрос НЕ чистый — не отправляем отдельный ответ по пункту,
    # продолжаем обычный пайплайн ниже (RAG / FULLREAD), чтобы ответить на всё целиком.




    # ====== FULLREAD: auto ======
    fr_mode = getattr(Cfg, "FULLREAD_MODE", "off")
    if fr_mode == "auto":
        _limit = int(getattr(Cfg, "DIRECT_MAX_CHARS", 80000))
        # пробуем дать модели ПОЛНЫЙ текст, если влазит
        full_text = _full_document_text(uid, doc_id, limit_chars=_limit + 1)
        if full_text and len(full_text) <= _limit:
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

            if STREAM_ENABLED and chat_with_gpt_stream is not None:
                try:
                    stream = chat_with_gpt_stream(messages, temperature=0.2, max_tokens=FINAL_MAX_TOKENS)  # type: ignore
                    await _stream_to_telegram(m, stream)
                    return
                except Exception as e:
                    logging.exception("auto fullread stream failed: %s", e)
            try:
                ans = chat_with_gpt(messages, temperature=0.2, max_tokens=FINAL_MAX_TOKENS)
                if ans:
                    await _send(m, _strip_unwanted_sections(ans))
                    return
            except Exception as e:
                logging.exception("auto fullread non-stream failed: %s", e)
        else:
            # документ большой → итеративное чтение (map→reduce)
            messages, err = _iterative_fullread_build_messages(uid, doc_id, q_text)
            if messages:
                if STREAM_ENABLED and chat_with_gpt_stream is not None:
                    try:
                        stream = chat_with_gpt_stream(messages, temperature=0.2, max_tokens=FINAL_MAX_TOKENS)  # type: ignore
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
                await _send(m, err)
                return


    # ====== FULLREAD: direct ======
    if fr_mode == "direct":
        fr = _fullread_try_answer(uid, doc_id, q_text)
        if isinstance(fr, tuple) and fr and fr[0] == "__STREAM__":
            messages = json.loads(fr[1])
            try:
                stream = chat_with_gpt_stream(messages, temperature=0.2, max_tokens=FINAL_MAX_TOKENS)  # type: ignore
                await _stream_to_telegram(m, stream)
                return
            except Exception as e:
                logging.exception("direct fullread stream failed: %s", e)
                # тихо падаем в обычный пайплайн
        elif isinstance(fr, str) and fr:
            await _send(m, _strip_unwanted_sections(fr))
            return
        # иначе — RAG ниже

    # ====== FULLREAD: iterative/digest ======
    if fr_mode in {"iterative", "digest"}:
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


    # ↓ НОВОЕ: если есть план подпунктов — включаем многошаговую подачу
    discovered_items = None
    if isinstance(facts, dict):
        discovered_items = (facts.get("coverage", {}).get("items")
                            or facts.get("general_subitems"))
    try:
        handled = await _run_multistep_answer(
            m, uid, doc_id, q_text, discovered_items=discovered_items  # отправит A→B→… и вернёт True
        )
        if handled:
            return
    except Exception as e:
        logging.exception("multistep pipeline failed, fallback to normal: %s", e)


        # обычный путь + явная инструкция по вербозности
    verbosity = _detect_verbosity(q_text)
    SAFE_RULES = ("Отвечай строго по приведённым фактам и цитатам из контекста. "
                "Если данных нет — так и скажи, без домыслов. Не придумывай номера/значения.")
    enriched_q = f"{SAFE_RULES}\n\n{q_text}\n\n{_verbosity_addendum(verbosity)}"

    # если хочется обновлять «последний упомянутый рисунок» — возьми из текста запроса
    figs_in_q = [_num_norm_fig(n) for n in FIG_NUM_RE.findall(q_text)]
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
                    "Не придумывай значения и номера, пиши только то, что видно или есть в тексте."
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
                    ans = chat_with_gpt_multimodal(
                        mm_prompt,
                        image_paths=img_paths,
                        system=mm_system,
                        temperature=0.2,
                        max_tokens=FINAL_MAX_TOKENS,
                    )
                    if ans:
                        await _send(m, _strip_unwanted_sections(ans))
                        return
    except Exception as e:
        logging.exception("multimodal answer path failed, falling back: %s", e)

    # старый путь RAG → генерация
    if STREAM_ENABLED and generate_answer_stream is not None:
        try:
            stream = generate_answer_stream(enriched_q, facts, language=intents.get("language", "ru"))
            await _stream_to_telegram(m, stream)
            return
        except Exception as e:
            logging.exception("stream answer failed, fallback to non-stream: %s", e)

    reply = generate_answer(enriched_q, facts, language=intents.get("language", "ru"))
    await _send(m, _strip_unwanted_sections(reply))



# ------------------------------ эмбеддинг-профиль ------------------------------

def _current_embedding_profile() -> str:
    dim = probe_embedding_dim(None)
    if dim:
        return f"emb={Cfg.POLZA_EMB}|dim={dim}"
    return f"emb={Cfg.POLZA_EMB}"

def _needs_reindex_by_embeddings(con, doc_id: int) -> bool:
    if not _table_has_columns(con, "documents", ["layout_profile"]):
        return True
    cur = con.cursor()
    cur.execute("SELECT layout_profile FROM documents WHERE id=?", (doc_id,))
    row = cur.fetchone()
    stored = (row["layout_profile"] or "") if row else ""
    if not stored:
        return True
    cur_model = Cfg.POLZA_EMB.strip().lower()
    stored_model = ""
    stored_dim = None
    for part in stored.split("|"):
        part = (part or "").strip().lower()
        if part.startswith("emb="):
            stored_model = part[4:]
        if part.startswith("dim="):
            try:
                stored_dim = int(part[4:])
            except Exception:
                stored_dim = None
    if stored_model and stored_model != cur_model:
        return True
    cur_dim = probe_embedding_dim(None)
    if cur_dim and stored_dim and stored_dim != cur_dim:
        return True
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
