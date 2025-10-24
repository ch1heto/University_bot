# app/bot.py
import re
import os
import html
import json
import logging
import asyncio
import time
from typing import Iterable, AsyncIterable, Optional, List, Tuple

from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import Command
from aiogram.exceptions import TelegramBadRequest
from aiogram.enums import ChatAction

# ---------- answer builder: пытаемся взять стримовую версию, фолбэк на нестримовую ----------
try:
    from .answer_builder import generate_answer, generate_answer_stream  # type: ignore
except Exception:
    from .answer_builder import generate_answer  # type: ignore
    generate_answer_stream = None  # стрима нет — будем фолбэкать

from .config import Cfg
from .db import (
    ensure_user, get_conn,
    set_document_indexer_version, get_document_indexer_version,
    CURRENT_INDEXER_VERSION,
    update_document_meta, delete_document_chunks,
    set_user_active_doc, get_user_active_doc,  # ⬅️ персист активного документа
)
from .parsing import parse_docx, parse_doc, save_upload
from .indexing import index_document
from .retrieval import (
    retrieve, build_context, invalidate_cache,
    retrieve_coverage, build_context_coverage,
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
    from .polza_client import probe_embedding_dim, chat_with_gpt, chat_with_gpt_stream  # type: ignore
except Exception:
    from .polza_client import probe_embedding_dim, chat_with_gpt  # type: ignore
    chat_with_gpt_stream = None

# НОВОЕ: оркестратор приёма/обогащения (OCR таблиц-картинок, нормализация чисел)
from .ingest_orchestrator import enrich_sections
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

# ↓ Новое: управляем «много сообщений» даже когда не упираемся в 4096
TG_SPLIT_TARGET: int = getattr(Cfg, "TG_SPLIT_TARGET", 1600)   # целевой размер части
TG_SPLIT_MAX_PARTS: int = getattr(Cfg, "TG_SPLIT_MAX_PARTS", 3)  # не больше 3 сообщений
_SPLIT_ANCHOR_RE = re.compile(
    r"(?m)^(?:### .+|## .+|\*\*[^\n]+?\*\*|\d+[).] .+|- .+)$"
)  # предпочитаемые границы (заголовки/списки)
STREAM_HEAD_START_MS: int = getattr(Cfg, "STREAM_HEAD_START_MS", 250)        # первый апдейт быстрее
FINAL_MAX_TOKENS: int = getattr(Cfg, "FINAL_MAX_TOKENS", 1600)
TYPE_INDICATION_EVERY_MS: int = getattr(Cfg, "TYPE_INDICATION_EVERY_MS", 2000)

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
    """Безопасно экранируем HTML и конвертируем самый частый Markdown в тг-HTML."""
    if not text:
        return ""
    # 1) экранируем всё
    txt = html.escape(text)

    # 2) кодовые спаны первыми
    txt = _MD_CODE_RE.sub(r"<code>\1</code>", txt)

    # 3) заголовки вида '# ...' → <b>...</b>
    txt = _MD_H_RE.sub(r"<b>\1</b>", txt)

    # 4) жирный и курсив
    txt = _MD_BOLD_RE.sub(r"<b>\1</b>", txt)
    txt = _MD_BOLD2_RE.sub(r"<b>\1</b>", txt)
    txt = _MD_ITALIC_RE.sub(r"<i>\1</i>", txt)
    txt = _MD_ITALIC2_RE.sub(r"<i>\1</i>", txt)

    # 5) зачистка «висячих» **
    txt = re.sub(r"(?<!\*)\*\*(?!\*)", "", txt)
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
                     max_parts: int = TG_SPLIT_MAX_PARTS,
                     hard: int = TG_MAX_CHARS) -> list[str]:
    """
    Дробим ответ на 2–3 логических сообщения:
    - стремимся к target символов на часть;
    - режем по якорям (###/списки/нумерация), если есть;
    - никогда не превышаем hard (лимит Telegram).
    """
    s = text or ""
    if not s:
        return []
    if len(s) <= target:
        return [s]

    parts: list[str] = []
    rest = s

    for _ in range(max_parts - 1):
        if len(rest) <= target:
            break
        # ищем последнюю «красивую» границу до target
        cut = -1
        for m in _SPLIT_ANCHOR_RE.finditer(rest[: min(len(rest), hard)]):
            if m.start() < target:
                cut = m.start()
        if cut <= 0:
            cut = _smart_cut_point(rest, min(hard, target))
        parts.append(rest[:cut].rstrip())
        rest = rest[cut:].lstrip()

    # остаток и, если нужно, сверхжёсткое разбиение по hard
    while rest:
        parts.append(rest[:hard])
        rest = rest[hard:]

    return parts

async def _send(m: types.Message, text: str):
    """Бережно отправляем длинный текст частями в HTML-режиме (нестримовый фолбэк)."""
    for chunk in _split_multipart(text or ""):
        await m.answer(_to_html(chunk), parse_mode="HTML", disable_web_page_preview=True)

# --------------------- STREAM: вспомогалки ---------------------

def _now_ms() -> int:
    return int(time.time() * 1000)

async def _typing_loop(chat_id: int, stop_event: asyncio.Event):
    """Периодически отправляет индикатор 'typing', пока не остановим."""
    try:
        while not stop_event.is_set():
            await bot.send_chat_action(chat_id, ChatAction.TYPING)
            await asyncio.wait_for(stop_event.wait(), timeout=TYPE_INDICATION_EVERY_MS / 1000)
    except asyncio.TimeoutError:
        pass
    except Exception:
        pass

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
            if STREAM_MODE == "multi" and sent_parts < TG_SPLIT_MAX_PARTS - 1 and len(current_text) >= TG_SPLIT_TARGET:
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
    cov_snips = (cov or {}).get("snippets") or []
    cov_map = (cov or {}).get("by_item") or {}  # { "1": [чанки], "2": [чанки], ... }

    # по очереди: A → send, B → send, ...
    for i, it in enumerate(items, start=1):
        ask = (it.get("ask") or "").strip()
        # контекст для конкретного подпункта
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
        header = f"<b>{i}. {html.escape(ask)}</b>\n\n"
        await _send(m, header + (part or "Не удалось сгенерировать ответ по этому подпункту."))
        subanswers.append(f"{header}{part}")

        # микропаузa, чтобы не упереться в rate/чаты
        await asyncio.sleep(MULTI_STEP_PAUSE_MS / 1000)

    # (опционально) финальный сводный блок
    if MULTI_STEP_FINAL_MERGE:
        try:
            merged = merge_subanswers(q_text, items, subanswers).strip()
            if merged:
                await _send(m, "<b>Итоговый сводный ответ</b>\n\n" + merged)
        except Exception as e:
            logging.exception("merge_subanswers failed: %s", e)

    return True

# summarizer (мягкий импорт)
try:
    from .summarizer import is_summary_intent, overview_context  # могут отсутствовать — есть фолбэки ниже
except Exception:
    def is_summary_intent(text: str) -> bool:
        return bool(re.search(r"\b(суть|кратко|основн|главн|summary|overview|итог|вывод)\w*\b",
                              text or "", re.IGNORECASE))

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

# vision-описание рисунков (мягкий импорт; если нет — отвечаем текстовым фолбэком)
try:
    from .summarizer import describe_figures as vision_describe_figures
except Exception:
    def vision_describe_figures(owner_id: int, doc_id: int, numbers: list[str]) -> str:
        if not numbers:
            return "Не указаны номера рисунков."
        return "Описания рисунков недоступны (vision-модуль не подключён)."

# ГОСТ-валидатор (мягкий импорт)
try:
    from .validators_gost import validate_gost, render_report
except Exception:
    validate_gost = None
    render_report = None


# «Активный документ» в памяти процесса
ACTIVE_DOC: dict[int, int] = {}  # user_id -> doc_id


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

def _find_existing_doc(con, owner_id: int, sha256: str | None, file_uid: str | None):
    """Поиск уже загруженного документа пользователя по sha/file_unique_id (если колонки есть)."""
    if not _table_has_columns(con, "documents", ["content_sha256", "file_uid"]):
        return None
    cur = con.cursor()
    if sha256 and file_uid:
        cur.execute(
            "SELECT id FROM documents WHERE owner_id=? AND (content_sha256=? OR file_uid=?) "
            "ORDER BY id DESC LIMIT 1",
            (owner_id, sha256, file_uid),
        )
    elif sha256:
        cur.execute(
            "SELECT id FROM documents WHERE owner_id=? AND content_sha256=? "
            "ORDER BY id DESC LIMIT 1",
            (owner_id, sha256),
        )
    elif file_uid:
        cur.execute(
            "SELECT id FROM documents WHERE owner_id=? AND file_uid=? "
            "ORDER BY id DESC LIMIT 1",
            (owner_id, file_uid),
        )
    row = cur.fetchone()
    return (row["id"] if row else None)

def _insert_document(con, owner_id: int, kind: str, path: str,
                     sha256: str | None, file_uid: str | None) -> int:
    cur = con.cursor()
    if _table_has_columns(con, "documents", ["content_sha256", "file_uid"]):
        cur.execute(
            "INSERT INTO documents(owner_id, kind, path, content_sha256, file_uid) VALUES(?,?,?,?,?)",
            (owner_id, kind, path, sha256, file_uid),
        )
    else:
        cur.execute(
            "INSERT INTO documents(owner_id, kind, path) VALUES(?,?,?)",
            (owner_id, kind, path),
        )
    doc_id = cur.lastrowid
    con.commit()
    return doc_id


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
    s = (s or "").strip()
    if len(s) <= limit:
        return s
    return s[:limit - 1].rstrip() + "…"


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
        "Привет! Я репетитор по твоей ВКР. Пришли .doc/.docx — я проиндексирую и буду объяснять содержание: главы простым языком, смысл таблиц/рисунков, конспекты к защите. Можешь прикрепить вопрос к файлу или написать его отдельным сообщением."
    )

# ------------------------------ /diag ------------------------------

@dp.message(Command("diag"))
async def cmd_diag(m: types.Message):
    uid = ensure_user(str(m.from_user.id))
    doc_id = ACTIVE_DOC.get(uid) or get_user_active_doc(uid)
    if not doc_id:
        await _send(m, "Активного документа нет. Пришлите файл (.doc/.docx) сначала.")
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
    r"(?i)\b(рис(?:\.|унок)?|figure|fig\.?)\s*(?:№\s*)?(\d+(?:[.,]\d+)*)\b(?:\s*[—\-–:\u2013\u2014]\s*(.+))?"
)

def _compose_figure_display(attrs_json: str | None, section_path: str, title_text: str | None) -> str:
    """Делаем красивый заголовок рисунка по приоритетам."""
    num = None
    tail = None
    if attrs_json:
        try:
            a = json.loads(attrs_json or "{}")
            num = (a.get("caption_num") or a.get("label") or "").strip()
            tail = (a.get("caption_tail") or a.get("title") or "").strip()
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
    base = re.sub(r"(?i)^\s*(рис(?:\.|унок)?|figure|fig\.?)\s*", "", base).strip(" —–-")
    return _shorten(base or "Рисунок", 160)

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
        cur.execute(
            "SELECT DISTINCT section_path, attrs, text FROM chunks "
            "WHERE owner_id=? AND doc_id=? AND (text LIKE '[Рисунок]%' OR lower(section_path) LIKE '%рисунок%') "
            "ORDER BY id ASC",
            (uid, doc_id),
        )
    rows = cur.fetchall() or []
    con.close()

    items: list[str] = []
    for r in rows:
        section_path = r["section_path"] or ""
        attrs_json = r["attrs"] if "attrs" in r.keys() else None
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
    con = get_conn()
    rc = _count_et(con, uid, doc_id, "reference") if need_refs else 1
    fc = _count_et(con, uid, doc_id, "figure") if need_figs else 1
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
            await _send(m, "Обновил индекс документа: добавлены распознанные рисунки/источники.")
        except Exception as e:
            logging.exception("self-heal reindex failed: %s", e)


# -------------------------- Сбор фактов --------------------------

def _gather_facts(uid: int, doc_id: int, intents: dict) -> dict:
    """
    Собираем ТОЛЬКО факты из БД/индекса, без генерации текста.
    """
    facts: dict[str, object] = {"doc_id": doc_id, "owner_id": uid}
    # флаг «точные числа как в документе»
    facts["exact_numbers"] = bool(intents.get("exact_numbers"))

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
        facts["tables"] = {
            "count": total_tables,
            "list": items[:intents["tables"]["limit"]],
            "more": max(0, len(items) - intents["tables"]["limit"]),
            "describe": [],
        }

        # Авто-описание для общего запроса про таблицы
        desc_cards = []
        if not intents["tables"].get("describe"):
            # возьмём первые 3–5 таблиц из списка
            bases = _distinct_table_basenames(uid, doc_id)[:min(5, intents["tables"]["limit"])]
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
        lst = _list_figures_db(uid, doc_id, limit=intents["figures"]["limit"])
        figs_block = {
            "count": int(lst.get("count") or 0),
            "list": list(lst.get("list") or []),
            "more": int(lst.get("more") or 0),
            "describe_lines": [],
        }

        if intents["figures"]["describe"]:
            try:
                desc_text = vision_describe_figures(uid, doc_id, intents["figures"]["describe"])
                lines = [ln.strip() for ln in (desc_text or "").splitlines() if ln.strip()]
                figs_block["describe_lines"] = lines[:25]
            except Exception as e:
                figs_block["describe_lines"] = [f"Не удалось получить описания рисунков: {e}"]

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

        facts["sources"] = {
            "count": len(items),
            "list": items[:intents["sources"]["limit"]],
            "more": max(0, len(items) - intents["sources"]["limit"]),
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

        # НОВОЕ: coverage-aware выборка под многопунктный вопрос
        cov = retrieve_coverage(
            owner_id=uid,
            doc_id=doc_id,
            question=intents["general_question"],
            # per_item_k/prelim_factor/backfill_k — с разумными дефолтами из retrieval.py
        )
        ctx = ""
        if cov and cov.get("snippets"):
            ctx = build_context_coverage(
                cov["snippets"],
                items_count=len(cov.get("items") or []) or None,
                # base_chars/per_item_bonus/hard_limit — дефолты из retrieval.py
            )

        # Фолбэк-ступени, если coverage-контекст не набрался
        if not ctx:
            ctx = best_context(uid, doc_id, intents["general_question"], max_chars=6000)
        if not ctx:
            hits = retrieve(uid, doc_id, intents["general_question"], top_k=12)  # было 8 → чуть шире
            if hits:
                ctx = build_context(hits)
        if not ctx:
            ctx = _first_chunks_context(uid, doc_id, n=12, max_chars=6000)

        if ctx:
            facts["general_ctx"] = ctx
        if vb:
            facts["verbatim_hits"] = vb
        # передаём подпункты и в coverage.items (для [Items]), и в general_subitems (для многошаговой подачи)
        if cov and cov.get("items"):
            facts["coverage"] = {"items": cov["items"]}
            # нормализуем general_subitems под многошаговый режим (id+ask)
            facts["general_subitems"] = [
                {"id": i + 1, "ask": s} if isinstance(s, str) else s
                for i, s in enumerate(cov["items"])
            ]

    # логируем маленький срез фактов (без огромных текстов)
    log_snapshot = dict(facts)
    if "general_ctx" in log_snapshot and isinstance(log_snapshot["general_ctx"], str):
        log_snapshot["general_ctx"] = log_snapshot["general_ctx"][:300] + "…" if len(log_snapshot["general_ctx"]) > 300 else log_snapshot["general_ctx"]
    if "summary_text" in log_snapshot and isinstance(log_snapshot["summary_text"], str):
        log_snapshot["summary_text"] = log_snapshot["summary_text"][:300] + "…" if len(log_snapshot["summary_text"]) > 300 else log_snapshot["summary_text"]
    logging.debug("FACTS: %s", json.dumps(log_snapshot, ensure_ascii=False))
    return facts


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
    if (Cfg.FULLREAD_MODE or "off") != "direct":
        return None

    full_text = _full_document_text(uid, doc_id, limit_chars=Cfg.DIRECT_MAX_CHARS + 1)
    if not full_text.strip():
        return None

    if len(full_text) > Cfg.DIRECT_MAX_CHARS:
        return None

    system_prompt = (
        "Ты ассистент по дипломным работам. Тебе дан ПОЛНЫЙ текст ВКР/документа.\n"
        "Отвечай строго по этому тексту, без внешних фактов. Если данных недостаточно — скажи об этом.\n"
        "Если вопрос про таблицы/рисунки — используй подписи и текст рядом; не придумывай номера/значения.\n"
        "Цитируй короткими фрагментами при необходимости, без ссылок на страницы."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": f"[Документ — полный текст]\n{full_text}"},
        {"role": "user", "content": q_text},
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
            text = "\n".join([t for t in buf if t.strip()])
            if text.strip():
                out.append(text.strip())
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
    reduce_tokens = int(getattr(Cfg, "FINAL_MAX_TOKENS", 900))

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
        "Собери из них связный ответ на вопрос. Не выдумывай новых цифр/таблиц. "
        "Если данных не хватает — отдельной строкой перечисли, чего не хватает."
    )
    messages = [
        {"role": "system", "content": sys_reduce},
        {"role": "assistant", "content": f"Сводные факты из документа:\n{ctx}"},
        {"role": "user", "content": question},
    ]
    return messages, None

# ------------------------------ загрузка файла ------------------------------

@dp.message(F.document)
async def handle_doc(m: types.Message):
    uid = ensure_user(str(m.from_user.id))
    doc = m.document

    # 1) скачиваем
    file = await bot.get_file(doc.file_id)
    stream = await bot.download_file(file.file_path)
    data = stream.read()
    stream.close()

    # 2) дедуп по содержимому + file_unique_id
    sha256 = sha256_bytes(data)
    file_uid = getattr(doc, "file_unique_id", None)

    con = get_conn()
    existing_id = _find_existing_doc(con, uid, sha256, file_uid)

    if existing_id:
        existing_ver = get_document_indexer_version(existing_id) or 0

        need_reindex = False
        try:
            if _needs_reindex_by_embeddings(con, existing_id):
                need_reindex = True
        except Exception:
            need_reindex = True

        if existing_ver < CURRENT_INDEXER_VERSION:
            need_reindex = True

        if need_reindex:
            filename = safe_filename(f"{m.from_user.id}_{doc.file_name}")
            path = save_upload(data, filename, Cfg.UPLOAD_DIR)
            update_document_meta(existing_id, path=path, content_sha256=sha256, file_uid=file_uid)
            con.close()

            try:
                sections = _parse_by_ext(path)
                sections = enrich_sections(sections, doc_kind=os.path.splitext(path)[1].lower().strip("."))
                if sum(len(s.get("text") or "") for s in sections) < 500 and not any(
                    s.get("element_type") in ("table", "table_row", "figure") for s in sections
                ):
                    await _send(m, "Похоже, файл не содержит текста/структур. Убедитесь, что загружен .docx с «живыми» таблицами. Если таблицы были картинками — я их распознаю автоматически.")
                    return

                delete_document_chunks(existing_id, uid)
                index_document(uid, existing_id, sections)
                invalidate_cache(uid, existing_id)

                set_document_indexer_version(existing_id, CURRENT_INDEXER_VERSION)
                update_document_meta(existing_id, layout_profile=_current_embedding_profile())

            except Exception as e:
                await _send(m, f"Не удалось переиндексировать документ #{existing_id}: {e}")
                return

            ACTIVE_DOC[uid] = existing_id
            set_user_active_doc(uid, existing_id)
            caption = (m.caption or "").strip()
            await _send(m, f"Документ #{existing_id} переиндексирован. Готов отвечать.")
            if caption:
                await respond_with_answer(m, uid, existing_id, caption)
            return

        con.close()
        ACTIVE_DOC[uid] = existing_id
        set_user_active_doc(uid, existing_id)
        caption = (m.caption or "").strip()
        await _send(m, f"Этот файл уже загружен ранее как документ #{existing_id}. Использую его.")
        if caption:
            await respond_with_answer(m, uid, existing_id, caption)
        return

    # 3) сохраняем
    filename = safe_filename(f"{m.from_user.id}_{doc.file_name}")
    path = save_upload(data, filename, Cfg.UPLOAD_DIR)

    # 4) парсим и ОБОГАЩАЕМ
    try:
        sections = _parse_by_ext(path)
        sections = enrich_sections(sections, doc_kind=os.path.splitext(path)[1].lower().strip("."))
    except Exception as e:
        await _send(m, f"Не удалось обработать файл: {e}")
        return

    # 5) проверка объёма — уже после enrich
    if sum(len(s.get("text") or "") for s in sections) < 500 and not any(
        s.get("element_type") in ("table", "table_row", "figure") for s in sections
    ):
        await _send(m, "Похоже, файл не содержит текста/структур. Загрузите текстовый DOC/DOCX; таблицы-картинки я распознаю автоматически.")
        return

    # 6) документ → БД и индексация
    kind = infer_doc_kind(doc.file_name)
    doc_id = _insert_document(con, uid, kind, path, sha256, file_uid)
    con.close()

    index_document(uid, doc_id, sections)
    invalidate_cache(uid, doc_id)
    set_document_indexer_version(doc_id, CURRENT_INDEXER_VERSION)
    update_document_meta(doc_id, layout_profile=_current_embedding_profile())

    ACTIVE_DOC[uid] = doc_id
    set_user_active_doc(uid, doc_id)

    caption = (m.caption or "").strip()
    if caption:
        await _send(m, f"Документ #{doc_id} проиндексирован. Отвечаю на ваш вопрос из подписи…")
        await respond_with_answer(m, uid, doc_id, caption)
    else:
        await _send(m, f"Готово. Документ #{doc_id} проиндексирован. Можете задавать вопросы по работе.")


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

    # ====== FULLREAD: auto ======
    mode = (Cfg.FULLREAD_MODE or "off")
    if mode == "auto":
        # пробуем дать модели ПОЛНЫЙ текст, если влазит
        full_text = _full_document_text(uid, doc_id, limit_chars=Cfg.DIRECT_MAX_CHARS + 1)
        if full_text and len(full_text) <= Cfg.DIRECT_MAX_CHARS:
            system_prompt = (
                "Ты ассистент по дипломным работам. Тебе дан ПОЛНЫЙ текст ВКР/документа.\n"
                "Отвечай строго по этому тексту, без внешних фактов. Если данных недостаточно — скажи об этом.\n"
                "Если вопрос про таблицы/рисунки — используй подписи и текст рядом; не придумывай номера/значения.\n"
                "Цитируй короткими фрагментами при необходимости, без ссылок на страницы."
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "assistant", "content": f"[Документ — полный текст]\n{full_text}"},
                {"role": "user", "content": q_text},
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
                    await _send(m, ans)
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
                        await _send(m, ans)
                        return
                except Exception as e:
                    logging.exception("auto iterative non-stream failed: %s", e)
            elif err:
                await _send(m, err)
                return


    # ====== FULLREAD: direct ======
    if (Cfg.FULLREAD_MODE or "off") == "direct":
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
            await _send(m, fr)
            return
        # иначе — RAG ниже

    # ====== FULLREAD: iterative/digest ======
    if (Cfg.FULLREAD_MODE or "off") in {"iterative", "digest"}:
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
                    await _send(m, ans)
                    return
            except Exception as e:
                logging.exception("iterative fullread non-stream failed: %s", e)
        else:
            if err:
                await _send(m, err)
                return
        # если что-то не вышло — проваливаемся в стандартный режим ниже

    # ====== Стандартный мульти-интент пайплайн (RAG) ======
    intents = detect_intents(q_text)
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

    # обычный путь
    # app/bot.py
    if STREAM_ENABLED and generate_answer_stream is not None:
        try:
            stream = generate_answer_stream(q_text, facts, language=intents.get("language", "ru"))
            await _stream_to_telegram(m, stream)
            return

        except Exception as e:
            logging.exception("stream answer failed, fallback to non-stream: %s", e)

    reply = generate_answer(q_text, facts, language=intents.get("language", "ru"))
    await _send(m, reply)


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

    # НОВОЕ: если в памяти нет — поднимем из БД (устойчивость к рестартам процесса)
    if not doc_id:
        persisted = get_user_active_doc(uid)
        if persisted:
            ACTIVE_DOC[uid] = persisted
            doc_id = persisted

    text = (m.text or "").strip()

    # Строгий режим: без активного документа не отвечаем по содержанию
    if not doc_id:
        if _is_greeting(text):
            await start(m)
        else:
            await _send(m, "Сначала пришлите файл (.doc/.docx). Без него я не отвечаю по содержанию.")
        return

    await respond_with_answer(m, uid, doc_id, text)
