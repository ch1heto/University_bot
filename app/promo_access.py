# app/promo_access.py
from __future__ import annotations

import os
import re
import sqlite3
import time
import secrets
import asyncio
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

# aiogram v3
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command

try:
    from aiogram import BaseMiddleware
except Exception:  # pragma: no cover
    from aiogram.dispatcher.middlewares.base import BaseMiddleware  # type: ignore

logger = logging.getLogger(__name__)

# Ожидание ответа админа для /promo_create (в памяти процесса)
_PENDING_PROMO_CREATE: dict[int, int] = {}  # admin_user_id -> created_ts
PROMO_CREATE_MIN = 1
PROMO_CREATE_MAX = 100

# ──────────────────────────────────────────────────────────────────────────────
# ENV helpers
# ──────────────────────────────────────────────────────────────────────────────

def _env_str(name: str, default: str = "") -> str:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val)

def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name, "")).strip())
    except Exception:
        return default

def _env_csv_ints(name: str, fallback_name: Optional[str] = None) -> set[int]:
    raw = os.getenv(name)
    if not raw and fallback_name:
        raw = os.getenv(fallback_name, "")
    raw = (raw or "").replace(";", ",")
    out: set[int] = set()
    for part in raw.split(","):
        part = part.strip().strip("'\"")
        if not part:
            continue
        try:
            out.add(int(part))
        except Exception:
            continue
    return out

def _env_csv_ints_list(name: str, default: str) -> list[int]:
    raw = _env_str(name, default).replace(";", ",")
    res: list[int] = []
    for p in raw.split(","):
        p = p.strip()
        if not p:
            continue
        try:
            res.append(int(p))
        except Exception:
            continue
    seen = set()
    uniq: list[int] = []
    for x in res:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq

# ──────────────────────────────────────────────────────────────────────────────
# Config (важно: значения читаются при импорте модуля)
# ──────────────────────────────────────────────────────────────────────────────

PROMO_DISABLED = _env_str("PROMO_DISABLED", "0").lower() in {"1", "true", "yes", "y", "on"}

# Админы: PROMO_ADMIN_IDS предпочтительнее, но поддержим ADMIN_IDS
ADMIN_IDS = _env_csv_ints("PROMO_ADMIN_IDS", fallback_name="ADMIN_IDS")

PROMO_DURATION_HOURS = _env_int("PROMO_DURATION_HOURS", 72)
PROMO_DURATION_SEC = max(1, PROMO_DURATION_HOURS) * 3600

PROMO_DB_PATH = _env_str("PROMO_DB_PATH", os.path.join(os.path.dirname(__file__), "promo_store.sqlite3"))

# Уведомления
PROMO_WARN_HOURS = _env_csv_ints_list("PROMO_WARN_HOURS", "24,7,1")
PROMO_WARN_CHECK_INTERVAL_SEC = max(30, _env_int("PROMO_WARN_CHECK_INTERVAL_SEC", 300))

# timezone для отображения
PROMO_TZ = _env_str("PROMO_TZ", "Europe/Amsterdam")

# Формат кода: XXXX-XXXX-XXXX
CODE_RE = re.compile(r"^[A-Z0-9]{4}(?:-[A-Z0-9]{4}){2}$")

# ──────────────────────────────────────────────────────────────────────────────
# DB schema
# ──────────────────────────────────────────────────────────────────────────────

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS promo_codes (
  code TEXT PRIMARY KEY,
  created_at_ts INTEGER NOT NULL,
  created_by_admin INTEGER NOT NULL,
  duration_sec INTEGER NOT NULL,
  max_redemptions INTEGER NOT NULL DEFAULT 1,
  redeemed_count INTEGER NOT NULL DEFAULT 0,
  is_revoked INTEGER NOT NULL DEFAULT 0,
  expires_at_ts INTEGER
);

CREATE TABLE IF NOT EXISTS user_access (
  user_id INTEGER PRIMARY KEY,
  activated_at_ts INTEGER NOT NULL,
  access_until_ts INTEGER NOT NULL,
  activated_by_code TEXT,
  updated_at_ts INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS promo_redemptions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  code TEXT NOT NULL,
  user_id INTEGER NOT NULL,
  redeemed_at_ts INTEGER NOT NULL,
  access_until_ts INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_redemptions_code ON promo_redemptions(code);
CREATE INDEX IF NOT EXISTS idx_redemptions_user ON promo_redemptions(user_id);

-- Уведомления: одно уведомление на порог для конкретного периода доступа (access_until_ts)
CREATE TABLE IF NOT EXISTS access_notifications (
  user_id INTEGER NOT NULL,
  access_until_ts INTEGER NOT NULL,
  threshold_hours INTEGER NOT NULL,
  sent_at_ts INTEGER NOT NULL,
  PRIMARY KEY (user_id, access_until_ts, threshold_hours)
);

CREATE INDEX IF NOT EXISTS idx_access_until ON user_access(access_until_ts);
"""

@contextmanager
def _db():
    con = sqlite3.connect(PROMO_DB_PATH, timeout=30, isolation_level=None)
    try:
        yield con
    finally:
        con.close()

def _init_db() -> None:
    with _db() as con:
        con.executescript(SCHEMA_SQL)

def _now_ts() -> int:
    return int(time.time())

def _is_admin(user_id: int) -> bool:
    return user_id in ADMIN_IDS

def _gen_code() -> str:
    raw = secrets.token_hex(6).upper()  # 12 hex chars
    return f"{raw[0:4]}-{raw[4:8]}-{raw[8:12]}"

# ──────────────────────────────────────────────────────────────────────────────
# Access logic
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AccessStatus:
    is_active: bool
    access_until_ts: Optional[int]
    seconds_left: Optional[int]

def get_access_status(user_id: int) -> AccessStatus:
    ts = _now_ts()
    with _db() as con:
        row = con.execute("SELECT access_until_ts FROM user_access WHERE user_id=?", (int(user_id),)).fetchone()
    if not row:
        return AccessStatus(False, None, None)
    until = int(row[0])
    left = until - ts
    if left <= 0:
        return AccessStatus(False, until, 0)
    return AccessStatus(True, until, left)

def _format_until_local(ts_epoch: int) -> str:
    try:
        from zoneinfo import ZoneInfo  # py3.9+
        tz = ZoneInfo(PROMO_TZ)
        dt = datetime.fromtimestamp(ts_epoch, tz=tz)
        return dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        dt = datetime.fromtimestamp(ts_epoch, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")

def _format_left(seconds_left: int) -> str:
    if seconds_left <= 0:
        return "0 мин"
    mins = seconds_left // 60
    hours = mins // 60
    mins = mins % 60
    if hours <= 0:
        return f"{mins} мин"
    return f"{hours} ч {mins} мин"

def create_promo_code(admin_id: int, duration_sec: int = PROMO_DURATION_SEC, max_redemptions: int = 1) -> str:
    code = _gen_code()
    ts = _now_ts()
    with _db() as con:
        con.execute("BEGIN IMMEDIATE")
        con.execute(
            "INSERT INTO promo_codes(code, created_at_ts, created_by_admin, duration_sec, max_redemptions) VALUES(?,?,?,?,?)",
            (code, ts, int(admin_id), int(duration_sec), int(max_redemptions)),
        )
        con.execute("COMMIT")
    return code

def revoke_promo_code(code: str) -> bool:
    code = (code or "").strip().upper()
    if not code:
        return False
    with _db() as con:
        con.execute("BEGIN IMMEDIATE")
        cur = con.execute("UPDATE promo_codes SET is_revoked=1 WHERE code=? AND redeemed_count=0", (code,))
        con.execute("COMMIT")
        return cur.rowcount > 0

def promo_info(code: str) -> Optional[dict]:
    code = (code or "").strip().upper()
    if not code:
        return None
    with _db() as con:
        row = con.execute(
            """SELECT code, created_at_ts, created_by_admin, duration_sec, max_redemptions,
                      redeemed_count, is_revoked, expires_at_ts
               FROM promo_codes WHERE code=?""",
            (code,),
        ).fetchone()
    if not row:
        return None
    return {
        "code": row[0],
        "created_at_ts": int(row[1]),
        "created_by_admin": int(row[2]),
        "duration_sec": int(row[3]),
        "max_redemptions": int(row[4]),
        "redeemed_count": int(row[5]),
        "is_revoked": int(row[6]),
        "expires_at_ts": None if row[7] is None else int(row[7]),
    }

def redeem_code(user_id: int, code: str) -> tuple[bool, str, Optional[int]]:
    """
    Возвращает (ok, message, access_until_ts).
    Продление: доливаем время:
      new_until = max(current_until, now) + duration
    Промокод одноразовый: max_redemptions=1.
    """
    code = (code or "").strip().upper()
    if not CODE_RE.match(code):
        return False, "Неверный формат промокода. Пример: ABCD-EFGH-1234", None

    now = _now_ts()

    with _db() as con:
        con.execute("BEGIN IMMEDIATE")

        row = con.execute(
            """SELECT duration_sec, max_redemptions, redeemed_count, is_revoked, expires_at_ts
               FROM promo_codes WHERE code=?""",
            (code,),
        ).fetchone()

        if not row:
            con.execute("ROLLBACK")
            return False, "Промокод не найден.", None

        duration_sec, max_red, redeemed_count, is_revoked, expires_at_ts = row
        duration_sec = int(duration_sec)
        max_red = int(max_red)
        redeemed_count = int(redeemed_count)
        is_revoked = int(is_revoked)

        if is_revoked:
            con.execute("ROLLBACK")
            return False, "Промокод отозван.", None

        if expires_at_ts is not None and now >= int(expires_at_ts):
            con.execute("ROLLBACK")
            return False, "Срок действия промокода истёк.", None

        if redeemed_count >= max_red:
            con.execute("ROLLBACK")
            return False, "Промокод уже использован.", None

        # текущий доступ (если есть)
        cur_row = con.execute(
            "SELECT access_until_ts FROM user_access WHERE user_id=?",
            (int(user_id),),
        ).fetchone()

        current_until = int(cur_row[0]) if cur_row else 0
        base = current_until if current_until > now else now
        new_until = base + duration_sec

        # upsert user_access
        con.execute(
            """
            INSERT INTO user_access(user_id, activated_at_ts, access_until_ts, activated_by_code, updated_at_ts)
            VALUES(?,?,?,?,?)
            ON CONFLICT(user_id) DO UPDATE SET
              activated_at_ts=excluded.activated_at_ts,
              access_until_ts=excluded.access_until_ts,
              activated_by_code=excluded.activated_by_code,
              updated_at_ts=excluded.updated_at_ts
            """,
            (int(user_id), now, int(new_until), code, now),
        )

        # помечаем промокод использованным
        con.execute(
            "UPDATE promo_codes SET redeemed_count = redeemed_count + 1 WHERE code=?",
            (code,),
        )

        # аудит
        con.execute(
            "INSERT INTO promo_redemptions(code, user_id, redeemed_at_ts, access_until_ts) VALUES(?,?,?,?)",
            (code, int(user_id), now, int(new_until)),
        )

        con.execute("COMMIT")

    msg = (
        "Доступ активирован.\n"
        f"Действует до: {_format_until_local(new_until)}\n"
        f"Осталось: {_format_left(int(new_until - now))}\n\n"
        "Проверить статус: /status"
    )
    return True, msg, int(new_until)

# ──────────────────────────────────────────────────────────────────────────────
# Middleware: блокируем всё, кроме разрешённых команд
# ──────────────────────────────────────────────────────────────────────────────

_ALLOWED_PREFIXES = (
    "/start",
    "/help",
    "/redeem",
    "/status",
    "/promo_create",
    "/promo_revoke",
    "/promo_info",
)

class PromoAccessMiddleware(BaseMiddleware):
    async def __call__(self, handler, event: types.TelegramObject, data):
        if PROMO_DISABLED:
            return await handler(event, data)

        # messages
        if isinstance(event, types.Message):
            if not event.from_user:
                return await handler(event, data)

            user_id = int(event.from_user.id)
            text = (event.text or "").strip()

            # команды
            if text.startswith("/"):
                # разрешённые команды
                if any(text.startswith(p) for p in _ALLOWED_PREFIXES):
                    if text.startswith("/promo_") and not _is_admin(user_id):
                        await event.answer("Недостаточно прав.")
                        return
                    return await handler(event, data)

                # остальные команды
                st = get_access_status(user_id)
                if not st.is_active and not _is_admin(user_id):
                    await event.answer("Доступ закрыт. Активируйте промокод: /redeem ABCD-EFGH-1234")
                    return
                return await handler(event, data)

            # не команда (текст/документы/медиа)
            if _is_admin(user_id):
                return await handler(event, data)

            st = get_access_status(user_id)
            if not st.is_active:
                await event.answer("Доступ закрыт. Активируйте промокод: /redeem ABCD-EFGH-1234")
                return

            return await handler(event, data)

        # callback queries
        if isinstance(event, types.CallbackQuery):
            if not event.from_user:
                return await handler(event, data)

            user_id = int(event.from_user.id)
            if _is_admin(user_id):
                return await handler(event, data)

            st = get_access_status(user_id)
            if not st.is_active:
                try:
                    await event.answer("Доступ закрыт. Активируйте промокод: /redeem ABCD-EFGH-1234", show_alert=True)
                except Exception:
                    pass
                return

            return await handler(event, data)

        return await handler(event, data)

# ──────────────────────────────────────────────────────────────────────────────
# Handlers (НЕ через Router, а обычные функции для dp.message.register)
# ──────────────────────────────────────────────────────────────────────────────

async def redeem_handler(message: types.Message):
    if PROMO_DISABLED:
        await message.answer("Система доступа по промокодам отключена.")
        return

    parts = (message.text or "").split(maxsplit=1)
    if len(parts) != 2:
        await message.answer("Использование: /redeem ABCD-EFGH-1234")
        return

    ok, msg, _ = redeem_code(int(message.from_user.id), parts[1])
    await message.answer(msg)

async def status_handler(message: types.Message):
    if PROMO_DISABLED:
        await message.answer("Система доступа по промокодам отключена.")
        return

    user_id = int(message.from_user.id)
    st = get_access_status(user_id)
    if not st.access_until_ts:
        await message.answer("Доступ не активирован. Активируйте промокод: /redeem ABCD-EFGH-1234")
        return

    if st.is_active:
        await message.answer(
            "Доступ активен.\n"
            f"Действует до: {_format_until_local(st.access_until_ts)}\n"
            f"Осталось: {_format_left(int(st.seconds_left or 0))}"
        )
    else:
        await message.answer(
            "Доступ истёк.\n"
            f"Истёк: {_format_until_local(st.access_until_ts)}\n\n"
            "Активируйте новый промокод: /redeem ABCD-EFGH-1234"
        )

async def promo_create_handler(message: types.Message):
    if PROMO_DISABLED:
        await message.answer("Система доступа по промокодам отключена.")
        return

    user_id = int(message.from_user.id)
    if not _is_admin(user_id):
        await message.answer("Недостаточно прав.")
        return

    parts = (message.text or "").split(maxsplit=1)

    # Если админ сразу передал число: /promo_create 10
    if len(parts) == 2:
        raw_n = parts[1].strip()
        if not raw_n.isdigit():
            await message.answer(f"Нужно число. Пример: /promo_create 10 (от {PROMO_CREATE_MIN} до {PROMO_CREATE_MAX})")
            return
        n = int(raw_n)
        if n < PROMO_CREATE_MIN or n > PROMO_CREATE_MAX:
            await message.answer(f"Число должно быть от {PROMO_CREATE_MIN} до {PROMO_CREATE_MAX}.")
            return

        codes = [create_promo_code(admin_id=user_id, duration_sec=PROMO_DURATION_SEC, max_redemptions=1) for _ in range(n)]
        await message.answer(
            f"Сгенерировано {n} промокодов (одноразовые, {PROMO_DURATION_HOURS} ч):\n" + "\n".join(codes)
        )
        return

    # Иначе — интерактивный режим: спрашиваем количество
    _PENDING_PROMO_CREATE[user_id] = _now_ts()
    await message.answer(
        f"Сколько промокодов сгенерировать? Отправьте число от {PROMO_CREATE_MIN} до {PROMO_CREATE_MAX}.\n"
        "Отмена: /cancel"
    )

async def cancel_handler(message: types.Message):
    user_id = int(message.from_user.id)
    if user_id in _PENDING_PROMO_CREATE:
        _PENDING_PROMO_CREATE.pop(user_id, None)
        await message.answer("Операция отменена.")
    else:
        await message.answer("Нет активной операции для отмены.")


async def promo_revoke_handler(message: types.Message):
    if PROMO_DISABLED:
        await message.answer("Система доступа по промокодам отключена.")
        return

    user_id = int(message.from_user.id)
    if not _is_admin(user_id):
        await message.answer("Недостаточно прав.")
        return

    parts = (message.text or "").split(maxsplit=1)
    if len(parts) != 2:
        await message.answer("Использование: /promo_revoke ABCD-EFGH-1234")
        return

    ok = revoke_promo_code(parts[1])
    if ok:
        await message.answer("Промокод отозван (если он ещё не был использован).")
    else:
        await message.answer("Не удалось отозвать: код не найден или уже использован.")

async def promo_create_count_input_handler(message: types.Message):
    # Срабатывает только для админов, у кого есть pending create
    if PROMO_DISABLED:
        return

    if not message.from_user:
        return
    user_id = int(message.from_user.id)

    if not _is_admin(user_id):
        return
    if user_id not in _PENDING_PROMO_CREATE:
        return

    text = (message.text or "").strip()

    # Если вдруг прислали команду — игнорируем (пусть обработают другие handlers)
    if text.startswith("/"):
        return

    if not text.isdigit():
        await message.answer(f"Нужно число от {PROMO_CREATE_MIN} до {PROMO_CREATE_MAX}. Отмена: /cancel")
        return

    n = int(text)
    if n < PROMO_CREATE_MIN or n > PROMO_CREATE_MAX:
        await message.answer(f"Число должно быть от {PROMO_CREATE_MIN} до {PROMO_CREATE_MAX}. Отмена: /cancel")
        return

    _PENDING_PROMO_CREATE.pop(user_id, None)

    codes = [create_promo_code(admin_id=user_id, duration_sec=PROMO_DURATION_SEC, max_redemptions=1) for _ in range(n)]
    await message.answer(
        f"Сгенерировано {n} промокодов (одноразовые, {PROMO_DURATION_HOURS} ч):\n" + "\n".join(codes)
    )


async def promo_info_handler(message: types.Message):
    if PROMO_DISABLED:
        await message.answer("Система доступа по промокодам отключена.")
        return

    user_id = int(message.from_user.id)
    if not _is_admin(user_id):
        await message.answer("Недостаточно прав.")
        return

    parts = (message.text or "").split(maxsplit=1)
    if len(parts) != 2:
        await message.answer("Использование: /promo_info ABCD-EFGH-1234")
        return

    info = promo_info(parts[1])
    if not info:
        await message.answer("Код не найден.")
        return

    created_dt = datetime.fromtimestamp(info["created_at_ts"], tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    expires_txt = "—"
    if info["expires_at_ts"]:
        expires_txt = datetime.fromtimestamp(info["expires_at_ts"], tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    await message.answer(
        "Информация по промокоду:\n"
        f"Код: {info['code']}\n"
        f"Создан: {created_dt}\n"
        f"Создал (admin_id): {info['created_by_admin']}\n"
        f"Длительность: {int(info['duration_sec']) // 3600} ч\n"
        f"Использований: {info['redeemed_count']}/{info['max_redemptions']}\n"
        f"Отозван: {'да' if info['is_revoked'] else 'нет'}\n"
        f"Годен до (UTC): {expires_txt}"
    )

# ──────────────────────────────────────────────────────────────────────────────
# Notifications worker
# ──────────────────────────────────────────────────────────────────────────────

async def _send_warning_if_needed(bot: Bot, user_id: int, access_until_ts: int, threshold_h: int) -> None:
    now = _now_ts()
    with _db() as con:
        try:
            con.execute(
                "INSERT INTO access_notifications(user_id, access_until_ts, threshold_hours, sent_at_ts) VALUES(?,?,?,?)",
                (int(user_id), int(access_until_ts), int(threshold_h), int(now)),
            )
        except sqlite3.IntegrityError:
            return

    try:
        await bot.send_message(
            chat_id=user_id,
            text=(
                f"Напоминание: до окончания доступа осталось примерно {threshold_h} ч.\n"
                "Проверить остаток: /status"
            ),
        )
    except Exception as e:
        logger.warning("Failed to send promo warning to %s: %s", user_id, e)

async def promo_notifications_worker(bot: Bot, stop_event: asyncio.Event) -> None:
    if PROMO_DISABLED:
        return

    thresholds = [h for h in PROMO_WARN_HOURS if h > 0]
    if not thresholds:
        return

    max_h = max(thresholds)

    while not stop_event.is_set():
        now = _now_ts()
        try:
            with _db() as con:
                rows = con.execute(
                    "SELECT user_id, access_until_ts FROM user_access WHERE access_until_ts > ? AND access_until_ts <= ?",
                    (int(now), int(now + max_h * 3600)),
                ).fetchall()

            for user_id, access_until_ts in rows:
                user_id = int(user_id)
                access_until_ts = int(access_until_ts)
                left = access_until_ts - now

                for h in thresholds:
                    if left <= h * 3600:
                        await _send_warning_if_needed(bot, user_id, access_until_ts, h)

        except Exception as e:
            logger.exception("promo_notifications_worker error: %s", e)

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=PROMO_WARN_CHECK_INTERVAL_SEC)
        except asyncio.TimeoutError:
            pass

# ──────────────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────────────

def setup_promo_access(dp: Dispatcher, bot: Bot) -> None:
    """
    Подключение:
      - инициализация БД
      - middleware на dp.update
      - регистрация команд через dp.message.register(...)
      - startup/shutdown для воркера уведомлений
    """
    _init_db()

    dp.update.middleware(PromoAccessMiddleware())

    # Регистрируем команды напрямую на dp (это критично для вашего случая)
    dp.message.register(redeem_handler, Command("redeem"))
    dp.message.register(status_handler, Command("status"))
    dp.message.register(help_handler, Command("help"))
    dp.message.register(promo_create_handler, Command("promo_create"))
    dp.message.register(promo_revoke_handler, Command("promo_revoke"))
    dp.message.register(promo_info_handler, Command("promo_info"))
    dp.message.register(cancel_handler, Command("cancel"))

    dp.message.register(promo_create_count_input_handler, lambda m: bool(getattr(m, "text", None)))

    stop_event = asyncio.Event()
    task_holder: dict[str, asyncio.Task] = {}

    async def _on_startup():
        if PROMO_DISABLED:
            return
        task_holder["task"] = asyncio.create_task(promo_notifications_worker(bot, stop_event))
        logger.info(
            "promo_access: notifications worker started; interval=%ss thresholds=%s",
            PROMO_WARN_CHECK_INTERVAL_SEC,
            PROMO_WARN_HOURS,
        )

    async def _on_shutdown():
        stop_event.set()
        t = task_holder.get("task")
        if t:
            t.cancel()
            try:
                await t
            except Exception:
                pass
        logger.info("promo_access: notifications worker stopped")

    dp.startup.register(_on_startup)
    dp.shutdown.register(_on_shutdown)

async def help_handler(message: types.Message):
    # /help должен быть доступен даже без активного доступа
    user_id = int(message.from_user.id)

    if _is_admin(user_id):
        await message.answer(
            "Команды администратора:\n"
            "/promo_create — создать одноразовый промокод на 72 часа\n"
            "/promo_info <КОД> — информация по промокоду (создатель, использован/нет, отозван/нет)\n"
            "/promo_revoke <КОД> — отозвать промокод (если ещё не использован)\n\n"
            "Команды пользователя:\n"
            "/redeem <КОД> — активировать промокод (доступ на 72 часа, время доливается)\n"
            "/status — проверить, сколько времени осталось\n"
        )
        return

    await message.answer(
        "Доступ к боту выдаётся по промокоду на 72 часа.\n\n"
        "Команды:\n"
        "/redeem <КОД> — активировать промокод\n"
        "/status — проверить, сколько времени осталось\n"
        "/help — помощь\n"
    )
