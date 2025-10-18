# app/paywall_stub.py
from __future__ import annotations

import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Optional, Tuple

# aiogram v3
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
try:
    # aiogram 3.x base middleware
    from aiogram import BaseMiddleware
except Exception:  # совместимость на всякий случай
    from aiogram.dispatcher.middlewares.base import BaseMiddleware  # type: ignore
from aiogram import Router

# ──────────────────────────────────────────────────────────────────────────────
# Конфигурация из окружения / .env (если python-dotenv установлен — подцепится)
# ──────────────────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv  # мягко
    load_dotenv()
except Exception:
    pass

def _env_int(key: str, default: int) -> int:
    """Безопасно читаем int из .env (обрезаем пробелы и кавычки)."""
    raw = os.getenv(key, str(default))
    if raw is None:
        return default
    raw = raw.strip().strip("'\"")
    try:
        return int(raw)
    except Exception:
        return default

def _env_str(key: str, default: str) -> str:
    raw = os.getenv(key, default)
    if raw is None:
        return default
    return raw.strip().strip("'\"")

FREE_MESSAGES = _env_int("PAYWALL_FREE_MESSAGES", 10)
SUB_DAYS = _env_int("PAYWALL_SUB_DAYS", 30)

AMOUNT = _env_int("PAYWALL_AMOUNT", 299)
CURRENCY = _env_str("PAYWALL_CURRENCY", "RUB")

PAY_LINK_TEMPLATE = _env_str(
    "PAYWALL_PAY_LINK_TEMPLATE",
    "https://yookassa.ru/demo?order={invoice_id}&user={user_id}",
)

# Админы: CSV/SSV со списком Telegram ID (поддерживаем ',' и ';')
def _parse_admin_ids(env_val: Optional[str]) -> set[int]:
    ids: set[int] = set()
    if not env_val:
        # бэкап: одиночная переменная PAYWALL_ADMIN_ID
        env_val = os.getenv("PAYWALL_ADMIN_ID", "")
    env_val = (env_val or "").replace(";", ",")
    for part in env_val.split(","):
        part = part.strip().strip("'\"")
        if not part:
            continue
        try:
            ids.add(int(part))
        except Exception:
            pass
    return ids

ADMIN_IDS = _parse_admin_ids(os.getenv("PAYWALL_ADMIN_IDS"))

# ──────────────────────────────────────────────────────────────────────────────
# Хранилище (отдельный SQLite, чтобы не трогать вашу БД)
# ──────────────────────────────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "paywall_store.sqlite3")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
  user_id INTEGER PRIMARY KEY,
  free_used INTEGER NOT NULL DEFAULT 0,
  sub_until_ts INTEGER,            -- UTC epoch seconds
  updated_ts INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS invoices (
  invoice_id TEXT PRIMARY KEY,
  user_id INTEGER NOT NULL,
  amount INTEGER NOT NULL,
  currency TEXT NOT NULL,
  status TEXT NOT NULL,            -- 'pending' | 'succeeded' | 'canceled'
  created_ts INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_invoices_user ON invoices(user_id);
"""

def _utc_now_ts() -> int:
    return int(time.time())

@contextmanager
def _db() -> Iterable[sqlite3.Connection]:
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()

def _init_db() -> None:
    with _db() as con:
        con.executescript(SCHEMA_SQL)

def _ensure_user(user_id: int) -> None:
    with _db() as con:
        con.execute(
            "INSERT OR IGNORE INTO users(user_id, free_used, sub_until_ts, updated_ts) VALUES(?,?,?,?)",
            (user_id, 0, None, _utc_now_ts()),
        )

def _get_user_state(user_id: int) -> Tuple[int, Optional[int]]:
    with _db() as con:
        row = con.execute(
            "SELECT free_used, sub_until_ts FROM users WHERE user_id=?",
            (user_id,),
        ).fetchone()
        if not row:
            return (0, None)
        return (int(row["free_used"] or 0), (int(row["sub_until_ts"]) if row["sub_until_ts"] is not None else None))

def _set_free_used(user_id: int, value: int) -> None:
    with _db() as con:
        con.execute(
            "UPDATE users SET free_used=?, updated_ts=? WHERE user_id=?",
            (value, _utc_now_ts(), user_id),
        )

def _inc_free_used(user_id: int) -> int:
    with _db() as con:
        con.execute(
            "UPDATE users SET free_used=COALESCE(free_used,0)+1, updated_ts=? WHERE user_id=?",
            (_utc_now_ts(), user_id),
        )
        row = con.execute("SELECT free_used FROM users WHERE user_id=?", (user_id,)).fetchone()
        return int(row["free_used"] or 0)

def _set_sub_until(user_id: int, until_ts: int) -> None:
    with _db() as con:
        con.execute(
            "UPDATE users SET sub_until_ts=?, updated_ts=? WHERE user_id=?",
            (until_ts, _utc_now_ts(), user_id),
        )

def _create_invoice(user_id: int, amount: int, currency: str) -> str:
    invoice_id = uuid.uuid4().hex[:12]
    with _db() as con:
        con.execute(
            "INSERT INTO invoices(invoice_id, user_id, amount, currency, status, created_ts) VALUES(?,?,?,?,?,?)",
            (invoice_id, user_id, amount, currency, "pending", _utc_now_ts()),
        )
    return invoice_id

def _mark_invoice_succeeded(invoice_id: str) -> Optional[int]:
    with _db() as con:
        row = con.execute(
            "SELECT user_id FROM invoices WHERE invoice_id=?",
            (invoice_id,),
        ).fetchone()
        if not row:
            return None
        con.execute(
            "UPDATE invoices SET status='succeeded' WHERE invoice_id=?",
            (invoice_id,),
        )
        return int(row["user_id"])

# ──────────────────────────────────────────────────────────────────────────────
# Бизнес-правила
# ──────────────────────────────────────────────────────────────────────────────
def _is_admin(user_id: int) -> bool:
    return user_id in ADMIN_IDS

def _is_sub_active(user_id: int) -> bool:
    _, until_ts = _get_user_state(user_id)
    if until_ts is None:
        return False
    return until_ts > _utc_now_ts()

def _subscription_until_str(ts: Optional[int]) -> str:
    if not ts:
        return "—"
    dt = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone()
    return dt.strftime("%d.%m.%Y %H:%M")

def _activate_subscription(user_id: int, days: int) -> int:
    until = _utc_now_ts() + int(days) * 24 * 3600
    _set_sub_until(user_id, until)
    _set_free_used(user_id, 0)
    return until

# ──────────────────────────────────────────────────────────────────────────────
# Инлайн-кнопки (заглушка оплаты)
# ──────────────────────────────────────────────────────────────────────────────
def _pay_keyboard(invoice_id: str, user_id: int) -> types.InlineKeyboardMarkup:
    pay_url = PAY_LINK_TEMPLATE.format(invoice_id=invoice_id, user_id=user_id)
    kb = [
        [types.InlineKeyboardButton(text="💳 Оплатить подписку", url=pay_url)],
        [types.InlineKeyboardButton(text="✅ Я оплатил (заглушка)", callback_data=f"pay:confirm:{invoice_id}")],
    ]
    return types.InlineKeyboardMarkup(inline_keyboard=kb)

# ──────────────────────────────────────────────────────────────────────────────
# Middleware: отслеживание лимита и показ pay-кнопки
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class PaywallConfig:
    free_messages: int = FREE_MESSAGES
    sub_days: int = SUB_DAYS
    amount: int = AMOUNT
    currency: str = CURRENCY

class PaywallMiddleware(BaseMiddleware):
    def __init__(self, bot: Bot, cfg: PaywallConfig):
        super().__init__()
        self.bot = bot
        self.cfg = cfg

    async def __call__(self, handler, event: types.Message, data):
        # применяем только к сообщениям
        if not isinstance(event, types.Message):
            return await handler(event, data)

        user_id = event.from_user.id if event.from_user else None
        if not user_id:
            return await handler(event, data)

        _ensure_user(user_id)

        # админы не лимитируются
        if _is_admin(user_id):
            return await handler(event, data)

        # команды не учитываем и не блокируем
        text = (event.text or "").strip()
        if text.startswith("/"):
            return await handler(event, data)

        # активная подписка — пропускаем
        if _is_sub_active(user_id):
            return await handler(event, data)

        # проверяем лимит ДО инкремента: на 11-м блокируем
        used, _ = _get_user_state(user_id)
        if used >= self.cfg.free_messages:
            # показываем клавиатуру оплаты (создадим "инвойс" заглушки)
            invoice_id = _create_invoice(user_id, self.cfg.amount, self.cfg.currency)
            kb = _pay_keyboard(invoice_id, user_id)
            await event.answer(
                f"Доступно {self.cfg.free_messages} бесплатных запросов. "
                "Лимит исчерпан. Чтобы продолжить — оформите подписку.",
                reply_markup=kb
            )
            return  # прерываем цепочку обработчиков

        # иначе учитываем этот запрос и пропускаем дальше
        _inc_free_used(user_id)
        return await handler(event, data)

# ──────────────────────────────────────────────────────────────────────────────
# Хэндлеры команд и колбэков
# ──────────────────────────────────────────────────────────────────────────────
router = Router(name="paywall")

@router.message(Command("status"))
async def cmd_status(m: types.Message):
    user_id = m.from_user.id
    _ensure_user(user_id)
    used, until_ts = _get_user_state(user_id)
    remaining = max(0, FREE_MESSAGES - used)
    status = "Активна" if _is_sub_active(user_id) else "Нет"
    sub_until = _subscription_until_str(until_ts)
    await m.answer(
        "📊 Статус:\n"
        f"• Подписка: {status}\n"
        f"• Действует до: {sub_until}\n"
        f"• Бесплатных сообщений осталось: {remaining}/{FREE_MESSAGES}"
    )

@router.message(Command("buy"))
async def cmd_buy(m: types.Message):
    user_id = m.from_user.id
    _ensure_user(user_id)
    invoice_id = _create_invoice(user_id, AMOUNT, CURRENCY)
    kb = _pay_keyboard(invoice_id, user_id)
    await m.answer(
        f"Подписка на {SUB_DAYS} дней за {AMOUNT} {CURRENCY}.\n"
        "Нажмите кнопку ниже для перехода к оплате.",
        reply_markup=kb
    )

@router.message(Command("reset_trial"))
async def cmd_reset_trial(m: types.Message):
    user_id = m.from_user.id
    if not _is_admin(user_id):
        await m.answer("Эта команда доступна только администраторам.")
        return
    # можно ресетить кому-то другому: /reset_trial <user_id>
    target_id = user_id
    parts = (m.text or "").split()
    if len(parts) > 1:
        try:
            target_id = int(parts[1])
        except Exception:
            await m.answer("Неверный user_id. Пример: /reset_trial 123456789")
            return
    _ensure_user(target_id)
    _set_free_used(target_id, 0)
    await m.answer(f"Счётчик бесплатных сообщений для user_id={target_id} сброшен.")

@router.message(Command("admin"))
async def cmd_admin(m: types.Message):
    """Панель администратора (только для ADMIN_IDS)."""
    if not _is_admin(m.from_user.id):
        await m.answer("Эта команда доступна только администраторам.")
        return
    await m.answer(
        "🛠 Админ-панель\n"
        "Команды:\n"
        "• /reset_trial <user_id> — сбросить счётчик пробных сообщений пользователю.\n"
        "• /status — посмотреть свой статус.\n"
        "• /buy — создать инвойс (демо) и кнопку оплаты для себя."
    )

@router.callback_query(F.data.startswith("pay:confirm:"))
async def cb_confirm_paid(cb: types.CallbackQuery):
    data = cb.data or ""
    invoice_id = data.split("pay:confirm:", 1)[-1].strip()
    user_id_from_invoice = _mark_invoice_succeeded(invoice_id)
    if not user_id_from_invoice:
        await cb.answer("Инвойс не найден", show_alert=True)
        return

    # Активация подписки
    until_ts = _activate_subscription(user_id_from_invoice, SUB_DAYS)

    # "Чек" (демо)
    until_human = _subscription_until_str(until_ts)
    receipt = (
        "🧾 Чек (демо)\n"
        f"• Номер заказа: {invoice_id}\n"
        f"• Сумма: {AMOUNT} {CURRENCY}\n"
        f"• Подписка активна до: {until_human}\n"
    )

    await cb.message.edit_reply_markup(reply_markup=None)
    await cb.message.answer("✅ Оплата подтверждена. Подписка активирована!")
    await cb.message.answer(receipt)
    await cb.answer("Готово", show_alert=False)

# ──────────────────────────────────────────────────────────────────────────────
# Публичная точка подключения
# ──────────────────────────────────────────────────────────────────────────────
def setup_paywall(dp: Dispatcher, bot: Bot) -> None:
    """
    Подключаем заглушку оплаты к вашему бот-ядру.
    Вызывать один раз после создания dp и bot.
    """
    _init_db()

    # Middleware — ставим на обработку всех апдейтов (так безопаснее для разных версий aiogram)
    dp.update.middleware(PaywallMiddleware(bot, PaywallConfig()))

    # Роутер с командами/колбэками
    dp.include_router(router)
