# app/__main__.py
import asyncio
import logging
from aiogram import Bot, Dispatcher

# Импортируем из bot.py нужные объекты
from .bot import bot, dp

logger = logging.getLogger(__name__)


async def main():
    logger.info("Запуск бота: начинаю polling…")
    # backoff_config: увеличенный таймаут между повторными попытками при обрыве
    # allowed_updates: получаем все типы обновлений
    await dp.start_polling(
        bot,
        # Увеличенные таймауты для нестабильных сетей (актуально для РФ-серверов)
        polling_timeout=30,
        # Закрываем предыдущие сессии при старте
        close_bot_session=True,
    )

if __name__ == "__main__":
    asyncio.run(main())