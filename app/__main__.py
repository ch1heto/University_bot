# app/__main__.py
# -*- coding: utf-8 -*-
"""
Точка входа для запуска бота: python -m app
"""

import asyncio
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('bot.log', encoding='utf-8')
    ]
)
log = logging.getLogger(__name__)


async def main():
    # Импортируем здесь чтобы логирование уже было настроено
    from .bot import dp, bot
    
    log.info("Бот запускается…")
    
    # Запускаем heartbeat если модуль есть
    try:
        from .heartbeat import start_heartbeat_task
        asyncio.create_task(start_heartbeat_task())
        log.info("Heartbeat task запущен")
    except ImportError:
        log.warning("Модуль heartbeat не найден — работаем без него")
    
    log.info("Start polling")
    
    try:
        await dp.start_polling(bot)
    except Exception as e:
        log.exception(f"Ошибка polling: {e}")
        raise


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Бот остановлен по Ctrl+C")
    except Exception as e:
        log.exception(f"Критическая ошибка: {e}")
        sys.exit(1)