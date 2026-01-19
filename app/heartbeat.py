# app/heartbeat.py
# -*- coding: utf-8 -*-
"""
Heartbeat модуль — обновляет файл каждые N секунд.
Watchdog проверяет этот файл и перезапускает бота если он устарел.
"""

import asyncio
import time
from pathlib import Path
from datetime import datetime
import logging

log = logging.getLogger(__name__)

HEARTBEAT_FILE = "bot_heartbeat.txt"
HEARTBEAT_INTERVAL = 30  # Обновлять каждые 30 секунд


def update_heartbeat():
    """Обновляет файл heartbeat."""
    try:
        Path(HEARTBEAT_FILE).write_text(
            f"{datetime.now().isoformat()}\n{time.time()}"
        )
    except Exception as e:
        log.debug(f"Ошибка записи heartbeat: {e}")


async def start_heartbeat_task(interval: int = HEARTBEAT_INTERVAL):
    """
    Фоновая задача — обновляет heartbeat каждые N секунд.
    Запускается при старте бота.
    """
    log.info(f"Heartbeat запущен (интервал: {interval} сек)")
    
    while True:
        update_heartbeat()
        await asyncio.sleep(interval)