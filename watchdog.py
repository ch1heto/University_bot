#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Watchdog для Telegram бота.

Что делает:
1. Запускает бота как subprocess
2. Следит — если процесс упал, перезапускает
3. Проверяет heartbeat файл — если устарел (бот завис), перезапускает
4. Защита от бесконечных перезапусков (макс 10 в час)

Использование:
    python watchdog.py                    # По умолчанию
    python watchdog.py --timeout 120      # Timeout 2 минуты
    
Расположение: в корне проекта (НЕ в папке app/)

Структура:
    project/
    ├── app/
    │   ├── bot.py
    │   ├── heartbeat.py    ← обновляет bot_heartbeat.txt
    │   └── __main__.py     ← точка входа
    ├── watchdog.py         ← этот файл
    └── bot_heartbeat.txt   ← создаётся автоматически
"""

import subprocess
import sys
import time
import signal
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Логирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [WATCHDOG] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('watchdog.log', encoding='utf-8')
    ]
)
log = logging.getLogger(__name__)


class Watchdog:
    def __init__(
        self,
        command: str = "python -m app",
        timeout: int = 120,
        restart_delay: int = 5,
        max_restarts: int = 10,
        heartbeat_file: str = "bot_heartbeat.txt",
    ):
        self.command = command
        self.timeout = timeout
        self.restart_delay = restart_delay
        self.max_restarts = max_restarts
        self.heartbeat_file = Path(heartbeat_file)
        
        self.process = None
        self.running = True
        self.restart_times = []
        
        # Graceful shutdown
        signal.signal(signal.SIGINT, self._stop)
        signal.signal(signal.SIGTERM, self._stop)
    
    def _stop(self, *args):
        """Остановка watchdog."""
        log.info("Останавливаем watchdog...")
        self.running = False
        self._kill_bot()
    
    def _kill_bot(self):
        """Убивает процесс бота."""
        if self.process:
            log.info(f"Останавливаем бота (PID: {self.process.pid})...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process = None
            log.info("Бот остановлен")
    
    def _start_bot(self) -> bool:
        """Запускает бота."""
        log.info(f"Запускаем бота: {self.command}")
        try:
            self.process = subprocess.Popen(
                self.command.split(),
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
            log.info(f"Бот запущен, PID: {self.process.pid}")
            return True
        except Exception as e:
            log.error(f"Ошибка запуска: {e}")
            return False
    
    def _is_alive(self) -> bool:
        """Проверяет, жив ли процесс."""
        if not self.process:
            return False
        return self.process.poll() is None
    
    def _check_heartbeat(self) -> bool:
        """Проверяет heartbeat файл."""
        if not self.heartbeat_file.exists():
            # Файла нет — даём боту время создать его (первые 60 сек)
            return True
        
        try:
            age = time.time() - self.heartbeat_file.stat().st_mtime
            if age > self.timeout:
                log.warning(f"Heartbeat устарел: {age:.0f} сек > {self.timeout} сек")
                return False
            return True
        except Exception as e:
            log.error(f"Ошибка чтения heartbeat: {e}")
            return True
    
    def _check_restart_limit(self) -> bool:
        """Проверяет лимит перезапусков."""
        now = time.time()
        # Оставляем только перезапуски за последний час
        self.restart_times = [t for t in self.restart_times if now - t < 3600]
        
        if len(self.restart_times) >= self.max_restarts:
            log.error(f"Лимит перезапусков ({self.max_restarts}/час). Пауза 30 мин...")
            time.sleep(1800)
            self.restart_times.clear()
        
        return True
    
    def _restart(self, reason: str):
        """Перезапускает бота."""
        log.warning(f"Перезапуск бота. Причина: {reason}")
        
        self._check_restart_limit()
        self.restart_times.append(time.time())
        
        self._kill_bot()
        
        # Удаляем старый heartbeat файл
        try:
            self.heartbeat_file.unlink(missing_ok=True)
        except:
            pass
        
        log.info(f"Ожидание {self.restart_delay} сек...")
        time.sleep(self.restart_delay)
        
        self._start_bot()
    
    def run(self):
        """Главный цикл."""
        log.info("=" * 50)
        log.info("WATCHDOG ЗАПУЩЕН")
        log.info(f"  Команда: {self.command}")
        log.info(f"  Timeout heartbeat: {self.timeout} сек")
        log.info(f"  Heartbeat файл: {self.heartbeat_file}")
        log.info("=" * 50)
        
        if not self._start_bot():
            log.error("Не удалось запустить бота!")
            return
        
        # Даём боту время на запуск перед проверкой heartbeat
        startup_grace_period = 60  # 60 секунд на старт
        startup_time = time.time()
        
        while self.running:
            time.sleep(5)  # Проверяем каждые 5 секунд
            
            # 1. Проверяем процесс
            if not self._is_alive():
                exit_code = self.process.returncode if self.process else "?"
                self._restart(f"процесс упал (exit code: {exit_code})")
                startup_time = time.time()
                continue
            
            # 2. Проверяем heartbeat (только после grace period)
            if time.time() - startup_time > startup_grace_period:
                if not self._check_heartbeat():
                    self._restart("heartbeat timeout (бот завис)")
                    startup_time = time.time()
                    continue
        
        self._kill_bot()
        log.info("Watchdog завершён")


def main():
    parser = argparse.ArgumentParser(
        description="Watchdog для Telegram бота",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python watchdog.py                          # Запуск по умолчанию
  python watchdog.py --timeout 60             # Timeout 1 минута  
  python watchdog.py -c "python run.py"       # Другая команда
        """
    )
    parser.add_argument(
        "-c", "--command",
        default="python -m app",
        help="Команда запуска бота (default: python -m app)"
    )
    parser.add_argument(
        "-t", "--timeout",
        type=int,
        default=120,
        help="Timeout heartbeat в секундах (default: 120)"
    )
    parser.add_argument(
        "-d", "--delay",
        type=int,
        default=5,
        help="Задержка перед перезапуском (default: 5)"
    )
    parser.add_argument(
        "-f", "--heartbeat-file",
        default="bot_heartbeat.txt",
        help="Файл heartbeat (default: bot_heartbeat.txt)"
    )
    
    args = parser.parse_args()
    
    watchdog = Watchdog(
        command=args.command,
        timeout=args.timeout,
        restart_delay=args.delay,
        heartbeat_file=args.heartbeat_file,
    )
    watchdog.run()


if __name__ == "__main__":
    main()