# app/config.py
import os
from pathlib import Path

# Подтягиваем .env (если библиотека есть — ок; если нет — тихо проигнорируем)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


class Cfg:
    # --- Telegram ---
    TG_TOKEN: str | None = os.getenv("TG_BOT_TOKEN")

    # --- Polza / OpenAI ---
    # Базовый URL Polza API (env имеет приоритет)
    BASE_POLZA: str = os.getenv("POLZA_BASE_URL", "https://api.polza.ai/api/v1")
    # Ключ доступа (ВАЖНО: именно POLZA_API_KEY)
    POLZA_KEY: str | None = os.getenv("POLZA_API_KEY")
    # Имена моделей (безопасные дефолты)
    POLZA_CHAT: str = os.getenv("POLZA_CHAT_MODEL", "openai/gpt-4o")
    POLZA_EMB: str = os.getenv("POLZA_EMB_MODEL", "openai/text-embedding-3-large")

    # --- PostgreSQL (на будущее) ---
    PG_HOST: str = os.getenv("PG_HOST", "localhost")
    PG_PORT: int = int(os.getenv("PG_PORT", "5432"))
    PG_DB: str = os.getenv("PG_DB", "vkr")
    PG_USER: str = os.getenv("PG_USER", "postgres")
    PG_PASSWORD: str = os.getenv("PG_PASSWORD", "postgres")

    # --- SQLite / файлы ---
    SQLITE_PATH: str = os.getenv("SQLITE_PATH", "./vkr.sqlite")
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./uploads")

    # Гарантируем наличие директорий при импортировании конфига
    _upload_dir = Path(UPLOAD_DIR)
    _upload_dir.mkdir(parents=True, exist_ok=True)

    _sqlite_parent = Path(SQLITE_PATH).resolve().parent
    _sqlite_parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def validate(cls) -> None:
        """
        Вызывай в точке входа перед запуском бота.
        Бросит понятную ошибку, если критичные переменные не заданы.
        """
        missing = []
        if not cls.POLZA_KEY:
            missing.append("POLZA_API_KEY")
        if not cls.TG_TOKEN:
            missing.append("TG_BOT_TOKEN")
        if missing:
            raise RuntimeError(
                "Не найдены переменные окружения: "
                + ", ".join(missing)
                + ". Проверьте .env и окружение."
            )
