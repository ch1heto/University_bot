# app/config.py
import os
from pathlib import Path

# Подтягиваем .env (если библиотека есть — ок; если нет — тихо проигнорируем)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name, str(default))).strip())
    except Exception:
        return default

def _env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return default if v is None else str(v).strip()


class Cfg:
    # --- Telegram ---
    TG_TOKEN: str | None = os.getenv("TG_BOT_TOKEN")

    # --- Polza / OpenAI ---
    # Базовый URL Polza API (env имеет приоритет); убираем хвостовые слэши для совместимости клиента
    _BASE_POLZA_RAW = os.getenv("POLZA_BASE_URL", "https://api.polza.ai/api/v1")
    BASE_POLZA: str = _BASE_POLZA_RAW.rstrip("/")
    # Гарантируем /v1 на конце (если в env указали базу без суффикса)
    if not BASE_POLZA.endswith("/v1"):
        BASE_POLZA = BASE_POLZA + "/v1"

    # Ключ доступа (ВАЖНО: именно POLZA_API_KEY)
    POLZA_KEY: str | None = os.getenv("POLZA_API_KEY")

    # Имена моделей (env → кодовые дефолты)
    # Чат/vision по умолчанию — «мини», можно сменить через POLZA_CHAT_MODEL
    POLZA_CHAT: str = os.getenv("POLZA_CHAT_MODEL", "openai/gpt-4o-mini")
    POLZA_EMB: str = os.getenv("POLZA_EMB_MODEL", "openai/text-embedding-3-large")

    # --- Опции распознавания изображений/парсинга (используются парсером, при желании) ---
    # Ширина «окна» в абзацах для поиска картинки вокруг подписи «Рисунок …»
    FIG_NEIGHBOR_WINDOW: int = _env_int("FIG_NEIGHBOR_WINDOW", 4)
    # Попытка извлекать изображения из PDF (если установлен PyMuPDF)
    PDF_EXTRACT_IMAGES: bool = _env_bool("PDF_EXTRACT_IMAGES", True)

    # --- Режим «модель просматривает файл» (FULLREAD) ---
    # Варианты: off | iterative | direct | digest
    FULLREAD_MODE: str = _env_str("FULLREAD_MODE", "off").lower()
    # Для iterative-ридера: максимум шагов «прочитать ещё»
    FULLREAD_MAX_STEPS: int = _env_int("FULLREAD_MAX_STEPS", 2)
    # Сколько символов подгружать за один шаг (сумма по выбранным секциям/страницам)
    FULLREAD_STEP_CHARS: int = _env_int("FULLREAD_STEP_CHARS", 14000)
    # Итоговый лимит контекста для финального ответа
    FULLREAD_CONTEXT_CHARS: int = _env_int("FULLREAD_CONTEXT_CHARS", 9000)
    # Direct-режим: допускаем «полный просмотр», если весь текст ≤ этого порога
    DIRECT_MAX_CHARS: int = _env_int("DIRECT_MAX_CHARS", 180_000)
    # Digest-режим: целевой размер дайджеста (токены условно; используем как «желаемую длину»)
    DIGEST_TOKENS_PER_SECTION: int = _env_int("DIGEST_TOKENS_PER_SECTION", 300)
    # Использовать vision при чтении рисунков в fullread (если есть изображения)
    FULLREAD_ENABLE_VISION: bool = _env_bool("FULLREAD_ENABLE_VISION", True)

    # --- PostgreSQL (на будущее) ---
    PG_HOST: str = os.getenv("PG_HOST", "localhost")
    PG_PORT: int = _env_int("PG_PORT", 5432)
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
        Также валидирует FULLREAD_MODE.
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

        allowed_modes = {"off", "iterative", "direct", "digest"}
        if (cls.FULLREAD_MODE or "off") not in allowed_modes:
            raise RuntimeError(
                f"FULLREAD_MODE='{cls.FULLREAD_MODE}' не поддерживается. "
                f"Допустимые значения: {', '.join(sorted(allowed_modes))}."
            )

    @classmethod
    def fullread_enabled(cls) -> bool:
        return (cls.FULLREAD_MODE or "off") != "off"
