# app/config.py
import os
from pathlib import Path

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
    _BASE_POLZA_RAW = os.getenv("POLZA_BASE_URL", "https://api.polza.ai/api/v1")
    BASE_POLZA: str = _BASE_POLZA_RAW.rstrip("/")
    if not BASE_POLZA.endswith("/v1"):
        BASE_POLZA = BASE_POLZA + "/v1"

    POLZA_KEY: str | None = os.getenv("POLZA_API_KEY")
    POLZA_CHAT: str = os.getenv("POLZA_CHAT_MODEL", "openai/gpt-4o-mini")
    POLZA_EMB: str = os.getenv("POLZA_EMB_MODEL", "openai/text-embedding-3-large")

    # --- Парсинг / распознавание ---
    FIG_NEIGHBOR_WINDOW: int = _env_int("FIG_NEIGHBOR_WINDOW", 4)
    PDF_EXTRACT_IMAGES: bool = _env_bool("PDF_EXTRACT_IMAGES", True)

    # --- FULLREAD режим (по умолчанию — iterativе) ---
    _FULLREAD_MODE_RAW: str = _env_str("FULLREAD_MODE", "iterative").lower()
    _FULLREAD_ALIASES = {
        "iter": "iterative",
        "iterative": "iterative",
        "auto": "auto",
        "direct": "direct",
        "digest": "digest",
        "off": "off",
    }
    FULLREAD_MODE: str = _FULLREAD_ALIASES.get(_FULLREAD_MODE_RAW, "iterative")

    FULLREAD_MAX_STEPS: int = _env_int("FULLREAD_MAX_STEPS", 4)
    FULLREAD_STEP_CHARS: int = _env_int("FULLREAD_STEP_CHARS", 18_000)
    FULLREAD_CHUNK_CHARS: int = _env_int("FULLREAD_CHUNK_CHARS", FULLREAD_STEP_CHARS)
    FULLREAD_MAX_SECTIONS: int = _env_int("FULLREAD_MAX_SECTIONS", 120)
    FULLREAD_CONTEXT_CHARS: int = _env_int("FULLREAD_CONTEXT_CHARS", 20_000)
    DIRECT_MAX_CHARS: int = _env_int("DIRECT_MAX_CHARS", 180_000)

    # Токен-бюджеты для map/reduce
    FULLREAD_MAP_TOKENS: int = _env_int("FULLREAD_MAP_TOKENS", 600)
    FULLREAD_REDUCE_TOKENS: int = _env_int("FULLREAD_REDUCE_TOKENS", 2400)
    DIGEST_TOKENS_PER_SECTION: int = _env_int("DIGEST_TOKENS_PER_SECTION", FULLREAD_MAP_TOKENS)

    FULLREAD_ENABLE_VISION: bool = _env_bool("FULLREAD_ENABLE_VISION", True)

    # --- Бюджеты генерации (увеличены) ---
    ANSWER_MAX_TOKENS: int = _env_int("ANSWER_MAX_TOKENS", 2400)
    EDITOR_MAX_TOKENS: int = _env_int("EDITOR_MAX_TOKENS", 1800)
    CRITIC_MAX_TOKENS: int = _env_int("CRITIC_MAX_TOKENS", 600)
    EXPLAIN_MAX_TOKENS: int = _env_int("EXPLAIN_MAX_TOKENS", 2400)
    EXPAND_MAX_TOKENS: int = _env_int("EXPAND_MAX_TOKENS", 2400)
    PLANNER_MAX_TOKENS: int = _env_int("PLANNER_MAX_TOKENS", 500)
    PART_MAX_TOKENS: int = _env_int("PART_MAX_TOKENS", 900)
    MERGE_MAX_TOKENS: int = _env_int("MERGE_MAX_TOKENS", 2400)
    FINAL_MAX_TOKENS: int = _env_int("FINAL_MAX_TOKENS", 2400)

    # --- Полные выгрузки таблиц (для TablesRaw) ---
    FULL_TABLE_MAX_ROWS: int = _env_int("FULL_TABLE_MAX_ROWS", 2000)
    FULL_TABLE_MAX_COLS: int = _env_int("FULL_TABLE_MAX_COLS", 60)
    FULL_TABLE_MAX_CHARS: int = _env_int("FULL_TABLE_MAX_CHARS", 60_000)

    # --- Декомпозиция многопунктных вопросов ---
    MULTI_PLAN_ENABLED: bool = _env_bool("MULTI_PLAN_ENABLED", True)
    MULTI_MIN_ITEMS: int = _env_int("MULTI_MIN_ITEMS", 2)
    MULTI_MAX_ITEMS: int = _env_int("MULTI_MAX_ITEMS", 12)
    MULTI_PASS_SCORE: int = _env_int("MULTI_PASS_SCORE", 85)

    # --- Стриминг в Telegram (много сообщений, а не правки) ---
    STREAM_ENABLED: bool = _env_bool("STREAM_ENABLED", True)
    STREAM_EDIT_INTERVAL_MS: int = _env_int("STREAM_EDIT_INTERVAL_MS", 1200)
    STREAM_MIN_CHARS: int = _env_int("STREAM_MIN_CHARS", 700)
    STREAM_MODE: str = _env_str("STREAM_MODE", "multi")  # "edit" | "multi"
    TG_MAX_CHARS: int = _env_int("TG_MAX_CHARS", 3900)
    STREAM_HEAD_START_MS: int = _env_int("STREAM_HEAD_START_MS", 250)
    TYPE_INDICATION_EVERY_MS: int = _env_int("TYPE_INDICATION_EVERY_MS", 2000)

    # --- PostgreSQL (резерв) ---
    PG_HOST: str = os.getenv("PG_HOST", "localhost")
    PG_PORT: int = _env_int("PG_PORT", 5432)
    PG_DB: str = os.getenv("PG_DB", "vkr")
    PG_USER: str = os.getenv("PG_USER", "postgres")
    PG_PASSWORD: str = os.getenv("PG_PASSWORD", "postgres")

    # --- SQLite / файлы ---
    SQLITE_PATH: str = os.getenv("SQLITE_PATH", "./vkr.sqlite")
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./uploads")

    _upload_dir = Path(UPLOAD_DIR)
    _upload_dir.mkdir(parents=True, exist_ok=True)

    _sqlite_parent = Path(SQLITE_PATH).resolve().parent
    _sqlite_parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def validate(cls) -> None:
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

        allowed = {"off", "iterative", "auto", "direct", "digest"}
        if (cls.FULLREAD_MODE or "off") not in allowed:
            raise RuntimeError(
                f"FULLREAD_MODE='{cls.FULLREAD_MODE}' не поддерживается. "
                f"Допустимые значения: {', '.join(sorted(allowed))}."
            )

    @classmethod
    def fullread_enabled(cls) -> bool:
        return (cls.FULLREAD_MODE or "off") != "off"

    @classmethod
    def fullread_mode(cls) -> str:
        return cls.FULLREAD_MODE
