# app/config.py
import os
from pathlib import Path
from enum import Enum

# Загружаем .env, если есть
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# ----------------------------- env helpers -----------------------------

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


# --- FSM состояния обработки документа (для детерминированной оркестрации) ---
class ProcessingState(str, Enum):
    IDLE = "idle"                 # Нет активного документа / ожидание
    DOWNLOADING = "downloading"   # Получаем файл от Telegram
    INDEXING = "indexing"         # Индексируем/парсим документ
    READY = "ready"               # Документ проиндексирован, можно отвечать
    ANSWERING = "answering"       # Генерация ответа (RAG/LLM)


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
    # Сколько соседних блоков проверять при поиске картинок вокруг подписи (DOCX)
    FIG_NEIGHBOR_WINDOW: int = _env_int("FIG_NEIGHBOR_WINDOW", 4)
    # Извлекать ли картинки из PDF (через PyMuPDF), если доступно
    PDF_EXTRACT_IMAGES: bool = _env_bool("PDF_EXTRACT_IMAGES", True)

    # ВКЛЮЧЕНО ДЛЯ ШАГА «Структурированный парсинг + OCR + кеш по хэшу»
    # Сохранять структурированный индекс секций в БД (таблица document_sections)
    SAVE_STRUCT_INDEX: bool = _env_bool("SAVE_STRUCT_INDEX", True)
    # OCR для изображений/сканов (используется индексатором, если подключён pytesseract/иное)
    OCR_ENABLED: bool = _env_bool("OCR_ENABLED", True)
    OCR_LANGS: str = _env_str("OCR_LANGS", "rus+eng")
    # Какой OCR-движок использовать: "tesseract" | "disabled" | др. (на будущее)
    OCR_ENGINE: str = _env_str("OCR_ENGINE", "tesseract")
    # Необязательный путь к бинарю tesseract (если не в PATH)
    OCR_TESSERACT_CMD: str = _env_str("OCR_TESSERACT_CMD", "")

    # --- FULLREAD режим ---
    # По умолчанию "auto": если документ влазит — даём модели целиком,
    # иначе читаем итеративно (map → reduce).
    _FULLREAD_MODE_RAW: str = _env_str("FULLREAD_MODE", "auto").lower()
    _FULLREAD_ALIASES = {
        "iter": "iterative",
        "iterative": "iterative",
        "auto": "auto",
        "direct": "direct",
        "digest": "digest",
        "off": "off",
    }
    FULLREAD_MODE: str = _FULLREAD_ALIASES.get(_FULLREAD_MODE_RAW, "auto")

    # Лимиты/бюджеты для FULLREAD
    FULLREAD_MAX_STEPS: int = _env_int("FULLREAD_MAX_STEPS", 6)
    FULLREAD_STEP_CHARS: int = _env_int("FULLREAD_STEP_CHARS", 20_000)
    FULLREAD_CHUNK_CHARS: int = _env_int("FULLREAD_CHUNK_CHARS", FULLREAD_STEP_CHARS)
    FULLREAD_MAX_SECTIONS: int = _env_int("FULLREAD_MAX_SECTIONS", 120)
    FULLREAD_CONTEXT_CHARS: int = _env_int("FULLREAD_CONTEXT_CHARS", 30_000)
    # Сколько символов позволяем "скормить" напрямую (если влезает):
    DIRECT_MAX_CHARS: int = _env_int("DIRECT_MAX_CHARS", 300_000)

    # Токен-бюджеты для map/reduce
    FULLREAD_MAP_TOKENS: int = _env_int("FULLREAD_MAP_TOKENS", 600)
    FULLREAD_REDUCE_TOKENS: int = _env_int("FULLREAD_REDUCE_TOKENS", 2_400)
    DIGEST_TOKENS_PER_SECTION: int = _env_int("DIGEST_TOKENS_PER_SECTION", 900)

    FULLREAD_ENABLE_VISION: bool = _env_bool("FULLREAD_ENABLE_VISION", True)

    # --- Бюджеты генерации ---
    ANSWER_MAX_TOKENS: int = _env_int("ANSWER_MAX_TOKENS", 2_400)
    EDITOR_MAX_TOKENS: int = _env_int("EDITOR_MAX_TOKENS", 1_800)
    CRITIC_MAX_TOKENS: int = _env_int("CRITIC_MAX_TOKENS", 600)
    EXPLAIN_MAX_TOKENS: int = _env_int("EXPLAIN_MAX_TOKENS", 2_400)
    EXPAND_MAX_TOKENS: int = _env_int("EXPAND_MAX_TOKENS", 2_400)
    PLANNER_MAX_TOKENS: int = _env_int("PLANNER_MAX_TOKENS", 500)
    PART_MAX_TOKENS: int = _env_int("PART_MAX_TOKENS", 900)
    MERGE_MAX_TOKENS: int = _env_int("MERGE_MAX_TOKENS", 2_400)
    # Используется в нескольких местах (stream/non-stream)
    FINAL_MAX_TOKENS: int = _env_int("FINAL_MAX_TOKENS", 2_400)

    # --- Полные выгрузки таблиц (TablesRaw) ---
    FULL_TABLE_MAX_ROWS: int = _env_int("FULL_TABLE_MAX_ROWS", 500)
    FULL_TABLE_MAX_COLS: int = _env_int("FULL_TABLE_MAX_COLS", 40)
    FULL_TABLE_MAX_CHARS: int = _env_int("FULL_TABLE_MAX_CHARS", 20_000)
    SECTION_TABLES_MAX: int = _env_int("SECTION_TABLES_MAX", 3)

    # --- Декомпозиция многопунктных вопросов ---
    MULTI_PLAN_ENABLED: bool = _env_bool("MULTI_PLAN_ENABLED", True)
    MULTI_MIN_ITEMS: int = _env_int("MULTI_MIN_ITEMS", 2)
    MULTI_MAX_ITEMS: int = _env_int("MULTI_MAX_ITEMS", 12)
    MULTI_PASS_SCORE: int = _env_int("MULTI_PASS_SCORE", 85)

    # Настройки подачи подпунктов (используются в bot.py)
    MULTI_STEP_SEND_ENABLED: bool = _env_bool("MULTI_STEP_SEND_ENABLED", True)
    MULTI_STEP_MIN_ITEMS: int = _env_int("MULTI_STEP_MIN_ITEMS", 2)
    MULTI_STEP_MAX_ITEMS: int = _env_int("MULTI_STEP_MAX_ITEMS", 8)
    MULTI_STEP_FINAL_MERGE: bool = _env_bool("MULTI_STEP_FINAL_MERGE", True)
    MULTI_STEP_PAUSE_MS: int = _env_int("MULTI_STEP_PAUSE_MS", 120)

    # --- Стриминг в Telegram ---
    STREAM_ENABLED: bool = _env_bool("STREAM_ENABLED", True)
    STREAM_EDIT_INTERVAL_MS: int = _env_int("STREAM_EDIT_INTERVAL_MS", 1_200)
    STREAM_MIN_CHARS: int = _env_int("STREAM_MIN_CHARS", 700)
    STREAM_MODE: str = _env_str("STREAM_MODE", "multi")  # "edit" | "multi"
    TG_MAX_CHARS: int = _env_int("TG_MAX_CHARS", 3_900)
    STREAM_HEAD_START_MS: int = _env_int("STREAM_HEAD_START_MS", 250)
    TYPE_INDICATION_EVERY_MS: int = _env_int("TYPE_INDICATION_EVERY_MS", 2_000)

    # --- PostgreSQL (резерв) ---
    PG_HOST: str = os.getenv("PG_HOST", "localhost")
    PG_PORT: int = _env_int("PG_PORT", 5432)
    PG_DB: str = os.getenv("PG_DB", "vkr")
    PG_USER: str = os.getenv("PG_USER", "postgres")
    PG_PASSWORD: str = os.getenv("PG_PASSWORD", "postgres")

    # --- Пути / файловая система ---
    SQLITE_PATH: str = os.getenv("SQLITE_PATH", "./vkr.sqlite")
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./uploads")
    # Опциональный кэш (например, для временных OCR-результатов/картинок)
    CACHE_DIR: str = os.getenv("CACHE_DIR", "./.cache")

    # Гарантируем наличие директорий
    _upload_dir = Path(UPLOAD_DIR)
    _upload_dir.mkdir(parents=True, exist_ok=True)

    _sqlite_parent = Path(SQLITE_PATH).resolve().parent
    _sqlite_parent.mkdir(parents=True, exist_ok=True)

    _cache_dir = Path(CACHE_DIR)
    _cache_dir.mkdir(parents=True, exist_ok=True)

    # =========================
    #   FSM / READY-БАРЬЕР
    # =========================
    FSM_ENABLED: bool = _env_bool("FSM_ENABLED", True)

    # Жёсткий контроль на пользователя: запрещает параллельные пайплайны для одного user_id.
    PER_USER_MAX_CONCURRENCY: int = _env_int("PER_USER_MAX_CONCURRENCY", 1)

    # Очередь запросов, пришедших до READY
    PENDING_QUEUE_MAX: int = _env_int("PENDING_QUEUE_MAX", 50)
    PENDING_QUEUE_TTL_SEC: int = _env_int("PENDING_QUEUE_TTL_SEC", 60 * 60 * 12)  # 12 часов

    # Таймауты стадий
    DOWNLOAD_TIMEOUT_SEC: int = _env_int("DOWNLOAD_TIMEOUT_SEC", 120)
    INDEX_TIMEOUT_SEC: int = _env_int("INDEX_TIMEOUT_SEC", 15 * 60)  # 15 минут

    # Барьер: ждать READY при обработке текстовых сообщений, если есть активный ingest
    WAIT_READY_ON_QUERY: bool = _env_bool("WAIT_READY_ON_QUERY", True)
    WAIT_READY_TIMEOUT_SEC: int = _env_int("WAIT_READY_TIMEOUT_SEC", 15 * 60)  # 15 минут
    READY_GRACE_SEC: int = _env_int("READY_GRACE_SEC", 2)  # пауза после READY перед ANSWERING

    # Идемпотентность индексации — по хэшу файла (не запускать повторно)
    INGEST_IDEMPOTENCY_BY_HASH: bool = _env_bool("INGEST_IDEMPOTENCY_BY_HASH", True)

    # Ретраи оркестрации индексации/загрузки
    RETRY_MAX_ATTEMPTS: int = _env_int("FSM_RETRY_MAX_ATTEMPTS", 3)
    RETRY_BASE_DELAY_MS: int = _env_int("FSM_RETRY_BASE_DELAY_MS", 500)

    # Диагностика и логирование событий FSM
    FSM_AUDIT_LOG_ENABLED: bool = _env_bool("FSM_AUDIT_LOG_ENABLED", True)

    # Текстовые шаблоны статусов (переопределяемые через .env)
    MSG_ACK_DOWNLOADING: str = _env_str(
        "MSG_ACK_DOWNLOADING",
        "Файл получен, начинаю загрузку…"
    )
    MSG_ACK_INDEXING: str = _env_str(
        "MSG_ACK_INDEXING",
        "Индексирую документ, это может занять немного времени…"
    )
    # Новое: сообщение, когда ещё нет активного документа — вопрос кладём в очередь
    MSG_NEED_FILE_QUEUED: str = _env_str(
        "MSG_NEED_FILE_QUEUED",
        "Сначала пришлите .doc/.docx. Я поставил ваш вопрос в очередь."
    )
    MSG_NOT_READY_QUEUED: str = _env_str(
        "MSG_NOT_READY_QUEUED",
        "Документ ещё готовится. Ваш запрос поставлен в очередь и будет выполнен сразу после индексации."
    )
    MSG_READY: str = _env_str(
        "MSG_READY",
        "Готово: документ проиндексирован. Перехожу к ответам."
    )
    MSG_INDEX_FAILED: str = _env_str(
        "MSG_INDEX_FAILED",
        "Не удалось проиндексировать документ. Попробуйте ещё раз или загрузите другой файл."
    )

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

        if cls.STREAM_MODE not in {"edit", "multi"}:
            raise RuntimeError("STREAM_MODE должен быть 'edit' или 'multi'.")

        if cls.PER_USER_MAX_CONCURRENCY < 1:
            raise RuntimeError("PER_USER_MAX_CONCURRENCY должен быть >= 1.")

        if cls.PENDING_QUEUE_MAX < 0:
            raise RuntimeError("PENDING_QUEUE_MAX не может быть отрицательным.")

        if cls.DOWNLOAD_TIMEOUT_SEC <= 0 or cls.INDEX_TIMEOUT_SEC <= 0:
            raise RuntimeError("DOWNLOAD_TIMEOUT_SEC и INDEX_TIMEOUT_SEC должны быть > 0.")

        if cls.WAIT_READY_ON_QUERY and cls.WAIT_READY_TIMEOUT_SEC <= 0:
            raise RuntimeError("WAIT_READY_TIMEOUT_SEC должен быть > 0, если WAIT_READY_ON_QUERY включён.")

    @classmethod
    def fullread_enabled(cls) -> bool:
        return (cls.FULLREAD_MODE or "off") != "off"

    @classmethod
    def fullread_mode(cls) -> str:
        return cls.FULLREAD_MODE
