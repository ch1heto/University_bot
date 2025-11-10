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

def _env_float(name: str, default: float) -> float:
    try:
        return float(str(os.getenv(name, str(default))).replace(",", ".").strip())
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

    # Фоллбэки имён переменных окружения для совместимости:
    #   POLZA_CHAT_MODEL | POLZA_MODEL | MODEL_CHAT
    _CHAT_CANDIDATES = [
        os.getenv("POLZA_CHAT_MODEL"),
        os.getenv("POLZA_MODEL"),
        os.getenv("MODEL_CHAT"),
    ]
    POLZA_CHAT: str = next((c for c in _CHAT_CANDIDATES if c and c.strip()), "openai/gpt-5")

    #   POLZA_VISION_MODEL | MODEL_VISION | <fallback: POLZA_CHAT>
    _VISION_CANDIDATES = [
        os.getenv("POLZA_VISION_MODEL"),
        os.getenv("MODEL_VISION"),
        POLZA_CHAT,
    ]
    POLZA_VISION_MODEL: str = next((c for c in _VISION_CANDIDATES if c and c.strip()), POLZA_CHAT)

    #   POLZA_EMB_MODEL | EMB_MODEL
    _EMB_CANDIDATES = [
        os.getenv("POLZA_EMB_MODEL"),
        os.getenv("EMB_MODEL"),
    ]
    POLZA_EMB: str = next((c for c in _EMB_CANDIDATES if c and c.strip()), "openai/text-embedding-3-large")

    # =========================
    #   VISION / РИСУНКИ
    # =========================
    # Включает подачу изображений напрямую в чат-модель (b64/url/file).
    VISION_ENABLED: bool = _env_bool("VISION_ENABLED", True)
    VISION_LANG: str = _env_str("VISION_LANG", "ru")  # язык ответов vision-пайплайна

    # Как передавать изображение в модель: "base64" | "url" | "file"
    VISION_IMAGE_TRANSPORT: str = _env_str("VISION_IMAGE_TRANSPORT", "base64").lower()

    # Ограничения и препроцесс
    VISION_MAX_IMAGES_PER_REQUEST: int = _env_int("VISION_MAX_IMAGES_PER_REQUEST", 4)
    VISION_MAX_IMAGE_BYTES: int = _env_int("VISION_MAX_IMAGE_BYTES", 8_000_000)  # 8 MB
    VISION_MAX_SIDE_PX: int = _env_int("VISION_MAX_SIDE_PX", 2048)              # даунскейл длинной стороны
    VISION_JPEG_QUALITY: int = _env_int("VISION_JPEG_QUALITY", 88)

    # Директория для кеша ответов vision
    VISION_CACHE_DIR: str = _env_str("VISION_CACHE_DIR", ".cache/vision")
    VISION_CACHE_TTL_SEC: int = _env_int("VISION_CACHE_TTL_SEC", 7 * 24 * 60 * 60)  # 7 дней

    # Разрешённые расширения изображений (через запятую)
    _VISION_ACCEPT_EXT_RAW = _env_str(
        "VISION_ACCEPT_EXT",
        "png,jpg,jpeg,gif,webp,bmp,tiff,tif"
    )
    VISION_ACCEPT_EXT: set[str] = {e.strip().lower().lstrip(".") for e in _VISION_ACCEPT_EXT_RAW.split(",") if e.strip()}

    # MIME по расширениям — может пригодиться клиенту
    VISION_MIME_BY_EXT: dict[str, str] = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
        "bmp": "image/bmp",
        "tif": "image/tiff",
        "tiff": "image/tiff",
    }

    # Таймаут запросов c изображениями больше, чем у обычного чата
    VISION_TIMEOUT_SEC: int = _env_int("VISION_TIMEOUT_SEC", 120)

    # Хинты для OCR/языков на картинках (передаются в system/meta при необходимости)
    VISION_OCR_HINT_LANGS: str = _env_str("VISION_OCR_HINT_LANGS", "rus+eng")

    # Если true — даже при наличии текстового контекста картинки отправляются ВСЕГДА,
    # когда вопрос касается рисунков/диаграмм/изображений.
    VISION_ENFORCE_INLINE_IMAGES: bool = _env_bool("VISION_ENFORCE_INLINE_IMAGES", True)

    # ---------- Управление двумя шагами анализа ----------
    # Шаг 1: осмысленное описание (Describe)
    VISION_DESCRIBE_ENABLED: bool = _env_bool("VISION_DESCRIBE_ENABLED", True)
    VISION_DESCRIBE_SENTENCES_MIN: int = _env_int("VISION_DESCRIBE_SENTENCES_MIN", 2)
    VISION_DESCRIBE_SENTENCES_MAX: int = _env_int("VISION_DESCRIBE_SENTENCES_MAX", 4)
    VISION_JSON_STRICT: bool = _env_bool("VISION_JSON_STRICT", True)  # требовать от vision JSON без "воды"

    # Шаг 2: извлечение значений (Extract)
    VISION_EXTRACT_VALUES_ENABLED: bool = _env_bool("VISION_EXTRACT_VALUES_ENABLED", True)
    VISION_EXTRACT_CONF_MIN: float = _env_float("VISION_EXTRACT_CONF_MIN", 0.70)  # нижний порог уверенности
    VISION_EXTRACT_MAX_ITEMS: int = _env_int("VISION_EXTRACT_MAX_ITEMS", 30)

    # Пост-валидации и форматирование чисел
    VISION_PIE_SUM_TARGET: float = _env_float("VISION_PIE_SUM_TARGET", 100.0)
    VISION_PIE_SUM_TOLERANCE_PP: float = _env_float("VISION_PIE_SUM_TOLERANCE_PP", 5.0)  # допуск в п.п.
    VISION_PERCENT_DECIMALS: int = _env_int("VISION_PERCENT_DECIMALS", 1)
    VISION_NUMBER_GROUPING: bool = _env_bool("VISION_NUMBER_GROUPING", True)  # 12 300 вместо 12300

    # Включать ли оговорку о погрешности, если числа с OCR/vision (а не из chart XML)
    VISION_APPEND_CAVEAT_FOR_OCR: bool = _env_bool("VISION_APPEND_CAVEAT_FOR_OCR", True)

    # --- Параметры модуля рисунков (figures.py) / PDF-растрирование векторов ---
    FIG_CACHE_DIR: str = _env_str("FIG_CACHE_DIR", ".cache/figures")
    FIG_MEDIA_LIMIT: int = _env_int("FIG_MEDIA_LIMIT", 12)
    FIG_VALUES_DEFAULT: bool = _env_bool("FIG_VALUES_DEFAULT", True)
    FIG_NEIGHBOR_WINDOW: int = _env_int("FIG_NEIGHBOR_WINDOW", 10)

    # Управление извлечением для DOCX/PDF (зеркалим переменные figures.py)
    FIG_DOCX_EXTRACT_IMAGES: bool = _env_bool("DOCX_EXTRACT_IMAGES", True)
    FIG_PDF_EXTRACT_IMAGES: bool = _env_bool("PDF_EXTRACT_IMAGES", True)
    FIG_DOCX_PDF_FALLBACK: bool = _env_bool("DOCX_PDF_FALLBACK", True)
    FIG_DOCX_PDF_BACKEND: str = _env_str("DOCX_TO_PDF_BACKEND", "auto").lower()  # auto|docx2pdf|libreoffice|off
    FIG_DOCX_PDF_ALWAYS: bool = _env_bool("DOCX_PDF_ALWAYS", False)

    FIG_PDF_CAPTION_MAX_DISTANCE_PX: int = _env_int("PDF_CAPTION_MAX_DISTANCE_PX", 300)
    FIG_PDF_MIN_IMAGE_AREA: int = _env_int("PDF_MIN_IMAGE_AREA", 20000)

    # Рендер векторов (PyMuPDF get_drawings → clip → PNG)
    FIG_PDF_VECTOR_RASTERIZE: bool = _env_bool("PDF_VECTOR_RASTERIZE", True)
    FIG_PDF_VECTOR_DPI: int = _env_int("PDF_VECTOR_DPI", 360)
    FIG_PDF_VECTOR_MIN_WIDTH_PX: int = _env_int("PDF_VECTOR_MIN_WIDTH_PX", 1200)
    FIG_PDF_VECTOR_PAD_PX: int = _env_int("PDF_VECTOR_PAD_PX", 16)
    FIG_PDF_VECTOR_MAX_DPI: int = _env_int("PDF_VECTOR_MAX_DPI", 600)

    FIG_CLEANUP_ON_NEW_DOC: bool = _env_bool("FIG_CLEANUP_ON_NEW_DOC", True)

    # ВКЛЮЧЕНО ДЛЯ ШАГА «Структурированный парсинг + OCR + кеш по хэшу»
    SAVE_STRUCT_INDEX: bool = _env_bool("SAVE_STRUCT_INDEX", True)
    OCR_ENABLED: bool = _env_bool("OCR_ENABLED", True)
    OCR_LANGS: str = _env_str("OCR_LANGS", "rus+eng")
    OCR_ENGINE: str = _env_str("OCR_ENGINE", "tesseract")
    OCR_TESSERACT_CMD: str = _env_str("OCR_TESSERACT_CMD", "")

    # --- FULLREAD режим ---
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
    DIRECT_MAX_CHARS: int = _env_int("DIRECT_MAX_CHARS", 300_000)

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
    CACHE_DIR: str = os.getenv("CACHE_DIR", "./.cache")

    # Гарантируем наличие директорий
    _upload_dir = Path(UPLOAD_DIR)
    _upload_dir.mkdir(parents=True, exist_ok=True)

    _sqlite_parent = Path(SQLITE_PATH).resolve().parent
    _sqlite_parent.mkdir(parents=True, exist_ok=True)

    _cache_dir = Path(CACHE_DIR)
    _cache_dir.mkdir(parents=True, exist_ok=True)

    # Кеш для vision (по умолчанию ./.cache/vision)
    _vision_cache_dir = Path(VISION_CACHE_DIR)
    _vision_cache_dir.mkdir(parents=True, exist_ok=True)

    # Кеш для figures (изображения + figures_index.json)
    _fig_cache_dir = Path(FIG_CACHE_DIR)
    _fig_cache_dir.mkdir(parents=True, exist_ok=True)

    # =========================
    #   FSM / READY-БАРЬЕР
    # =========================
    FSM_ENABLED: bool = _env_bool("FSM_ENABLED", True)
    PER_USER_MAX_CONCURRENCY: int = _env_int("PER_USER_MAX_CONCURRENCY", 1)
    PENDING_QUEUE_MAX: int = _env_int("PENDING_QUEUE_MAX", 50)
    PENDING_QUEUE_TTL_SEC: int = _env_int("PENDING_QUEUE_TTL_SEC", 60 * 60 * 12)

    DOWNLOAD_TIMEOUT_SEC: int = _env_int("DOWNLOAD_TIMEOUT_SEC", 120)
    INDEX_TIMEOUT_SEC: int = _env_int("INDEX_TIMEOUT_SEC", 15 * 60)

    WAIT_READY_ON_QUERY: bool = _env_bool("WAIT_READY_ON_QUERY", True)
    WAIT_READY_TIMEOUT_SEC: int = _env_int("WAIT_READY_TIMEOUT_SEC", 15 * 60)
    READY_GRACE_SEC: int = _env_int("READY_GRACE_SEC", 2)

    INGEST_IDEMPOTENCY_BY_HASH: bool = _env_bool("INGEST_IDEMPOTENCY_BY_HASH", True)
    RETRY_MAX_ATTEMPTS: int = _env_int("FSM_RETRY_MAX_ATTEMPTS", 3)
    RETRY_BASE_DELAY_MS: int = _env_int("FSM_RETRY_BASE_DELAY_MS", 500)
    FSM_AUDIT_LOG_ENABLED: bool = _env_bool("FSM_AUDIT_LOG_ENABLED", True)

    MSG_ACK_DOWNLOADING: str = _env_str(
        "MSG_ACK_DOWNLOADING",
        "Файл получен, начинаю загрузку…"
    )
    MSG_ACK_INDEXING: str = _env_str(
        "MSG_ACK_INDEXING",
        "Индексирую документ, это может занять немного времени…"
    )
    MSG_GREET: str = _env_str(
        "MSG_GREET",
        "Привет! Я репетитор по твоей ВКР. Пришлите файл ВКР — и я помогу по содержанию."
    )
    MSG_NEED_FILE_QUEUED: str = _env_str(
        "MSG_NEED_FILE_QUEUED",
        "Сначала пришлите файл ВКР. Я поставил ваш вопрос в очередь."
    )
    # Исправлено имя поля (была ошибка в букве), оставим обратную совместимость ниже.
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

    # ----- Аналитический слой (vision_analyzer) -----
    ANALYZE_WITH_NUMBERS: bool = _env_bool("ANALYZE_WITH_NUMBERS", True)  # включать числа в связный ответ
    ANALYZER_ENABLE_CACHE: bool = _env_bool("ANALYZER_ENABLE_CACHE", True)
    ANALYZER_CACHE_TTL_SEC: int = _env_int("ANALYZER_CACHE_TTL_SEC", 7 * 24 * 60 * 60)

    # ----------------------------- Helpers -----------------------------

    @classmethod
    def vision_transport(cls) -> str:
        """Нормализованный способ доставки изображений в модель: 'base64' | 'url' | 'file'."""
        return (cls.VISION_IMAGE_TRANSPORT or "base64").strip().lower()

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

        # Vision sanity checks (не фейлим, если выключено)
        if cls.VISION_ENABLED:
            if cls.VISION_IMAGE_TRANSPORT not in {"base64", "url", "file"}:
                raise RuntimeError("VISION_IMAGE_TRANSPORT должен быть 'base64', 'url' или 'file'.")
            if cls.VISION_MAX_IMAGES_PER_REQUEST < 1:
                raise RuntimeError("VISION_MAX_IMAGES_PER_REQUEST должен быть >= 1.")
            if cls.VISION_MAX_IMAGE_BYTES <= 0:
                raise RuntimeError("VISION_MAX_IMAGE_BYTES должен быть > 0.")
            if cls.VISION_MAX_SIDE_PX <= 0:
                raise RuntimeError("VISION_MAX_SIDE_PX должен быть > 0.")
            if not cls.VISION_ACCEPT_EXT:
                raise RuntimeError("VISION_ACCEPT_EXT не должен быть пустым.")
            if not (0.0 <= cls.VISION_EXTRACT_CONF_MIN <= 1.0):
                raise RuntimeError("VISION_EXTRACT_CONF_MIN должен быть в диапазоне [0.0, 1.0].")

        # Figures sanity
        if cls.FIG_MEDIA_LIMIT < 1:
            raise RuntimeError("FIG_MEDIA_LIMIT должен быть >= 1.")
        if not cls.FIG_CACHE_DIR:
            raise RuntimeError("FIG_CACHE_DIR не должен быть пустым.")
        if not Path(cls.FIG_CACHE_DIR).exists():
            raise RuntimeError(f"Директория FIG_CACHE_DIR='{cls.FIG_CACHE_DIR}' не существует (должна была создаться).")

        # Backward-compat alias (если где-то обращались к старому опечатанному имени)
        # MSG_NOT_READY_QUEУED — с русской буквой 'У'
        setattr(cls, "MSG_NOT_READY_QUEУED", cls.MSG_NOT_READY_QUEUED)

    @classmethod
    def fullread_enabled(cls) -> bool:
        return (cls.FULLREAD_MODE or "off") != "off"

    @classmethod
    def fullread_mode(cls) -> str:
        return cls.FULLREAD_MODE

    @classmethod
    def vision_active(cls) -> bool:
        """Глобальный флаг готовности подачи изображений в модель."""
        return bool(cls.VISION_ENABLED and cls.POLZA_VISION_MODEL)

    @classmethod
    def vision_model(cls) -> str:
        """Выбор модели для мультимодального чата (с запасным вариантом)."""
        return cls.POLZA_VISION_MODEL or cls.POLZA_CHAT

    @classmethod
    def is_image_ext_allowed(cls, ext: str | None) -> bool:
        """Проверка расширения файла изображения (без точки)."""
        if not ext:
            return False
        return ext.lower().lstrip(".") in cls.VISION_ACCEPT_EXT
