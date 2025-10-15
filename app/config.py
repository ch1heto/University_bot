import os
try:
    from dotenv import load_dotenv  # optional, но удобно
    load_dotenv()
except Exception:
    pass

class Cfg:
    TG_TOKEN     = os.getenv("TG_BOT_TOKEN")
    POLZA_KEY    = os.getenv("POLZA_API_KEY")
    POLZA_CHAT   = os.getenv("POLZA_CHAT_MODEL")
    POLZA_EMB    = os.getenv("POLZA_EMB_MODEL")
    BASE_POLZA   = "https://api.polza.ai/api/v1"

    # PG остаётся на будущее — не используется в SQLite-режиме
    PG_HOST      = os.getenv("PG_HOST", "localhost")
    PG_PORT      = int(os.getenv("PG_PORT", "5432"))
    PG_DB        = os.getenv("PG_DB", "vkr")
    PG_USER      = os.getenv("PG_USER", "postgres")
    PG_PASSWORD  = os.getenv("PG_PASSWORD", "postgres")

    # SQLite
    SQLITE_PATH  = os.getenv("SQLITE_PATH", "./vkr.sqlite")

    UPLOAD_DIR   = os.getenv("UPLOAD_DIR", "./uploads")
    os.makedirs(UPLOAD_DIR, exist_ok=True)
