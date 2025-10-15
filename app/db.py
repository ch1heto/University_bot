import sqlite3
from pathlib import Path
from .config import Cfg

def get_conn():
    # WAL даёт «один писатель, много читателей» без блокировок
    Path(Cfg.SQLITE_PATH).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(Cfg.SQLITE_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con

def ensure_user(tg_id: str) -> int:
    con = get_conn(); cur = con.cursor()
    cur.execute("INSERT OR IGNORE INTO users(tg_id) VALUES (?)", (tg_id,))
    cur.execute("SELECT id FROM users WHERE tg_id = ?", (tg_id,))
    uid = cur.fetchone()["id"]
    con.commit(); con.close()
    return uid
