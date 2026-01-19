import os
import sqlite3
import pytest

@pytest.mark.skipif(not os.getenv("BOT_DB_PATH"), reason="Set BOT_DB_PATH to sqlite path with chunks")
def test_real_index_contains_ch3_cfo():
    db = os.environ["BOT_DB_PATH"]
    owner_id = int(os.getenv("TEST_OWNER_ID", "1"))
    doc_id = int(os.getenv("TEST_DOC_ID", "1"))

    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    cur.execute("SELECT COUNT(*) as n FROM chunks WHERE owner_id=? AND doc_id=? AND text LIKE '%ЦФО%'", (owner_id, doc_id))
    n_cfo = cur.fetchone()["n"]

    cur.execute("SELECT COUNT(*) as n FROM chunks WHERE owner_id=? AND doc_id=? AND text LIKE '%Таблица 3.1%'", (owner_id, doc_id))
    n_tbl = cur.fetchone()["n"]

    con.close()

    assert n_cfo > 0, "In real index: no 'ЦФО' found in chunks -> ingest/index issue"
    assert n_tbl > 0, "In real index: no 'Таблица 3.1' found -> ingest/index issue"
