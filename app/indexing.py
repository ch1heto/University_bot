import numpy as np
from .db import get_conn
from .polza_client import embeddings
from .chunking import split_into_chunks

def index_document(owner_id:int, doc_id:int, sections:list[dict]):
    rows_text, rows_meta = [], []
    for s in sections:
        for ch in split_into_chunks(s["text"]):
            rows_text.append(ch)
            rows_meta.append({"page": s.get("page"),
                              "section_path": s.get("section_path") or s.get("title","")})
    if not rows_text:
        return
    vecs = embeddings(rows_text)  # list[list[float]]
    con = get_conn(); cur = con.cursor()
    for text, vec, meta in zip(rows_text, vecs, rows_meta):
        blob = np.asarray(vec, dtype=np.float32).tobytes()
        cur.execute(
            "INSERT INTO chunks(doc_id, owner_id, page, section_path, text, embedding) "
            "VALUES(?,?,?,?,?,?)",
            (doc_id, owner_id, meta["page"], meta["section_path"], text, blob)
        )
    con.commit(); con.close()
