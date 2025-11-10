#!/usr/bin/env python3
import os, json, sqlite3, argparse, re
try:
    # если есть конфиг проекта — возьмём пути оттуда
    from app.config import Cfg
    DB_PATH = Cfg.SQLITE_PATH
    UPLOAD_DIR = Cfg.UPLOAD_DIR
except Exception:
    DB_PATH = os.getenv("SQLITE_PATH", "./vkr.sqlite")
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")

def _parse_chart_data(attrs: dict):
    """Возвращает (chart_type, pairs[(label,value,unit)]) из attrs.chart_data (или близких схем)."""
    a = attrs or {}
    raw = (a.get("chart_data")
           or (a.get("chart") or {}).get("data")
           or a.get("data")
           or a.get("series"))
    ctype = (a.get("chart_type")
             or (a.get("chart") or {}).get("type")
             or a.get("type"))

    pairs = []
    if isinstance(raw, list):
        for it in raw:
            if not isinstance(it, dict): 
                continue
            label = str(it.get("label") or it.get("category") or it.get("name") or "").strip()
            val   = it.get("value")
            if val is None:
                val = it.get("y") or it.get("v")
            unit  = (it.get("unit") or "").strip()
            if label or val is not None:
                pairs.append((label, val, unit))
    elif isinstance(raw, dict) and raw.get("categories") and raw.get("series"):
        cats = list(raw.get("categories") or [])
        s0 = (raw.get("series") or [{}])[0] or {}
        vals = s0.get("values") or s0.get("data") or []
        unit = (s0.get("unit") or "").strip()
        for i in range(min(len(cats), len(vals))):
            pairs.append((str(cats[i]), vals[i], unit))
    return ctype, pairs

def _exists_any(path: str) -> tuple[bool, str]:
    """Проверяем как есть и с UPLOAD_DIR для относительных путей."""
    if os.path.isabs(path):
        return os.path.exists(path), path
    p1 = path
    p2 = os.path.join(UPLOAD_DIR, path)
    if os.path.exists(p1): 
        return True, os.path.abspath(p1)
    if os.path.exists(p2): 
        return True, os.path.abspath(p2)
    return False, os.path.abspath(p1)

def main():
    ap = argparse.ArgumentParser(description="Проверка наличия изображений/данных для рисунка в БД.")
    ap.add_argument("--db", default=DB_PATH, help="Путь к SQLite (по умолчанию из config.py/ENV).")
    ap.add_argument("--num", required=True, help="Номер рисунка, напр. 4 или 2.1")
    ap.add_argument("--doc-id", type=int, help="Документ id (опционально).")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    where = "element_type='figure'"
    params = []
    if args.doc_id:
        where += " AND doc_id=?"; params.append(args.doc_id)

    # ищем по attrs.caption_num / attrs.label и по section_path
    like1 = f'%\\"caption_num\\": \\"{args.num}\\"%'
    like2 = f'%\\"label\\": \\"{args.num}\\"%'
    where += " AND (attrs LIKE ? OR attrs LIKE ? OR section_path LIKE ?)"
    params += [like1, like2, f"%Рисунок {args.num}%"]

    sql = f"""SELECT id, owner_id, doc_id, page, section_path, text, attrs
              FROM chunks
              WHERE {where}
              ORDER BY id ASC
              LIMIT 20"""
    cur.execute(sql, params)
    rows = cur.fetchall()

    if not rows:
        print(f"[!] Не найдено записей для Рисунка {args.num}. Проверь номер/документ.")
        return

    for i, r in enumerate(rows, 1):
        print("="*80)
        print(f"[{i}] doc_id={r['doc_id']} page={r['page']} section_path={r['section_path']}")
        txt = (r["text"] or "").strip()
        if txt:
            print("Текст/подпись:", (txt[:180] + "…") if len(txt) > 180 else txt)

        # attrs -> dict
        try:
            attrs = json.loads(r["attrs"] or "{}")
        except Exception:
            attrs = {}
        images = list(attrs.get("images") or [])
        ctype, pairs = _parse_chart_data(attrs)

        # images
        if images:
            print("images (из attrs):")
            for p in images:
                ok, real = _exists_any(p)
                mark = "OK" if ok else "MISSING"
                print(f"  - {p}  -> {mark}  ({real})")
        else:
            print("images: [] (в attrs нет путей к файлам)")

        # chart_data
        if pairs:
            print(f"chart_data: type={ctype or 'unknown'}, {len(pairs)} пар значений. Превью:")
            for lab, val, unit in pairs[:12]:
                u = f" {unit}" if unit else ""
                print(f"  — {lab}: {val}{u}")
        else:
            print("chart_data: отсутствует")

    conn.close()

if __name__ == "__main__":
    main()
