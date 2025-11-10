# scripts/vision_diag.py
from __future__ import annotations
import argparse, os, json, re, sys
from typing import List, Optional

# подключаем приложение
sys.path.append(os.path.abspath("."))

from app.config import Cfg
from app.db import get_conn
from app.polza_client import (
    vision_describe,
    vision_extract_values,
    chat_with_gpt,
    _image_part_for,           # используем для быстрого data:URL
)
from app.retrieval import describe_figures_by_numbers

FIG_NUM_RE = re.compile(r"(?i)\b(?:рис\w*|figure|fig\.?)\s*([A-Za-zА-Яа-я]?\s*[\d.,]+)")

def _print(title: str, payload):
    print(f"\n=== {title} ===")
    if isinstance(payload, (dict, list)):
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(str(payload).strip())

def smoke_direct(image_path: str):
    _print("CONFIG",
           {"vision_active": Cfg.vision_active(),
            "vision_model": Cfg.vision_model(),
            "transport": Cfg.VISION_IMAGE_TRANSPORT})
    if not os.path.exists(image_path):
        _print("SMOKE", f"Файл не найден: {image_path}")
        return False
    res = vision_describe(image_path, lang="ru")
    ok = bool((res or {}).get("description"))
    _print("SMOKE → vision_describe(image)", res)
    if not ok:
        _print("SMOKE RESULT", "❌ Модель не ответила по картинке. Смотрим ключ/модель/лог сети.")
    else:
        _print("SMOKE RESULT", "✅ Картинка ушла напрямую в модель.")
    return ok

def db_audit(uid: int, doc_id: int):
    con = get_conn(); cur = con.cursor()
    cur.execute("""SELECT COUNT(*) AS c FROM chunks
                   WHERE owner_id=? AND doc_id=? AND element_type='figure'""",
                (uid, doc_id))
    total_figs = int(cur.fetchone()["c"] or 0)
    cur.execute("""SELECT section_path, attrs FROM chunks
                   WHERE owner_id=? AND doc_id=? AND element_type='figure'
                   ORDER BY id ASC LIMIT 200""", (uid, doc_id))
    rows = cur.fetchall() or []
    with_images, existing = 0, 0
    sample_nums: List[str] = []
    for r in rows:
        attrs = {}
        try:
            attrs = json.loads(r["attrs"] or "{}")
        except Exception:
            pass
        imgs = list((attrs.get("images") or []))
        if imgs:
            with_images += 1
            for p in imgs:
                if isinstance(p, str) and os.path.exists(p):
                    existing += 1; break
        # вытащим номер, если есть
        num = (attrs.get("caption_num") or attrs.get("label") or "")
        if not num:
            m = re.search(r"(?i)\bрисунок\s+([A-Za-zА-Яа-я]?\s*[\d.,]+)", r["section_path"] or "")
            if m: num = m.group(1)
        if num and len(sample_nums) < 8:
            sample_nums.append(str(num).replace(" ", "").replace(",", "."))
    con.close()

    _print("DB AUDIT",
           {"figures_total": total_figs,
            "figures_with_attrs_images": with_images,
            "figures_with_existing_file": existing,
            "sample_nums_for_test": sample_nums})
    return total_figs, with_images, existing, sample_nums

def retrieval_cards(uid: int, doc_id: int, nums: List[str]):
    if not nums:
        _print("RETRIEVAL", "Нет номеров для проверки.")
        return []
    cards = describe_figures_by_numbers(uid, doc_id, nums, sample_chunks=1, use_vision=False, lang="ru") or []
    brief = []
    for c in cards:
        brief.append({
            "num": c.get("num"),
            "display": c.get("display"),
            "images_cnt": len(c.get("images") or []),
            "first_image_exists": bool((c.get("images") or []) and os.path.exists((c.get("images") or [None])[0])),
            "has_highlights": bool(c.get("highlights")),
        })
    _print("RETRIEVAL CARDS (images presence)", brief)
    return cards

def direct_chat_with_image(image_path: str):
    part = _image_part_for(image_path)
    if not part:
        _print("CHAT TEST", "Не удалось собрать image_url part.")
        return False
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Кратко опиши изображение одним предложением."},
            part,
        ],
    }]
    ans = chat_with_gpt(messages, temperature=0.0, max_tokens=120)
    _print("CHAT COMPLETIONS (with image_url)", ans)
    return bool(ans.strip())

def values_from_first_card(cards):
    for c in cards:
        imgs = c.get("images") or []
        if imgs:
            res = vision_extract_values(imgs[:1], caption_hint=(c.get("highlights") or [None])[0], lang="ru")
            _print("VISION VALUES (first card)", res)
            return
    _print("VISION VALUES", "Нет карточек с изображениями.")

def main():
    ap = argparse.ArgumentParser(description="Диагностика цепочки vision")
    ap.add_argument("--image", help="Путь к картинке для прямого смока")
    ap.add_argument("--uid", type=int, help="owner_id пользователя")
    ap.add_argument("--doc-id", type=int, help="id документа")
    ap.add_argument("--nums", nargs="*", help="Конкретные номера рисунков (например: 4 2.1 3)")
    args = ap.parse_args()

    if args.image:
        smoke_direct(args.image)
        direct_chat_with_image(args.image)

    if args.uid and args.doc_id:
        total, with_imgs, existing, sample = db_audit(args.uid, args.doc_id)
        nums = args.nums or sample[:6]
        cards = retrieval_cards(args.uid, args.doc_id, nums)
        values_from_first_card(cards)

    if not (args.image or (args.uid and args.doc_id)):
        print("Запуск: \n"
              "  python scripts/vision_diag.py --image ./tests/chart.png\n"
              "  python scripts/vision_diag.py --uid 123 --doc-id 7 --nums 4 5 6")

if __name__ == "__main__":
    main()
