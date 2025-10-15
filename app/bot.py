import os
from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import Command

from .config import Cfg
from .db import ensure_user, get_conn
from .parsing import parse_docx, parse_pdf, save_upload
from .indexing import index_document
from .retrieval import (
    retrieve, build_context, invalidate_cache,
    _mk_table_pattern, _mk_figure_pattern, keyword_find
)
from .polza_client import chat
from .ace import ace_once, SYS_ANSWER

bot = Bot(Cfg.TG_TOKEN)
dp = Dispatcher()

# Память режима в ОЗУ (MVP). Можно вынести в БД.
PRECISE_MODE: dict[int, bool] = {}   # user_id -> True/False
ACTIVE_DOC:   dict[int, int] = {}    # user_id -> doc_id


@dp.message(Command("start"))
async def start(m: types.Message):
    ensure_user(str(m.from_user.id))
    await m.answer(
        "Привет! Загрузите ВКР (DOCX/PDF) командой /upload.\n"
        "Команды: /mode — переключить 'Точный режим', /doc — показать активный документ."
    )


@dp.message(Command("mode"))
async def mode(m: types.Message):
    uid = ensure_user(str(m.from_user.id))
    PRECISE_MODE[uid] = not PRECISE_MODE.get(uid, False)
    await m.answer(f"Точный режим: {'ВКЛ' if PRECISE_MODE[uid] else 'ВЫКЛ'}")


@dp.message(Command("doc"))
async def doc(m: types.Message):
    uid = ensure_user(str(m.from_user.id))
    doc_id = ACTIVE_DOC.get(uid)
    await m.answer(f"Активный документ: {doc_id if doc_id else 'не выбран (загрузите /upload)'}")


@dp.message(Command("upload"))
async def ask_upload(m: types.Message):
    await m.answer("Пришлите файл ВКР (DOCX или PDF). Можно добавить подпись — я сразу отвечу на неё после индексации.")


async def respond_with_answer(m: types.Message, uid: int, doc_id: int, q_text: str):
    """Единый путь ответа (используется и после загрузки, и в обычных текстах)."""
    q_text = (q_text or "").strip()
    if not q_text:
        await m.answer("Вопрос пустой. Напишите, что именно вас интересует по ВКР.")
        return

    # 1) keyword-fallback: «таблица/рисунок 3.1»
    pat = _mk_table_pattern(q_text) or _mk_figure_pattern(q_text)
    if pat:
        hits_kw = keyword_find(uid, doc_id, pat)
        if hits_kw:
            txt = "\n\n".join(
                [f"Нашёл упоминание на стр. {h['page']} ({h['section_path']}):\n«{h['snippet']}»"
                 for h in hits_kw]
            )
            await m.answer(txt)
            return

    # 2) обычный RAG
    hits = retrieve(uid, doc_id, q_text, top_k=8)
    if not hits or hits[0]["score"] < 0.20:
        await m.answer("В загруженной работе нет достаточных оснований ответить. "
                       "Уточните раздел или добавьте нужные фрагменты.")
        return

    ctx = build_context(hits)

    # 3) режим
    if PRECISE_MODE.get(uid, False):
        reply = ace_once(q_text, ctx)  # ACE-lite
    else:
        reply = chat([
            {"role": "system", "content": SYS_ANSWER},
            {"role": "assistant", "content": f"Контекст:\n{ctx}"},
            {"role": "user", "content": q_text}
        ], temperature=0.2)

    # 4) ограничения Telegram
    for chunk in (reply[i:i+3900] for i in range(0, len(reply), 3900)):
        await m.answer(chunk)


@dp.message(F.document)
async def handle_doc(m: types.Message):
    uid = ensure_user(str(m.from_user.id))
    doc = m.document
    file = await bot.get_file(doc.file_id)
    stream = await bot.download_file(file.file_path)
    data = stream.read()
    stream.close()

    filename = f"{m.from_user.id}_{doc.file_name}"
    path = save_upload(data, filename, Cfg.UPLOAD_DIR)

    # распарсим
    kind = "thesis"
    if filename.lower().endswith(".docx"):
        sections = parse_docx(path)
    elif filename.lower().endswith(".pdf"):
        sections = parse_pdf(path)
    else:
        await m.answer("Поддерживаю только .docx и .pdf в MVP.")
        return

    # детектор «пустого/скан-PDF»
    total_chars = sum(len(s.get("text") or "") for s in sections)
    if total_chars < 500:
        await m.answer("Похоже, файл пустой или это скан-PDF без текста. "
                       "Загрузите, пожалуйста, DOCX или текстовый PDF.")
        return

    # документ → БД (SQLite) и индексация
    con = get_conn()
    cur = con.cursor()
    cur.execute("INSERT INTO documents(owner_id, kind, path) VALUES(?,?,?)", (uid, kind, path))
    doc_id = cur.lastrowid
    con.commit()
    con.close()

    index_document(uid, doc_id, sections)
    invalidate_cache(uid, doc_id)
    ACTIVE_DOC[uid] = doc_id

    caption = (m.caption or "").strip()
    if caption:
        await m.answer(f"Документ #{doc_id} проиндексирован. Отвечаю на ваш вопрос из подписи…")
        await respond_with_answer(m, uid, doc_id, caption)
    else:
        await m.answer(f"Готово. Документ #{doc_id} проиндексирован. Спросите что-нибудь по ВКР!")


@dp.message(F.text & ~F.via_bot)
async def qa(m: types.Message):
    uid = ensure_user(str(m.from_user.id))
    doc_id = ACTIVE_DOC.get(uid)
    if not doc_id:
        await m.answer("Сначала загрузите ВКР командой /upload.")
        return
    await respond_with_answer(m, uid, doc_id, m.text)
