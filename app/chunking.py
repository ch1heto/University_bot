import re

def split_into_chunks(text:str, max_chars=1800, overlap=200) -> list[str]:
    # Грубое разбиение по абзацам/точкам
    paras = re.split(r"\n{2,}", text.strip())
    chunks, cur = [], ""
    for para in paras:
        if len(cur) + len(para) + 1 <= max_chars:
            cur += ("\n\n" if cur else "") + para
        else:
            if cur: chunks.append(cur)
            cur = para[-(max_chars-overlap):] if len(para)>max_chars else para
    if cur: chunks.append(cur)
    return chunks
