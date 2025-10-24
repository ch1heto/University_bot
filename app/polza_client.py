# app/polza_client.py
from __future__ import annotations

import base64
import hashlib
import json
import logging
import mimetypes
import os
import re
from typing import List, Dict, Any, Optional, Union, Tuple, Iterable

from openai import OpenAI

from .config import Cfg  # ключи/модели/базовый URL


# -----------------------------
# Клиент Polza (OpenAI-совместимый)
# -----------------------------
_client = OpenAI(
    base_url=(Cfg.BASE_POLZA or "").rstrip("/"),
    api_key=Cfg.POLZA_KEY,
)

__all__ = [
    "embeddings",
    "chat_with_gpt",
    "chat_with_gpt_stream",
    "probe_embedding_dim",
    "vision_describe",
    "vision_describe_many",
]


# -----------------------------
# Внутренние хелперы
# -----------------------------

# Удаляем управляющие символы, которые могут поломать JSON в SSE потоках (кроме \t, \r, \n)
_CTRL_BAD = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")

def _sanitize_text_for_json(s: str) -> str:
    """
    Убираем символы управления + U+2028/U+2029 (line/para sep), которые иногда не экранируются.
    """
    t = (s or "")
    t = _CTRL_BAD.sub(" ", t)
    t = t.replace("\u2028", " ").replace("\u2029", " ")
    return t


def _sanitize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Санитизируем текстовые фрагменты сообщений:
    - message['content'] если это str;
    - content-парт {"type":"text","text":...} если контент — массив частей.
    Остальные части (image_url и т.п.) не трогаем.
    """
    out: List[Dict[str, Any]] = []
    for m in (messages or []):
        mm = dict(m)
        c = mm.get("content")
        if isinstance(c, str):
            mm["content"] = _sanitize_text_for_json(c)
        elif isinstance(c, list):
            parts: List[Any] = []
            for p in c:
                if isinstance(p, dict) and p.get("type") == "text":
                    q = dict(p)
                    q["text"] = _sanitize_text_for_json(str(q.get("text", "")))
                    parts.append(q)
                else:
                    parts.append(p)
            mm["content"] = parts
        out.append(mm)
    return out


def _chunk_text(s: str, maxlen: int = 480) -> Iterable[str]:
    """
    Аккуратно режем длинный текст для отправки частями.
    """
    s = s or ""
    if not s:
        return []
    i, n = 0, len(s)

    def _cut_point(seg: str, limit: int) -> int:
        if len(seg) <= limit:
            return len(seg)
        for token in ("\n", ". ", " "):
            pos = seg.rfind(token, 0, limit)
            if pos != -1:
                return pos + (0 if token == "\n" else len(token))
        return limit

    while i < n:
        cut = _cut_point(s[i:], maxlen)
        yield s[i:i + cut]
        i += cut


def _file_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _detect_mime(path: str, data: Optional[bytes] = None) -> str:
    # 1) по расширению
    mime, _ = mimetypes.guess_type(path)
    if mime:
        return mime
    # 2) по простым сигнатурам
    b = data or b""
    if b.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if b.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if b.startswith(b"GIF87a") or b.startswith(b"GIF89a"):
        return "image/gif"
    if b[:4] in (b"II*\x00", b"MM\x00*"):
        return "image/tiff"
    return "application/octet-stream"


def _to_data_url(path: str) -> str:
    raw = _file_bytes(path)
    mime = _detect_mime(path, raw)
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sanitize_description(s: str) -> str:
    """
    Приводим ответ под вставку в «… изображено ___». Убираем вводные, сжимаем до 1 предложения.
    """
    t = (s or "").strip()
    repl = {
        "На рисунке показано": "",
        "На рисунке изображено": "",
        "На изображении показано": "",
        "На изображении изображено": "",
        "На фото показано": "",
        "На фото изображено": "",
    }
    for k, v in repl.items():
        t = t.replace(k, v)
    t = t.replace("Визуализируется", "изображена")
    t = " ".join(t.split())
    if "." in t:
        t = t.split(".")[0] + "."
    return t if t else "содержимое изображения (описание не распознано)."


# Простенький in-memory кэш (на процесс)
_VISION_CACHE: Dict[Tuple[str, str, str], Dict[str, Any]] = {}


# -----------------------------
# Публичные API
# -----------------------------

def embeddings(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    try:
        resp = _client.embeddings.create(
            model=Cfg.POLZA_EMB,
            input=texts,
        )
        return [item.embedding for item in resp.data]
    except Exception as e:
        logging.exception("Ошибка при получении эмбеддингов: %s", e)
        raise


def chat_with_gpt(
    messages: List[Dict[str, Any]],
    temperature: float = 0.2,
    max_tokens: int = 800,
) -> str:
    """
    Нестримовый чат-вызов (с санацией сообщений).
    """
    try:
        smsg = _sanitize_messages(messages)
        cmpl = _client.chat.completions.create(
            model=Cfg.POLZA_CHAT,
            messages=smsg,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = cmpl.choices[0].message.content or ""
        return (content or "").strip()
    except Exception as e:
        logging.error("Ошибка при запросе к чат-модели: %s", e)
        return "Произошла ошибка при обработке запроса. Попробуйте позже."


def chat_with_gpt_stream(
    messages: List[Dict[str, Any]],
    temperature: float = 0.2,
    max_tokens: int = 800,
) -> Iterable[str]:
    """
    Стриминговая версия chat-комплишна (с санацией входа и устойчивым фолбэком).
    Если стрим ломается ДО первой дельты — автоматически переключаемся на нестрим и
    отдаём ответ кусками. Если ПОСЛЕ — просто завершаем поток без исключения.
    """
    def _gen():
        stream = None
        yielded_any = False
        try:
            smsg = _sanitize_messages(messages)
            stream = _client.chat.completions.create(
                model=Cfg.POLZA_CHAT,
                messages=smsg,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            # OpenAI SDK v1: по чанкам; контент в choices[0].delta.content
            for chunk in stream:
                try:
                    choice = (chunk.choices or [None])[0]
                    if not choice:
                        continue
                    delta = getattr(choice, "delta", None)
                    if not delta:
                        continue
                    text = getattr(delta, "content", None) or ""
                    if text:
                        yielded_any = True
                        yield text
                except Exception as inner_e:
                    # Спорный/неполный chunk — пропускаем
                    logging.debug("stream chunk parse warning: %s", inner_e)
                    continue
        except Exception as e:
            # Самая частая причина по логам: JSONDecodeError в openai._streaming.sse.json()
            logging.error("chat_with_gpt_stream: stream aborted: %s", e)
            if not yielded_any:
                # Мягкий фолбэк: единый ответ → режем на части
                try:
                    answer = chat_with_gpt(messages, temperature=temperature, max_tokens=max_tokens)
                    for part in _chunk_text(answer, 480):
                        yield part
                except Exception as e2:
                    logging.exception("chat_with_gpt_stream fallback failed: %s", e2)
        finally:
            try:
                if stream and hasattr(stream, "close"):
                    stream.close()
            except Exception:
                pass

    return _gen()


def probe_embedding_dim(default: int | None = None) -> int | None:
    try:
        vecs = embeddings(["__probe__"])
        if not vecs:
            return default
        return len(vecs[0])
    except Exception:
        return default


# -----------------------------
# Vision
# -----------------------------

def _vision_messages(
    content_parts: List[Dict[str, Any]],
    *,
    system_hint: Optional[str],
    temperature: float,
    max_tokens: int,
    want_tags: bool
) -> str:
    """
    Вызов chat.completions с контентом-массивом (image_url + текст).
    Если want_tags=True — просим JSON (response_format=json_object); при ошибке — повторяем без него.
    """
    sys = system_hint or (
        "Ты помощник по дипломам. Дай краткое, деловое описание картинки, "
        "одним предложением, без вводных типа 'На рисунке', 'На изображении', "
        "без местоимений и оценок. Сначала предмет изображения, затем контекст/назначение, "
        "затем 1–3 ключевые детали. Лаконично."
    )

    # Готовим санитизированный контент (как массив частей)
    sanitized = _sanitize_messages([{"role": "user", "content": content_parts}])[0]["content"]

    try:
        kwargs: Dict[str, Any] = dict(
            model=Cfg.POLZA_CHAT,
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": sanitized},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if want_tags:
            kwargs["response_format"] = {"type": "json_object"}
        cmpl = _client.chat.completions.create(**kwargs)
        return (cmpl.choices[0].message.content or "").strip()
    except Exception as e:
        logging.warning("vision JSON response_format failed, retrying without it: %s", e)
        try:
            cmpl = _client.chat.completions.create(
                model=Cfg.POLZA_CHAT,
                messages=[
                    {"role": "system", "content": sys},
                    {"role": "user", "content": sanitized},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return (cmpl.choices[0].message.content or "").strip()
        except Exception as e2:
            logging.exception("vision_describe: обе попытки не удались: %s", e2)
            return ""


def vision_describe(
    image_or_images: Union[str, List[str]],
    lang: str = "ru",
    *,
    want_tags: bool = True,
    temperature: float = 0.0,
    max_tokens: int = 180,
    system_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Описывает изображение(я) «для подписи»: возвращает JSON {description, tags}.
    """
    paths: List[str] = [image_or_images] if isinstance(image_or_images, str) else list(image_or_images or [])
    paths = [p for p in paths if p and os.path.exists(p)]
    if not paths:
        return {"description": "изображение отсутствует или не доступно.", "tags": []}

    # Кэш: все пути -> сводим в один ключ
    shas = [_sha256_file(p) for p in paths]
    multi_key = hashlib.sha256(("|".join(shas)).encode("utf-8")).hexdigest()
    cache_key = (multi_key, (lang or "ru").lower(), Cfg.POLZA_CHAT or "openai/gpt-4o")
    if cache_key in _VISION_CACHE:
        return _VISION_CACHE[cache_key]

    # Языковая подсказка
    lang_hint = {
        "ru": "Отвечай на русском.",
        "en": "Answer in English.",
        "kk": "Қазақ тілінде жауап бер.",
    }.get((lang or "ru").lower(), "Отвечай на русском.")

    instruction = {
        "ru": (
            "Верни JSON с полями 'description' и 'tags' (массив коротких тегов). "
            "description — одно предложение для подстановки после слов 'изображено ...'."
        ),
        "en": (
            "Return JSON with 'description' and 'tags' (array of short tags). "
            "description must be a single sentence that can follow 'shows ...'."
        ),
        "kk": (
            "JSON қайтар: 'description' және 'tags' (қысқа тегтер тізімі). "
            "description бір сөйлем болсын."
        ),
    }.get((lang or "ru").lower(), "Верни JSON с полями 'description' и 'tags'.")

    # Контент: текст + 1..N изображений (data URL)
    content_parts: List[Dict[str, Any]] = [{"type": "text", "text": f"{lang_hint} {instruction}"}]
    for p in paths:
        content_parts.append({"type": "image_url", "image_url": {"url": _to_data_url(p)}})

    raw = _vision_messages(
        content_parts,
        system_hint=system_hint,
        temperature=temperature,
        max_tokens=max_tokens,
        want_tags=want_tags,
    )

    # Стараемся распарсить JSON; если не получилось — используем как есть
    desc: str = ""
    tags: List[str] = []
    if raw:
        try:
            obj = json.loads(raw)
            desc = str(obj.get("description", "")).strip()
            tgs = obj.get("tags", [])
            if isinstance(tgs, list):
                tags = [str(x).strip() for x in tgs if str(x).strip()]
        except Exception:
            desc = raw.strip()

    desc = _sanitize_description(desc)
    res = {"description": desc, "tags": tags}
    _VISION_CACHE[cache_key] = res
    return res


def vision_describe_many(
    images: List[str],
    lang: str = "ru",
    *,
    per_image_limit: int = 4,  # параметр сохранён для обратной совместимости
) -> List[Dict[str, Any]]:
    """
    One-in → one-out: на каждый путь — один словарь результата.
    Повторяющиеся изображения кешируются по SHA256.
    """
    out: List[Dict[str, Any]] = []
    for p in images or []:
        if not p or not os.path.exists(p):
            out.append({"description": "файл не найден", "tags": []})
            continue
        try:
            out.append(vision_describe(p, lang=lang))
        except Exception as e:
            logging.exception("vision_describe_many: ошибка на файле %s: %s", p, e)
            out.append({"description": "описание недоступно из-за ошибки обработки.", "tags": []})
    return out
