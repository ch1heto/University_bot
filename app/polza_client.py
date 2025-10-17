# app/polza_client.py
from __future__ import annotations

import base64
import hashlib
import json
import logging
import mimetypes
import os
from typing import List, Dict, Any, Optional, Union, Tuple

from openai import OpenAI

from .config import Cfg  # ключи/модели/базовый URL


# -----------------------------
# Клиент Polza (OpenAI-совместимый)
# -----------------------------
_client = OpenAI(
    base_url=Cfg.BASE_POLZA,   # напр.: "https://api.polza.ai/api/v1"
    api_key=Cfg.POLZA_KEY,
)

__all__ = [
    "embeddings",
    "chat_with_gpt",
    "probe_embedding_dim",
    "vision_describe",
    "vision_describe_many",
]


# -----------------------------
# Внутренние хелперы
# -----------------------------

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
    Приводим ответ под вставку в «… изображено ___». Убираем служебные фразы,
    сжимаем пробелы, оставляем одно предложение.
    """
    t = (s or "").strip()

    # убираем дежурные вводные
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

    # ещё одна частая конструкция
    t = t.replace("Визуализируется", "изображена")

    # одна строка
    t = " ".join(t.split())

    # обрежем до первой точки, если модель разошлась
    if "." in t:
        t = t.split(".")[0] + "."

    return t if t else "содержимое изображения (описание не распознано)."


# Простенький in-memory кэш (на процесс): (sha256, lang, model) -> {"description": ..., "tags": [...]}
_VISION_CACHE: Dict[Tuple[str, str, str], Dict[str, Any]] = {}


# -----------------------------
# Публичные API
# -----------------------------

def embeddings(texts: List[str]) -> List[List[float]]:
    """
    Получить эмбеддинги для списка текстов.
    Возвращает список векторов float[]. Пустой ввод -> пустой список.
    """
    if not texts:
        return []
    try:
        resp = _client.embeddings.create(
            model=Cfg.POLZA_EMB,   # напр.: "openai/text-embedding-3-large"
            input=texts,
        )
        return [item.embedding for item in resp.data]
    except Exception as e:
        logging.exception(f"Ошибка при получении эмбеддингов: {e}")
        raise


def chat_with_gpt(
    messages: List[Dict[str, Any]],
    temperature: float = 0.2,
    max_tokens: int = 800,
) -> str:
    """
    Отправить диалог в чат-модель и вернуть текстовый ответ.
    Совместимо с Polza (OpenAI Chat Completions API).
    """
    try:
        cmpl = _client.chat.completions.create(
            model=Cfg.POLZA_CHAT,  # напр.: "openai/gpt-4o"
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = cmpl.choices[0].message.content or ""
        return (content or "").strip()
    except Exception as e:
        logging.error(f"Ошибка при запросе к чат-модели: {e}")
        return "Произошла ошибка при обработке запроса. Попробуйте позже."


def probe_embedding_dim(default: int | None = None) -> int | None:
    """
    Вспомогательно: вернуть размерность текущей эмбеддинг-модели.
    Полезно, чтобы сверять с сохранённой размерностью в индексе.
    """
    try:
        vec = embeddings(["__probe__"])[0]
        return len(vec)
    except Exception:
        return default


# -----------------------------
# Vision: описание изображений через GPT-4o
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
    Вспомогательный вызов chat.completions с fallback:
    1) пробуем с response_format=json_object (если want_tags=True);
    2) при ошибке повторяем без response_format.
    """
    sys = system_hint or (
        "Ты помощник по дипломам. Дай краткое, деловое описание картинки, "
        "одним предложением, без вводных типа 'На рисунке', 'На изображении', "
        "без местоимений и оценок. Сначала предмет изображения, затем контекст/назначение, "
        "затем 1–3 ключевые детали. Лаконично."
    )

    # Первая попытка — с response_format (если нужно строго JSON)
    try:
        cmpl = _client.chat.completions.create(
            model=Cfg.POLZA_CHAT,
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": content_parts},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"} if want_tags else None,
        )
        return (cmpl.choices[0].message.content or "").strip()
    except Exception as e:
        # Лояльный фолбэк — без response_format
        logging.warning(f"vision JSON response_format failed, retrying without it: {e}")
        try:
            cmpl = _client.chat.completions.create(
                model=Cfg.POLZA_CHAT,
                messages=[
                    {"role": "system", "content": sys},
                    {"role": "user", "content": content_parts},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return (cmpl.choices[0].message.content or "").strip()
        except Exception as e2:
            logging.exception(f"vision_describe: обе попытки не удались: {e2}")
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

    • image_or_images — путь к файлу или список путей.
    • lang — язык ответа ("ru"|"en"|"kk"), по умолчанию русский.
    • want_tags — если True, попросим компактные теги (тип изображения/объекты).
    • Возвращает: {"description": "...", "tags": ["...","..."]}

    Реализация: Chat Completions с контентом вида:
    [{"type":"text","text":...}, {"type":"image_url","image_url":{"url":"data:...base64"}}]
    """
    paths: List[str] = [image_or_images] if isinstance(image_or_images, str) else list(image_or_images or [])
    paths = [p for p in paths if p and os.path.exists(p)]
    if not paths:
        return {"description": "изображение отсутствует или не доступно.", "tags": []}

    # кэш: все пути -> сводим в один ключ
    shas = [_sha256_file(p) for p in paths]
    multi_key = hashlib.sha256(("|".join(shas)).encode("utf-8")).hexdigest()
    cache_key = (multi_key, (lang or "ru").lower(), Cfg.POLZA_CHAT or "openai/gpt-4o")
    if cache_key in _VISION_CACHE:
        return _VISION_CACHE[cache_key]

    # языковая подсказка
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

    # контент: текст + 1..N изображений (data URL)
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

    # стараемся распарсить JSON; если не получилось — используем как есть
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
    per_image_limit: int = 4,  # параметр сохранён для обратной совместимости, сейчас не батчим
) -> List[Dict[str, Any]]:
    """
    Возвращает **one-in → one-out**: на каждый путь — один словарь результата.
    Внутри использует кэш по SHA256, так что повторяющиеся изображения не перегружают модель.

    ПРИМЕЧАНИЕ: Раньше функция могла возвращать один результат на целый батч.
    Теперь гарантируется длина результата == длине входного списка.
    """
    out: List[Dict[str, Any]] = []
    for p in images or []:
        if not p or not os.path.exists(p):
            out.append({"description": "файл не найден", "tags": []})
            continue
        try:
            out.append(vision_describe(p, lang=lang))
        except Exception as e:
            logging.exception(f"vision_describe_many: ошибка на файле {p}: {e}")
            out.append({"description": "описание недоступно из-за ошибки обработки.", "tags": []})
    return out
