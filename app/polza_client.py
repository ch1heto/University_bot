# app/polza_client.py
from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import mimetypes
import os
import re
import time
from typing import List, Dict, Any, Optional, Union, Tuple, Iterable

from openai import OpenAI
import requests

from .config import Cfg  # ключи/модели/базовый URL
from .templates import build_describe_prompt, build_extract_prompt

# Pillow — опционально (даунскейл/сжатие). Если нет, используем raw-байты.
try:
    from PIL import Image  # type: ignore
    _PIL_OK = True
except Exception:
    _PIL_OK = False


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
    # новое:
    "chat_with_gpt_multimodal",
    "chat_with_gpt_stream_multimodal",
    "probe_embedding_dim",
    "vision_describe",
    "vision_describe_many",
    "vision_extract_values",
]


# -----------------------------
# Внутренние хелперы
# -----------------------------

# Удаляем управляющие символы, которые могут поломать JSON в SSE потоках (кроме \t, \r, \n)
_CTRL_BAD = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")

def _sanitize_text_for_json(s: str) -> str:
    """
    Убираем символы управления + U+2028/U+2029 (line/para sep), а также одиночные обратные слэши
    на конце строки, которые иногда ломают сериализацию у провайдеров.
    """
    t = (s or "")
    t = _CTRL_BAD.sub(" ", t)
    t = t.replace("\u2028", " ").replace("\u2029", " ")
    # Простейший кейс «незавершённой строки»: финальный одиночный бэкслэш
    if t.endswith("\\"):
        t = t[:-1] + " "
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


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _resize_and_encode_base64(path: str) -> Tuple[str, int]:
    """
    Возвращает (data_url, size_bytes). Учитывает лимиты из конфига.
    Если Pillow недоступен — кодируем как есть (может быть больше лимита).
    """
    raw = _file_bytes(path)
    mime = _detect_mime(path, raw)

    # Если PIL есть — даунскейлим длинную сторону и сохраняем в JPEG/PNG
    if _PIL_OK:
        try:
            img = Image.open(io.BytesIO(raw))
            img = img.convert("RGB")  # во избежание экзотических режимов
            max_side = int(getattr(Cfg, "VISION_MAX_SIDE_PX", 2048) or 2048)
            w, h = img.size
            scale = 1.0
            if max(w, h) > max_side:
                scale = max_side / float(max(w, h))
                new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
                img = img.resize(new_size, Image.LANCZOS)

            # JPEG предпочтительнее по размеру
            out = io.BytesIO()
            img.save(out, format="JPEG", quality=int(getattr(Cfg, "VISION_JPEG_QUALITY", 88) or 88), optimize=True)
            data = out.getvalue()
            out.close()

            # Если всё равно слишком большое — ещё раз ужмём качеством
            max_bytes = int(getattr(Cfg, "VISION_MAX_IMAGE_BYTES", 2_500_000) or 2_500_000)
            if len(data) > max_bytes:
                q = max(40, int((getattr(Cfg, "VISION_JPEG_QUALITY", 88) or 88) * 0.75))
                out2 = io.BytesIO()
                img.save(out2, format="JPEG", quality=q, optimize=True)
                data2 = out2.getvalue()
                out2.close()
                if len(data2) < len(data):
                    data = data2

            mime = "image/jpeg"
            b64 = base64.b64encode(data).decode("ascii")
            return (f"data:{mime};base64,{b64}", len(data))
        except Exception:
            # Фолбэк — как есть
            pass

    b64 = base64.b64encode(raw).decode("ascii")
    return (f"data:{mime};base64,{b64}", len(raw))


def _path_is_url(p: str) -> bool:
    return bool(re.match(r"^https?://", str(p).strip(), flags=re.IGNORECASE))


def _image_part_for(path: str) -> Optional[Dict[str, Any]]:
    """
    Готовит message part под OpenAI/Polza:
      {"type":"image_url","image_url":{"url":"data:..."}}
    Транспорт выбирается по Cfg.VISION_IMAGE_TRANSPORT.
    """
    if not path:
        return None
    # фильтр расширений (мягкий)
    ext = os.path.splitext(path)[1].lstrip(".").lower()
    try:
        if ext and hasattr(Cfg, "is_image_ext_allowed") and not Cfg.is_image_ext_allowed(ext):
            # всё равно попробуем — многие экспортируют без расширения/с вебпом и т.п.
            pass
    except Exception:
        pass

    if _path_is_url(path):
        return {"type": "image_url", "image_url": {"url": path}}

    transport = (getattr(Cfg, "VISION_IMAGE_TRANSPORT", "base64") or "base64").lower()

    if transport == "url":
        # локальный путь при режиме url — мягкий фолбэк в base64 (data:)
        data_url, _ = _resize_and_encode_base64(path)
        return {"type": "image_url", "image_url": {"url": data_url}}

    # базовый случай — base64 (data:)
    data_url, _ = _resize_and_encode_base64(path)
    return {"type": "image_url", "image_url": {"url": data_url}}


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
        "Рисунок показывает": "",
        "Данное изображение показывает": "",
        "Изображение показывает": "",
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

# Дисковый кэш (переживает рестарт процесса)
def _vcache_path(key: Tuple[str, str, str]) -> str:
    # ключ может быть длинным — хэшируем в имя файла
    h = hashlib.sha256("|".join(key).encode("utf-8")).hexdigest()
    return os.path.join(getattr(Cfg, "VISION_CACHE_DIR", "/tmp/vision_cache"), f"{h}.json")

def _vcache_get(key: Tuple[str, str, str]) -> Optional[Dict[str, Any]]:
    try:
        fp = _vcache_path(key)
        if os.path.exists(fp):
            with open(fp, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return None

def _vcache_put(key: Tuple[str, str, str], data: Dict[str, Any]) -> None:
    try:
        os.makedirs(getattr(Cfg, "VISION_CACHE_DIR", "/tmp/vision_cache"), exist_ok=True)
        fp = _vcache_path(key)
        payload = dict(data)
        payload["_cached_at"] = int(time.time())
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
    except Exception:
        pass


# -----------------------------
# Публичные API — текст
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
    **extra: Any,
) -> str:
    """
    Нестримовый чат-вызов (с санацией сообщений).
    Поддерживает сообщения с image_url-партами (data: или http/https).
    Допускает дополнительные параметры, напр. response_format={"type":"json_object"}.
    """
    try:
        smsg = _sanitize_messages(messages)
        # Выбор модели: если в сообщениях есть image_url — используем vision-модель.
        use_vision = any(
            isinstance(m.get("content"), list) and any(
                isinstance(p, dict) and p.get("type") == "image_url"
                for p in m["content"]
            )
            for m in smsg
        )
        model = Cfg.vision_model() if (use_vision and Cfg.vision_active()) else Cfg.POLZA_CHAT

        # Фильтруем extra, чтобы не перетереть критичные ключи
        blocked = {"model", "messages", "temperature", "max_tokens", "stream"}
        pass_extra = {k: v for k, v in (extra or {}).items() if k not in blocked}

        cmpl = _client.chat.completions.create(
            model=model,
            messages=smsg,
            temperature=temperature,
            max_tokens=max_tokens,
            **pass_extra,
        )
        content = cmpl.choices[0].message.content or ""
        return (content or "").strip()
    except Exception as e:
        logging.error("Ошибка при запросе к чат-модели: %s", e)
        return "Произошла ошибка при обработке запроса. Попробуйте позже."


def _iter_sse_lines(resp) -> Iterable[str]:
    """
    Читаем SSE как БАЙТЫ и декодируем только UTF-8 (серверные charset-ы игнорируем),
    чтобы не ловить mojibake на cp1251/latin-1.
    """
    for raw in resp.iter_lines(decode_unicode=False):
        if not raw:
            continue
        try:
            line = raw.decode("utf-8", "replace")
        except Exception:
            # на всякий случай: если что-то совсем нетипичное
            line = raw.decode("latin-1", "replace")
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            yield line[5:].strip()
        else:
            yield line.strip()


def chat_with_gpt_stream(
    messages: List[Dict[str, Any]],
    temperature: float = 0.2,
    max_tokens: int = 800,
) -> Iterable[str]:
    """
    Стриминговая версия с ручным парсингом SSE (устойчиво к «битым» строкам JSON).
    Если поток обрывается до первой дельты — мягкий фолбэк на нестрим.
    Поддерживает image_url-парты.
    """
    def _gen():
        yielded_any = False
        resp = None
        try:
            smsg = _sanitize_messages(messages)

            # Если есть image_url — переключаемся на vision-модель.
            use_vision = any(
                isinstance(m.get("content"), list) and any(
                    isinstance(p, dict) and p.get("type") == "image_url"
                    for p in m["content"]
                )
                for m in smsg
            )
            model = Cfg.vision_model() if (use_vision and Cfg.vision_active()) else Cfg.POLZA_CHAT

            payload = {
                "model": model,
                "messages": smsg,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
            }
            headers = {
                "Authorization": f"Bearer {Cfg.POLZA_KEY}",
                "Content-Type": "application/json; charset=utf-8",
                "Accept": "text/event-stream",
                "Accept-Charset": "utf-8",
            }
            url = f"{(Cfg.BASE_POLZA or '').rstrip('/')}/chat/completions"
            # таймауты
            vision_timeout = float(getattr(Cfg, "VISION_TIMEOUT_SEC", 120) or 120.0)
            resp = requests.post(
                url,
                headers=headers,
                json=payload,
                stream=True,
                timeout=(10, vision_timeout if use_vision else 300),
            )
            resp.raise_for_status()

            pending: Optional[str] = None
            for data in _iter_sse_lines(resp):
                if data == "[DONE]":
                    break

                # Если до этого получили неполный JSON — склеим
                if pending:
                    data = pending + data
                    pending = None

                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    # Пришёл обрезанный JSON — ждём продолжение
                    pending = data
                    continue

                # Поддерживаем два формата:
                # 1) {"delta":"..."} — упрощённый
                # 2) {"choices":[{"delta":{"content":"..."}}]} — openai-совместимый
                delta = obj.get("delta")
                if delta is None:
                    chs = obj.get("choices") or []
                    if chs:
                        delta = (chs[0].get("delta") or {}).get("content")

                if delta:
                    yielded_any = True
                    yield str(delta)

            # Если поток завершился, а в pending висит валидный JSON — доберём
            if pending:
                try:
                    obj = json.loads(pending)
                    delta = obj.get("delta")
                    if delta is None:
                        chs = obj.get("choices") or []
                        if chs:
                            delta = (chs[0].get("delta") or {}).get("content")
                    if delta:
                        yielded_any = True
                        yield str(delta)
                except Exception:
                    pass

        except Exception as e:
            logging.error("chat_with_gpt_stream: stream aborted: %s", e)
            if not yielded_any:
                # Фолбэк на нестрим — режем на части, чтобы UI продолжал «капать»
                try:
                    answer = chat_with_gpt(messages, temperature=temperature, max_tokens=max_tokens)
                    for part in _chunk_text(answer, 480):
                        yield part
                except Exception as e2:
                    logging.exception("chat_with_gpt_stream fallback failed: %s", e2)
        finally:
            try:
                if resp is not None:
                    resp.close()
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
# Новое: простые мультимодальные обёртки (текст + изображения)
# -----------------------------

def _make_image_parts(paths: Optional[List[str]]) -> List[Dict[str, Any]]:
    if not paths:
        return []
    max_n = int(getattr(Cfg, "VISION_MAX_IMAGES_PER_REQUEST", 4) or 4)
    parts: List[Dict[str, Any]] = []
    for p in (paths or [])[:max_n]:
        if not p:
            continue
        if _path_is_url(p) or os.path.exists(p):
            part = _image_part_for(p)
            if part:
                parts.append(part)
    return parts

def chat_with_gpt_multimodal(
    prompt: str,
    image_paths: Optional[List[str]] = None,
    *,
    system: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 800,
    **extra: Any,
) -> str:
    """
    Упрощённый вызов чата с прикреплением 0..N изображений.
    Если VISION_DISABLED — упадём в обычный текстовый режим (изображения игнорируются).
    """
    user_content: List[Dict[str, Any]] = [{"type": "text", "text": prompt or ""}]
    user_content += _make_image_parts(image_paths)
    messages: List[Dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_content})
    return chat_with_gpt(messages, temperature=temperature, max_tokens=max_tokens, **extra)

def chat_with_gpt_stream_multimodal(
    prompt: str,
    image_paths: Optional[List[str]] = None,
    *,
    system: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 800,
) -> Iterable[str]:
    """
    Стрим-вариант мультимодального чата (текст + изображения).
    """
    user_content: List[Dict[str, Any]] = [{"type": "text", "text": prompt or ""}]
    user_content += _make_image_parts(image_paths)
    messages: List[Dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_content})
    return chat_with_gpt_stream(messages, temperature=temperature, max_tokens=max_tokens)


# -----------------------------
# Vision — low/high level
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
            model=Cfg.vision_model() if Cfg.vision_active() else Cfg.POLZA_CHAT,
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
            # Если ошибка доступа к модели (401/403) — мягко фоллбэкаем на чат-модель
            fallback_model = Cfg.vision_model() if Cfg.vision_active() else Cfg.POLZA_CHAT
            status = getattr(e, "status_code", None)
            if status in (401, 403):
                fallback_model = Cfg.POLZA_CHAT

            cmpl = _client.chat.completions.create(
                model=fallback_model,
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


def _norm_unit(u: Optional[str]) -> str:
    t = (u or "").strip().lower()
    if t in {"%", "percent", "perc", "pct"}:
        return "%"
    if t in {"‰", "permil", "per mille"}:
        return "‰"
    if t in {"pp", "п.п.", "пп"}:
        return "п.п."
    if t in {"pcs", "шт", "штук", "ед", "units", "unit"}:
        return "шт"
    if t in {"rub", "₽", "руб", "руб.", "р", "р."}:
        return "₽"
    if t in {"eur", "€", "евро"}:
        return "€"
    if t in {"usd", "$", "доллар", "долл."}:
        return "$"
    if t in {"тыс", "тыс.", "k"}:
        return "тыс."
    if t in {"млн", "млн.", "mln", "m"}:
        return "млн"
    return t


def vision_describe(
    image_or_images: Union[str, List[str]],
    lang: str = "ru",
    *,
    temperature: float = 0.0,
    max_tokens: int = 180,
    system_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Описывает изображение(я) «для подписи»: возвращает JSON {description, tags}.
    Вшивает картинки как image_url (data: base64) либо http(s) ссылку — по Cfg.VISION_IMAGE_TRANSPORT.
    Даунскейлит/сжимает до лимитов из конфига.
    """
    if not Cfg.vision_active():
        return {"description": "модуль анализа изображений отключён (VISION_ENABLED=False).", "tags": []}

    paths: List[str] = [image_or_images] if isinstance(image_or_images, str) else list(image_or_images or [])
    # фильтруем по существованию, но оставляем http(s)
    norm_paths: List[str] = []
    for p in paths:
        if not p:
            continue
        if _path_is_url(p) or os.path.exists(p):
            norm_paths.append(p)

    if not norm_paths:
        return {"description": "изображение отсутствует или не доступно.", "tags": []}

    # Лимит по количеству изображений
    norm_paths = norm_paths[: max(1, int(getattr(Cfg, "VISION_MAX_IMAGES_PER_REQUEST", 4) or 4))]

    # Кэш по набору изображений
    shas: List[str] = []
    for p in norm_paths:
        shas.append(p if _path_is_url(p) else _sha256_file(p))
    multi_key = hashlib.sha256(("|".join(shas)).encode("utf-8")).hexdigest()
    cache_key = (multi_key, (lang or "ru").lower(), Cfg.vision_model() or Cfg.POLZA_CHAT)

    # 1) память процесса
    if cache_key in _VISION_CACHE:
        return _VISION_CACHE[cache_key]
    # 2) дисковый кэш
    disk = _vcache_get(cache_key)
    if disk:
        _VISION_CACHE[cache_key] = disk
        return disk

    # Шаблон промпта (RU/EN)
    tmpl = build_describe_prompt(
        caption=None,
        context=None,
        lang=lang,
        sentences_min=Cfg.VISION_DESCRIBE_SENTENCES_MIN,
        sentences_max=Cfg.VISION_DESCRIBE_SENTENCES_MAX,
        require_json=Cfg.VISION_JSON_STRICT,
    )

    content_parts: List[Dict[str, Any]] = [{"type": "text", "text": tmpl["user"]}]
    for p in norm_paths:
        part = _image_part_for(p)
        if part:
            content_parts.append(part)

    raw = _vision_messages(
        content_parts,
        system_hint=(system_hint or tmpl["system"]),
        temperature=temperature,
        max_tokens=max_tokens,
        want_tags=Cfg.VISION_JSON_STRICT,  # просим JSON
    )

    # Стараемся распарсить JSON; если не получилось — используем как есть (текст модели)
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
            desc = (raw or "").strip()

    # Санитайз + фоллбэк, если после санитайза получилось пусто/заглушка
    sanitized = _sanitize_description(desc)
    _fallback_placeholder = "содержимое изображения (описание не распознано)."
    if (not sanitized or sanitized == _fallback_placeholder) and (raw or "").strip():
        # Возьмём первое предложение «как есть» из сырого текста
        r = (raw or "").strip()
        sanitized = (r.split(".")[0] + ".") if "." in r else r

    res = {"description": sanitized, "tags": tags}
    _VISION_CACHE[cache_key] = res
    _vcache_put(cache_key, res)
    return res


def vision_describe_many(
    images: List[str],
    lang: str = "ru",
    *,
    per_image_limit: int = 4,  # параметр сохранён для обратной совместимости
) -> List[Dict[str, Any]]:
    """
    One-in → one-out: на каждый путь — один словарь результата.
    Повторяющиеся изображения кешируются (по SHA256 для локальных и по URL-строке для http/https).
    """
    if not Cfg.vision_active():
        return [{"description": "модуль анализа изображений отключён (VISION_ENABLED=False).", "tags": []} for _ in (images or [])]

    out: List[Dict[str, Any]] = []
    for p in images or []:
        if not p or (not _path_is_url(p) and not os.path.exists(p)):
            out.append({"description": "файл не найден", "tags": []})
            continue
        try:
            out.append(vision_describe(p, lang=lang))
        except Exception as e:
            logging.exception("vision_describe_many: ошибка на файле %s: %s", p, e)
            out.append({"description": "описание недоступно из-за ошибки обработки.", "tags": []})
    return out


# -----------------------------
# Новое: извлечение значений с изображений (структурированный JSON)
# -----------------------------

def vision_extract_values(
    image_or_images: Union[str, List[str]],
    *,
    caption_hint: Optional[str] = None,
    ocr_hint: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 1400,
    lang: str = "ru",
) -> Dict[str, Any]:
    """
    Высокоуровневая обёртка для извлечения чисел/процентов/подписей/единиц.
    Возвращает объект JSON-совместимого словаря:
      {
        "type": "pie|bar|line|diagram|table|other",
        "units": {"x":"...","y":"..."},
        "axes": {"x_labels":["..."],"y_labels":["..."]},
        "legend": ["..."],
        "data": [{"label":"...","series":"optional","value":"...","unit":"optional","conf":0.0..1.0}],
        "raw_text": ["..."],
        "warnings": ["..."]
      }
    Если распознать не удалось — вернёт {"type":"other","warnings":[...]}.
    """
    if not Cfg.vision_active():
        return {"type": "other", "warnings": ["vision отключён (VISION_ENABLED=False)."]}

    # Нормализуем список путей
    paths: List[str] = [image_or_images] if isinstance(image_or_images, str) else list(image_or_images or [])
    norm_paths: List[str] = []
    for p in paths:
        if not p:
            continue
        if _path_is_url(p) or os.path.exists(p):
            norm_paths.append(p)

    if not norm_paths:
        return {"type": "other", "warnings": ["изображение отсутствует или недоступно."]}

    # Ограничим количество изображений
    norm_paths = norm_paths[: max(1, int(getattr(Cfg, "VISION_MAX_IMAGES_PER_REQUEST", 4) or 4))]

    # Кэш по (изображения + подсказки)
    sig_parts: List[str] = []
    for p in norm_paths:
        sig_parts.append(p if _path_is_url(p) else _sha256_file(p))
    if caption_hint:
        sig_parts.append("cap:" + caption_hint.strip())
    if ocr_hint:
        # не включаем весь OCR-текст целиком, только его хэш
        sig_parts.append("ocr:" + hashlib.sha256(ocr_hint.encode("utf-8")).hexdigest())
    multi_key = hashlib.sha256(("|".join(sig_parts)).encode("utf-8")).hexdigest()
    cache_key = ("values|" + multi_key, (lang or "ru").lower(), Cfg.vision_model() or Cfg.POLZA_CHAT)

    # 1) память процесса
    if cache_key in _VISION_CACHE:
        return _VISION_CACHE[cache_key]
    # 2) дисковый кэш
    disk = _vcache_get(cache_key)
    if disk:
        _VISION_CACHE[cache_key] = disk
        return disk

    # Инструкция (язык)
    lang_hint = {
        "ru": "Отвечай на русском.",
        "en": "Answer in English.",
        "kk": "Қазақ тілінде жауап бер.",
    }.get((lang or "ru").lower(), "Отвечай на русском.")

    # Контент: текст + изображения
    # Готовим контекст для шаблона: подпись + укороченный OCR-текст
    ctx_lines: List[str] = []
    if caption_hint:
        ctx_lines.append(f"Caption: {caption_hint}")
    if ocr_hint:
        cut = (ocr_hint[:2000] + "…") if len(ocr_hint or "") > 2000 else (ocr_hint or "")
        ctx_lines.append(f"OCR: {cut}")
    ctx_text = "\n".join(ctx_lines) if ctx_lines else None

    tmpl = build_extract_prompt(
        caption=caption_hint,
        context=ctx_text,
        lang=lang,
        max_items=Cfg.VISION_EXTRACT_MAX_ITEMS,
        conf_min=Cfg.VISION_EXTRACT_CONF_MIN,
        percent_decimals=Cfg.VISION_PERCENT_DECIMALS,
        require_json=Cfg.VISION_JSON_STRICT,
    )

    content_parts: List[Dict[str, Any]] = [{"type": "text", "text": tmpl["user"]}]
    for p in norm_paths:
        part = _image_part_for(p)
        if part:
            content_parts.append(part)

    raw = _vision_messages(
        content_parts,
        system_hint=tmpl["system"],
        temperature=temperature,
        max_tokens=max_tokens,
        want_tags=Cfg.VISION_JSON_STRICT,  # всегда требуем JSON-объект
    )

    # Парсинг JSON (устойчивый)
    def _parse_first_json(text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            s = str(text)
            start = s.find("{")
            end = s.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(s[start:end + 1])
                except Exception:
                    return None
        return None

    obj = _parse_first_json(raw)
    if not obj or not isinstance(obj, dict):
        res = {"type": "other", "warnings": ["не удалось распарсить JSON ответа модели."], "raw_text": [raw or ""]}
        _VISION_CACHE[cache_key] = res
        _vcache_put(cache_key, res)
        return res

    # ----------------- НОРМАЛИЗАЦИЯ И ГАРАНТИИ ПОЛЕЙ -----------------
    try:
        obj.setdefault("type", "other")
        obj.setdefault("units", {})
        obj.setdefault("axes", {})
        obj.setdefault("legend", [])
        obj.setdefault("data", [])
        obj.setdefault("raw_text", [])
        obj.setdefault("warnings", [])

        # Легенда — просто список строк
        if isinstance(obj.get("legend"), list):
            obj["legend"] = [str(x).strip() for x in obj["legend"] if str(x).strip()]

        # Нормализуем элементы данных: label/series/value/unit/conf
        norm_rows: List[Dict[str, Any]] = []
        rows = obj.get("data") if isinstance(obj.get("data"), list) else []
        for it in rows:
            if not isinstance(it, dict):
                continue
            label = str(it.get("label") or "").strip()
            series = str(it.get("series")).strip() if it.get("series") is not None else ""
            value_raw = it.get("value")
            unit_raw = it.get("unit")
            conf_raw = it.get("conf", it.get("confidence", it.get("score", None)))

            # value всегда строкой (анализатор сам парсит float/%, 0..1 и пр.)
            value = str(value_raw if value_raw is not None else "").strip()

            # unit: если пусто, но в value есть %/‰/п.п. — извлечём
            unit = str(unit_raw or "").strip()
            if not unit:
                if re.search(r"%\s*$", value):
                    unit = "%"
                elif re.search(r"‰\s*$", value):
                    unit = "‰"
                elif re.search(r"(?:п\.п\.|пп)\s*$", value, flags=re.IGNORECASE):
                    unit = "п.п."
            unit = _norm_unit(unit)

            # conf → float в [0,1]; если нет — 1.0
            try:
                conf = float(conf_raw) if conf_raw is not None else 1.0
            except Exception:
                conf = 1.0
            # клип
            if conf < 0.0:
                conf = 0.0
            if conf > 1.0:
                conf = 1.0

            norm_rows.append({
                "label": label,
                "series": series,
                "value": value,
                "unit": unit,
                "conf": conf,
            })
        obj["data"] = norm_rows

        # Очистим axes (если есть)
        axes = obj.get("axes", {})
        if isinstance(axes, dict):
            if "x_labels" in axes and isinstance(axes["x_labels"], list):
                axes["x_labels"] = [str(x).strip() for x in axes["x_labels"] if str(x).strip()]
            if "y_labels" in axes and isinstance(axes["y_labels"], list):
                axes["y_labels"] = [str(x).strip() for x in axes["y_labels"] if str(x).strip()]
            obj["axes"] = axes

        # Доп. поле units можно мягко нормализовать (если модель его заполнила)
        units = obj.get("units", {})
        if isinstance(units, dict):
            for k in list(units.keys()):
                units[k] = _norm_unit(str(units[k]))
            obj["units"] = units

        # Тип — в нижний регистр для предсказуемости
        t = str(obj.get("type") or "other").strip().lower()
        obj["type"] = t

    except Exception as e:
        logging.warning("vision_extract_values: normalization failed: %s", e)

    _VISION_CACHE[cache_key] = obj
    _vcache_put(cache_key, obj)
    return obj
