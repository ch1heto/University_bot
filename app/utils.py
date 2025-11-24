# app/utils.py
from __future__ import annotations
import os
import re
import io
import json
import time
import hashlib
import unicodedata
from pathlib import Path
from typing import Iterable, Iterator, Any, Optional, Tuple, List, Dict

# Локальная конфигурация (для путей к кэшу/настройкам OCR)
try:
    from .config import Cfg
except Exception:  # на случай ранних импортов в тестах
    class _TmpCfg:
        CACHE_DIR = "./.cache"
        OCR_ENABLED = True
        OCR_LANGS = "rus+eng"
        OCR_ENGINE = "tesseract"
        OCR_TESSERACT_CMD = ""
        UPLOAD_DIR = "./uploads"
    Cfg = _TmpCfg()  # type: ignore


# ------------------------- ТЕКСТ / НОРМАЛИЗАЦИЯ -------------------------

_WS_RE = re.compile(r"\s+")
_NUM_RE = re.compile(r"\b\d+(?:[.,]\d+)?%?\b")

def norm_ws(s: str | None) -> str:
    """Нормализуем пробелы, убираем неразрывные."""
    if not s:
        return ""
    s = s.replace("\xa0", " ")
    return _WS_RE.sub(" ", s).strip()

def extract_numbers(s: str | None) -> List[str]:
    """Выделяет числа/проценты (как строки)."""
    return _NUM_RE.findall(s or "")

def clip(s: str, max_len: int = 200) -> str:
    """Обрезка строки с многоточием."""
    s = s or ""
    return s if len(s) <= max_len else (s[: max_len - 1] + "…")

def _smart_cut_point(text: str, start: int, limit: int) -> int:
    """
    Ищем «хорошую» границу разрыва (двойной перевод строки / перевод строки / точка / пробел)
    в окне [start, start+limit]. Возвращаем индекс конца чанка.
    """
    end = min(len(text), start + limit)
    window = text[start:end]
    # приоритетные места разрыва
    for pat in ("\n\n", "\n", ". ", " "):
        pos = window.rfind(pat)
        if pos != -1 and (start + pos) - start > int(limit * 0.6):  # не рвём слишком рано
            return start + pos + len(pat)
    return end

def split_for_telegram(s: str, limit: int = 3900) -> Iterator[str]:
    """
    Режем длинные сообщения под ограничения Telegram, стараясь рвать по «хорошим» границам.
    Важно: уже готовый текст должен быть безопасен для HTML (экранирование делается снаружи).
    """
    s = s or ""
    if len(s) <= limit:
        yield s
        return
    i = 0
    while i < len(s):
        j = _smart_cut_point(s, i, limit)
        chunk = s[i:j].rstrip()
        if chunk:
            yield chunk
        i = j


# ------------------------- ФАЙЛЫ / ХЕШИ -------------------------

def sha256_bytes(data: bytes) -> str:
    """SHA-256 от байтов (hex)."""
    h = hashlib.sha256()
    h.update(data or b"")
    return h.hexdigest()

def sha256_file(path: str | os.PathLike[str]) -> str:
    """SHA-256 от файла (стримовым чтением)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def ensure_dir(path: str | os.PathLike[str]) -> Path:
    """Создать директорию, если её нет. Возвращает Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_bytes(data: bytes, filename: str, dirpath: str = None) -> str:
    """
    Сохранить байты в файл (создаст папку при необходимости). Возвращает абсолютный путь.
    Имя проходит через safe_filename для страховки.
    """
    if dirpath is None:
        dirpath = getattr(Cfg, "UPLOAD_DIR", "./uploads")
    d = ensure_dir(dirpath)
    safe_name = safe_filename(filename)
    fp = d / safe_name
    fp.write_bytes(data or b"")
    return str(fp.resolve())

def file_ext_lower(name: str) -> str:
    """Расширение файла в нижнем регистре (с точкой), либо ''."""
    try:
        return (Path(name).suffix or "").lower()
    except Exception:
        return ""


# ------------------------- БЕЗОПАСНОЕ ИМЯ ФАЙЛА -------------------------

_SLUG_BAD = re.compile(r"[^0-9A-Za-zА-Яа-яЁё _.\-]+")
_MULTI_DOT = re.compile(r"\.{2,}")
_PATH_SEP = re.compile(r"[\\/]+")
_WINDOWS_RESERVED = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}

def safe_filename(name: str, max_len: int = 180) -> str:
    """
    Делает «безопасное» имя файла:
    - нормализует Unicode (NFKC),
    - вычищает опасные символы, слэши, двойные точки,
    - избегает зарезервированных имён Windows,
    - ограничивает длину (с сохранением расширения, если возможно).
    """
    name = name or "file"
    name = unicodedata.normalize("NFKC", name)
    name = name.replace("\x00", "")
    name = _PATH_SEP.sub("_", name)          # убираем / и \
    name = _SLUG_BAD.sub("_", name)          # убираем неожиданные символы
    name = _MULTI_DOT.sub(".", name)         # схлопываем цепочки точек
    name = name.strip(" .\t\r\n")
    if not name:
        name = "file"

    stem = Path(name).stem or "file"
    ext = (Path(name).suffix or "")
    # Windows reserved
    if stem.lower() in _WINDOWS_RESERVED:
        stem = f"_{stem}"

    # Ограничение длины
    if len(stem) + len(ext) > max_len:
        # оставляем расширение (до 12 символов)
        ext = ext[:12]
        room = max_len - len(ext)
        stem = stem[: max(1, room - 1)]
    out = (stem + ext) if (stem + ext) else "file"
    return out


# ------------------------- ЛЁГКИЙ КЭШ НА ДИСКЕ -------------------------

def _cache_ns_dir(namespace: str = "common") -> Path:
    """Путь к namespace в кэше."""
    base = getattr(Cfg, "CACHE_DIR", "./.cache")
    return ensure_dir(Path(base) / namespace)

def cache_key_for_bytes(data: bytes) -> str:
    """Удобный ключ кэша по SHA-256 (hex)."""
    return sha256_bytes(data)

def cache_path(key: str, *, namespace: str = "common", ext: str = "bin") -> Path:
    """Полный путь кэша по ключу и расширению."""
    key = re.sub(r"[^0-9a-fA-F]", "", key)[:64] or "key"
    ext = ext.lstrip(".")
    return _cache_ns_dir(namespace) / f"{key}.{ext}"

def cache_put_bytes(data: bytes, *, namespace: str = "common", ext: str = "bin", key: str | None = None) -> Tuple[str, str]:
    """
    Кладём байты в кэш. Возвращает (absolute_path, key).
    Если key не задан — используем sha256(data).
    """
    k = key or cache_key_for_bytes(data or b"")
    fp = cache_path(k, namespace=namespace, ext=ext)
    if not fp.exists():  # не перезаписываем
        fp.write_bytes(data or b"")
    return str(fp.resolve()), k

def cache_get_bytes(key: str, *, namespace: str = "common", ext: str = "bin") -> Optional[bytes]:
    """Читает байты из кэша по ключу/расширению."""
    fp = cache_path(key, namespace=namespace, ext=ext)
    if fp.exists():
        try:
            return fp.read_bytes()
        except Exception:
            return None
    return None

def cache_put_json(obj: Any, *, namespace: str = "common", key: str, pretty: bool = False) -> str:
    """Кладёт JSON в кэш. Возвращает абсолютный путь файла."""
    fp = cache_path(key, namespace=namespace, ext="json")
    data = json.dumps(obj, ensure_ascii=False, indent=2 if pretty else None, default=str)
    fp.write_text(data, encoding="utf-8")
    return str(fp.resolve())

def cache_get_json(key: str, *, namespace: str = "common", default: Any = None) -> Any:
    """Читает JSON из кэша, либо default."""
    fp = cache_path(key, namespace=namespace, ext="json")
    if not fp.exists():
        return default
    try:
        return json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        return default




# ------------------------- JSON УТИЛИТЫ -------------------------

def to_json(obj: Any, *, pretty: bool = False) -> str:
    """Безопасный json.dumps с ensure_ascii=False."""
    if pretty:
        return json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)

def from_json(s: str | None, default: Any = None) -> Any:
    """json.loads с дефолтом при ошибке."""
    if not s:
        return default
    try:
        return json.loads(s)
    except Exception:
        return default


# ------------------------- ТАЙМИНГ / ЛОГГИНГ -------------------------

class Stopwatch:
    """Контекст-менеджер для измерения времени."""
    def __init__(self, tag: str = "") -> None:
        self.tag = tag
        self.start = 0.0
        self.elapsed = 0.0

    def __enter__(self) -> "Stopwatch":
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.elapsed = time.perf_counter() - self.start

def time_call(fn):
    """Декоратор: печатает время выполнения функции (для отладки)."""
    def _wrap(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            dt = (time.perf_counter() - t0) * 1000.0
            print(f"[time_call] {fn.__name__}: {dt:.1f} ms")
    return _wrap


# ------------------------- ПРОЧЕЕ -------------------------

def coalesce(*vals, default: Any = None) -> Any:
    """Первое «правдивое» значение, иначе default."""
    for v in vals:
        if v:
            return v
    return default

def file_is_probably_pdf(path: str | os.PathLike[str]) -> bool:
    """Быстрая эвристика по сигнатуре PDF."""
    try:
        with open(path, "rb") as f:
            head = f.read(5)
        return head == b"%PDF-"
    except Exception:
        return False

def infer_doc_kind(filename: str | None) -> str:
    """
    Грубая типизация по расширению:
    'thesis' для .doc/.docx/.pdf, иначе 'file'.
    """
    ext = file_ext_lower(filename or "")
    if ext in {".doc", ".docx", ".pdf"}:
        return "thesis"
    return "file"


# ------------------------- РАЗБИТИЕ ДЛЯ ДОЛГИХ ТЕКСТОВ -------------------------

def split_document(text: str, max_chunk_size: int = 3000) -> List[str]:
    """
    Разбивает текст на части, чтобы каждая часть не превышала max_chunk_size символов.
    Использует ту же логику, что и split_for_telegram, чтобы нарезка была одинаковой
    и для Telegram, и для долгих текстов.
    """
    text = text or ""
    if not text:
        return []
    if len(text) <= max_chunk_size:
        return [text]
    # переиспользуем уже настроенный алгоритм «умной» нарезки
    return list(split_for_telegram(text, limit=max_chunk_size))


# ------------------------- УТИЛИТЫ ДЛЯ «Структурированного парсинга + кеша» -------------------------

def cache_store_bytes_by_hash(data: bytes, *, namespace: str = "blobs", ext: str = "bin") -> Tuple[str, str]:
    """
    Сохраняет байты в кэш по их SHA-256 (без перезаписи).
    Возвращает (absolute_path, key).
    Удобно для картинок, извлечённых из DOCX/PDF.
    """
    key = cache_key_for_bytes(data or b"")
    path, _ = cache_put_bytes(data or b"", namespace=namespace, ext=ext, key=key)
    return path, key

def cache_store_json_by_key(key: str, obj: Any, *, namespace: str = "indexes") -> str:
    """
    Сохраняет JSON (например, структурированный индекс документа) под заданным ключом.
    Возвращает абсолютный путь к файлу.
    """
    return cache_put_json(obj, namespace=namespace, key=key, pretty=False)

def cache_load_json_by_key(key: str, *, namespace: str = "indexes", default: Any = None) -> Any:
    """Загружает JSON по ключу/namespace (если нет — default)."""
    return cache_get_json(key, namespace=namespace, default=default)
