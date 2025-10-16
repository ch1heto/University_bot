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

def split_for_telegram(s: str, limit: int = 3900) -> Iterator[str]:
    """Режем длинные сообщения под ограничения Telegram."""
    s = s or ""
    for i in range(0, len(s), limit):
        yield s[i : i + limit]

# ------------------------- ФАЙЛЫ / ХЕШИ -------------------------

def sha256_bytes(data: bytes) -> str:
    """SHA-256 от байтов."""
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

def save_bytes(data: bytes, filename: str, dirpath: str = "./uploads") -> str:
    """Сохранить байты в файл (создаст папку при необходимости). Возвращает абсолютный путь."""
    d = ensure_dir(dirpath)
    fp = d / filename
    fp.write_bytes(data)
    return str(fp)

def file_ext_lower(name: str) -> str:
    """Расширение файла в нижнем регистре (с точкой), либо ''."""
    try:
        return (Path(name).suffix or "").lower()
    except Exception:
        return ""

# ------------------------- БЕЗОПАСНОЕ ИМЯ ФАЙЛА -------------------------

_SLUG_BAD = re.compile(r"[^0-9A-Za-zА-Яа-яЁё _.\-]+")
_MULTI_DOT = re.compile(r"\.{2,}")

def safe_filename(name: str, max_len: int = 180) -> str:
    """
    Делает «безопасное» имя файла:
    - нормализует Unicode (NFKC),
    - вычищает опасные символы, двойные точки,
    - ограничивает длину.
    """
    name = name or "file"
    name = unicodedata.normalize("NFKC", name)
    name = name.replace("\x00", "")
    name = _SLUG_BAD.sub("_", name)
    name = name.strip(" .\t\r\n")
    name = _MULTI_DOT.sub(".", name)
    if not name:
        name = "file"
    if len(name) > max_len:
        stem = Path(name).stem[: max_len - 16] or "file"
        ext = (Path(name).suffix or "")[:12]
        name = stem + ext
    return name

# ------------------------- JSON -------------------------

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

def infer_doc_kind(filename: str | None) -> str:
    """
    Грубая типизация по расширению:
    'thesis' для .doc/.docx/.pdf, иначе 'file'.
    """
    ext = file_ext_lower(filename or "")
    if ext in {".doc", ".docx", ".pdf"}:
        return "thesis"
    return "file"

# ------------------------- НОВАЯ ФУНКЦИЯ: РАЗБИТИЕ ДОКУМЕНТА НА ЧАСТИ -------------------------

def split_document(text: str, max_chunk_size: int = 3000) -> List[str]:
    """
    Разбивает текст на части, чтобы каждая часть не превышала заданный размер в символах.
    Это нужно для обработки больших документов, которые могут не поместиться в запрос к GPT-5.
    
    :param text: Текст документа, который нужно разделить.
    :param max_chunk_size: Максимальный размер части в символах.
    :return: Список частей текста, каждая из которых имеет размер не более max_chunk_size.
    """
    # Разбиваем текст на строки, чтобы сохранить структуру
    chunks = []
    current_chunk = ""
    
    # Разделяем текст на строки
    for line in text.splitlines():
        # Добавляем строку в текущий чанк, если не превышает максимальный размер
        if len(current_chunk) + len(line) + 1 <= max_chunk_size:
            current_chunk += (line + "\n")
        else:
            # Если превышает, сохраняем текущий чанк и начинаем новый
            chunks.append(current_chunk.strip())
            current_chunk = line + "\n"
    
    # Добавляем последний чанк, если он не пустой
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks
