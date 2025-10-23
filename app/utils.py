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
    """
    Сохранить байты в файл (создаст папку при необходимости). Возвращает абсолютный путь.
    Имя проходит через safe_filename для страховки.
    """
    d = ensure_dir(dirpath)
    safe_name = safe_filename(filename)
    fp = d / safe_name
    fp.write_bytes(data)
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
    Ставит разрывы по «хорошим» границам (параграф/строка/слово), когда возможно.
    """
    text = text or ""
    if len(text) <= max_chunk_size:
        return [text] if text else []

    parts: List[str] = []
    i = 0
    while i < len(text):
        j = _smart_cut_point(text, i, max_chunk_size)
        chunk = text[i:j].strip()
        if chunk:
            parts.append(chunk)
        i = j
    return parts
