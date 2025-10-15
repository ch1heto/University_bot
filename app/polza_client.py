import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import List, Dict
from .config import Cfg

# ----------------- HTTP сессия с ретраями -----------------

HDR = {
    "Authorization": f"Bearer {Cfg.POLZA_KEY}",
    "Content-Type": "application/json",
}

# Таймауты: (connect, read)
TIMEOUT = (15, 120)

_session = requests.Session()
_retries = Retry(
    total=3,            # всего попыток
    connect=3,
    read=3,
    backoff_factor=1.5, # 0s, 1.5s, 3s, 4.5s ...
    status_forcelist=[429, 502, 503, 504],
    allowed_methods=frozenset({"GET", "POST"}),
)
_adapter = HTTPAdapter(max_retries=_retries)
_session.mount("https://", _adapter)
_session.mount("http://", _adapter)

def _post(path: str, *, json: dict) -> requests.Response:
    url = f"{Cfg.BASE_POLZA}{path}"
    return _session.post(url, headers=HDR, json=json, timeout=TIMEOUT)

# ----------------- Публичные функции -----------------

def embeddings(texts: List[str]) -> List[List[float]]:
    """
    Возвращает список эмбеддингов для переданных текстов.
    В случае сетевой/HTTP-ошибки возбуждает исключение — вызывающая сторона
    (retrieval._embed_query) уже оборачивает вызов в try/except.
    """
    r = _post(
        "/embeddings",
        json={"model": Cfg.POLZA_EMB, "input": texts},
    )
    r.raise_for_status()
    data = r.json().get("data", [])
    return [item["embedding"] for item in data]

def chat(messages: List[Dict], temperature: float = 0.2, max_tokens: int = 800) -> str:
    """
    Возвращает ответ модели как строку.
    При сетевой/HTTP-ошибке НЕ падает — выдаёт вежливый фолбэк,
    чтобы бот продолжал работу.
    """
    try:
        r = _post(
            "/chat/completions",
            json={
                "model": Cfg.POLZA_CHAT,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException:
        return ("Сейчас нет связи с модельным сервером. "
                "Повторите запрос чуть позже.")
