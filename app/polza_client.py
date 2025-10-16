# app/polza_client.py
from typing import List, Dict, Any
import logging
from openai import OpenAI

from .config import Cfg  # берём ключ, базовый URL и имена моделей из одного места


# Один общий клиент Polza/OpenAI
_client = OpenAI(
    base_url=Cfg.BASE_POLZA,   # например: "https://api.polza.ai/api/v1"
    api_key=Cfg.POLZA_KEY,     # .env: POLZA_API_KEY=...
)

__all__ = ["embeddings", "chat_with_gpt", "probe_embedding_dim"]


def embeddings(texts: List[str]) -> List[List[float]]:
    """
    Получить эмбеддинги для списка текстов.
    Возвращает список векторов float[]. Пустой ввод -> пустой список.
    """
    if not texts:
        return []
    try:
        resp = _client.embeddings.create(
            model=Cfg.POLZA_EMB,   # например: "openai/text-embedding-3-large"
            input=texts,
        )
        # Новый SDK возвращает объекты, а не словари
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
    """
    try:
        cmpl = _client.chat.completions.create(
            model=Cfg.POLZA_CHAT,  # например: "openai/gpt-4o"
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = cmpl.choices[0].message.content or ""
        return content.strip()
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
