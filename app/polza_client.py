import requests
from .config import Cfg

HDR = {"Authorization": f"Bearer {Cfg.POLZA_KEY}", "Content-Type": "application/json"}

def embeddings(texts:list[str]) -> list[list[float]]:
    r = requests.post(f"{Cfg.BASE_POLZA}/embeddings", headers=HDR,
                      json={"model": Cfg.POLZA_EMB, "input": texts}, timeout=60)
    r.raise_for_status()
    data = r.json()["data"]
    return [d["embedding"] for d in data]

def chat(messages:list[dict], temperature=0.2, max_tokens=800) -> str:
    r = requests.post(f"{Cfg.BASE_POLZA}/chat/completions", headers=HDR, timeout=60,
                      json={"model": Cfg.POLZA_CHAT, "messages": messages,
                            "temperature": temperature, "max_tokens": max_tokens})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]
