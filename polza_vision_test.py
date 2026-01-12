# polza_vision_test.py
# Тест: реально ли Polza видит картинку, если отправить её как base64 data URL.

import os
import base64
import json
import requests

# 1. Ключ Polza: либо из переменной окружения, либо руками подставь сюда.
POLZA_API_KEY = os.getenv("POLZA_API_KEY")

# 2. Тот же endpoint, что в логах бота
BASE_URL = "https://api.polza.ai/api/v1/chat/completions"

# 3. Модель — подставь сюда ту же, что используешь в боте
POLZA_MODEL = "gpt-5"  # например, "gpt-4o-mini" или что у тебя стоит


def call_polza_vision_with_base64(image_path: str) -> None:
    print(f"Используем файл: {image_path}")
    if not os.path.isfile(image_path):
        print("❌ Файл не найден, проверь путь/имя файла.")
        return

    # читаем картинку и кодируем в base64
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    # формируем payload в формате мультимодального Chat Completions
    payload = {
        "model": POLZA_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Опиши это изображение максимально подробно на русском языке.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            # ВАЖНО: в API мы шлём НЕ путь V:\..., а data URL с base64
                            "url": f"data:image/png;base64,{img_b64}",
                        },
                    },
                ],
            }
        ],
        "max_tokens": 512,
    }

    headers = {
        "Authorization": f"Bearer {POLZA_API_KEY}",
        "Content-Type": "application/json",
    }

    resp = requests.post(BASE_URL, headers=headers, json=payload, timeout=120)
    print("HTTP status:", resp.status_code)
    try:
        data = resp.json()
        print(json.dumps(data, ensure_ascii=False, indent=2))
    except Exception:
        # если по какой-то причине не JSON — покажем как есть
        print(resp.text)


if __name__ == "__main__":
    # ТВОЙ ПУТЬ К ПАПКЕ С МЕДИА
    folder = r"V:\work\tg_bot_Universe\runtime\media\5e06ce89"

    # Имя файла — подставь реальное название, например:
    # img_aaa78ac6.png  или img_aaa78ac6.jpg  (посмотри расширение в проводнике)
    image_name = "img_aaa78ac6.png"

    image_path = os.path.join(folder, image_name)
    call_polza_vision_with_base64(image_path)
