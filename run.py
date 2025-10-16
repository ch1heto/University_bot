# run.py
import asyncio
import logging
from app.bot import dp, bot   # у тебя уже есть bot.py с dp и bot

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

async def main():
    logging.info("Бот запускается…")
    await dp.start_polling(bot)
    logging.info("Бот остановлен.")

if __name__ == "__main__":
    asyncio.run(main())
