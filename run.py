import asyncio
from app.bot import dp, bot

if __name__ == "__main__":
    asyncio.run(dp.start_polling(bot))
