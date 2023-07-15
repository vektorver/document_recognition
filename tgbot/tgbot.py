from aiogram import Bot, Dispatcher, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram import Bot, Dispatcher, executor, types
from dotenv import dotenv_values
import asyncio

dotenv_config = dotenv_values(".env")
loop = asyncio.get_event_loop()
bot = Bot(token=dotenv_config['API_TOKEN'], loop=loop)
dp = Dispatcher(bot, storage=MemoryStorage())
storage = MemoryStorage()

from handlers.base import *
from handlers.predict import *


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)


