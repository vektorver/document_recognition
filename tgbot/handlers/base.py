from tgbot import dp, bot, dotenv_config
from utils.request import RequestToServer
from aiogram import types


@dp.message_handler(commands=['start'])
async def start(message: types.Message):
    await bot.send_message(message.chat.id, 'Привет, ты зарегистрирован! Для распознавания текста отправь мне фото с документом.')
    request = RequestToServer(dotenv_config['SERVER'])
    _, status_code = request.create_user(message.chat.id)
    if status_code == 422:
        await bot.send_message(message.chat.id, 'Рад видеть тебя снова!')

    answer, status_code = request.set_state(message.chat.id, 'registered')


@dp.message_handler(commands=['get_excel'])
async def get_excel(message: types.Message):
    request = RequestToServer(dotenv_config['SERVER'])
    answer, status_code = request.get_user(message.chat.id)

    if status_code == 404:
        await bot.send_message(message.chat.id, 'Пользователь незарегистрирован. Нажмите /start!')

    answer, status_code = request.get_excel(message.chat.id)
    if status_code == 200:
        await bot.send_message(message.chat.id, 'Файл с распознанными данными отправлен!')
        await bot.send_document(message.chat.id, open(answer['filename'], 'rb'))
    else:
        await bot.send_message(message.chat.id, 'Что-то пошло не так!')