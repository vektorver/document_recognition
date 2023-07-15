from tgbot import dp, bot, dotenv_config
from utils.request import RequestToServer
from aiogram import types
import random
import string

def random_string(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))


# if photo was sent
@dp.message_handler(content_types=['photo'])
async def recognize(message: types.Message):
    await bot.send_message(message.chat.id, 'Фото получено...')
    request = RequestToServer(dotenv_config['SERVER'])
    answer, status_code = request.get_user(message.chat.id)
    if status_code == 200:
        if answer['state'] == 'registered':
            file_info = await bot.get_file(message.photo[-1].file_id)
            downloaded_file = await bot.download_file(file_info.file_path)
            filename = f'/uploads/{random_string(10)}.png'
            with open(filename, 'wb') as new_file:
                new_file.write(downloaded_file.read())
            await bot.send_message(message.chat.id, 'Распознаю текст!')

            answer, status_code = request.predict(message.chat.id, filename)
            if status_code == 200:
                # create string with key: value on every string exclude id and tg_chat_id
                # rename keys to russian
                keys_eng = ['name', 'surname', 'patronymic', 'birth_date', 'birth_place', 'passport_series', 'passport_number', 'passport_date', 'passport_code', 'passport_issued_by', 'passport_issued_by_code']
                keys_rus = ['Имя', 'Фамилия', 'Отчество', 'Дата рождения', 'Место рождения', 'Серия паспорта', 'Номер паспорта', 'Дата выдачи паспорта', 'Код подразделения', 'Кем выдан паспорт', 'Код подразделения']
                answer = {keys_rus[keys_eng.index(key)]: value for key, value in answer.items() if key not in ['id', 'tg_chat_id']}
                answer = '\n'.join([f'{key}: {value}' for key, value in answer.items()])
                # await bot.send_message(message.chat.id, str(answer))
                # send photo /uploads/result.png

                with open('/uploads/result.png', 'rb') as photo:
                    await bot.send_photo(message.chat.id, photo, caption=answer)

            else:
                await bot.send_message(message.chat.id, 'Ошибка распознавания.')
