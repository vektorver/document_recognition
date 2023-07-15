# document_recognition

# Приложение для распознавания документов

Для запуска неоьходимо скачать веса моделей: 
https://drive.google.com/file/d/10yAFdULEBsE7to1G3yeIZdIcOq3y9-AN/view?usp=sharing
https://drive.google.com/file/d/1-0wQ0I7FtuEu3ANMwOmezYhel7n3f4bR/view?usp=sharing

И расположить их по пути api/model/weights

В файле .env необходимо указать API_TOKEN для Telegram бота

## Запуск docker
```
docker compose up --build
```
Команда организует сборку и запуск контейнеров в одном приложении. Создается новая база данных, инициализируется сервис распознавания документов. Запускается бот в Telegram. 

## Документация сервиса 

Swagger по маршруту **/docs**

Redoc по маршруту **/redoc**