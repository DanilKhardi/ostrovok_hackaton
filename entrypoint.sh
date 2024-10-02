#!/bin/bash

# Путь к входному CSV файлу
input_csv="$1"

# Запуск Python скрипта
python app.py --content "$input_csv"

# Сохранение результата в контейнере
result_file="predict.csv"
python -c "open('$result_file', 'w').close()"

# Сохранение результата на локальной машине
docker cp $(docker ps -q):$result_file ./local_predict.csv

