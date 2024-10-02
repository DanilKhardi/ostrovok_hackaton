#!/bin/bash

# Путь к входному CSV файлу
input_csv="$1"

python app.py --content "$input_csv"

# Сохранение результата в контейнере
result_file="output_predict.csv"
python -c "open('$result_file', 'w').close()"
