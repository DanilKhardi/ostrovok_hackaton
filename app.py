import joblib
import argparse
import pandas as pd
from pathlib import Path
import sys
import csv


# Путь к файлам моделей и предобработчика
MODEL_1 = './models/xgb_classifier_y1.joblib'
MODEL_2 = './models/xgb_classifier_y2.joblib'
MODEL_3 = './models/xgb_classifier_y3_new.joblib'
VECTORIZER = './prep/count_vectorizer_X.joblib'
LABEL_ENCODER_1 = './prep/label_encoder_y1.joblib'
LABEL_ENCODER_2 = './prep/label_encoder_y2.joblib'
LABEL_ENCODER_3 = './prep/label_encoder_y3.joblib'


def load_enocder():
    encoders = []
    for enc in (LABEL_ENCODER_1, LABEL_ENCODER_2, LABEL_ENCODER_3):
        with open(enc, 'rb') as f:
            encoders.append(joblib.load(f))
    return encoders


def load_vectorizer():
    vectorizers = []
    with open(VECTORIZER, 'rb') as f:
        vectorizers.append(joblib.load(f))
    return vectorizers


def load_models():
        models = []
        for model in (MODEL_1, MODEL_2, MODEL_3):
            with open(model, 'rb') as f:
                models.append(joblib.load(f))
        return models


def read_csv_custom(path_file):
    return pd.read_csv(path_file, on_bad_lines="warn").dropna().squeeze() # Возвращаем объект Series


def get_filename(path_file):
    return Path(path_file).name.split(".")[0] # Возвращаем имя файла


def save_csv(path_file, X, y1, y2, y3):
    df_result = pd.DataFrame()
    df_result["rate_name"] = X
    df_result["class"] = "undefined"
    df_result["quality"] = "undefined"
    df_result["bathroom"] = "undefined"
    df_result["bedding"] = y1
    df_result["capacity"] = y2
    df_result["club"] = "undefined"
    df_result["bedrooms"] = "undefined"
    df_result["balcony"] = "undefined"
    df_result["view"] = y3
    df_result["floor"] = "undefined"

    df_result.to_csv(f"./{get_filename(path_file)}_result.csv", index=None)

    # print(df_result.values.tolist())
    print(f"Все прошло отлично, результаты сохранены в файл {get_filename(path_file)}_result.csv")
    # write_row(f"{get_filename(path_file)}_result.csv") # Вывод результата в stdout


def write_row(path_file):
    result = csv.writer(sys.stdout, lineterminator='\n')
    with open(path_file) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            result.writerow(row)


def predict_category(X):
    le_y1, le_y2, le_y3 = load_enocder()
    vectorizer = load_vectorizer()[0]
    xgb_clf1, xgb_clf2, xgb_clf3 = load_models()

    X_count = vectorizer.transform(X)

    y1_predict = xgb_clf1.predict(X_count)
    y2_predict = xgb_clf2.predict(X_count)
    y3_predict = xgb_clf3.predict(X_count)

    y1_decode = le_y1.inverse_transform(y1_predict)
    y2_decode = le_y2.inverse_transform(y2_predict)
    y3_decode = le_y3.inverse_transform(y3_predict)

    return y1_decode, y2_decode, y3_decode



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--content', help='A path to rates CSV file')
    args = parser.parse_args()

    X = read_csv_custom(args.content)
    y1, y2, y3 = predict_category(X)
    save_csv(args.content, X, y1, y2, y3)

