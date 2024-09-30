import streamlit as st
import numpy as np
import joblib

# Путь к файлам моделей и предобработчика
MODEL_1 = './models/xgb_classifier_y1.joblib'
MODEL_2 = './models/xgb_classifier_y2.joblib'
MODEL_3 = './models/xgb_classifier_y3_new.joblib'

VECTORIZER = './prep/count_vectorizer_X.joblib'

LABEL_ENCODER_1 = './prep/label_encoder_y1.joblib'
LABEL_ENCODER_2 = './prep/label_encoder_y2.joblib'
LABEL_ENCODER_3 = './prep/label_encoder_y3.joblib'

models = []
vectorizers = []
encoders = []


for model in (MODEL_1, MODEL_2, MODEL_3):
    with open(model, 'rb') as f:
        models.append(joblib.load(f))


with open(VECTORIZER, 'rb') as f:
        vectorizers.append(joblib.load(f))

for enc in (LABEL_ENCODER_1, LABEL_ENCODER_2, LABEL_ENCODER_3):
    with open(enc, 'rb') as f:
        encoders.append(joblib.load(f))


le_y1 = encoders[0]
le_y2 = encoders[1]
le_y3 = encoders[2]

vectorizer = vectorizers[0]

xgb_clf1 = models[0]
xgb_clf2 = models[1]
xgb_clf3 = models[2]

X_count = vectorizer.transform([input()])

y1_predict = xgb_clf1.predict(X_count)
y2_predict = xgb_clf2.predict(X_count)
y3_predict = xgb_clf3.predict(X_count)

y1_predict_prob = xgb_clf1.predict_proba(X_count)
y2_predict_prob = xgb_clf2.predict_proba(X_count)
y3_predict_prob = xgb_clf3.predict_proba(X_count)

y1_decode = le_y1.inverse_transform(y1_predict)
y2_decode = le_y2.inverse_transform(y2_predict)
y3_decode = le_y3.inverse_transform(y3_predict)

print(f"Предсказания: {y1_predict} {y2_predict} {y3_predict}")
print(f"Предсказания: {y1_decode} {y2_decode} {y3_decode}")
print(f"Вероятность: {y1_predict_prob} {y2_predict_prob} {y3_predict_prob}")



# # Основной интерфейс Streamlit
# st.title("Модель предсказания")

# # Текстовое поле для многоканального ввода
# text_input = st.text_area("Input Rate Name:")

# # Кнопка для запуска предсказания
# if st.button("Предсказать"):
#     # Преобразование текста при помощи предобработчика
#     X_count = vectorizer.transform([text_input])
    
#     # Предсказание
#     y1_predict = xgb_clf1.predict(X_count)
#     y2_predict = xgb_clf2.predict(X_count)
#     y3_predict = xgb_clf3.predict(X_count)
    
#     # Получение категорий
#     # categories = ["Категория 1", "Категория 2", "Категория 3"]  # Замените на ваши фактические категории
    
#     # Вывод результата
#     st.success(f"Предсказания: {y1_predict} {y2_predict} {y3_predict}")
    
#     # Вывод вероятностей
#     # probabilities = model.predict_proba(processed_text)[0]
#     # st.info(f"Вероятности: {dict(zip(categories, probabilities))}")
