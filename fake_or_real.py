from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = FastAPI()

# Загрузка модели и векторизатора
with open('src/model_lr.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('src/vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

# Модель данных
class NewsItem(BaseModel):
    title: str
    text: str

# Пример обработки данных
@app.post("/predict/")
def predict(news: NewsItem):
    # Преобразование текста в TF-IDF вектор
    text_vector = vectorizer.transform([news.text])
    
    # Получение предсказания от модели
    prediction = model.predict(text_vector)
    
    # Интерпретация предсказания (0 или 1)
    label = "FAKE" if prediction[0] == 0 else "REAL"
    
    return {"title": news.title, "prediction": label}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Fake News Detection API"}
