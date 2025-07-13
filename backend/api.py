from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import html
import re
import os
import string
from nltk.corpus import stopwords
from fastapi.middleware.cors import CORSMiddleware


# Load saved model and vectorizer
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "model", "spam_detector_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "model", "tfidf_vectorizer.pkl")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

#model = joblib.load("../model/spam_detector_model.pkl")
#vectorizer = joblib.load("../model/tfidf_vectorizer.pkl")

stop_words = set(stopwords.words("english"))

app = FastAPI(title="SMS Spam Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify ["http://127.0.0.1:5500"] if you want stricter
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    text: str

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

@app.post("/predict")
def predict(message: Message):
    cleaned = clean_text(message.text)
    vect = vectorizer.transform([cleaned])
    proba = model.predict_proba(vect)[0]
    spam_prob = proba[1]
    pred = int(spam_prob >= 0.4)
    result = "spam" if pred == 1 else "ham"
    return {
        "prediction": result,
        "spam_probability": round(spam_prob, 4)  # Return e.g., 0.8732
    }

