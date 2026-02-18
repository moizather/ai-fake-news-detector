import pickle
import os

# Get absolute path to this file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "ml", "fake_news_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "ml", "tfidf_vectorizer.pkl")

# Load trained model and vectorizer
model = pickle.load(open(MODEL_PATH, "rb"))
vectorizer = pickle.load(open(VECTORIZER_PATH, "rb"))

def predict_text(text: str) -> str:
    if not text or not text.strip():
        return "Please send some text"

    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)

    if prediction[0] == 1:
        return "Fake News"
    else:
        return "Real News"
