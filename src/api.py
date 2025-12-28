from fastapi import FastAPI
import joblib
from pydantic import BaseModel
from preprocess import preprocess_text

class Item(BaseModel):
    text: str

app = FastAPI()

vectorizer = joblib.load("../model/tfidf.pkl")
kmeans = joblib.load("../model/kmeans.pkl")

@app.post("/predict")
def predict(item: Item):
    clean = preprocess_text(item.text)
    print("CLEAN:", clean)
    X = vectorizer.transform([clean])
    cluster = int(kmeans.predict(X)[0])

    return {
        "cluster": cluster
    }
