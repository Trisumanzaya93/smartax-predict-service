from fastapi import FastAPI
from pathlib import Path
import joblib
from pydantic import BaseModel, Field
from typing import List
from src.preprocess import preprocess_text

class Item(BaseModel):
    text: str = Field(
        ...,
        min_length=5,
        max_length=500,
        example="tidak paham cara lapor spt"
    )

class BatchRequest(BaseModel):
  texts: List[str] = Field(
      ...,
      min_items=1,
      max_items=500,
      example=[
          "tidak paham cara lapor spt",
          "bingung pajak umkm"
      ]
)

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent.parent
vectorizer = joblib.load(BASE_DIR / "model" / "tfidf.pkl")
kmeans = joblib.load(BASE_DIR / "model" / "kmeans.pkl")

@app.post("/predict")
def predict(item: Item):
    clean = preprocess_text(item.text)
    print("CLEAN:", clean)
    X = vectorizer.transform([clean])
    cluster = int(kmeans.predict(X)[0])

    return {
        "raw_text": item.text,
        "clean_text": clean,
        "cluster": cluster
    }

@app.post("/predict/batch")
def predict_batch(payload: BatchRequest):
    clean_texts = [preprocess_text(t) for t in payload.texts]
    X = vectorizer.transform(clean_texts)
    clusters = kmeans.predict(X)

    return [
        {
            "raw_text": raw,
            "clean_text": clean,
            "cluster": int(cluster)
        }
        for raw, clean, cluster in zip(payload.texts, clean_texts, clusters)
    ]