from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import pickle, re, os

app = FastAPI(title="AeroStream API")
model, transformer = None, None

def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower().strip()

@app.on_event("startup")
async def startup():
    global model, transformer
    if os.path.exists("./models/best_model.pkl"):
        model = pickle.load(open("./models/best_model.pkl", "rb"))
        transformer = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        print("Model loaded!")
    else:
        print("No model found - run train_model.py first")

class TextInput(BaseModel):
    text: str

@app.post("/predict")
async def predict(data: TextInput):
    if not model:
        raise HTTPException(503, "Model not loaded")
    embedding = transformer.encode([clean_text(data.text)])
    return {
        "sentiment": model.predict(embedding)[0],
        "confidence": float(max(model.predict_proba(embedding)[0]))
    }

@app.get("/health")
async def health():
    return {"status": "ok", "model": model is not None}
