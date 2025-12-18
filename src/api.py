from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import pickle
import re
import os

app = FastAPI(title="AeroStream Sentiment API")

model = None
transformer = None

@app.on_event("startup")
async def load_model():
    global model, transformer
    print("Loading model...")
    
    if not os.path.exists("./models/best_model.pkl"):
        print("WARNING: Model not found! Train the model first using: python src/train_model.py")
        print("API will run but predictions will fail until model is trained.")
        return
    
    with open("./models/best_model.pkl", "rb") as f:
        model = pickle.load(f)
    transformer = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    print("Ready!")

class TextInput(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    sentiment: str
    confidence: float

def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower().strip()

@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: TextInput):
    if not model or not transformer:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    clean = clean_text(input_data.text)
    embedding = transformer.encode([clean])
    
    prediction = model.predict(embedding)[0]
    confidence = float(max(model.predict_proba(embedding)[0]))
    
    return PredictionResponse(sentiment=prediction, confidence=confidence)

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}
