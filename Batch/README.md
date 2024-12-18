# AeroStream Batch

Sentiment analysis for airline reviews.

## Setup
```bash
pip install -r requirements.txt
python src/data_processing.py
python src/train_model.py
```

## Run API
```bash
uvicorn src.api:app --reload
```

## Docker
```bash
docker-compose up --build
```

```bash
docker-compose up --build
```

## API Endpoints

- `POST /predict` - Predict sentiment
  ```json
  {
    "text": "Great flight experience!"
  }
  ```

- `GET /health` - Health check

## Features

- ✅ Data loading from Hugging Face
- ✅ Text cleaning and normalization
- ✅ Multilingual embeddings
- ✅ ChromaDB vector storage
- ✅ Multiple ML models
- ✅ REST API
- ✅ Docker support
