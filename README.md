# AeroStream - Airline Sentiment Analysis

A simple sentiment analysis system for airline reviews using ChromaDB and machine learning.

## Project Structure

```
AeroStream-/
├── data/               # Data storage
├── notebooks/          # Jupyter notebooks
│   └── exploratory_analysis.ipynb
├── src/               # Source code
│   ├── data_processing.py
│   ├── train_model.py
│   └── api.py
├── models/            # Saved models
├── chroma_db/         # Vector database
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Process Data
```bash
python src/data_processing.py
```
Loads dataset from Hugging Face, cleans text, generates embeddings, and stores in ChromaDB.

### 3. Train Model
```bash
python src/train_model.py
```
Trains classifiers and saves the best model.

### 4. Run API
```bash
cd src
uvicorn api:app --reload
```
API available at `http://localhost:8000`

## Docker Deployment

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
