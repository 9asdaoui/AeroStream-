# AeroStream - Complete Documentation

## 📚 Table of Contents
1. [Project Overview](#project-overview)
2. [Core Concepts](#core-concepts)
3. [Architecture & Data Flow](#architecture--data-flow)
4. [Technologies Explained](#technologies-explained)
5. [Step-by-Step Process](#step-by-step-process)
6. [Use Cases & Examples](#use-cases--examples)

---

## 🎯 Project Overview

**What does this project do?**

AeroStream analyzes airline customer reviews and automatically determines if the sentiment is:
- **Positive** 😊 (customer is happy)
- **Negative** 😞 (customer is unhappy)
- **Neutral** 😐 (customer is neither happy nor unhappy)

**Real-world example:**
```
Input: "The flight was delayed for 3 hours and no one helped us!"
Output: Negative sentiment (Confidence: 92%)

Input: "Amazing service! The crew was friendly and helpful."
Output: Positive sentiment (Confidence: 95%)
```

---

## 🧠 Core Concepts

### 1. Sentiment Analysis

**What is it?**
Sentiment analysis is teaching a computer to understand emotions in text, just like humans do.

**Simple analogy:**
Imagine you're reading restaurant reviews. You can quickly tell if someone liked or disliked the food by reading their words. Sentiment analysis teaches computers to do the same thing.

**How it works:**
```
Text → Analysis → Emotion Label
"Great flight!" → [Process] → Positive
"Terrible service!" → [Process] → Negative
```

---

### 2. Text Embeddings

**What are embeddings?**
Embeddings convert words into numbers (vectors) that capture their meaning. Similar words get similar numbers.

**Simple analogy:**
Think of embeddings like GPS coordinates for words:
- "happy" might be at coordinates [0.8, 0.9, 0.2]
- "joyful" might be at coordinates [0.81, 0.88, 0.19] (very close!)
- "sad" might be at coordinates [-0.7, -0.8, 0.1] (far away)

**Why do we need them?**
Computers can't understand words like "good" or "bad" directly. They only understand numbers. Embeddings translate language into math.

**Example transformation:**
```python
Text: "Great flight experience"
↓
Embedding: [0.234, -0.567, 0.891, ..., 0.123]  # 384 numbers
          ↑
    This vector captures the meaning
```

**Key properties:**
- Similar meanings → Similar vectors
- "excellent" and "great" have vectors close to each other
- "terrible" and "awful" have vectors close to each other
- "excellent" and "terrible" have vectors far apart

---

### 3. Sentence Transformers

**What is it?**
Sentence Transformers is a special model that converts entire sentences into embeddings (not just individual words).

**Model we use:**
`paraphrase-multilingual-MiniLM-L12-v2`

Let's break down the name:
- **paraphrase**: Good at understanding when sentences mean the same thing
- **multilingual**: Works with multiple languages (English, French, Spanish, etc.)
- **MiniLM**: A smaller, faster version of a larger model
- **L12**: Has 12 layers of neural network processing
- **v2**: Version 2 (improved version)

**Example:**
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Convert text to numbers
embedding = model.encode("The flight was amazing!")
# Result: array of 384 numbers representing the meaning
```

**Why this model?**
- Fast and lightweight
- Maintains good accuracy
- Understands context (not just individual words)

---

### 4. ChromaDB (Vector Database)

**What is it?**
ChromaDB is a specialized database for storing and searching embeddings (vectors).

**Simple analogy:**
Think of it like a library, but instead of organizing books by title or author, it organizes them by their content and meaning. If you ask for "books about adventure," it finds similar books even if they don't have "adventure" in the title.

**Why not use a regular database?**

| Regular Database (MySQL) | Vector Database (ChromaDB) |
|--------------------------|----------------------------|
| Finds exact matches | Finds similar meanings |
| "flight delay" ≠ "plane late" | "flight delay" ≈ "plane late" |
| Slower for similarity searches | Optimized for similarity |
| Stores text as-is | Stores numerical vectors |

**What we store in ChromaDB:**

```python
For each review:
{
    "id": "1",
    "embedding": [0.234, -0.567, ...],  # 384 numbers
    "document": "Great flight experience",  # Original text
    "metadata": {"label": "positive"}  # Sentiment label
}
```

**Why use it?**
1. **Efficient retrieval**: Get all embeddings quickly for training
2. **Organized storage**: Separate collections for train/test data
3. **Scalability**: Can handle millions of embeddings
4. **Persistence**: Data saved to disk, not lost when program stops

**Our collections:**
```
ChromaDB
├── train_collection (training data embeddings)
└── test_collection (test data embeddings)
```

---

### 5. Machine Learning Models

We train two different models and pick the best one:

#### A. Logistic Regression

**What is it?**
A simple, fast algorithm that draws a line (or boundary) between positive and negative sentiments.

**Simple analogy:**
Imagine plotting reviews on a graph:
- Positive reviews on one side
- Negative reviews on the other side
- Logistic regression draws a line to separate them

**Example:**
```
Positive reviews ✓✓✓
                    |  ← This line separates them
Negative reviews ✗✗✗
```

**Pros:**
- Very fast
- Easy to understand
- Works well for simple problems

**Cons:**
- Might miss complex patterns

#### B. Random Forest

**What is it?**
An ensemble of many decision trees working together.

**Simple analogy:**
Instead of asking one expert, you ask 100 experts and take a vote. Each "tree" makes a prediction, and the majority wins.

**Example:**
```
Review: "Flight was okay but food was bad"

Tree 1: Negative
Tree 2: Neutral
Tree 3: Negative
...
Tree 100: Negative

Final prediction: Negative (majority vote)
```

**Pros:**
- More accurate for complex patterns
- Handles non-linear relationships
- Robust to noise

**Cons:**
- Slower than Logistic Regression
- Requires more memory

---

## 🏗️ Architecture & Data Flow

### Complete Pipeline

```
1. DATA LOADING
   ↓
   Hugging Face Dataset
   "7Xan7der7/usairlinesentiment"
   
2. DATA CLEANING
   ↓
   • Remove URLs, mentions
   • Remove special characters
   • Convert to lowercase
   • Remove duplicates
   • Handle missing values
   
3. EMBEDDING GENERATION
   ↓
   Sentence Transformers Model
   Text → 384-dimensional vectors
   
4. STORAGE
   ↓
   ChromaDB Collections
   • train_collection
   • test_collection
   
5. MODEL TRAINING
   ↓
   Retrieve embeddings from ChromaDB
   Train multiple models
   
6. EVALUATION
   ↓
   Test on unseen data
   Pick best model
   
7. DEPLOYMENT
   ↓
   FastAPI REST API
   Docker containers
```

### Relationship Between Components

```
┌─────────────────────────────────────────────────────┐
│                  USER REQUEST                        │
│           "Analyze: Great flight!"                   │
└─────────────────────┬───────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│                   FastAPI                            │
│              (REST API Layer)                        │
└─────────────────────┬───────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│              Text Cleaning                           │
│   Remove noise, lowercase, normalize                │
└─────────────────────┬───────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│          Sentence Transformers                       │
│     Convert text → embedding vector                  │
└─────────────────────┬───────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│            Trained ML Model                          │
│    (Logistic Regression or Random Forest)           │
│         Predict: Positive/Negative/Neutral           │
└─────────────────────┬───────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│                   RESPONSE                           │
│   {"sentiment": "positive", "confidence": 0.95}      │
└─────────────────────────────────────────────────────┘
```

---

## 🔧 Technologies Explained

### 1. Python Libraries

#### datasets (Hugging Face)
```python
from datasets import load_dataset

# Load pre-prepared dataset from Hugging Face
dataset = load_dataset("7Xan7der7/usairlinesentiment")
```

**Purpose:** Easy access to thousands of datasets without manual downloading
**Why use it:** Dataset is already cleaned, split, and ready to use

#### pandas
```python
import pandas as pd

# Work with data in table format
df = pd.DataFrame(dataset['train'])
df.head()  # View first 5 rows
```

**Purpose:** Data manipulation and analysis
**Why use it:** Makes it easy to filter, clean, and explore data

#### scikit-learn
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
accuracy = accuracy_score(y_test, predictions)
```

**Purpose:** Machine learning algorithms and evaluation tools
**Why use it:** Industry-standard, reliable, well-documented

### 2. Text Processing

**Regular Expressions (regex)**
```python
import re

# Remove URLs
text = re.sub(r'http\S+', '', text)

# Remove mentions like @username
text = re.sub(r'@\w+', '', text)

# Remove special characters
text = re.sub(r'[^\w\s]', '', text)
```

**Purpose:** Pattern matching and text cleaning
**Example transformation:**
```
Before: "@JohnDoe Flight delayed!! https://bit.ly/123 😡"
After: "flight delayed"
```

### 3. FastAPI

**What is it?**
A modern Python framework for building APIs (Application Programming Interfaces).

**Simple analogy:**
FastAPI is like a waiter in a restaurant:
- You (client) make a request: "I want to know the sentiment of this text"
- FastAPI (waiter) takes your request to the kitchen (ML model)
- The kitchen processes it
- FastAPI brings back the result

**Example:**
```python
from fastapi import FastAPI

app = FastAPI()

@app.post("/predict")
async def predict(text: str):
    # Process text
    # Get prediction
    return {"sentiment": "positive", "confidence": 0.95}
```

**Why FastAPI?**
- Automatic API documentation
- Fast performance
- Type validation built-in
- Async support (can handle multiple requests simultaneously)

### 4. Docker & Docker Compose

#### Docker

**What is it?**
Docker packages your application and all its dependencies into a container.

**Simple analogy:**
Think of Docker like a shipping container:
- You pack everything needed (code, libraries, settings)
- It works the same way on any ship (computer)
- No "it works on my machine" problems

**Dockerfile explained:**
```dockerfile
FROM python:3.11-slim           # Start with Python installed
WORKDIR /app                    # Create working directory
COPY requirements.txt .         # Copy dependencies list
RUN pip install -r requirements.txt  # Install dependencies
COPY . .                        # Copy all code
EXPOSE 8000                     # Open port 8000
CMD ["uvicorn", "src.api:app"]  # Start the application
```

#### Docker Compose

**What is it?**
Orchestrates multiple Docker containers that work together.

**Our setup:**
```yaml
services:
  aerostream-api:       # Service 1: API
    ports: 8000
    
  jupyter:              # Service 2: Jupyter notebooks
    ports: 8888
```

**Why use it?**
- Start everything with one command: `docker-compose up`
- Containers can communicate with each other
- Easy to manage multiple services

---

## 📝 Step-by-Step Process

### Step 1: Data Loading & Cleaning

**File:** `src/data_processing.py`

```python
# Load data
dataset = load_dataset("7Xan7der7/usairlinesentiment")
```

**What happens:**
1. Downloads dataset from Hugging Face
2. Splits into train and test sets
3. Converts to pandas DataFrame for easy manipulation

**Cleaning steps:**

```python
def clean_text(text):
    # 1. Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # 2. Remove mentions (@username)
    text = re.sub(r'@\w+', '', text)
    
    # 3. Remove punctuation and special chars
    text = re.sub(r'[^\w\s]', '', text)
    
    # 4. Lowercase
    return text.lower().strip()
```

**Example transformation:**
```
BEFORE: "@UnitedAir Flight UA123 DELAYED!! 😡 http://bit.ly/complaint"
AFTER:  "flight ua123 delayed"
```

**Why each step?**

| Step | Why? |
|------|------|
| Remove URLs | URLs don't contain sentiment, just noise |
| Remove mentions | @username doesn't help predict sentiment |
| Remove punctuation | "good!" and "good" should be treated the same |
| Lowercase | "GOOD", "Good", "good" should all be the same |

**Remove duplicates:**
```python
df = df.drop_duplicates(subset=['text'])
```
**Why?** Duplicate reviews can bias the model

**Handle missing values:**
```python
df = df.dropna(subset=['text', 'airline_sentiment'])
```
**Why?** Can't train on incomplete data

### Step 2: Generate Embeddings

```python
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
embeddings = model.encode(texts, show_progress_bar=True)
```

**What happens:**
1. Model loads (downloads if first time)
2. Each text is converted to 384 numbers
3. Progress bar shows completion

**Example:**
```python
Input: "Great service!"
Output: [0.234, -0.567, 0.891, ..., 0.123]  # 384 numbers
```

**Why 384 dimensions?**
This is the model's output size. Think of it as 384 different features describing the text:
- Feature 1 might capture "positivity"
- Feature 2 might capture "formality"
- Feature 3 might capture "urgency"
- etc.

### Step 3: Store in ChromaDB

```python
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.create_collection(name="train_collection")

collection.add(
    embeddings=embeddings.tolist(),
    documents=texts,
    metadatas=[{"label": label} for label in labels],
    ids=[str(i) for i in range(len(texts))]
)
```

**What happens:**
1. Create/connect to ChromaDB
2. Create a collection (like a table)
3. Add embeddings with metadata

**Storage structure:**
```
chroma_db/
├── train_collection/
│   ├── embedding_0: [0.234, -0.567, ...]  → "Great flight!" → positive
│   ├── embedding_1: [-0.789, 0.123, ...]  → "Terrible service" → negative
│   └── ...
└── test_collection/
    └── ...
```

**Why separate train/test collections?**
- **train_collection**: Used to train the model
- **test_collection**: Used to evaluate (model has never seen this data)

### Step 4: Train Models

**File:** `src/train_model.py`

```python
# Load embeddings from ChromaDB
X_train, y_train = load_embeddings_from_chromadb("train_collection")
X_test, y_test = load_embeddings_from_chromadb("test_collection")

# Train multiple models
models = {
    'logistic_regression': LogisticRegression(),
    'random_forest': RandomForestClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
```

**What happens:**

1. **Load data:**
   ```
   X_train: [[0.234, ...], [0.567, ...], ...]  # Embeddings
   y_train: ['positive', 'negative', ...]       # Labels
   ```

2. **Train model:**
   - Model learns patterns from embeddings
   - Associates certain vector patterns with positive/negative

3. **Example learning:**
   ```
   Model learns:
   - Vectors around [0.8, 0.9, ...] → Usually positive
   - Vectors around [-0.7, -0.8, ...] → Usually negative
   ```

### Step 5: Evaluation

```python
# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Detailed report
print(classification_report(y_test, y_pred))
```

**Output example:**
```
              precision    recall  f1-score   support

    positive       0.89      0.92      0.90      1000
    negative       0.85      0.88      0.86       800
     neutral       0.72      0.65      0.68       400

    accuracy                           0.85      2200
```

**What these metrics mean:**

- **Precision**: Of all reviews we labeled as positive, how many were actually positive?
  - Example: If we predict 100 as positive, but only 89 are truly positive → 89% precision

- **Recall**: Of all actual positive reviews, how many did we find?
  - Example: There are 100 positive reviews, we found 92 of them → 92% recall

- **F1-score**: Harmonic mean of precision and recall (balanced metric)

- **Support**: Number of actual occurrences in the dataset

**Pick best model:**
```python
if accuracy_logistic > accuracy_rf:
    best_model = logistic_regression
else:
    best_model = random_forest
```

### Step 6: Save Model

```python
import pickle

with open("./models/best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
```

**What is pickle?**
Serializes (saves) Python objects to disk so you can load them later.

**Why save?**
- Don't need to retrain every time
- Can deploy the same model to production
- Faster startup (just load, don't train)

### Step 7: Deploy API

**File:** `src/api.py`

```python
@app.post("/predict")
async def predict(input_data: TextInput):
    # 1. Clean text
    clean = clean_text(input_data.text)
    
    # 2. Generate embedding
    embedding = transformer.encode([clean])
    
    # 3. Predict
    prediction = model.predict(embedding)[0]
    confidence = max(model.predict_proba(embedding)[0])
    
    return {"sentiment": prediction, "confidence": confidence}
```

**Request flow:**
```
1. User sends: {"text": "Great flight!"}
   ↓
2. API cleans: "great flight"
   ↓
3. Create embedding: [0.234, -0.567, ...]
   ↓
4. Model predicts: "positive" (95% confidence)
   ↓
5. API returns: {"sentiment": "positive", "confidence": 0.95}
```

---

## 💡 Use Cases & Examples

### Use Case 1: Customer Service Monitoring

**Scenario:** Airline wants to monitor customer satisfaction in real-time

**How AeroStream helps:**
```python
# Process incoming reviews
review = "The staff was rude and unhelpful"
response = api.predict(review)
# Result: {"sentiment": "negative", "confidence": 0.93}

# Alert customer service team for negative reviews
if response['sentiment'] == 'negative' and response['confidence'] > 0.8:
    alert_team(review)
```

**Business value:**
- Identify unhappy customers quickly
- Prioritize responses
- Prevent escalation

### Use Case 2: Product Improvement

**Scenario:** Identify most common complaints

**How AeroStream helps:**
```python
# Analyze all reviews
negative_reviews = [review for review in all_reviews 
                   if predict(review)['sentiment'] == 'negative']

# Find common patterns
common_issues = analyze_keywords(negative_reviews)
# Example output: ["delay", "luggage", "customer service"]
```

**Business value:**
- Data-driven decisions
- Focus improvements where needed
- Track improvement over time

### Use Case 3: Competitive Analysis

**Scenario:** Compare sentiment across airlines

**How AeroStream helps:**
```python
airlines = ['United', 'Delta', 'American']

for airline in airlines:
    reviews = get_reviews(airline)
    sentiments = [predict(r)['sentiment'] for r in reviews]
    
    positive_rate = sentiments.count('positive') / len(sentiments)
    print(f"{airline}: {positive_rate:.2%} positive")
```

**Business value:**
- Benchmark against competitors
- Identify best practices
- Market positioning

### Use Case 4: Marketing Analytics

**Scenario:** Measure campaign effectiveness

**How AeroStream helps:**
```python
# Before campaign
before_reviews = get_reviews(date_range='before_campaign')
before_sentiment = analyze_sentiment(before_reviews)

# After campaign
after_reviews = get_reviews(date_range='after_campaign')
after_sentiment = analyze_sentiment(after_reviews)

# Compare
improvement = after_sentiment - before_sentiment
```

**Business value:**
- Measure ROI of campaigns
- Adjust marketing strategy
- Validate messaging

---

## 🔄 Complete Example Walkthrough

Let's trace a single review through the entire system:

### Input Review
```
"@UnitedAir Your customer service is TERRIBLE!!! Flight UA123 delayed 5hrs 😡 http://bit.ly/complaint"
```

### Step 1: Cleaning
```python
clean_text("@UnitedAir Your customer service is TERRIBLE!!! Flight UA123 delayed 5hrs 😡 http://bit.ly/complaint")
```
**Result:** `"your customer service is terrible flight ua123 delayed 5hrs"`

### Step 2: Embedding
```python
model.encode("your customer service is terrible flight ua123 delayed 5hrs")
```
**Result:** `[−0.723, −0.891, 0.234, ..., −0.567]` (384 numbers)

**Why these specific numbers?**
- Negative words push values negative
- "terrible", "delayed" contribute to negative direction
- The model learned these patterns from training data

### Step 3: Prediction
```python
model.predict([−0.723, −0.891, 0.234, ..., −0.567])
```

**Internal process:**
```
Logistic Regression calculation:
- Multiply embedding by learned weights
- Sum the results
- Apply sigmoid function
- If > 0.5: positive, else: negative
```

**Result:** `"negative"`

### Step 4: Confidence
```python
model.predict_proba([−0.723, −0.891, 0.234, ..., −0.567])
```
**Result:** `[0.03, 0.94, 0.03]` → negative=94%, neutral=3%, positive=3%

### Step 5: API Response
```json
{
    "sentiment": "negative",
    "confidence": 0.94
}
```

---

## 🎓 Key Takeaways

1. **Embeddings are crucial**: They translate human language into math computers understand

2. **ChromaDB organizes efficiently**: Stores millions of vectors with fast retrieval

3. **Multiple models = better results**: Test different approaches, pick the best

4. **Clean data = better predictions**: Garbage in, garbage out

5. **APIs make it accessible**: Anyone can use the model without knowing the internals

6. **Docker ensures consistency**: Works the same everywhere

---

## 🚀 Quick Reference

### Run the project:
```bash
# 1. Process data
python src/data_processing.py

# 2. Train models
python src/train_model.py

# 3. Start API
cd src && uvicorn api:app --reload

# Or use Docker
docker-compose up
```

### Make a prediction:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Great flight experience!"}'
```

### Response:
```json
{
  "sentiment": "positive",
  "confidence": 0.95
}
```

---

## 📊 Performance Tips

1. **Batch processing**: Process multiple reviews at once
   ```python
   embeddings = model.encode(texts_list)  # Faster than one-by-one
   ```

2. **GPU acceleration**: Use CUDA if available
   ```python
   model = SentenceTransformer('model-name', device='cuda')
   ```

3. **Cache embeddings**: Store frequently used embeddings
   
4. **API optimization**: Use async for concurrent requests

---

## 🔍 Troubleshooting

**Q: Model predictions are poor**
- Check data quality
- Try different models
- Increase training data
- Tune hyperparameters

**Q: ChromaDB errors**
- Check disk space
- Verify path permissions
- Clear and recreate collections

**Q: API is slow**
- Enable GPU if available
- Batch requests
- Use smaller embedding model
- Add caching layer

---

**This documentation covers the complete system. Each concept builds on the previous one to create a powerful, production-ready sentiment analysis system! 🚀**
