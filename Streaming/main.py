import re, os, time, uuid, random, psycopg2, requests
from datetime import datetime

API_URL = os.getenv("API_URL", "http://batch-api:8000")
DB_CONFIG = {"host": os.getenv("DB_HOST", "postgres"), "port": 5432, "database": "aerostream", "user": "aerostream", "password": "aerostream123"}
AIRLINES = ["United", "Delta", "American", "Southwest", "JetBlue", "Spirit"]

def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower().strip()

def detect_airline(text):
    for a in AIRLINES:
        if a.lower() in text.lower():
            return a
    return random.choice(AIRLINES)

def get_prediction(text):
    try:
        r = requests.post(f"{API_URL}/predict", json={"text": text}, timeout=10)
        return r.json() if r.status_code == 200 else None
    except:
        return None

def get_db():
    for _ in range(5):
        try: return psycopg2.connect(**DB_CONFIG)
        except: time.sleep(2)
    raise Exception("DB connection failed")

def init_db():
    conn = get_db()
    conn.cursor().execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY, original_text TEXT, cleaned_text TEXT,
            airline VARCHAR(50), sentiment VARCHAR(20), confidence FLOAT,
            batch_id VARCHAR(50), created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    conn.commit()
    conn.close()

def save_predictions(predictions, batch_id):
    conn = get_db()
    cur = conn.cursor()
    for p in predictions:
        cur.execute("INSERT INTO predictions (original_text, cleaned_text, airline, sentiment, confidence, batch_id) VALUES (%s,%s,%s,%s,%s,%s)",
            (p["original"], p["cleaned"], p["airline"], p["sentiment"], p["confidence"], batch_id))
    conn.commit()
    conn.close()


def process_batch(reviews):
    batch_id = str(uuid.uuid4())[:8]
    print(f"\n[{batch_id}] Processing {len(reviews)} reviews...")
    
    predictions = []
    for text in reviews:
        cleaned = clean_text(text)
        if len(cleaned) < 3: continue
        result = get_prediction(text)
        if result:
            predictions.append({"original": text, "cleaned": cleaned, "airline": detect_airline(text), 
                               "sentiment": result["sentiment"], "confidence": result["confidence"]})
    
    if predictions: save_predictions(predictions, batch_id)
    summary = {}
    for p in predictions: summary[p["sentiment"]] = summary.get(p["sentiment"], 0) + 1
    print(f"Results: {summary}")
    return {"batch_id": batch_id, "count": len(predictions)}

def generate_reviews(n=10):
    templates = {
        "positive": ["Great flight with {a}!", "Love {a}, best service!", "{a} never disappoints!"],
        "negative": ["Terrible {a} experience, delayed!", "{a} lost my luggage!", "Worst {a} service ever!"],
        "neutral": ["My {a} flight was okay.", "{a} was average."]
    }
    reviews = []
    for _ in range(n):
        sent = random.choices(["negative", "positive", "neutral"], weights=[0.4, 0.35, 0.25])[0]
        reviews.append(random.choice(templates[sent]).format(a=random.choice(AIRLINES)))
    return reviews

def run_pipeline():
    print(f"\n{'='*40}\nAeroStream - {datetime.now()}\n{'='*40}")
    init_db()
    return process_batch(generate_reviews(10))

if __name__ == "__main__":
    run_pipeline()
