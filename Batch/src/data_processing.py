import re, pandas as pd, chromadb
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower().strip()

def load_data():
    print("Loading dataset from HuggingFace...")
    df = pd.DataFrame(load_dataset("7Xan7der7/us_airline_sentiment")['train'])
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['airline_sentiment'])
    
    for d in [train_df, test_df]:
        d.drop_duplicates(subset=['text'], inplace=True)
        d.dropna(subset=['text', 'airline_sentiment'], inplace=True)
        d['clean_text'] = d['text'].apply(clean_text)
    
    print(f"Train: {len(train_df)} | Test: {len(test_df)}")
    return train_df, test_df

def store_embeddings(df, embeddings, name):
    print(f"Storing {name}...")
    client = chromadb.PersistentClient(path="./chroma_db")
    try: client.delete_collection(name)
    except: pass
    
    collection = client.create_collection(name)
    for i in range(0, len(df), 5000):
        end = min(i + 5000, len(df))
        collection.add(
            embeddings=embeddings[i:end].tolist(),
            documents=df['clean_text'].iloc[i:end].tolist(),
            metadatas=[{"label": l} for l in df['airline_sentiment'].iloc[i:end]],
            ids=[str(j) for j in range(i, end)]
        )
    print(f"Done: {len(df)} embeddings")

def main():
    train_df, test_df = load_data()
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    print("Generating embeddings...")
    store_embeddings(train_df, model.encode(train_df['clean_text'].tolist(), show_progress_bar=True), "train_collection")
    store_embeddings(test_df, model.encode(test_df['clean_text'].tolist(), show_progress_bar=True), "test_collection")
    print("Done!")

if __name__ == "__main__":
    main()
