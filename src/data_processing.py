import re
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
import chromadb

def clean_text(text):
    """Clean text: remove URLs, mentions, special chars"""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower().strip()

def load_and_clean_data():
    """Load dataset from HuggingFace and clean it"""
    print("Loading dataset...")
    dataset = load_dataset("7Xan7der7/us_airline_sentiment")
    
    # Load and split data (only train split available)
    df = pd.DataFrame(dataset['train'])
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['airline_sentiment'])
    
    train_df = train_df.drop_duplicates(subset=['text'])
    test_df = test_df.drop_duplicates(subset=['text'])
    
    train_df = train_df.dropna(subset=['text', 'airline_sentiment'])
    test_df = test_df.dropna(subset=['text', 'airline_sentiment'])
    
    train_df['clean_text'] = train_df['text'].apply(clean_text)
    test_df['clean_text'] = test_df['text'].apply(clean_text)
    
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    print(f"Class distribution:\n{train_df['airline_sentiment'].value_counts()}")
    
    return train_df, test_df

def generate_embeddings(df, model):
    """Generate embeddings using Sentence Transformers"""
    print("Generating embeddings...")
    return model.encode(df['clean_text'].tolist(), show_progress_bar=True)

def store_in_chromadb(df, embeddings, collection_name):
    """Store embeddings in ChromaDB"""
    print(f"Storing in {collection_name}...")
    
    client = chromadb.PersistentClient(path="./chroma_db")
    
    try:
        client.delete_collection(name=collection_name)
    except:
        pass
    
    collection = client.create_collection(name=collection_name)
    
    # Add in batches of 5000 to avoid ChromaDB batch limit
    batch_size = 5000
    for i in range(0, len(df), batch_size):
        end_idx = min(i + batch_size, len(df))
        collection.add(
            embeddings=embeddings[i:end_idx].tolist(),
            documents=df['clean_text'].iloc[i:end_idx].tolist(),
            metadatas=[{"label": label} for label in df['airline_sentiment'].iloc[i:end_idx].tolist()],
            ids=[str(j) for j in range(i, end_idx)]
        )
    
    print(f"Stored {len(df)} embeddings")

def main():
    train_df, test_df = load_and_clean_data()
    
    print("Loading embedding model...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    train_embeddings = generate_embeddings(train_df, model)
    test_embeddings = generate_embeddings(test_df, model)
    
    store_in_chromadb(train_df, train_embeddings, "train_collection")
    store_in_chromadb(test_df, test_embeddings, "test_collection")
    
    print("Data processing complete!")

if __name__ == "__main__":
    main()
