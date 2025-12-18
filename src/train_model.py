import numpy as np
import chromadb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

def load_embeddings_from_chromadb(collection_name):
    """Load embeddings and labels from ChromaDB"""
    print(f"Loading {collection_name}...")
    
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(name=collection_name)
    
    results = collection.get(include=['embeddings', 'metadatas'])
    
    embeddings = np.array(results['embeddings'])
    labels = [meta['label'] for meta in results['metadatas']]
    
    return embeddings, labels

def train_models(X_train, y_train):
    """Train multiple models"""
    models = {
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    return trained_models

def evaluate_models(models, X_test, y_test):
    """Evaluate models and return best one"""
    best_model = None
    best_score = 0
    best_name = ""
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n{name.upper()}:")
        print(f"Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))
        
        if accuracy > best_score:
            best_score = accuracy
            best_model = model
            best_name = name
    
    print(f"\nBest: {best_name} ({best_score:.4f})")
    return best_model, best_name

def save_model(model):
    """Save best model"""
    os.makedirs("./models", exist_ok=True)
    
    with open("./models/best_model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    print("Model saved to ./models/best_model.pkl")

def main():
    X_train, y_train = load_embeddings_from_chromadb("train_collection")
    X_test, y_test = load_embeddings_from_chromadb("test_collection")
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    models = train_models(X_train, y_train)
    best_model, _ = evaluate_models(models, X_test, y_test)
    save_model(best_model)
    
    print("Training complete!")

if __name__ == "__main__":
    main()
