import numpy as np, chromadb, pickle, os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_from_chromadb(name):
    client = chromadb.PersistentClient(path="./chroma_db")
    results = client.get_collection(name).get(include=['embeddings', 'metadatas'])
    return np.array(results['embeddings']), [m['label'] for m in results['metadatas']]

def main():
    X_train, y_train = load_from_chromadb("train_collection")
    X_test, y_test = load_from_chromadb("test_collection")
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    
    models = {
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    best, best_score = None, 0
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, model.predict(X_test)))
        if acc > best_score:
            best, best_score = model, acc
    
    os.makedirs("./models", exist_ok=True)
    pickle.dump(best, open("./models/best_model.pkl", "wb"))
    print(f"\nBest model saved! Accuracy: {best_score:.4f}")

if __name__ == "__main__":
    main()
