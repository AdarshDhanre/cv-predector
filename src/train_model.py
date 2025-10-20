# src/train_model.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
from sentence_transformers import SentenceTransformer

# CONFIG
DATA_CSV = "../data/labeled_dataset.csv"  # columns: job_id, resume_text, jd_text, label (1 = good, 0 = bad)
MODEL_DIR = "../models"
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data(path):
    df = pd.read_csv(path)
    # Expect columns: 'jd_text', 'resume_text', 'label'
    return df

def train_classifier(df):
    # Create pairwise text: combine JD + resume for classifier
    X_text = (df['jd_text'].fillna('') + ' [SEP] ' + df['resume_text'].fillna('')).values
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(X_text, y, test_size=0.2, random_state=42, stratify=y)

    tf = TfidfVectorizer(max_features=30000, ngram_range=(1,2))
    X_train_tfidf = tf.fit_transform(X_train)
    X_test_tfidf = tf.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_tfidf, y_train)

    preds = clf.predict(X_test_tfidf)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    joblib.dump(tf, os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
    joblib.dump(clf, os.path.join(MODEL_DIR, "clf_logreg.joblib"))
    print("Saved classifier and vectorizer.")

def save_embedder():
    # We'll use SentenceTransformer and let it auto-download
    model_name = "all-MiniLM-L6-v2"
    embedder = SentenceTransformer(model_name)
    embedder.save(os.path.join(MODEL_DIR, "embedder"))
    print("Saved embedder to models/embedder")

def main():
    if not os.path.exists(DATA_CSV):
        print("No labeled data found at", DATA_CSV, ". You can skip classifier training.")
    else:
        df = load_data(DATA_CSV)
        train_classifier(df)
    save_embedder()

if __name__ == "__main__":
    main()
