import os
import re
import string
import joblib
import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Download NLTK data if needed
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

stop_words = set(stopwords.words('english'))
lemm = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = [
        lemm.lemmatize(word)
        for word in text.split()
        if word not in stop_words
    ]
    return ' '.join(tokens)

def load_and_preprocess_data():
    csv_file = "spam.csv" if os.path.exists("spam.csv") else "SMS.csv"
    try:
        df = pd.read_csv(csv_file, encoding="latin-1")
    except Exception:
        df = pd.read_csv(csv_file, encoding="utf-8")

    if 'v1' in df.columns and 'v2' in df.columns:
        df = df[['v1', 'v2']]
        df.columns = ['label', 'message']

    df = df.drop_duplicates()
    df['clean_message'] = df['message'].apply(clean_text)
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    
    # Balance dataset via oversampling minority class (spam)
    df_ham = df[df.label == 0]
    df_spam = df[df.label == 1]
    df_spam_oversampled = resample(df_spam, replace=True, n_samples=len(df_ham), random_state=42)
    df_balanced = pd.concat([df_ham, df_spam_oversampled]).reset_index(drop=True)
    
    return df, df_balanced

def train_model():
    df_original, df_balanced = load_and_preprocess_data()
    
    print("=== SMS Spam SVM Classifier Training ===")
    print("Original Dataset Counts:\n", df_original['label'].value_counts().to_dict())
    print("Balanced Dataset Counts:\n", df_balanced['label'].value_counts().to_dict())
    
    X = df_balanced['clean_message']
    y = df_balanced['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('svm', LinearSVC(random_state=42))
    ])
    
    # Hyperparameter tuning
    param_grid = {'svm__C': [0.1, 1, 10]}
    grid = GridSearchCV(pipeline, param_grid, cv=5)
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    pred = best_model.predict(X_test)
    
    print("\nBest Parameters:", grid.best_params_)
    print("Accuracy :", accuracy_score(y_test, pred))
    print("Precision:", precision_score(y_test, pred))
    print("Recall   :", recall_score(y_test, pred))
    print("F1-Score :", f1_score(y_test, pred))
    print("\nClassification Report:\n", classification_report(y_test, pred))
    
    # 5-Fold Cross Validation
    scores = cross_val_score(best_model, df_balanced['clean_message'], df_balanced['label'], cv=5)
    print("5-Fold CV Scores:", scores)
    print("Mean CV Accuracy:", scores.mean())
    
    # Save trained model
    joblib.dump(best_model, "sms_spam_pipeline.pkl")
    print("\nModel saved to sms_spam_pipeline.pkl!")

if __name__ == "__main__":
    train_model()
