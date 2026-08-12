import os
import re
import string
import joblib
import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

# Download NLTK corpora if needed
for resource in ['stopwords', 'wordnet']:
    try:
        nltk.data.find(f'corpora/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = [
        lemmatizer.lemmatize(word)
        for word in text.split()
        if word not in stop_words
    ]
    return ' '.join(tokens)

def load_data(filepath="spam.csv"):
    if not os.path.exists(filepath):
        from download_data import download_dataset
        filepath = download_dataset()
        
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except Exception:
        df = pd.read_csv(filepath, encoding='latin-1')
        
    # Standardize columns
    if 'v1' in df.columns and 'v2' in df.columns:
        df = df[['v1', 'v2']].copy()
        df.columns = ['label', 'message']
    elif 'label' in df.columns and 'message' in df.columns:
        df = df[['label', 'message']].copy()
    else:
        df.columns = ['label', 'message']
        
    df = df.drop_duplicates(subset=['message']).reset_index(drop=True)
    df['clean_message'] = df['message'].apply(clean_text)
    
    # Map label ham->0, spam->1 cleanly
    df['label'] = df['label'].astype(str).str.strip().str.lower()
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    df = df.dropna(subset=['label']).reset_index(drop=True)
    df['label'] = df['label'].astype(int)
        
    return df

def train_and_evaluate():
    print("=" * 60)
    print("      SMS SPAM CLASSIFIER - TRAINING & EVALUATION PIPELINE")
    print("=" * 60)
    
    df = load_data()
    print(f"[*] Dataset Loaded & Cleaned: {len(df)} samples after deduplication.")
    print(f"    - Ham count : {sum(df['label'] == 0)}")
    print(f"    - Spam count: {sum(df['label'] == 1)}")
    
    X = df['clean_message']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print("\n[*] Benchmarking Baseline Models...")
    models = {
        'Multinomial Naive Bayes': MultinomialNB(),
        'Linear SVC': LinearSVC(C=1.0, random_state=42, max_iter=2000),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    benchmark_results = []
    
    for name, model in models.items():
        pipe = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)),
            ('clf', model)
        ])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        
        # Calculate ROC AUC if probability or decision function supported
        if hasattr(pipe.named_steps['clf'], 'predict_proba'):
            probs = pipe.predict_proba(X_test)[:, 1]
            auc_val = roc_auc_score(y_test, probs)
        elif hasattr(pipe.named_steps['clf'], 'decision_function'):
            probs = pipe.decision_function(X_test)
            auc_val = roc_auc_score(y_test, probs)
        else:
            auc_val = np.nan
            
        benchmark_results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': auc_val
        })
        
    benchmark_df = pd.DataFrame(benchmark_results)
    print("\n" + benchmark_df.to_string(index=False))
    
    print("\n[*] Performing Hyperparameter Tuning for LinearSVC...")
    tune_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ('clf', CalibratedClassifierCV(estimator=LinearSVC(random_state=42, max_iter=2000)))
    ])
    
    param_grid = {
        'tfidf__max_features': [3000, 5000, 8000],
        'tfidf__min_df': [1, 2],
        'clf__estimator__C': [0.1, 0.5, 1.0, 2.0]
    }
    
    grid = GridSearchCV(tune_pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    print(f"    - Best Params: {grid.best_params_}")
    print(f"    - Best CV F1-Score: {grid.best_score_:.4f}")
    
    # Final Test Set Evaluation
    test_preds = best_model.predict(X_test)
    test_probs = best_model.predict_proba(X_test)[:, 1]
    
    final_acc = accuracy_score(y_test, test_preds)
    final_prec = precision_score(y_test, test_preds)
    final_rec = recall_score(y_test, test_preds)
    final_f1 = f1_score(y_test, test_preds)
    final_auc = roc_auc_score(y_test, test_probs)
    
    print("\n[*] Final Model Performance on Test Set:")
    print(f"    - Accuracy : {final_acc:.4f}")
    print(f"    - Precision: {final_prec:.4f}")
    print(f"    - Recall   : {final_rec:.4f}")
    print(f"    - F1-Score : {final_f1:.4f}")
    print(f"    - ROC-AUC  : {final_auc:.4f}")
    
    print("\n[*] Classification Report:")
    print(classification_report(y_test, test_preds, target_names=['Ham', 'Spam']))
    
    print("\n[*] Confusion Matrix:")
    print(confusion_matrix(y_test, test_preds))
    
    # Save Pipeline
    model_filename = "sms_spam_pipeline.pkl"
    joblib.dump(best_model, model_filename)
    print(f"\n[+] Trained model pipeline successfully saved to '{model_filename}'!")

if __name__ == "__main__":
    train_and_evaluate()
