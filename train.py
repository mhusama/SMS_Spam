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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

stop_words = set(stopwords.words('english'))
lemm = WordNetLemmatizer()

HAM_SAMPLE_COUNT = 3600
SPAM_SAMPLE_COUNTS = (1200, 1800, 2400, 3000, 3600)
REPORT_FILE = "spam_ratio_experiment_report.txt"


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

def train_model():
    csv_file = "spam.csv" if os.path.exists("spam.csv") else "SMS.csv"
    try:
        df = pd.read_csv(csv_file, encoding="latin-1")
    except Exception:
        df = pd.read_csv(csv_file, encoding="utf-8")

    if 'v1' in df.columns and 'v2' in df.columns:
        df = df[['v1', 'v2']]
        df.columns = ['label', 'message']

    df = df.drop_duplicates().reset_index(drop=True)
    df['clean_message'] = df['message'].apply(clean_text)
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    
    # 1. SPLIT FIRST (LEAK-FREE)
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_message'], df['label'], test_size=0.20, random_state=42, stratify=df['label']
    )
    
    # 2. Compare the original training distribution with fixed ham/spam ratios.
    train_df = pd.DataFrame({'clean_message': X_train, 'label': y_train})
    train_ham = train_df[train_df.label == 0]
    train_spam = train_df[train_df.label == 1]

    if len(train_ham) < HAM_SAMPLE_COUNT:
        raise ValueError(
            f"The training split has {len(train_ham)} ham messages; "
            f"{HAM_SAMPLE_COUNT} are required for the ratio experiment."
        )

    experiment_sets = [("OG:OG (original distribution)", train_df)]
    sampled_ham = resample(
        train_ham,
        replace=False,
        n_samples=HAM_SAMPLE_COUNT,
        random_state=42
    )
    for spam_count in SPAM_SAMPLE_COUNTS:
        sampled_spam = resample(
            train_spam,
            replace=True,
            n_samples=spam_count,
            random_state=42
        )
        experiment_sets.append(
            (f"{HAM_SAMPLE_COUNT}:{spam_count}",
             pd.concat([sampled_ham, sampled_spam]).reset_index(drop=True))
        )

    print("=== SMS Spam SVM Classifier Ratio Experiment ===")
    print("Raw Train counts:", y_train.value_counts().to_dict())
    print("Pure Test counts:", y_test.value_counts().to_dict())

    param_grid = {'svm__C': [0.1, 1, 10]}
    results = []
    final_model = None

    for ratio, experiment_df in experiment_sets:
        X_experiment = experiment_df['clean_message']
        y_experiment = experiment_df['label']
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('svm', LinearSVC(random_state=42))
        ])
        grid = GridSearchCV(pipeline, param_grid, cv=5)
        grid.fit(X_experiment, y_experiment)

        best_model = grid.best_estimator_
        pred = best_model.predict(X_test)
        tn, ham_as_spam, spam_as_ham, tp = confusion_matrix(
            y_test, pred, labels=[0, 1]
        ).ravel()
        result = {
            'ratio': ratio,
            'train_counts': y_experiment.value_counts().to_dict(),
            'best_params': grid.best_params_,
            'cv_accuracy': grid.best_score_,
            'accuracy': accuracy_score(y_test, pred),
            'ham_as_spam': int(ham_as_spam),
            'spam_as_ham': int(spam_as_ham),
            'precision': precision_score(y_test, pred),
            'recall': recall_score(y_test, pred),
            'f1': f1_score(y_test, pred),
            'classification_report': classification_report(y_test, pred),
        }
        results.append(result)

        print(
            f"\n{ratio} (ham:spam) | "
            f"train={result['train_counts']} | "
            f"accuracy={result['accuracy']:.4f} | "
            f"precision={result['precision']:.4f} | "
            f"recall={result['recall']:.4f} | "
            f"f1={result['f1']:.4f}"
        )
        print("Best Parameters:", grid.best_params_)

        if ratio == f"{HAM_SAMPLE_COUNT}:{SPAM_SAMPLE_COUNTS[-1]}":
            final_model = best_model

    report_lines = [
        "SMS Spam SVM - Ham/Spam Training Ratio Experiment",
        "=" * 54,
        f"Dataset: {csv_file}",
        "Split: stratified 80% train / 20% test, random_state=42",
        "Test set: same untouched holdout for every experiment",
        f"Training split original counts (label 0=ham, 1=spam): {y_train.value_counts().to_dict()}",
        f"Test split counts (label 0=ham, 1=spam): {y_test.value_counts().to_dict()}",
        "",
        "Holdout metrics by training ratio (ham:spam):",
    ]
    for result in results:
        report_lines.extend([
            "",
            f"Ratio: {result['ratio']}",
            f"  Training counts (ham:spam): "
            f"{result['train_counts'].get(0, 0)}:{result['train_counts'].get(1, 0)}",
            f"  Best SVM parameters: {result['best_params']}",
            f"  5-fold CV accuracy: {result['cv_accuracy']:.4f}",
            f"  Test accuracy: {result['accuracy']:.4f}",
            f"  Test precision: {result['precision']:.4f}",
            f"  Test recall: {result['recall']:.4f}",
            f"  Test F1-score: {result['f1']:.4f}",
            "  Classification report:",
            *[f"    {line}" for line in result['classification_report'].rstrip().splitlines()],
        ])

    report_lines.extend([
        "",
        "Saved model: sms_spam_pipeline.pkl",
        f"Saved model training ratio: {HAM_SAMPLE_COUNT}:{SPAM_SAMPLE_COUNTS[-1]} (ham:spam)",
        "",
        "FINAL COMPARISON - HOLDOUT TEST RESULTS",
        "Ratio (ham:spam)             Accuracy   Ham reported as spam   Spam reported as ham",
        "----------------------------  --------   --------------------   --------------------",
    ])
    for result in results:
        report_lines.append(
            f"{result['ratio']:<28}  "
            f"{result['accuracy'] * 100:>6.2f}%"
            f"{result['ham_as_spam']:>22}"
            f"{result['spam_as_ham']:>23}"
        )
    report_lines.extend([
        "",
        "Ham reported as spam = ham messages incorrectly classified as spam.",
        "Spam reported as ham = spam messages incorrectly classified as ham.",
        "",
    ])
    with open(REPORT_FILE, "w", encoding="utf-8") as report_file:
        report_file.write("\n".join(report_lines))

    if final_model is None:
        raise RuntimeError("The 3600:3600 experiment did not produce a model.")
    joblib.dump(final_model, "sms_spam_pipeline.pkl")
    print(f"\nExperiment report saved to {REPORT_FILE}")
    print("Model saved to sms_spam_pipeline.pkl!")

if __name__ == "__main__":
    train_model()
