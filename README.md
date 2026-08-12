# SMS Spam Classification using Support Vector Machine (SVM)

This repository contains an SMS Spam Classification model built using Python, TF-IDF feature extraction, and Support Vector Machines (`LinearSVC`).

It includes the baseline notebook (`SMS_Spam_SVM_old.ipynb`), the updated leak-free notebook (`SMS_Spam_SVM_new.ipynb`), training & prediction scripts, and a dataset breakdown.

---

## 📌 Methodology & Data Leakage Prevention

To ensure rigorous and reliable evaluation, we implement a **leak-free training pipeline**:

1. **Split First (80% Train / 20% Test)**: We split the original 5,169 deduplicated SMS messages into a Training set (4,135 samples) and a pure, unseen Testing set (1,034 samples) **BEFORE** performing any oversampling.
2. **Oversample Training Set ONLY**: To handle class imbalance (3,613 Ham vs 522 Spam in training data), we oversample the minority Spam class **ONLY within the training set**, producing a 50/50 balanced training set of **7,226 training samples** (3,613 Ham and 3,613 Spam).
3. **Pure Test Set Evaluation**: The model is evaluated on the 1,034 pure, unseen test messages (903 Ham, 131 Spam).

---

## 📊 Performance Comparison (`_old` vs `_new`)

| Metric | `SMS_Spam_SVM_old.ipynb` (Imbalanced Train Data) | `SMS_Spam_SVM_new.ipynb` (Leak-Free Balanced Train Data) |
|---|---|---|
| **Training Set Size** | 4,135 samples (3,613 Ham, 522 Spam) | **7,226 samples** (3,613 Ham, 3,613 Spam) |
| **Pure Test Set Size** | 1,034 samples (903 Ham, 131 Spam) | 1,034 samples (903 Ham, 131 Spam) |
| **Test Accuracy** | 98.16% | **97.78%** |
| **Spam Precision** | 95.90% | **91.54%** |
| **Spam Recall (Catch Rate)**| 89.31% *(Missed 14 spams)* | **90.84%** *(Missed 12 spams)* |
| **F1-Score** | 0.9249 | **0.9119** |
| **5-Fold CV Mean Score (Training Folds)** | 97.89% | **99.63%** |

---

## 📁 Repository Files

- `SMS_Spam_SVM_new.ipynb` — Leak-free notebook with train/test split prior to oversampling, LinearSVC training, 5-fold CV, and predictions.
- `SMS_Spam_SVM_old.ipynb` — Reference copy of the original baseline notebook.
- `train.py` — Python script replicating the leak-free dataset split, training set balancing, 5-fold CV, and pipeline export (`sms_spam_pipeline.pkl`).
- `predict.py` — Command-line tool to test custom SMS messages.
- `download_data.py` — Script to ensure `spam.csv` dataset is available locally.
- `requirements.txt` — Python dependencies (`scikit-learn`, `pandas`, `nltk`, `wordcloud`, `matplotlib`, `seaborn`).
- `.gitignore` — Git ignore file.

---

## 🚀 How to Run

### Installation
```bash
pip install -r requirements.txt
```

### Running the Notebooks
Launch Jupyter Notebook to view and run `SMS_Spam_SVM_new.ipynb` or `SMS_Spam_SVM_old.ipynb`:
```bash
jupyter notebook SMS_Spam_SVM_new.ipynb
```

### Automated Training & Prediction
```bash
python train.py
python predict.py "Congratulations! You have won a free gift card. Call 0800111222 now to claim."
```
