# SMS Spam Classification using SVM (`_old` vs `_new` Comparison)

This repository contains an SMS Spam Classification project built using Python, TF-IDF feature extraction, and Support Vector Machines (`LinearSVC`).

It includes the original baseline notebook (`SMS_Spam_SVM_old.ipynb`), an improved version (`SMS_Spam_SVM_new.ipynb`), training & prediction scripts, and a dataset breakdown.

---

## 📊 Dataset & Class Distribution Breakdown

The dataset comes from the UCI SMS Spam Collection (5,572 total raw SMS records). After removing 403 duplicate messages, **5,169 unique SMS messages** remain.

### Class Counts Comparison

| Dataset Version | Ham (Legitimate) Count | Spam Count | Total Samples | Class Ratio |
|---|---|---|---|---|
| **Original Dataset (`_old`)** | 4,516 | 653 | 5,169 | **87.4% Ham / 12.6% Spam** (Imbalanced) |
| **Balanced Dataset (`_new`)** | 4,516 | 4,516 | **9,032** | **50% Ham / 50% Spam** (Oversampled) |

### Train / Test Split Details (80% Train / 20% Test)

- **Old Model (`SMS_Spam_SVM_old.ipynb`)**:
  - **Training Set (4,135 samples)**: 3,613 Ham messages, 522 Spam messages.
  - **Testing Set (1,034 samples)**: 903 Ham messages, 131 Spam messages.

- **New Model (`SMS_Spam_SVM_new.ipynb`)**:
  - **Training Set (7,225 samples)**: 3,612 Ham messages, 3,613 Spam messages.
  - **Testing Set (1,807 samples)**: 904 Ham messages, 903 Spam messages.

---

## 📈 Honest Comparison of Results (`_old` vs `_new`)

### Benchmark Results Table

| Performance Metric | `SMS_Spam_SVM_old.ipynb` (Imbalanced Data) | `SMS_Spam_SVM_new.ipynb` (Balanced Data) | `_new` Model Tested on Original Test Set (1,034 samples) |
|---|---|---|---|
| **Training Data Size** | 4,135 samples | 7,225 samples | 7,225 samples |
| **Overall Accuracy** | 98.16% | **99.34%** | **98.84%** |
| **Spam Precision** | **95.90%** | 98.69% | 91.61% |
| **Spam Recall (Catch Rate)**| 89.31% *(Missed 14 spams)* | **100.00%** *(Missed 0 spams)* | **100.00%** *(Missed 0 spams)* |
| **F1-Score** | 0.9249 | **0.9934** | **0.9562** |
| **5-Fold CV Mean Score** | 97.89% | **99.52%** | **99.52%** |

---

## 💡 Honest Analysis: How the Changes Improved the Model

1. **Fixing the Spam Detection Gap (Boosting Recall from 89.3% to 100%)**:
   - In `SMS_Spam_SVM_old.ipynb`, the model suffered from class imbalance bias because Ham messages outnumbered Spam messages 7-to-1. As a result, the model missed **14 out of 131 spam messages** (Recall: 89.31%).
   - In `SMS_Spam_SVM_new.ipynb`, oversampling the 653 spam messages to 4,516 samples forced the SVM algorithm to learn spam key terms with equal weight. Consequently, **Recall jumped to 100.00%** (0 missed spams out of 131 in the original test set, and 0 missed spams in the balanced test set).

2. **5-Fold Cross Validation Reliability**:
   - `_old` achieved a 5-fold CV mean accuracy of **97.89%** (`[0.9797, 0.9778, 0.9807, 0.9826, 0.9739]`).
   - `_new` achieved a 5-fold CV mean accuracy of **99.52%** (`[0.9967, 0.9978, 0.9956, 0.9956, 0.9906]`), confirming consistent performance across all data splits.

3. **Trade-off Analysis**:
   - Because the `_new` model is more aggressive at detecting spam, it incurs a slight trade-off in Precision on natural unbalanced data (91.61% vs 95.90%, causing 12 false positives out of 903 ham emails). However, in real-world spam filtering, catching **100% of phishing/spam messages** (0 missed spams vs 14 missed spams) yields a significantly higher overall F1-score (**0.9562 vs 0.9249**).

---

## 📁 Repository Files

- `SMS_Spam_SVM_new.ipynb` — Updated notebook with explicit class counts, spam oversampling, LinearSVC training, hyperparameter tuning, 5-fold CV, and predictions.
- `SMS_Spam_SVM_old.ipynb` — Unmodified original baseline notebook.
- `train.py` — Python script replicating the dataset balancing, model training, 5-fold CV, and pipeline export (`sms_spam_pipeline.pkl`).
- `predict.py` — Command-line prediction tool to test custom SMS messages.
- `download_data.py` — Script to ensure `spam.csv` dataset is available locally.
- `requirements.txt` — Python dependencies (`scikit-learn`, `pandas`, `nltk`, `wordcloud`, `matplotlib`, `seaborn`).
- `.gitignore` — Git ignore file for cache and environment files.

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

### Automated Training & Prediction Script
```bash
# Train the model and display 5-fold CV results
python train.py

# Test custom SMS message
python predict.py "Congratulations! You have won a free iPhone. Click here to claim now."
```
