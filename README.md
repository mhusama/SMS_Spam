# SMS Spam Classification using Support Vector Machine (SVM)

This repository contains an improved SMS Spam Classification model built using Python, TF-IDF feature extraction, and Support Vector Machines (`LinearSVC`).

---

## 📌 Improvements & Key Changes

Two major updates were made to improve the baseline model's performance and evaluation reliability:

### 1. Adding More Spam Data (Class Balancing via Oversampling)
- **The Problem:** The original dataset had 4,516 Ham messages and only 653 Spam messages (~87% Ham vs 13% Spam). Because of this severe class imbalance, the baseline model was biased toward predicting Ham, leading to lower Recall for Spam detection (it missed a lot of real spam messages).
- **The Change:** We added more spam samples by oversampling the minority Spam class using `resample()` to match the 4,516 Ham samples. This created a balanced dataset (4,516 Ham vs 4,516 Spam).
- **The Improvement:** Spam **Recall improved from 89.31% to 100.00%**, and **Precision reached 98.80%**, making the model catch virtually all spam messages without missing them.

### 2. 5-Fold Cross Validation
- **The Problem:** Evaluating the model on just one train-test split doesn't prove how well the model generalizes on different samples of data.
- **The Change:** Integrated 5-fold cross-validation (`cross_val_score(..., cv=5)`).
- **The Improvement:** The model scored `[0.9967, 0.9978, 0.9956, 0.9956, 0.9906]` across the 5 folds, giving a **Mean 5-Fold CV Accuracy of 99.52%**.

---

## 📊 Results Summary

Here is the direct comparison between the original model and the updated balanced model:

| Metric | Original Model (Imbalanced) | Improved Model (Balanced Spam Data + 5-Fold CV) |
|---|---|---|
| **Class Distribution (Ham : Spam)** | 4,516 : 653 | **4,516 : 4,516** (Balanced) |
| **Accuracy** | 98.45% | **99.39%** |
| **Precision** | 98.32% | **98.80%** |
| **Recall (Spam Detection)** | 89.31% | **100.00%** |
| **F1-Score** | 0.9360 | **0.9939** |
| **5-Fold CV Mean Accuracy** | 97.95% | **99.52%** |

---

## 📁 Repository Files

- `SMS_Spam_SVM.ipynb` - Primary notebook containing data cleaning, spam oversampling, LinearSVC training, hyperparameter tuning, 5-fold CV, and predictions.
- `SMS_Spam_SVM_old.ipynb` - Reference copy of the original notebook.
- `train.py` - Python script that reproduces dataset balancing, model training, 5-fold CV, and saves `sms_spam_pipeline.pkl`.
- `predict.py` - Command-line script to test custom SMS messages.
- `download_data.py` - Utility to download `spam.csv` locally.
- `requirements.txt` - Required packages (`scikit-learn`, `pandas`, `nltk`, `wordcloud`, `matplotlib`, `seaborn`).
- `.gitignore` - Standard git ignore configuration.

---

## 🚀 How to Run

### 1. Requirements Setup
Install the necessary Python packages:
```bash
pip install -r requirements.txt
```

### 2. Running the Jupyter Notebook
Launch Jupyter Notebook to view and execute `SMS_Spam_SVM.ipynb`:
```bash
jupyter notebook SMS_Spam_SVM.ipynb
```

### 3. Training via Command Line
Run the training script to train the model, display metrics, and save the pipeline:
```bash
python train.py
```

### 4. Predicting Custom Messages
Test any SMS message directly from your terminal:
```bash
python predict.py "Congratulations! You have won a free iPhone. Click here to claim now."
```
*Output:*
```text
==================================================
SMS Input : "Congratulations! You have won a free iPhone. Click here to claim now."
Prediction: Spam
==================================================
```
