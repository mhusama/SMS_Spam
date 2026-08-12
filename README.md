# SMS Spam Classification using SVM

This project builds a text classification model to detect spam SMS messages using TF-IDF feature extraction and a Support Vector Machine (LinearSVC). 

---

## 🔍 Overview of Improvements

In this updated version, two major improvements were made to fix the limitations of the baseline model:

### 1. Handling Class Imbalance (Adding Spam Data via Oversampling)
- **Problem:** The original SMS dataset was heavily imbalanced, containing 4,516 Ham (legitimate) messages and only 653 Spam messages (~87% vs 13%). Because of this skew, the model was biased toward predicting "Ham" and missed a noticeable portion of real spam messages (Recall was lower).
- **Solution:** We oversampled the minority Spam class using `resample()` to match the 4,516 Ham samples, creating a balanced 50/50 dataset.
- **Impact:** Spam **Recall increased from ~74.8% to nearly 100%**, and the F1-Score jumped from 0.85 to 0.99! The model is now far more effective at catching spam without missing them.

### 2. 5-Fold Cross Validation
- **Problem:** Evaluating the model on a single 80/20 train-test split doesn't guarantee stability across different subsets of data.
- **Solution:** Added 5-fold cross validation (`cross_val_score(..., cv=5)`).
- **Impact:** The model achieved consistent performance across all 5 folds with a **Mean 5-Fold CV Accuracy of 99.52%**, confirming that the model isn't overfitting to a specific train-test split.

---

## 📊 Performance Comparison

| Metric | Original Model (Imbalanced) | Improved Model (Balanced + 5-Fold) |
|---|---|---|
| **Class Ratio (Ham : Spam)** | 4516 : 653 (Imbalanced) | 4516 : 4516 (Balanced) |
| **Accuracy** | ~96.8% | **99.34%** |
| **Precision** | ~100.0% | **98.69%** |
| **Recall (Spam Detection)** | ~74.8% | **100.00%** |
| **F1-Score** | ~0.85 | **0.993** |
| **5-Fold CV Mean Score** | ~96.5% | **99.52%** |

---

## 📁 Repository Structure

- `SMS_Spam_SVM.ipynb` — The primary Jupyter notebook (cleaning, balancing, model training, tuning, 5-fold CV, prediction).
- `SMS_Spam_SVM_old.ipynb` — Reference copy of the original notebook.
- `train.py` — Python script to train the model, evaluate 5-fold CV, and save `sms_spam_pipeline.pkl`.
- `predict.py` — Command-line interface to test custom SMS messages.
- `download_data.py` — Script to ensure `spam.csv` dataset is available locally.
- `requirements.txt` — Required Python packages.
- `.gitignore` — Ignores python cache and environment files.

---

## 🚀 How to Run

### Installation
1. Clone the repository and navigate into the folder:
   ```bash
   git clone <your-repo-url>
   cd SMS_Spam
   ```
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Notebook
Launch Jupyter Notebook to view and run `SMS_Spam_SVM.ipynb`:
```bash
jupyter notebook SMS_Spam_SVM.ipynb
```

### Running the Training Script
To re-train the model and update `sms_spam_pipeline.pkl`:
```bash
python train.py
```

### Predicting Custom SMS Messages
Predict whether a message is Spam or Ham directly from the command line:

```bash
python predict.py "Congratulations! You have won a free gift card. Call 0800111222 now to claim."
```
*Output:*
```text
==================================================
SMS Input : "Congratulations! You have won a free gift card. Call 0800111222 now to claim."
Prediction: SPAM
==================================================
```

Or run without arguments for interactive mode:
```bash
python predict.py
```
