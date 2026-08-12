# 📱 SMS Spam Classification using Machine Learning

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-orange.svg)
![NLTK](https://img.shields.io/badge/NLTK-3.8%2B-green.svg)
![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)

An end-to-end Machine Learning project that classifies SMS messages into **Ham** (legitimate) or **Spam** (unsolicited commercial/phishing messages). It includes dataset preprocessing, exploratory data analysis (EDA), multi-model benchmarking, hyperparameter optimization, model serialization, and an interactive CLI prediction interface.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Dataset Description](#-dataset-description)
- [Methodology & Pipeline](#-methodology--pipeline)
- [Model Performance Benchmark](#-model-performance-benchmark)
- [Project Structure](#-project-structure)
- [Installation & Requirements](#-installation--requirements)
- [Usage Guide](#-usage-guide)
  - [1. Running the Notebook](#1-running-the-notebook)
  - [2. Automated Training Script](#2-automated-training-script)
  - [3. CLI Prediction Tool](#3-cli-prediction-tool)
- [Future Enhancements](#-future-enhancements)
- [License](#-license)

---

## 📌 Project Overview
Unsolicited messages (Spam) pose significant security risks, ranging from marketing clutter to malicious phishing attacks. This repository provides a robust NLP and Machine Learning solution trained on the benchmark [UCI SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).

By leveraging **TF-IDF Feature Vectorization** paired with a **Calibrated Linear Support Vector Machine (LinearSVC)** classifier, the model achieves an impressive **97.97% Accuracy** and an **F1-Score of 91.95%** on unseen test data.

---

## ✨ Key Features
- **Data Preprocessing**: Handles duplicates, lowercasing, regex filtering (URLs, emails, digits, punctuation), NLTK stopword filtering, and WordNet lemmatization.
- **Exploratory Data Analysis (EDA)**: Character length & word count distributions, class imbalance analysis, and WordCloud visualizations.
- **Multi-Model Benchmarking**: Evaluates four classifiers:
  1. *Multinomial Naive Bayes*
  2. *Linear Support Vector Classifier (LinearSVC)*
  3. *Logistic Regression*
  4. *Random Forest Classifier*
- **Hyperparameter Optimization**: 5-Fold Stratified `GridSearchCV` to optimize TF-IDF feature parameters and Support Vector regularization.
- **Probability Calibration**: Uses `CalibratedClassifierCV` to extract true prediction confidence percentages.
- **Model Serialization**: Exported pipeline via `joblib` (`sms_spam_pipeline.pkl`) for production integration.
- **CLI Prediction Tool**: User-friendly `predict.py` command-line utility for testing real-time SMS messages.

---

## 📊 Dataset Description
The dataset contains **5,572 raw SMS messages** written in English, tagged as `ham` or `spam`.

| Metric | Raw Dataset | Cleaned Dataset (Deduplicated) |
|---|---|---|
| Total Messages | 5,572 | 5,169 |
| Ham Count (Legitimate) | 4,825 (86.6%) | 4,516 (87.4%) |
| Spam Count | 747 (13.4%) | 653 (12.6%) |

*Note: Automated deduplication removed 403 duplicate messages to prevent data leakage between train and test splits.*

---

## 🛠️ Methodology & Pipeline

1. **Text Normalization**:
   - Convert all text to lower case.
   - Strip URLs (`http://...`, `www...`), email addresses, numbers, and punctuation.
   - Tokenize text into words.
   - Filter out standard English stopwords (e.g., *the, is, at, which*).
   - Apply WordNet Lemmatization to map tokens to base forms (e.g., *running* $\rightarrow$ *run*).

2. **Feature Extraction**:
   - **TF-IDF Vectorization** with unigrams and bigrams (`ngram_range=(1, 2)`).
   - Sublinear term frequency scaling (`sublinear_tf=True`) to smooth word count impact.
   - Feature space constrained to top 3,000 informative n-grams.

3. **Validation Strategy**:
   - 80/20 Stratified Train-Test Split to preserve class proportions.
   - 5-Fold Stratified Cross-Validation during hyperparameter tuning.

---

## 📈 Model Performance Benchmark

### Baseline Comparison (Default Parameters)
| Classifier Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Linear SVC** | **98.26%** | **95.94%** | **90.08%** | **92.91%** | **0.9951** |
| Random Forest | 97.68% | 97.35% | 83.97% | 90.16% | 0.9890 |
| Multinomial Naive Bayes | 96.81% | 100.00% | 74.81% | 85.59% | 0.9870 |
| Logistic Regression | 95.55% | 95.70% | 67.94% | 79.46% | 0.9921 |

### Final Tuned Calibrated LinearSVC Results (Test Set)
- **Accuracy**: `97.97%`
- **Precision**: `92.31%`
- **Recall**: `91.60%`
- **F1-Score**: `91.95%`
- **ROC-AUC**: `0.9948`

#### Confusion Matrix
```
               Predicted HAM    Predicted SPAM
Actual HAM          893               10
Actual SPAM          11              120
```

---

## 📁 Project Structure

```
SMS_Spam/
├── SMS_Spam_SVM.ipynb      # Main Jupyter Notebook (EDA, Modeling, Evaluation)
├── train.py                # Standalone automated training & evaluation script
├── predict.py              # CLI tool for interactive text prediction
├── download_data.py        # Dataset downloader & validator script
├── requirements.txt        # Python package dependencies
├── .gitignore              # Git ignore configuration
├── spam.csv                # SMS Spam dataset (5,572 records)
├── SMS.csv                 # Dataset fallback copy
├── sms_spam_pipeline.pkl   # Serialized trained model pipeline
└── README.md               # Project documentation
```

---

## ⚙️ Installation & Requirements

### Prerequisites
- Python `3.9` or higher
- `pip` package manager

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/SMS_Spam.git
   cd SMS_Spam
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage Guide

### 1. Running the Notebook
Open the interactive Jupyter notebook to view step-by-step EDA, charts, and visualizations:
```bash
jupyter notebook SMS_Spam_SVM.ipynb
```

### 2. Automated Training Script
Run the automated pipeline script to train the classifiers, run grid search, print evaluation benchmarks, and save the pipeline pickle:
```bash
python train.py
```

### 3. CLI Prediction Tool
Predict custom SMS text directly from the terminal:

**Single Message Prediction:**
```bash
python predict.py "URGENT! You have won a 1 week free holiday to Spain. Call 09061104000 now!"
```
*Output:*
```text
==================================================
SMS Input : "URGENT! You have won a 1 week free holiday to Spain. Call 09061104000 now!"
Prediction: SPAM
Confidence: 99.95%
==================================================
```

**Interactive Mode:**
Run without arguments to enter interactive prediction mode:
```bash
python predict.py
```

---

## 🔮 Future Enhancements
- Deploy as a REST API using **FastAPI** or **Flask**.
- Build an interactive Web UI using **Streamlit** or **Gradio**.
- Explore Transformer-based language models (e.g., `BERT` or `RoBERTa` fine-tuning).

---

## 📄 License
This project is open-source and licensed under the [MIT License](LICENSE).
