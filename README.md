# SMS Spam Classification with a Linear SVM

This project trains a text classifier to label SMS messages as **ham** (not spam) or **spam**. It also compares model performance across several ham-to-spam training ratios.

## How it works

1. `train.py` reads `spam.csv` if it exists; otherwise it reads `SMS.csv`.
2. It removes duplicate rows, cleans the message text, and encodes ham as `0` and spam as `1`.
3. It makes one stratified 80% training / 20% test split (`random_state=42`). The same held-out test set is used for every ratio experiment.
4. It compares the original training distribution with five sampled training sets: `3600:1200`, `3600:1800`, `3600:2400`, `3600:3000`, and `3600:3600` (ham:spam).
5. Each training set is converted to TF-IDF features and used to train a Linear SVM. Five-fold cross-validation selects the best `C` from `0.1`, `1`, and `10`.
6. It evaluates each selected model on the held-out test set, writes `spam_ratio_experiment_report.txt`, and saves the model from the `3600:3600` experiment as `sms_spam_pipeline.pkl`.

### Important note about oversampling

The training split contains 3,613 ham and 522 spam messages. In the fixed-ratio experiments, ham is sampled without replacement and spam is sampled **with replacement**. Thus, for example, `3600:3600` contains 3,600 spam training entries, but not 3,600 distinct spam messages: some of the 522 available spam examples are repeated. This does not add new or unique messages to the CSV.

Because the repeated spam examples are created before five-fold cross-validation, copies of a message can occur in different folds. The cross-validation scores may therefore be optimistic. The held-out test scores are the more useful figures for comparing these experiments.

## Results

Results below are from the current `spam_ratio_experiment_report.txt`. All rows use the same test set of 1,034 messages (903 ham and 131 spam).

| Training ratio (ham:spam) | Test accuracy | Spam precision | Spam recall | Spam F1 |
| --- | ---: | ---: | ---: | ---: |
| Original distribution (3613:522) | **98.16%** | 95.90% | 89.31% | **92.49%** |
| 3600:1200 | 97.97% | 93.65% | 90.08% | 91.83% |
| 3600:1800 | 97.97% | 94.35% | 89.31% | 91.76% |
| 3600:2400 | 98.07% | 93.02% | **91.60%** | 92.31% |
| 3600:3000 | 97.68% | 92.13% | 89.31% | 90.70% |
| 3600:3600 | 97.68% | 92.13% | 89.31% | 90.70% |

The original-distribution model achieved the highest test accuracy and F1 in this run. The `3600:2400` model identified the largest share of spam messages (highest spam recall). Increasing the number of oversampled spam entries did not consistently improve test performance.

The saved `sms_spam_pipeline.pkl` is specifically the model trained with the `3600:3600` ratio; it is not the best-scoring model from the comparison. To change which experiment is saved, update the final-model selection in `train.py`.

## Run the project

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the models and regenerate the report:

```bash
python train.py
```

If neither dataset CSV is available, run the downloader first:

```bash
python download_data.py
```

Predict an SMS message:

```bash
python predict.py "Congratulations, you have won a prize"
```

Running `python predict.py` without a message starts an interactive prompt. If the saved model file is missing, the predictor trains the models first.

## Project files

- `train.py` — text preprocessing, ratio experiments, evaluation, report generation, and model saving.
- `predict.py` — loads the saved model and predicts whether a message is ham or spam.
- `download_data.py` — downloads the SMS dataset and saves it as `spam.csv` and `SMS.csv`.
- `spam_ratio_experiment_report.txt` — detailed metrics and comparison from the latest training run.
- `sms_spam_pipeline.pkl` — saved model used by the predictor.
