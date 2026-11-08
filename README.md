# Gender Classification App

## Problem statement

Compare five classification algorithms on the Gender Gap in Spanish Wikipedia dataset and provide an interactive app for evaluating saved models on a labeled CSV. The app accepts the original `data.csv` and selects its fixed held-out split, or the prepared `test_data.csv` directly; it never retrains models on uploaded data.

## Dataset description

The dataset is **Gender Gap in Spanish WP**, published in the [UCI Machine Learning Repository](https://www.ics.uci.edu/dataset/852/gender%2Bgap%2Bin%2Bspanish%2Bwp), DOI [10.24432/C5XP52](https://doi.org/10.24432/C5XP52). It contains 4,746 editor records. The target is `gender` (0 = unknown, 1 = male, 2 = female). The alternative label fields `C_api` and `C_man` are excluded from the predictors to prevent label leakage. The dataset has 18 remaining candidate input columns; `firstDay` and `lastDay` are converted into year, month, day, and hour features.

The data has no missing values or duplicate rows. The classes are imbalanced, so the evaluation reports macro-averaged precision, recall, F1, and one-vs-rest macro AUC in addition to accuracy and MCC. The split is stratified, with a fixed random seed for reproducibility.

## GitHub Repository Link
https://github.com/amoldeokate-07/gender-gap-classification-app

## Streamlit app link
https://hbcpaatjxo8etp76dvclek.streamlit.app/

## Models used

Run `python train_models.py` to train the five models, save their pipelines in `model/`, generate `test_data.csv`, and write the evaluation results to `metrics.json`.

| ML Model Name | Accuracy | AUC (OvR macro) | Precision (macro) | Recall (macro) | F1 (macro) | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.4568 | 0.6348 | 0.4366 | 0.4877 | 0.4119 | 0.1747 |
| Decision Tree | 0.5200 | 0.5577 | 0.4099 | 0.4164 | 0.4122 | 0.1014 |
| K-Nearest Neighbors | 0.5537 | 0.5872 | 0.4271 | 0.3824 | 0.3855 | 0.0849 |
| Gaussian Naive Bayes | 0.3758 | 0.6519 | 0.5678 | 0.3925 | 0.2996 | 0.0946 |
| Support Vector Machine | 0.4768 | 0.6449 | 0.4230 | 0.4581 | 0.4117 | 0.1550 |

### Observations on model performance

The Decision Tree has the highest macro F1 (0.4122), narrowly ahead of Logistic Regression (0.4119) and SVM (0.4117). KNN has the highest accuracy (0.5537), but its macro F1 is lower (0.3855), showing that its performance is less even across classes. Gaussian Naive Bayes has the highest macro AUC (0.6519) and macro precision (0.5678), but the lowest accuracy and macro F1. Logistic Regression has the highest MCC (0.1747), though all MCC scores are modest. The held-out set has class counts 320, 559, and 71 respectively, so the minority female class has much less support; review the per-class report before selecting a single winner. By macro F1, Decision Tree is the leading model in this run.

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong balanced class-sensitive scores; best MCC and second-highest macro F1 in this run. |
| Decision Tree | Best macro F1, with moderate accuracy and lower MCC/AUC than the top alternatives. |
| K-Nearest Neighbors | Highest accuracy, but lower macro recall/F1; majority-class performance dominates. |
| Gaussian Naive Bayes | Highest macro AUC and precision, but lowest accuracy and macro F1. |
| Support Vector Machine | Competitive macro AUC, recall, and F1; slightly behind Logistic Regression and Decision Tree on macro F1. |
| Overall winner | Decision Tree by macro F1 (0.4122); Logistic Regression has the highest MCC (0.1747). |

## Run locally

```bash
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```
