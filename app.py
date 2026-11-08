"""Interactive evaluation interface for the saved Assignment 2 models."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from train_models import RANDOM_STATE, TARGET, TEST_SIZE, prepare_features


ROOT = Path(__file__).resolve().parent
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knearest_neighbors.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}


@st.cache_resource
def load_bundle(filename: str):
    return joblib.load(ROOT / "model" / filename)


st.set_page_config(page_title="Wikipedia Editor Gender Classifier", page_icon="📊", layout="wide")
st.title("Wikipedia Editor Gender Classification")
st.write("Evaluate one of the five saved classifiers on a labeled CSV. The app does not retrain models on uploaded data.")
st.caption("Target classes: 0 = unknown, 1 = male, 2 = female. Metrics use macro averaging for class-sensitive scores.")

uploaded = st.file_uploader("Upload data.csv or the prepared test_data.csv", type=["csv"])
model_name = st.selectbox("Choose a classifier", list(MODEL_FILES))

if uploaded is not None:
    try:
        data = pd.read_csv(uploaded)
        bundle = load_bundle(MODEL_FILES[model_name])
        target = bundle["target"]
        expected = bundle["feature_columns"]
        if data.empty:
            st.error("The uploaded CSV has no rows.")
        elif target not in data.columns:
            st.error(f"The uploaded CSV must include the target column '{target}' so the app can calculate evaluation metrics.")
        else:
            # Full data.csv contains raw timestamp columns and the two alternative
            # gender labels. Recreate the same fixed stratified holdout used during
            # training so the app evaluates only the reserved test rows.
            if "firstDay" in data.columns or "lastDay" in data.columns:
                y_all = data[target].astype(int)
                X_all = prepare_features(data)
                _, X, _, y = train_test_split(
                    X_all, y_all, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_all
                )
                st.info(f"Loaded the full dataset and selected its fixed stratified {TEST_SIZE:.0%} test split ({len(X)} rows).")
            else:
                missing = [column for column in expected if column not in data.columns]
                if missing:
                    st.error("CSV is missing required model input columns: " + ", ".join(missing))
                    st.stop()
                X = data[expected]
                y = data[target].astype(int)

            if not set(y.unique()).issubset(set(bundle["labels"])):
                st.error("The target column contains class values not represented by the saved model.")
            else:
                estimator = bundle["model"]
                prediction = estimator.predict(X)
                probability = estimator.predict_proba(X)
                labels = bundle["labels"]
                scores = {
                    "Accuracy": accuracy_score(y, prediction),
                    "AUC (OvR macro)": roc_auc_score(y, probability, multi_class="ovr", average="macro", labels=labels),
                    "Precision (macro)": precision_score(y, prediction, average="macro", zero_division=0),
                    "Recall (macro)": recall_score(y, prediction, average="macro", zero_division=0),
                    "F1 (macro)": f1_score(y, prediction, average="macro", zero_division=0),
                    "MCC": matthews_corrcoef(y, prediction),
                }
                cells = st.columns(len(scores))
                for cell, (label, value) in zip(cells, scores.items()):
                    cell.metric(label, f"{value:.4f}")
                st.subheader("Confusion matrix")
                st.dataframe(pd.DataFrame(confusion_matrix(y, prediction, labels=labels), index=labels, columns=labels), use_container_width=True)
                st.subheader("Classification report")
                report = classification_report(y, prediction, labels=labels, zero_division=0, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)
                st.subheader("Predictions preview")
                preview = X.copy()
                preview[target] = y.to_numpy()
                preview["predicted_gender"] = prediction
                st.dataframe(preview.head(20), use_container_width=True)
    except Exception as exc:
        st.error(f"Could not evaluate this CSV: {exc}")
else:
    st.info("Upload data.csv to have the app select the same held-out split, or upload test_data.csv directly.")
