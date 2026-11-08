"""Train and save the five classifiers required for Assignment 2."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data.csv"
TARGET = "gender"
DROP_COLUMNS = {TARGET, "C_api", "C_man"}
RANDOM_STATE = 42
TEST_SIZE = 0.20


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert compact timestamp columns to numeric calendar features."""
    X = frame.drop(columns=[c for c in DROP_COLUMNS if c in frame.columns]).copy()
    for col in ("firstDay", "lastDay"):
        if col in X:
            dates = pd.to_datetime(X.pop(col).astype("Int64").astype(str), format="%Y%m%d%H%M%S", errors="coerce")
            X[f"{col}_year"] = dates.dt.year
            X[f"{col}_month"] = dates.dt.month
            X[f"{col}_day"] = dates.dt.day
            X[f"{col}_hour"] = dates.dt.hour
    return X


def main() -> None:
    data = pd.read_csv(SOURCE)
    y = data[TARGET].astype(int)
    X = prepare_features(data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        "K-Nearest Neighbors": Pipeline([
            ("scale", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=7, weights="distance", n_jobs=-1)),
        ]),
        "Gaussian Naive Bayes": Pipeline([
            ("scale", StandardScaler()),
            ("classifier", GaussianNB()),
        ]),
        "Support Vector Machine": Pipeline([
            ("scale", StandardScaler()),
            ("classifier", SVC(probability=True, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
    }

    model_dir = ROOT / "model"
    model_dir.mkdir(exist_ok=True)
    metric_rows = {}
    labels = sorted(y.unique().tolist())
    for name, estimator in models.items():
        estimator.fit(X_train, y_train)
        predictions = estimator.predict(X_test)
        probabilities = estimator.predict_proba(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "auc_ovr_macro": roc_auc_score(y_test, probabilities, multi_class="ovr", average="macro", labels=labels),
            "precision_macro": precision_score(y_test, predictions, average="macro", zero_division=0),
            "recall_macro": recall_score(y_test, predictions, average="macro", zero_division=0),
            "f1_macro": f1_score(y_test, predictions, average="macro", zero_division=0),
            "mcc": matthews_corrcoef(y_test, predictions),
            "labels": labels,
            "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
        }
        filename = name.lower().replace(" ", "_").replace("-", "") + ".joblib"
        joblib.dump({"model": estimator, "feature_columns": X.columns.tolist(), "labels": labels, "target": TARGET}, model_dir / filename)
        metric_rows[name] = metrics
        print(f"{name}: accuracy={metrics['accuracy']:.4f}, macro-F1={metrics['f1_macro']:.4f}")

    test = X_test.copy()
    test[TARGET] = y_test
    test.to_csv(ROOT / "test_data.csv", index=False)
    (ROOT / "metrics.json").write_text(json.dumps(metric_rows, indent=2))
    print(f"Saved {len(models)} models, metrics.json, and {len(test)} labeled test rows.")


if __name__ == "__main__":
    main()
