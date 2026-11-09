"""Train and evaluate the three requested career classifiers."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import math
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from .data_preprocessing import load_dataset, ROOT
from .feature_extraction import build_vectorizer

MODEL_DIR = ROOT / "models"

def train_models(data_path=None, save=True):
    df = load_dataset(data_path) if data_path else load_dataset()
    X, y = df.skills_text, df.career
    counts = y.value_counts()
    test_count = math.ceil(len(y) * 0.25)
    stratify = y if counts.min() >= 2 and test_count >= y.nunique() and (len(y) - test_count) >= y.nunique() else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=stratify)
    estimators = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced_subsample"),
        "Support Vector Machine": LinearSVC(class_weight="balanced", random_state=42),
    }
    results, fitted = {}, {}
    for name, estimator in estimators.items():
        pipe = Pipeline([("tfidf", build_vectorizer()), ("classifier", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        results[name] = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_test, pred, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, pred, labels=sorted(y.unique())).tolist(),
            "labels": sorted(y.unique()),
            "report": classification_report(y_test, pred, output_dict=True, zero_division=0),
        }
        fitted[name] = pipe
    best_name = max(results, key=lambda n: results[n]["f1"])
    bundle = {"model": fitted[best_name], "model_name": best_name, "metrics": results,
              "career_profiles": pd.DataFrame({"career": df["career"], "skills_text": df["skills_list"].apply(lambda values: ",".join(values))}).to_dict("records"),
              "dataset_rows": int(len(df)), "career_count": int(y.nunique())}
    if save:
        MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump(bundle, MODEL_DIR / "skillsync_bundle.joblib")
        with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as f:
            json.dump({k: {m: v for m, v in val.items() if m not in ("confusion_matrix", "report")} for k, val in results.items()}, f, indent=2)
    return bundle

def load_or_train(force=False):
    path = MODEL_DIR / "skillsync_bundle.joblib"
    if path.exists() and not force:
        try:
            return joblib.load(path)
        except Exception:
            pass
    return train_models()
