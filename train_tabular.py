"""
Tabular Dataset: Breast Cancer Wisconsin
Binary Classification: Malignant (M=1) vs Benign (B=0)
Dataset: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
Place 'data.csv' inside datasets/tabular/
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
import joblib
import json
import os

DATASET_PATH = "datasets/tabular/data.csv"
MODEL_PATH   = "models/tabular_model.pkl"
SCALER_PATH  = "models/tabular_scaler.pkl"
META_PATH    = "models/tabular_meta.json"


def train():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found at '{DATASET_PATH}'.\n"
            "Download from Kaggle: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data\n"
            "and place 'data.csv' in datasets/tabular/"
        )

    df = pd.read_csv(DATASET_PATH)

    # Drop ID and unnamed last column if present
    df.drop(columns=["id"], errors="ignore", inplace=True)
    df.drop(columns=[c for c in df.columns if "Unnamed" in c], inplace=True)

    # Encode target: M -> 1, B -> 0
    df["diagnosis"] = (df["diagnosis"] == "M").astype(int)

    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]

    feature_names = list(X.columns)

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Logistic Regression
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_sc, y_train)

    y_pred  = model.predict(X_test_sc)
    y_proba = model.predict_proba(X_test_sc)[:, 1]

    acc    = accuracy_score(y_test, y_pred)
    roc    = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred,
                                   target_names=["Benign", "Malignant"],
                                   output_dict=True)
    cm     = confusion_matrix(y_test, y_pred).tolist()

    # Save artefacts
    joblib.dump(model,  MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    meta = {
        "accuracy": round(acc, 4),
        "roc_auc":  round(roc, 4),
        "report":   report,
        "confusion_matrix": cm,
        "feature_names": feature_names,
        "classes": ["Benign (B)", "Malignant (M)"],
        "train_samples": len(X_train),
        "test_samples":  len(X_test),
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Accuracy : {acc:.4f}")
    print(f"ROC-AUC  : {roc:.4f}")
    print("Model saved to", MODEL_PATH)
    return meta


def predict_single(feature_values: list):
    """Predict for one sample given a list of feature values."""
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)

    x = np.array(feature_values, dtype=float).reshape(1, -1)
    x_sc = scaler.transform(x)
    pred  = model.predict(x_sc)[0]
    proba = model.predict_proba(x_sc)[0]

    return {
        "prediction": int(pred),
        "label": meta["classes"][pred],
        "probability_benign":    round(float(proba[0]), 4),
        "probability_malignant": round(float(proba[1]), 4),
    }


if __name__ == "__main__":
    train()
