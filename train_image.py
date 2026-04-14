"""
Image Dataset: Chest X-Ray Pneumonia
Binary Classification: NORMAL (0) vs PNEUMONIA (1)
Dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
Place images in:
  datasets/images/train/NORMAL/
  datasets/images/train/PNEUMONIA/
  datasets/images/test/NORMAL/
  datasets/images/test/PNEUMONIA/
"""

import os
import json
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
import joblib

IMG_SIZE   = (64, 64)          # resize all images to this
TRAIN_DIR  = "datasets/images/chest_xray/chest_xray/train"
TEST_DIR   = "datasets/images/chest_xray/chest_xray/test"
MODEL_PATH = "models/image_model.pkl"
SCALER_PATH= "models/image_scaler.pkl"
META_PATH  = "models/image_meta.json"
CLASSES    = ["NORMAL", "PNEUMONIA"]   # index = label


def load_images(directory, max_per_class=500):
    """Load images from directory/CLASSNAME/ folders, flatten to 1-D vectors."""
    X, y = [], []
    for label, cls in enumerate(CLASSES):
        cls_dir = os.path.join(directory, cls)
        if not os.path.isdir(cls_dir):
            print(f"Warning: {cls_dir} not found, skipping.")
            continue
        files = [f for f in os.listdir(cls_dir)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        files = files[:max_per_class]
        print(f"  Loading {len(files)} images from {cls_dir}")
        for fname in files:
            try:
                img = Image.open(os.path.join(cls_dir, fname)).convert("L")
                img = img.resize(IMG_SIZE)
                X.append(np.array(img, dtype=np.float32).flatten())
                y.append(label)
            except Exception as e:
                print(f"  Skipped {fname}: {e}")
    return np.array(X), np.array(y)


def train():
    if not os.path.isdir(TRAIN_DIR):
        raise FileNotFoundError(
            f"Training directory '{TRAIN_DIR}' not found.\n"
            "Download from Kaggle: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia\n"
            "and organise into datasets/images/train/ and datasets/images/test/"
        )

    print("Loading training images...")
    X_train, y_train = load_images(TRAIN_DIR, max_per_class=500)
    print("Loading test images...")
    X_test,  y_test  = load_images(TEST_DIR,  max_per_class=200)

    if len(X_train) == 0:
        raise ValueError("No training images found. Check your dataset directory.")

    # Normalize pixel values to [0,1]
    X_train = X_train / 255.0
    X_test  = X_test  / 255.0

    # Scale (zero-mean, unit variance per pixel)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print("Training Logistic Regression...")
    model = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42, C=0.1)
    model.fit(X_train_sc, y_train)

    y_pred  = model.predict(X_test_sc)
    y_proba = model.predict_proba(X_test_sc)[:, 1]

    acc    = accuracy_score(y_test, y_pred)
    roc    = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred,
                                   target_names=CLASSES,
                                   output_dict=True)
    cm     = confusion_matrix(y_test, y_pred).tolist()

    joblib.dump(model,  MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    meta = {
        "accuracy": round(acc, 4),
        "roc_auc":  round(roc, 4),
        "report":   report,
        "confusion_matrix": cm,
        "classes": CLASSES,
        "img_size": list(IMG_SIZE),
        "train_samples": len(X_train),
        "test_samples":  len(X_test),
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Accuracy : {acc:.4f}")
    print(f"ROC-AUC  : {roc:.4f}")
    print("Model saved to", MODEL_PATH)
    return meta


def predict_image_file(filepath: str):
    """Predict class of a single image file."""
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)

    img_size = tuple(meta["img_size"])
    img = Image.open(filepath).convert("L").resize(img_size)
    x   = np.array(img, dtype=np.float32).flatten() / 255.0
    x_sc = scaler.transform(x.reshape(1, -1))

    pred  = model.predict(x_sc)[0]
    proba = model.predict_proba(x_sc)[0]

    return {
        "prediction": int(pred),
        "label": meta["classes"][pred],
        "probability_normal":    round(float(proba[0]), 4),
        "probability_pneumonia": round(float(proba[1]), 4),
    }


if __name__ == "__main__":
    train()
