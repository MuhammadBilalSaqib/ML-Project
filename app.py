"""
Flask Application — Assignment 01
Logistic Regression: Tabular + Image Binary Classification
"""

import os
import json
import uuid
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "assignment01-secret-key"

UPLOAD_FOLDER   = "static/uploads"
ALLOWED_IMAGES  = {"png", "jpg", "jpeg"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_IMAGES


def model_exists(model_path):
    return os.path.exists(model_path)


# ── Home ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    tabular_trained = model_exists("models/tabular_model.pkl")
    image_trained   = model_exists("models/image_model.pkl")
    return render_template("index.html",
                           tabular_trained=tabular_trained,
                           image_trained=image_trained)


# ── Tabular ───────────────────────────────────────────────────────────────────

@app.route("/tabular")
def tabular():
    meta = None
    if model_exists("models/tabular_meta.json"):
        with open("models/tabular_meta.json") as f:
            meta = json.load(f)
    feature_names = meta["feature_names"] if meta else []
    return render_template("tabular.html", meta=meta, feature_names=feature_names)


@app.route("/tabular/train", methods=["POST"])
def tabular_train():
    try:
        from train_tabular import train
        meta = train()
        flash(f"Tabular model trained! Accuracy: {meta['accuracy']*100:.2f}%", "success")
    except FileNotFoundError as e:
        flash(str(e), "error")
    except Exception as e:
        flash(f"Training failed: {e}", "error")
    return redirect(url_for("tabular"))


@app.route("/tabular/predict", methods=["POST"])
def tabular_predict():
    if not model_exists("models/tabular_model.pkl"):
        return jsonify({"error": "Model not trained yet. Train the model first."}), 400

    with open("models/tabular_meta.json") as f:
        meta = json.load(f)

    feature_names = meta["feature_names"]
    try:
        values = [float(request.form.get(f, 0)) for f in feature_names]
    except ValueError:
        return jsonify({"error": "Invalid input values."}), 400

    from train_tabular import predict_single
    result = predict_single(values)
    return jsonify(result)


# ── Image ─────────────────────────────────────────────────────────────────────

@app.route("/image")
def image():
    meta = None
    if model_exists("models/image_meta.json"):
        with open("models/image_meta.json") as f:
            meta = json.load(f)
    return render_template("image.html", meta=meta)


@app.route("/image/train", methods=["POST"])
def image_train():
    try:
        from train_image import train
        meta = train()
        flash(f"Image model trained! Accuracy: {meta['accuracy']*100:.2f}%", "success")
    except FileNotFoundError as e:
        flash(str(e), "error")
    except Exception as e:
        flash(f"Training failed: {e}", "error")
    return redirect(url_for("image"))


@app.route("/image/predict", methods=["POST"])
def image_predict():
    if not model_exists("models/image_model.pkl"):
        return jsonify({"error": "Model not trained yet."}), 400

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file. Upload a PNG/JPG image."}), 400

    filename = str(uuid.uuid4()) + "_" + secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        from train_image import predict_image_file
        result = predict_image_file(filepath)
        result["image_url"] = "/" + filepath.replace("\\", "/")
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    app.run(debug=True, port=5000)
