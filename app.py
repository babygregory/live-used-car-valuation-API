import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from joblib import load

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")

MODEL_PATH = os.path.join(ARTIFACT_DIR, "model.pkl")
PREPROCESSOR_PATH = os.path.join(ARTIFACT_DIR, "preprocessor.pkl")

app = Flask(__name__, template_folder="templates", static_folder="static")

# -----------------------------
# Load model artifacts once
# -----------------------------
model = load(MODEL_PATH)
preprocessor = load(PREPROCESSOR_PATH)


def safe_float(value, default=None):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default=None):
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def build_features(payload: dict) -> pd.DataFrame:
    """
    Build the exact training-time feature schema:
    numeric_features = ["car_age", "log_mileage", "engine_cc"]
    categorical_features = ["make_model", "Variant", "Transm", "Color"]
    """
    current_year = datetime.now().year

    make = str(payload.get("make", "")).strip()
    model_name = str(payload.get("model", "")).strip()

    variant = payload.get("variant")
    transm = payload.get("transm")
    color = payload.get("color")

    year = safe_int(payload.get("year"), default=current_year)
    mileage = safe_float(payload.get("mileage"), default=0.0)
    engine_cc = safe_float(payload.get("engine_cc"), default=np.nan)

    # Basic guardrails
    year = max(1980, min(current_year, year))
    mileage = max(0.0, mileage)

    car_age = max(0, current_year - year)
    log_mileage = np.log1p(mileage)
    make_model = f"{make} {model_name}".strip()

    row = {
        "car_age": car_age,
        "log_mileage": log_mileage,
        "engine_cc": engine_cc,
        "make_model": make_model if make_model else np.nan,
        "Variant": variant if variant not in ("", None) else np.nan,
        "Transm": transm if transm not in ("", None) else np.nan,
        "Color": color if color not in ("", None) else np.nan,
    }

    return pd.DataFrame([row])


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)

        make = str(payload.get("make", "")).strip()
        model_name = str(payload.get("model", "")).strip()

        if not make or not model_name:
            return jsonify({
                "error": "Both 'make' and 'model' are required."
            }), 400

        features_df = build_features(payload)
        transformed = preprocessor.transform(features_df)

        pred_log = model.predict(transformed)[0]
        pred_price = float(np.expm1(pred_log))

        return jsonify({
            "predicted_price": round(pred_price, 2),
            "currency": "RM",
            "features_used": {
                "make": make,
                "model": model_name,
                "variant": payload.get("variant"),
                "year": payload.get("year"),
                "mileage": payload.get("mileage"),
                "engine_cc": payload.get("engine_cc"),
                "transm": payload.get("transm"),
                "color": payload.get("color")
            }
        })

    except Exception as e:
        return jsonify({
            "error": "Prediction failed",
            "details": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True)