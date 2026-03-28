"""
Smart Orchard API — thin Flask layer; logic lives in sibling modules.
"""

from __future__ import annotations

import os
from io import BytesIO

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

from config import CLASS_NAMES, CONFIDENCE_THRESHOLD_PCT, MODEL_PATH
from explanations import estimate_severity, get_explanation, get_treatment
from prediction import ensure_model_loaded, get_model, predict_image
from preprocessing import preprocess_image
from validation import validate_leaf

app = Flask(__name__)
CORS(app)


@app.route("/health", methods=["GET"])
def health():
    ensure_model_loaded()
    return jsonify(
        {
            "status": "ok",
            "model_loaded": get_model() is not None,
            "model_path": str(MODEL_PATH),
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Missing file. Send multipart field 'file'."}), 400

    file = request.files["file"]
    if file is None or file.filename == "":
        return jsonify({"error": "No file selected or empty filename."}), 400

    try:
        raw = file.read()
        if not raw:
            return jsonify({"error": "Empty file body."}), 400
        pil_image = Image.open(BytesIO(raw))
    except Exception as e:
        return jsonify({"error": f"Could not read image: {e}"}), 400

    try:
        batch = preprocess_image(pil_image)
    except Exception as e:
        return jsonify({"error": f"Could not preprocess image: {e}"}), 400

    ok_leaf, leaf_err = validate_leaf(batch)
    if not ok_leaf:
        return jsonify({"status": "invalid", "message": leaf_err})

    try:
        pred = predict_image(batch)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    if not np.isfinite(pred["confidence"]):
        return jsonify(
            {
                "status": "uncertain",
                "message": "Model is not confident. Please upload a clear apple leaf image.",
            }
        )

    if pred["confidence"] < CONFIDENCE_THRESHOLD_PCT:
        return jsonify(
            {
                "status": "uncertain",
                "message": "Model is not confident. Please upload a clear apple leaf image.",
            }
        )

    idx = max(0, min(pred["class_index"], len(CLASS_NAMES) - 1))
    label = CLASS_NAMES[idx]

    return jsonify(
        {
            "status": "ok",
            "disease": label,
            "confidence": round(float(pred["confidence"]), 2),
            "top2": pred["top2"],
            "explanation": get_explanation(label),
            "treatment": get_treatment(label),
            "severity": estimate_severity(label, batch),
        }
    )


if __name__ == "__main__":
    try:
        ensure_model_loaded()
    except Exception as e:
        print(f"Startup model load note: {e}")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
