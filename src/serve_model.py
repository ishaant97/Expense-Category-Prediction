# src/serve_model.py
from flask import Flask, request, jsonify
import tensorflow as tf
import joblib
import numpy as np
import os

# -----------------------------
# 1. Load model, label encoder, and vectorizer
# -----------------------------
MODEL_PATH = os.path.join("../models", "expense_mlp_model.keras")
ENCODER_PATH = os.path.join("../models", "label_encoder.joblib")
VECTORIZER_PATH = os.path.join("../models", "vectorizer.joblib")

print("🔄 Loading model, encoder, and vectorizer...")
model = tf.keras.models.load_model(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
print("✅ Model, encoder, and vectorizer loaded successfully!")

# -----------------------------
# 2. Create Flask app
# -----------------------------
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "<h2>Expense Category Prediction API is Running 🚀</h2>"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Missing 'text' field in request body"}), 400

        text = data["text"].strip()
        if not text:
            return jsonify({"error": "Empty text provided"}), 400

        # Transform input using loaded vectorizer
        X = vectorizer.transform([text]).toarray()

        # Predict category probabilities
        preds = model.predict(X)
        idx = int(np.argmax(preds))
        category = label_encoder.inverse_transform([idx])[0]
        confidence = float(preds[0][idx])

        return jsonify({
            "text": text,
            "predicted_category": category,
            "confidence": round(confidence, 3)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # host='0.0.0.0' allows external access (important for Render deployment)
    app.run(host="0.0.0.0", port=5000, debug=True)