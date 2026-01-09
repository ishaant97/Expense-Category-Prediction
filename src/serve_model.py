from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import joblib
import numpy as np
import os
import re

import nltk

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading NLTK data...")
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    print("NLTK data downloaded")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


IMPORTANT_WORDS = {'rent', 'food', 'doctor', 'hospital', 'gym', 'movie', 'bus',
                   'train', 'uber', 'ola', 'electricity', 'water', 'bill', 'grocery',
                   'college', 'school', 'tuition', 'fuel', 'cafe', 'restaurant',
                   'netflix', 'spotify', 'prime', 'metro', 'cab', 'auto'}

stop_words = set(stopwords.words("english")) - IMPORTANT_WORDS
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """Enhanced text cleaning that preserves important category indicators"""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    words = text.split()
    
    words = [lemmatizer.lemmatize(word) for word in words 
             if word not in stop_words or len(word) > 2]
    return " ".join(words)


CONFIDENCE_THRESHOLD = 0.5
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

MODEL_PATH = os.path.join("../models", "expense_mlp.keras")
ENCODER_PATH = os.path.join("../models", "label_encoder.joblib")
VECTORIZER_PATH = os.path.join("../models", "vectorizer.joblib")

print("=" * 60)
print("🔄 Loading Expense Categorization System...")
print("=" * 60)


DL_MODEL_AVAILABLE = False
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    
    print(f"✅ DL Model loaded: {MODEL_PATH}")
    print(f"✅ Label encoder loaded: {ENCODER_PATH}")
    print(f"✅ Vectorizer loaded: {VECTORIZER_PATH}")
    print(f"\n DL Model Info:")
    print(f"   Categories: {list(label_encoder.classes_)}")
    print(f"   Features: {len(vectorizer.get_feature_names_out())}")
    print(f"   Confidence threshold: {CONFIDENCE_THRESHOLD}")
    DL_MODEL_AVAILABLE = True
except Exception as e:
    print(f"Error loading DL model: {e}")
    print("Please ensure you've trained the model first:")
    print("  python mlp_train.py")



print("=" * 60)
print(f"🎯 Available Models:")
print(f"   - Deep Learning: {'✅ Yes (93.33% accuracy)' if DL_MODEL_AVAILABLE else '❌ No'}")
print("=" * 60)
print("✅ API Ready to serve predictions!")
print("=" * 60)


app = Flask(__name__)

CORS(app)


def predict_with_dl(text):
    """Predict using Deep Learning model"""
    if not DL_MODEL_AVAILABLE:
        return {"error": "DL model not available"}
    
    cleaned_text = clean_text(text)
    if not cleaned_text:
        return {"error": "Text preprocessing resulted in empty string"}
    
    X = vectorizer.transform([cleaned_text]).toarray()
    preds = model.predict(X, verbose=0)
    
    idx = int(np.argmax(preds[0]))
    confidence = float(preds[0][idx])
    category = label_encoder.inverse_transform([idx])[0]
    

    is_uncertain = confidence < CONFIDENCE_THRESHOLD
    if is_uncertain:
        original_prediction = category
        category = "Miscellaneous"
    else:
        original_prediction = None
    
    return {
        "category": category,
        "confidence": round(confidence, 4),
        "is_uncertain": is_uncertain,
        "original_prediction": original_prediction,
        "model": "deep_learning"
    }


@app.route("/", methods=["GET"])
def home():
    """Home endpoint with API information"""
    return jsonify({
        "message": "🚀 Expense Category Prediction API",
        "version": "3.0 - Hybrid (DL + Gemini AI)",
        "dl_model_accuracy": "93.33%",
        "categories": list(label_encoder.classes_) if DL_MODEL_AVAILABLE else [],
        "available_models": {
            "deep_learning": DL_MODEL_AVAILABLE,
        },
        "endpoints": {
            "POST /predict": "Predict using DL model (default, fast)",
            "POST /predict/gemini": "Predict using Gemini AI (requires API key)",
            "POST /predict/hybrid": "Predict using BOTH models + comparison",
            "POST /predict/auto": "Auto-select best model (DL first, Gemini fallback)",
            "GET /config": "Get current configuration",
            "POST /config": "Update confidence threshold",
            "GET /health": "Check API health"
        },
        "example_request": {
            "url": "/predict",
            "method": "POST",
            "body": {
                "text": "pizza from dominos"
            }
        }
    })

@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict expense category using Deep Learning model (default)
    
    Request body:
        {
            "text": "expense description",
            "include_all_predictions": false  // optional
        }
    
    Response:
        {
            "success": true,
            "text": "original text",
            "predicted_category": "category name",
            "confidence": 0.95,
            "is_uncertain": false,
            "model_used": "deep_learning"
        }
    """
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({
                "error": "Missing 'text' field in request body",
                "example": {"text": "pizza from dominos"}
            }), 400

        raw_text = data["text"].strip()
        if not raw_text:
            return jsonify({"error": "Empty text provided"}), 400

        # Use DL model
        result = predict_with_dl(raw_text)
        
        if "error" in result:
            return jsonify(result), 500
        
        response = {
            "success": True,
            "text": raw_text,
            "predicted_category": result["category"],
            "confidence": result["confidence"],
            "is_uncertain": result.get("is_uncertain", False),
            "model_used": "deep_learning"
        }
        
        if result.get("original_prediction"):
            response["original_prediction"] = result["original_prediction"]
            response["message"] = f"Low confidence. Suggested: {result['original_prediction']}, marked as Miscellaneous."
        
        # Include all predictions if requested
        if data.get("include_all_predictions", False) and DL_MODEL_AVAILABLE:
            cleaned_text = clean_text(raw_text)
            X = vectorizer.transform([cleaned_text]).toarray()
            preds = model.predict(X, verbose=0)
            all_predictions = {
                label_encoder.inverse_transform([i])[0]: round(float(preds[0][i]), 4)
                for i in range(len(preds[0]))
            }
            response["all_predictions"] = dict(sorted(all_predictions.items(), 
                                                     key=lambda x: x[1], reverse=True))
        
        return jsonify(response)

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }), 500


@app.route("/health", methods=["GET"])
def health():
    """API health check endpoint"""
    try:
        dl_status = "not available"
        if DL_MODEL_AVAILABLE:
            try:
                test_text = "test"
                cleaned = clean_text(test_text)
                X = vectorizer.transform([cleaned]).toarray()
                _ = model.predict(X, verbose=0)
                dl_status = "healthy"
            except Exception as e:
                dl_status = f"error: {str(e)}"
        
        return jsonify({
            "status": "healthy",
            "api_version": "3.0 - Hybrid",
            "models": {
                "deep_learning": {
                    "available": DL_MODEL_AVAILABLE,
                    "status": dl_status,
                    "accuracy": "93.33%" if DL_MODEL_AVAILABLE else None
                }
            },
            "endpoints_available": {
                "/predict": DL_MODEL_AVAILABLE,
            }
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 Starting Flask API Server...")
    print("=" * 60)
    
    port = int(os.environ.get("PORT", 5000))
    
    print("Server will be available at:")
    print(f"   - Local: http://127.0.0.1:{port}")
    print(f"   - Network: http://0.0.0.0:{port}")
    print("\n Available endpoints:")
    print("   GET  /              - API info")
    print("   POST /predict       - Predict single expense")
    print("   GET  /config        - Get configuration")
    print("   POST /config        - Update configuration")
    print("   GET  /health        - Health check")
    print("=" * 60)
    print("\n Test with:")
    print(f'   curl -X POST http://localhost:{port}/predict \\')
    print('        -H "Content-Type: application/json" \\')
    print('        -d \'{"text": "pizza from dominos"}\'')
    print("\n" + "=" * 60 + "\n")
    
    # For production deployment, disable debug mode
    debug_mode = os.environ.get("FLASK_ENV") != "production"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)