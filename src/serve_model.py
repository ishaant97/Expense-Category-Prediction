# src/serve_model.py
"""
Hybrid Expense Categorization API - Deep Learning + Gemini AI
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import joblib
import numpy as np
import os
import re

# === NLP Preprocessing Setup ===
import nltk

# Download NLTK data at runtime (for deployment environments)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("📥 Downloading NLTK data...")
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    print("✅ NLTK data downloaded")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# === Gemini Integration (Optional) ===
try:
    from gemini_predictor import GeminiExpensePredictor
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("ℹ️  Gemini module not found (optional). Install: pip install google-generativeai")

# Keep important category-indicating words
IMPORTANT_WORDS = {'rent', 'food', 'doctor', 'hospital', 'gym', 'movie', 'bus',
                   'train', 'uber', 'ola', 'electricity', 'water', 'bill', 'grocery',
                   'college', 'school', 'tuition', 'fuel', 'cafe', 'restaurant',
                   'netflix', 'spotify', 'prime', 'metro', 'cab', 'auto'}

stop_words = set(stopwords.words("english")) - IMPORTANT_WORDS
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """Enhanced text cleaning that preserves important category indicators"""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)  # remove punctuation
    words = text.split()
    # Lemmatize and keep important words even if they're stopwords
    words = [lemmatizer.lemmatize(word) for word in words 
             if word not in stop_words or len(word) > 2]
    return " ".join(words)

# === Configuration ===
CONFIDENCE_THRESHOLD = 0.5  # Predictions below this will be marked as "Miscellaneous"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Optional: for Gemini integration

# === Load DL model and artifacts ===
# Using the improved model trained with data augmentation
MODEL_PATH = os.path.join("../models", "expense_mlp_best.keras")
ENCODER_PATH = os.path.join("../models", "label_encoder_best.joblib")
VECTORIZER_PATH = os.path.join("../models", "vectorizer_best.joblib")

print("=" * 60)
print("🔄 Loading Expense Categorization System...")
print("=" * 60)

# Load Deep Learning Model
DL_MODEL_AVAILABLE = False
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    
    print(f"✅ DL Model loaded: {MODEL_PATH}")
    print(f"✅ Label encoder loaded: {ENCODER_PATH}")
    print(f"✅ Vectorizer loaded: {VECTORIZER_PATH}")
    print(f"\n📊 DL Model Info:")
    print(f"   Categories: {list(label_encoder.classes_)}")
    print(f"   Features: {len(vectorizer.get_feature_names_out())}")
    print(f"   Confidence threshold: {CONFIDENCE_THRESHOLD}")
    DL_MODEL_AVAILABLE = True
except Exception as e:
    print(f"❌ Error loading DL model: {e}")
    print("Please ensure you've trained the model first:")
    print("  python train_with_augmented_data.py")

# Initialize Gemini (if available and API key set)
gemini_predictor = None
if GEMINI_AVAILABLE and GEMINI_API_KEY:
    try:
        gemini_predictor = GeminiExpensePredictor(GEMINI_API_KEY)
        print(f"✅ Gemini AI initialized")
    except Exception as e:
        print(f"⚠️  Gemini initialization failed: {e}")
        print("   Gemini endpoints will not be available")

print("=" * 60)
print(f"🎯 Available Models:")
print(f"   - Deep Learning: {'✅ Yes (93.33% accuracy)' if DL_MODEL_AVAILABLE else '❌ No'}")
print(f"   - Gemini AI: {'✅ Yes' if gemini_predictor else '❌ No (Set GEMINI_API_KEY to enable)'}")
print("=" * 60)
print("✅ API Ready to serve predictions!")
print("=" * 60)

# === Create Flask app ===
app = Flask(__name__)

# Enable CORS for all routes (allows your website to make API calls)
CORS(app)

# === Helper Functions ===

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
    
    # Apply confidence threshold
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

def predict_with_gemini(text):
    """Predict using Gemini AI"""
    if not gemini_predictor:
        return {"error": "Gemini not available. Set GEMINI_API_KEY environment variable."}
    
    result = gemini_predictor.predict(text)
    return result

# === API Endpoints ===

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
            "gemini_ai": gemini_predictor is not None
        },
        "endpoints": {
            "POST /predict": "Predict using DL model (default, fast)",
            "POST /predict/gemini": "Predict using Gemini AI (requires API key)",
            "POST /predict/hybrid": "Predict using BOTH models + comparison",
            "POST /predict/auto": "Auto-select best model (DL first, Gemini fallback)",
            "POST /batch_predict": "Batch predictions (DL model)",
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


@app.route("/predict/gemini", methods=["POST"])
def predict_gemini():
    """
    Predict expense category using Gemini AI only
    
    Request body:
        {
            "text": "expense description"
        }
    
    Response:
        {
            "success": true,
            "text": "original text",
            "predicted_category": "category name",
            "confidence": 0.95,
            "reasoning": "explanation from Gemini",
            "model_used": "gemini_ai"
        }
    """
    try:
        if not gemini_predictor:
            return jsonify({
                "error": "Gemini AI not available",
                "message": "Set GEMINI_API_KEY environment variable to enable Gemini predictions",
                "available_models": ["deep_learning"]
            }), 503

        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({
                "error": "Missing 'text' field in request body",
                "example": {"text": "pizza from dominos"}
            }), 400

        raw_text = data["text"].strip()
        if not raw_text:
            return jsonify({"error": "Empty text provided"}), 400

        # Use Gemini
        result = predict_with_gemini(raw_text)
        
        if "error" in result:
            return jsonify(result), 500
        
        response = {
            "success": True,
            "text": raw_text,
            "predicted_category": result["category"],
            "confidence": result["confidence"],
            "reasoning": result.get("reasoning", ""),
            "model_used": "gemini_ai"
        }
        
        return jsonify(response)

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }), 500


@app.route("/predict/hybrid", methods=["POST"])
def predict_hybrid():
    """
    Predict using BOTH Deep Learning and Gemini AI models for comparison
    
    Request body:
        {
            "text": "expense description"
        }
    
    Response:
        {
            "success": true,
            "text": "original text",
            "deep_learning": {...},
            "gemini_ai": {...},
            "agreement": true/false,
            "recommended": "category name"
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

        # Get predictions from both models
        dl_result = predict_with_dl(raw_text) if DL_MODEL_AVAILABLE else {"error": "DL model not available"}
        gemini_result = predict_with_gemini(raw_text) if gemini_predictor else {"error": "Gemini not available"}
        
        # Build response
        response = {
            "success": True,
            "text": raw_text,
            "deep_learning": dl_result,
            "gemini_ai": gemini_result
        }
        
        # Check agreement
        if "error" not in dl_result and "error" not in gemini_result:
            dl_category = dl_result["category"]
            gemini_category = gemini_result["category"]
            
            response["agreement"] = (dl_category == gemini_category)
            
            # Recommend based on confidence
            if response["agreement"]:
                response["recommended"] = dl_category
                response["recommendation_reason"] = "Both models agree"
            else:
                # Choose the one with higher confidence
                if gemini_result["confidence"] > dl_result["confidence"]:
                    response["recommended"] = gemini_category
                    response["recommendation_reason"] = "Gemini has higher confidence"
                else:
                    response["recommended"] = dl_category
                    response["recommendation_reason"] = "DL model has higher confidence"
        elif "error" not in dl_result:
            response["recommended"] = dl_result["category"]
            response["recommendation_reason"] = "Only DL model available"
        elif "error" not in gemini_result:
            response["recommended"] = gemini_result["category"]
            response["recommendation_reason"] = "Only Gemini available"
        
        return jsonify(response)

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }), 500


@app.route("/predict/auto", methods=["POST"])
def predict_auto():
    """
    Auto-select best model: Try DL first (fast), fallback to Gemini if uncertain
    
    Request body:
        {
            "text": "expense description"
        }
    
    Response:
        {
            "success": true,
            "text": "original text",
            "predicted_category": "category name",
            "confidence": 0.95,
            "model_used": "deep_learning" or "gemini_ai",
            "fallback_used": true/false
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

        # Try DL first (fast)
        if DL_MODEL_AVAILABLE:
            dl_result = predict_with_dl(raw_text)
            
            # If confident, use DL result
            if "error" not in dl_result and not dl_result.get("is_uncertain", False):
                return jsonify({
                    "success": True,
                    "text": raw_text,
                    "predicted_category": dl_result["category"],
                    "confidence": dl_result["confidence"],
                    "model_used": "deep_learning",
                    "fallback_used": False,
                    "reason": "DL model confident"
                })
        
        # Fallback to Gemini if DL uncertain or unavailable
        if gemini_predictor:
            gemini_result = predict_with_gemini(raw_text)
            
            if "error" not in gemini_result:
                return jsonify({
                    "success": True,
                    "text": raw_text,
                    "predicted_category": gemini_result["category"],
                    "confidence": gemini_result["confidence"],
                    "reasoning": gemini_result.get("reasoning", ""),
                    "model_used": "gemini_ai",
                    "fallback_used": True,
                    "reason": "DL model uncertain, using Gemini"
                })
        
        # If we get here, both failed or unavailable
        if DL_MODEL_AVAILABLE and "error" not in dl_result:
            # Return DL result even if uncertain
            return jsonify({
                "success": True,
                "text": raw_text,
                "predicted_category": dl_result["category"],
                "confidence": dl_result["confidence"],
                "model_used": "deep_learning",
                "fallback_used": False,
                "reason": "Only DL model available (uncertain)",
                "is_uncertain": True
            })
        
        return jsonify({
            "error": "No models available for prediction"
        }), 503

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }), 500


@app.route("/config", methods=["GET", "POST"])
def config():
    """
    Get or update API configuration
    
    GET: Returns current configuration
    POST: Update confidence threshold
        {
            "confidence_threshold": 0.5
        }
    """
    global CONFIDENCE_THRESHOLD
    
    if request.method == "POST":
        data = request.get_json()
        if "confidence_threshold" in data:
            try:
                new_threshold = float(data["confidence_threshold"])
                if 0 <= new_threshold <= 1:
                    CONFIDENCE_THRESHOLD = new_threshold
                    return jsonify({
                        "success": True,
                        "message": "Threshold updated successfully",
                        "confidence_threshold": CONFIDENCE_THRESHOLD
                    })
                else:
                    return jsonify({
                        "success": False,
                        "error": "Threshold must be between 0 and 1"
                    }), 400
            except ValueError:
                return jsonify({
                    "success": False,
                    "error": "Invalid threshold value"
                }), 400
    
    return jsonify({
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "categories": list(label_encoder.classes_),
        "num_features": len(vectorizer.get_feature_names_out()),
        "model_type": "MLP with data augmentation"
    })

@app.route("/health", methods=["GET"])
def health():
    """API health check endpoint"""
    try:
        # Test DL model if available
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
                },
                "gemini_ai": {
                    "available": gemini_predictor is not None,
                    "status": "ready" if gemini_predictor else "not configured (set GEMINI_API_KEY)"
                }
            },
            "endpoints_available": {
                "/predict": DL_MODEL_AVAILABLE,
                "/predict/gemini": gemini_predictor is not None,
                "/predict/hybrid": DL_MODEL_AVAILABLE or gemini_predictor is not None,
                "/predict/auto": DL_MODEL_AVAILABLE or gemini_predictor is not None,
                "/batch_predict": DL_MODEL_AVAILABLE
            }
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    """
    Predict categories for multiple expenses at once
    
    Request body:
        {
            "texts": ["expense 1", "expense 2", ...]
        }
    
    Response:
        {
            "success": true,
            "predictions": [...]
        }
    """
    try:
        data = request.get_json()
        if not data or "texts" not in data:
            return jsonify({
                "error": "Missing 'texts' field in request body",
                "example": {"texts": ["pizza delivery", "uber ride"]}
            }), 400
        
        texts = data["texts"]
        if not isinstance(texts, list):
            return jsonify({"error": "texts must be a list"}), 400
        
        if len(texts) == 0:
            return jsonify({"error": "texts list is empty"}), 400
        
        if len(texts) > 100:
            return jsonify({"error": "Maximum 100 texts per batch request"}), 400
        
        predictions = []
        
        for raw_text in texts:
            if not raw_text or not raw_text.strip():
                predictions.append({
                    "text": raw_text,
                    "error": "Empty text"
                })
                continue
            
            try:
                cleaned_text = clean_text(raw_text.strip())
                
                if not cleaned_text:
                    predictions.append({
                        "text": raw_text,
                        "error": "Preprocessing resulted in empty string"
                    })
                    continue
                
                X = vectorizer.transform([cleaned_text]).toarray()
                preds = model.predict(X, verbose=0)
                
                idx = int(np.argmax(preds[0]))
                confidence = float(preds[0][idx])
                
                if confidence < CONFIDENCE_THRESHOLD:
                    category = "Miscellaneous"
                    is_uncertain = True
                    original_prediction = label_encoder.inverse_transform([idx])[0]
                else:
                    category = label_encoder.inverse_transform([idx])[0]
                    is_uncertain = False
                    original_prediction = None
                
                pred_result = {
                    "text": raw_text,
                    "predicted_category": category,
                    "confidence": round(confidence, 4),
                    "is_uncertain": is_uncertain
                }
                
                if is_uncertain:
                    pred_result["original_prediction"] = original_prediction
                
                predictions.append(pred_result)
                
            except Exception as e:
                predictions.append({
                    "text": raw_text,
                    "error": str(e)
                })
        
        return jsonify({
            "success": True,
            "total": len(texts),
            "predictions": predictions
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 Starting Flask API Server...")
    print("=" * 60)
    
    # Get port from environment variable (for Render/Heroku deployment)
    port = int(os.environ.get("PORT", 5000))
    
    print("📍 Server will be available at:")
    print(f"   - Local: http://127.0.0.1:{port}")
    print(f"   - Network: http://0.0.0.0:{port}")
    print("\n📋 Available endpoints:")
    print("   GET  /              - API info")
    print("   POST /predict       - Predict single expense")
    print("   POST /batch_predict - Predict multiple expenses")
    print("   GET  /config        - Get configuration")
    print("   POST /config        - Update configuration")
    print("   GET  /health        - Health check")
    print("=" * 60)
    print("\n💡 Test with:")
    print(f'   curl -X POST http://localhost:{port}/predict \\')
    print('        -H "Content-Type: application/json" \\')
    print('        -d \'{"text": "pizza from dominos"}\'')
    print("\n" + "=" * 60 + "\n")
    
    # For production deployment, disable debug mode
    debug_mode = os.environ.get("FLASK_ENV") != "production"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)