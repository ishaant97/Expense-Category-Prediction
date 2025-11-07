# src/serve_model.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import joblib
import numpy as np
import os
import re

# === NLP Preprocessing Setup ===
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

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

# === Load model and artifacts ===
# Using the improved model trained with data augmentation
MODEL_PATH = os.path.join("../models", "expense_mlp_best.keras")
ENCODER_PATH = os.path.join("../models", "label_encoder_best.joblib")
VECTORIZER_PATH = os.path.join("../models", "vectorizer_best.joblib")

print("=" * 60)
print("🔄 Loading Improved Expense Categorization Model...")
print("=" * 60)

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    
    print(f"✅ Model loaded: {MODEL_PATH}")
    print(f"✅ Label encoder loaded: {ENCODER_PATH}")
    print(f"✅ Vectorizer loaded: {VECTORIZER_PATH}")
    print(f"\n📊 Model Info:")
    print(f"   Categories: {list(label_encoder.classes_)}")
    print(f"   Features: {len(vectorizer.get_feature_names_out())}")
    print(f"   Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print("=" * 60)
    print("✅ API Ready to serve predictions!")
    print("=" * 60)
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Please ensure you've trained the model first:")
    print("  python train_with_augmented_data.py")
    raise

# === Create Flask app ===
app = Flask(__name__)

# Enable CORS for all routes (allows your website to make API calls)
CORS(app)

@app.route("/", methods=["GET"])
def home():
    """Home endpoint with API information"""
    return jsonify({
        "message": "🚀 Expense Category Prediction API",
        "version": "2.0 - Improved Model",
        "model_accuracy": "93.33%",
        "categories": list(label_encoder.classes_),
        "endpoints": {
            "POST /predict": "Predict expense category",
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
    Predict expense category from text description
    
    Request body:
        {
            "text": "expense description"
        }
    
    Response:
        {
            "text": "original text",
            "cleaned_text": "preprocessed text",
            "predicted_category": "category name",
            "confidence": 0.95,
            "is_uncertain": false,
            "all_predictions": {...}  // optional
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

        # NLP preprocessing
        cleaned_text = clean_text(raw_text)
        
        if not cleaned_text:
            return jsonify({
                "error": "Text preprocessing resulted in empty string",
                "original_text": raw_text
            }), 400

        # Vectorize and predict
        X = vectorizer.transform([cleaned_text]).toarray()
        preds = model.predict(X, verbose=0)
        
        # Get prediction details
        idx = int(np.argmax(preds[0]))
        confidence = float(preds[0][idx])
        
        # Get all predictions (optional, controlled by request)
        include_all = data.get("include_all_predictions", False)
        all_predictions = None
        
        if include_all:
            all_predictions = {
                label_encoder.inverse_transform([i])[0]: float(preds[0][i])
                for i in range(len(preds[0]))
            }
            # Sort by confidence
            all_predictions = dict(sorted(all_predictions.items(), 
                                        key=lambda x: x[1], reverse=True))
        
        # Apply confidence threshold
        if confidence < CONFIDENCE_THRESHOLD:
            category = "Miscellaneous"
            is_uncertain = True
            original_prediction = label_encoder.inverse_transform([idx])[0]
        else:
            category = label_encoder.inverse_transform([idx])[0]
            is_uncertain = False
            original_prediction = None

        # Build response
        response = {
            "success": True,
            "text": raw_text,
            "cleaned_text": cleaned_text,
            "predicted_category": category,
            "confidence": round(confidence, 4),
            "is_uncertain": is_uncertain
        }
        
        # Include original prediction if marked as miscellaneous
        if is_uncertain:
            response["original_prediction"] = original_prediction
            response["message"] = f"Low confidence ({round(confidence, 4)}). Suggested category: {original_prediction}, but marked as Miscellaneous."
        
        # Add all predictions if requested
        if all_predictions:
            response["all_predictions"] = all_predictions

        return jsonify(response)

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
        # Quick prediction test
        test_text = "test"
        cleaned = clean_text(test_text)
        X = vectorizer.transform([cleaned]).toarray()
        _ = model.predict(X, verbose=0)
        
        return jsonify({
            "status": "healthy",
            "model_loaded": True,
            "api_version": "2.0",
            "model_accuracy": "93.33%"
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