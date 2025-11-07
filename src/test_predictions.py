# src/test_predictions.py
"""
Test the trained model with sample predictions.
"""

import numpy as np
import joblib
import tensorflow as tf
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load preprocessing utilities
IMPORTANT_WORDS = {'rent', 'food', 'doctor', 'hospital', 'gym', 'movie', 'bus',
                   'train', 'uber', 'ola', 'electricity', 'water', 'bill', 'grocery',
                   'college', 'school', 'tuition', 'fuel', 'cafe', 'restaurant',
                   'netflix', 'spotify', 'prime', 'metro', 'cab', 'auto'}

stop_words = set(stopwords.words("english")) - IMPORTANT_WORDS
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """Clean text for prediction"""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words 
             if word not in stop_words or len(word) > 2]
    return " ".join(words)

# Load model and artifacts
print("🔄 Loading model...")
model = tf.keras.models.load_model("../models/expense_mlp_best.keras")
label_encoder = joblib.load("../models/label_encoder_best.joblib")
vectorizer = joblib.load("../models/vectorizer_best.joblib")
print("✅ Model loaded!\n")

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.5

def predict_expense(text):
    """Predict category for an expense description"""
    # Clean text
    cleaned = clean_text(text)
    
    # Vectorize
    X = vectorizer.transform([cleaned]).toarray()
    
    # Predict
    probs = model.predict(X, verbose=0)[0]
    idx = int(np.argmax(probs))
    confidence = float(probs[idx])
    
    # Apply threshold
    if confidence < CONFIDENCE_THRESHOLD:
        category = "Miscellaneous"
        is_uncertain = True
        original = label_encoder.inverse_transform([idx])[0]
    else:
        category = label_encoder.inverse_transform([idx])[0]
        is_uncertain = False
        original = None
    
    return {
        'text': text,
        'cleaned': cleaned,
        'category': category,
        'confidence': confidence,
        'is_uncertain': is_uncertain,
        'original_prediction': original,
        'all_probabilities': {label_encoder.inverse_transform([i])[0]: float(probs[i]) 
                             for i in range(len(probs))}
    }

# Test cases covering all categories
test_cases = [
    # Food
    "pizza from dominos",
    "breakfast at starbucks",
    "ordered biryani online",
    "dinner with friends",
    
    # Transport
    "uber to office",
    "bus fare to college",
    "fuel for my bike",
    "metro card recharge",
    
    # Rent
    "monthly apartment rent",
    "house rent for december",
    
    # Utilities
    "electricity bill payment",
    "internet broadband bill",
    "mobile phone bill",
    
    # Entertainment
    "netflix monthly subscription",
    "movie tickets pvr",
    "spotify premium",
    
    # Groceries
    "vegetables from supermarket",
    "weekly grocery shopping",
    "milk and bread",
    
    # Health
    "gym membership renewal",
    "doctor consultation fee",
    "medical checkup",
    
    # Education
    "college tuition fee",
    "bought textbooks",
    "online course payment",
    
    # Edge cases - unclear
    "random purchase",
    "miscellaneous expense",
    "bought something",
]

print("=" * 80)
print("TESTING PREDICTIONS")
print("=" * 80)

correct = 0
total = 0

for text in test_cases:
    result = predict_expense(text)
    
    print(f"\n📝 Input: {result['text']}")
    print(f"   Cleaned: {result['cleaned']}")
    print(f"   ➡️  Category: {result['category']}")
    print(f"   📊 Confidence: {result['confidence']:.3f}")
    
    if result['is_uncertain']:
        print(f"   ⚠️  Uncertain (original: {result['original_prediction']})")
    
    # Show top 3 predictions
    sorted_probs = sorted(result['all_probabilities'].items(), 
                          key=lambda x: x[1], reverse=True)
    print(f"   Top 3:")
    for cat, prob in sorted_probs[:3]:
        print(f"      {cat:15s}: {prob:.3f}")

print("\n" + "=" * 80)
print("TESTING COMPLETE")
print("=" * 80)

# Interactive mode
print("\n💡 Interactive Mode - Test your own expenses!")
print("   (Type 'quit' to exit)\n")

while True:
    try:
        text = input("Enter expense description: ").strip()
        if text.lower() in ['quit', 'exit', 'q']:
            break
        
        if not text:
            continue
        
        result = predict_expense(text)
        print(f"\n   Category: {result['category']}")
        print(f"   Confidence: {result['confidence']:.3f}")
        
        if result['is_uncertain']:
            print(f"   ⚠️  Low confidence - Original prediction: {result['original_prediction']}")
        
        # Show top 3
        sorted_probs = sorted(result['all_probabilities'].items(),
                             key=lambda x: x[1], reverse=True)
        print(f"   Top predictions:")
        for cat, prob in sorted_probs[:3]:
            print(f"      {cat:15s}: {prob:.3f}")
        print()
        
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"   Error: {e}\n")

print("\n👋 Goodbye!")