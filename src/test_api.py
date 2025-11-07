"""
Test client for the Expense Category Prediction API
Demonstrates how to make requests to the Flask API
"""

import requests
import json

# API base URL
BASE_URL = "http://localhost:5000"

def test_home():
    """Test the home endpoint"""
    print("\n" + "=" * 70)
    print("Testing GET / (API Info)")
    print("=" * 70)
    
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

def test_health():
    """Test health check endpoint"""
    print("\n" + "=" * 70)
    print("Testing GET /health")
    print("=" * 70)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

def test_config():
    """Test config endpoint"""
    print("\n" + "=" * 70)
    print("Testing GET /config")
    print("=" * 70)
    
    response = requests.get(f"{BASE_URL}/config")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

def test_single_prediction(text):
    """Test single prediction"""
    print("\n" + "=" * 70)
    print(f"Testing POST /predict with: '{text}'")
    print("=" * 70)
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json={"text": text}
    )
    print(f"Status: {response.status_code}")
    result = response.json()
    print(json.dumps(result, indent=2))
    
    return result

def test_prediction_with_all_probs(text):
    """Test prediction with all probabilities"""
    print("\n" + "=" * 70)
    print(f"Testing POST /predict with all predictions: '{text}'")
    print("=" * 70)
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json={
            "text": text,
            "include_all_predictions": True
        }
    )
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

def test_batch_prediction():
    """Test batch prediction"""
    print("\n" + "=" * 70)
    print("Testing POST /batch_predict")
    print("=" * 70)
    
    texts = [
        "pizza from dominos",
        "uber to airport",
        "monthly rent payment",
        "electricity bill",
        "gym membership",
        "netflix subscription",
        "grocery shopping",
        "doctor consultation"
    ]
    
    response = requests.post(
        f"{BASE_URL}/batch_predict",
        json={"texts": texts}
    )
    print(f"Status: {response.status_code}")
    result = response.json()
    
    if result.get("success"):
        print(f"\nTotal predictions: {result['total']}")
        print("\nResults:")
        for pred in result['predictions']:
            if 'error' not in pred:
                print(f"  '{pred['text']:30s}' → {pred['predicted_category']:15s} ({pred['confidence']:.3f})")
            else:
                print(f"  '{pred['text']:30s}' → ERROR: {pred['error']}")
    else:
        print(json.dumps(result, indent=2))

def test_update_threshold():
    """Test updating confidence threshold"""
    print("\n" + "=" * 70)
    print("Testing POST /config (Update threshold to 0.7)")
    print("=" * 70)
    
    response = requests.post(
        f"{BASE_URL}/config",
        json={"confidence_threshold": 0.7}
    )
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

def run_all_tests():
    """Run all API tests"""
    print("\n" + "=" * 70)
    print("🧪 EXPENSE CATEGORY PREDICTION API TEST SUITE")
    print("=" * 70)
    
    try:
        # Basic tests
        test_home()
        test_health()
        test_config()
        
        # Single predictions for each category
        print("\n\n" + "=" * 70)
        print("🎯 TESTING PREDICTIONS FOR ALL CATEGORIES")
        print("=" * 70)
        
        test_cases = [
            # Food
            ("pizza from dominos", "Food"),
            ("breakfast at starbucks", "Food"),
            
            # Transport
            ("uber to office", "Transport"),
            ("bus fare payment", "Transport"),
            
            # Rent
            ("monthly apartment rent", "Rent"),
            
            # Utilities
            ("electricity bill payment", "Utilities"),
            ("internet broadband", "Utilities"),
            
            # Entertainment
            ("netflix subscription", "Entertainment"),
            ("movie tickets", "Entertainment"),
            
            # Groceries
            ("grocery shopping", "Groceries"),
            ("vegetables from market", "Groceries"),
            
            # Health
            ("gym membership", "Health"),
            ("doctor consultation", "Health"),
            
            # Education
            ("tuition fee payment", "Education"),
            ("textbook purchase", "Education"),
            
            # Uncertain cases
            ("random purchase", "Miscellaneous (expected)"),
            ("bought something", "Miscellaneous (expected)"),
        ]
        
        correct = 0
        total = 0
        
        for text, expected in test_cases:
            result = test_single_prediction(text)
            if result.get("success"):
                predicted = result["predicted_category"]
                confidence = result["confidence"]
                
                # Check if prediction matches expected (roughly)
                if expected.startswith("Miscellaneous") or expected.lower() in predicted.lower() or predicted.lower() in expected.lower():
                    correct += 1
                    print(f"✅ PASS - Expected: {expected}")
                else:
                    print(f"❌ FAIL - Expected: {expected}, Got: {predicted}")
                
                total += 1
        
        print(f"\n📊 Test Results: {correct}/{total} predictions matched expectations")
        
        # Test with all probabilities
        test_prediction_with_all_probs("pizza delivery")
        
        # Batch prediction
        test_batch_prediction()
        
        # Config update
        test_update_threshold()
        
        # Reset threshold
        print("\n" + "=" * 70)
        print("Resetting threshold to 0.5")
        print("=" * 70)
        requests.post(f"{BASE_URL}/config", json={"confidence_threshold": 0.5})
        
        print("\n" + "=" * 70)
        print("✅ ALL TESTS COMPLETED!")
        print("=" * 70)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to API")
        print("Make sure the Flask server is running:")
        print("  python serve_model.py")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")

if __name__ == "__main__":
    # Run all tests
    run_all_tests()
    
    print("\n\n💡 You can also test individual functions:")
    print("   test_single_prediction('your text here')")
    print("   test_batch_prediction()")
    print("   test_update_threshold()")
