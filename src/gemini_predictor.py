# src/gemini_predictor.py
"""
Gemini API integration for expense categorization
"""
import os
import google.generativeai as genai
from typing import Dict, Optional

class GeminiExpensePredictor:
    """Expense categorization using Google Gemini API"""
    
    # Supported categories (matching your DL model)
    CATEGORIES = [
        "Food",
        "Transport", 
        "Utilities",
        "Rent",
        "Health",
        "Education",
        "Entertainment",
        "Groceries",
        "Miscellaneous"
    ]
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini predictor
        
        Args:
            api_key: Google Gemini API key (or set GEMINI_API_KEY env variable)
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Use Gemini 1.5 Flash model (faster and more cost-effective)
        # Alternative: 'gemini-1.5-pro' for more complex tasks
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.model_name = 'gemini-1.5-flash'
        
        # Create prompt template
        self.prompt_template = """You are an expense categorization expert. 

Categorize the following expense description into EXACTLY ONE of these categories:
{categories}

Expense description: "{expense}"

Rules:
1. Return ONLY the category name, nothing else
2. Choose the MOST appropriate category
3. If uncertain, use "Miscellaneous"
4. Be consistent with common expense patterns

Category:"""
    
    def predict(self, text: str, return_confidence: bool = True) -> Dict:
        """
        Predict expense category using Gemini
        
        Args:
            text: Expense description
            return_confidence: Whether to estimate confidence (always True for Gemini)
            
        Returns:
            {
                "category": str,
                "confidence": float,
                "raw_response": str,
                "reasoning": str (optional),
                "model": "gemini-1.5-flash"
            }
        """
        if not text or not text.strip():
            return {
                "category": "Miscellaneous",
                "confidence": 0.0,
                "error": "Empty text provided",
                "model": self.model_name
            }
        
        try:
            # Create prompt
            prompt = self.prompt_template.format(
                categories=", ".join(self.CATEGORIES),
                expense=text.strip()
            )
            
            # Generate response
            response = self.model.generate_content(prompt)
            raw_category = response.text.strip()
            
            # Clean and validate category
            category = self._validate_category(raw_category)
            
            # Estimate confidence (Gemini doesn't provide confidence scores)
            # We use heuristics based on response clarity
            confidence = self._estimate_confidence(raw_category, category)
            
            return {
                "category": category,
                "confidence": confidence,
                "raw_response": raw_category,
                "model": self.model_name,
                "reasoning": raw_category if len(raw_category) > len(category) else None
            }
            
        except Exception as e:
            return {
                "category": "Miscellaneous",
                "confidence": 0.0,
                "error": str(e),
                "model": self.model_name
            }
    
    def _validate_category(self, raw_category: str) -> str:
        """
        Validate and clean Gemini's response
        
        Args:
            raw_category: Raw response from Gemini
            
        Returns:
            Validated category name
        """
        # Clean the response
        cleaned = raw_category.strip().strip('"').strip("'")
        
        # Check if it matches any known category (case-insensitive)
        for category in self.CATEGORIES:
            if category.lower() in cleaned.lower():
                return category
        
        # If exact match
        if cleaned in self.CATEGORIES:
            return cleaned
        
        # Default to Miscellaneous if no match
        return "Miscellaneous"
    
    def _estimate_confidence(self, raw_response: str, final_category: str) -> float:
        """
        Estimate confidence based on response clarity
        
        Since Gemini doesn't provide confidence scores, we estimate based on:
        - Response length (shorter = more confident)
        - Exact category match
        - Additional text in response
        
        Args:
            raw_response: Raw Gemini response
            final_category: Validated category
            
        Returns:
            Estimated confidence (0.0 to 1.0)
        """
        # If response is just the category name, high confidence
        if raw_response.strip().strip('"').strip("'") == final_category:
            return 0.95
        
        # If category name appears in response, medium-high confidence
        if final_category.lower() in raw_response.lower():
            return 0.85
        
        # If we had to default to Miscellaneous, low confidence
        if final_category == "Miscellaneous":
            return 0.50
        
        # Default medium confidence
        return 0.75
    
    def batch_predict(self, texts: list) -> list:
        """
        Predict categories for multiple expenses
        
        Args:
            texts: List of expense descriptions
            
        Returns:
            List of prediction dictionaries
        """
        return [self.predict(text) for text in texts]


# Singleton instance (optional, for easier use)
_gemini_predictor = None

def get_gemini_predictor(api_key: Optional[str] = None) -> GeminiExpensePredictor:
    """Get or create Gemini predictor instance"""
    global _gemini_predictor
    
    if _gemini_predictor is None:
        _gemini_predictor = GeminiExpensePredictor(api_key)
    
    return _gemini_predictor


if __name__ == "__main__":
    # Test the Gemini predictor
    import sys
    
    # Check if API key is provided
    if len(sys.argv) > 1:
        api_key = sys.argv[1]
    else:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("❌ Error: GEMINI_API_KEY not set")
            print("\nUsage:")
            print("  python gemini_predictor.py YOUR_API_KEY")
            print("  or set GEMINI_API_KEY environment variable")
            sys.exit(1)
    
    print("=" * 60)
    print("🧪 Testing Gemini Expense Predictor")
    print("=" * 60)
    
    try:
        predictor = GeminiExpensePredictor(api_key)
        
        # Test cases
        test_expenses = [
            "pizza from dominos",
            "uber to airport",
            "monthly rent payment",
            "electricity bill",
            "gym membership",
            "netflix subscription",
            "grocery shopping at walmart",
            "doctor appointment"
        ]
        
        print("\n📋 Test Results:\n")
        for expense in test_expenses:
            result = predictor.predict(expense)
            print(f"Expense: {expense}")
            print(f"  → Category: {result['category']}")
            print(f"  → Confidence: {result['confidence']:.2%}")
            if 'error' in result:
                print(f"  → Error: {result['error']}")
            print()
        
        print("=" * 60)
        print("✅ Testing complete!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
