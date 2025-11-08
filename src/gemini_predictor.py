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
        
        # Use Gemini 2.5 Flash - fast, cost-effective, and stable
        # This is the recommended model for simple categorization tasks
        # Available models checked with your API key on 2025-11-08
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.model_name = 'gemini-2.5-flash'
        
        # Create prompt template - allows flexible categorization
        self.prompt_template = """You are an expense categorization expert.

Analyze the following expense description and categorize it with a meaningful, specific one-word category.

Expense description: "{expense}"

Instructions:
1. Return ONLY ONE WORD - the category name
2. Use a clear, descriptive category that best represents the expense
3. Common categories include: Food, Transport, Utilities, Rent, Health, Education, Entertainment, Groceries, Shopping, Fitness, Travel, Insurance, Subscription, Bills, Fuel, Dining, Coffee, etc.
4. Be specific and meaningful (e.g., "Coffee" instead of just "Food" for Starbucks)
5. Use title case (capitalize first letter)
6. Think about what makes sense for financial tracking

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
            # Create prompt (no longer using predefined categories)
            prompt = self.prompt_template.format(expense=text.strip())
            
            # Generate response
            response = self.model.generate_content(prompt)
            raw_category = response.text.strip()
            
            # Clean and validate category
            category = self._clean_category(raw_category)
            
            # Estimate confidence based on response quality
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
    
    def _clean_category(self, raw_category: str) -> str:
        """
        Clean and validate Gemini's response
        
        Args:
            raw_category: Raw response from Gemini
            
        Returns:
            Cleaned category name (one word, title case)
        """
        # Clean the response - remove quotes, extra spaces
        cleaned = raw_category.strip().strip('"').strip("'").strip()
        
        # If multiple words, take the first meaningful word
        # (in case Gemini returned extra text)
        words = cleaned.split()
        if words:
            # Take the first word that looks like a category
            category = words[0]
            
            # Capitalize first letter (title case)
            category = category.capitalize()
            
            # Remove any trailing punctuation
            category = category.rstrip('.,!?;:')
            
            return category
        
        # Fallback if somehow empty
        return "Miscellaneous"
    
    def _estimate_confidence(self, raw_response: str, final_category: str) -> float:
        """
        Estimate confidence based on response clarity
        
        Since Gemini doesn't provide confidence scores, we estimate based on:
        - Response length (one word = high confidence)
        - Response clarity
        - Presence of extra explanation
        
        Args:
            raw_response: Raw Gemini response
            final_category: Cleaned category
            
        Returns:
            Estimated confidence (0.0 to 1.0)
        """
        # Clean response for comparison
        cleaned_response = raw_response.strip().strip('"').strip("'")
        
        # If response is exactly one word (just the category), very high confidence
        if len(cleaned_response.split()) == 1:
            return 0.95
        
        # If response is the category plus minor formatting, high confidence
        if cleaned_response.lower() == final_category.lower():
            return 0.90
        
        # If response contains the category, medium-high confidence
        if final_category.lower() in cleaned_response.lower():
            return 0.85
        
        # Otherwise, medium confidence
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
