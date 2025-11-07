# src/check_gemini_models.py
"""
Script to check available Gemini models with your API key
"""
import os
import sys
import google.generativeai as genai

def check_available_models(api_key=None):
    """
    Check which Gemini models are available with your API key
    
    Args:
        api_key: Your Gemini API key (or set GEMINI_API_KEY env variable)
    """
    # Get API key
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("❌ Error: GEMINI_API_KEY not found")
        print("\nUsage:")
        print("  python check_gemini_models.py YOUR_API_KEY")
        print("  or set GEMINI_API_KEY environment variable")
        return
    
    print("=" * 70)
    print("🔍 Checking Available Gemini Models")
    print("=" * 70)
    print(f"API Key: {api_key[:10]}...{api_key[-4:]}")
    print()
    
    try:
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # List all available models
        print("📋 Available Models:")
        print("-" * 70)
        
        models_found = []
        generative_models = []
        
        for model in genai.list_models():
            models_found.append(model.name)
            
            # Check if it supports generateContent
            if 'generateContent' in model.supported_generation_methods:
                generative_models.append(model.name)
                print(f"\n✅ {model.name}")
                print(f"   Display Name: {model.display_name}")
                print(f"   Description: {model.description[:80]}..." if len(model.description) > 80 else f"   Description: {model.description}")
                print(f"   Supported Methods: {', '.join(model.supported_generation_methods)}")
                print(f"   Input Token Limit: {model.input_token_limit:,}")
                print(f"   Output Token Limit: {model.output_token_limit:,}")
        
        print("\n" + "=" * 70)
        print(f"📊 Summary:")
        print(f"   Total models found: {len(models_found)}")
        print(f"   Models supporting generateContent: {len(generative_models)}")
        print("=" * 70)
        
        if generative_models:
            print("\n🎯 Recommended Models for Expense Categorization:")
            print("-" * 70)
            
            for model_name in generative_models:
                # Extract simple name (remove 'models/' prefix)
                simple_name = model_name.replace('models/', '')
                
                if 'flash' in simple_name.lower():
                    print(f"⚡ {simple_name} - Fast and cost-effective (RECOMMENDED)")
                elif 'pro' in simple_name.lower():
                    print(f"🚀 {simple_name} - More powerful for complex tasks")
                else:
                    print(f"📌 {simple_name}")
            
            print("\n💡 To use a model, update gemini_predictor.py line 45:")
            print(f"   self.model = genai.GenerativeModel('{generative_models[0].replace('models/', '')}')")
            
        else:
            print("\n❌ No models supporting generateContent found!")
            print("   This might indicate an API key issue or regional restrictions.")
        
        print("\n" + "=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error checking models: {e}")
        print("\nPossible reasons:")
        print("  1. Invalid API key")
        print("  2. API key doesn't have access to Gemini models")
        print("  3. Network/firewall issues")
        print("  4. Regional restrictions")
        print("\n💡 Get a valid API key from: https://makersuite.google.com/app/apikey")


if __name__ == "__main__":
    # Check if API key is provided as argument
    if len(sys.argv) > 1:
        api_key = sys.argv[1]
    else:
        api_key = os.getenv("GEMINI_API_KEY")
    
    check_available_models(api_key)
