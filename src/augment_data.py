# src/augment_data.py
"""
Advanced data augmentation script to expand the expense dataset.
Creates realistic variations of existing expenses.
"""

import pandas as pd
import random
import re

# Set seed for reproducibility
random.seed(42)

# Category-specific patterns and templates
CATEGORY_PATTERNS = {
    'Food': [
        'lunch at {place}',
        'dinner at {place}',
        'breakfast from {place}',
        'ordered {item} online',
        '{item} from {place}',
        'coffee and {item}',
        'snacks from {place}',
        '{meal} with {person}',
        'ate {item} at {place}',
        'food delivery from {place}',
        '{item} and {item2} at cafe',
        'quick bite at {place}',
    ],
    'Transport': [
        'uber to {destination}',
        'bus to {destination}',
        'metro ride to {destination}',
        'cab to {destination}',
        'auto to {destination}',
        'train to {destination}',
        'taxi to {destination}',
        'bus fare for {destination}',
        'fuel for {vehicle}',
        'diesel for {vehicle}',
        'petrol refill',
        'ola cab to {destination}',
        '{transport} fare',
    ],
    'Rent': [
        'monthly rent {month}',
        'rent for {month}',
        'apartment rent',
        'house rent payment',
        'hostel rent',
        'room rent',
        'rent payment {month}',
        'paid monthly rent',
    ],
    'Utilities': [
        'electricity bill {month}',
        'water bill {month}',
        'internet bill',
        'mobile bill',
        'broadband payment',
        'power bill {month}',
        'phone bill',
        'wifi payment',
        'cable tv bill',
        'gas bill {month}',
        'postpaid mobile bill',
    ],
    'Entertainment': [
        'movie at {place}',
        'netflix subscription',
        'spotify subscription',
        'prime video payment',
        'movie tickets',
        'concert tickets',
        'gaming subscription',
        'youtube premium',
        'cinema tickets at {place}',
        'streaming service payment',
        'movie night at {place}',
    ],
    'Groceries': [
        'grocery from {place}',
        'vegetables and fruits',
        'weekly groceries',
        'monthly grocery shopping',
        'bought {item} and {item2}',
        'supermarket shopping',
        'grocery delivery',
        'vegetables from market',
        'fruits and vegetables',
        'grocery items from {place}',
        'milk and bread',
    ],
    'Health': [
        'doctor consultation',
        'medical checkup',
        'dentist appointment',
        'gym membership',
        'yoga class',
        'medicine purchase',
        'health checkup',
        'hospital visit',
        'pharmacy bill',
        'fitness class',
        'gym renewal',
        'online doctor consultation',
    ],
    'Education': [
        'tuition fee',
        'course fee payment',
        'textbook purchase',
        'online course subscription',
        'stationery items',
        'notebook and pens',
        'college fee',
        'coaching classes',
        'study material',
        'semester fee',
        'educational course',
    ],
}

# Replacement words for variety
REPLACEMENTS = {
    'place': ['mcdonalds', 'kfc', 'dominos', 'cafe coffee day', 'starbucks', 
              'subway', 'burger king', 'restaurant', 'hotel', 'dhaba', 'canteen',
              'food court', 'bistro', 'pizzeria'],
    'destination': ['office', 'airport', 'home', 'college', 'mall', 'station',
                   'market', 'hospital', 'gym', 'work', 'downtown', 'city center'],
    'vehicle': ['bike', 'car', 'scooter', 'motorcycle', 'vehicle'],
    'month': ['january', 'february', 'march', 'april', 'may', 'june',
              'july', 'august', 'september', 'october', 'november', 'december'],
    'item': ['pizza', 'burger', 'sandwich', 'biryani', 'pasta', 'noodles',
             'rice', 'roti', 'dal', 'chicken', 'salad'],
    'item2': ['fries', 'coke', 'juice', 'dessert', 'coffee', 'tea'],
    'meal': ['lunch', 'dinner', 'breakfast'],
    'person': ['friends', 'family', 'colleague', 'roommate'],
    'transport': ['bus', 'metro', 'auto', 'cab'],
}

def generate_augmented_samples(category, num_samples=10):
    """Generate synthetic samples for a category"""
    samples = []
    patterns = CATEGORY_PATTERNS.get(category, [])
    
    for _ in range(num_samples):
        if not patterns:
            continue
            
        pattern = random.choice(patterns)
        
        # Replace placeholders with random values
        text = pattern
        for placeholder, options in REPLACEMENTS.items():
            if '{' + placeholder + '}' in text:
                text = text.replace('{' + placeholder + '}', random.choice(options))
        
        samples.append({'description': text, 'category': category})
    
    return samples

def augment_dataset(input_file, output_file, samples_per_category=15):
    """
    Augment the dataset by generating additional samples.
    
    Args:
        input_file: Path to original CSV
        output_file: Path to save augmented CSV
        samples_per_category: Number of additional samples to generate per category
    """
    # Load original data
    df = pd.read_csv(input_file)
    
    print("=" * 60)
    print("DATA AUGMENTATION")
    print("=" * 60)
    print(f"\n📊 Original dataset: {len(df)} samples")
    print("\nOriginal category distribution:")
    print(df['category'].value_counts())
    
    # Generate augmented samples for each category
    augmented_samples = []
    
    for category in df['category'].unique():
        category_samples = generate_augmented_samples(category, samples_per_category)
        augmented_samples.extend(category_samples)
        print(f"\n✨ Generated {len(category_samples)} samples for '{category}'")
    
    # Create augmented dataframe
    df_augmented = pd.DataFrame(augmented_samples)
    
    # Combine original and augmented data
    df_combined = pd.concat([df, df_augmented], ignore_index=True)
    
    # Shuffle
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n📈 Augmented dataset: {len(df_combined)} samples")
    print("\nAugmented category distribution:")
    print(df_combined['category'].value_counts())
    
    # Save
    df_combined.to_csv(output_file, index=False)
    print(f"\n✅ Saved augmented dataset to: {output_file}")
    
    return df_combined

if __name__ == "__main__":
    # Create augmented dataset
    augment_dataset(
        input_file='../data/expenses.csv',
        output_file='../data/expenses_augmented.csv',
        samples_per_category=20  # Generate 20 additional samples per category
    )
    
    print("\n" + "=" * 60)
    print("AUGMENTATION COMPLETE!")
    print("=" * 60)
    print("\nYou can now train on the augmented dataset:")
    print("  python mlp_train_improved.py")
