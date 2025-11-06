# src/check_data.py
import pandas as pd

# load the CSV file
df = pd.read_csv('../data/expenses.csv')

print("=== DATA SAMPLE ===")
print(df.head(10).to_string(index=False))

print("\n=== BASIC STATS ===")
print("Total rows:", len(df))
print("Columns:", list(df.columns))

print("\nCategory counts:")
print(df['category'].value_counts().to_string())

print("\nUnique categories:", df['category'].nunique())

print("\nRandom sample:")
print(df.sample(5).to_string(index=False))
