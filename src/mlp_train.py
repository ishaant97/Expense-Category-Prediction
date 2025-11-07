# src/mlp_train.py
import pandas as pd
import numpy as np
import re
import os
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# === 0. Preprocessing utilities ===
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# === 1. Load and preprocess data ===
df = pd.read_csv('../data/expenses.csv')
df["cleaned_description"] = df["description"].astype(str).apply(clean_text)

texts = df["cleaned_description"].tolist()
labels = df["category"].astype(str).tolist()

# === 2. Encode labels ===
le = LabelEncoder()
y = le.fit_transform(labels)
num_classes = len(le.classes_)

# === 3. TF-IDF vectorization ===
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(texts).toarray()

# === 4. Split data ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 5. Build the MLP model ===
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# === 6. Compile the model ===
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# === 7. Train the model ===
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=4,
    verbose=1
)

# === 8. Evaluate on test data ===
y_pred_probs = model.predict(X_test)
y_pred = y_pred_probs.argmax(axis=1)
y_pred_confidence = y_pred_probs.max(axis=1)

print("\n=== TEST PERFORMANCE ===")
print("Test Accuracy:", accuracy_score(y_test, y_pred))

# Print confidence statistics
print("\n=== CONFIDENCE STATISTICS ===")
print(f"Mean confidence: {y_pred_confidence.mean():.3f}")
print(f"Min confidence: {y_pred_confidence.min():.3f}")
print(f"Max confidence: {y_pred_confidence.max():.3f}")
print(f"Median confidence: {np.median(y_pred_confidence):.3f}")

# Evaluate with confidence threshold
CONFIDENCE_THRESHOLD = 0.5  # Adjust this based on your needs
low_confidence_mask = y_pred_confidence < CONFIDENCE_THRESHOLD
print(f"\nPredictions below {CONFIDENCE_THRESHOLD} confidence: {low_confidence_mask.sum()} out of {len(y_test)} ({100*low_confidence_mask.sum()/len(y_test):.1f}%)")

present_labels = np.unique(y_test)
present_names = [le.classes_[i] for i in present_labels]

print("\nClassification Report:")
print(classification_report(y_test, y_pred,
                            labels=present_labels,
                            target_names=present_names,
                            zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=present_labels))

# === 9. Save model, encoder, and vectorizer ===
os.makedirs("../models", exist_ok=True)
model.save("../models/expense_mlp_model.keras")
joblib.dump(le, "../models/label_encoder.joblib")
joblib.dump(vectorizer, "../models/vectorizer.joblib")
print("\n✅ Model, label encoder, and vectorizer saved successfully!")