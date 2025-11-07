# src/mlp_train_improved.py
"""
Improved MLP training script with:
- Data augmentation
- Better preprocessing
- Optimized architecture for small datasets
- Class weight balancing
"""

import pandas as pd
import numpy as np
import re
import os
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import random

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# === 0. Text Preprocessing (IMPROVED) ===
import nltk
try:
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
except:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

# Keep important words that indicate categories
IMPORTANT_WORDS = {'rent', 'food', 'doctor', 'hospital', 'gym', 'movie', 'bus', 
                   'train', 'uber', 'electricity', 'water', 'bill', 'grocery',
                   'college', 'school', 'tuition', 'fuel', 'cafe', 'restaurant'}

stop_words = set(stopwords.words("english")) - IMPORTANT_WORDS
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """Improved text cleaning that preserves important category indicators"""
    text = text.lower()
    # Keep numbers as they might be important (e.g., "2 pizzas")
    text = re.sub(r"[^\w\s\d]", " ", text)  # keep alphanumeric
    words = text.split()
    # Lemmatize but keep important words
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words or len(word) > 2]
    return " ".join(words)

# === 1. Data Augmentation ===
def augment_text(text, category):
    """
    Create variations of text to expand dataset.
    This helps with small datasets by creating similar but slightly different examples.
    """
    augmented = []
    
    # Synonym replacement for common words
    replacements = {
        'bought': ['purchased', 'got', 'ordered'],
        'paid': ['spent on', 'payment for'],
        'monthly': ['this month', 'month'],
        'bill': ['payment', 'charge'],
        'from': ['at'],
        'to': ['towards'],
    }
    
    words = text.lower().split()
    
    # Create 2-3 variations per text
    for _ in range(2):
        new_words = words.copy()
        # Replace 1-2 words randomly
        for i, word in enumerate(new_words):
            if word in replacements and random.random() > 0.5:
                new_words[i] = random.choice(replacements[word])
        
        augmented_text = ' '.join(new_words)
        if augmented_text != text.lower():  # Don't add duplicates
            augmented.append((augmented_text, category))
    
    return augmented

# === 2. Load and preprocess data ===
print("=" * 60)
print("LOADING AND PREPROCESSING DATA")
print("=" * 60)

df = pd.read_csv('../data/expenses.csv')
print(f"\n📊 Original dataset size: {len(df)} samples")
print(f"📂 Categories: {df['category'].nunique()}")
print("\nCategory distribution:")
print(df['category'].value_counts())

# Apply data augmentation to increase dataset size
augmented_data = []
for _, row in df.iterrows():
    # Keep original
    augmented_data.append((row['description'], row['category']))
    # Add augmented versions
    augmented_data.extend(augment_text(row['description'], row['category']))

# Create augmented dataframe
df_augmented = pd.DataFrame(augmented_data, columns=['description', 'category'])
print(f"\n✨ Augmented dataset size: {len(df_augmented)} samples")
print("\nAugmented category distribution:")
print(df_augmented['category'].value_counts())

# Clean text
df_augmented["cleaned_description"] = df_augmented["description"].astype(str).apply(clean_text)

# Remove any empty strings after cleaning
df_augmented = df_augmented[df_augmented["cleaned_description"].str.strip() != ""]

texts = df_augmented["cleaned_description"].tolist()
labels = df_augmented["category"].astype(str).tolist()

print(f"\n📝 Final dataset size after cleaning: {len(texts)} samples")

# === 3. Encode labels ===
le = LabelEncoder()
y = le.fit_transform(labels)
num_classes = len(le.classes_)

print(f"\n🏷️  Number of classes: {num_classes}")
print(f"Classes: {list(le.classes_)}")

# === 4. TF-IDF vectorization (OPTIMIZED) ===
# Reduce max_features to avoid overfitting on small dataset
# Rule of thumb: features should be ~5-10x less than samples
max_features = min(200, len(texts) // 3)
print(f"\n🔢 Using {max_features} TF-IDF features (optimized for dataset size)")

vectorizer = TfidfVectorizer(
    max_features=max_features,
    ngram_range=(1, 2),  # Include bigrams for better context
    min_df=1,  # Minimum document frequency
    max_df=0.8  # Ignore terms that appear in >80% of documents
)
X = vectorizer.fit_transform(texts).toarray()

print(f"Feature matrix shape: {X.shape}")

# === 5. Split data with stratification ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y  # Ensure balanced distribution in train/test
)

print(f"\n📊 Train size: {len(X_train)}, Test size: {len(X_test)}")

# === 6. Compute class weights to handle imbalance ===
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights_array))

print("\n⚖️  Class weights (to handle imbalance):")
for idx, weight in class_weights.items():
    print(f"  {le.classes_[idx]}: {weight:.2f}")

# === 7. Build IMPROVED MLP model ===
print("\n" + "=" * 60)
print("BUILDING MODEL")
print("=" * 60)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    
    # First hidden layer - larger to capture patterns
    tf.keras.layers.Dense(128, activation='relu', 
                         kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    
    # Second hidden layer
    tf.keras.layers.Dense(64, activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    
    # Third hidden layer
    tf.keras.layers.Dense(32, activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.2),
    
    # Output layer
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.summary()

# === 8. Compile with optimized settings ===
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# === 9. Training with callbacks ===
print("\n" + "=" * 60)
print("TRAINING MODEL")
print("=" * 60)

# Callbacks for better training
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

# Better batch size for small dataset
batch_size = max(8, len(X_train) // 20)
print(f"\nUsing batch size: {batch_size}")

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,  # Use early stopping to prevent overfitting
    batch_size=batch_size,
    class_weight=class_weights,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# === 10. Comprehensive Evaluation ===
print("\n" + "=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)

y_pred_probs = model.predict(X_test)
y_pred = y_pred_probs.argmax(axis=1)
y_pred_confidence = y_pred_probs.max(axis=1)

print("\n=== TEST PERFORMANCE ===")
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Training history
print("\n=== TRAINING HISTORY ===")
print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Total epochs trained: {len(history.history['accuracy'])}")

# Confidence statistics
print("\n=== CONFIDENCE STATISTICS ===")
print(f"Mean confidence: {y_pred_confidence.mean():.3f}")
print(f"Min confidence: {y_pred_confidence.min():.3f}")
print(f"Max confidence: {y_pred_confidence.max():.3f}")
print(f"Median confidence: {np.median(y_pred_confidence):.3f}")

# Confidence threshold analysis
for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
    low_conf = (y_pred_confidence < threshold).sum()
    print(f"Predictions below {threshold} confidence: {low_conf}/{len(y_test)} ({100*low_conf/len(y_test):.1f}%)")

# Classification report
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred,
                            target_names=le.classes_,
                            zero_division=0))

# Confusion matrix
print("\n=== CONFUSION MATRIX ===")
print("Rows: True labels, Columns: Predicted labels")
print("Classes order:", list(le.classes_))
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Per-class accuracy
print("\n=== PER-CLASS ACCURACY ===")
for i, class_name in enumerate(le.classes_):
    class_mask = (y_test == i)
    if class_mask.sum() > 0:
        class_acc = accuracy_score(y_test[class_mask], y_pred[class_mask])
        print(f"{class_name:15s}: {class_acc:.3f} ({class_mask.sum()} samples)")

# === 11. Save model and artifacts ===
print("\n" + "=" * 60)
print("SAVING MODEL")
print("=" * 60)

os.makedirs("../models", exist_ok=True)
model.save("../models/expense_mlp_model_improved.keras")
joblib.dump(le, "../models/label_encoder_improved.joblib")
joblib.dump(vectorizer, "../models/vectorizer_improved.joblib")

# Save training history
import json
history_dict = {
    'accuracy': [float(x) for x in history.history['accuracy']],
    'val_accuracy': [float(x) for x in history.history['val_accuracy']],
    'loss': [float(x) for x in history.history['loss']],
    'val_loss': [float(x) for x in history.history['val_loss']]
}
with open("../models/training_history.json", 'w') as f:
    json.dump(history_dict, f, indent=2)

print("\n✅ Saved:")
print("  - Model: ../models/expense_mlp_model_improved.keras")
print("  - Label Encoder: ../models/label_encoder_improved.joblib")
print("  - Vectorizer: ../models/vectorizer_improved.joblib")
print("  - Training History: ../models/training_history.json")

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
