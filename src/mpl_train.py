# src/train_with_augmented_data.py
"""
Training script that uses the augmented dataset for better accuracy.
This script first generates augmented data, then trains the improved model.
"""

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
from sklearn.utils.class_weight import compute_class_weight
import random

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Import augmentation function
import sys
sys.path.append(os.path.dirname(__file__))
from augment_data import augment_dataset

# === 0. Text Preprocessing ===
import nltk
try:
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
except:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

IMPORTANT_WORDS = {'rent', 'food', 'doctor', 'hospital', 'gym', 'movie', 'bus',
                   'train', 'uber', 'ola', 'electricity', 'water', 'bill', 'grocery',
                   'college', 'school', 'tuition', 'fuel', 'cafe', 'restaurant',
                   'netflix', 'spotify', 'prime', 'metro', 'cab', 'auto'}

stop_words = set(stopwords.words("english")) - IMPORTANT_WORDS
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """Enhanced text cleaning"""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words 
             if word not in stop_words or len(word) > 2]
    return " ".join(words)

# === 1. Generate or load augmented data ===
print("=" * 70)
print("STEP 1: DATA PREPARATION")
print("=" * 70)

augmented_file = '../data/expenses_augmented.csv'

# Generate augmented dataset
if not os.path.exists(augmented_file):
    print("\n📝 Generating augmented dataset...")
    df = augment_dataset(
        input_file='../data/expenses.csv',
        output_file=augmented_file,
        samples_per_category=20
    )
else:
    print(f"\n📂 Loading existing augmented dataset from {augmented_file}")
    df = pd.read_csv(augmented_file)

print(f"\n📊 Dataset size: {len(df)} samples")
print(f"📂 Categories: {df['category'].nunique()}")
print("\nCategory distribution:")
category_counts = df['category'].value_counts()
for cat, count in category_counts.items():
    print(f"  {cat:15s}: {count:3d} samples")

# === 2. Preprocess text ===
print("\n" + "=" * 70)
print("STEP 2: TEXT PREPROCESSING")
print("=" * 70)

df["cleaned_description"] = df["description"].astype(str).apply(clean_text)
df = df[df["cleaned_description"].str.strip() != ""]

texts = df["cleaned_description"].tolist()
labels = df["category"].astype(str).tolist()

print(f"\n✅ Processed {len(texts)} samples")

# Show some examples
print("\n📝 Sample preprocessed texts:")
for i in random.sample(range(len(texts)), min(5, len(texts))):
    print(f"  Original: {df.iloc[i]['description']}")
    print(f"  Cleaned:  {texts[i]}")
    print(f"  Category: {labels[i]}\n")

# === 3. Encode labels ===
le = LabelEncoder()
y = le.fit_transform(labels)
num_classes = len(le.classes_)

print(f"🏷️  Classes ({num_classes}): {list(le.classes_)}")

# === 4. Optimized TF-IDF ===
print("\n" + "=" * 70)
print("STEP 3: FEATURE EXTRACTION")
print("=" * 70)

# Optimal feature count based on dataset size
max_features = min(300, len(texts) // 2)
print(f"\n🔢 TF-IDF Configuration:")
print(f"  Max features: {max_features}")
print(f"  N-gram range: (1, 2) - unigrams and bigrams")
print(f"  Min document frequency: 2")
print(f"  Max document frequency: 0.85")

vectorizer = TfidfVectorizer(
    max_features=max_features,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.85,
    sublinear_tf=True  # Use sublinear TF scaling
)
X = vectorizer.fit_transform(texts).toarray()

print(f"\n✅ Feature matrix shape: {X.shape}")
print(f"   Samples: {X.shape[0]}, Features: {X.shape[1]}")

# Show top features
feature_names = vectorizer.get_feature_names_out()
print(f"\n📊 Sample features: {list(feature_names[:20])}")

# === 5. Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\n📊 Split: Train={len(X_train)}, Test={len(X_test)}")

# === 6. Class weights ===
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights_array))

print("\n⚖️  Class weights:")
for idx, weight in class_weights.items():
    print(f"  {le.classes_[idx]:15s}: {weight:.3f}")

# === 7. Build model ===
print("\n" + "=" * 70)
print("STEP 4: MODEL ARCHITECTURE")
print("=" * 70)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    
    # Layer 1
    tf.keras.layers.Dense(256, activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    
    # Layer 2
    tf.keras.layers.Dense(128, activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    
    # Layer 3
    tf.keras.layers.Dense(64, activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.3),
    
    # Output
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

print("\n📐 Model architecture:")
model.summary()

# === 8. Compile ===
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# === 9. Train ===
print("\n" + "=" * 70)
print("STEP 5: TRAINING")
print("=" * 70)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=7,
    min_lr=0.00001,
    verbose=1
)

batch_size = max(16, len(X_train) // 25)
print(f"\n⚙️  Training configuration:")
print(f"  Batch size: {batch_size}")
print(f"  Max epochs: 150")
print(f"  Early stopping patience: 20")
print(f"  Learning rate reduction: enabled")

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=150,
    batch_size=batch_size,
    class_weight=class_weights,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# === 10. Evaluate ===
print("\n" + "=" * 70)
print("STEP 6: EVALUATION")
print("=" * 70)

y_pred_probs = model.predict(X_test, verbose=0)
y_pred = y_pred_probs.argmax(axis=1)
y_pred_confidence = y_pred_probs.max(axis=1)

# Overall metrics
test_acc = accuracy_score(y_test, y_pred)
print(f"\n🎯 TEST ACCURACY: {test_acc:.4f} ({test_acc*100:.2f}%)")

print(f"\n📈 Training Summary:")
print(f"  Best val accuracy: {max(history.history['val_accuracy']):.4f}")
print(f"  Final train accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"  Epochs trained: {len(history.history['accuracy'])}")

# Confidence analysis
print(f"\n🎲 Confidence Statistics:")
print(f"  Mean:   {y_pred_confidence.mean():.3f}")
print(f"  Median: {np.median(y_pred_confidence):.3f}")
print(f"  Min:    {y_pred_confidence.min():.3f}")
print(f"  Max:    {y_pred_confidence.max():.3f}")

print(f"\n📊 Confidence Threshold Analysis:")
for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
    low = (y_pred_confidence < threshold).sum()
    pct = 100 * low / len(y_test)
    print(f"  Below {threshold}: {low:3d}/{len(y_test)} ({pct:5.1f}%)")

# Classification report
print(f"\n📋 Classification Report:")
print(classification_report(y_test, y_pred,
                            target_names=le.classes_,
                            zero_division=0))

# Confusion matrix
print(f"\n🔢 Confusion Matrix:")
print("   Rows=True, Cols=Predicted")
cm = confusion_matrix(y_test, y_pred)
print("\n   ", "  ".join([f"{c[:4]:>4s}" for c in le.classes_]))
for i, row in enumerate(cm):
    print(f"{le.classes_[i]:>8s}", "  ".join([f"{val:4d}" for val in row]))

# Per-class accuracy
print(f"\n🎯 Per-Class Accuracy:")
for i, class_name in enumerate(le.classes_):
    mask = (y_test == i)
    if mask.sum() > 0:
        acc = accuracy_score(y_test[mask], y_pred[mask])
        print(f"  {class_name:15s}: {acc:.3f} ({mask.sum():2d} samples)")

# === 11. Save ===
print("\n" + "=" * 70)
print("STEP 7: SAVING ARTIFACTS")
print("=" * 70)

os.makedirs("../models", exist_ok=True)

model.save("../models/expense_mlp_best.keras")
joblib.dump(le, "../models/label_encoder_best.joblib")
joblib.dump(vectorizer, "../models/vectorizer_best.joblib")

# Save config
import json
config = {
    'dataset_size': len(df),
    'test_accuracy': float(test_acc),
    'num_classes': int(num_classes),
    'classes': list(le.classes_),
    'max_features': int(max_features),
    'confidence_threshold_recommended': 0.5,
    'training_epochs': len(history.history['accuracy'])
}

with open("../models/model_config.json", 'w') as f:
    json.dump(config, f, indent=2)

history_dict = {
    'accuracy': [float(x) for x in history.history['accuracy']],
    'val_accuracy': [float(x) for x in history.history['val_accuracy']],
    'loss': [float(x) for x in history.history['loss']],
    'val_loss': [float(x) for x in history.history['val_loss']]
}

with open("../models/training_history_best.json", 'w') as f:
    json.dump(history_dict, f, indent=2)

print("\n✅ Saved:")
print("  📦 ../models/expense_mlp_best.keras")
print("  📦 ../models/label_encoder_best.joblib")
print("  📦 ../models/vectorizer_best.joblib")
print("  📦 ../models/model_config.json")
print("  📦 ../models/training_history_best.json")

print("\n" + "=" * 70)
print("🎉 TRAINING COMPLETE!")
print("=" * 70)
print(f"\n✨ Final Test Accuracy: {test_acc*100:.2f}%")
print("\n💡 Next steps:")
print("  1. Test predictions: python test_predictions.py")
print("  2. Deploy model in your application")
print("  3. Monitor confidence scores for uncertain predictions")
