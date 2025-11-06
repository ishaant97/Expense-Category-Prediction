# src/mlp_train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf

# === 1. Load data ===
df = pd.read_csv('../data/expenses.csv')
texts = df['description'].astype(str).tolist()
labels = df['category'].astype(str).tolist()

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
    tf.keras.layers.Input(shape=(X_train.shape[1],)),    # input layer
    tf.keras.layers.Dense(64, activation='relu'),         # hidden layer 1
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),         # hidden layer 2
    tf.keras.layers.Dense(num_classes, activation='softmax')  # output layer
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
y_pred = model.predict(X_test).argmax(axis=1)

print("\n=== TEST PERFORMANCE ===")
print("Test Accuracy:", accuracy_score(y_test, y_pred))

# figure out which labels actually appear in y_test
present_labels = np.unique(y_test)
present_names = [le.classes_[i] for i in present_labels]

print("\nClassification Report:")
print(classification_report(y_test, y_pred,
                            labels=present_labels,
                            target_names=present_names,
                            zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=present_labels))
