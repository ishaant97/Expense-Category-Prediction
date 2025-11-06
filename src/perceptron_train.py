# src/perceptron_train.py
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# === 1. Load data ===
df = pd.read_csv('../data/expenses.csv')
texts = df['description'].values
labels = df['category'].values

# === 2. Convert text to numeric features ===
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(texts).toarray()  # shape: (N, D)

# === 3. Encode labels (one-hot) ===
categories = np.unique(labels)
cat_to_idx = {c: i for i, c in enumerate(categories)}
y_indices = np.array([cat_to_idx[c] for c in labels])
Y = np.eye(len(categories))[y_indices]  # one-hot labels

# === 4. Split data ===
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# === 5. Initialize weights ===
D = X_train.shape[1]
C = Y_train.shape[1]
W = np.random.randn(D, C) * 0.01  # small random weights
b = np.zeros((1, C))

# === 6. Define helper functions ===
def softmax(z):
    z -= np.max(z, axis=1, keepdims=True)  # stability
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy_loss(Y_true, Y_pred):
    m = Y_true.shape[0]
    log_likelihood = -np.log(Y_pred[range(m), np.argmax(Y_true, axis=1)] + 1e-9)
    return np.sum(log_likelihood) / m

# === 7. Train with gradient descent ===
learning_rate = 0.5
epochs = 100

for epoch in range(epochs):
    # forward
    Z = np.dot(X_train, W) + b
    Y_pred = softmax(Z)

    # loss
    loss = cross_entropy_loss(Y_train, Y_pred)

    # gradient
    m = X_train.shape[0]
    dZ = (Y_pred - Y_train) / m
    dW = np.dot(X_train.T, dZ)
    db = np.sum(dZ, axis=0, keepdims=True)

    # update
    W -= learning_rate * dW
    b -= learning_rate * db

    if (epoch + 1) % 10 == 0:
        preds = np.argmax(Y_pred, axis=1)
        acc = accuracy_score(np.argmax(Y_train, axis=1), preds)
        print(f"Epoch {epoch+1:3d} | Loss: {loss:.4f} | Train Acc: {acc:.3f}")

# === 8. Evaluate on test data ===
Z_test = np.dot(X_test, W) + b
Y_pred_test = softmax(Z_test)
pred_classes = np.argmax(Y_pred_test, axis=1)
true_classes = np.argmax(Y_test, axis=1)

print("\n=== TEST PERFORMANCE ===")
print("Accuracy:", accuracy_score(true_classes, pred_classes))

# Only keep categories that actually appear in the test set
present_labels = np.unique(true_classes)
present_names = [categories[i] for i in present_labels]

print("\nClassification Report:")
print(classification_report(true_classes,
                            pred_classes,
                            labels=present_labels,
                            target_names=present_names,
                            zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(true_classes, pred_classes, labels=present_labels))