import os
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

FEATURES_DIR = "../features"

TRAIN_PATH = f"{FEATURES_DIR}/train_vectors.pkl"
TEST_PATH = f"{FEATURES_DIR}/test_vectors.pkl"
os.makedirs(FEATURES_DIR, exist_ok=True)

# ----------------------------
# 1. Load vectorized data
# ----------------------------
X_train, y_train = joblib.load(TRAIN_PATH)
X_test, y_test = joblib.load(TEST_PATH)

# Convert sparse TF-IDF matrices to dense arrays
X_train = X_train.toarray()
X_test = X_test.toarray()

print("Data loaded.")
print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")

# ----------------------------
# 2. Encode labels
# ----------------------------
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# One-hot encoding for categorical crossentropy
y_train_cat = to_categorical(y_train_enc)
y_test_cat = to_categorical(y_test_enc)

# ----------------------------
# 3. Build deep learning model
# ----------------------------
model = Sequential([
    Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(y_train_cat.shape[1], activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nTraining deep learning model...")
history = model.fit(
    X_train, y_train_cat,
    validation_split=0.1,
    epochs=10,
    batch_size=32,
    verbose=1
)

# ----------------------------
# 4. Evaluate model
# ----------------------------
y_pred_probs = model.predict(X_test)
y_pred = y_pred_probs.argmax(axis=1)

print("\nModel Evaluation")
print("-------------------")
print(f"Accuracy: {accuracy_score(y_test_enc, y_pred):.4f}\n")
print("Classification Report:")
print(classification_report(y_test_enc, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test_enc, y_pred))

# ----------------------------
# 5. Save model
# ----------------------------
model_path = f"{FEATURES_DIR}/deep_learning_model.h5"
model.save(model_path)
print(f"\nModel saved to {model_path}")