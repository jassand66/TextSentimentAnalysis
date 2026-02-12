import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

FEATURES_DIR = "../features"

TRAIN_PATH = f"{FEATURES_DIR}/train_vectors.pkl"
TEST_PATH = f"{FEATURES_DIR}/test_vectors.pkl"

# Load vectorized data
X_train, y_train = joblib.load(TRAIN_PATH)
X_test, y_test = joblib.load(TEST_PATH)

print("Data loaded.")
print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")

# create Logistic Regression model
model = LogisticRegression(
    max_iter=1000,       # ensure convergence
    n_jobs=-1            # use all CPU cores
)

print("\nTraining model...")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluate model
print("\nModel Evaluation")
print("-------------------")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(model, f"{FEATURES_DIR}/logistic_regression_model.pkl")
print("\nModel saved.")
print(X_train.shape)