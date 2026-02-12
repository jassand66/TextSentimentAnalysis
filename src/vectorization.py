import pandas as pd
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer




INPUT_DIR = "../split_data"
OUTPUT_DIR = "../features"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_PATH = f"{INPUT_DIR}/train.parquet"
TEST_PATH = f"{INPUT_DIR}/test.parquet"

# Load Data 
train_df = pd.read_parquet(TRAIN_PATH)
test_df = pd.read_parquet(TEST_PATH)

X_train_text = train_df["text"]      # change if your column name differs
X_test_text = test_df["text"]

y_train = train_df["sentiment"]
y_test = test_df["sentiment"]

print("Data loaded.")
print(f"Train samples: {len(X_train_text)}")
print(f"Test samples: {len(X_test_text)}")

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(
    max_features=10000,       
    ngram_range=(1,2),        # unigrams + bigrams
    stop_words="english"      # remove standard English stopwords
)

# FIT ONLY ON TRAIN DATA
X_train_vectors = vectorizer.fit_transform(X_train_text)


X_test_vectors = vectorizer.transform(X_test_text)

print("Vectorization complete.")
print(f"Train vector shape: {X_train_vectors.shape}")
print(f"Test vector shape: {X_test_vectors.shape}")

# Save Outputs
joblib.dump(vectorizer, f"{OUTPUT_DIR}/tfidf_vectorizer.pkl")
joblib.dump((X_train_vectors, y_train), f"{OUTPUT_DIR}/train_vectors.pkl")
joblib.dump((X_test_vectors, y_test), f"{OUTPUT_DIR}/test_vectors.pkl")

print("Features and vectorizer saved.")
