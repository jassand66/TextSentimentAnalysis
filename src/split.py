import pandas as pd
from sklearn.model_selection import train_test_split
import os
import glob

INPUT_DIR = "../cleaned_data"
OUTPUT_DIR = "../split_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load cleaned files
parquet_files = glob.glob(os.path.join(INPUT_DIR, "*.parquet"))

if not parquet_files:
    raise FileNotFoundError("No parquet files found in cleaned_data folder.")

df_list = [pd.read_parquet(file) for file in parquet_files]
df = pd.concat(df_list, ignore_index=True)

print(f"Loaded {len(parquet_files)} parquet files")
print(f"Total rows: {len(df)}")

# Double check field name
df["sentiment"] = df["sentiment"].str.lower().str.strip()

expected_labels = {"positive", "neutral", "negative"}
actual_labels = set(df["sentiment"].unique())

if not actual_labels.issubset(expected_labels):
    raise ValueError(f"Unexpected labels found: {actual_labels - expected_labels}")

# Perform data split(test/train)
X = df.drop(columns=["sentiment"])
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.15,
    stratify=y,
    random_state=42
)

train_df = X_train.copy()
train_df["sentiment"] = y_train

test_df = X_test.copy()
test_df["sentiment"] = y_test


train_df.to_parquet(f"{OUTPUT_DIR}/train.parquet", index=False)
test_df.to_parquet(f"{OUTPUT_DIR}/test.parquet", index=False)

print("Split complete!")
print(f"Train size: {len(train_df)}")
print(f"Test size: {len(test_df)}")

print("\nClass distribution (Train):")
print(train_df["sentiment"].value_counts(normalize=True))

print("\nClass distribution (Test):")
print(test_df["sentiment"].value_counts(normalize=True))
