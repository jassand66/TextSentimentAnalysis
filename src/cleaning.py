import os
import pandas as pd
from cleantext import clean
from sklearn.preprocessing import LabelEncoder

RAW_CSV = "../raw_data/messages.csv"
OUTPUT_PARQUET = "../cleaned_data/cleaned_messages.parquet"

os.makedirs(os.path.dirname(OUTPUT_PARQUET), exist_ok=True)

df = pd.read_csv(RAW_CSV)

df = df.dropna(subset=["text", "selected_text"]).drop_duplicates()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    cleaned = clean(text)
    cleaned = cleaned.lower().strip()
    return cleaned

df["text"] = df["text"].apply(clean_text)
df["selected_text"] = df["selected_text"].apply(clean_text)

le = LabelEncoder()
df["label"] = le.fit_transform(df["sentiment"])

df[["text", "selected_text", "sentiment", "label"]].to_parquet(
    OUTPUT_PARQUET,
    index=False
)

print(f"Cleaned data saved to {OUTPUT_PARQUET}")