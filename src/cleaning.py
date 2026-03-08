import os
import pandas as pd
from cleantext import clean
import emoji
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# Paths
RAW_CSV = "../raw_data/messages.csv"
OUTPUT_PARQUET = "../cleaned_data/cleaned_messages.parquet"
os.makedirs(os.path.dirname(OUTPUT_PARQUET), exist_ok=True)

# Load data
df = pd.read_csv(RAW_CSV)

# Drop missing or duplicate rows
df = df.dropna(subset=['text', 'selected_text']).drop_duplicates()

# Define text cleaning function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # Use cleantext for basic cleaning
    cleaned = clean(text)
    
    # Additional manual cleaning
    cleaned = cleaned.lower()
    cleaned = cleaned.strip()
    
    return cleaned

# Apply cleaning
df['text'] = df['text'].apply(clean_text)
df['selected_text'] = df['selected_text'].apply(clean_text)

# Encode labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['sentiment'])

# Convert text to features
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Save cleaned data
df[['text', 'selected_text', 'sentiment', 'label']].to_parquet(OUTPUT_PARQUET, index=False)
print(f"Cleaned data saved to {OUTPUT_PARQUET}")