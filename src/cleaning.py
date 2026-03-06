from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, udf
from pyspark.sql.types import StringType
from pyspark.ml.feature import StringIndexer, Tokenizer, StopWordsRemover, CountVectorizer

# Simple emoji replacement map
emoji_map = {
    ":)": "smile",
    ":-)": "smile",
    ":(": "sad",
    ":-(": "sad",
    ";)": "wink",
    "<3": "heart"
}

# Simple common typo fixes, look into Python library for spell fixing
typo_map = {
    "juss": "just",
    "berkeleyy": "berkeley",
    "donbt": "don't",
    "wierd": "weird",
    "obesed": "obese",
    #Add more here when needed
}

def replace_emojis_and_typos(text):
    for emoji, token in emoji_map.items():
        text = text.replace(emoji, f" {token} ")
    for typo, correction in typo_map.items():
        text = text.replace(typo, correction)
    return text

# Register UDF
replace_udf = udf(replace_emojis_and_typos, StringType())

def clean_sentiment_data(input_csv, output_parquet):
    spark = SparkSession.builder.appName("SentimentDataCleaning").getOrCreate()

    df = spark.read.csv(input_csv, header=True, inferSchema=True)

    df = df.dropna(subset=['text', 'selected_text']).dropDuplicates()

    # Lowercase
    df = df.withColumn('text', lower(col('text')))
    df = df.withColumn('selected_text', lower(col('selected_text')))

    # Remove URLs
    df = df.withColumn('text', regexp_replace('text', r'http\S+', ''))
    df = df.withColumn('selected_text', regexp_replace('selected_text', r'http\S+', ''))

    # Remove extra whitespace
    df = df.withColumn('text', regexp_replace('text', '\s+', ' '))
    df = df.withColumn('selected_text', regexp_replace('selected_text', '\s+', ' '))

    # Replace emojis and fix typos
    df = df.withColumn('text', replace_udf(col('text')))
    df = df.withColumn('selected_text', replace_udf(col('selected_text')))

    # Encode sentiment labels
    indexer = StringIndexer(inputCol='sentiment', outputCol='label')
    df = indexer.fit(df).transform(df)

    # Tokenization
    tokenizer = Tokenizer(inputCol='text', outputCol='words')
    df = tokenizer.transform(df)

    # Remove stop words
    remover = StopWordsRemover(inputCol='words', outputCol='filtered')
    df = remover.transform(df)

    # Convert to features using Bag-of-Words
    vectorizer = CountVectorizer(inputCol='filtered', outputCol='features')
    vector_model = vectorizer.fit(df)
    df = vector_model.transform(df)

    df.write.parquet(output_parquet, mode='overwrite')

    spark.stop()
    print(f"Data cleaned and saved to {output_parquet}")


if __name__ == "__main__":
    input_csv = "../raw_data/messages.csv"
    output_parquet = "../cleaned_data/cleaned_messages.parquet"
    clean_sentiment_data(input_csv, output_parquet)