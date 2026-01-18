# src/clean_data.pypop
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, length
from pyspark.ml.feature import StringIndexer, Tokenizer, StopWordsRemover, CountVectorizer

def clean_sentiment_data(input_csv, output_parquet):
    # 1. Initialize Spark session
    spark = SparkSession.builder.appName("SentimentDataCleaning").getOrCreate()

    # 2. Load CSV from same folder
    df = spark.read.csv(input_csv, header=True, inferSchema=True)

    # 3. Drop rows with nulls and duplicates
    df = df.dropna(subset=['text', 'selected_text']).dropDuplicates()

    # 4. Text cleaning 
    # Lowercase
    df = df.withColumn('text', lower(col('text')))
    df = df.withColumn('selected_text', lower(col('selected_text')))
    # Remove extra whitespace
    df = df.withColumn('text', regexp_replace('text', '\s+', ' '))
    df = df.withColumn('selected_text', regexp_replace('selected_text', '\s+', ' '))

    # 5. Encode sentiment labels as numerics
    indexer = StringIndexer(inputCol='sentiment', outputCol='label')
    df = indexer.fit(df).transform(df)

    # 6. Tokenization 
    tokenizer = Tokenizer(inputCol='text', outputCol='words')
    df = tokenizer.transform(df)

    # Remove stop words
    remover = StopWordsRemover(inputCol='words', outputCol='filtered')
    df = remover.transform(df)


    # 7. Convert to features using Bag-of-Words
    vectorizer = CountVectorizer(inputCol='filtered', outputCol='features')
    vector_model = vectorizer.fit(df)
    df = vector_model.transform(df)

    # 8. Save cleaned data as Parquet (in src/cleaned_data.parquet)
    df.write.parquet(output_parquet, mode='overwrite')

    spark.stop()
    print(f"Data cleaned and saved to {output_parquet}")


if __name__ == "__main__":
    input_csv = "../raw_data/messages.csv"             
    output_parquet = "cleaned_messages.parquet"  
    clean_sentiment_data(input_csv, output_parquet)
