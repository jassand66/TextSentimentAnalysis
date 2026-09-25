# Text Sentiment Analysis

Classifies text messages and tweets as positive, neutral, or negative. There are two branches: `main` uses a classical ML model, `deep-learning` uses a neural network. Both run through the same cleaning, splitting, and vectorization steps, so the only real difference between them is the model.

| Branch | Model |
|---|---|
| [`main`](https://github.com/jassand66/TextSentimentAnalysis/tree/main) | Logistic Regression (scikit-learn) |
| [`deep-learning`](https://github.com/jassand66/TextSentimentAnalysis/tree/deep-learning) | Feedforward neural network (Keras) |

## Why two branches

I wanted a baseline before jumping to a neural net, and I wanted to see if the extra complexity actually helps on this dataset. Keeping the preprocessing identical across both branches makes it a fair comparison instead of just guessing.

## Pipeline

1. `src/cleaning.py` - cleans and normalizes the raw text, encodes sentiment labels (PySpark)
2. `src/split.py` - 80/20 stratified train/test split
3. `src/vectorization.py` - TF-IDF (unigrams + bigrams, max 18,000 features), fit on the training set only
4. `src/train.py` - trains and evaluates the model (this is the step that differs by branch)

On `main`, there's also `src/extract_words.py`, which pulls the logistic regression coefficients to show the words that push predictions positive or negative. It's a quick way to check the model learned something reasonable instead of just memorizing noise.

The `deep-learning` branch trains a small `Sequential` model in Keras: two dense hidden layers (256 then 128 units, ReLU, L2 regularization, 50% dropout), softmax output, trained with Adam and categorical crossentropy.

## Results

| Model | Accuracy | Notes |
|---|---|---|
| Logistic Regression (`main`) | 82 % | |
| Neural network (`deep-learning`) | 78 % | 	Slightly lower than logistic regression(unnecessary layers)|

Run `train.py` on each branch to get these numbers.

## Repo structure

```
raw_data/       original dataset
cleaned_data/   output of cleaning.py
split_data/     train/test parquet files
features/       vectorizer, vectorized features, trained models
src/            pipeline scripts
```

## Running it

```bash
cd src
python cleaning.py
python split.py
python vectorization.py
python train.py
```

Run `python extract_words.py` afterward on `main` to see the top predictive words.

## Notes / possible improvements

- Add a requirements.txt for each branch (PySpark and TensorFlow have pretty different dependencies)
- Fill in the results table with real numbers
- Re-enable and tune the early stopping callback on the deep-learning branch (currently commented out)
- Interpretability on the deep-learning branch would need a different approach than the coefficient method used on main, since that doesn't apply to a neural net - permutation importance would be a reasonable option
