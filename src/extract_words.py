import joblib
import numpy as np

MODEL_PATH = "../features/logistic_regression_model.pkl"
VECTORIZER_PATH = "../features/tfidf_vectorizer.pkl"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

feature_names = vectorizer.get_feature_names_out()
coefs = model.coef_

if len(coefs) == 1:  # Binary classification
    coefs = coefs[0]
    top_positive_idx = np.argsort(coefs)[-20:][::-1]
    top_negative_idx = np.argsort(coefs)[:20]

    print("Top Positive Words:")
    print([feature_names[i] for i in top_positive_idx])

    print("\nTop Negative Words:")
    print([feature_names[i] for i in top_negative_idx])
else:  # Multi-class
    classes = model.classes_
    for i, class_label in enumerate(classes):
        top_idx = np.argsort(coefs[i])[-20:][::-1]
        print(f"Top words for class {class_label}:")
        print([feature_names[j] for j in top_idx])