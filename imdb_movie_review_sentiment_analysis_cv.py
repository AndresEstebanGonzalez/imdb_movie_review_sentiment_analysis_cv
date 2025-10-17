"""
IMDB Sentiment Analysis with Cross-Validation

Builds a sentiment analysis pipeline using Logistic Regression
and TF-IDF vectorization. Includes text cleaning, 3-fold
cross-validation, and reports the Accuracy
for performance evaluation.
"""
#Import re
import re
#Import save pipeline
from joblib import dump
#Import Pandas
import pandas as pd
#Import split and CV
from sklearn.model_selection import StratifiedKFold, cross_validate
#Import Pipeline
from sklearn.pipeline import Pipeline
#Import function transformer
from sklearn.preprocessing import FunctionTransformer
#Import vectroizer
from sklearn.feature_extraction.text import TfidfVectorizer
#Import model
from sklearn.linear_model import LogisticRegression

#Import data
DATA_PATH = "data/IMDB Dataset.csv"
imdb_df = pd.read_csv(DATA_PATH, encoding="latin-1")
#Map sentiment values
imdb_df["sentiment"] = imdb_df["sentiment"].map({"negative":0, "positive":1})
#Clean text function
def clean_text(reviews):
    '''Removes punctuation, HTML tags and strips reviews'''
    cleaned = []
    for text in reviews:
        text = text.strip()
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"[^A-Za-z0-9'\s]", "", text)
        cleaned.append(text)
    return cleaned
#Pipeline
pipeline = Pipeline([
    ("cleaner", FunctionTransformer(clean_text, validate=False)),
    ("vectorizer", TfidfVectorizer(
        max_features=20000,
        ngram_range=(1,2),
        lowercase=True,
        stop_words="english",
        min_df= 2,
        max_df= 0.9,
        sublinear_tf=True)),
    ("model", LogisticRegression(
        max_iter=1000,
        random_state=1,
        solver="liblinear"))
])
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=1
)
scoring = ["accuracy", "f1", "precision", "recall"]
#Cross Validation score
cv_scores = cross_validate(
    pipeline,
    imdb_df["review"],
    imdb_df["sentiment"],
    cv=cv,
    scoring= scoring,
    n_jobs=-1,
    return_train_score=False
)

for metric in scoring:
    metric_results = cv_scores[f"test_{metric}"]
    mean = metric_results.mean()
    std = metric_results.std()
    print(
        f"{metric.capitalize():<10} "
        f"mean±std: {mean:.4f} ± {std:.4f}  -> {metric_results}"
    )

#Save pipeline
PIPELINE_PATH = "models/imdb_logreg_cv_pipeline.joblib"
dump(pipeline, PIPELINE_PATH)
