import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

df = pd.read_csv("labeled_headlines.csv")

# This pipeline bundles the two steps: vectorizing text and classifying it.
text_clf_pipeline = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LogisticRegression())])

print("Training the sentiment model...")
text_clf_pipeline.fit(df['headline'], df['sentiment'])
print("Training complete.")

# save the model
joblib.dump(text_clf_pipeline, "models/traditional_sentiment_model.pkl")
print("Model saved to models/traditional_sentiment_model.pkl")

