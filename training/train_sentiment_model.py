import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib

df = pd.read_csv("../labeled_headlines.csv")

df.dropna(subset=['headline'], inplace=True)

X_train, X_test, y_train, y_test = train_test_split(df['headline'], df['sentiment'], test_size=0.2, random_state=42)

# This pipeline bundles the two steps: vectorizing text and classifying it.
text_clf_pipeline = Pipeline([('tfidf', TfidfVectorizer()),('clf', LogisticRegression(max_iter=1000))])

text_clf_pipeline.fit(X_train, y_train) 

predictions = text_clf_pipeline.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print(f"Overall Accuracy: {accuracy_score(y_test, predictions):.2%}")

joblib.dump(text_clf_pipeline, "models/traditional_sentiment_model.pkl")


