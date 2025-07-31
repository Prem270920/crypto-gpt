import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib

df = pd.read_csv("../labeled_headlines.csv")

df.dropna(subset=['headline'], inplace=True)

X_train, X_test, y_train, y_test = train_test_split(df['headline'], df['sentiment'], test_size=0.2, random_state=42)

pipeline = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LogisticRegression()),])

#parameters for GridSearchCV to test with single words and word pairs
parameters = {'tfidf__ngram_range': [(1, 1), (1, 2)], 'clf': [LogisticRegression(max_iter=1000), LinearSVC(max_iter=1000)]}

#Performing the Grid Search
grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, cv=5, verbose=1)

grid_search.fit(X_train, y_train)

#Evaluate the best Model
best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print(f"Overall Accuracy: {accuracy_score(y_test, predictions):.2%}")

joblib.dump(best_model, "../models/traditional_sentiment_model.pkl")


