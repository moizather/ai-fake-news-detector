"""
FAKE NEWS DETECTION - MODEL TRAINING SCRIPT

Steps performed in this file:
1. Load Fake and Real news datasets
2. Add labels (1 = Fake, 0 = Real)
3. Combine datasets
4. Split into train and test data
5. Convert text into numerical features using TF-IDF
6. Train Naive Bayes model
7. Evaluate accuracy
8. Save model and vectorizer for backend use
"""

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# 1️⃣ Load datasets
print("Loading datasets...")
fake = pd.read_csv("../data/Fake.csv")
real = pd.read_csv("../data/True.csv")

# 2️⃣ Add labels
fake["label"] = 1   # Fake news
real["label"] = 0   # Real news

# 3️⃣ Combine datasets
df = pd.concat([fake, real], ignore_index=True)
df = df[["text", "label"]]

print("Total samples:", len(df))

# 4️⃣ Split data
X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# 5️⃣ TF-IDF Vectorization
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 6️⃣ Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 7️⃣ Evaluate model
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# 8️⃣ Save model and vectorizer
pickle.dump(model, open("fake_news_model.pkl", "wb"))
pickle.dump(vectorizer, open("tfidf_vectorizer.pkl", "wb"))

print("Model and vectorizer saved successfully")
