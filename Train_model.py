import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv("Spam_ham.csv", encoding="latin-1")

# Keep required columns
df = df[["v1", "v2"]]
df.columns = ["Category", "Message"]

# -----------------------------
# 2. Encode labels
# -----------------------------
df["Category"] = df["Category"].map({"spam": 1, "ham": 0})

# -----------------------------
# 3. Separate X and y
# -----------------------------
x = df["Message"]
y = df["Category"]

# -----------------------------
# 4. Train-test split
# -----------------------------
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# -----------------------------
# 5. TF-IDF Vectorization
# -----------------------------
tfidf = TfidfVectorizer(stop_words="english")
x_train_tfidf = tfidf.fit_transform(x_train)
x_test_tfidf = tfidf.transform(x_test)

# -----------------------------
# 6. Train Naive Bayes model
# -----------------------------
model = MultinomialNB()
model.fit(x_train_tfidf, y_train)

# -----------------------------
# 7. Prediction & Accuracy
# -----------------------------
y_pred = model.predict(x_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# -----------------------------
# 8. Save model & vectorizer
# -----------------------------
joblib.dump(model, "spam_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

print("Model and TF-IDF vectorizer saved successfully!")
