import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

# Dataset path
DATA_PATH = os.path.join("..", "data", "spam.csv")

# Read CSV with BOM-safe encoding
df = pd.read_csv(
    DATA_PATH,
    encoding="utf-8-sig"   
)

# Normalize column names
df.columns = df.columns.str.strip().str.lower()

# Rename BOM-affected column safely
df.rename(columns=lambda x: x.replace("\ufeff", ""), inplace=True)

# Select required columns
df = df[["label", "message"]]

# Map labels to numbers
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Drop missing values
df.dropna(inplace=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df["message"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# Vectorization
vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# Save Model and Vectorizer

MODEL_DIR = os.path.join("..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(model, os.path.join(MODEL_DIR, "spam_model.pkl"))
joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.pkl"))

print("\n✅ Model and vectorizer saved successfully!")
print("📁 Saved in:", os.path.abspath(MODEL_DIR))
