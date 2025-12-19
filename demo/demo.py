import joblib
import os

# Get project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model paths
MODEL_PATH = os.path.join(BASE_DIR, "models", "spam_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "vectorizer.pkl")

# Load model and vectorizer
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

print("\n📧 Email Spam Classifier Demo")
print("Type 'exit' to quit\n")

while True:
    msg = input("Enter email message: ")
    if msg.lower() == "exit":
        print("Exiting demo...")
        break

    vec = vectorizer.transform([msg])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0][1]

    if pred == 1:
        print(f"🚨 SPAM (confidence: {prob:.2f})\n")
    else:
        print(f"✅ HAM (confidence: {1 - prob:.2f})\n")
