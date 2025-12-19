
import joblib

model = joblib.load("models/spam_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

print("Email Spam Classifier Demo (type exit to quit)")

while True:
    msg = input("Enter message: ")
    if msg.lower() == "exit":
        break
    vec = vectorizer.transform([msg])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0][1]

    if pred == 1:
        print(f"SPAM (probability {prob:.2f})")
    else:
        print(f"HAM (probability {1-prob:.2f})")
