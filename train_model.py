
import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Dataset location
DATA_PATH = "data/SMSSpamCollection"

# Check dataset
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        "Dataset not found! Put SMSSpamCollection inside the data folder."
    )

# Load dataset
df = pd.read_csv(
    DATA_PATH,
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8"
)

# Clean data
df = df.dropna()
df["label"] = df["label"].map({"ham": 0, "spam": 1})
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

X = df["message"]
y = df["label"]

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create machine learning pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

# Train model
model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)

print("Model training completed!")
print(f"Total messages: {len(df)}")
print(f"Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["Ham (Normal)", "Spam"]
))

# Save trained model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/spam_model.pkl")

print("\nModel saved successfully!")
print("File location: models/spam_model.pkl")
