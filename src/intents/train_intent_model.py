import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "apple_support_labeled.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "intent_classifier.pkl"


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

print("=" * 80)
print("HIVER SUPPORT AGENT — TRAIN INTENT MODEL")
print("=" * 80)

print(f"\nLoading: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

print(f"Total rows: {len(df):,}")


# ---------------------------------------------------------
# Clean data
# ---------------------------------------------------------

df["customer_text"] = (
    df["customer_text"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df["intent"] = (
    df["intent"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df = df[
    (df["customer_text"] != "")
    & (df["intent"] != "")
].copy()

print(f"Usable rows: {len(df):,}")


# ---------------------------------------------------------
# Distribution
# ---------------------------------------------------------

print("\nIntent distribution:")

print(
    df["intent"]
    .value_counts()
    .to_string()
)


# ---------------------------------------------------------
# Train / validation split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    df["customer_text"],
    df["intent"],
    test_size=0.20,
    random_state=42,
    stratify=df["intent"],
)

print("\nTraining rows:", len(X_train))
print("Validation rows:", len(X_test))


# ---------------------------------------------------------
# TF-IDF + Logistic Regression
# ---------------------------------------------------------

print("\nBuilding model...")

model = Pipeline(
    [
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                min_df=2,
                max_features=30000,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
            ),
        ),
    ]
)


# ---------------------------------------------------------
# Train
# ---------------------------------------------------------

print("\nTraining...")

model.fit(X_train, y_train)


# ---------------------------------------------------------
# Evaluate
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("MODEL EVALUATION")
print("=" * 80)

predictions = model.predict(X_test)

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0,
    )
)


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print("=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)

print(f"\nSaved model:")
print(MODEL_PATH)