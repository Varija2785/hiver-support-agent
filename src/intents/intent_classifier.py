import re
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "apple_support_pairs.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "apple_support_labeled.csv"


# ---------------------------------------------------------
# Intent keywords
# ---------------------------------------------------------

INTENT_KEYWORDS = {
    "battery_issue": [
        "battery",
        "battery drain",
        "battery draining",
        "battery life",
        "charge",
        "charging",
        "not charging",
    ],

    "ios_update_issue": [
        "ios update",
        "software update",
        "ios 11",
        "update",
        "updated",
        "latest update",
        "new update",
        "after update",
    ],

    "app_issue": [
        "app",
        "apps",
        "application",
        "app not working",
        "app crashes",
        "app won't open",
        "app freezing",
    ],

    "iphone_freezing": [
        "freezing",
        "freeze",
        "freezes",
        "frozen",
        "lag",
        "lagging",
        "slow",
        "unresponsive",
        "restart",
        "restarting",
    ],

    "itunes_issue": [
        "itunes",
        "sync",
        "itunes error",
        "itunes restore",
        "itunes backup",
        "itunes connect",
    ],

    "apple_id_issue": [
        "apple id",
        "account",
        "login",
        "sign in",
        "password",
        "authentication",
        "icloud account",
    ],

    "wifi_connectivity": [
        "wifi",
        "wi-fi",
        "internet",
        "network",
        "connection",
        "connected",
        "disconnect",
        "bluetooth",
    ],

    "control_center": [
        "control center",
        "control centre",
        "swipe up",
        "control panel",
        "wifi control",
        "bluetooth control",
        "music control",
    ],

    "apple_music": [
        "apple music",
        "music",
        "songs",
        "song",
        "music library",
        "playlist",
        "music app",
        "sync music",
        "headphones",
    ],

    "touch_screen": [
        "touch",
        "touch screen",
        "touchscreen",
        "screen",
        "touch id",
        "fingerprint",
        "not responding",
    ],

    "facetime": [
        "facetime",
        "face time",
        "video call",
        "facetime call",
        "facetime not ringing",
    ],

    "iphone_device_issue": [
        "iphone",
        "phone",
        "device",
        "iphone issue",
        "iphone problem",
        "phone problem",
    ],

    "ipad_issue": [
        "ipad",
        "ipad issue",
        "ipad problem",
        "ipad not working",
    ],

    "feedback": [
        "feedback",
        "suggestion",
        "suggest",
        "feature request",
        "comment",
    ],

    "support_followup": [
        "waiting",
        "waiting for response",
        "response",
        "reply",
        "dm",
        "direct message",
        "message",
        "sent you a dm",
    ],
}


# ---------------------------------------------------------
# Cleaning
# ---------------------------------------------------------

def clean_text(text):
    text = str(text).lower()

    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"&\w+;", " ", text)

    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

print("=" * 80)
print("HIVER SUPPORT AGENT — INTENT CLASSIFICATION")
print("=" * 80)

print(f"\nLoading: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

print(f"Original rows: {len(df):,}")


df["customer_text"] = (
    df["customer_text"]
    .fillna("")
    .astype(str)
)

df["clean_text"] = df["customer_text"].apply(clean_text)


# ---------------------------------------------------------
# Classification
# ---------------------------------------------------------

def classify_intent(text):
    scores = {}

    for intent, keywords in INTENT_KEYWORDS.items():

        score = 0

        for keyword in keywords:
            keyword_clean = clean_text(keyword)

            if keyword_clean in text:
                # Longer phrases receive more weight
                score += len(keyword_clean.split())

        scores[intent] = score

    best_intent = max(scores, key=scores.get)

    if scores[best_intent] == 0:
        return "iphone_device_issue"

    return best_intent


print("\nClassifying messages...")

df["intent"] = df["clean_text"].apply(classify_intent)


# ---------------------------------------------------------
# Confidence / score
# ---------------------------------------------------------

def get_score(text, intent):
    score = 0

    for keyword in INTENT_KEYWORDS[intent]:
        keyword_clean = clean_text(keyword)

        if keyword_clean in text:
            score += len(keyword_clean.split())

    return score


df["intent_score"] = df.apply(
    lambda row: get_score(
        row["clean_text"],
        row["intent"],
    ),
    axis=1,
)


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

output_columns = [
    "customer_tweet_id",
    "brand_tweet_id",
    "customer_text",
    "brand_text",
    "intent",
    "intent_score",
]

available_columns = [
    column
    for column in output_columns
    if column in df.columns
]

result = df[available_columns].copy()

result.to_csv(
    OUTPUT_PATH,
    index=False,
)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("INTENT DISTRIBUTION")
print("=" * 80)

distribution = (
    result["intent"]
    .value_counts()
    .rename_axis("intent")
    .reset_index(name="count")
)

distribution["percentage"] = (
    distribution["count"] / len(result) * 100
)

print(distribution.to_string(index=False))


print("\n" + "=" * 80)
print("DONE")
print("=" * 80)

print(f"\nSaved: {OUTPUT_PATH}")