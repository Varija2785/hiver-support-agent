import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer


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


# ---------------------------------------------------------
# Load
# ---------------------------------------------------------

print("Loading AppleSupport data...")

df = pd.read_csv(DATA_PATH)

df["customer_text"] = (
    df["customer_text"]
    .fillna("")
    .astype(str)
)


# ---------------------------------------------------------
# Basic text cleaning
# ---------------------------------------------------------

def clean_text(text):
    text = text.lower()

    # Remove URLs
    text = re.sub(r"https?://\S+", " ", text)

    # Remove Twitter mentions
    text = re.sub(r"@\w+", " ", text)

    # Remove HTML entities
    text = re.sub(r"&\w+;", " ", text)

    # Keep words/numbers
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


df["clean_text"] = df["customer_text"].apply(clean_text)

# Remove exact duplicate customer messages for discovery.
# This prevents repeated wording from dominating clusters.
df = df.drop_duplicates(subset=["clean_text"]).reset_index(drop=True)

print(f"Unique customer messages for discovery: {len(df):,}")


# ---------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------

print("\nBuilding TF-IDF representation...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.95,
    max_features=20000,
    sublinear_tf=True,
)

X = vectorizer.fit_transform(df["clean_text"])

print(f"TF-IDF matrix: {X.shape}")


# ---------------------------------------------------------
# Clustering
# ---------------------------------------------------------

N_CLUSTERS = 12

print(f"\nClustering into {N_CLUSTERS} candidate groups...")

model = MiniBatchKMeans(
    n_clusters=N_CLUSTERS,
    random_state=42,
    batch_size=1024,
    n_init=10,
)

df["cluster"] = model.fit_predict(X)


# ---------------------------------------------------------
# Top terms per cluster
# ---------------------------------------------------------

terms = np.array(vectorizer.get_feature_names_out())

print("\n" + "=" * 80)
print("TOP TERMS BY CLUSTER")
print("=" * 80)

cluster_info = []

for cluster_id in range(N_CLUSTERS):

    mask = df["cluster"] == cluster_id

    cluster_size = mask.sum()

    center = model.cluster_centers_[cluster_id]

    top_indices = center.argsort()[::-1][:15]

    top_terms = terms[top_indices].tolist()

    cluster_info.append(
        {
            "cluster": cluster_id,
            "size": cluster_size,
            "percentage": cluster_size / len(df) * 100,
            "top_terms": ", ".join(top_terms),
        }
    )

    print(
        f"\nCluster {cluster_id}"
        f" | {cluster_size:,} messages"
        f" | {cluster_size / len(df) * 100:.2f}%"
    )

    print("Terms:", ", ".join(top_terms))


# ---------------------------------------------------------
# Representative examples
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("REPRESENTATIVE CUSTOMER MESSAGES")
print("=" * 80)

for cluster_id in range(N_CLUSTERS):

    cluster_rows = df[df["cluster"] == cluster_id]

    print("\n" + "-" * 80)
    print(f"CLUSTER {cluster_id} ({len(cluster_rows):,} messages)")
    print("-" * 80)

    # Deterministic examples
    examples = cluster_rows.sample(
        min(8, len(cluster_rows)),
        random_state=42,
    )

    for i, text in enumerate(examples["customer_text"], 1):
        print(f"{i}. {text}")


# ---------------------------------------------------------
# Save discovery dataset
# ---------------------------------------------------------

cluster_df = df[
    [
        "customer_tweet_id",
        "brand_tweet_id",
        "customer_text",
        "brand_text",
        "cluster",
    ]
].copy()

cluster_path = OUTPUT_DIR / "intent_discovery_clusters.csv"

cluster_df.to_csv(
    cluster_path,
    index=False,
)


# ---------------------------------------------------------
# Save cluster summary
# ---------------------------------------------------------

summary_df = pd.DataFrame(cluster_info)

summary_path = OUTPUT_DIR / "intent_cluster_summary.csv"

summary_df.to_csv(
    summary_path,
    index=False,
)


print("\n" + "=" * 80)
print("DONE")
print("=" * 80)

print(f"Saved: {cluster_path}")
print(f"Saved: {summary_path}")