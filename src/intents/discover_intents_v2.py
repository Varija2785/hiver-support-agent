import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "apple_support_pairs.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("APPLE SUPPORT — IMPROVED INTENT DISCOVERY")
print("=" * 80)

print(f"\nLoading: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

df["customer_text"] = (
    df["customer_text"]
    .fillna("")
    .astype(str)
)


print(f"Original rows: {len(df):,}")


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text):
    text = text.lower()

    # URLs
    text = re.sub(r"https?://\S+", " ", text)

    # Twitter mentions
    text = re.sub(r"@\w+", " ", text)

    # HTML entities
    text = re.sub(r"&\w+;", " ", text)

    # Keep alphanumeric text
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


df["clean_text"] = df["customer_text"].apply(clean_text)


# ============================================================
# REMOVE GENERIC / LOW-INFORMATION MESSAGES
# ============================================================

GENERIC_MESSAGES = {
    "thanks",
    "thank you",
    "thankyou",
    "thank",
    "thanks so much",
    "thanks a lot",
    "okay",
    "ok",
    "yes",
    "no",
    "yep",
    "yup",
    "yeah",
    "great",
    "awesome",
    "perfect",
    "cool",
    "sure",
    "done",
    "got it",
    "will do",
    "i did",
    "never mind",
    "nevermind",
}

before = len(df)

df = df[
    ~df["clean_text"].isin(GENERIC_MESSAGES)
].copy()

# Remove very short messages.
# These are usually confirmations, device names,
# versions, or fragments rather than complete intents.
df = df[
    df["clean_text"].str.len() >= 30
].copy()

# Remove duplicate wording.
df = df.drop_duplicates(
    subset=["clean_text"]
).reset_index(drop=True)

print(
    f"After removing low-information messages: "
    f"{len(df):,}"
)

print(
    f"Removed: {before - len(df):,}"
)


# ============================================================
# TF-IDF
# ============================================================

print("\nBuilding TF-IDF representation...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=8,
    max_df=0.90,
    max_features=30000,
    sublinear_tf=True,
)

X = vectorizer.fit_transform(df["clean_text"])

print(f"TF-IDF matrix: {X.shape}")


# ============================================================
# CLUSTERING
# ============================================================

N_CLUSTERS = 15

print(
    f"\nClustering into {N_CLUSTERS} candidate intent groups..."
)

model = MiniBatchKMeans(
    n_clusters=N_CLUSTERS,
    random_state=42,
    batch_size=2048,
    n_init=10,
)

df["cluster"] = model.fit_predict(X)


# ============================================================
# TOP TERMS
# ============================================================

terms = np.array(
    vectorizer.get_feature_names_out()
)

print("\n" + "=" * 80)
print("CANDIDATE INTENT GROUPS")
print("=" * 80)

cluster_info = []

for cluster_id in range(N_CLUSTERS):

    mask = df["cluster"] == cluster_id

    cluster_size = int(mask.sum())

    center = model.cluster_centers_[cluster_id]

    top_indices = center.argsort()[::-1][:15]

    top_terms = terms[top_indices].tolist()

    cluster_info.append(
        {
            "cluster": cluster_id,
            "size": cluster_size,
            "percentage": (
                cluster_size / len(df) * 100
            ),
            "top_terms": ", ".join(top_terms),
        }
    )

    print(
        f"\nCluster {cluster_id}"
        f" | {cluster_size:,} messages"
        f" | {cluster_size / len(df) * 100:.2f}%"
    )

    print(
        "Terms:",
        ", ".join(top_terms)
    )


# ============================================================
# REPRESENTATIVE EXAMPLES
# ============================================================

print("\n" + "=" * 80)
print("REPRESENTATIVE EXAMPLES")
print("=" * 80)

representative_rows = []

for cluster_id in range(N_CLUSTERS):

    cluster_indices = np.where(
        df["cluster"].values == cluster_id
    )[0]

    if len(cluster_indices) == 0:
        continue

    cluster_matrix = X[cluster_indices]

    center = model.cluster_centers_[cluster_id]

    similarities = cosine_similarity(
        cluster_matrix,
        center.reshape(1, -1)
    ).ravel()

    # Closest examples to the cluster center
    top_positions = np.argsort(
        similarities
    )[::-1][:10]

    selected_indices = cluster_indices[
        top_positions
    ]

    print(
        "\n" + "-" * 80
    )

    print(
        f"CLUSTER {cluster_id} "
        f"({len(cluster_indices):,} messages)"
    )

    print("-" * 80)

    for rank, idx in enumerate(
        selected_indices,
        1
    ):

        customer = df.iloc[idx]["customer_text"]

        print(
            f"{rank}. {customer}"
        )

        representative_rows.append(
            {
                "cluster": cluster_id,
                "rank": rank,
                "customer_text": customer,
                "brand_text": df.iloc[idx]["brand_text"],
            }
        )


# ============================================================
# SAVE CLUSTERS
# ============================================================

cluster_df = df[
    [
        "customer_tweet_id",
        "brand_tweet_id",
        "customer_text",
        "brand_text",
        "cluster",
    ]
].copy()

cluster_path = (
    OUTPUT_DIR
    / "intent_discovery_v2.csv"
)

cluster_df.to_csv(
    cluster_path,
    index=False
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    cluster_info
)

summary_path = (
    OUTPUT_DIR
    / "intent_discovery_v2_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)


# ============================================================
# SAVE REPRESENTATIVE EXAMPLES
# ============================================================

examples_df = pd.DataFrame(
    representative_rows
)

examples_path = (
    OUTPUT_DIR
    / "intent_discovery_v2_examples.csv"
)

examples_df.to_csv(
    examples_path,
    index=False
)


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 80)
print("DISCOVERY COMPLETE")
print("=" * 80)

print(f"\nSaved:")
print(cluster_path)
print(summary_path)
print(examples_path)