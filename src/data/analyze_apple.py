import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "apple_support_pairs.csv"

print("=" * 70)
print("APPLE SUPPORT DATASET ANALYSIS")
print("=" * 70)

print(f"\nLoading: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

print("\nDataset shape:")
print(f"Rows:    {len(df):,}")
print(f"Columns: {len(df.columns)}")

print("\nColumns:")
for column in df.columns:
    print(f"  - {column}")

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing = df.isna().sum()

for column, count in missing.items():
    percentage = count / len(df) * 100
    print(f"{column:25s} {count:10,} ({percentage:6.2f}%)")

df["customer_text"] = df["customer_text"].fillna("").astype(str)
df["brand_text"] = df["brand_text"].fillna("").astype(str)

print("\n" + "=" * 70)
print("DUPLICATES")
print("=" * 70)

print(
    f"Duplicate customer tweets: "
    f"{df['customer_tweet_id'].duplicated().sum():,}"
)

print(
    f"Duplicate brand replies:   "
    f"{df['brand_tweet_id'].duplicated().sum():,}"
)

df["customer_length"] = df["customer_text"].str.len()
df["brand_length"] = df["brand_text"].str.len()

print("\n" + "=" * 70)
print("MESSAGE LENGTHS")
print("=" * 70)

print("\nCustomer message:")
print(df["customer_length"].describe())

print("\nBrand response:")
print(df["brand_length"].describe())

print("\n" + "=" * 70)
print("EMPTY / VERY SHORT MESSAGES")
print("=" * 70)

empty_customer = (df["customer_text"].str.strip() == "").sum()
empty_brand = (df["brand_text"].str.strip() == "").sum()

short_customer = (df["customer_length"] < 10).sum()
short_brand = (df["brand_length"] < 10).sum()

print(f"Empty customer messages:      {empty_customer:,}")
print(f"Empty brand responses:        {empty_brand:,}")
print(f"Customer messages <10 chars:  {short_customer:,}")
print(f"Brand responses <10 chars:    {short_brand:,}")

print("\n" + "=" * 70)
print("SAMPLE CUSTOMER MESSAGES")
print("=" * 70)

sample = df["customer_text"].sample(
    min(20, len(df)),
    random_state=42
)

for i, text in enumerate(sample, 1):
    print(f"\n{i}. {text}")

print("\n" + "=" * 70)
print("SAMPLE CONVERSATION PAIRS")
print("=" * 70)

sample_pairs = df.sample(
    min(10, len(df)),
    random_state=42
)

for i, row in enumerate(sample_pairs.itertuples(), 1):
    print(f"\n--- Example {i} ---")
    print(f"Customer: {row.customer_text}")
    print(f"Apple:    {row.brand_text}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

stats = {
    "total_pairs": len(df),
    "unique_customers": df["customer_author_id"].nunique(),
    "unique_customer_tweets": df["customer_tweet_id"].nunique(),
    "unique_brand_replies": df["brand_tweet_id"].nunique(),
    "duplicate_customer_tweets": df["customer_tweet_id"].duplicated().sum(),
    "duplicate_brand_replies": df["brand_tweet_id"].duplicated().sum(),
    "empty_customer_messages": empty_customer,
    "empty_brand_responses": empty_brand,
    "avg_customer_length": df["customer_length"].mean(),
    "avg_brand_length": df["brand_length"].mean(),
}

for key, value in stats.items():
    if isinstance(value, float):
        print(f"{key:30s}: {value:.2f}")
    else:
        print(f"{key:30s}: {value:,}")

print("\nAnalysis complete.")