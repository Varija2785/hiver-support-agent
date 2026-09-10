# Hiver SDE Intern — AI Support Agent

Initial implementation for the Hiver take-home assignment.

## Step 1: AppleSupport extraction

The raw Kaggle file is processed in chunks so the full ~500 MB CSV does not need to be loaded into memory at once.

```bash
python src/data/extract_apple.py --input twcs.csv --output data/processed/apple_support_pairs.csv
```

The extracted dataset contains customer tweets that are explicitly linked by the Twitter dataset's `response_tweet_id` field to an `AppleSupport` response.

## Current artifact

`data/processed/apple_support_pairs.csv`

Next: exploratory analysis and data-driven intent discovery.
