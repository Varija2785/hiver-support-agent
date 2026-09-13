# Hiver SDE Intern — AI Support Agent 🤖

An AI-powered customer support agent built as a Hiver SDE internship take-home project. The system processes AppleSupport customer conversations, discovers support intents, classifies incoming messages, and retrieves relevant responses.

## 🎯 Project Overview

The project is built around the Twitter Customer Support dataset and focuses on creating a data-driven support workflow.

The pipeline is designed to:

1. Extract AppleSupport customer-response pairs from the raw dataset
2. Explore and analyze the extracted conversations
3. Discover support intents from the data
4. Classify customer messages into relevant intents
5. Retrieve appropriate responses

## 📊 Dataset Processing

The raw Kaggle dataset is approximately 500 MB, so it is processed in chunks instead of loading the complete CSV into memory.

The extraction step identifies customer tweets that are explicitly linked to an `AppleSupport` response through the dataset's `response_tweet_id` field.

### Run the extraction

```bash
python src/data/extract_apple.py --input twcs.csv --output data/processed/apple_support_pairs.csv
