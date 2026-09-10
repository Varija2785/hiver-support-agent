import argparse
import pandas as pd

COLS = [
    'tweet_id', 'author_id', 'inbound', 'created_at', 'text',
    'response_tweet_id', 'in_response_to_tweet_id'
]


def extract(input_csv: str, output_csv: str, brand: str = 'AppleSupport'):
    targets = set()
    brand_rows = []

    for chunk in pd.read_csv(input_csv, usecols=COLS, chunksize=500_000):
        rows = chunk[chunk['author_id'].eq(brand)]
        brand_rows.append(rows)
        for value in rows['response_tweet_id'].dropna().astype(str):
            for tid in value.split(','):
                try:
                    targets.add(int(float(tid)))
                except ValueError:
                    continue

    brand_rows = pd.concat(brand_rows, ignore_index=True)
    customer_rows = []
    for chunk in pd.read_csv(input_csv, usecols=COLS, chunksize=500_000):
        rows = chunk[chunk['tweet_id'].isin(targets) & chunk['inbound'].eq(True)]
        if not rows.empty:
            customer_rows.append(rows)

    customers = pd.concat(customer_rows, ignore_index=True).drop_duplicates('tweet_id')
    customer_map = customers.set_index('tweet_id')
    records = []

    for _, brand_row in brand_rows.iterrows():
        if pd.isna(brand_row['response_tweet_id']):
            continue
        for raw_id in str(brand_row['response_tweet_id']).split(','):
            try:
                customer_id = int(float(raw_id))
            except ValueError:
                continue
            if customer_id not in customer_map.index:
                continue
            customer = customer_map.loc[customer_id]
            if isinstance(customer, pd.DataFrame):
                customer = customer.iloc[0]
            records.append({
                'customer_tweet_id': customer_id,
                'brand_tweet_id': int(brand_row['tweet_id']),
                'customer_author_id': str(customer['author_id']),
                'customer_created_at': customer['created_at'],
                'customer_text': customer['text'],
                'brand_created_at': brand_row['created_at'],
                'brand_text': brand_row['text'],
            })

    pd.DataFrame(records).drop_duplicates(
        ['customer_tweet_id', 'brand_tweet_id']
    ).to_csv(output_csv, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    extract(args.input, args.output)
