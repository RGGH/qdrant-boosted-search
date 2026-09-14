import math

import pandas as pd
from qdrant_client import QdrantClient, models


QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "hm_products"

TRANSACTIONS_FILE = "data/transactions_train.csv"

TRANSACTION_CHUNK_SIZE = 500_000
QDRANT_BATCH_SIZE = 1_000

client = QdrantClient(QDRANT_URL)


def calculate_last_purchase_dates():
    """Find the most recent purchase date for every article."""

    print("📖 Reading transaction data...")

    last_purchase = {}

    for chunk in pd.read_csv(
        TRANSACTIONS_FILE,
        usecols=["t_dat", "article_id"],
        parse_dates=["t_dat"],
        dtype={"article_id": "int64"},
        chunksize=TRANSACTION_CHUNK_SIZE,
    ):
        # Find the latest purchase date for each article
        latest = chunk.groupby("article_id")["t_dat"].max()

        for article_id, date in latest.items():
            article_id = int(article_id)

            if (
                article_id not in last_purchase
                or date > last_purchase[article_id]
            ):
                last_purchase[article_id] = date

        print(f"   processed {len(chunk):,} transactions")

    print(
        f"\n📦 Found recent purchase dates for "
        f"{len(last_purchase):,} products"
    )

    return last_purchase


def create_recency_scores(last_purchase):
    """Convert purchase dates into normalized 0-1 recency scores."""

    latest_date = max(last_purchase.values())

    # Find the oldest purchase in the dataset.
    earliest_date = min(last_purchase.values())

    total_days = (latest_date - earliest_date).days

    print(f"\n📅 Earliest purchase: {earliest_date.date()}")
    print(f"📅 Latest purchase:   {latest_date.date()}")

    scores = {}

    for article_id, purchase_date in last_purchase.items():
        age_days = (latest_date - purchase_date).days

        if total_days == 0:
            recency = 1.0
        else:
            # Newest = 1.0
            # Oldest = 0.0
            recency = 1.0 - (age_days / total_days)

        scores[article_id] = recency

    return scores


def update_qdrant(recency_scores):
    """Write recency scores into existing Qdrant products."""

    print("\n🚀 Updating Qdrant...")

    offset = None
    total = 0
    matched = 0
    no_purchases = 0

    while True:
        products, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=QDRANT_BATCH_SIZE,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )

        if not products:
            break

        operations = []

        for product in products:
            article_id = int(product.id)

            recency = recency_scores.get(article_id, 0.0)

            if article_id in recency_scores:
                matched += 1
            else:
                no_purchases += 1

            operations.append(
                models.SetPayloadOperation(
                    set_payload=models.SetPayload(
                        payload={
                            "recency": float(recency),
                        },
                        points=[product.id],
                    )
                )
            )

        client.batch_update_points(
            collection_name=COLLECTION_NAME,
            update_operations=operations,
        )

        total += len(products)

        print(
            f"✅ Updated {total:,} products "
            f"| matched={matched:,} "
            f"| no purchases={no_purchases:,}"
        )

        offset = next_offset

        if offset is None:
            break

    print("\n🎉 Finished!")
    print(f"Products updated:          {total:,}")
    print(f"Products with transactions: {matched:,}")
    print(f"Products with no purchases: {no_purchases:,}")


def main():
    last_purchase = calculate_last_purchase_dates()

    recency_scores = create_recency_scores(last_purchase)

    print("\n📊 Recency statistics:")
    print(f"Min:    {min(recency_scores.values()):.4f}")
    print(f"Median: {pd.Series(recency_scores).median():.4f}")
    print(f"Max:    {max(recency_scores.values()):.4f}")

    update_qdrant(recency_scores)


if __name__ == "__main__":
    main()
