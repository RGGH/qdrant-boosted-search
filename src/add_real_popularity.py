# add_real_popularity.py

import math

import pandas as pd
from qdrant_client import QdrantClient, models


QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "hm_products"

TRANSACTIONS_FILE = "data/transactions_train.csv"

TRANSACTION_CHUNK_SIZE = 500_000
QDRANT_BATCH_SIZE = 1_000

client = QdrantClient(QDRANT_URL)


def calculate_purchase_counts():
    """Count how many times each article was purchased."""

    print("📖 Reading transaction data...")

    purchase_counts = {}

    for chunk in pd.read_csv(
        TRANSACTIONS_FILE,
        usecols=["article_id"],
        dtype={"article_id": "int64"},
        chunksize=TRANSACTION_CHUNK_SIZE,
    ):
        counts = chunk["article_id"].value_counts()

        for article_id, count in counts.items():
            purchase_counts[article_id] = (
                purchase_counts.get(article_id, 0) + int(count)
            )

        print(f"   processed {len(chunk):,} transactions")

    print(
        f"\n📦 Found {len(purchase_counts):,} products "
        "with at least one purchase"
    )

    return purchase_counts


def create_popularity_scores(purchase_counts):
    """Convert purchase counts into normalized 0-1 popularity scores."""

    # Log scaling prevents extremely popular products
    # from completely dominating the ranking.
    scores = {
        article_id: math.log1p(count)
        for article_id, count in purchase_counts.items()
    }

    max_score = max(scores.values())

    return {
        article_id: score / max_score
        for article_id, score in scores.items()
    }


def update_qdrant(popularity_scores):
    """Update the popularity payload for every Qdrant (H&M) product."""

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

            popularity = popularity_scores.get(article_id, 0.0)

            if article_id in popularity_scores:
                matched += 1
            else:
                no_purchases += 1

            operations.append(
                models.SetPayloadOperation(
                    set_payload=models.SetPayload(
                        payload={
                            "popularity": float(popularity),
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
            f"| purchases={matched:,} "
            f"| no purchases={no_purchases:,}"
        )

        offset = next_offset

        if offset is None:
            break

    print("\n🎉 Finished!")
    print(f"Products updated:       {total:,}")
    print(f"Products with purchases: {matched:,}")
    print(f"Products with no purchases: {no_purchases:,}")


def main():
    purchase_counts = calculate_purchase_counts()

    popularity_scores = create_popularity_scores(purchase_counts)

    update_qdrant(popularity_scores)


if __name__ == "__main__":
    main()