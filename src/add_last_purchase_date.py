# src/add_last_purchase_date.py

import pandas as pd
from qdrant_client import QdrantClient, models

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "hm_products"

TRANSACTIONS_FILE = "data/transactions_train.csv"

TRANSACTION_CHUNK_SIZE = 500_000
QDRANT_BATCH_SIZE = 1_000

client = QdrantClient(QDRANT_URL)


def calculate_last_purchase_dates():
    print("📖 Reading transaction data...")

    last_purchase = {}

    for chunk in pd.read_csv(
        TRANSACTIONS_FILE,
        usecols=["t_dat", "article_id"],
        parse_dates=["t_dat"],
        dtype={"article_id": "int64"},
        chunksize=TRANSACTION_CHUNK_SIZE,
    ):
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
        f"\n📦 Found last purchase dates for "
        f"{len(last_purchase):,} products"
    )

    return last_purchase


def update_qdrant(last_purchase):
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

            if article_id in last_purchase:
                purchase_date = last_purchase[article_id]

                # Qdrant datetime payload format
                date_string = purchase_date.strftime(
                    "%Y-%m-%dT00:00:00Z"
                )

                matched += 1

                payload = {
                    "last_purchase_date": date_string
                }

            else:
                # Products with no purchases
                no_purchases += 1

                payload = {
                    "last_purchase_date": "1970-01-01T00:00:00Z"
                }

            operations.append(
                models.SetPayloadOperation(
                    set_payload=models.SetPayload(
                        payload=payload,
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
    print(f"Products updated:       {total:,}")
    print(f"With purchase history:  {matched:,}")
    print(f"Without purchases:     {no_purchases:,}")


def main():
    last_purchase = calculate_last_purchase_dates()
    update_qdrant(last_purchase)


if __name__ == "__main__":
    main()
