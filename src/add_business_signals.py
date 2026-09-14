import random

from qdrant_client import QdrantClient, models

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "hm_products"

BATCH_SIZE = 1_000

client = QdrantClient(QDRANT_URL)


def add_popularity_scores():
    """Add a synthetic popularity score to every product."""

    # Makes the demo reproducible.
    random.seed(42)

    offset = None
    total_updated = 0

    while True:
        # Fetch one page of products.
        products, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=BATCH_SIZE,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )

        if not products:
            break

        operations = []

        for product in products:
            popularity = random.random()

            operations.append(
                models.SetPayloadOperation(
                    set_payload=models.SetPayload(
                        payload={
                            "popularity": popularity,
                        },
                        points=[product.id],
                    )
                )
            )

        # Send all updates in this batch in one request.
        client.batch_update_points(
            collection_name=COLLECTION_NAME,
            update_operations=operations,
        )

        total_updated += len(products)

        print(f"✅ Updated {total_updated:,} products")

        # Qdrant gives us the offset for the next page.
        offset = next_offset

        if offset is None:
            break

    print(f"\n🎉 Finished. Updated {total_updated:,} products.")


if __name__ == "__main__":
    add_popularity_scores()