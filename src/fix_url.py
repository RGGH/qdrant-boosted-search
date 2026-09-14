from pathlib import Path

from qdrant_client import QdrantClient, models


QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "hm_products"

DATA_DIR = Path("/home/moo/Documents/python/hnm/data")
IMAGES_DIR = DATA_DIR / "images"

IMAGE_BASE_URL = "http://localhost:8000/images"

BATCH_SIZE = 1_000


client = QdrantClient(QDRANT_URL)


def get_image_url(article_id: int) -> str | None:
    """
    Convert an H&M article ID such as:

        108775015

    into:

        http://localhost:8000/images/010/0108775015.jpg
    """

    # H&M image filenames are 10 digits.
    image_id = str(article_id).zfill(10)

    image_path = (
        IMAGES_DIR
        / image_id[:3]
        / f"{image_id}.jpg"
    )

    if not image_path.is_file():
        return None

    return f"{IMAGE_BASE_URL}/{image_id[:3]}/{image_id}.jpg"


def update_image_urls():
    offset = None

    total_seen = 0
    total_updated = 0
    total_missing = 0

    while True:
        products, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=BATCH_SIZE,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        if not products:
            break

        operations = []

        for product in products:
            total_seen += 1

            payload = product.payload or {}

            article_id = payload.get("article_id")

            if article_id is None:
                print(f"⚠️ No article_id for Qdrant point {product.id}")
                total_missing += 1
                continue

            url = get_image_url(article_id)

            if url is None:
                total_missing += 1
                continue

            operations.append(
                models.SetPayloadOperation(
                    set_payload=models.SetPayload(
                        payload={
                            "image_url": url,
                        },
                        points=[product.id],
                    )
                )
            )

        if operations:
            client.batch_update_points(
                collection_name=COLLECTION_NAME,
                update_operations=operations,
            )

            total_updated += len(operations)

        print(
            f"Seen: {total_seen:,} | "
            f"Updated: {total_updated:,} | "
            f"Missing: {total_missing:,}"
        )

        offset = next_offset

        if offset is None:
            break

    print()
    print("🎉 Finished")
    print(f"Seen:    {total_seen:,}")
    print(f"Updated: {total_updated:,}")
    print(f"Missing: {total_missing:,}")


if __name__ == "__main__":
    update_image_urls()