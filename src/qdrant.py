from qdrant_client import QdrantClient

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "hm_products"

client = QdrantClient(QDRANT_URL)


def search_products(
    query_vector: list[float],
    limit: int = 10,
):
    """Retrieve the most semantically relevant products."""

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        using="dense",
        limit=limit,
        with_payload=True,
    )

    return results.points