from embeddings import embed_query
from qdrant import search_products


def search(query: str, limit: int = 10):
    """Run semantic product search."""

    query_vector = embed_query(query)

    return search_products(
        query_vector=query_vector,
        limit=limit,
    )


if __name__ == "__main__":
    query = "black summer dress for a wedding"

    results = search(query)

    print(f'\n🔎 "{query}"\n')

    for rank, result in enumerate(results, 1):
        product = result.payload

        print(
            f"{rank:2}. "
            f"{product['name']} | "
            f"{product['product_type']} | "
            f"{product['colour']} | "
            f"score={result.score:.4f}"
        )