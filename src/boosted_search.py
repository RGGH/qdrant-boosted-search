from embeddings import embed_query
from qdrant import client, COLLECTION_NAME
from qdrant_client import models


def semantic_search(query: str, limit: int = 10):
    query_vector = embed_query(query)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        using="dense",
        limit=limit,
        with_payload=True,
    )

    return results.points


def boosted_search(
    query: str,
    popularity_weight: float,
    recency_weight: float,
    limit: int = 10,
):
    query_vector = embed_query(query)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=models.Prefetch(
            query=query_vector,
            using="dense",
            limit=100,
        ),
        query=models.FormulaQuery(
            formula=models.SumExpression(
                sum=[
                    "$score",
                    models.MultExpression(
                        mult=[
                            popularity_weight,
                            "popularity",
                        ]
                    ),
                    models.MultExpression(
                        mult=[
                            recency_weight,
                            "recency",
                        ]
                    ),
                ]
            ),
            defaults={
                "popularity": 0.0,
                "recency": 0.0,
            },
        ),
        limit=limit,
        with_payload=True,
    )

    return results.points


def print_results(title: str, results):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

    print(
        f"{'Rank':<6}"
        f"{'Product':<32}"
        f"{'Score':<10}"
        f"{'Popularity':<14}"
        f"{'Recency':<10}"
    )

    print("-" * 100)

    for rank, result in enumerate(results, 1):
        product = result.payload

        print(
            f"{rank:<6}"
            f"{product['name'][:30]:<32}"
            f"{result.score:<10.4f}"
            f"{product.get('popularity', 0):<14.3f}"
            f"{product.get('recency', 0):<10.3f}"
        )


if __name__ == "__main__":
    query = "black summer dress for a wedding"

    print(f'\n🔎 Query: "{query}"')

    # Baseline
    baseline = semantic_search(query)

    print_results(
        "BASELINE — SEMANTIC ONLY",
        baseline,
    )

    # Experiment A
    experiment_a = boosted_search(
        query,
        popularity_weight=0.05,
        recency_weight=0.05,
    )

    print_results(
        "EXPERIMENT A — 0.05 POPULARITY + 0.05 RECENCY",
        experiment_a,
    )

    # Experiment B
    experiment_b = boosted_search(
        query,
        popularity_weight=0.10,
        recency_weight=0.05,
    )

    print_results(
        "EXPERIMENT B — 0.10 POPULARITY + 0.05 RECENCY",
        experiment_b,
    )

    # Experiment C
    experiment_c = boosted_search(
        query,
        popularity_weight=0.20,
        recency_weight=0.10,
    )

    print_results(
        "EXPERIMENT C — 0.20 POPULARITY + 0.10 RECENCY",
        experiment_c,
    )