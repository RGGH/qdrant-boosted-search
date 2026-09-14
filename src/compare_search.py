from datetime import datetime, timezone

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

    # Current time becomes the "ideal" freshness target.
    current_time = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    results = client.query_points(
        collection_name=COLLECTION_NAME,

        # Step 1:
        # retrieve semantic candidates
        prefetch=models.Prefetch(
            query=query_vector,
            using="dense",
            limit=100,
        ),

        # Step 2:
        # rerank those candidates using Qdrant Formula Query
        query=models.FormulaQuery(
            formula=models.SumExpression(
                sum=[
                    # Original semantic similarity
                    "$score",

                    # Popularity boost
                    models.MultExpression(
                        mult=[
                            popularity_weight,
                            "popularity",
                        ]
                    ),

                    # Native Qdrant exponential decay
                    models.MultExpression(
                        mult=[
                            recency_weight,
                            models.ExpDecayExpression(
                                exp_decay=models.DecayParamsExpression(
                                    x=models.DatetimeKeyExpression(
                                        datetime_key="last_purchase_date"
                                    ),
                                    target=models.DatetimeExpression(
                                        datetime=current_time
                                    ),
                                    scale=86400 * 30,
                                    midpoint=0.5,
                                )
                            ),
                        ]
                    ),
                ]
            ),

            defaults={
                "popularity": 0.0,
            },
        ),

        limit=limit,
        with_payload=True,
    )

    return results.points


def print_results(title: str, results):
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)

    print(
        f"{'Rank':<6}"
        f"{'Product':<32}"
        f"{'Score':<10}"
        f"{'Popularity':<14}"
        f"{'Last Purchase':<22}"
    )

    print("-" * 110)

    for rank, result in enumerate(results, 1):
        product = result.payload

        print(
            f"{rank:<6}"
            f"{product['name'][:30]:<32}"
            f"{result.score:<10.4f}"
            f"{product.get('popularity', 0):<14.3f}"
            f"{str(product.get('last_purchase_date', 'N/A')):<22}"
        )


if __name__ == "__main__":
    query = "black summer dress for a wedding"

    print(f'\n🔎 Query: "{query}"')

    # ---------------------------------------------------------
    # BASELINE
    # ---------------------------------------------------------

    baseline = semantic_search(query)

    print_results(
        "BASELINE — SEMANTIC ONLY",
        baseline,
    )

    # ---------------------------------------------------------
    # EXPERIMENT A
    # ---------------------------------------------------------

    experiment_a = boosted_search(
        query,
        popularity_weight=0.05,
        recency_weight=0.05,
    )

    print_results(
        "EXPERIMENT A — 0.05 POPULARITY + 0.05 EXP DECAY",
        experiment_a,
    )

    # ---------------------------------------------------------
    # EXPERIMENT B
    # ---------------------------------------------------------

    experiment_b = boosted_search(
        query,
        popularity_weight=0.10,
        recency_weight=0.05,
    )

    print_results(
        "EXPERIMENT B — 0.10 POPULARITY + 0.05 EXP DECAY",
        experiment_b,
    )

    # ---------------------------------------------------------
    # EXPERIMENT C
    # ---------------------------------------------------------

    experiment_c = boosted_search(
        query,
        popularity_weight=0.20,
        recency_weight=0.10,
    )

    print_results(
        "EXPERIMENT C — 0.20 POPULARITY + 0.10 EXP DECAY",
        experiment_c,
    )