from datetime import datetime
import math

import matplotlib.pyplot as plt

from embeddings import embed_query
from qdrant import client, COLLECTION_NAME
from qdrant_client import models


# ============================================================
# CONFIG
# ============================================================

QUERY = "black summer dress for a wedding"

POPULARITY_WEIGHT = 0.20

# Increased for the demo so viewers can clearly see the effect.
RECENCY_WEIGHT = 0.25

# Qdrant ExpDecay parameters
DECAY_DAYS = 30
MIDPOINT = 0.5

# Last date in the H&M transaction dataset
DATASET_END_DATE = "2020-09-22T00:00:00Z"

OUTPUT_FILE = "decay_effect.png"


# ============================================================
# SEARCH
# ============================================================

def boosted_search(query: str, limit: int = 10):

    query_vector = embed_query(query)

    results = client.query_points(
        collection_name=COLLECTION_NAME,

        # First retrieve semantic candidates
        prefetch=models.Prefetch(
            query=query_vector,
            using="dense",
            limit=100,
        ),

        # Then rerank using Qdrant Formula Query
        query=models.FormulaQuery(
            formula=models.SumExpression(
                sum=[
                    # Semantic relevance
                    "$score",

                    # Popularity boost
                    models.MultExpression(
                        mult=[
                            POPULARITY_WEIGHT,
                            "popularity",
                        ]
                    ),

                    # Qdrant exponential decay
                    models.MultExpression(
                        mult=[
                            RECENCY_WEIGHT,
                            models.ExpDecayExpression(
                                exp_decay=models.DecayParamsExpression(
                                    x=models.DatetimeKeyExpression(
                                        datetime_key="last_purchase_date"
                                    ),
                                    target=models.DatetimeExpression(
                                        datetime=DATASET_END_DATE
                                    ),
                                    scale=DECAY_DAYS * 86400,
                                    midpoint=MIDPOINT,
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


# ============================================================
# REPRODUCE THE DECAY FOR THE VISUALIZATION
# ============================================================

def calculate_decay(age_days: float) -> float:

    return math.exp(
        -math.log(1 / MIDPOINT)
        * age_days
        / DECAY_DAYS
    )


def get_decay_details(payload):

    date_string = payload.get("last_purchase_date")

    if not date_string:
        return 0.0, 0.0

    purchase_date = datetime.fromisoformat(
        date_string.replace("Z", "+00:00")
    )

    target_date = datetime.fromisoformat(
        DATASET_END_DATE.replace("Z", "+00:00")
    )

    age_days = (
        target_date - purchase_date
    ).total_seconds() / 86400

    decay = calculate_decay(age_days)

    return age_days, decay


# ============================================================
# CREATE VISUALIZATION
# ============================================================

def visualize(results):

    products = []

    for rank, result in enumerate(results, 1):

        payload = result.payload

        popularity = float(
            payload.get("popularity", 0.0)
        )

        age_days, decay = get_decay_details(payload)

        popularity_boost = (
            POPULARITY_WEIGHT * popularity
        )

        recency_boost = (
            RECENCY_WEIGHT * decay
        )

        # Qdrant's final score is:
        #
        # semantic
        # + popularity boost
        # + recency boost
        #
        semantic_score = (
            result.score
            - popularity_boost
            - recency_boost
        )

        products.append(
            {
                "rank": rank,
                "name": payload.get(
                    "name",
                    "Unknown product",
                )[:30],
                "semantic": semantic_score,
                "popularity": popularity_boost,
                "recency": recency_boost,
                "decay": decay,
                "age": age_days,
                "final": result.score,
            }
        )

    # Reverse so rank 1 is displayed at the top
    products.reverse()

    names = [
        product["name"]
        for product in products
    ]

    semantic = [
        product["semantic"]
        for product in products
    ]

    popularity = [
        product["popularity"]
        for product in products
    ]

    recency = [
        product["recency"]
        for product in products
    ]

    final_scores = [
        product["final"]
        for product in products
    ]

    # ========================================================
    # MATPLOTLIB
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(13, 8)
    )

    # Semantic score
    ax.barh(
        names,
        semantic,
        label="Semantic relevance",
    )

    # Popularity boost
    ax.barh(
        names,
        popularity,
        left=semantic,
        label="Popularity boost",
    )

    # Recency boost
    popularity_left = [
        s + p
        for s, p in zip(
            semantic,
            popularity,
        )
    ]

    ax.barh(
        names,
        recency,
        left=popularity_left,
        label="Exponential recency boost",
    )

    # ========================================================
    # FINAL SCORE LABELS
    # ========================================================

    for i, score in enumerate(final_scores):

        ax.text(
            score + 0.01,
            i,
            f"{score:.3f}",
            va="center",
            fontsize=10,
        )

    # ========================================================
    # TITLES
    # ========================================================

    ax.set_title(
        "How Exponential Decay Affects Search Ranking",
        fontsize=17,
        pad=20,
    )

    ax.text(
        0,
        1.01,
        f'Query: "{QUERY}"',
        transform=ax.transAxes,
        fontsize=11,
    )

    ax.set_xlabel(
        "Final Qdrant ranking score"
    )

    ax.set_ylabel(
        "Product"
    )

    ax.legend()

    ax.grid(
        axis="x",
        alpha=0.25,
    )

    # ========================================================
    # CLEANUP
    # ========================================================

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_FILE,
        dpi=180,
        bbox_inches="tight",
    )

    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        f'\n🔎 Query: "{QUERY}"'
    )

    print(
        "\n🚀 Running Qdrant boosted search..."
    )

    results = boosted_search(
        QUERY,
        limit=10,
    )

    print(
        "\n📊 Results:"
    )

    for rank, result in enumerate(
        results,
        1,
    ):

        payload = result.payload

        age_days, decay = get_decay_details(
            payload
        )

        recency_boost = (
            RECENCY_WEIGHT * decay
        )

        print(
            f"{rank:2}. "
            f"{payload['name'][:35]:<35} "
            f"score={result.score:.4f} "
            f"age={age_days:.0f}d "
            f"decay={decay:.3f} "
            f"recency_boost={recency_boost:.3f}"
        )

    print(
        "\n🎨 Creating visualization..."
    )

    visualize(results)

    print(
        f"\n✅ Saved to {OUTPUT_FILE}"
    )
