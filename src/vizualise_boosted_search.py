from datetime import datetime, timezone
import math

import matplotlib.pyplot as plt
import pandas as pd

from embeddings import embed_query
from qdrant import client, COLLECTION_NAME
from qdrant_client import models


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

QUERY = "black summer dress for a wedding"

POPULARITY_WEIGHT = 0.20

# Increased deliberately for the demo so the effect is obvious.
RECENCY_WEIGHT = 0.25

# Qdrant exponential decay parameters
DECAY_DAYS = 30
MIDPOINT = 0.5

# H&M transaction dataset ends on this date.
# If your transaction file reports a different latest date,
# change this value.
DATASET_END_DATE = "2020-09-22T00:00:00Z"

OUTPUT_CURVE = "exp_decay_curve.png"
OUTPUT_RANKING = "boosted_ranking_effect.png"


# ---------------------------------------------------------
# QDRANT SEARCH
# ---------------------------------------------------------

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

        # Then let Qdrant rerank them
        query=models.FormulaQuery(
            formula=models.SumExpression(
                sum=[
                    "$score",

                    # Popularity
                    models.MultExpression(
                        mult=[
                            POPULARITY_WEIGHT,
                            "popularity",
                        ]
                    ),

                    # Qdrant native exponential decay
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


# ---------------------------------------------------------
# REPRODUCE QDRANT'S EXP DECAY FOR VISUALIZATION
# ---------------------------------------------------------

def calculate_decay(age_days):
    """
    Same exponential decay we're asking Qdrant to calculate.

    With midpoint=0.5 and scale=30 days:

        0 days  -> 1.00
        30 days -> 0.50
        60 days -> 0.25
        90 days -> 0.125
    """

    return math.exp(
        -math.log(1 / MIDPOINT)
        * age_days
        / DECAY_DAYS
    )


# ---------------------------------------------------------
# PLOT 1 — EXPONENTIAL DECAY CURVE
# ---------------------------------------------------------

def plot_decay_curve():

    ages = list(range(0, 181))
    decay_scores = [
        calculate_decay(age)
        for age in ages
    ]

    plt.figure(figsize=(10, 6))

    plt.plot(
        ages,
        decay_scores,
        linewidth=3,
    )

    plt.axhline(
        MIDPOINT,
        linestyle="--",
        linewidth=1,
    )

    plt.axvline(
        DECAY_DAYS,
        linestyle="--",
        linewidth=1,
    )

    plt.scatter(
        [0, 30, 60, 90],
        [
            calculate_decay(0),
            calculate_decay(30),
            calculate_decay(60),
            calculate_decay(90),
        ],
        s=60,
    )

    plt.text(
        30,
        0.53,
        "30 days → 0.50",
        ha="center",
    )

    plt.text(
        60,
        0.28,
        "60 days → 0.25",
        ha="center",
    )

    plt.text(
        90,
        0.15,
        "90 days → 0.125",
        ha="center",
    )

    plt.title(
        "Qdrant Exponential Decay — Product Freshness",
        fontsize=16,
    )

    plt.xlabel("Days since last purchase")
    plt.ylabel("Recency score")

    plt.xlim(0, 180)
    plt.ylim(0, 1.05)

    plt.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(OUTPUT_CURVE, dpi=150)
    plt.close()

    print(f"📈 Saved {OUTPUT_CURVE}")


# ---------------------------------------------------------
# CALCULATE CONTRIBUTIONS FOR SEARCH RESULTS
# ---------------------------------------------------------

def get_decay_for_product(product):
    date_string = product.get("last_purchase_date")

    if not date_string:
        return 0.0

    purchase_date = datetime.fromisoformat(
        date_string.replace("Z", "+00:00")
    )

    target_date = datetime.fromisoformat(
        DATASET_END_DATE.replace("Z", "+00:00")
    )

    age_days = (
        target_date - purchase_date
    ).total_seconds() / 86400

    return calculate_decay(age_days)


def prepare_ranking_data(results):

    rows = []

    for rank, result in enumerate(results, 1):

        payload = result.payload

        popularity = float(
            payload.get("popularity", 0.0)
        )

        decay = get_decay_for_product(payload)

        popularity_boost = (
            POPULARITY_WEIGHT * popularity
        )

        recency_boost = (
            RECENCY_WEIGHT * decay
        )

        semantic_score = (
            result.score
            - popularity_boost
            - recency_boost
        )

        rows.append(
            {
                "rank": rank,
                "name": payload.get(
                    "name",
                    "Unknown product",
                )[:28],
                "semantic": semantic_score,
                "popularity": popularity_boost,
                "recency": recency_boost,
                "final": result.score,
                "decay": decay,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# PLOT 2 — EFFECT ON REAL SEARCH RESULTS
# ---------------------------------------------------------

def plot_ranking_effect(results):

    df = prepare_ranking_data(results)

    # Reverse so rank 1 appears at the top
    df = df.iloc[::-1]

    plt.figure(figsize=(12, 8))

    plt.barh(
        df["name"],
        df["semantic"],
        label="Semantic relevance",
    )

    plt.barh(
        df["name"],
        df["popularity"],
        left=df["semantic"],
        label="Popularity boost",
    )

    plt.barh(
        df["name"],
        df["recency"],
        left=df["semantic"] + df["popularity"],
        label="Exponential recency boost",
    )

    for _, row in df.iterrows():

        plt.text(
            row["final"] + 0.005,
            row["name"],
            f"{row['final']:.3f}",
            va="center",
        )

    plt.title(
        f'How Qdrant Exponential Decay Changes Ranking\n'
        f'"{QUERY}"',
        fontsize=16,
    )

    plt.xlabel("Final ranking score")
    plt.ylabel("Product")

    plt.legend()

    plt.grid(
        axis="x",
        alpha=0.25,
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_RANKING, dpi=150)
    plt.close()

    print(f"📊 Saved {OUTPUT_RANKING}")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":

    print(f'\n🔎 Query: "{QUERY}"')

    print("\n1️⃣  Generating exponential decay curve...")
    plot_decay_curve()

    print("\n2️⃣  Running boosted Qdrant search...")

    results = boosted_search(
        QUERY,
        limit=10,
    )

    print("\nResults:")

    for rank, result in enumerate(results, 1):

        product = result.payload

        decay = get_decay_for_product(product)

        print(
            f"{rank:2}. "
            f"{product['name'][:35]:<35} "
            f"final={result.score:.4f} "
            f"decay={decay:.3f} "
            f"recency_boost={RECENCY_WEIGHT * decay:.4f}"
        )

    print("\n3️⃣  Generating ranking visualization...")
    plot_ranking_effect(results)

    print("\n🎉 Done!")
