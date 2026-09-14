from embeddings import embed_query
from qdrant import client, COLLECTION_NAME
from qdrant_client import models

import matplotlib.pyplot as plt
import numpy as np


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


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def plot_score_by_rank(experiments: dict, top_n: int = 10, save_path: str | None = None):
    """
    Line plot comparing score-by-rank across experiments.

    `experiments` maps a label (e.g. "Baseline") to a list of result points.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for label, results in experiments.items():
        results = results[:top_n]
        ranks = np.arange(1, len(results) + 1)
        scores = [r.score for r in results]
        ax.plot(ranks, scores, marker="o", label=label)

    ax.set_xlabel("Rank")
    ax.set_ylabel("Score")
    ax.set_title("Score by Rank Across Experiments")
    ax.set_xticks(np.arange(1, top_n + 1))
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_top_product_scores(experiments: dict, top_n: int = 5, save_path: str | None = None):
    """
    Grouped bar chart comparing scores of the top-N products from each
    experiment (bars grouped by rank position, one color per experiment).
    """
    labels = list(experiments.keys())
    n_groups = top_n
    n_bars = len(labels)

    fig, ax = plt.subplots(figsize=(12, 6))

    bar_width = 0.8 / n_bars
    x = np.arange(n_groups)

    for i, label in enumerate(labels):
        results = experiments[label][:top_n]
        scores = [r.score for r in results]
        # pad in case an experiment returned fewer than top_n results
        scores += [0] * (top_n - len(scores))
        offset = (i - (n_bars - 1) / 2) * bar_width
        ax.bar(x + offset, scores, width=bar_width, label=label)

    ax.set_xlabel("Rank position")
    ax.set_ylabel("Score")
    ax.set_title(f"Top {top_n} Scores by Experiment")
    ax.set_xticks(x)
    ax.set_xticklabels([f"#{i + 1}" for i in range(n_groups)])
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_popularity_vs_recency(experiments: dict, top_n: int = 10, save_path: str | None = None):
    """
    Scatter plot of popularity vs recency for the top-N results of each
    experiment, so you can see how the boosting shifts which products surface.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    for label, results in experiments.items():
        results = results[:top_n]
        popularity = [r.payload.get("popularity", 0) for r in results]
        recency = [r.payload.get("recency", 0) for r in results]
        ax.scatter(popularity, recency, label=label, s=80, alpha=0.7)

    ax.set_xlabel("Popularity")
    ax.set_ylabel("Recency")
    ax.set_title("Popularity vs Recency of Top Results")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_score_decomposition(
    results,
    popularity_weight: float,
    recency_weight: float,
    title: str,
    top_n: int = 10,
    save_path: str | None = None,
):
    """
    Stacked bar chart that decomposes each boosted score back into its
    three ingredients: semantic score, popularity contribution, and
    recency contribution. This is what actually shows *how much* the
    boost is doing, rather than just that the total went up.
    """
    results = results[:top_n]

    names = [r.payload["name"][:20] for r in results]
    semantic = []
    pop_contrib = []
    rec_contrib = []

    for r in results:
        popularity = r.payload.get("popularity", 0)
        recency = r.payload.get("recency", 0)
        p_contrib = popularity_weight * popularity
        r_contrib = recency_weight * recency
        base = r.score - p_contrib - r_contrib

        semantic.append(base)
        pop_contrib.append(p_contrib)
        rec_contrib.append(r_contrib)

    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(x, semantic, label="Semantic score", color="#4C72B0")
    ax.bar(x, pop_contrib, bottom=semantic, label="Popularity boost", color="#DD8452")
    bottom2 = np.array(semantic) + np.array(pop_contrib)
    ax.bar(x, rec_contrib, bottom=bottom2, label="Recency boost", color="#55A868")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(f"Score Decomposition — {title}")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_rank_movement(
    baseline,
    experiment,
    experiment_label: str,
    top_n: int = 10,
    save_path: str | None = None,
):
    """
    Bump chart connecting each product's baseline rank to its rank in a
    boosted experiment. Makes risers, fallers, and new entrants obvious
    at a glance, instead of burying the effect in raw scores.
    """
    baseline_top = baseline[:top_n]
    experiment_top = experiment[:top_n]

    baseline_rank = {r.id: (i + 1, r.payload["name"]) for i, r in enumerate(baseline_top)}
    experiment_rank = {r.id: (i + 1, r.payload["name"]) for i, r in enumerate(experiment_top)}

    all_ids = set(baseline_rank) | set(experiment_rank)

    fig, ax = plt.subplots(figsize=(8, max(6, len(all_ids) * 0.6)))

    max_rank = top_n + 1  # off-chart slot for products absent from a side

    for pid in all_ids:
        b_rank, b_name = baseline_rank.get(pid, (max_rank, None))
        e_rank, e_name = experiment_rank.get(pid, (max_rank, None))
        name = b_name or e_name

        if b_rank < e_rank:
            color = "#C44E52"  # fell
        elif b_rank > e_rank:
            color = "#55A868"  # rose
        else:
            color = "#8C8C8C"  # unchanged

        ax.plot([0, 1], [b_rank, e_rank], marker="o", color=color, linewidth=2)
        ax.text(-0.05, b_rank, f"{name} (#{b_rank})", ha="right", va="center", fontsize=9)
        ax.text(1.05, e_rank, f"{name} (#{e_rank})", ha="left", va="center", fontsize=9)

    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(max_rank + 0.5, 0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Baseline", experiment_label])
    ax.set_yticks([])
    ax.set_title(f"Rank Movement: Baseline → {experiment_label}")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


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

    # ------------------------------------------------------------------
    # Visualize all experiments together
    # ------------------------------------------------------------------
    all_experiments = {
        "Baseline": baseline,
        "Exp A (0.05 / 0.05)": experiment_a,
        "Exp B (0.10 / 0.05)": experiment_b,
        "Exp C (0.20 / 0.10)": experiment_c,
    }

    plot_score_by_rank(all_experiments, top_n=10, save_path="score_by_rank.png")
    plot_top_product_scores(all_experiments, top_n=5, save_path="top_product_scores.png")
    plot_popularity_vs_recency(all_experiments, top_n=10, save_path="popularity_vs_recency.png")

    # These are the ones that actually show what the boost is *doing*:
    plot_score_decomposition(
        experiment_c,
        popularity_weight=0.20,
        recency_weight=0.10,
        title="Exp C (0.20 / 0.10)",
        top_n=10,
        save_path="score_decomposition_exp_c.png",
    )

    plot_rank_movement(
        baseline,
        experiment_c,
        experiment_label="Exp C (0.20 / 0.10)",
        top_n=10,
        save_path="rank_movement_exp_c.png",
    )

    plt.show()