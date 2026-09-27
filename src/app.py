import math
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from embeddings import embed_query
from qdrant import client, COLLECTION_NAME
from qdrant_client import models


st.set_page_config(page_title="Boosted Search", layout="centered")


# --- Decay function config -------------------------------------------------
#
# The "recency" payload field is a precomputed value in [0, 1], where 1.0
# means "most recent". We decay it toward 0 as it moves away from that
# target, using one of Qdrant's three decay shapes.

RECENCY_TARGET = 1.0
RECENCY_SCALE = 0.5
RECENCY_MIDPOINT = 0.5

DECAY_LABELS = {
    "linear": "Linear",
    "exp": "Exponential",
    "gauss": "Gaussian",
}

# --- Colour filter config ---------------------------------------------------
#
# "colour" is a plain string field in the payload (e.g. "Black"). For best
# performance at scale you'd typically also call:
#   client.create_payload_index(
#       collection_name=COLLECTION_NAME,
#       field_name="colour",
#       field_schema=models.PayloadSchemaType.KEYWORD,
#   )
# Unindexed filtering also works, just slower on large collections.

COLOUR_OPTIONS = [
    "Any",
    "Black",
    "White",
    "Grey",
    "Red",
    "Orange",
    "Yellow",
    "Green",
    "Blue",
    "Turquoise",
    "Purple",
    "Pink",
    "Beige",
    "Brown",
    "Gold",
    "Silver",
    "Bronze",
]

# How many candidates to consider when working out each item's "pure
# vector similarity" rank, for the rank-movement chart. Matches the
# prefetch limit used by boosted_search so the two rankings are over the
# same pool.
VECTOR_RANK_POOL_SIZE = 100


def build_colour_filter(colour: str | None):
    """Shared colour filter builder, used both for the boosted query's
    prefetch and for the plain vector-only ranking query, so both rank
    over the exact same candidate pool."""

    if colour and colour != "Any":
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="colour",
                    match=models.MatchValue(value=colour),
                )
            ]
        )

    return None


def build_recency_decay_expression(decay_type: str, params: models.DecayParamsExpression):
    """Build the Qdrant decay Expression matching the chosen decay_type."""

    if decay_type == "linear":
        return models.LinDecayExpression(lin_decay=params)
    elif decay_type == "exp":
        return models.ExpDecayExpression(exp_decay=params)
    elif decay_type == "gauss":
        return models.GaussDecayExpression(gauss_decay=params)

    raise ValueError(f"Unknown decay_type: {decay_type}")


def parse_last_purchase_date(last_purchase_date: str):
    """Parse an ISO 8601 string (e.g. '2020-07-22T00:00:00Z') into a
    timezone-aware datetime, or None if missing."""

    if not last_purchase_date:
        return None

    return datetime.fromisoformat(last_purchase_date.replace("Z", "+00:00"))


@st.cache_data(ttl=3600)
def get_dataset_anchor_date():
    """Fetch the single most recent last_purchase_date across the *entire*
    collection (not just the current top 5), once per hour, to use as a
    fixed "today" reference for the days_since_purchase column.

    Requires a payload index on last_purchase_date for order_by to work;
    if that's missing (or the query fails for any other reason) this
    falls back to None, and the caller uses the most recent date within
    the current result set instead.
    """

    try:
        records, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1,
            order_by=models.OrderBy(
                key="last_purchase_date",
                direction=models.Direction.DESC,
            ),
            with_payload=["last_purchase_date"],
            with_vectors=False,
        )
    except Exception:
        return None

    if not records:
        return None

    raw_date = (records[0].payload or {}).get("last_purchase_date")
    if raw_date is None:
            return None
    return parse_last_purchase_date(raw_date)



def days_since_purchase(purchase_dt, anchor_dt) -> float:
    """Days between purchase_dt and anchor_dt, rounded to 1 decimal place.

    We anchor "today" to the most recent last_purchase_date in the result
    set rather than the real-world current date, since this dataset is
    several years old: this way the freshest item(s) in the results show
    up as 0 (or close to it) days old, and older items count up from
    there, instead of everything showing ~2000+ days old.
    """

    if purchase_dt is None or anchor_dt is None:
        return float("nan")

    delta = anchor_dt - purchase_dt

    return round(delta.total_seconds() / 86400, 1)


def recency_decay_value(
    recency: float,
    decay_type: str,
    target: float = RECENCY_TARGET,
    scale: float = RECENCY_SCALE,
    midpoint: float = RECENCY_MIDPOINT,
) -> float:
    """Python-side mirror of Qdrant's decay formulas, used locally to
    recover the vector-only score and to draw the score-components chart."""

    diff = abs(recency - target)

    if decay_type == "linear":
        return max(0.0, 1 - (1 - midpoint) * diff / scale)
    elif decay_type == "exp":
        return math.exp(math.log(midpoint) * diff / scale)
    elif decay_type == "gauss":
        return math.exp(math.log(midpoint) * (diff / scale) ** 2)

    raise ValueError(f"Unknown decay_type: {decay_type}")


def boosted_search(
    query_vector,
    popularity_weight: float,
    recency_weight: float,
    decay_type: str,
    colour: str | None = None,
    limit: int = 10,
):
    recency_decay_params = models.DecayParamsExpression(
        x="recency",
        target=RECENCY_TARGET,
        scale=RECENCY_SCALE,
        midpoint=RECENCY_MIDPOINT,
    )

    recency_decay_expression = build_recency_decay_expression(
        decay_type, recency_decay_params
    )

    # Filter the candidate set to the chosen colour *before* reranking, so
    # the formula only ever scores/returns matching points.
    prefetch_filter = build_colour_filter(colour)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=models.Prefetch(
            query=query_vector,
            using="dense",
            limit=100,
            filter=prefetch_filter,
        ),
        query=models.FormulaQuery(
            formula=models.SumExpression(
                sum=[
                    "$score",
                    models.MultExpression(
                        mult=[popularity_weight, "popularity"]
                    ),
                    models.MultExpression(
                        mult=[recency_weight, recency_decay_expression]
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


def get_vector_only_ranks(query_vector, colour: str | None = None, limit: int = VECTOR_RANK_POOL_SIZE):
    """Rank every candidate by vector similarity alone (no popularity/
    recency boost), over the same colour-filtered pool the boosted search
    draws its prefetch from. Returns {point_id: 1-indexed rank}, so callers
    can look up "where would this item have ranked without boosting?".
    """

    colour_filter = build_colour_filter(colour)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        using="dense",
        query_filter=colour_filter,
        limit=limit,
        with_payload=False,
    )

    return {point.id: rank for rank, point in enumerate(results.points, start=1)}


def plot_decay_score_comparison(scores_by_decay: dict, selected_decay_type: str):
    """Compare the actual top-5 final_score curve for this exact search
    (query + weights) across all three decay functions, so users can see
    how swapping decay type would have reshaped these specific results.

    scores_by_decay: {decay_key: [final_score, ...]} from 3 real queries,
    one per decay type, ranks in order (best first).
    """

    colors = {
        "linear": "#4C72B0",
        "exp": "#DD8452",
        "gauss": "#55A868",
    }

    fig, ax = plt.subplots(figsize=(7, 4))

    max_len = 1

    for key, label in DECAY_LABELS.items():
        scores = scores_by_decay.get(key, [])

        if not scores:
            continue

        max_len = max(max_len, len(scores))
        ranks = list(range(1, len(scores) + 1))
        is_selected = key == selected_decay_type

        ax.plot(
            ranks,
            scores,
            marker="o",
            markersize=6,
            label=label,
            color=colors[key],
            linewidth=2.5 if is_selected else 1.5,
            alpha=1.0 if is_selected else 0.5,
        )

    ax.set_xlabel("Result rank")
    ax.set_ylabel("final_score")
    ax.set_title("Same search, 3 decay functions")
    ax.set_xticks(range(1, max_len + 1))
    ax.legend(title="Decay type", loc="upper right")

    plt.tight_layout()

    return fig


def plot_rank_movement(rows, pool_size: int = VECTOR_RANK_POOL_SIZE):
    """Slope chart: left column is each item's rank under vector similarity
    alone, right column is its rank after popularity/recency boosting.
    A line rising from a low position (further down the left column) up to
    a high one on the right shows the boost pulling that item up; a flat
    line means the item was already near the top on vectors alone.

    Both columns are drawn at evenly spaced vertical slots (1, 2, 3, ...)
    rather than at the literal rank numbers. The boosted side is always a
    tight 1..n already; if we plotted the vector-only side at its real
    values (which can range up into the hundreds) on the *same* numeric
    axis, the boosted side would get compressed into an unreadable sliver
    near the top. Slots keep both sides equally legible; the real rank
    number is still shown as a text label next to each point.

    rows: list of dicts with at least "item" (name) and "boosted_rank"
    (1-indexed final position). "vector_rank" may be missing/None if the
    item didn't appear in the top `pool_size` vector-only candidates.
    """

    n = len(rows)

    def vector_sort_key(row):
        vr = row.get("vector_rank")
        return vr if vr is not None else pool_size + 1

    # Slot 1 = best (lowest) vector-only rank among these items, slot n =
    # worst, ties broken by original order.
    left_slot_by_index = {}
    for slot, (original_index, _row) in enumerate(
        sorted(enumerate(rows), key=lambda pair: vector_sort_key(pair[1])),
        start=1,
    ):
        left_slot_by_index[original_index] = slot

    fig, ax = plt.subplots(figsize=(6.5, max(3, 0.9 * n)))

    left_x, right_x = 0.0, 1.0

    for i, row in enumerate(rows):
        boosted_slot = row["boosted_rank"]  # already an even 1..n slot
        left_slot = left_slot_by_index[i]
        vector_rank = row.get("vector_rank")

        vector_rank_label = (
            f">{pool_size}" if vector_rank is None else str(vector_rank)
        )

        if left_slot > boosted_slot:
            color = "#55A868"  # green: boosting pulled it up
        elif left_slot < boosted_slot:
            color = "#C44E52"  # red: boosting pushed it down
        else:
            color = "#8C8C8C"  # grey: unchanged

        ax.plot(
            [left_x, right_x],
            [left_slot, boosted_slot],
            marker="o",
            markersize=6,
            color=color,
            linewidth=2.2,
            zorder=2,
        )

        ax.text(
            left_x - 0.04,
            left_slot,
            f"{row['item']}  ({vector_rank_label})",
            ha="right",
            va="center",
            fontsize=9,
        )
        ax.text(
            right_x + 0.04,
            boosted_slot,
            f"#{boosted_slot}",
            ha="left",
            va="center",
            fontsize=9,
        )

    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(n + 0.5, 0.5)  # inverted: slot 1 at the top
    ax.set_xticks([left_x, right_x])
    ax.set_xticklabels(["Vector-only rank", "Boosted rank"])
    ax.set_yticks([])
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.set_title("Where boosting moved each result")

    plt.tight_layout()

    return fig


st.title("Boosted Search")
st.caption("Semantic search, re-ranked with popularity and recency signals")

query = st.text_input(
    "Search query",
    placeholder="black summer dress for a wedding",
)

col1, col2 = st.columns(2)

with col1:
    popularity_weight = st.slider(
        "Popularity weight", 0.0, 1.0, 0.20, 0.05
    )

with col2:
    recency_weight = st.slider(
        "Recency weight", 0.0, 1.0, 0.25, 0.05
    )

decay_type = st.selectbox(
    "Recency decay function",
    options=list(DECAY_LABELS.keys()),
    format_func=lambda key: DECAY_LABELS[key],
    index=0,
)

colour = st.selectbox(
    "Colour",
    options=COLOUR_OPTIONS,
    index=0,
)

run = st.button(
    "Search",
    type="primary",
    disabled=not query,
)


if run and query:

    with st.spinner("Searching..."):
        query_vector = embed_query(query)

        points = boosted_search(
            query_vector,
            popularity_weight,
            recency_weight,
            decay_type,
            colour,
            limit=10,
        )

    top5 = points[:5]

    if not top5:
        st.warning("No results found.")

    else:
        # First pass: pull out payload fields and parse purchase dates, so
        # we can find the most recent one to use as our "today" anchor.
        parsed = []

        for p in top5:
            payload = p.payload or {}

            parsed.append(
                {
                    "point": p,
                    "payload": payload,
                    "purchase_dt": parse_last_purchase_date(
                        payload.get("last_purchase_date")
                    ),
                }
            )

        purchase_dts = [
            r["purchase_dt"] for r in parsed if r["purchase_dt"] is not None
        ]

        # Prefer a fixed anchor from the whole collection; fall back to the
        # most recent date within this result set if that's unavailable
        # (e.g. no payload index on last_purchase_date yet).
        anchor_dt = get_dataset_anchor_date()

        if anchor_dt is None:
            anchor_dt = max(purchase_dts) if purchase_dts else None

        # Rank each of these top-5 items by vector similarity alone (no
        # popularity/recency boost), over the same candidate pool, so we
        # can show how much the boost moved them.
        vector_ranks = get_vector_only_ranks(query_vector, colour)

        rows = []

        for i, r in enumerate(parsed):
            p = r["point"]
            payload = r["payload"]

            name = (
                payload.get("prod_name")
                or payload.get("name")
                or str(p.id)
            )

            popularity = payload.get("popularity", 0.0)
            recency = payload.get("recency", 0.0)
            image_url = payload.get("image_url")
            item_colour = payload.get("colour", "-")

            recency_decay = recency_decay_value(recency, decay_type)
            days_since = days_since_purchase(r["purchase_dt"], anchor_dt)

            vector_score = (
                p.score
                - popularity_weight * popularity
                - recency_weight * recency_decay
            )

            rows.append(
                {
                    "item": name,
                    "image_url": image_url,
                    "final_score": p.score,
                    "vector_score": vector_score,
                    "popularity": popularity,
                    "recency": recency,
                    "recency_decay": recency_decay,
                    "days_since_purchase": days_since,
                    "colour": item_colour,
                    "boosted_rank": i + 1,
                    "vector_rank": vector_ranks.get(p.id),
                }
            )

        df = pd.DataFrame(rows)

        st.subheader("Top 5 results")

        for _, row in df.iterrows():
            col_image, col_info = st.columns([1, 4])

            with col_image:
                image_url = row["image_url"]

                if isinstance(image_url, str) and image_url.strip():
                    st.image(image_url, width=140)
                else:
                    st.caption("No image")

            with col_info:
                st.markdown(f"### {row['item']}")

                st.caption(
                    f"Final score: {row['final_score']:.3f}  ·  "
                    f"Vector: {row['vector_score']:.3f}"
                )

                st.caption(
                    f"Colour: {row['colour']}  ·  "
                    f"Popularity: {row['popularity']:.3f}  ·  "
                    f"Recency: {row['recency']:.3f}  ·  "
                    f"Days since purchase: {row['days_since_purchase']:.1f}"
                )

            st.divider()

        display_df = df[
            [
                "item",
                "colour",
                "final_score",
                "vector_score",
                "popularity",
                "recency",
                "days_since_purchase",
                "vector_rank",
                "boosted_rank",
            ]
        ]

        st.dataframe(
            display_df.style.format(
                {
                    "final_score": "{:.3f}",
                    "vector_score": "{:.3f}",
                    "popularity": "{:.3f}",
                    "recency": "{:.3f}",
                    "days_since_purchase": "{:.1f}",
                },
                na_rep="-",
            ),
            hide_index=True,
            width='stretch'
        )

        st.subheader("Score components")

        # Weight popularity/recency the same way the final_score formula does,
        # so the stacked bar heights actually equal final_score and stay in
        # the same descending order as the table above.
        chart_df = df.set_index("item")[["vector_score"]].copy()
        chart_df["popularity"] = (
            df.set_index("item")["popularity"] * popularity_weight
        )
        chart_df["recency"] = (
            df.set_index("item")["recency_decay"] * recency_weight
        )

        fig, ax = plt.subplots(figsize=(7, 4))

        chart_df.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            color=[
                "#4C72B0",
                "#DD8452",
                "#55A868",
            ],
        )

        ax.set_ylabel("Contribution to final score")
        ax.set_xlabel("")
        ax.legend(title="Component", loc="upper right")

        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()

        st.pyplot(fig)

        st.subheader("Where boosting moved things")
        st.caption(
            "Left column: rank if we only used vector similarity, within "
            f"the same top-{VECTOR_RANK_POOL_SIZE} candidate pool. Right "
            "column: rank after adding popularity and recency. Green rising "
            "lines are items the boost pulled up; red falling lines are "
            "items it pushed down; grey lines were already in place."
        )

        st.pyplot(plot_rank_movement(rows))

        st.subheader("How other decay functions would have ranked this search")

        other_decay_types = [k for k in DECAY_LABELS if k != decay_type]

        with st.spinner("Running the same search with the other decay functions..."):
            scores_by_decay = {decay_type: [p.score for p in top5]}

            for other_type in other_decay_types:
                other_points = boosted_search(
                    query_vector,
                    popularity_weight,
                    recency_weight,
                    other_type,
                    colour,
                    limit=10,
                )
                scores_by_decay[other_type] = [
                    p.score for p in other_points[:5]
                ]

        st.caption(
            "Same query and weights, re-run with each decay function — "
            "ranks line up by position (1st, 2nd, ...), not by item, "
            "since different decay shapes can promote different items."
        )

        st.pyplot(plot_decay_score_comparison(scores_by_decay, decay_type))