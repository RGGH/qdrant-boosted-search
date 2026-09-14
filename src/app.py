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
    query: str,
    popularity_weight: float,
    recency_weight: float,
    decay_type: str,
    colour: str | None = None,
    limit: int = 10,
):
    query_vector = embed_query(query)

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
    prefetch_filter = None

    if colour and colour != "Any":
        prefetch_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="colour",
                    match=models.MatchValue(value=colour),
                )
            ]
        )

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
        points = boosted_search(
            query,
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

        rows = []

        for r in parsed:
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

        st.subheader("How other decay functions would have ranked this search")

        other_decay_types = [k for k in DECAY_LABELS if k != decay_type]

        with st.spinner("Running the same search with the other decay functions..."):
            scores_by_decay = {decay_type: [p.score for p in top5]}

            for other_type in other_decay_types:
                other_points = boosted_search(
                    query,
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
