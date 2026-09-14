import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from embeddings import embed_query
from qdrant import client, COLLECTION_NAME
from qdrant_client import models


st.set_page_config(page_title="Boosted Search", layout="centered")


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
                        mult=[popularity_weight, "popularity"]
                    ),
                    models.MultExpression(
                        mult=[recency_weight, "recency"]
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
            limit=10,
        )

    top5 = points[:5]

    if not top5:
        st.warning("No results found.")

    else:
        rows = []

        for p in top5:
            payload = p.payload or {}

            name = (
                payload.get("prod_name")
                or payload.get("name")
                or str(p.id)
            )

            popularity = payload.get("popularity", 0.0)
            recency = payload.get("recency", 0.0)
            image_url = payload.get("image_url")

            vector_score = (
                p.score
                - popularity_weight * popularity
                - recency_weight * recency
            )

            rows.append(
                {
                    "item": name,
                    "image_url": image_url,
                    "final_score": p.score,
                    "vector_score": vector_score,
                    "popularity": popularity,
                    "recency": recency,
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
                    f"Popularity: {row['popularity']:.3f}  ·  "
                    f"Recency: {row['recency']:.3f}"
                )

            st.divider()

        display_df = df[
            [
                "item",
                "final_score",
                "vector_score",
                "popularity",
                "recency",
            ]
        ]

        st.dataframe(
            display_df.style.format(
                {
                    "final_score": "{:.3f}",
                    "vector_score": "{:.3f}",
                    "popularity": "{:.3f}",
                    "recency": "{:.3f}",
                }
            ),
            hide_index=True,
            use_container_width=True,
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
            df.set_index("item")["recency"] * recency_weight
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