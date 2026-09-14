from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"

model = SentenceTransformer(MODEL_NAME)


def embed_query(query: str) -> list[float]:
    """Convert a text query into the same 384-dim space as our products."""
    vector = model.encode(
        query,
        normalize_embeddings=True,
    )

    return vector.tolist()