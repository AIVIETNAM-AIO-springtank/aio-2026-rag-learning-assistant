"""Embedding wrapper for the upgraded RAG pipeline."""

from .config import EMBED_MODEL


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Return Ollama embeddings for a list of texts."""
    if not texts:
        return []

    import ollama

    response = ollama.embed(model=EMBED_MODEL, input=texts)
    return response["embeddings"]
