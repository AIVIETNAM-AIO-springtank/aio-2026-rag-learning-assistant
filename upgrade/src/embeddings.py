"""Embedding wrapper skeleton for the upgraded RAG pipeline."""


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Return embeddings for a list of texts.

    The real Ollama integration is implemented in a later task. Empty input is
    handled now so imports and simple smoke tests are stable.
    """
    if not texts:
        return []
    raise NotImplementedError("Ollama embedding integration is implemented in a later task.")

