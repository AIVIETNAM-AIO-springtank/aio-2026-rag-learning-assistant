"""Retriever skeleton for the upgraded RAG pipeline."""


def retrieve(question: str, k: int = 4) -> list[dict]:
    """Retrieve relevant chunks for a question."""
    if not question.strip():
        return []
    raise NotImplementedError("Retriever implementation is handled in a later task.")

