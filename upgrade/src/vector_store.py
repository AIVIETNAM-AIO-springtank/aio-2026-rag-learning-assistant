"""Vector store skeleton for the upgraded RAG pipeline."""


class VectorStore:
    """Placeholder vector store interface for future ChromaDB implementation."""

    def add_chunks(self, chunks: list[str]) -> None:
        """Add chunks to the vector store."""
        raise NotImplementedError("Persistent ChromaDB storage is implemented in a later task.")

