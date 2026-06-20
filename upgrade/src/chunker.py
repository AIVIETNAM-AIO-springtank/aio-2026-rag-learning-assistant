"""Text chunking utilities for the upgraded RAG pipeline."""


def chunk_text(text: str, size: int = 1000, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks.

    This mirrors the baseline behavior at skeleton stage. Later tasks will add
    metadata-aware chunking.
    """
    if not text:
        return []

    paragraphs = [paragraph.strip() for paragraph in text.split("\n") if paragraph.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 1 <= size:
            current += paragraph + "\n"
            continue

        if current:
            chunks.append(current.strip())
        current = (current[-overlap:] + paragraph + "\n") if overlap else (paragraph + "\n")

    if current.strip():
        chunks.append(current.strip())

    return chunks

