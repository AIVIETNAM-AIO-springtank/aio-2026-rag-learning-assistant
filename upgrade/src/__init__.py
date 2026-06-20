"""Core modules for the upgraded AIO 2026 Learning Assistant."""

from .chunker import chunk_text
from .embeddings import embed_texts

__all__ = ["chunk_text", "embed_texts"]

