"""Prompt templates for the upgraded RAG assistant."""

RAG_PROMPT = """Bạn là trợ lý học tập AIO 2026.
Chỉ trả lời dựa trên ngữ cảnh được cung cấp.
Nếu không tìm thấy thông tin trong ngữ cảnh, hãy nói rằng chưa tìm thấy thông tin.

Ngữ cảnh:
{context}

Câu hỏi:
{question}

Trả lời:
"""

