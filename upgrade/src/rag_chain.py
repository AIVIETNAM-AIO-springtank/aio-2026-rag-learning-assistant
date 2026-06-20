"""RAG chain skeleton for the upgraded learning assistant."""


def answer_question(question: str) -> str:
    """Answer a question using the upgraded RAG pipeline."""
    if not question.strip():
        return "Vui lòng nhập câu hỏi."
    raise NotImplementedError("RAG generation is implemented in a later task.")

