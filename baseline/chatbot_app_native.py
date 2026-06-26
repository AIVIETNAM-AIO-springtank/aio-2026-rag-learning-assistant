"""Baseline Streamlit PDF RAG chatbot using Gemini and session ChromaDB.

This baseline stays intentionally simple for demos:
- PDF text is extracted with pypdf.
- Chunks and query are embedded through Gemini's embedding API.
- ChromaDB is in-memory per Streamlit session.
- Answers are generated with Gemini.
"""

from __future__ import annotations

import os
import tempfile
import time
from collections.abc import Iterable
from typing import Any

import chromadb
import pypdf
import requests
import streamlit as st


GEMINI_GENERATION_MODEL = os.getenv("GEMINI_GENERATION_MODEL", "gemini-1.5-flash")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004")
DEFAULT_K = int(os.getenv("DEFAULT_K", "2"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
REQUEST_TIMEOUT = int(os.getenv("GEMINI_REQUEST_TIMEOUT", "60"))

PROMPT = """Bạn là trợ lý hỏi đáp tài liệu học tập.
Chỉ dùng các đoạn ngữ cảnh dưới đây để trả lời câu hỏi.
Nếu ngữ cảnh không có thông tin, hãy nói: "Tôi chưa tìm thấy thông tin này trong tài liệu."
Không bịa, không suy đoán ngoài ngữ cảnh.
Trả lời ngắn gọn, chính xác, bằng tiếng Việt.

Ngữ cảnh:
{context}

Câu hỏi:
{question}

Trả lời:
"""


def get_gemini_api_key() -> str:
    """Return Gemini API key from Streamlit secrets or environment variables."""
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        key = ""
    return key or os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")


def gemini_endpoint(model: str, action: str, api_key: str) -> str:
    """Build a Gemini REST endpoint URL."""
    return f"https://generativelanguage.googleapis.com/v1beta/models/{model}:{action}?key={api_key}"


def embed_texts(texts: list[str], api_key: str | None = None) -> list[list[float]]:
    """Embed texts with Gemini and return vector lists."""
    if not texts:
        return []

    key = api_key or get_gemini_api_key()
    if not key:
        raise RuntimeError("Missing GEMINI_API_KEY. Add it to Streamlit secrets or environment variables.")

    embeddings: list[list[float]] = []
    url = gemini_endpoint(GEMINI_EMBEDDING_MODEL, "embedContent", key)
    for text in texts:
        payload = {
            "model": f"models/{GEMINI_EMBEDDING_MODEL}",
            "content": {"parts": [{"text": text}]},
        }
        response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        embeddings.append(data["embedding"]["values"])
    return embeddings


def generate_answer(prompt: str, api_key: str | None = None) -> str:
    """Generate an answer with Gemini from a completed prompt."""
    key = api_key or get_gemini_api_key()
    if not key:
        raise RuntimeError("Missing GEMINI_API_KEY. Add it to Streamlit secrets or environment variables.")

    url = gemini_endpoint(GEMINI_GENERATION_MODEL, "generateContent", key)
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 512},
    }
    response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    candidates = data.get("candidates", [])
    if not candidates:
        return "Tôi chưa nhận được câu trả lời từ Gemini."
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(part.get("text", "") for part in parts).strip()


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into simple overlapping chunks."""
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


def extract_pdf_text(uploaded_file: Any) -> str:
    """Extract text from an uploaded PDF file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        path = tmp.name

    try:
        reader = pypdf.PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    finally:
        os.unlink(path)


def process_pdf(uploaded_file: Any) -> tuple[Any, int]:
    """Read PDF, chunk text, embed chunks and store them in session ChromaDB."""
    text = extract_pdf_text(uploaded_file)
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("PDF không có text để index. Hãy thử PDF text-based khác.")

    client = chromadb.Client()
    collection = client.get_or_create_collection(f"rag_{int(time.time())}")
    collection.add(
        ids=[str(index) for index in range(len(chunks))],
        documents=chunks,
        embeddings=embed_texts(chunks),
    )
    return collection, len(chunks)


def rag(question: str, collection: Any, k: int = DEFAULT_K) -> str:
    """Retrieve context from ChromaDB and generate an answer with Gemini."""
    result = collection.query(query_embeddings=embed_texts([question]), n_results=k)
    context = "\n\n".join(result["documents"][0])
    prompt = PROMPT.format(context=context, question=question)
    return generate_answer(prompt)


def stream_words(text: str) -> Iterable[str]:
    """Yield answer words so Streamlit can display a lightweight stream."""
    for word in text.split(" "):
        yield word + " "


def init_session_state() -> None:
    """Initialize Streamlit session state for the baseline app."""
    defaults = {"collection": None, "pdf_name": "", "chat_history": []}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def main() -> None:
    """Run the Streamlit baseline app."""
    st.set_page_config(page_title="PDF RAG Chatbot - Gemini", layout="wide", initial_sidebar_state="expanded")
    init_session_state()

    st.title("PDF RAG Assistant: Baseline Gemini")

    with st.sidebar:
        st.subheader("Cấu hình")
        st.caption(f"Generation model: `{GEMINI_GENERATION_MODEL}`")
        st.caption(f"Embedding model: `{GEMINI_EMBEDDING_MODEL}`")
        if not get_gemini_api_key():
            st.warning("Chưa cấu hình GEMINI_API_KEY trong Streamlit secrets hoặc environment.")

        st.subheader("Upload tài liệu")
        uploaded_file = st.file_uploader("Chọn file PDF", type="pdf")
        if uploaded_file and st.button("Xử lý PDF", use_container_width=True):
            with st.spinner("Đang xử lý PDF và tạo embedding..."):
                try:
                    st.session_state.collection, chunk_count = process_pdf(uploaded_file)
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.chat_history = []
                    st.success(f"Đã index {chunk_count} chunks")
                    st.info(st.session_state.pdf_name)
                except Exception as exc:
                    st.error(f"Không xử lý được PDF: {exc}")

        if st.button("Xóa lịch sử chat", use_container_width=True):
            st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if st.session_state.collection is None:
        st.info("Upload và xử lý PDF trước khi chat.")
        st.chat_input("Nhập câu hỏi...", disabled=True)
        return

    question = st.chat_input("Nhập câu hỏi của bạn...")
    if not question:
        return

    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Đang suy nghĩ..."):
            try:
                answer = rag(question, st.session_state.collection)
                st.write_stream(stream_words(answer))
            except Exception as exc:
                answer = f"Lỗi khi gọi Gemini: {exc}"
                st.error(answer)

    st.session_state.chat_history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
