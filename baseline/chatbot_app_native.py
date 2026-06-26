"""Baseline Streamlit PDF RAG chatbot using Gemini and session ChromaDB.

This baseline is optimized for Streamlit Cloud demos:
- PDF text is extracted with pypdf.
- Chunks and queries use a local hashing embedding, so upload/index does not
  call Gemini embedding APIs.
- ChromaDB is in-memory per Streamlit session.
- Answers are generated with Gemini.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import tempfile
import time
from collections.abc import Iterable
from typing import Any

import chromadb
import pypdf
import requests
import streamlit as st


GEMINI_GENERATION_MODEL = os.getenv("GEMINI_GENERATION_MODEL", "gemini-1.5-flash")
LOCAL_EMBEDDING_MODEL = "local-hash-embedding"
LOCAL_EMBEDDING_DIM = int(os.getenv("LOCAL_EMBEDDING_DIM", "384"))
DEFAULT_K = int(os.getenv("DEFAULT_K", "2"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
REQUEST_TIMEOUT = int(os.getenv("GEMINI_REQUEST_TIMEOUT", "60"))
TOKEN_PATTERN = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)

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


def token_index(token: str, dimension: int = LOCAL_EMBEDDING_DIM) -> int:
    """Map a token to a stable vector index."""
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(digest, 16) % dimension


def embed_one(text: str, dimension: int = LOCAL_EMBEDDING_DIM) -> list[float]:
    """Embed text locally with a normalized hashing bag-of-words vector."""
    vector = [0.0] * dimension
    for token in TOKEN_PATTERN.findall(text.lower()):
        vector[token_index(token, dimension)] += 1.0

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts locally without calling an external embedding API."""
    return [embed_one(text) for text in texts]


def safe_post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    """POST JSON and raise sanitized errors that do not expose API keys."""
    response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
    if response.status_code >= 400:
        detail = response.text[:300].replace("\n", " ")
        raise RuntimeError(f"Gemini API error {response.status_code}: {detail}")
    return response.json()


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
    data = safe_post_json(url, payload)
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
        st.caption(f"Embedding: `{LOCAL_EMBEDDING_MODEL}` ({LOCAL_EMBEDDING_DIM} dim)")
        if not get_gemini_api_key():
            st.warning("Chưa cấu hình GEMINI_API_KEY trong Streamlit secrets hoặc environment.")

        st.subheader("Upload tài liệu")
        uploaded_file = st.file_uploader("Chọn file PDF", type="pdf")
        if uploaded_file and st.button("Xử lý PDF", use_container_width=True):
            with st.spinner("Đang xử lý PDF và tạo embedding local..."):
                try:
                    st.session_state.collection, chunk_count = process_pdf(uploaded_file)
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.chat_history = []
                    st.success(f"Đã index {chunk_count} chunks")
                    st.info(st.session_state.pdf_name)
                except Exception:
                    st.error("Không xử lý được PDF. Hãy kiểm tra file PDF có text và thử lại.")

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
