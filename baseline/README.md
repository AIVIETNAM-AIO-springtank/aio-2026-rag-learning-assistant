# Baseline - PDF RAG Chatbot With Gemini

This folder contains the simple baseline PDF RAG Chatbot implementation for Streamlit Cloud demos.

## Purpose

- Upload a PDF.
- Extract text with `pypdf`.
- Store local hash embeddings in session-only ChromaDB.
- Use Gemini only for answer generation.
- Keep the baseline simple and easy to demo.

## Streamlit Cloud Secret

Do not hard-code API keys in the repository. Add this secret in Streamlit Cloud:

```toml
GEMINI_API_KEY = "your_key_here"
```

## Run

```bash
streamlit run baseline/chatbot_app_native.py
```

Optional environment variables:

```bash
GEMINI_GENERATION_MODEL=gemini-2.5-flash
LOCAL_EMBEDDING_DIM=384
DEFAULT_K=2
```

## Rules

- Treat this folder as reference-only unless a Task ID explicitly targets baseline.
- Do not add upgrade features here.
- Technical improvements should be implemented in `upgrade/`.
