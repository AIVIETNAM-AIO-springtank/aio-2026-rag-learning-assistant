"""Configuration defaults for the upgraded RAG assistant."""

import os

APP_TITLE = "AIO 2026 Learning Assistant"
LLM_MODEL = os.getenv("LLM_MODEL", "vicuna:7b-v1.5-q5_1")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")
DEFAULT_K = int(os.getenv("DEFAULT_K", "4"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
NOTION_TOKEN = os.getenv("NOTION_TOKEN", "")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID", "")
