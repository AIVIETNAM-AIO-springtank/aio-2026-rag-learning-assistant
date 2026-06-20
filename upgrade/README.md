# Upgrade - AIO 2026 Learning Assistant

This folder contains the improved implementation built from the baseline PDF RAG flow.

## Target Improvements

- Modular RAG pipeline.
- PDF metadata and source citation.
- Persistent ChromaDB storage.
- Configurable Ollama models.
- Short chat memory.
- Future Notion CSV/API ingestion.

## Current State

This is a skeleton created for project structure separation. Feature implementation will be handled by later Task IDs.

## Import Check

```bash
python -c "from upgrade.src import chunk_text, embed_texts; print(chunk_text('hello world')); print(embed_texts([]))"
```

