"""Smoke tests for Sprint 1 project structure and upgrade imports."""

import importlib
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class SprintOneSmokeTests(unittest.TestCase):
    """Validate the baseline/upgrade split and importable upgrade skeleton."""

    def test_required_folders_exist(self) -> None:
        """Check the required top-level project folders."""
        for folder in ["baseline", "upgrade", "docs", "tests"]:
            self.assertTrue((ROOT / folder).exists(), folder)

    def test_baseline_app_exists(self) -> None:
        """Check that the baseline Streamlit app is present."""
        self.assertTrue((ROOT / "baseline" / "chatbot_app_native.py").exists())

    def test_upgrade_modules_import(self) -> None:
        """Check that core upgrade skeleton modules are importable."""
        for module_name in [
            "upgrade.src.config",
            "upgrade.src.chunker",
            "upgrade.src.embeddings",
            "upgrade.src.vector_store",
            "upgrade.src.retriever",
            "upgrade.src.rag_chain",
            "upgrade.src.prompts",
        ]:
            importlib.import_module(module_name)

    def test_chunker_sample_text(self) -> None:
        """Check that chunk_text runs with sample text."""
        from upgrade.src.chunker import chunk_text

        self.assertEqual(chunk_text("hello world"), ["hello world"])

    def test_embed_empty_input(self) -> None:
        """Check that embed_texts handles empty input without Ollama."""
        from upgrade.src.embeddings import embed_texts

        self.assertEqual(embed_texts([]), [])


if __name__ == "__main__":
    unittest.main()

