"""Tests for the Gemini-based baseline helpers without network calls."""

import unittest
from unittest.mock import Mock, patch

from baseline import chatbot_app_native as app


class BaselineGeminiTests(unittest.TestCase):
    """Validate baseline Gemini wrappers and simple chunking behavior."""

    def test_chunk_text_sample(self) -> None:
        """chunk_text should return a single chunk for short text."""
        self.assertEqual(app.chunk_text("hello world"), ["hello world"])

    @patch("baseline.chatbot_app_native.requests.post")
    def test_embed_texts_uses_gemini_response(self, mock_post: Mock) -> None:
        """embed_texts should parse Gemini embedding values."""
        mock_response = Mock()
        mock_response.json.return_value = {"embedding": {"values": [0.1, 0.2]}}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        self.assertEqual(app.embed_texts(["hello"], api_key="test-key"), [[0.1, 0.2]])

    @patch("baseline.chatbot_app_native.requests.post")
    def test_generate_answer_uses_gemini_response(self, mock_post: Mock) -> None:
        """generate_answer should parse Gemini candidate text."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "Xin chào"}]}}],
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        self.assertEqual(app.generate_answer("prompt", api_key="test-key"), "Xin chào")


if __name__ == "__main__":
    unittest.main()
