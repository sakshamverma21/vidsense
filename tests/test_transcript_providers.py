import unittest
from unittest.mock import patch

from vidsense import config, corpus


class TranscriptProviderTests(unittest.TestCase):
    def test_supadata_segments_are_converted_from_milliseconds(self):
        payload = {
            "lang": "en",
            "content": [
                {"text": " First segment ", "offset": 1250, "duration": 750},
                {"text": "Second segment", "offset": 2000, "duration": None},
                {"text": "  ", "offset": 3000, "duration": 1000},
            ],
        }
        with patch.object(corpus, "get_supadata_json", return_value=payload):
            segments = corpus._fetch_supadata_transcript("abc12345678")

        self.assertEqual(
            segments,
            [
                {"text": "First segment", "start": 1.25, "duration": 0.75},
                {"text": "Second segment", "start": 2.0, "duration": 0.0},
            ],
        )

    def test_auto_mode_prefers_managed_provider_when_configured(self):
        expected = [{"text": "Managed", "start": 0.0, "duration": 1.0}]
        with patch.object(config, "TRANSCRIPT_PROVIDER", "auto"), patch.object(
            config, "has_supadata", return_value=True
        ), patch.object(corpus, "_fetch_supadata_transcript", return_value=expected) as managed, patch.object(
            corpus, "_fetch_direct_transcript"
        ) as direct:
            self.assertEqual(corpus._fetch_transcript("abc12345678"), expected)

        managed.assert_called_once_with("abc12345678")
        direct.assert_not_called()

    def test_auto_mode_keeps_zero_cost_direct_path_without_key(self):
        expected = [{"text": "Direct", "start": 0.0, "duration": 1.0}]
        with patch.object(config, "TRANSCRIPT_PROVIDER", "auto"), patch.object(
            config, "has_supadata", return_value=False
        ), patch.object(corpus, "_fetch_direct_transcript", return_value=expected) as direct:
            self.assertEqual(corpus._fetch_transcript("abc12345678"), expected)

        direct.assert_called_once_with("abc12345678")

    def test_unknown_provider_is_rejected(self):
        with patch.object(config, "TRANSCRIPT_PROVIDER", "unknown"):
            with self.assertRaisesRegex(RuntimeError, "auto, direct, or supadata"):
                corpus._fetch_transcript("abc12345678")

    def test_managed_unavailable_transcript_becomes_caption_status(self):
        with patch.object(
            corpus,
            "get_supadata_json",
            return_value={"error": "transcript-unavailable"},
        ):
            with self.assertRaisesRegex(RuntimeError, "Captions are unavailable"):
                corpus._fetch_supadata_transcript("abc12345678")


if __name__ == "__main__":
    unittest.main()
