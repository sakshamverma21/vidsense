import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.embeddings import Embeddings

from vidsense import config, corpus


class TinyEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[float(len(text)), 1.0, 0.0] for text in texts]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


class SavedCorpusTests(unittest.TestCase):
    def test_saved_transcript_build_and_restore_use_real_faiss_without_provider_or_disk(self):
        transcript = [{"text": "The video explains gradient descent and learning.", "start": 12.0, "duration": 4.0}]
        stored = []
        with tempfile.TemporaryDirectory() as directory, patch.object(config, "CACHE_DIR", Path(directory)), patch.object(
            config, "get_embeddings", return_value=TinyEmbeddings()
        ), patch.object(corpus, "_fetch_transcript") as provider:
            collection = corpus.build_corpus(
                "Private", [{"video_id": "abc12345678", "title": "Learning"}],
                transcript_loader=lambda _video: transcript, video_saved=stored.append,
                persist_cache=False, cache_namespace="private-one", verbose=False,
            )
            self.assertEqual(list(Path(directory).iterdir()), [])
            self.assertEqual(collection.search("gradient", k=1)[0].metadata["timestamp"], 12.0)
            restored = corpus.restore_corpus("Private", [{
                "video_id": stored[0].video_id, "title": stored[0].title,
                "transcript": stored[0].transcript,
            }])
            self.assertEqual(restored.search("gradient", k=1)[0].metadata["video_id"], "abc12345678")
        provider.assert_not_called()

    def test_oversized_transcript_is_rejected_before_persistence(self):
        with patch.object(config, "MAX_TRANSCRIPT_CHARS", 5), patch.object(
            config, "get_embeddings", return_value=TinyEmbeddings()
        ), patch.object(corpus, "_fetch_transcript", return_value=[{"text": "Much too large", "start": 0, "duration": 1}]):
            with self.assertRaisesRegex(RuntimeError, "No usable transcripts"):
                corpus.build_corpus("Private", [{"video_id": "abc12345678"}], persist_cache=False, verbose=False)
