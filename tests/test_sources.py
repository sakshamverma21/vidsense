import unittest
from unittest.mock import patch

from vidsense import config, sources
from vidsense.sources import resolve_sources, split_inputs
from vidsense.utils import extract_playlist_id, extract_video_id


class SourceResolverTests(unittest.TestCase):
    def test_youtube_id_helpers_support_watch_and_playlist_context_urls(self):
        self.assertEqual(extract_video_id("https://www.youtube.com/watch?v=abc12345678&list=PL1"), "abc12345678")
        self.assertEqual(extract_playlist_id("https://www.youtube.com/playlist?list=PL1"), "PL1")
        self.assertEqual(extract_playlist_id("https://www.youtube.com/watch?v=abc12345678&list=PL1"), "PL1")

    def test_split_inputs_handles_multiline_and_comma_paste(self):
        self.assertEqual(split_inputs([" one, two\nthree "]), ["one", "two", "three"])

    def test_resolver_deduplicates_and_reports_invalid_source(self):
        def metadata(url, video_id):
            return {"title": f"Title {video_id}", "url": url}

        resolved = resolve_sources(
            ["https://youtu.be/abc12345678", "https://www.youtube.com/watch?v=abc12345678", "not-a-link"],
            metadata_lookup=metadata,
        )
        self.assertEqual([spec["video_id"] for spec in resolved.specs], ["abc12345678"])
        self.assertEqual(resolved.sources[1]["status"], "duplicate")
        self.assertEqual(resolved.sources[2]["status"], "invalid")

    def test_playlist_limit_is_deterministic_and_warns(self):
        calls = []

        def playlist(url, maximum):
            calls.append(maximum)
            return "Playlist", [
                {"id": "first000001", "title": "First"},
                {"id": "second00002", "title": "Second"},
            ], True

        resolved = resolve_sources(
            ["https://youtu.be/direct00001", "https://www.youtube.com/playlist?list=PL1"],
            max_videos=2,
            metadata_lookup=lambda url, vid: {"title": "Direct", "url": url},
            playlist_lookup=playlist,
        )
        self.assertEqual(calls, [1])
        self.assertEqual([spec["video_id"] for spec in resolved.specs], ["direct00001", "first000001"])
        self.assertEqual(resolved.sources[1]["status"], "limited")
        self.assertTrue(resolved.warnings)

    def test_inaccessible_playlist_is_reported_without_hiding_valid_video(self):
        resolved = resolve_sources(
            ["https://youtu.be/abc12345678", "https://www.youtube.com/playlist?list=PL1"],
            metadata_lookup=lambda url, vid: {"title": "Working video", "url": url},
            playlist_lookup=lambda url, maximum: (_ for _ in ()).throw(RuntimeError("private playlist")),
        )
        self.assertEqual([spec["video_id"] for spec in resolved.specs], ["abc12345678"])
        self.assertEqual(resolved.sources[1]["status"], "error")

    def test_managed_playlist_resolver_preserves_ids_titles_and_limit(self):
        def managed_json(path, params):
            if path.endswith("/playlist"):
                return {"title": "Managed playlist"}
            self.assertEqual(params["limit"], 3)
            return {"videoIds": ["first000001", "second00002", "third000003"]}

        with patch.object(sources, "_supadata_json", side_effect=managed_json), patch.object(
            sources,
            "_oembed_metadata",
            side_effect=lambda url, video_id: {"title": f"Title {video_id}", "url": url},
        ):
            title, entries, clipped = sources._supadata_playlist_entries("playlist-url", 2)

        self.assertEqual(title, "Managed playlist")
        self.assertEqual([entry["id"] for entry in entries], ["first000001", "second00002"])
        self.assertEqual(entries[0]["title"], "Title first000001")
        self.assertTrue(clipped)

    def test_auto_mode_routes_source_resolution_through_managed_playlist(self):
        with patch.object(config, "TRANSCRIPT_PROVIDER", "auto"), patch.object(
            config, "has_supadata", return_value=True
        ), patch.object(
            sources,
            "_supadata_playlist_entries",
            return_value=("Cloud playlist", [{"id": "first000001", "title": "First"}], False),
        ) as managed:
            resolved = resolve_sources(["https://www.youtube.com/playlist?list=PL1"])

        managed.assert_called_once()
        self.assertEqual(resolved.specs[0]["video_id"], "first000001")


if __name__ == "__main__":
    unittest.main()
