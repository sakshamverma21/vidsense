import io
import json
import unittest
from unittest.mock import patch

from vidsense import config, supadata


class SupadataClientTests(unittest.TestCase):
    def setUp(self):
        self.previous_request_time = supadata._last_request_at

    def tearDown(self):
        supadata._last_request_at = self.previous_request_time

    def test_requests_share_the_configured_minimum_interval(self):
        response = io.BytesIO(json.dumps({"content": []}).encode("utf-8"))
        supadata._last_request_at = 10.0

        with patch.object(config, "get_supadata_api_key", return_value="test-key"), patch.object(
            config, "SUPADATA_MIN_INTERVAL_SECONDS", 1.1
        ), patch.object(
            supadata, "monotonic", side_effect=[10.2, 11.5]
        ), patch.object(
            supadata, "sleep"
        ) as sleep_mock, patch.object(
            supadata.urllib.request, "urlopen", return_value=response
        ):
            payload = supadata.get_json("/v1/transcript", {"url": "video-url"})

        self.assertEqual(payload, {"content": []})
        sleep_mock.assert_called_once()
        self.assertAlmostEqual(sleep_mock.call_args.args[0], 0.9)


if __name__ == "__main__":
    unittest.main()
