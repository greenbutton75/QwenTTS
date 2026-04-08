import importlib
import os
import sys
import unittest
from unittest.mock import MagicMock, patch


def _seed_env() -> None:
    defaults = {
        "TASK_BASE_URL": "https://rixtrema.net/api/async_task_manager",
        "USER_TOKEN": "user-token",
        "SYSTEM_TOKEN": "system-token",
        "FINGERPRINT": "fingerprint",
        "AWS_ACCESS_KEY_ID": "test-access-key",
        "AWS_SECRET_ACCESS_KEY": "test-secret-key",
        "AWS_REGION": "us-east-1",
        "S3_BUCKET_NAME": "test-bucket",
        "QWEN_TTS_BASE_URL": "http://127.0.0.1:8000",
    }
    for key, value in defaults.items():
        os.environ[key] = value


class TaskWorkerClientTests(unittest.TestCase):
    def _reload_modules(self):
        sys.modules.pop("task_worker.config", None)
        sys.modules.pop("task_worker.qwen_client", None)
        config = importlib.import_module("task_worker.config")
        qwen_client = importlib.import_module("task_worker.qwen_client")
        return config, qwen_client

    def test_phrase_poll_interval_defaults_to_five_seconds(self) -> None:
        _seed_env()
        os.environ.pop("PHRASE_POLL_INTERVAL", None)

        config, _ = self._reload_modules()

        self.assertEqual(config.PHRASE_POLL_INTERVAL, 5)

    def test_qwen_client_uses_configurable_request_timeouts(self) -> None:
        _seed_env()
        os.environ["QWEN_TTS_HEALTH_TIMEOUT_SECONDS"] = "7"
        os.environ["QWEN_TTS_PROFILE_REQUEST_TIMEOUT_SECONDS"] = "181"
        os.environ["QWEN_TTS_PHRASE_REQUEST_TIMEOUT_SECONDS"] = "123"
        os.environ["QWEN_TTS_SPLICE_REQUEST_TIMEOUT_SECONDS"] = "145"
        os.environ["QWEN_TTS_STATUS_TIMEOUT_SECONDS"] = "17"

        _, qwen_client = self._reload_modules()

        response = MagicMock()
        response.json.return_value = {"status": "ok"}
        with patch.object(qwen_client.requests, "get", return_value=response) as get_mock, \
             patch.object(qwen_client.requests, "post", return_value=response) as post_mock:
            qwen_client.health_check()
            qwen_client.create_profile("19338", "voice-1", "Voice", None, False)
            qwen_client.get_profile_status("19338", "voice-1")
            qwen_client.create_phrase("19338", "voice-1", "phrase-1", "Hello there")
            qwen_client.create_phrase_splice("19338", "voice-1", "phrase-1", "Hi Kevin,", "this is Eugene")
            qwen_client.get_phrase_status("19338", "phrase-1")

        self.assertEqual(get_mock.call_args_list[0].kwargs["timeout"], 7)
        self.assertEqual(post_mock.call_args_list[0].kwargs["timeout"], 181)
        self.assertEqual(get_mock.call_args_list[1].kwargs["timeout"], 17)
        self.assertEqual(post_mock.call_args_list[1].kwargs["timeout"], 123)
        self.assertEqual(post_mock.call_args_list[2].kwargs["timeout"], 145)
        self.assertEqual(get_mock.call_args_list[2].kwargs["timeout"], 17)


if __name__ == "__main__":
    unittest.main()
