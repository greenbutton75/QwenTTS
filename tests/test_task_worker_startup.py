import importlib
import os
import sys
import types
import unittest
from unittest.mock import patch


def _seed_env() -> None:
    defaults = {
        "USER_TOKEN": "user-token",
        "SYSTEM_TOKEN": "system-token",
        "FINGERPRINT": "fingerprint",
        "AWS_ACCESS_KEY_ID": "test-access-key",
        "AWS_SECRET_ACCESS_KEY": "test-secret-key",
        "AWS_REGION": "us-east-1",
        "S3_BUCKET_NAME": "test-bucket",
        "PHRASE_SPLICE_PAUSE_MS": "220",
        "PHRASE_SPLICE_CROSSFADE_MS": "10",
        "QWEN_TTS_READY_TIMEOUT_SECONDS": "5",
        "QWEN_TTS_READY_POLL_INTERVAL": "1",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def _install_test_stubs() -> None:
    fake_health = types.ModuleType("task_worker.health")

    class FakeHealthState:
        def __init__(self) -> None:
            self.last_error = ""

        def set_error(self, msg: str) -> None:
            self.last_error = msg

        def inc_phrase_success(self) -> None:
            return None

        def inc_phrase_failed(self) -> None:
            return None

        def inc_profile_success(self) -> None:
            return None

        def inc_profile_failed(self) -> None:
            return None

        def inc_phrase_grouped(self, count: int = 1) -> None:
            return None

        def inc_phrase_splice_path(self, count: int = 1) -> None:
            return None

        def inc_phrase_fallback_full(self, count: int = 1) -> None:
            return None

        def inc_splice_failure(self, count: int = 1) -> None:
            return None

        def mark_profile_poll(self, seen: int) -> None:
            return None

        def mark_phrase_poll(self, seen: int) -> None:
            return None

    fake_health.HealthState = FakeHealthState
    sys.modules["task_worker.health"] = fake_health

    fake_qwen_client = types.ModuleType("task_worker.qwen_client")
    fake_qwen_client.create_phrase = lambda *args, **kwargs: {}
    fake_qwen_client.create_phrase_splice = lambda *args, **kwargs: {}
    fake_qwen_client.create_profile = lambda *args, **kwargs: {}
    fake_qwen_client.get_phrase_status = lambda *args, **kwargs: {"status": "done"}
    fake_qwen_client.get_profile_status = lambda *args, **kwargs: {"status": "done"}
    fake_qwen_client.health_check = lambda: {"status": "ok"}
    sys.modules["task_worker.qwen_client"] = fake_qwen_client

    fake_s3_utils = types.ModuleType("task_worker.s3_utils")
    fake_s3_utils.object_exists = lambda key: True
    fake_s3_utils.read_json = lambda key: {"status": "done"}
    sys.modules["task_worker.s3_utils"] = fake_s3_utils

    fake_task_api = types.ModuleType("task_worker.task_api")
    fake_task_api.complete_task = lambda *args, **kwargs: None
    fake_task_api.failed_task = lambda *args, **kwargs: None
    fake_task_api.list_tasks = lambda *args, **kwargs: []
    fake_task_api.task_id_from_record = lambda rec: "1"
    fake_task_api.task_params_from_record = lambda rec: {}
    fake_task_api.update_progress = lambda *args, **kwargs: None
    sys.modules["task_worker.task_api"] = fake_task_api


_seed_env()
_install_test_stubs()

sys.modules.pop("task_worker.config", None)
sys.modules.pop("task_worker.worker", None)

task_worker_module = importlib.import_module("task_worker.worker")


class TaskWorkerStartupTests(unittest.TestCase):
    def test_wait_for_qwen_tts_ready_retries_until_health_is_ok(self) -> None:
        state = task_worker_module.HealthState()
        with patch.object(
            task_worker_module,
            "health_check",
            side_effect=[RuntimeError("connection refused"), {"status": "ok"}],
        ) as health_mock, patch.object(task_worker_module.time, "sleep") as sleep_mock:
            task_worker_module._wait_for_qwen_tts_ready(state)

        self.assertEqual(health_mock.call_count, 2)
        self.assertEqual(sleep_mock.call_count, 1)
        self.assertIn("connection refused", state.last_error)

    def test_submit_and_wait_splice_phrase_uses_configured_pause(self) -> None:
        with patch.object(task_worker_module, "create_phrase_splice") as splice_mock, \
             patch.object(task_worker_module, "_wait_phrase", return_value={"status": "done", "public_url": "ok"}):
            ok, err = task_worker_module._submit_and_wait_splice_phrase(
                task_id="1",
                support_id="72290",
                voice_id="voice-1",
                phrase_id="phrase-1",
                greeting="Hi Joel,",
                body="this is Eugene from RiXtrema",
                state=task_worker_module.HealthState(),
            )

        self.assertTrue(ok)
        self.assertEqual(err, "")
        self.assertEqual(splice_mock.call_args.kwargs["pause_ms"], 220)
        self.assertEqual(splice_mock.call_args.kwargs["crossfade_ms"], 10)


if __name__ == "__main__":
    unittest.main()
