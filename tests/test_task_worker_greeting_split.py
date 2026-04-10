import importlib
import os
import sys
import types
import unittest


def _seed_env() -> None:
    defaults = {
        "USER_TOKEN": "user-token",
        "SYSTEM_TOKEN": "system-token",
        "FINGERPRINT": "fingerprint",
        "AWS_ACCESS_KEY_ID": "test-access-key",
        "AWS_SECRET_ACCESS_KEY": "test-secret-key",
        "AWS_REGION": "us-east-1",
        "S3_BUCKET_NAME": "test-bucket",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def _install_test_stubs() -> None:
    fake_health = types.ModuleType("task_worker.health")

    class FakeHealthState:
        pass

    fake_health.HealthState = FakeHealthState
    sys.modules.setdefault("task_worker.health", fake_health)

    fake_qwen_client = types.ModuleType("task_worker.qwen_client")
    fake_qwen_client.create_phrase = lambda *args, **kwargs: {}
    fake_qwen_client.create_phrase_splice = lambda *args, **kwargs: {}
    fake_qwen_client.create_profile = lambda *args, **kwargs: {}
    fake_qwen_client.get_phrase_status = lambda *args, **kwargs: {"status": "done"}
    fake_qwen_client.get_profile_status = lambda *args, **kwargs: {"status": "done"}
    fake_qwen_client.health_check = lambda: {"status": "ok"}
    sys.modules.setdefault("task_worker.qwen_client", fake_qwen_client)

    fake_s3_utils = types.ModuleType("task_worker.s3_utils")
    fake_s3_utils.object_exists = lambda key: True
    fake_s3_utils.read_json = lambda key: {"status": "done"}
    sys.modules.setdefault("task_worker.s3_utils", fake_s3_utils)

    fake_task_api = types.ModuleType("task_worker.task_api")
    fake_task_api.complete_task = lambda *args, **kwargs: None
    fake_task_api.failed_task = lambda *args, **kwargs: None
    fake_task_api.list_tasks = lambda *args, **kwargs: []
    fake_task_api.task_id_from_record = lambda rec: "1"
    fake_task_api.task_params_from_record = lambda rec: {}
    fake_task_api.update_progress = lambda *args, **kwargs: None
    sys.modules.setdefault("task_worker.task_api", fake_task_api)


_seed_env()
_install_test_stubs()

task_worker_module = importlib.import_module("task_worker.worker")


class TaskWorkerGreetingSplitTests(unittest.TestCase):
    def test_split_accepts_greeting_variants(self) -> None:
        cases = [
            ("Hi Kevin, this is Fred", ("Hi Kevin,", "this is Fred")),
            ("Hi, Kevin, this is Fred", ("Hi, Kevin,", "this is Fred")),
            ("Hi, Kevin. this is Fred", ("Hi, Kevin.", "this is Fred")),
            ("Hi , Kevin, this is Fred", ("Hi , Kevin,", "this is Fred")),
            ("Hi! Kevin, this is Fred", ("Hi! Kevin,", "this is Fred")),
            ("Hello Kevin! This is Fred", ("Hello Kevin!", "This is Fred")),
            ("Hello, Kevin! This is Fred", ("Hello, Kevin!", "This is Fred")),
            ("Hello, Kevin. This is Fred", ("Hello, Kevin.", "This is Fred")),
            ("hello, Kevin! This is Fred", ("hello, Kevin!", "This is Fred")),
            ("Hello,O'Neil! This is Fred", ("Hello,O'Neil!", "This is Fred")),
            ("Hi, Anne-Marie, this is Fred", ("Hi, Anne-Marie,", "this is Fred")),
        ]

        for text, expected in cases:
            with self.subTest(text=text):
                result = task_worker_module._split_greeting_body(text)
                self.assertEqual(result, expected)

    def test_split_rejects_missing_separator(self) -> None:
        result = task_worker_module._split_greeting_body("HiKevin, this is Fred")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
