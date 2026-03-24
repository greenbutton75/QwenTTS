import importlib
import os
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np


def _seed_env() -> None:
    defaults = {
        "S3_BUCKET_NAME": "test-bucket",
        "AWS_ACCESS_KEY_ID": "test-access-key",
        "AWS_SECRET_ACCESS_KEY": "test-secret-key",
        "AWS_REGION": "us-east-1",
        "ADMIN_USER": "admin",
        "ADMIN_PASSWORD": "secret",
        "SQLITE_PATH": os.path.join(tempfile.gettempdir(), "qwentts_test_queue.db"),
        "LOG_DIR": os.path.join(tempfile.gettempdir(), "qwentts_test_logs"),
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def _install_test_stubs() -> None:
    fake_qwen_tts = types.ModuleType("qwen_tts")
    fake_qwen_tts.Qwen3TTSModel = object
    fake_qwen_tts.VoiceClonePromptItem = object
    sys.modules.setdefault("qwen_tts", fake_qwen_tts)

    fake_db = types.ModuleType("server.db")

    class FakeTaskDB:
        def __init__(self, path: str) -> None:
            self.path = path

        def enqueue(self, *args, **kwargs):
            return 1

        def get_next_due(self):
            return None

        def stats(self):
            return {}

        def stats_by_type(self):
            return {}

        def list_recent(self, limit: int = 50):
            return []

    fake_db.TaskDB = FakeTaskDB
    sys.modules.setdefault("server.db", fake_db)

    fake_tts = types.ModuleType("server.tts")
    fake_tts.bytes_to_wav_file = lambda data, suffix=".wav": "tmp.wav"
    fake_tts.create_voice_prompt = lambda ref_audio, ref_text, x_vector_only: None
    fake_tts.generate_voice = lambda text, voice_prompt: (np.zeros(8, dtype=np.float32), 24000)
    fake_tts.load_audio = lambda path: (np.zeros(8, dtype=np.float32), 24000)
    fake_tts.write_wav_temp = lambda wav, sr: "tmp.wav"
    fake_tts.splice_speech_segments = lambda **kwargs: np.zeros(8, dtype=np.float32)
    fake_tts.wav_from_bytes = lambda data: (np.zeros(8, dtype=np.float32), 24000)
    fake_tts.wav_to_bytes = lambda wav, sr: b"WAV"
    sys.modules.setdefault("server.tts", fake_tts)


_seed_env()
_install_test_stubs()

from server.cache_utils import body_cache_hash, prompt_fingerprint

server_worker = importlib.import_module("server.worker")


class ServerCacheLogicTests(unittest.TestCase):
    def test_prompt_fingerprint_changes_when_prompt_content_changes(self) -> None:
        base_prompt = {
            "ref_code": np.array([[1, 2, 3]], dtype=np.int16),
            "ref_spk_embedding": np.array([0.1, 0.2, 0.3], dtype=np.float32),
            "x_vector_only_mode": False,
            "icl_mode": True,
            "ref_text": "hello world",
        }
        changed_prompt = {
            **base_prompt,
            "ref_spk_embedding": np.array([0.9, 0.2, 0.3], dtype=np.float32),
        }

        self.assertNotEqual(
            prompt_fingerprint(base_prompt),
            prompt_fingerprint(changed_prompt),
        )

    def test_body_cache_hash_changes_when_prompt_content_changes(self) -> None:
        prompt_a = {
            "ref_code": np.array([[1, 2, 3]], dtype=np.int16),
            "ref_spk_embedding": np.array([0.1, 0.2], dtype=np.float32),
            "x_vector_only_mode": False,
            "icl_mode": True,
            "ref_text": "sample one",
        }
        prompt_b = {
            **prompt_a,
            "ref_code": np.array([[4, 5, 6]], dtype=np.int16),
        }

        hash_a = body_cache_hash("support-1", "voice-1", "Shared body", "1.7B", "English", prompt_a)
        hash_b = body_cache_hash("support-1", "voice-1", "Shared body", "1.7B", "English", prompt_b)

        self.assertNotEqual(hash_a, hash_b)

    def test_body_cache_hash_changes_when_generation_config_changes(self) -> None:
        prompt = {
            "ref_code": np.array([[1, 2, 3]], dtype=np.int16),
            "ref_spk_embedding": np.array([0.1, 0.2], dtype=np.float32),
            "x_vector_only_mode": False,
            "icl_mode": True,
            "ref_text": "sample one",
        }
        greedy_hash = body_cache_hash(
            "support-1",
            "voice-1",
            "Shared body",
            "1.7B",
            "English",
            prompt,
            generation_config={"do_sample": False, "non_streaming_mode": True},
        )
        sampled_hash = body_cache_hash(
            "support-1",
            "voice-1",
            "Shared body",
            "1.7B",
            "English",
            prompt,
            generation_config={"do_sample": True, "temperature": 0.9},
        )

        self.assertNotEqual(greedy_hash, sampled_hash)

    def test_profile_refresh_clears_splice_cache_and_records_prompt_fingerprint(self) -> None:
        worker = server_worker.Worker(db=MagicMock(), logger=MagicMock())
        prompt_item = SimpleNamespace(
            ref_code=np.array([[1, 2, 3]], dtype=np.int16),
            ref_spk_embedding=np.array([0.1, 0.2], dtype=np.float32),
            x_vector_only_mode=False,
            icl_mode=True,
            ref_text="reference text",
        )
        payload = {
            "support_id": "support-1",
            "voice_id": "voice-1",
            "voice_name": "Kevin",
            "ref_text": "reference text",
            "x_vector_only": False,
            "s3_sample_key": "support/support-1/voices/voice-1/sample.wav",
        }

        with patch.object(server_worker, "download_bytes", return_value=b"sample"), \
             patch.object(server_worker, "bytes_to_wav_file", return_value="tmp.wav"), \
             patch.object(server_worker, "load_audio", return_value=(np.zeros(8, dtype=np.float32), 24000)), \
             patch.object(server_worker, "create_voice_prompt", return_value=prompt_item), \
             patch.object(server_worker, "delete_prefix") as delete_prefix_mock, \
             patch.object(server_worker, "upload_file"), \
             patch.object(server_worker, "upload_torch") as upload_torch_mock, \
             patch.object(server_worker, "write_json") as write_json_mock, \
             patch.object(server_worker.os, "remove"):
            worker._handle_profile(payload)

        delete_prefix_mock.assert_called_once_with("support/support-1/voices/voice-1/splice_cache/")
        upload_payload = upload_torch_mock.call_args.args[1]
        voice_json = write_json_mock.call_args.args[1]
        self.assertEqual(
            voice_json["prompt_fingerprint"],
            prompt_fingerprint(upload_payload),
        )


if __name__ == "__main__":
    unittest.main()
