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

    class FakeVoiceClonePromptItem:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    fake_qwen_tts.VoiceClonePromptItem = FakeVoiceClonePromptItem
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
    fake_tts.generate_voice_with_similarity_retry = (
        lambda text, voice_prompt, reference_embedding, min_similarity, max_attempts, **kwargs:
        (
            np.zeros(8, dtype=np.float32),
            24000,
            0.99,
            1,
            True,
            {
                "similarity_passed": 1,
                "onset_artifact": 0,
                "onset_checked": 1,
                "onset_passed": 1,
                "duration_artifact": 0,
                "duration_checked": 1,
                "duration_passed": 1,
                "ending_artifact": 0,
                "ending_checked": 1,
                "ending_passed": 1,
                "preroll_artifact": 0,
                "preroll_checked": 1,
                "preroll_passed": 1,
                "start_passed": 1,
                "greeting_passed": 1,
            },
        )
    )
    fake_tts.greeting_splice_generate_configs = lambda: ({"max_new_tokens": 256}, {"max_new_tokens": 256})
    fake_tts.is_fatal_cuda_error = lambda exc_or_text: False
    fake_tts.clean_output_audio = (
        lambda wav, sr: (
            wav,
            sr,
            {"trimmed": 0, "leading_ms": 0, "trailing_ms": 0, "original_ms": 0, "cleaned_ms": 0},
        )
    )
    fake_tts.clean_output_audio_preserve_start = fake_tts.clean_output_audio
    fake_tts.clean_output_audio_for_greeting = lambda text, wav, sr: fake_tts.clean_output_audio(wav, sr)
    fake_tts.clean_output_audio_for_spliced_phrase = fake_tts.clean_output_audio
    fake_tts.clean_reference_audio = (
        lambda wav, sr: (
            wav,
            sr,
            {"trimmed": 0, "leading_ms": 0, "trailing_ms": 0, "original_ms": 0, "cleaned_ms": 0},
        )
    )
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

    def test_body_cache_hash_changes_when_output_trim_config_changes(self) -> None:
        prompt = {
            "ref_code": np.array([[1, 2, 3]], dtype=np.int16),
            "ref_spk_embedding": np.array([0.1, 0.2], dtype=np.float32),
            "x_vector_only_mode": False,
            "icl_mode": True,
            "ref_text": "sample one",
        }
        trimmed_hash = body_cache_hash(
            "support-1",
            "voice-1",
            "Shared body",
            "1.7B",
            "English",
            prompt,
            generation_config={
                "voice_clone": {"do_sample": False},
                "output_trim": {"enabled": True, "pad_ms": 30},
            },
        )
        untrimmed_hash = body_cache_hash(
            "support-1",
            "voice-1",
            "Shared body",
            "1.7B",
            "English",
            prompt,
            generation_config={
                "voice_clone": {"do_sample": False},
                "output_trim": {"enabled": False, "pad_ms": 0},
            },
        )

        self.assertNotEqual(trimmed_hash, untrimmed_hash)

    def test_body_cache_hash_changes_when_trim_algorithm_version_changes(self) -> None:
        prompt = {
            "ref_code": np.array([[1, 2, 3]], dtype=np.int16),
            "ref_spk_embedding": np.array([0.1, 0.2], dtype=np.float32),
            "x_vector_only_mode": False,
            "icl_mode": True,
            "ref_text": "sample one",
        }
        hash_v1 = body_cache_hash(
            "support-1",
            "voice-1",
            "Shared body",
            "1.7B",
            "English",
            prompt,
            generation_config={
                "voice_clone": {"do_sample": False},
                "output_trim": {"enabled": True, "algorithm_version": "v1"},
            },
        )
        hash_v2 = body_cache_hash(
            "support-1",
            "voice-1",
            "Shared body",
            "1.7B",
            "English",
            prompt,
            generation_config={
                "voice_clone": {"do_sample": False},
                "output_trim": {"enabled": True, "algorithm_version": "v2"},
            },
        )

        self.assertNotEqual(hash_v1, hash_v2)

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
             patch.object(server_worker, "clean_reference_audio", return_value=(np.zeros(8, dtype=np.float32), 24000, {"trimmed": 1, "leading_ms": 200, "trailing_ms": 0, "original_ms": 1200, "cleaned_ms": 1000})), \
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
        self.assertEqual(voice_json["reference_trim"]["leading_ms"], 200)

    def test_full_phrase_greeting_uses_full_attempt_limit(self) -> None:
        worker = server_worker.Worker(db=MagicMock(), logger=MagicMock())
        payload = {
            "support_id": "support-1",
            "voice_id": "voice-1",
            "phrase_id": "phrase-1",
            "text": "Hi Dennis. I would like to learn more about your business.",
        }
        prompt_data = {
            "ref_code": np.array([[1, 2, 3]], dtype=np.int16),
            "ref_spk_embedding": np.array([0.1, 0.2], dtype=np.float32),
            "x_vector_only_mode": False,
            "icl_mode": True,
            "ref_text": "reference text",
        }
        quality = {
            "similarity_passed": 1,
            "onset_artifact": 0,
            "onset_checked": 1,
            "onset_passed": 1,
            "duration_artifact": 0,
            "duration_checked": 1,
            "duration_passed": 1,
            "ending_artifact": 0,
            "ending_checked": 1,
            "ending_passed": 1,
            "preroll_artifact": 0,
            "preroll_checked": 1,
            "preroll_passed": 1,
            "start_passed": 1,
            "greeting_passed": 1,
        }
        with patch.object(server_worker, "download_torch", return_value=prompt_data), \
             patch.object(server_worker, "generate_voice_with_similarity_retry", return_value=(np.zeros(8, dtype=np.float32), 24000, 0.91, 2, True, quality)) as retry_mock, \
             patch.object(server_worker, "clean_output_audio_for_greeting", return_value=(np.zeros(8, dtype=np.float32), 24000, {"trimmed": 0, "leading_ms": 0, "trailing_ms": 0, "original_ms": 0, "cleaned_ms": 0})), \
             patch.object(server_worker, "write_wav_temp", return_value="tmp.wav"), \
             patch.object(server_worker, "upload_file"), \
             patch.object(server_worker, "create_presigned_url", return_value="https://example.invalid/audio.wav"), \
             patch.object(server_worker, "write_json"), \
             patch.object(server_worker.os, "remove"), \
             patch.object(server_worker, "GREETING_FULL_PHRASE_MAX_ATTEMPTS", 4):
            worker._handle_phrase(payload)

        self.assertEqual(retry_mock.call_args.kwargs["max_attempts"], 4)
        self.assertEqual(retry_mock.call_args.kwargs["text"], "Hi Dennis.")

    def test_full_phrase_split_uses_short_greeting_probe_then_single_full_render(self) -> None:
        worker = server_worker.Worker(db=MagicMock(), logger=MagicMock())
        payload = {
            "support_id": "support-1",
            "voice_id": "voice-1",
            "phrase_id": "phrase-2",
            "text": "Hi Dennis. I would like to learn more about your business.",
        }
        prompt_data = {
            "ref_code": np.array([[1, 2, 3]], dtype=np.int16),
            "ref_spk_embedding": np.array([0.1, 0.2], dtype=np.float32),
            "x_vector_only_mode": False,
            "icl_mode": True,
            "ref_text": "reference text",
        }
        quality = {
            "similarity_passed": 1,
            "onset_artifact": 0,
            "onset_checked": 1,
            "onset_passed": 1,
            "duration_artifact": 1,
            "duration_checked": 1,
            "duration_passed": 0,
            "ending_artifact": 0,
            "ending_checked": 1,
            "ending_passed": 1,
            "preroll_artifact": 0,
            "preroll_checked": 1,
            "preroll_passed": 1,
            "start_passed": 1,
            "greeting_passed": 1,
        }
        with patch.object(server_worker, "download_torch", return_value=prompt_data), \
             patch.object(server_worker, "generate_voice_with_similarity_retry", return_value=(np.zeros(8, dtype=np.float32), 24000, 0.91, 2, True, quality)) as retry_mock, \
             patch.object(server_worker, "generate_voice", return_value=(np.zeros(8, dtype=np.float32), 24000)) as full_generate_mock, \
             patch.object(server_worker, "clean_output_audio", return_value=(np.zeros(8, dtype=np.float32), 24000, {"trimmed": 0, "leading_ms": 0, "trailing_ms": 0, "original_ms": 0, "cleaned_ms": 0})) as clean_mock, \
             patch.object(server_worker, "write_wav_temp", return_value="tmp.wav"), \
             patch.object(server_worker, "upload_file"), \
             patch.object(server_worker, "create_presigned_url", return_value="https://example.invalid/audio.wav"), \
             patch.object(server_worker, "write_json"), \
             patch.object(server_worker.os, "remove"), \
             patch.object(server_worker, "GREETING_FULL_PHRASE_MAX_ATTEMPTS", 2):
            worker._handle_phrase(payload)

        self.assertEqual(retry_mock.call_count, 1)
        self.assertEqual(retry_mock.call_args.kwargs["text"], "Hi Dennis.")
        self.assertEqual(retry_mock.call_args.kwargs["max_attempts"], 2)
        self.assertEqual(retry_mock.call_args.kwargs["timing_fields"]["probe_mode"], 1)
        self.assertEqual(full_generate_mock.call_count, 1)
        self.assertEqual(full_generate_mock.call_args.args[0], payload["text"])
        clean_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
