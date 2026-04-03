import importlib
import os
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np


def _seed_env() -> None:
    defaults = {
        "S3_BUCKET_NAME": "test-bucket",
        "AWS_ACCESS_KEY_ID": "test-access-key",
        "AWS_SECRET_ACCESS_KEY": "test-secret-key",
        "AWS_REGION": "us-east-1",
        "ADMIN_USER": "admin",
        "ADMIN_PASSWORD": "secret",
        "VOICE_CLONE_DO_SAMPLE": "false",
        "VOICE_CLONE_NON_STREAMING_MODE": "true",
        "VOICE_CLONE_MAX_NEW_TOKENS": "2048",
        "VOICE_CLONE_REPETITION_PENALTY": "1.02",
    }
    for key, value in defaults.items():
        os.environ[key] = value


def _install_qwen_stub() -> None:
    fake_qwen_tts = types.ModuleType("qwen_tts")
    fake_qwen_tts.Qwen3TTSModel = object
    fake_qwen_tts.VoiceClonePromptItem = object
    sys.modules["qwen_tts"] = fake_qwen_tts

    fake_soundfile = types.ModuleType("soundfile")
    fake_soundfile.read = lambda *args, **kwargs: (np.zeros(8, dtype=np.float32), 24000)
    fake_soundfile.write = lambda *args, **kwargs: None
    sys.modules["soundfile"] = fake_soundfile

    fake_librosa = types.ModuleType("librosa")
    fake_librosa.resample = lambda y, orig_sr, target_sr: y
    sys.modules["librosa"] = fake_librosa

    fake_pydub = types.ModuleType("pydub")

    class FakeAudioSegment:
        frame_rate = 24000

        @classmethod
        def from_file(cls, *args, **kwargs):
            return cls()

        def set_channels(self, *args, **kwargs):
            return self

        def get_array_of_samples(self):
            return [0] * 8

    fake_pydub.AudioSegment = FakeAudioSegment
    sys.modules["pydub"] = fake_pydub


_seed_env()
_install_qwen_stub()

sys.modules.pop("server.config", None)
sys.modules.pop("server.tts", None)

server_tts = importlib.import_module("server.tts")


class FakeModel:
    def __init__(self) -> None:
        self.kwargs = None

    def generate_voice_clone(self, **kwargs):
        self.kwargs = kwargs
        return [np.zeros(8, dtype=np.float32)], 24000


class ServerGenerationStabilityTests(unittest.TestCase):
    def test_generate_voice_uses_stable_clone_defaults(self) -> None:
        model = FakeModel()
        with patch.object(server_tts, "_get_model", return_value=model):
            wav, sr = server_tts.generate_voice("Hello there", voice_prompt=["prompt"])

        self.assertEqual(sr, 24000)
        self.assertEqual(wav.shape[0], 8)
        self.assertIsNotNone(model.kwargs)
        self.assertFalse(model.kwargs["do_sample"])
        self.assertTrue(model.kwargs["non_streaming_mode"])
        self.assertEqual(model.kwargs["max_new_tokens"], 2048)
        self.assertEqual(model.kwargs["repetition_penalty"], 1.02)
        self.assertNotIn("temperature", model.kwargs)
        self.assertNotIn("top_k", model.kwargs)
        self.assertNotIn("top_p", model.kwargs)

    def test_similarity_retry_accepts_second_attempt(self) -> None:
        wav = np.zeros(8, dtype=np.float32)
        with patch.object(server_tts, "generate_voice", side_effect=[(wav, 24000), (wav, 24000)]) as generate_mock, \
             patch.object(server_tts, "speaker_similarity", side_effect=[0.21, 0.81]):
            out_wav, sr, similarity, attempts, passed = server_tts.generate_voice_with_similarity_retry(
                text="Hi, Kevin,",
                voice_prompt=["prompt"],
                reference_embedding=np.array([0.1, 0.2], dtype=np.float32),
                min_similarity=0.55,
                max_attempts=3,
            )

        self.assertTrue(np.array_equal(out_wav, wav))
        self.assertEqual(sr, 24000)
        self.assertAlmostEqual(similarity, 0.81, places=6)
        self.assertEqual(attempts, 2)
        self.assertTrue(passed)
        self.assertEqual(generate_mock.call_count, 2)

    def test_similarity_retry_returns_best_attempt_when_threshold_not_met(self) -> None:
        wav1 = np.full(8, 0.1, dtype=np.float32)
        wav2 = np.full(8, 0.2, dtype=np.float32)
        wav3 = np.full(8, 0.3, dtype=np.float32)
        with patch.object(server_tts, "generate_voice", side_effect=[(wav1, 24000), (wav2, 24000), (wav3, 24000)]), \
             patch.object(server_tts, "speaker_similarity", side_effect=[0.20, 0.48, 0.35]):
            out_wav, sr, similarity, attempts, passed = server_tts.generate_voice_with_similarity_retry(
                text="Hi, Kevin,",
                voice_prompt=["prompt"],
                reference_embedding=np.array([0.1, 0.2], dtype=np.float32),
                min_similarity=0.55,
                max_attempts=3,
            )

        self.assertTrue(np.array_equal(out_wav, wav2))
        self.assertEqual(sr, 24000)
        self.assertAlmostEqual(similarity, 0.48, places=6)
        self.assertEqual(attempts, 3)
        self.assertFalse(passed)

    def test_trim_audio_edges_removes_leading_and_trailing_silence(self) -> None:
        silence = np.zeros(2400, dtype=np.float32)
        voice = np.full(4800, 0.2, dtype=np.float32)
        wav = np.concatenate([silence, voice, silence])

        cleaned, stats = server_tts.trim_audio_edges(
            wav,
            sr=24000,
            pad_ms=20,
            max_leading_ms=500,
            max_trailing_ms=500,
        )

        self.assertLess(cleaned.shape[0], wav.shape[0])
        self.assertGreater(stats["leading_ms"], 0)
        self.assertGreater(stats["trailing_ms"], 0)
        self.assertEqual(stats["trimmed"], 1)

    def test_trim_audio_edges_removes_leading_noise_burst(self) -> None:
        noise = np.random.default_rng(123).normal(0.0, 0.08, 2400).astype(np.float32)
        t = np.linspace(0.0, 0.25, 6000, endpoint=False, dtype=np.float32)
        voice = (0.24 * np.sin(2.0 * np.pi * 180.0 * t)).astype(np.float32)
        wav = np.concatenate([noise, voice, np.zeros(1200, dtype=np.float32)])

        cleaned, stats = server_tts.trim_audio_edges(
            wav,
            sr=24000,
            pad_ms=20,
            max_leading_ms=500,
            max_trailing_ms=500,
        )

        self.assertLess(cleaned.shape[0], wav.shape[0])
        self.assertGreater(stats["leading_ms"], 0)
        self.assertEqual(stats["trimmed"], 1)

    def test_clean_reference_audio_respects_disabled_flag(self) -> None:
        wav = np.concatenate([np.zeros(1200, dtype=np.float32), np.full(2400, 0.2, dtype=np.float32)])
        with patch.object(server_tts, "REFERENCE_AUDIO_TRIM_ENABLED", False):
            cleaned, sr, stats = server_tts.clean_reference_audio(wav, 24000)

        self.assertTrue(np.array_equal(cleaned, wav))
        self.assertEqual(sr, 24000)
        self.assertEqual(stats["trimmed"], 0)


if __name__ == "__main__":
    unittest.main()
