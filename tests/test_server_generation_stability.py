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


if __name__ == "__main__":
    unittest.main()
