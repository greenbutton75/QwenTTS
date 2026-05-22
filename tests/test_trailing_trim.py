"""Unit tests for trim_trailing_silence (server.tts).

Uses real numpy audio (loud noise = speech-like, zeros = silence) so the VAD
path runs for real; only qwen_tts / soundfile / pydub are stubbed for import.
"""

import importlib
import os
import sys
import types
import unittest

import numpy as np


def _seed_env() -> None:
    for k, v in {
        "S3_BUCKET_NAME": "b", "AWS_ACCESS_KEY_ID": "k", "AWS_SECRET_ACCESS_KEY": "s",
        "AWS_REGION": "us-east-1", "ADMIN_USER": "a", "ADMIN_PASSWORD": "p",
    }.items():
        os.environ.setdefault(k, v)


def _install_stub() -> None:
    q = types.ModuleType("qwen_tts")
    q.Qwen3TTSModel = object
    q.VoiceClonePromptItem = object
    sys.modules["qwen_tts"] = q
    sfm = types.ModuleType("soundfile")
    sfm.read = lambda *a, **k: (np.zeros(8, dtype=np.float32), 24000)
    sfm.write = lambda *a, **k: None
    sys.modules["soundfile"] = sfm
    lr = types.ModuleType("librosa")
    lr.resample = lambda y, orig_sr, target_sr: y
    sys.modules["librosa"] = lr
    pd = types.ModuleType("pydub")
    pd.AudioSegment = type("AS", (), {"frame_rate": 24000})
    sys.modules["pydub"] = pd


_seed_env()
_install_stub()
for m in ("server.config", "server.tts"):
    sys.modules.pop(m, None)
tts = importlib.import_module("server.tts")

SR = 24000


def _speech(seconds, rng):
    # Loud broadband noise -> classified as speech-like by the VAD (rms >= strong).
    return (0.4 * rng.standard_normal(int(SR * seconds))).astype(np.float32)


class TrimTrailingSilenceTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_trims_long_trailing_silence(self):
        sig = np.concatenate([_speech(1.5, self.rng), np.zeros(int(SR * 4), dtype=np.float32)])
        out, stats = tts.trim_trailing_silence(sig, SR, pad_ms=500, min_trailing_ms=700)
        self.assertEqual(stats["trimmed"], 1)
        out_s = len(out) / SR
        # Speech ~1.5s + 0.5s pad ~= 2.0s; well under the original 5.5s.
        self.assertLess(out_s, 2.6)
        self.assertGreater(out_s, 1.5)

    def test_keeps_audio_without_trailing_silence(self):
        sig = _speech(2.0, self.rng)
        out, stats = tts.trim_trailing_silence(sig, SR, pad_ms=500, min_trailing_ms=700)
        self.assertEqual(stats["trimmed"], 0)
        self.assertEqual(len(out), len(sig))

    def test_short_trailing_silence_is_kept(self):
        # 300ms trailing < min_trailing_ms(700) -> no trim.
        sig = np.concatenate([_speech(1.5, self.rng), np.zeros(int(SR * 0.3), dtype=np.float32)])
        out, stats = tts.trim_trailing_silence(sig, SR, pad_ms=500, min_trailing_ms=700)
        self.assertEqual(stats["trimmed"], 0)

    def test_spliced_cleanup_applies_trailing_trim(self):
        sig = np.concatenate([_speech(2.0, self.rng), np.zeros(int(SR * 5), dtype=np.float32)])
        out, sr, stats = tts.clean_output_audio_for_spliced_phrase(sig, SR)
        self.assertEqual(sr, SR)
        self.assertEqual(stats["trailing_silence_trimmed"], 1)
        self.assertLess(len(out) / SR, 3.0)


if __name__ == "__main__":
    unittest.main()
