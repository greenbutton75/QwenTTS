"""Unit tests for server.asr.

These run without a GPU or faster-whisper: the pure analysis logic
(``analyze_transcript``, WER/CER) is tested directly, and ``transcribe_audio``
is tested against a mocked Whisper model.
"""

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
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


_seed_env()
sys.modules.pop("server.config", None)
sys.modules.pop("server.asr", None)
asr = importlib.import_module("server.asr")


class NormalizationTests(unittest.TestCase):
    def test_normalize_strips_punctuation_and_case(self) -> None:
        self.assertEqual(asr.normalize_text("Hi, Dennis."), "hi dennis")
        self.assertEqual(asr.normalize_text("  Hello   World!! "), "hello world")
        self.assertEqual(asr.normalize_text(None), "")

    def test_word_error_rate(self) -> None:
        self.assertEqual(asr.word_error_rate("hi dennis", "hi dennis"), 0.0)
        self.assertEqual(asr.word_error_rate("hi dennis", "hi"), 0.5)
        self.assertEqual(asr.word_error_rate("", ""), 0.0)
        self.assertEqual(asr.word_error_rate("", "extra"), 1.0)

    def test_char_error_rate(self) -> None:
        self.assertEqual(asr.char_error_rate("hi", "hi"), 0.0)
        self.assertAlmostEqual(asr.char_error_rate("hi", "ho"), 0.5)


class AnalyzeTranscriptTests(unittest.TestCase):
    def test_clean_greeting(self) -> None:
        report = asr.analyze_transcript("Hi, Dennis.", "Hi, Dennis.")
        self.assertEqual(report.wer, 0.0)
        self.assertFalse(report.has_prefix_extra)
        self.assertFalse(report.has_suffix_clipped)
        self.assertFalse(report.has_reference_leak)

    def test_prefix_extra(self) -> None:
        # English ASR emits a spurious latin syllable before the greeting.
        report = asr.analyze_transcript("ma Hi Dennis", "Hi, Dennis.")
        self.assertTrue(report.has_prefix_extra)
        self.assertEqual(report.prefix_extra_tokens, ["ma"])

    def test_suffix_clipped(self) -> None:
        report = asr.analyze_transcript("Hi De", "Hi, Dennis.")
        self.assertTrue(report.has_suffix_clipped)

    def test_no_suffix_clip_when_last_word_present(self) -> None:
        report = asr.analyze_transcript("Hi Dennis", "Hi, Dennis.")
        self.assertFalse(report.has_suffix_clipped)

    def test_reference_leak(self) -> None:
        report = asr.analyze_transcript(
            "me i hope hi dennis",
            "Hi, Dennis.",
            ref_text="Me, I hope you are well.",
        )
        self.assertTrue(report.has_reference_leak)

    def test_no_reference_leak_without_ref_text(self) -> None:
        report = asr.analyze_transcript("me i hope hi dennis", "Hi, Dennis.")
        self.assertFalse(report.has_reference_leak)

    def test_empty_transcript(self) -> None:
        report = asr.analyze_transcript("", "Hi, Dennis.", no_speech_prob=0.9)
        self.assertEqual(report.wer, 1.0)
        self.assertTrue(report.has_suffix_clipped)
        self.assertFalse(report.has_prefix_extra)
        self.assertEqual(report.no_speech_prob, 0.9)

    def test_fuzzy_name_match(self) -> None:
        # Whisper mis-spells the name by one char -> still counts as present.
        report = asr.analyze_transcript("Hi Denis", "Hi, Dennis.")
        self.assertFalse(report.has_suffix_clipped)


class FakeWord:
    def __init__(self, word: str) -> None:
        self.word = word


class FakeSegment:
    def __init__(self, text: str, no_speech_prob: float, avg_logprob: float) -> None:
        self.text = text
        self.no_speech_prob = no_speech_prob
        self.avg_logprob = avg_logprob
        self.words = [FakeWord(w) for w in text.split()]


class FakeWhisperModel:
    def __init__(self, segments) -> None:
        self._segments = segments

    def transcribe(self, audio, **kwargs):
        return iter(self._segments), types.SimpleNamespace(language="en")


class TranscribeAudioTests(unittest.TestCase):
    def test_transcribe_with_mocked_model(self) -> None:
        fake = FakeWhisperModel([FakeSegment(" Hi, Dennis.", 0.01, -0.2)])
        wav = np.zeros(16000, dtype=np.float32)
        with patch.object(asr, "_get_asr_model", return_value=fake):
            report = asr.transcribe_audio(wav, 16000, "Hi, Dennis.")
        self.assertEqual(report.transcript_raw, "Hi, Dennis.")
        self.assertEqual(report.wer, 0.0)
        self.assertAlmostEqual(report.no_speech_prob, 0.01)
        self.assertAlmostEqual(report.confidence_score, -0.2)

    def test_transcribe_prefix_extra_with_mocked_model(self) -> None:
        fake = FakeWhisperModel([FakeSegment("ma Hi Dennis", 0.02, -0.3)])
        wav = np.zeros(16000, dtype=np.float32)
        with patch.object(asr, "_get_asr_model", return_value=fake):
            report = asr.transcribe_audio(wav, 16000, "Hi, Dennis.")
        self.assertTrue(report.has_prefix_extra)


if __name__ == "__main__":
    unittest.main()
