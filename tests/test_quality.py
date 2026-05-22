"""Unit tests for server.quality.

Importing server.quality pulls in server.tts, so we stub qwen_tts / soundfile /
librosa / pydub the same way the existing generation-stability tests do. The
model and ASR calls are patched, so no GPU or Whisper is required.
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


def _install_qwen_stub() -> None:
    fake_qwen_tts = types.ModuleType("qwen_tts")
    fake_qwen_tts.Qwen3TTSModel = object
    fake_qwen_tts.VoiceClonePromptItem = object
    sys.modules["qwen_tts"] = fake_qwen_tts

    fake_soundfile = types.ModuleType("soundfile")
    fake_soundfile.read = lambda *a, **k: (np.zeros(8, dtype=np.float32), 24000)
    fake_soundfile.write = lambda *a, **k: None
    sys.modules["soundfile"] = fake_soundfile

    fake_librosa = types.ModuleType("librosa")
    fake_librosa.resample = lambda y, orig_sr, target_sr: y
    sys.modules["librosa"] = fake_librosa

    fake_pydub = types.ModuleType("pydub")

    class FakeAudioSegment:
        frame_rate = 24000

        @classmethod
        def from_file(cls, *a, **k):
            return cls()

        def set_channels(self, *a, **k):
            return self

        def get_array_of_samples(self):
            return [0] * 8

    fake_pydub.AudioSegment = FakeAudioSegment
    sys.modules["pydub"] = fake_pydub


_seed_env()
_install_qwen_stub()
for mod in ("server.config", "server.tts", "server.asr", "server.quality"):
    sys.modules.pop(mod, None)

quality = importlib.import_module("server.quality")
asr = importlib.import_module("server.asr")


def _asr_report(wer=0.0, prefix=False, suffix=False, refleak=False):
    return asr.ASRReport(
        transcript_raw="hi dennis",
        transcript_normalized="hi dennis",
        target_normalized="hi dennis",
        wer=wer,
        cer=wer,
        has_prefix_extra=prefix,
        prefix_extra_tokens=["ma"] if prefix else [],
        has_suffix_clipped=suffix,
        has_reference_leak=refleak,
        no_speech_prob=0.01,
        confidence_score=-0.2,
    )


class CompositeScoreTests(unittest.TestCase):
    def test_clean_candidate_score(self) -> None:
        report = quality.QualityReport(
            similarity=0.8,
            similarity_passed=True,
            asr=_asr_report(wer=0.0),
            asr_passed=True,
        )
        # 1.0 * 0.8 + 1.5 * (1 - 0) = 2.3
        self.assertAlmostEqual(quality.composite_score(report), 2.3)

    def test_prefix_extra_penalizes(self) -> None:
        clean = quality.QualityReport(0.8, True, _asr_report(wer=0.0), True)
        dirty = quality.QualityReport(0.8, True, _asr_report(wer=0.0, prefix=True), True)
        self.assertAlmostEqual(
            quality.composite_score(clean) - quality.composite_score(dirty), 0.5
        )

    def test_reference_leak_is_heaviest_penalty(self) -> None:
        clean = quality.QualityReport(0.8, True, _asr_report(), True)
        leak = quality.QualityReport(0.8, True, _asr_report(refleak=True), True)
        self.assertAlmostEqual(
            quality.composite_score(clean) - quality.composite_score(leak), 1.0
        )

    def test_artifact_penalizes(self) -> None:
        clean = quality.QualityReport(0.8, True, _asr_report(), True)
        artifact = quality.QualityReport(
            0.8, True, _asr_report(), True, onset_artifact=True
        )
        self.assertAlmostEqual(
            quality.composite_score(clean) - quality.composite_score(artifact), 0.3
        )

    def test_score_without_asr(self) -> None:
        report = quality.QualityReport(0.8, True, None, True)
        # No ASR term contribution beyond max(0, 1-0)=1 -> 0.8 + 1.5
        self.assertAlmostEqual(quality.composite_score(report), 2.3)

    def test_duration_artifact_penalizes(self) -> None:
        clean = quality.QualityReport(0.8, True, _asr_report(), True)
        longish = quality.QualityReport(0.8, True, _asr_report(), True, duration_artifact=True)
        self.assertAlmostEqual(
            quality.composite_score(clean) - quality.composite_score(longish), 0.3
        )


class EvaluateCandidateTests(unittest.TestCase):
    def test_clean_full_phrase(self) -> None:
        wav = np.zeros(24000, dtype=np.float32)
        with patch.object(quality.tts, "speaker_similarity", return_value=0.8), patch.object(
            quality.tts,
            "detect_body_boundary_artifacts",
            return_value={"start_artifact": 0, "trailing_rebound_artifact": 0, "clipped_ending_artifact": 0},
        ), patch.object(quality, "transcribe_audio", return_value=_asr_report(wer=0.0)):
            report = quality.evaluate_candidate(
                wav=wav,
                sr=24000,
                text="Hi, Dennis.",
                reference_embedding=np.ones(8, dtype=np.float32),
                target_text="Hi, Dennis.",
                is_greeting=False,
                do_asr=True,
            )
        self.assertTrue(report.all_checks_passed)
        self.assertTrue(report.similarity_passed)
        self.assertTrue(report.asr_passed)
        self.assertAlmostEqual(report.composite_score, 2.3)

    def test_low_similarity_fails_checks(self) -> None:
        wav = np.zeros(24000, dtype=np.float32)
        with patch.object(quality.tts, "speaker_similarity", return_value=0.10), patch.object(
            quality.tts,
            "detect_body_boundary_artifacts",
            return_value={"start_artifact": 0, "trailing_rebound_artifact": 0, "clipped_ending_artifact": 0},
        ), patch.object(quality, "transcribe_audio", return_value=_asr_report(wer=0.0)):
            report = quality.evaluate_candidate(
                wav=wav,
                sr=24000,
                text="Hi, Dennis.",
                reference_embedding=np.ones(8, dtype=np.float32),
                target_text="Hi, Dennis.",
            )
        self.assertFalse(report.similarity_passed)
        self.assertFalse(report.all_checks_passed)

    def test_reference_leak_fails_asr_gate(self) -> None:
        wav = np.zeros(24000, dtype=np.float32)
        with patch.object(quality.tts, "speaker_similarity", return_value=0.8), patch.object(
            quality.tts,
            "detect_body_boundary_artifacts",
            return_value={"start_artifact": 0, "trailing_rebound_artifact": 0, "clipped_ending_artifact": 0},
        ), patch.object(quality, "transcribe_audio", return_value=_asr_report(wer=0.0, refleak=True)):
            report = quality.evaluate_candidate(
                wav=wav,
                sr=24000,
                text="Hi, Dennis.",
                reference_embedding=np.ones(8, dtype=np.float32),
                target_text="Hi, Dennis.",
            )
        self.assertFalse(report.asr_passed)
        self.assertFalse(report.all_checks_passed)

    def test_do_asr_false_skips_transcription(self) -> None:
        wav = np.zeros(24000, dtype=np.float32)
        with patch.object(quality.tts, "speaker_similarity", return_value=0.8), patch.object(
            quality.tts,
            "detect_body_boundary_artifacts",
            return_value={"start_artifact": 0, "trailing_rebound_artifact": 0, "clipped_ending_artifact": 0},
        ), patch.object(quality, "transcribe_audio", side_effect=AssertionError("should not be called")):
            report = quality.evaluate_candidate(
                wav=wav,
                sr=24000,
                text="Hi, Dennis.",
                reference_embedding=np.ones(8, dtype=np.float32),
                target_text="Hi, Dennis.",
                do_asr=False,
            )
        self.assertIsNone(report.asr)
        self.assertTrue(report.asr_passed)


class DiagnosticFieldsTests(unittest.TestCase):
    def test_fields_shape(self) -> None:
        report = quality.QualityReport(0.8, True, _asr_report(wer=0.1), True)
        report.composite_score = quality.composite_score(report)
        fields = quality.diagnostic_phrase_fields(report)
        self.assertEqual(fields["schema_version"], 2)
        self.assertEqual(fields["asr_transcript"], "hi dennis")
        self.assertEqual(fields["quality_gate_decision"], "skipped")
        self.assertIn("composite_score", fields)

    def test_fields_with_no_asr(self) -> None:
        report = quality.QualityReport(0.8, True, None, True)
        fields = quality.diagnostic_phrase_fields(report)
        self.assertIsNone(fields["asr_transcript"])
        self.assertIsNone(fields["asr_wer"])


if __name__ == "__main__":
    unittest.main()
