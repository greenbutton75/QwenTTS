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
    def test_is_fatal_cuda_error_detects_device_assert(self) -> None:
        self.assertTrue(server_tts.is_fatal_cuda_error("CUDA error: device-side assert triggered"))
        self.assertFalse(server_tts.is_fatal_cuda_error("timeout waiting for phrase"))

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

    def test_similarity_retry_does_not_pretrim_candidate_audio(self) -> None:
        wav = np.zeros(8, dtype=np.float32)
        with patch.object(server_tts, "generate_voice", return_value=(wav, 24000)), \
             patch.object(server_tts, "speaker_similarity", return_value=0.81), \
             patch.object(server_tts, "detect_greeting_onset_artifact", return_value={"artifact": 0, "checked": 1}), \
             patch.object(server_tts, "detect_greeting_leading_preroll_artifact", return_value={"artifact": 0, "checked": 1}), \
             patch.object(server_tts, "detect_greeting_clipped_ending_artifact", return_value={"artifact": 0, "checked": 1}), \
             patch.object(server_tts, "clean_output_audio", side_effect=AssertionError("should not pretrim")):
            out_wav, sr, similarity, attempts, passed, quality = server_tts.generate_voice_with_similarity_retry(
                text="Hi, Kevin,",
                voice_prompt=["prompt"],
                reference_embedding=np.array([0.1, 0.2], dtype=np.float32),
                min_similarity=0.55,
                max_attempts=3,
            )

        self.assertTrue(np.array_equal(out_wav, wav))
        self.assertEqual(sr, 24000)
        self.assertAlmostEqual(similarity, 0.81, places=6)
        self.assertEqual(attempts, 1)
        self.assertTrue(passed)
        self.assertEqual(quality["start_passed"], 1)
        self.assertEqual(quality["greeting_passed"], 1)

    def test_similarity_retry_accepts_second_attempt(self) -> None:
        wav = np.zeros(8, dtype=np.float32)
        with patch.object(server_tts, "generate_voice", side_effect=[(wav, 24000), (wav, 24000)]) as generate_mock, \
             patch.object(server_tts, "speaker_similarity", side_effect=[0.21, 0.81]), \
             patch.object(server_tts, "detect_greeting_onset_artifact", return_value={"artifact": 0, "checked": 1}), \
             patch.object(server_tts, "detect_greeting_leading_preroll_artifact", return_value={"artifact": 0, "checked": 1}), \
             patch.object(server_tts, "detect_greeting_clipped_ending_artifact", return_value={"artifact": 0, "checked": 1}):
            out_wav, sr, similarity, attempts, passed, quality = server_tts.generate_voice_with_similarity_retry(
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
        self.assertEqual(quality["onset_artifact"], 0)
        self.assertEqual(generate_mock.call_count, 2)

    def test_similarity_retry_returns_best_attempt_when_threshold_not_met(self) -> None:
        wav1 = np.full(8, 0.1, dtype=np.float32)
        wav2 = np.full(8, 0.2, dtype=np.float32)
        wav3 = np.full(8, 0.3, dtype=np.float32)
        with patch.object(server_tts, "generate_voice", side_effect=[(wav1, 24000), (wav2, 24000), (wav3, 24000)]), \
             patch.object(server_tts, "speaker_similarity", side_effect=[0.20, 0.48, 0.35]), \
             patch.object(server_tts, "detect_greeting_onset_artifact", return_value={"artifact": 0, "checked": 1}), \
             patch.object(server_tts, "detect_greeting_leading_preroll_artifact", return_value={"artifact": 0, "checked": 1}), \
             patch.object(server_tts, "detect_greeting_clipped_ending_artifact", return_value={"artifact": 0, "checked": 1}):
            out_wav, sr, similarity, attempts, passed, quality = server_tts.generate_voice_with_similarity_retry(
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
        self.assertEqual(quality["onset_artifact"], 0)

    def test_similarity_retry_rejects_long_preroll_candidate(self) -> None:
        wav_bad = np.full(8, 0.1, dtype=np.float32)
        wav_good = np.full(8, 0.2, dtype=np.float32)
        with patch.object(server_tts, "generate_voice", side_effect=[(wav_bad, 24000), (wav_good, 24000)]), \
             patch.object(server_tts, "speaker_similarity", side_effect=[0.82, 0.79]), \
             patch.object(server_tts, "detect_greeting_onset_artifact", return_value={"artifact": 0, "checked": 1}), \
             patch.object(server_tts, "detect_greeting_clipped_ending_artifact", return_value={"artifact": 0, "checked": 1}), \
             patch.object(
                 server_tts,
                 "detect_greeting_leading_preroll_artifact",
                 side_effect=[{"artifact": 1, "checked": 1}, {"artifact": 0, "checked": 1}],
             ):
            out_wav, sr, similarity, attempts, passed, quality = server_tts.generate_voice_with_similarity_retry(
                text="Hello Kevin,",
                voice_prompt=["prompt"],
                reference_embedding=np.array([0.1, 0.2], dtype=np.float32),
                min_similarity=0.55,
                max_attempts=3,
            )

        self.assertTrue(np.array_equal(out_wav, wav_good))
        self.assertEqual(sr, 24000)
        self.assertAlmostEqual(similarity, 0.79, places=6)
        self.assertEqual(attempts, 2)
        self.assertTrue(passed)
        self.assertEqual(quality["preroll_artifact"], 0)
        self.assertEqual(quality["start_passed"], 1)
        self.assertEqual(quality["greeting_passed"], 1)

    def test_similarity_retry_rejects_clipped_greeting_ending(self) -> None:
        wav_bad = np.full(8, 0.1, dtype=np.float32)
        wav_good = np.full(8, 0.2, dtype=np.float32)
        with patch.object(server_tts, "generate_voice", side_effect=[(wav_bad, 24000), (wav_good, 24000)]), \
             patch.object(server_tts, "speaker_similarity", side_effect=[0.83, 0.80]), \
             patch.object(server_tts, "detect_greeting_onset_artifact", return_value={"artifact": 0, "checked": 1}), \
             patch.object(server_tts, "detect_greeting_leading_preroll_artifact", return_value={"artifact": 0, "checked": 1}), \
             patch.object(
                 server_tts,
                 "detect_greeting_clipped_ending_artifact",
                 side_effect=[{"artifact": 1, "checked": 1}, {"artifact": 0, "checked": 1}],
             ):
            out_wav, sr, similarity, attempts, passed, quality = server_tts.generate_voice_with_similarity_retry(
                text="Hi, Dennis.",
                voice_prompt=["prompt"],
                reference_embedding=np.array([0.1, 0.2], dtype=np.float32),
                min_similarity=0.55,
                max_attempts=3,
            )

        self.assertTrue(np.array_equal(out_wav, wav_good))
        self.assertEqual(sr, 24000)
        self.assertAlmostEqual(similarity, 0.80, places=6)
        self.assertEqual(attempts, 2)
        self.assertTrue(passed)
        self.assertEqual(quality["ending_artifact"], 0)
        self.assertEqual(quality["greeting_passed"], 1)

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

    def test_clean_output_audio_removes_multi_second_leading_silence(self) -> None:
        silence = np.zeros(24000 * 7, dtype=np.float32)
        voice = np.full(4800, 0.2, dtype=np.float32)
        wav = np.concatenate([silence, voice])

        cleaned, sr, stats = server_tts.clean_output_audio(wav, 24000)

        self.assertEqual(sr, 24000)
        self.assertLess(cleaned.shape[0], wav.shape[0] // 3)
        self.assertGreaterEqual(stats["leading_ms"], 6500)
        self.assertEqual(stats["trimmed"], 1)

    def test_clean_output_audio_compacts_long_internal_silence(self) -> None:
        voice_a = np.full(4800, 0.2, dtype=np.float32)
        silence = np.zeros(24000 * 5, dtype=np.float32)
        voice_b = np.full(4800, 0.2, dtype=np.float32)
        wav = np.concatenate([voice_a, silence, voice_b])

        cleaned, sr, stats = server_tts.clean_output_audio(wav, 24000)

        self.assertEqual(sr, 24000)
        self.assertLess(cleaned.shape[0], wav.shape[0] - (24000 * 3))
        self.assertEqual(stats["internal_silence_compressed"], 1)
        self.assertGreaterEqual(stats["internal_silence_removed_ms"], 3500)

    def test_clean_output_audio_preserve_start_uses_larger_leading_pad(self) -> None:
        wav = np.full(16, 0.1, dtype=np.float32)
        edge_stats = {"trimmed": 0, "leading_ms": 0, "trailing_ms": 0, "original_ms": 1, "cleaned_ms": 1}
        silence_stats = {"compressed": 0, "spans": 0, "removed_ms": 0, "original_ms": 1, "cleaned_ms": 1}
        with patch.object(server_tts, "trim_audio_edges", return_value=(wav, edge_stats)) as trim_mock, \
             patch.object(server_tts, "compact_internal_silences", return_value=(wav, silence_stats)):
            server_tts.clean_output_audio_preserve_start(wav, 24000)

        self.assertEqual(trim_mock.call_args.kwargs["pad_ms"], server_tts.GREETING_OUTPUT_TRIM_PAD_MS)

    def test_clean_output_audio_for_short_greeting_disables_trailing_trim(self) -> None:
        wav = np.full(16, 0.1, dtype=np.float32)
        edge_stats = {"trimmed": 0, "leading_ms": 0, "trailing_ms": 0, "original_ms": 1, "cleaned_ms": 1}
        boundary_stats = {"trimmed": 0, "leading_ms": 0, "trailing_ms": 0, "original_ms": 1, "cleaned_ms": 1}
        silence_stats = {"compressed": 0, "spans": 0, "removed_ms": 0, "original_ms": 1, "cleaned_ms": 1}
        with patch.object(server_tts, "trim_audio_edges", return_value=(wav, edge_stats)) as trim_mock, \
             patch.object(server_tts, "trim_low_energy_boundary_artifacts", return_value=(wav, boundary_stats)), \
             patch.object(server_tts, "compact_internal_silences", return_value=(wav, silence_stats)):
            server_tts.clean_output_audio_for_greeting("Hi, Dennis.", wav, 24000)

        self.assertEqual(trim_mock.call_args.kwargs["max_trailing_ms"], 0)
        self.assertEqual(trim_mock.call_args.kwargs["pad_ms"], server_tts.GREETING_OUTPUT_TRIM_PAD_MS)

    def test_clean_output_audio_without_leading_trim_disables_leading_trim(self) -> None:
        wav = np.full(16, 0.1, dtype=np.float32)
        edge_stats = {"trimmed": 0, "leading_ms": 0, "trailing_ms": 0, "original_ms": 1, "cleaned_ms": 1}
        boundary_stats = {"trimmed": 0, "leading_ms": 0, "trailing_ms": 0, "original_ms": 1, "cleaned_ms": 1}
        silence_stats = {"compressed": 0, "spans": 0, "removed_ms": 0, "original_ms": 1, "cleaned_ms": 1}
        with patch.object(server_tts, "trim_audio_edges", return_value=(wav, edge_stats)) as trim_mock, \
             patch.object(server_tts, "trim_low_energy_boundary_artifacts", return_value=(wav, boundary_stats)) as boundary_mock, \
             patch.object(server_tts, "compact_internal_silences", return_value=(wav, silence_stats)):
            server_tts.clean_output_audio_without_leading_trim(wav, 24000)

        self.assertEqual(trim_mock.call_args.kwargs["max_leading_ms"], 0)
        self.assertTrue(boundary_mock.call_args.kwargs["allow_leading_artifact_trim"])

    def test_clean_output_audio_without_leading_trim_removes_long_low_energy_preroll(self) -> None:
        sr = 24000
        rng = np.random.default_rng(21)
        preroll = rng.normal(0.0, 0.001, int(sr * 6.0)).astype(np.float32)
        t = np.linspace(0.0, 1.0, int(sr * 1.0), endpoint=False, dtype=np.float32)
        speech = (0.16 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
        wav = np.concatenate([preroll, speech, speech])

        cleaned, out_sr, stats = server_tts.clean_output_audio_without_leading_trim(wav, sr)

        self.assertEqual(out_sr, sr)
        self.assertLess(cleaned.shape[0], wav.shape[0] - int(sr * 5.0))
        self.assertGreaterEqual(stats["boundary_leading_ms"], 5000)
        self.assertEqual(stats["boundary_artifact_trimmed"], 1)

    def test_clean_output_audio_without_leading_trim_preserves_soft_real_start(self) -> None:
        sr = 24000
        rng = np.random.default_rng(23)
        breath = rng.normal(0.0, 0.010, int(sr * 0.08)).astype(np.float32)
        t = np.linspace(0.0, 1.0, int(sr * 1.0), endpoint=False, dtype=np.float32)
        speech = (0.12 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
        wav = np.concatenate([breath, speech, speech * 1.02])

        cleaned, out_sr, stats = server_tts.clean_output_audio_without_leading_trim(wav, sr)

        self.assertEqual(out_sr, sr)
        self.assertGreaterEqual(cleaned.shape[0], wav.shape[0] - int(sr * 0.2))
        self.assertEqual(stats["boundary_leading_ms"], 0)

    def test_clean_output_audio_trims_weak_trailing_tail(self) -> None:
        sr = 24000
        t = np.linspace(0.0, 1.2, int(sr * 1.2), endpoint=False, dtype=np.float32)
        speech = (0.17 * np.sin(2.0 * np.pi * 210.0 * t)).astype(np.float32)
        tail_t = np.linspace(0.0, 1.2, int(sr * 1.2), endpoint=False, dtype=np.float32)
        tail = (0.035 * np.sin(2.0 * np.pi * 190.0 * tail_t)).astype(np.float32)
        gap = int(sr * 0.08)
        period = gap * 2
        for start in range(0, tail.shape[0], period):
            tail[start : start + gap] = 0.0
        wav = np.concatenate([speech, speech * 0.98, tail])

        cleaned, out_sr, stats = server_tts.clean_output_audio_without_leading_trim(wav, sr)

        self.assertEqual(out_sr, sr)
        self.assertLess(cleaned.shape[0], wav.shape[0] - int(sr * 0.3))
        self.assertGreaterEqual(stats["boundary_trailing_ms"], 400)
        self.assertEqual(stats["boundary_artifact_trimmed"], 1)

    def test_clean_output_audio_trims_long_low_clarity_prefix_and_suffix(self) -> None:
        sr = 24000
        rng = np.random.default_rng(31)
        prefix_t = np.linspace(0.0, 20.0, int(sr * 20.0), endpoint=False, dtype=np.float32)
        prefix = (
            0.030 * np.sin(2.0 * np.pi * 120.0 * prefix_t)
            + 0.0105 * np.sin(2.0 * np.pi * 240.0 * prefix_t)
        ).astype(np.float32)
        prefix += rng.normal(0.0, 0.0015, prefix.shape[0]).astype(np.float32)

        speech_t = np.linspace(0.0, 22.0, int(sr * 22.0), endpoint=False, dtype=np.float32)
        carrier = np.sin(2.0 * np.pi * (180.0 + 30.0 * np.sin(2.0 * np.pi * 2.8 * speech_t)) * speech_t)
        mod = 0.55 + 0.45 * np.sin(2.0 * np.pi * 4.0 * speech_t)
        speech = (0.085 * carrier * mod).astype(np.float32)
        speech += rng.normal(0.0, 0.0025, speech.shape[0]).astype(np.float32)

        suffix_t = np.linspace(0.0, 20.0, int(sr * 20.0), endpoint=False, dtype=np.float32)
        suffix = (
            0.030 * np.sin(2.0 * np.pi * 110.0 * suffix_t)
            + 0.009 * np.sin(2.0 * np.pi * 220.0 * suffix_t)
        ).astype(np.float32)
        suffix += rng.normal(0.0, 0.0012, suffix.shape[0]).astype(np.float32)

        wav = np.concatenate([prefix, speech, suffix])

        cleaned, out_sr, stats = server_tts.clean_output_audio_without_leading_trim(wav, sr)

        self.assertEqual(out_sr, sr)
        self.assertLess(cleaned.shape[0], wav.shape[0] - int(sr * 15.0))
        self.assertGreaterEqual(stats["clarity_leading_ms"], 10000)
        self.assertGreaterEqual(stats["clarity_trailing_ms"], 10000)
        self.assertEqual(stats["clarity_boundary_trimmed"], 1)

    def test_refine_local_clarity_boundaries_trims_residual_trailing_buzz(self) -> None:
        sr = 24000
        rng = np.random.default_rng(33)

        lead_t = np.linspace(0.0, 2.0, int(sr * 2.0), endpoint=False, dtype=np.float32)
        lead = (0.050 * np.sin(2.0 * np.pi * 170.0 * lead_t)).astype(np.float32)
        lead += rng.normal(0.0, 0.0020, lead.shape[0]).astype(np.float32)

        speech_t = np.linspace(0.0, 24.0, int(sr * 24.0), endpoint=False, dtype=np.float32)
        carrier = np.sin(2.0 * np.pi * (185.0 + 25.0 * np.sin(2.0 * np.pi * 3.2 * speech_t)) * speech_t)
        mod = 0.55 + 0.45 * np.sin(2.0 * np.pi * 4.3 * speech_t)
        speech = (0.085 * carrier * mod).astype(np.float32)
        speech += rng.normal(0.0, 0.002, speech.shape[0]).astype(np.float32)

        tail_t = np.linspace(0.0, 4.0, int(sr * 4.0), endpoint=False, dtype=np.float32)
        tail = (0.012 * np.sin(2.0 * np.pi * 70.0 * tail_t)).astype(np.float32)
        tail += rng.normal(0.0, 0.0008, tail.shape[0]).astype(np.float32)

        wav = np.concatenate([lead, speech, tail])

        cleaned, stats = server_tts.refine_local_clarity_boundaries(wav, sr, pad_ms=server_tts.OUTPUT_AUDIO_TRIM_PAD_MS)

        self.assertLess(cleaned.shape[0], wav.shape[0] - int(sr * 2.5))
        self.assertGreaterEqual(stats["trailing_ms"], 2000)
        self.assertEqual(stats["trimmed"], 1)

    def test_detect_greeting_onset_artifact_flags_stationary_voiced_start(self) -> None:
        sr = 24000
        t = np.linspace(0.0, 0.35, int(sr * 0.35), endpoint=False, dtype=np.float32)
        voiced = (0.09 * np.sin(2.0 * np.pi * 180.0 * t)).astype(np.float32)
        wav = np.concatenate([voiced, np.full(4800, 0.2, dtype=np.float32)])

        stats = server_tts.detect_greeting_onset_artifact("Hi Kevin,", wav, sr)

        self.assertEqual(stats["checked"], 1)
        self.assertEqual(stats["artifact"], 1)

    def test_detect_greeting_onset_artifact_allows_breathy_hi_start(self) -> None:
        sr = 24000
        rng = np.random.default_rng(7)
        breath = rng.normal(0.0, 0.015, int(sr * 0.08)).astype(np.float32)
        t = np.linspace(0.0, 0.27, int(sr * 0.27), endpoint=False, dtype=np.float32)
        vowel = (0.09 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
        wav = np.concatenate([breath, vowel, np.full(4800, 0.2, dtype=np.float32)])

        stats = server_tts.detect_greeting_onset_artifact("Hello Kevin,", wav, sr)

        self.assertEqual(stats["checked"], 1)
        self.assertEqual(stats["artifact"], 0)

    def test_detect_greeting_leading_preroll_artifact_flags_long_low_energy_preroll(self) -> None:
        sr = 24000
        rng = np.random.default_rng(9)
        preroll = rng.normal(0.0, 0.010, int(sr * 2.5)).astype(np.float32)
        t = np.linspace(0.0, 0.8, int(sr * 0.8), endpoint=False, dtype=np.float32)
        speech = (0.18 * np.sin(2.0 * np.pi * 210.0 * t)).astype(np.float32)
        wav = np.concatenate([preroll, speech, speech])

        stats = server_tts.detect_greeting_leading_preroll_artifact("Hello Kevin,", wav, sr)

        self.assertEqual(stats["checked"], 1)
        self.assertEqual(stats["artifact"], 1)
        self.assertGreaterEqual(stats["strong_start_ms"], 1500)

    def test_detect_greeting_leading_preroll_artifact_allows_prompt_speech_start(self) -> None:
        sr = 24000
        rng = np.random.default_rng(11)
        breath = rng.normal(0.0, 0.010, int(sr * 0.08)).astype(np.float32)
        t = np.linspace(0.0, 1.0, int(sr * 1.0), endpoint=False, dtype=np.float32)
        speech = (0.12 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
        wav = np.concatenate([breath, speech, speech * 1.05])

        stats = server_tts.detect_greeting_leading_preroll_artifact("Hi Kevin,", wav, sr)

        self.assertEqual(stats["checked"], 1)
        self.assertEqual(stats["artifact"], 0)

    def test_detect_greeting_clipped_ending_artifact_flags_short_abrupt_hi_name(self) -> None:
        sr = 24000
        t = np.linspace(0.0, 0.46, int(sr * 0.46), endpoint=False, dtype=np.float32)
        wav = (0.14 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)

        stats = server_tts.detect_greeting_clipped_ending_artifact("Hi, Dennis.", wav, sr)

        self.assertEqual(stats["checked"], 1)
        self.assertEqual(stats["artifact"], 1)
        self.assertLessEqual(stats["duration_ms"], stats["expected_min_ms"])
        self.assertGreaterEqual(stats["tail_speech_ratio"], 0.65)

    def test_detect_greeting_clipped_ending_artifact_allows_short_greeting_with_release(self) -> None:
        sr = 24000
        t = np.linspace(0.0, 0.78, int(sr * 0.78), endpoint=False, dtype=np.float32)
        body = (0.12 * np.sin(2.0 * np.pi * 210.0 * t)).astype(np.float32)
        fade = np.linspace(1.0, 0.0, int(sr * 0.12), dtype=np.float32)
        wav = body.copy()
        wav[-fade.size :] *= fade

        stats = server_tts.detect_greeting_clipped_ending_artifact("Hi, Dennis.", wav, sr)

        self.assertEqual(stats["checked"], 1)
        self.assertEqual(stats["artifact"], 0)

    def test_detect_body_boundary_artifacts_flags_false_start(self) -> None:
        sr = 24000
        wav = np.zeros(int(sr * 2.0), dtype=np.float32)
        mask = np.zeros(200, dtype=bool)
        mask[0:18] = True
        mask[32:88] = True
        with patch.object(server_tts, "_speech_frame_mask", return_value=(mask, int(sr * 0.02), int(sr * 0.01))):
            stats = server_tts.detect_body_boundary_artifacts("I hope you're doing well today.", wav, sr)

        self.assertEqual(stats["checked"], 1)
        self.assertEqual(stats["start_artifact"], 1)
        self.assertEqual(stats["passed"], 0)

    def test_detect_body_boundary_artifacts_flags_trailing_rebound(self) -> None:
        sr = 24000
        wav = np.zeros(int(sr * 2.3), dtype=np.float32)
        mask = np.zeros(230, dtype=bool)
        mask[10:150] = True
        mask[175:196] = True
        with patch.object(server_tts, "_speech_frame_mask", return_value=(mask, int(sr * 0.02), int(sr * 0.01))):
            stats = server_tts.detect_body_boundary_artifacts("I hope you're doing well today.", wav, sr)

        self.assertEqual(stats["checked"], 1)
        self.assertEqual(stats["trailing_rebound_artifact"], 1)
        self.assertEqual(stats["passed"], 0)

    def test_detect_body_boundary_artifacts_allows_clean_body(self) -> None:
        sr = 24000
        wav = np.zeros(int(sr * 2.2), dtype=np.float32)
        mask = np.zeros(220, dtype=bool)
        mask[8:205] = True
        with patch.object(server_tts, "_speech_frame_mask", return_value=(mask, int(sr * 0.02), int(sr * 0.01))):
            stats = server_tts.detect_body_boundary_artifacts("I hope you're doing well today.", wav, sr)

        self.assertEqual(stats["checked"], 1)
        self.assertEqual(stats["passed"], 1)
        self.assertEqual(stats["start_artifact"], 0)
        self.assertEqual(stats["trailing_rebound_artifact"], 0)

    def test_generate_body_with_quality_retry_retries_after_bad_candidate(self) -> None:
        wav_bad = np.full(16, 0.1, dtype=np.float32)
        wav_good = np.full(16, 0.2, dtype=np.float32)
        trim_stats = {"trimmed": 0, "leading_ms": 0, "trailing_ms": 0, "original_ms": 1, "cleaned_ms": 1}
        with patch.object(server_tts, "generate_voice", side_effect=[(wav_bad, 24000), (wav_good, 24000)]) as generate_mock, \
             patch.object(server_tts, "clean_output_audio", side_effect=[(wav_bad, 24000, trim_stats), (wav_good, 24000, trim_stats)]), \
             patch.object(
                 server_tts,
                 "detect_body_boundary_artifacts",
                 side_effect=[
                     {"checked": 1, "passed": 0, "start_artifact": 1, "trailing_rebound_artifact": 0},
                     {"checked": 1, "passed": 1, "start_artifact": 0, "trailing_rebound_artifact": 0},
                 ],
             ):
            out_wav, sr, attempts, passed, quality, out_trim = server_tts.generate_body_with_quality_retry(
                "I hope you're doing well today.",
                ["prompt"],
                max_attempts=3,
            )

        self.assertTrue(np.array_equal(out_wav, wav_good))
        self.assertEqual(sr, 24000)
        self.assertEqual(attempts, 2)
        self.assertTrue(passed)
        self.assertEqual(quality["passed"], 1)
        self.assertEqual(out_trim, trim_stats)
        self.assertIsNone(generate_mock.call_args_list[0].kwargs.get("generate_config"))
        self.assertEqual(generate_mock.call_args_list[1].kwargs.get("generate_config"), server_tts._BODY_RETRY_GENERATE_CONFIG)

    def test_content_aware_splice_preserves_full_greeting_tail(self) -> None:
        greeting = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        body = np.array([0.5, 0.6, 0.7], dtype=np.float32)
        with patch.object(server_tts, "_find_boundary_sample", return_value=1), \
             patch.object(server_tts, "_normalize_loudness_rms", side_effect=lambda signal, target_lufs: signal):
            out = server_tts.splice_speech_segments(
                greeting_wav=greeting,
                body_wav=body,
                sample_rate=24000,
                pause_ms=0,
                crossfade_ms=0,
                content_aware=True,
                target_lufs=-16.0,
            )

        expected = np.concatenate([greeting, body[1:]], axis=0)
        self.assertTrue(np.array_equal(out, expected))


if __name__ == "__main__":
    unittest.main()
