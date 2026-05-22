"""Unit tests for server.candidate_pool (best-of-N greeting selection).

Model generation and quality evaluation are patched, so no GPU/Whisper needed.
"""

import importlib
import os
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np


def _seed_env() -> None:
    for key, value in {
        "S3_BUCKET_NAME": "test-bucket",
        "AWS_ACCESS_KEY_ID": "k",
        "AWS_SECRET_ACCESS_KEY": "s",
        "AWS_REGION": "us-east-1",
        "ADMIN_USER": "admin",
        "ADMIN_PASSWORD": "secret",
    }.items():
        os.environ.setdefault(key, value)


def _install_qwen_stub() -> None:
    fake = types.ModuleType("qwen_tts")
    fake.Qwen3TTSModel = object
    fake.VoiceClonePromptItem = object
    sys.modules["qwen_tts"] = fake
    sf = types.ModuleType("soundfile")
    sf.read = lambda *a, **k: (np.zeros(8, dtype=np.float32), 24000)
    sf.write = lambda *a, **k: None
    sys.modules["soundfile"] = sf
    lr = types.ModuleType("librosa")
    lr.resample = lambda y, orig_sr, target_sr: y
    sys.modules["librosa"] = lr
    pd = types.ModuleType("pydub")

    class _AS:
        frame_rate = 24000

        @classmethod
        def from_file(cls, *a, **k):
            return cls()

        def set_channels(self, *a, **k):
            return self

        def get_array_of_samples(self):
            return [0] * 8

    pd.AudioSegment = _AS
    sys.modules["pydub"] = pd


_seed_env()
_install_qwen_stub()
for mod in ("server.config", "server.tts", "server.asr", "server.quality", "server.candidate_pool"):
    sys.modules.pop(mod, None)

cp = importlib.import_module("server.candidate_pool")
quality = importlib.import_module("server.quality")


def _report(score, passed):
    r = quality.QualityReport(similarity=0.8, similarity_passed=True, asr=None, asr_passed=True)
    r.composite_score = score
    r.all_checks_passed = passed
    return r


class SelectBestTests(unittest.TestCase):
    def _cand(self, label, score, passed):
        return cp.Candidate(
            spec=cp.CandidateSpec(label, {}, None),
            wav=np.zeros(8, dtype=np.float32),
            sr=24000,
            generate_latency_ms=10,
            report=_report(score, passed),
            score=score,
        )

    def test_prefers_clean_even_if_lower_score(self) -> None:
        # clean candidate scores lower than a dirty one -> clean still wins.
        clean = self._cand("sample_lo", 1.7, True)
        dirty = self._cand("greedy", 2.5, False)
        best = cp.select_best([dirty, clean])
        self.assertEqual(best.spec.label, "sample_lo")

    def test_falls_back_to_highest_score_when_none_clean(self) -> None:
        a = self._cand("greedy", -0.3, False)
        b = self._cand("sample_hi", 0.9, False)
        best = cp.select_best([a, b])
        self.assertEqual(best.spec.label, "sample_hi")

    def test_picks_highest_among_clean(self) -> None:
        a = self._cand("sample_lo", 1.5, True)
        b = self._cand("sample_hi", 2.2, True)
        best = cp.select_best([a, b])
        self.assertEqual(best.spec.label, "sample_hi")

    def test_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            cp.select_best([])


class GenerateCandidatesTests(unittest.TestCase):
    def test_generates_n_and_scores(self) -> None:
        scores = iter([2.3, 1.0, -0.5, 0.4])

        def fake_eval(**kwargs):
            return _report(next(scores), True)

        with patch.object(cp.tts, "generate_voice", return_value=(np.zeros(8, dtype=np.float32), 24000)) as gen, \
             patch.object(cp.quality, "evaluate_candidate", side_effect=fake_eval):
            cands = cp.generate_greeting_candidates(
                text="Hi, Dennis.",
                voice_prompt=[object()],
                reference_embedding=np.ones(4, dtype=np.float32),
                target_text="Hi, Dennis.",
                n=4,
                do_asr=False,
            )
        self.assertEqual(len(cands), 4)
        self.assertEqual(gen.call_count, 4)
        self.assertEqual(cands[0].score, 2.3)
        # best selection over these scores -> 2.3 (all marked passed here)
        self.assertEqual(cp.select_best(cands).score, 2.3)

    def test_specs_first_is_greedy(self) -> None:
        specs = cp._build_specs(4)
        self.assertEqual(specs[0].label, "greedy")
        self.assertFalse(specs[0].generate_config["do_sample"])
        self.assertTrue(specs[1].generate_config["do_sample"])

    def test_build_specs_extends_beyond_base(self) -> None:
        specs = cp._build_specs(6)
        self.assertEqual(len(specs), 6)


if __name__ == "__main__":
    unittest.main()
