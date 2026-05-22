"""Unit tests for the benchmark aggregation core (pure, no server/GPU)."""

import unittest

from bench import metrics, report


def _rec(voice_tag, category, score, **flags):
    qr = {"composite_score": score, "similarity": 0.8, "similarity_passed": True}
    qr.update(flags)
    return {
        "voice_tag": voice_tag,
        "text_category": category,
        "latency_total_ms": 500,
        "latency_generate_ms": 400,
        "latency_asr_ms": 100,
        "quality_report": qr,
    }


class IsFailTests(unittest.TestCase):
    def test_clean_is_not_fail(self) -> None:
        self.assertFalse(metrics.is_fail({"composite_score": 2.3, "similarity_passed": True}))

    def test_negative_score_is_fail(self) -> None:
        self.assertTrue(metrics.is_fail({"composite_score": -0.1, "similarity_passed": True}))

    def test_hard_flag_is_fail(self) -> None:
        self.assertTrue(metrics.is_fail({"composite_score": 2.0, "asr_prefix_extra": True}))

    def test_similarity_fail(self) -> None:
        self.assertTrue(metrics.is_fail({"composite_score": 2.0, "similarity_passed": False}))

    def test_none_report_is_fail(self) -> None:
        self.assertTrue(metrics.is_fail(None))


class AggregateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            _rec("clean_male", "normal", 2.3),
            _rec("clean_male", "normal", 2.1),
            _rec("clean_male", "stress", -0.2),  # fail (score)
            _rec("noisy_female", "normal", 1.0, asr_prefix_extra=True),  # fail (flag)
            _rec("noisy_female", "stress", 1.5),
        ]

    def test_overall_fail_rate(self) -> None:
        summary = metrics.aggregate(self.records)
        self.assertEqual(summary["total_generations"], 5)
        self.assertEqual(summary["overall_fail_count"], 2)
        self.assertEqual(summary["overall_fail_rate"], 0.4)

    def test_per_category_and_voice(self) -> None:
        summary = metrics.aggregate(self.records)
        self.assertEqual(summary["per_category_fail_rate"]["normal"], round(1 / 3, 4))
        self.assertEqual(summary["per_category_fail_rate"]["stress"], 0.5)
        self.assertEqual(summary["per_voice_fail_rate"]["clean_male"], round(1 / 3, 4))
        self.assertEqual(summary["per_voice_fail_rate"]["noisy_female"], 0.5)

    def test_artifact_breakdown(self) -> None:
        summary = metrics.aggregate(self.records)
        self.assertEqual(summary["artifact_breakdown"]["asr_prefix_extra"], 1)
        self.assertEqual(summary["artifact_breakdown"]["onset_artifact"], 0)

    def test_distributions_present(self) -> None:
        summary = metrics.aggregate(self.records)
        self.assertIn("median", summary["composite_score_distribution"])
        self.assertIsNotNone(summary["latency_total_ms"]["median"])

    def test_worst_cases_sorted(self) -> None:
        worst = metrics.worst_cases(self.records, n=2)
        self.assertEqual(worst[0]["quality_report"]["composite_score"], -0.2)


class ReportTests(unittest.TestCase):
    def test_render_markdown_runs(self) -> None:
        records = [_rec("v1", "normal", 2.3), _rec("v1", "stress", -0.5)]
        summary = metrics.aggregate(records)
        summary["run_id"] = "test_run"
        summary["worst_cases"] = metrics.worst_cases(records, n=2)
        md = report.render_markdown(summary)
        self.assertIn("QwenTTS Quality Benchmark", md)
        self.assertIn("overall_fail_rate", md.lower().replace(" ", "_"))


if __name__ == "__main__":
    unittest.main()
