"""Compare two benchmark runs and print the metric deltas.

Usage:
    python -m bench.compare bench/results/baseline_v1 bench/results/phase2_v1
"""

import json
import os
import sys
from typing import Any, Dict, Optional


def _load_summary(run_dir: str) -> Dict[str, Any]:
    with open(os.path.join(run_dir, "summary.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def _delta(before: Optional[float], after: Optional[float]) -> str:
    if before is None or after is None:
        return "n/a"
    diff = after - before
    rel = f" ({diff / before * 100:+.1f}%)" if before else ""
    return f"{before} -> {after} ({diff:+.4f}){rel}"


def compare(before: Dict[str, Any], after: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"# Benchmark comparison: {before.get('run_id')} -> {after.get('run_id')}")
    lines.append("")
    lines.append(f"overall_fail_rate: {_delta(before.get('overall_fail_rate'), after.get('overall_fail_rate'))}")
    lines.append("")

    lines.append("## Artifact breakdown deltas")
    b_art = before.get("artifact_breakdown") or {}
    a_art = after.get("artifact_breakdown") or {}
    for flag in sorted(set(b_art) | set(a_art)):
        lines.append(f"- {flag}: {_delta(b_art.get(flag, 0), a_art.get(flag, 0))}")
    lines.append("")

    lines.append("## Distribution medians")
    for metric in ("similarity_distribution", "wer_distribution", "composite_score_distribution"):
        b_med = (before.get(metric) or {}).get("median")
        a_med = (after.get(metric) or {}).get("median")
        lines.append(f"- {metric}.median: {_delta(b_med, a_med)}")
    lines.append("")

    lines.append("## Per-category fail rate")
    b_cat = before.get("per_category_fail_rate") or {}
    a_cat = after.get("per_category_fail_rate") or {}
    for cat in sorted(set(b_cat) | set(a_cat)):
        lines.append(f"- {cat}: {_delta(b_cat.get(cat), a_cat.get(cat))}")

    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python -m bench.compare <before_dir> <after_dir>")
        raise SystemExit(2)
    print(compare(_load_summary(sys.argv[1]), _load_summary(sys.argv[2])))
