"""Render a benchmark summary.json into human-readable Markdown + HTML."""

import json
import os
from typing import Any, Dict


def _dist_line(name: str, dist: Dict[str, Any]) -> str:
    return (
        f"| {name} | {dist.get('median')} | {dist.get('p10')} | "
        f"{dist.get('p90')} | {dist.get('p95')} |"
    )


def render_markdown(summary: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"# QwenTTS Quality Benchmark - {summary.get('run_id', '')}")
    lines.append("")
    lines.append(f"- Total generations: **{summary.get('total_generations')}**")
    lines.append(f"- Overall fail rate: **{summary.get('overall_fail_rate')}** "
                 f"({summary.get('overall_fail_count')} failures)")
    lines.append("")

    lines.append("## Fail rate by text category")
    lines.append("")
    lines.append("| Category | Fail rate |")
    lines.append("|---|---|")
    for cat, rate in (summary.get("per_category_fail_rate") or {}).items():
        lines.append(f"| {cat} | {rate} |")
    lines.append("")

    lines.append("## Fail rate by voice")
    lines.append("")
    lines.append("| Voice | Fail rate |")
    lines.append("|---|---|")
    for voice, rate in (summary.get("per_voice_fail_rate") or {}).items():
        lines.append(f"| {voice} | {rate} |")
    lines.append("")

    lines.append("## Artifact breakdown (count)")
    lines.append("")
    lines.append("| Artifact | Count |")
    lines.append("|---|---|")
    for flag, count in (summary.get("artifact_breakdown") or {}).items():
        lines.append(f"| {flag} | {count} |")
    lines.append("")

    lines.append("## Distributions")
    lines.append("")
    lines.append("| Metric | median | p10 | p90 | p95 |")
    lines.append("|---|---|---|---|---|")
    lines.append(_dist_line("similarity", summary.get("similarity_distribution") or {}))
    lines.append(_dist_line("asr_wer", summary.get("wer_distribution") or {}))
    lines.append(_dist_line("composite_score", summary.get("composite_score_distribution") or {}))
    lines.append(_dist_line("latency_total_ms", summary.get("latency_total_ms") or {}))
    lines.append(_dist_line("latency_generate_ms", summary.get("latency_generate_ms") or {}))
    lines.append(_dist_line("latency_asr_ms", summary.get("latency_asr_ms") or {}))
    lines.append("")

    worst = summary.get("worst_cases") or []
    if worst:
        lines.append("## Top worst cases (lowest composite score) - for manual QA")
        lines.append("")
        lines.append("| Voice | Text | Score | Audio |")
        lines.append("|---|---|---|---|")
        for rec in worst:
            qr = rec.get("quality_report") or {}
            text = (rec.get("text") or "")[:60].replace("|", "/")
            lines.append(
                f"| {rec.get('voice_tag')} | {text} | "
                f"{qr.get('composite_score')} | {rec.get('audio_path')} |"
            )
        lines.append("")

    return "\n".join(lines)


def write_report(summary: Dict[str, Any], output_dir: str) -> None:
    md = render_markdown(summary)
    with open(os.path.join(output_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(md)
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>QwenTTS Benchmark</title></head><body><pre>"
        + md.replace("&", "&amp;").replace("<", "&lt;")
        + "</pre></body></html>"
    )
    with open(os.path.join(output_dir, "report.html"), "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    import sys

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        print(render_markdown(json.load(f)))
