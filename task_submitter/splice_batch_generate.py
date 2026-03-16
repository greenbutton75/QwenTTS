import argparse
from pathlib import Path
from typing import List

import requests


DEFAULT_GREETINGS = [
    "Hi Lloyd,",
    "Hi Emma,",
    "Hi Michael,",
    "Hi Sophia,",
    "Hi Daniel,",
    "Hi Olivia,",
    "Hi James,",
    "Hi Isabella,",
    "Hi Benjamin,",
    "Hi Charlotte,",
]


def _read_greetings(path: str) -> List[str]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text:
            rows.append(text)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-generate splice TTS with one body and many greetings.")
    parser.add_argument("--base-url", required=True, help="Example: http://108.179.129.245:57489")
    parser.add_argument("--support-id", required=True)
    parser.add_argument("--voice-id", required=True)
    parser.add_argument("--body", required=True)
    parser.add_argument("--pause-ms", type=int, default=120)
    parser.add_argument("--crossfade-ms", type=int, default=10)
    parser.add_argument("--mode", choices=["wav_splice", "latent_concat"], default="wav_splice")
    parser.add_argument("--target-lufs", type=float, default=-16.0)
    parser.add_argument("--content-aware", dest="content_aware", action="store_true")
    parser.add_argument("--no-content-aware", dest="content_aware", action="store_false")
    parser.set_defaults(content_aware=True)
    parser.add_argument("--out-dir", default="splice-tests")
    parser.add_argument("--greetings-file", default="", help="Optional text file: one greeting per line.")
    parser.add_argument("--timeout-sec", type=int, default=900)
    args = parser.parse_args()

    greetings = DEFAULT_GREETINGS
    if args.greetings_file:
        greetings = _read_greetings(args.greetings_file)
    if not greetings:
        raise RuntimeError("No greetings to process.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    health = requests.get(f"{args.base_url}/health", timeout=30)
    health.raise_for_status()
    print(f"health={health.text}")

    for idx, greeting in enumerate(greetings, start=1):
        payload = {
            "support_id": args.support_id,
            "voice_id": args.voice_id,
            "greeting": greeting,
            "body": args.body,
            "pause_ms": args.pause_ms,
            "crossfade_ms": args.crossfade_ms,
            "content_aware": bool(args.content_aware),
            "target_lufs": float(args.target_lufs),
            "mode": args.mode,
        }

        safe_name = "".join(ch if ch.isalnum() else "_" for ch in greeting).strip("_").lower()
        out_path = out_dir / f"{idx:02d}_{safe_name[:60]}.wav"

        with requests.post(
            f"{args.base_url}/phrases/splice-test",
            json=payload,
            stream=True,
            timeout=args.timeout_sec,
        ) as resp:
            resp.raise_for_status()
            with out_path.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
            cache = resp.headers.get("X-Body-Cache", "unknown")
            cache_key = resp.headers.get("X-Body-Cache-Key", "")
            mode = resp.headers.get("X-Mode", "?")
            strategy = resp.headers.get("X-Splice-Strategy", "?")
            print(
                f"[{idx}/{len(greetings)}] saved={out_path} "
                f"cache={cache} key={cache_key[:12]} mode={mode} strategy={strategy}"
            )


if __name__ == "__main__":
    main()
