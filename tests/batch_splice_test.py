import itertools
import json
import os
from pathlib import Path

import requests

BASE_URL = "http://108.179.129.245:57489"
SUPPORT_ID = "63180"
VOICE_ID = "ab196dbb-3ef0-4ee4-ae15-1c4701b203c4"

GREETING = "Hi Lloyd,"
BODY = (
    "This is Alex, Director of Customer Support from Rixtrema. "
    "This might be the most unusual voicemail you get today. "
)

PAUSES_MS = [120, 160, 200, 240]
CROSSFADE_MS = [10, 20, 30]

OUT_DIR = Path("splice-tests")
TIMEOUT_SEC = 999


def ensure_ok(response: requests.Response) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError:
        print("---- HTTP ERROR ----")
        print("Status:", response.status_code)
        print("Headers:", dict(response.headers))
        print("Body:", response.text[:2000])
        raise


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Health check
    r = requests.get(f"{BASE_URL}/health", timeout=30)
    ensure_ok(r)
    print("Health:", r.text)

    # Optional profile check
    r = requests.get(
        f"{BASE_URL}/profiles/{VOICE_ID}",
        params={"support_id": SUPPORT_ID},
        timeout=30,
    )
    if r.status_code == 200:
        data = r.json()
        print("Profile status:", data.get("status"))
    else:
        print("Profile check failed:", r.status_code, r.text[:500])

    total = len(PAUSES_MS) * len(CROSSFADE_MS)
    idx = 0

    for pause_ms, crossfade_ms in itertools.product(PAUSES_MS, CROSSFADE_MS):
        idx += 1
        payload = {
            "support_id": SUPPORT_ID,
            "voice_id": VOICE_ID,
            "greeting": GREETING,
            "body": BODY,
            "pause_ms": pause_ms,
            "crossfade_ms": crossfade_ms,
        }

        out_file = OUT_DIR / f"splice_p{pause_ms}_c{crossfade_ms}.wav"
        print(f"[{idx}/{total}] Request -> pause={pause_ms}, crossfade={crossfade_ms}")

        try:
            with requests.post(
                f"{BASE_URL}/phrases/splice-test",
                json=payload,
                timeout=TIMEOUT_SEC,
                stream=True,
            ) as r:
                ensure_ok(r)
                with open(out_file, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 64):
                        if chunk:
                            f.write(chunk)
            print(f"Saved: {out_file}")
        except Exception as e:
            print(f"FAILED p={pause_ms} c={crossfade_ms}: {e}")

    print("Done.")


if __name__ == "__main__":
    main()