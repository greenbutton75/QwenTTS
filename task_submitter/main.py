import argparse

from .s3_upload import upload_sample
from .task_api import create_task


def cmd_create_profile(args: argparse.Namespace) -> None:
    key = upload_sample(args.support_id, args.voice_id, args.sample)
    task_id = create_task(
        "QWEN_TTS_CREATE_PROFILE",
        None,
        {
            "support_id": args.support_id,
            "voice_id": args.voice_id,
            "voice_name": args.voice_name,
            "ref_text": args.ref_text,
        },
    )
    print(f"Uploaded sample to s3://{key}")
    print(f"Created task: {task_id}")


def cmd_create_phrase(args: argparse.Namespace) -> None:
    task_id = create_task(
        "QWEN_TTS_PHRASE",
        None,
        {
            "support_id": args.support_id,
            "voice_id": args.voice_id,
            "phrase_id": args.phrase_id,
            "text": args.text,
        },
    )
    print(f"Created task: {task_id}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="task_submitter")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("create-profile", help="Upload sample and create profile task.")
    p1.add_argument("--support-id", required=True)
    p1.add_argument("--voice-id", required=True)
    p1.add_argument("--voice-name", required=True)
    p1.add_argument("--ref-text", required=True)
    p1.add_argument("--sample", required=True, help="Path to mp3/wav/m4a sample file.")
    p1.set_defaults(func=cmd_create_profile)

    p2 = sub.add_parser("create-phrase", help="Create phrase task.")
    p2.add_argument("--support-id", required=True)
    p2.add_argument("--voice-id", required=True)
    p2.add_argument("--phrase-id", required=True)
    p2.add_argument("--text", required=True)
    p2.set_defaults(func=cmd_create_phrase)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
