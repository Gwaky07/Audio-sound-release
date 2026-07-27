#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audio_sound.console import configure_utf8_stdio
from audio_sound.pair_evaluation import evaluate_audio_pair


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare a source/processed audio pair with format, swallow, spectral, and gain gates."
    )
    parser.add_argument("--source", required=True, help="Original source audio/video path.")
    parser.add_argument("--processed", required=True, help="Processed candidate audio path.")
    parser.add_argument("--output", help="Optional JSON report path under scratch/ or output/.")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--ffprobe-bin", default="ffprobe")
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_utf8_stdio()
    args = build_parser().parse_args(argv)
    report = evaluate_audio_pair(
        source=Path(args.source),
        processed=Path(args.processed),
        ffmpeg_bin=args.ffmpeg_bin,
        ffprobe_bin=args.ffprobe_bin,
    )
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        print(output_path)
    else:
        print(text)
    return 1 if report["quality_guard"].get("release_blocked") else 0


if __name__ == "__main__":
    raise SystemExit(main())
