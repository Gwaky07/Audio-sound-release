from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audio_sound.pipeline import NoiseWindow, duck_audio_file_in_place  # noqa: E402


@dataclass(frozen=True)
class DuckWindowSpec:
    start_seconds: float
    end_seconds: float
    floor_gain: float
    fade_ms: int


def _parse_time_window(value: str) -> NoiseWindow:
    parts = [item.strip() for item in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("window spec must be start,end")
    start_seconds = float(parts[0])
    end_seconds = float(parts[1])
    if end_seconds <= start_seconds:
        raise argparse.ArgumentTypeError("window end must be greater than start")
    return NoiseWindow(start_seconds=start_seconds, end_seconds=end_seconds)


def _parse_duck_window(value: str) -> DuckWindowSpec:
    parts = [item.strip() for item in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("duck spec must be start,end,gain,fade_ms")
    start_seconds = float(parts[0])
    end_seconds = float(parts[1])
    floor_gain = float(parts[2])
    fade_ms = int(parts[3])
    if end_seconds <= start_seconds:
        raise argparse.ArgumentTypeError("duck window end must be greater than start")
    if not 0.0 <= floor_gain <= 1.0:
        raise argparse.ArgumentTypeError("duck window gain must be between 0 and 1")
    if fade_ms < 0:
        raise argparse.ArgumentTypeError("duck window fade_ms must be >= 0")
    return DuckWindowSpec(
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        floor_gain=floor_gain,
        fade_ms=fade_ms,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply exact mute/duck windows to an existing mono WAV."
    )
    parser.add_argument("input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report")
    parser.add_argument(
        "--mute-window",
        action="append",
        default=[],
        type=_parse_time_window,
        help="Exact hard-mute window in start,end seconds. Can be repeated.",
    )
    parser.add_argument(
        "--duck-window",
        action="append",
        default=[],
        type=_parse_duck_window,
        help="Exact duck window in start,end,gain,fade_ms. Can be repeated.",
    )
    return parser.parse_args()


def _write_report(
    report_path: Path,
    *,
    input_path: Path,
    output_path: Path,
    mute_windows: list[NoiseWindow],
    duck_windows: list[DuckWindowSpec],
) -> None:
    payload = {
        "input": str(input_path),
        "output": str(output_path),
        "mute_window_count": len(mute_windows),
        "duck_window_count": len(duck_windows),
        "mute_windows": [asdict(window) for window in mute_windows],
        "duck_windows": [asdict(window) for window in duck_windows],
    }
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    report_path = (
        Path(args.report).expanduser().resolve()
        if args.report
        else output_path.with_suffix(output_path.suffix + ".report.json")
    )

    mute_windows: list[NoiseWindow] = list(args.mute_window)
    duck_windows: list[DuckWindowSpec] = list(args.duck_window)
    if not mute_windows and not duck_windows:
        raise SystemExit("No windows provided.")

    shutil.copy2(input_path, output_path)

    if mute_windows:
        duck_audio_file_in_place(
            output_path,
            windows=mute_windows,
            floor_gain=0.0,
            fade_ms=0,
        )

    for spec in duck_windows:
        duck_audio_file_in_place(
            output_path,
            windows=[NoiseWindow(start_seconds=spec.start_seconds, end_seconds=spec.end_seconds)],
            floor_gain=spec.floor_gain,
            fade_ms=spec.fade_ms,
        )

    _write_report(
        report_path,
        input_path=input_path,
        output_path=output_path,
        mute_windows=mute_windows,
        duck_windows=duck_windows,
    )
    print(f"Output: {output_path}")
    print(f"Report: {report_path}")
    print(f"Mute windows: {len(mute_windows)}")
    print(f"Duck windows: {len(duck_windows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
