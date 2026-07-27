from __future__ import annotations

import argparse
import json
import shutil
import wave
from array import array
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .console import configure_utf8_stdio
from .pipeline import (
    NoiseWindow,
    _analysis_samples,
    detect_silence_candidates,
    duck_audio_file_in_place,
    infer_breath_onset_windows,
    infer_breath_windows_from_silence_edges,
    merge_noise_windows,
)


@dataclass(frozen=True)
class NarrowConfig:
    low_threshold_db: float = -50.0
    high_threshold_db: float = -36.0
    silence_min_duration: float = 0.12
    min_breath_ms: int = 60
    max_breath_ms: int = 180
    pre_roll_ms: int = 10
    fade_ms: int = 8
    floor_gain: float = 0.12
    analysis_hop_ms: int = 12
    analysis_scan_ms: int = 180
    noise_floor_ms: int = 100
    speech_start_ratio: float = 0.88
    breath_over_noise_ratio: float = 3.1
    speech_over_breath_ratio: float = 2.6
    speech_confirm_frames: int = 4
    narrow_max_ms: int = 90
    narrow_min_ms: int = 45
    max_window_end_after_silence_ms: int = 160
    max_gap_merge_ms: int = 8
    min_preceding_silence_ms: int = 160
    cleanup_pre_silence: bool = True
    pre_silence_keep_tail_ms: int = 55
    pre_silence_min_duration_ms: int = 220
    pre_silence_floor_gain: float = 0.0
    pre_silence_fade_ms: int = 0


def _load_samples(audio_path: Path) -> tuple[array, int, int]:
    with wave.open(str(audio_path), "rb") as reader:
        params = reader.getparams()
        if params.sampwidth != 2 or params.nchannels <= 0:
            raise ValueError("Narrow onset cleanup expects 16-bit WAV audio")
        raw_frames = reader.readframes(params.nframes)
    samples = array("h")
    samples.frombytes(raw_frames)
    return _analysis_samples(params, samples), params.framerate, params.nchannels


def _build_detection_windows(
    audio_path: Path,
    *,
    ffmpeg_bin: str,
    config: NarrowConfig,
) -> tuple[list[dict[str, float]], list[dict[str, float]], list[NoiseWindow]]:
    low_threshold_silences = detect_silence_candidates(
        audio_path,
        ffmpeg_bin=ffmpeg_bin,
        threshold_db=config.low_threshold_db,
        min_duration=config.silence_min_duration,
    )
    high_threshold_silences = detect_silence_candidates(
        audio_path,
        ffmpeg_bin=ffmpeg_bin,
        threshold_db=config.high_threshold_db,
        min_duration=config.silence_min_duration,
    )

    seed_windows = infer_breath_onset_windows(
        low_threshold_silences,
        high_threshold_silences,
        min_breath_seconds=config.min_breath_ms / 1000.0,
        max_breath_seconds=config.max_breath_ms / 1000.0,
        pre_roll_seconds=config.pre_roll_ms / 1000.0,
    )

    samples, sample_rate, _ = _load_samples(audio_path)
    inferred_windows = infer_breath_windows_from_silence_edges(
        samples,
        sample_rate=sample_rate,
        silences=low_threshold_silences,
        config={
            "analysis_hop_ms": config.analysis_hop_ms,
            "analysis_scan_ms": config.analysis_scan_ms,
            "noise_floor_ms": config.noise_floor_ms,
            "min_breath_ms": config.min_breath_ms,
            "max_breath_ms": config.max_breath_ms,
            "speech_start_ratio": config.speech_start_ratio,
            "breath_over_noise_ratio": config.breath_over_noise_ratio,
            "speech_over_breath_ratio": config.speech_over_breath_ratio,
            "speech_confirm_frames": config.speech_confirm_frames,
            "pre_roll_ms": config.pre_roll_ms,
        },
    )

    merged = merge_noise_windows(
        [*seed_windows, *inferred_windows],
        max_gap_seconds=config.max_gap_merge_ms / 1000.0,
    )
    return low_threshold_silences, high_threshold_silences, merged


def _find_preceding_silence_end(
    window: NoiseWindow,
    silences: list[dict[str, float]],
) -> dict[str, float] | None:
    matches = [
        item
        for item in silences
        if float(item["end_seconds"]) <= window.start_seconds + 0.05
    ]
    if not matches:
        return None
    return max(matches, key=lambda item: float(item["end_seconds"]))


def _narrow_windows(
    windows: list[NoiseWindow],
    *,
    low_threshold_silences: list[dict[str, float]],
    config: NarrowConfig,
) -> tuple[list[NoiseWindow], list[dict[str, Any]]]:
    refined: list[NoiseWindow] = []
    report_items: list[dict[str, Any]] = []

    for window in windows:
        duration_ms = (window.end_seconds - window.start_seconds) * 1000.0
        if duration_ms < config.narrow_min_ms:
            continue

        preceding = _find_preceding_silence_end(window, low_threshold_silences)
        if preceding is None:
            continue

        silence_end = float(preceding["end_seconds"])
        preceding_silence_ms = float(preceding["duration_seconds"]) * 1000.0
        if preceding_silence_ms < config.min_preceding_silence_ms:
            continue
        onset_span_ms = (window.end_seconds - silence_end) * 1000.0
        if onset_span_ms <= 0 or onset_span_ms > config.max_window_end_after_silence_ms:
            continue

        narrow_span_ms = min(config.narrow_max_ms, max(config.narrow_min_ms, onset_span_ms))
        narrowed = NoiseWindow(
            start_seconds=max(window.start_seconds, window.end_seconds - narrow_span_ms / 1000.0),
            end_seconds=window.end_seconds,
        )
        refined.append(narrowed)
        report_items.append(
            {
                "source_start_seconds": round(window.start_seconds, 3),
                "source_end_seconds": round(window.end_seconds, 3),
                "narrow_start_seconds": round(narrowed.start_seconds, 3),
                "narrow_end_seconds": round(narrowed.end_seconds, 3),
                "source_duration_ms": round(duration_ms, 1),
                "narrow_duration_ms": round((narrowed.end_seconds - narrowed.start_seconds) * 1000.0, 1),
                "preceding_silence_end_seconds": round(silence_end, 3),
                "preceding_silence_duration_seconds": round(float(preceding["duration_seconds"]), 3),
                "preceding_silence_duration_ms": round(preceding_silence_ms, 1),
                "distance_from_silence_end_ms": round((narrowed.start_seconds - silence_end) * 1000.0, 1),
            }
        )

    narrowed_windows = merge_noise_windows(refined, max_gap_seconds=config.max_gap_merge_ms / 1000.0)
    return narrowed_windows, report_items


def _pre_silence_windows(
    samples: array,
    sample_rate: int,
    *,
    low_threshold_silences: list[dict[str, float]],
    config: NarrowConfig,
) -> tuple[list[NoiseWindow], list[dict[str, Any]]]:
    if not config.cleanup_pre_silence:
        return [], []

    refined: list[NoiseWindow] = []
    report_items: list[dict[str, Any]] = []
    hop_samples = max(1, int(sample_rate * (config.analysis_hop_ms / 1000.0)))
    scan_samples = max(hop_samples, int(sample_rate * (config.analysis_scan_ms / 1000.0)))

    for preceding in low_threshold_silences:
        silence_start = float(preceding["start_seconds"])
        silence_end = float(preceding["end_seconds"])
        silence_duration_ms = float(preceding["duration_seconds"]) * 1000.0
        if silence_duration_ms < config.pre_silence_min_duration_ms:
            continue

        silence_end_index = min(len(samples), max(0, int(silence_end * sample_rate)))
        scan_end_index = min(len(samples), silence_end_index + scan_samples)
        if scan_end_index <= silence_end_index:
            continue

        frame_rms: list[float] = []
        index = silence_end_index
        while index + hop_samples <= scan_end_index:
            frame = samples[index : index + hop_samples]
            frame_rms.append((sum(float(value) * float(value) for value in frame) / len(frame)) ** 0.5)
            index += hop_samples
        if not frame_rms:
            continue

        peak_rms = max(frame_rms)
        if peak_rms <= 0:
            continue

        threshold = peak_rms * config.speech_start_ratio
        onset_index: int | None = None
        for idx in range(len(frame_rms)):
            confirm_slice = frame_rms[idx : idx + config.speech_confirm_frames]
            if len(confirm_slice) < config.speech_confirm_frames:
                break
            if sum(confirm_slice) / len(confirm_slice) >= threshold:
                onset_index = idx
                break

        if onset_index is None:
            continue

        onset_offset_ms = onset_index * config.analysis_hop_ms
        if onset_offset_ms > config.max_window_end_after_silence_ms:
            continue

        keep_tail_seconds = max(0.0, config.pre_silence_keep_tail_ms / 1000.0)
        cleanup_end = max(silence_start, silence_end - keep_tail_seconds)
        if cleanup_end <= silence_start:
            continue

        refined_window = NoiseWindow(
            start_seconds=silence_start,
            end_seconds=cleanup_end,
        )
        refined.append(refined_window)
        report_items.append(
            {
                "silence_start_seconds": round(silence_start, 3),
                "silence_end_seconds": round(silence_end, 3),
                "silence_duration_ms": round(silence_duration_ms, 1),
                "cleanup_start_seconds": round(refined_window.start_seconds, 3),
                "cleanup_end_seconds": round(refined_window.end_seconds, 3),
                "cleanup_duration_ms": round(
                    (refined_window.end_seconds - refined_window.start_seconds) * 1000.0,
                    1,
                ),
                "kept_tail_ms": round((silence_end - refined_window.end_seconds) * 1000.0, 1),
                "speech_onset_after_silence_ms": round(onset_offset_ms, 1),
            }
        )

    cleanup_windows = merge_noise_windows(refined, max_gap_seconds=config.max_gap_merge_ms / 1000.0)
    return cleanup_windows, report_items


def _write_report(
    report_path: Path,
    *,
    input_path: Path,
    output_path: Path,
    config: NarrowConfig,
    low_silences: list[dict[str, float]],
    high_silences: list[dict[str, float]],
    raw_windows: list[NoiseWindow],
    narrow_windows: list[NoiseWindow],
    refined_items: list[dict[str, Any]],
    pre_silence_windows: list[NoiseWindow],
    pre_silence_items: list[dict[str, Any]],
) -> None:
    payload = {
        "input": str(input_path),
        "output": str(output_path),
        "config": asdict(config),
        "low_threshold_silence_count": len(low_silences),
        "high_threshold_silence_count": len(high_silences),
        "raw_window_count": len(raw_windows),
        "narrow_window_count": len(narrow_windows),
        "pre_silence_window_count": len(pre_silence_windows),
        "raw_windows": [
            {
                "start_seconds": round(window.start_seconds, 3),
                "end_seconds": round(window.end_seconds, 3),
                "duration_ms": round((window.end_seconds - window.start_seconds) * 1000.0, 1),
            }
            for window in raw_windows
        ],
        "narrow_windows": [
            {
                "start_seconds": round(window.start_seconds, 3),
                "end_seconds": round(window.end_seconds, 3),
                "duration_ms": round((window.end_seconds - window.start_seconds) * 1000.0, 1),
            }
            for window in narrow_windows
        ],
        "refined_items": refined_items,
        "pre_silence_windows": [
            {
                "start_seconds": round(window.start_seconds, 3),
                "end_seconds": round(window.end_seconds, 3),
                "duration_ms": round((window.end_seconds - window.start_seconds) * 1000.0, 1),
            }
            for window in pre_silence_windows
        ],
        "pre_silence_items": pre_silence_items,
    }
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply one narrow pre-speech onset cleanup pass to an existing mono WAV."
    )
    parser.add_argument("input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--low-threshold-db", type=float, default=NarrowConfig.low_threshold_db)
    parser.add_argument("--high-threshold-db", type=float, default=NarrowConfig.high_threshold_db)
    parser.add_argument("--silence-min-duration", type=float, default=NarrowConfig.silence_min_duration)
    parser.add_argument("--min-breath-ms", type=int, default=NarrowConfig.min_breath_ms)
    parser.add_argument("--max-breath-ms", type=int, default=NarrowConfig.max_breath_ms)
    parser.add_argument("--pre-roll-ms", type=int, default=NarrowConfig.pre_roll_ms)
    parser.add_argument("--fade-ms", type=int, default=NarrowConfig.fade_ms)
    parser.add_argument("--floor-gain", type=float, default=NarrowConfig.floor_gain)
    parser.add_argument("--analysis-hop-ms", type=int, default=NarrowConfig.analysis_hop_ms)
    parser.add_argument("--analysis-scan-ms", type=int, default=NarrowConfig.analysis_scan_ms)
    parser.add_argument("--noise-floor-ms", type=int, default=NarrowConfig.noise_floor_ms)
    parser.add_argument("--speech-start-ratio", type=float, default=NarrowConfig.speech_start_ratio)
    parser.add_argument("--breath-over-noise-ratio", type=float, default=NarrowConfig.breath_over_noise_ratio)
    parser.add_argument("--speech-over-breath-ratio", type=float, default=NarrowConfig.speech_over_breath_ratio)
    parser.add_argument("--speech-confirm-frames", type=int, default=NarrowConfig.speech_confirm_frames)
    parser.add_argument("--narrow-max-ms", type=int, default=NarrowConfig.narrow_max_ms)
    parser.add_argument("--narrow-min-ms", type=int, default=NarrowConfig.narrow_min_ms)
    parser.add_argument(
        "--max-window-end-after-silence-ms",
        type=int,
        default=NarrowConfig.max_window_end_after_silence_ms,
    )
    parser.add_argument("--max-gap-merge-ms", type=int, default=NarrowConfig.max_gap_merge_ms)
    parser.add_argument("--min-preceding-silence-ms", type=int, default=NarrowConfig.min_preceding_silence_ms)
    parser.add_argument("--cleanup-pre-silence", action=argparse.BooleanOptionalAction, default=NarrowConfig.cleanup_pre_silence)
    parser.add_argument("--pre-silence-keep-tail-ms", type=int, default=NarrowConfig.pre_silence_keep_tail_ms)
    parser.add_argument("--pre-silence-min-duration-ms", type=int, default=NarrowConfig.pre_silence_min_duration_ms)
    parser.add_argument("--pre-silence-floor-gain", type=float, default=NarrowConfig.pre_silence_floor_gain)
    parser.add_argument("--pre-silence-fade-ms", type=int, default=NarrowConfig.pre_silence_fade_ms)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    configure_utf8_stdio()
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    report_path = (
        Path(args.report).expanduser().resolve()
        if args.report
        else output_path.with_suffix(output_path.suffix + ".report.json")
    )

    config = NarrowConfig(
        low_threshold_db=args.low_threshold_db,
        high_threshold_db=args.high_threshold_db,
        silence_min_duration=args.silence_min_duration,
        min_breath_ms=args.min_breath_ms,
        max_breath_ms=args.max_breath_ms,
        pre_roll_ms=args.pre_roll_ms,
        fade_ms=args.fade_ms,
        floor_gain=args.floor_gain,
        analysis_hop_ms=args.analysis_hop_ms,
        analysis_scan_ms=args.analysis_scan_ms,
        noise_floor_ms=args.noise_floor_ms,
        speech_start_ratio=args.speech_start_ratio,
        breath_over_noise_ratio=args.breath_over_noise_ratio,
        speech_over_breath_ratio=args.speech_over_breath_ratio,
        speech_confirm_frames=args.speech_confirm_frames,
        narrow_max_ms=args.narrow_max_ms,
        narrow_min_ms=args.narrow_min_ms,
        max_window_end_after_silence_ms=args.max_window_end_after_silence_ms,
        max_gap_merge_ms=args.max_gap_merge_ms,
        min_preceding_silence_ms=args.min_preceding_silence_ms,
        cleanup_pre_silence=args.cleanup_pre_silence,
        pre_silence_keep_tail_ms=args.pre_silence_keep_tail_ms,
        pre_silence_min_duration_ms=args.pre_silence_min_duration_ms,
        pre_silence_floor_gain=args.pre_silence_floor_gain,
        pre_silence_fade_ms=args.pre_silence_fade_ms,
    )

    low_silences, high_silences, raw_windows = _build_detection_windows(
        input_path,
        ffmpeg_bin=args.ffmpeg_bin,
        config=config,
    )
    samples, sample_rate, _ = _load_samples(input_path)
    narrow_windows, refined_items = _narrow_windows(
        raw_windows,
        low_threshold_silences=low_silences,
        config=config,
    )
    pre_silence_windows, pre_silence_items = _pre_silence_windows(
        samples,
        sample_rate,
        low_threshold_silences=low_silences,
        config=config,
    )

    if args.dry_run:
        _write_report(
            report_path,
            input_path=input_path,
            output_path=output_path,
            config=config,
            low_silences=low_silences,
            high_silences=high_silences,
            raw_windows=raw_windows,
            narrow_windows=narrow_windows,
            refined_items=refined_items,
            pre_silence_windows=pre_silence_windows,
            pre_silence_items=pre_silence_items,
        )
        print(f"Dry run report: {report_path}")
        print(f"Raw windows: {len(raw_windows)}")
        print(f"Narrow windows: {len(narrow_windows)}")
        print(f"Pre-silence windows: {len(pre_silence_windows)}")
        return 0

    shutil.copy2(input_path, output_path)
    if pre_silence_windows:
        duck_audio_file_in_place(
            output_path,
            windows=pre_silence_windows,
            floor_gain=config.pre_silence_floor_gain,
            fade_ms=config.pre_silence_fade_ms,
        )
    if narrow_windows:
        duck_audio_file_in_place(
            output_path,
            windows=narrow_windows,
            floor_gain=config.floor_gain,
            fade_ms=config.fade_ms,
        )

    _write_report(
        report_path,
        input_path=input_path,
        output_path=output_path,
        config=config,
        low_silences=low_silences,
        high_silences=high_silences,
        raw_windows=raw_windows,
        narrow_windows=narrow_windows,
        refined_items=refined_items,
        pre_silence_windows=pre_silence_windows,
        pre_silence_items=pre_silence_items,
    )

    print(f"Output: {output_path}")
    print(f"Report: {report_path}")
    print(f"Raw windows: {len(raw_windows)}")
    print(f"Narrow windows: {len(narrow_windows)}")
    print(f"Pre-silence windows: {len(pre_silence_windows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
