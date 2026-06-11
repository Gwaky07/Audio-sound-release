from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import wave
from array import array
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from .config import PROJECT_ROOT, resolve_repo_python
from .pipeline import (
    NoiseWindow,
    detect_silence_candidates,
    duck_audio_file_in_place,
    ensure_tool,
    measure_segment_levels,
    run_command,
    utc_timestamp_slug,
)


LOUDNESS_LINE_PATTERNS = {
    "integrated_lufs": re.compile(r"Input Integrated:\s*(-?\d+(?:\.\d+)?)\s+LUFS"),
    "true_peak_dbtp": re.compile(r"Input True Peak:\s*(-?\d+(?:\.\d+)?)\s+dBTP"),
    "lra_lu": re.compile(r"Input LRA:\s*(-?\d+(?:\.\d+)?)\s+LU"),
    "threshold_lufs": re.compile(r"Input Threshold:\s*(-?\d+(?:\.\d+)?)\s+LUFS"),
}
MEAN_VOLUME_PATTERN = re.compile(r"mean_volume:\s*(-?\d+(?:\.\d+)?)\s+dB")
MAX_VOLUME_PATTERN = re.compile(r"max_volume:\s*(-?\d+(?:\.\d+)?)\s+dB")


@dataclass(frozen=True)
class FocusWindow:
    label: str
    start_seconds: float
    duration_seconds: float


@dataclass(frozen=True)
class ExactDuckWindow:
    start_seconds: float
    end_seconds: float
    floor_gain: float
    fade_ms: int


@dataclass(frozen=True)
class BridgeCleanupConfig:
    silence_threshold_db: float = -44.0
    silence_min_duration: float = 0.035
    min_seed_silence_duration: float = 0.07
    require_multi_seed_window: bool = True
    tiny_gap_merge_seconds: float = 0.055
    bridge_gap_seconds: float = 0.12
    bridge_peak_db: float = -18.0
    bridge_rms_db: float = -26.0
    protected_gap_seconds: float = 0.075
    protected_bridge_peak_db: float = -22.0
    protected_bridge_rms_db: float = -30.0
    protected_silence_max_seconds: float = 0.42
    protected_context_probe_seconds: float = 0.12
    protected_context_peak_db: float = -18.0
    protected_context_rms_db: float = -30.0
    short_trim_seconds: float = 0.006
    long_trim_seconds: float = 0.010
    trim_switch_seconds: float = 0.20
    min_window_seconds: float = 0.05
    fade_ms: float = 0.0


@dataclass(frozen=True)
class ResidueCleanupConfig:
    silence_threshold_db: float = -44.0
    silence_min_duration: float = 0.09
    min_preceding_silence_seconds: float = 0.35
    max_residue_duration_seconds: float = 0.28
    residue_peak_db: float = -20.0
    residue_rms_db: float = -34.0
    trim_start_seconds: float = 0.008
    trim_end_seconds: float = 0.012
    min_window_seconds: float = 0.05
    floor_gain: float = 0.0
    fade_ms: float = 0.0


@dataclass(frozen=True)
class HardMuteCleanupConfig:
    silence_threshold_db: float = -46.0
    silence_min_duration: float = 0.05
    trim_start_seconds: float = 0.02
    trim_end_seconds: float = 0.02
    min_window_seconds: float = 0.05
    fade_ms: float = 0.0
    protected_silence_max_seconds: float = 0.42
    protected_neighbor_min_seconds: float = 0.08
    protected_neighbor_max_seconds: float = 0.75
    protected_neighbor_peak_db: float = -18.0
    protected_neighbor_rms_db: float = -30.0


@dataclass(frozen=True)
class WorkflowMode:
    name: str
    preset_name: str
    attenuation_db: float
    target_lufs: float | None
    suffix: str
    apply_narrow_cleanup: bool
    apply_bridge_cleanup: bool
    description: str


WORKFLOW_MODES: dict[str, WorkflowMode] = {
    "reference-style": WorkflowMode(
        name="reference-style",
        preset_name="fast",
        attenuation_db=18.0,
        target_lufs=-24.7,
        suffix="技能稳定版",
        apply_narrow_cleanup=True,
        apply_bridge_cleanup=True,
        description="User-approved spoken-word reference style: breath-first cleanup, lower loudness, and tighter pause cleanup.",
    ),
    "reference-legacy": WorkflowMode(
        name="reference-legacy",
        preset_name="fast",
        attenuation_db=18.0,
        target_lufs=-24.7,
        suffix="细丝桥接清理版",
        apply_narrow_cleanup=False,
        apply_bridge_cleanup=False,
        description="Legacy approved cleanup chain: extreme pre-silence cleanup, hard mute, then aggressive bridge cleanup.",
    ),
    "final": WorkflowMode(
        name="final",
        preset_name="final",
        attenuation_db=18.0,
        target_lufs=None,
        suffix="技能交付版",
        apply_narrow_cleanup=False,
        apply_bridge_cleanup=False,
        description="Balanced final-delivery spoken-word cleanup with lighter post-silence surgery.",
    ),
    "repair-soft": WorkflowMode(
        name="repair-soft",
        preset_name="repair-soft",
        attenuation_db=18.0,
        target_lufs=None,
        suffix="技能柔和修复版",
        apply_narrow_cleanup=False,
        apply_bridge_cleanup=False,
        description="Gentler repair mode for mouth noise and breathing when hard repair clips speech edges.",
    ),
    "repair": WorkflowMode(
        name="repair",
        preset_name="repair",
        attenuation_db=18.0,
        target_lufs=None,
        suffix="技能强修复版",
        apply_narrow_cleanup=False,
        apply_bridge_cleanup=False,
        description="Stronger repair mode for obvious breathing, mouth clicks, and forward sibilance.",
    ),
    "voice-isolate": WorkflowMode(
        name="voice-isolate",
        preset_name="voice-isolate",
        attenuation_db=18.0,
        target_lufs=None,
        suffix="技能噪声隔离版",
        apply_narrow_cleanup=False,
        apply_bridge_cleanup=False,
        description="Noise-window-driven cleanup for room noise and non-voice residue from the same recording.",
    ),
    "review": WorkflowMode(
        name="review",
        preset_name="review",
        attenuation_db=18.0,
        target_lufs=None,
        suffix="技能审查版",
        apply_narrow_cleanup=False,
        apply_bridge_cleanup=False,
        description="Cleanup plus QA-oriented markers and reports for manual follow-up.",
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stable repo-local audio skill workflow with step spectrograms and loudness reporting."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the stable repository workflow.")
    run_parser.add_argument("input_path")
    run_parser.add_argument("--mode", default="reference-legacy", choices=sorted(WORKFLOW_MODES))
    run_parser.add_argument("--output-dir")
    run_parser.add_argument(
        "--delivery-dir",
        help="Directory for final user-facing audio. Defaults to output/修音成品.",
    )
    run_parser.add_argument(
        "--delivery-prefix",
        default="修音版",
        help="Prefix for final user-facing audio names.",
    )
    run_parser.add_argument(
        "--keep-intermediate-audio",
        action="store_true",
        help="Keep internal WAV/MP3 stage files for troubleshooting.",
    )
    run_parser.add_argument("--recursive", action="store_true")
    run_parser.add_argument("--target-lufs", type=float)
    run_parser.add_argument("--attenuation-db", type=float)
    run_parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    run_parser.add_argument("--ffprobe-bin", default="ffprobe")
    run_parser.add_argument("--python-executable", default=resolve_repo_python(PROJECT_ROOT))
    run_parser.add_argument(
        "--focus-window",
        action="append",
        default=[],
        help="Focused final spectrogram window formatted as label,start_seconds,duration_seconds. Repeatable.",
    )
    run_parser.add_argument(
        "--noise-window",
        action="append",
        default=[],
        help="Noise-only sample region in seconds, formatted as start:end. Useful with voice-isolate.",
    )
    run_parser.add_argument(
        "--exact-mute-window",
        action="append",
        default=[],
        help="Exact final hard-mute window formatted as start_seconds,end_seconds. Repeatable.",
    )
    run_parser.add_argument(
        "--exact-duck-window",
        action="append",
        default=[],
        help="Exact final duck window formatted as start_seconds,end_seconds,floor_gain,fade_ms. Repeatable.",
    )
    run_parser.add_argument("--skip-spectrograms", action="store_true")
    run_parser.add_argument("--skip-bridge-cleanup", action="store_true")
    run_parser.add_argument("--spectrogram-start", type=float, default=0.0)
    run_parser.add_argument("--spectrogram-duration", type=float, default=30.0)

    describe_parser = subparsers.add_parser("describe-modes", help="Print workflow modes as JSON.")
    describe_parser.add_argument("--mode", choices=sorted(WORKFLOW_MODES))
    return parser


def parse_focus_window(raw_value: str) -> FocusWindow:
    parts = [item.strip() for item in raw_value.split(",")]
    if len(parts) != 3:
        raise ValueError("focus window must be label,start_seconds,duration_seconds")
    label = parts[0]
    start_seconds = float(parts[1])
    duration_seconds = float(parts[2])
    if not label:
        raise ValueError("focus window label must be non-empty")
    if start_seconds < 0.0:
        raise ValueError("focus window start_seconds must be >= 0")
    if duration_seconds <= 0.0:
        raise ValueError("focus window duration_seconds must be > 0")
    return FocusWindow(
        label=_slug_label(label),
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
    )


def parse_exact_mute_window(raw_value: str) -> NoiseWindow:
    parts = [item.strip() for item in raw_value.split(",")]
    if len(parts) != 2:
        raise ValueError("exact mute window must be start_seconds,end_seconds")
    start_seconds = float(parts[0])
    end_seconds = float(parts[1])
    if start_seconds < 0.0:
        raise ValueError("exact mute window start_seconds must be >= 0")
    if end_seconds <= start_seconds:
        raise ValueError("exact mute window end_seconds must be > start_seconds")
    return NoiseWindow(start_seconds=start_seconds, end_seconds=end_seconds)


def parse_exact_duck_window(raw_value: str) -> ExactDuckWindow:
    parts = [item.strip() for item in raw_value.split(",")]
    if len(parts) != 4:
        raise ValueError("exact duck window must be start_seconds,end_seconds,floor_gain,fade_ms")
    start_seconds = float(parts[0])
    end_seconds = float(parts[1])
    floor_gain = float(parts[2])
    fade_ms = int(parts[3])
    if start_seconds < 0.0:
        raise ValueError("exact duck window start_seconds must be >= 0")
    if end_seconds <= start_seconds:
        raise ValueError("exact duck window end_seconds must be > start_seconds")
    if not 0.0 <= floor_gain <= 1.0:
        raise ValueError("exact duck window floor_gain must be between 0 and 1")
    if fade_ms < 0:
        raise ValueError("exact duck window fade_ms must be >= 0")
    return ExactDuckWindow(
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        floor_gain=floor_gain,
        fade_ms=fade_ms,
    )


def describe_modes(mode_name: str | None = None) -> dict[str, Any]:
    if mode_name is None:
        return {name: asdict(mode) for name, mode in WORKFLOW_MODES.items()}
    return asdict(resolve_mode(mode_name))


def resolve_mode(mode_name: str) -> WorkflowMode:
    try:
        return WORKFLOW_MODES[mode_name]
    except KeyError as exc:
        raise ValueError(f"Unknown workflow mode: {mode_name}") from exc


def measure_audio_metrics(
    audio_path: Path,
    *,
    ffmpeg_bin: str,
    measurement_target_lufs: float,
    measurement_target_tp: float = -3.0,
    measurement_target_lra: float = 7.0,
) -> dict[str, float | None]:
    ffmpeg = ensure_tool(ffmpeg_bin)
    volumedetect = run_command(
        [
            ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-i",
            str(audio_path),
            "-af",
            "volumedetect",
            "-f",
            "null",
            "-",
        ]
    )
    if volumedetect.returncode != 0:
        raise RuntimeError(volumedetect.stderr.strip() or volumedetect.stdout.strip() or "volumedetect failed")

    loudnorm = run_command(
        [
            ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-i",
            str(audio_path),
            "-af",
            (
                f"loudnorm=I={measurement_target_lufs}:"
                f"LRA={measurement_target_lra}:"
                f"TP={measurement_target_tp}:print_format=summary"
            ),
            "-f",
            "null",
            "-",
        ]
    )
    if loudnorm.returncode != 0:
        raise RuntimeError(loudnorm.stderr.strip() or loudnorm.stdout.strip() or "loudnorm measurement failed")

    metrics: dict[str, float | None] = {
        "mean_volume_db": _parse_single_float(MEAN_VOLUME_PATTERN, volumedetect.stderr),
        "max_volume_db": _parse_single_float(MAX_VOLUME_PATTERN, volumedetect.stderr),
        "integrated_lufs": None,
        "true_peak_dbtp": None,
        "lra_lu": None,
        "threshold_lufs": None,
    }
    for key, pattern in LOUDNESS_LINE_PATTERNS.items():
        metrics[key] = _parse_single_float(pattern, loudnorm.stderr)
    return metrics


def render_spectrogram_png(
    audio_path: Path,
    output_path: Path,
    *,
    ffmpeg_bin: str,
    start_seconds: float = 0.0,
    duration_seconds: float = 30.0,
) -> None:
    ffmpeg = ensure_tool(ffmpeg_bin)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-nostdin",
        "-ss",
        _format_seconds(start_seconds),
        "-t",
        _format_seconds(duration_seconds),
        "-i",
        str(audio_path),
        "-lavfi",
        "showspectrumpic=s=1800x900:legend=1:mode=combined:color=viridis:scale=log",
        "-frames:v",
        "1",
        "-update",
        "1",
        str(output_path),
    ]
    completed = run_command(command)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "spectrogram render failed")


def build_bridge_cleanup_windows_from_silences(
    silences: list[dict[str, float]],
    *,
    samples: array,
    sample_rate: int,
    config: BridgeCleanupConfig,
) -> tuple[
    list[NoiseWindow],
    list[dict[str, float | bool]],
    list[dict[str, float | bool | None | str]],
]:
    seed_windows: list[NoiseWindow] = []
    seed_debug: list[dict[str, float | bool | None | str]] = []
    total_samples = len(samples)
    for item in silences:
        duration_seconds = float(item["duration_seconds"])
        if duration_seconds < config.min_seed_silence_duration:
            continue
        window = NoiseWindow(
            start_seconds=float(item["start_seconds"]),
            end_seconds=float(item["end_seconds"]),
        )
        protected = False
        protection_reason: str | None = None
        left_context_peak_db: float | None = None
        left_context_rms_db: float | None = None
        right_context_peak_db: float | None = None
        right_context_rms_db: float | None = None
        if duration_seconds <= config.protected_silence_max_seconds:
            left_start_index = max(
                0,
                int((window.start_seconds - config.protected_context_probe_seconds) * sample_rate),
            )
            left_end_index = max(0, min(total_samples, int(window.start_seconds * sample_rate)))
            right_start_index = max(0, min(total_samples, int(window.end_seconds * sample_rate)))
            right_end_index = max(
                right_start_index,
                min(total_samples, int((window.end_seconds + config.protected_context_probe_seconds) * sample_rate)),
            )
            if left_end_index > left_start_index and right_end_index > right_start_index:
                left_context_peak_db, left_context_rms_db = measure_segment_levels(
                    samples,
                    start_index=left_start_index,
                    end_index=left_end_index,
                )
                right_context_peak_db, right_context_rms_db = measure_segment_levels(
                    samples,
                    start_index=right_start_index,
                    end_index=right_end_index,
                )
                if (
                    left_context_peak_db >= config.protected_context_peak_db
                    and left_context_rms_db >= config.protected_context_rms_db
                    and right_context_peak_db >= config.protected_context_peak_db
                    and right_context_rms_db >= config.protected_context_rms_db
                ):
                    protected = True
                    protection_reason = "continuous_speech_context"

        seed_debug.append(
            {
                "seed_start_seconds": round(window.start_seconds, 3),
                "seed_end_seconds": round(window.end_seconds, 3),
                "seed_duration_seconds": round(duration_seconds, 3),
                "protected": protected,
                "protection_reason": protection_reason,
                "left_context_peak_db": (
                    round(left_context_peak_db, 3) if left_context_peak_db is not None else None
                ),
                "left_context_rms_db": (
                    round(left_context_rms_db, 3) if left_context_rms_db is not None else None
                ),
                "right_context_peak_db": (
                    round(right_context_peak_db, 3) if right_context_peak_db is not None else None
                ),
                "right_context_rms_db": (
                    round(right_context_rms_db, 3) if right_context_rms_db is not None else None
                ),
            }
        )
        if not protected:
            seed_windows.append(window)
    if not seed_windows:
        return [], [], seed_debug

    merged_seed_windows: list[NoiseWindow] = []
    merged_seed_counts: list[int] = []
    merge_debug: list[dict[str, float | bool]] = []
    for window in seed_windows:
        if not merged_seed_windows:
            merged_seed_windows.append(window)
            merged_seed_counts.append(1)
            continue

        previous = merged_seed_windows[-1]
        gap_seconds = window.start_seconds - previous.end_seconds
        merge_allowed = False
        peak_db: float | None = None
        rms_db: float | None = None

        if gap_seconds <= config.tiny_gap_merge_seconds:
            merge_allowed = True
        elif gap_seconds <= config.bridge_gap_seconds:
            peak_db, rms_db = measure_segment_levels(
                samples,
                start_index=int(previous.end_seconds * sample_rate),
                end_index=int(window.start_seconds * sample_rate),
            )
            bridge_peak_limit = config.bridge_peak_db
            bridge_rms_limit = config.bridge_rms_db
            if gap_seconds >= config.protected_gap_seconds:
                bridge_peak_limit = min(bridge_peak_limit, config.protected_bridge_peak_db)
                bridge_rms_limit = min(bridge_rms_limit, config.protected_bridge_rms_db)
            if peak_db <= bridge_peak_limit and rms_db <= bridge_rms_limit:
                merge_allowed = True

        merge_debug.append(
            {
                "previous_end_seconds": round(previous.end_seconds, 3),
                "current_start_seconds": round(window.start_seconds, 3),
                "gap_seconds": round(gap_seconds, 3),
                "merged": merge_allowed,
                "gap_peak_db": round(peak_db, 3) if peak_db is not None else None,
                "gap_rms_db": round(rms_db, 3) if rms_db is not None else None,
                "bridge_peak_limit_db": round(bridge_peak_limit, 3) if peak_db is not None else None,
                "bridge_rms_limit_db": round(bridge_rms_limit, 3) if rms_db is not None else None,
            }
        )

        if merge_allowed:
            merged_seed_windows[-1] = NoiseWindow(
                start_seconds=previous.start_seconds,
                end_seconds=max(previous.end_seconds, window.end_seconds),
            )
            merged_seed_counts[-1] += 1
        else:
            merged_seed_windows.append(window)
            merged_seed_counts.append(1)

    final_windows: list[NoiseWindow] = []
    for window, seed_count in zip(merged_seed_windows, merged_seed_counts):
        if config.require_multi_seed_window and seed_count < 2:
            continue
        duration_seconds = window.end_seconds - window.start_seconds
        if duration_seconds < config.min_window_seconds:
            continue
        trim_seconds = (
            config.short_trim_seconds
            if duration_seconds < config.trim_switch_seconds
            else config.long_trim_seconds
        )
        start_seconds = window.start_seconds + trim_seconds
        end_seconds = window.end_seconds - trim_seconds
        if end_seconds - start_seconds >= config.min_window_seconds:
            final_windows.append(
                NoiseWindow(
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                )
            )
    return final_windows, merge_debug, seed_debug


def apply_bridge_cleanup_to_file(
    input_wav: Path,
    output_wav: Path,
    *,
    ffmpeg_bin: str,
    config: BridgeCleanupConfig,
) -> dict[str, Any]:
    silences = detect_silence_candidates(
        input_wav,
        ffmpeg_bin=ffmpeg_bin,
        threshold_db=config.silence_threshold_db,
        min_duration=config.silence_min_duration,
    )
    params, samples = _load_wave_samples(input_wav)
    windows, merge_debug, seed_debug = build_bridge_cleanup_windows_from_silences(
        silences,
        samples=samples,
        sample_rate=params.framerate,
        config=config,
    )
    shutil.copy2(input_wav, output_wav)
    if windows:
        duck_audio_file_in_place(
            output_wav,
            windows=windows,
            floor_gain=0.0,
            fade_ms=config.fade_ms,
        )
    return {
        "silence_candidate_count": len(silences),
        "mute_window_count": len(windows),
        "protected_seed_count": sum(1 for item in seed_debug if item["protected"]),
        "seed_debug": seed_debug,
        "merge_debug": merge_debug,
        "mute_windows": [
            {
                "start_seconds": round(window.start_seconds, 3),
                "end_seconds": round(window.end_seconds, 3),
                "duration_ms": round((window.end_seconds - window.start_seconds) * 1000.0, 1),
            }
            for window in windows
        ],
        "config": asdict(config),
    }


def build_hardmute_cleanup_windows_from_silences(
    silences: list[dict[str, float]],
    *,
    samples: array,
    sample_rate: int,
    config: HardMuteCleanupConfig,
) -> tuple[list[NoiseWindow], list[dict[str, float | bool | None | str]]]:
    silence_windows = [
        NoiseWindow(
            start_seconds=float(item["start_seconds"]),
            end_seconds=float(item["end_seconds"]),
        )
        for item in silences
    ]
    windows: list[NoiseWindow] = []
    debug_items: list[dict[str, float | bool | None | str]] = []
    for index, window in enumerate(silence_windows):
        raw_duration_seconds = window.end_seconds - window.start_seconds
        start_seconds = window.start_seconds + config.trim_start_seconds
        end_seconds = window.end_seconds - config.trim_end_seconds
        trimmed_duration_seconds = end_seconds - start_seconds
        protected = False
        protection_reason: str | None = None
        left_gap_duration_seconds: float | None = None
        right_gap_duration_seconds: float | None = None
        left_gap_peak_db: float | None = None
        left_gap_rms_db: float | None = None
        right_gap_peak_db: float | None = None
        right_gap_rms_db: float | None = None

        if 0 < index < (len(silence_windows) - 1):
            previous = silence_windows[index - 1]
            following = silence_windows[index + 1]
            left_gap_duration_seconds = window.start_seconds - previous.end_seconds
            right_gap_duration_seconds = following.start_seconds - window.end_seconds
            if (
                raw_duration_seconds <= config.protected_silence_max_seconds
                and config.protected_neighbor_min_seconds
                <= left_gap_duration_seconds
                <= config.protected_neighbor_max_seconds
                and config.protected_neighbor_min_seconds
                <= right_gap_duration_seconds
                <= config.protected_neighbor_max_seconds
            ):
                left_gap_peak_db, left_gap_rms_db = measure_segment_levels(
                    samples,
                    start_index=int(previous.end_seconds * sample_rate),
                    end_index=int(window.start_seconds * sample_rate),
                )
                right_gap_peak_db, right_gap_rms_db = measure_segment_levels(
                    samples,
                    start_index=int(window.end_seconds * sample_rate),
                    end_index=int(following.start_seconds * sample_rate),
                )
                if (
                    left_gap_peak_db >= config.protected_neighbor_peak_db
                    and left_gap_rms_db >= config.protected_neighbor_rms_db
                    and right_gap_peak_db >= config.protected_neighbor_peak_db
                    and right_gap_rms_db >= config.protected_neighbor_rms_db
                ):
                    protected = True
                    protection_reason = "continuous_speech_neighbors"

        applied = False
        if not protected and trimmed_duration_seconds >= config.min_window_seconds:
            windows.append(
                NoiseWindow(
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                )
            )
            applied = True

        debug_items.append(
            {
                "silence_start_seconds": round(window.start_seconds, 3),
                "silence_end_seconds": round(window.end_seconds, 3),
                "silence_duration_seconds": round(raw_duration_seconds, 3),
                "trimmed_start_seconds": round(start_seconds, 3),
                "trimmed_end_seconds": round(end_seconds, 3),
                "trimmed_duration_seconds": round(trimmed_duration_seconds, 3),
                "applied": applied,
                "protected": protected,
                "protection_reason": protection_reason,
                "left_gap_duration_seconds": (
                    round(left_gap_duration_seconds, 3)
                    if left_gap_duration_seconds is not None
                    else None
                ),
                "right_gap_duration_seconds": (
                    round(right_gap_duration_seconds, 3)
                    if right_gap_duration_seconds is not None
                    else None
                ),
                "left_gap_peak_db": round(left_gap_peak_db, 3) if left_gap_peak_db is not None else None,
                "left_gap_rms_db": round(left_gap_rms_db, 3) if left_gap_rms_db is not None else None,
                "right_gap_peak_db": round(right_gap_peak_db, 3) if right_gap_peak_db is not None else None,
                "right_gap_rms_db": round(right_gap_rms_db, 3) if right_gap_rms_db is not None else None,
            }
        )
    return windows, debug_items


def apply_hardmute_cleanup_to_file(
    input_wav: Path,
    output_wav: Path,
    *,
    ffmpeg_bin: str,
    config: HardMuteCleanupConfig,
) -> dict[str, Any]:
    silences = detect_silence_candidates(
        input_wav,
        ffmpeg_bin=ffmpeg_bin,
        threshold_db=config.silence_threshold_db,
        min_duration=config.silence_min_duration,
    )
    params, samples = _load_wave_samples(input_wav)
    windows, debug_items = build_hardmute_cleanup_windows_from_silences(
        silences,
        samples=samples,
        sample_rate=params.framerate,
        config=config,
    )
    shutil.copy2(input_wav, output_wav)
    if windows:
        duck_audio_file_in_place(
            output_wav,
            windows=windows,
            floor_gain=0.0,
            fade_ms=config.fade_ms,
        )
    return {
        "silence_count": len(silences),
        "window_count": len(windows),
        "protected_window_count": sum(1 for item in debug_items if item["protected"]),
        "total_window_ms": round(
            sum((window.end_seconds - window.start_seconds) * 1000.0 for window in windows),
            1,
        ),
        "first_30s_windows": [
            {
                "start_seconds": round(window.start_seconds, 3),
                "end_seconds": round(window.end_seconds, 3),
                "duration_ms": round((window.end_seconds - window.start_seconds) * 1000.0, 1),
            }
            for window in windows
            if window.start_seconds < 30.0
        ],
        "debug_items": debug_items,
        "config": asdict(config),
    }


def apply_exact_cleanup_to_file(
    input_wav: Path,
    output_wav: Path,
    *,
    mute_windows: Sequence[NoiseWindow],
    duck_windows: Sequence[ExactDuckWindow],
) -> dict[str, Any]:
    shutil.copy2(input_wav, output_wav)
    if mute_windows:
        duck_audio_file_in_place(
            output_wav,
            windows=mute_windows,
            floor_gain=0.0,
            fade_ms=0,
        )
    for spec in duck_windows:
        duck_audio_file_in_place(
            output_wav,
            windows=[
                NoiseWindow(
                    start_seconds=spec.start_seconds,
                    end_seconds=spec.end_seconds,
                )
            ],
            floor_gain=spec.floor_gain,
            fade_ms=spec.fade_ms,
        )
    return {
        "mute_window_count": len(mute_windows),
        "duck_window_count": len(duck_windows),
        "mute_windows": [
            {
                "start_seconds": round(window.start_seconds, 3),
                "end_seconds": round(window.end_seconds, 3),
                "duration_ms": round((window.end_seconds - window.start_seconds) * 1000.0, 1),
            }
            for window in mute_windows
        ],
        "duck_windows": [asdict(window) for window in duck_windows],
    }


def build_residue_cleanup_windows_from_silences(
    silences: list[dict[str, float]],
    *,
    samples: array,
    sample_rate: int,
    config: ResidueCleanupConfig,
) -> tuple[list[NoiseWindow], list[dict[str, float | bool | None]]]:
    windows: list[NoiseWindow] = []
    debug_items: list[dict[str, float | bool | None]] = []
    silence_windows = [
        NoiseWindow(
            start_seconds=float(item["start_seconds"]),
            end_seconds=float(item["end_seconds"]),
        )
        for item in silences
    ]

    for previous, current in zip(silence_windows, silence_windows[1:]):
        previous_duration = previous.end_seconds - previous.start_seconds
        gap_start = previous.end_seconds
        gap_end = current.start_seconds
        gap_duration = gap_end - gap_start
        accepted = False
        peak_db: float | None = None
        rms_db: float | None = None
        trimmed_start = gap_start
        trimmed_end = gap_end

        if previous_duration >= config.min_preceding_silence_seconds:
            if 0.0 < gap_duration <= config.max_residue_duration_seconds:
                peak_db, rms_db = measure_segment_levels(
                    samples,
                    start_index=int(gap_start * sample_rate),
                    end_index=int(gap_end * sample_rate),
                )
                if peak_db <= config.residue_peak_db and rms_db <= config.residue_rms_db:
                    trimmed_start = gap_start + config.trim_start_seconds
                    trimmed_end = gap_end - config.trim_end_seconds
                    if trimmed_end - trimmed_start >= config.min_window_seconds:
                        windows.append(
                            NoiseWindow(
                                start_seconds=trimmed_start,
                                end_seconds=trimmed_end,
                            )
                        )
                        accepted = True

        debug_items.append(
            {
                "previous_silence_end_seconds": round(previous.end_seconds, 3),
                "next_silence_start_seconds": round(current.start_seconds, 3),
                "previous_silence_duration_seconds": round(previous_duration, 3),
                "gap_duration_seconds": round(gap_duration, 3),
                "accepted": accepted,
                "gap_peak_db": round(peak_db, 3) if peak_db is not None else None,
                "gap_rms_db": round(rms_db, 3) if rms_db is not None else None,
                "trimmed_start_seconds": round(trimmed_start, 3) if accepted else None,
                "trimmed_end_seconds": round(trimmed_end, 3) if accepted else None,
            }
        )

    return windows, debug_items


def apply_residue_cleanup_to_file(
    input_wav: Path,
    output_wav: Path,
    *,
    ffmpeg_bin: str,
    config: ResidueCleanupConfig,
) -> dict[str, Any]:
    silences = detect_silence_candidates(
        input_wav,
        ffmpeg_bin=ffmpeg_bin,
        threshold_db=config.silence_threshold_db,
        min_duration=config.silence_min_duration,
    )
    params, samples = _load_wave_samples(input_wav)
    windows, debug_items = build_residue_cleanup_windows_from_silences(
        silences,
        samples=samples,
        sample_rate=params.framerate,
        config=config,
    )
    shutil.copy2(input_wav, output_wav)
    if windows:
        duck_audio_file_in_place(
            output_wav,
            windows=windows,
            floor_gain=config.floor_gain,
            fade_ms=config.fade_ms,
        )
    return {
        "silence_candidate_count": len(silences),
        "residue_window_count": len(windows),
        "debug_items": debug_items,
        "residue_windows": [
            {
                "start_seconds": round(window.start_seconds, 3),
                "end_seconds": round(window.end_seconds, 3),
                "duration_ms": round((window.end_seconds - window.start_seconds) * 1000.0, 1),
            }
            for window in windows
        ],
        "config": asdict(config),
    }


def apply_narrow_cleanup_to_file(
    input_wav: Path,
    output_wav: Path,
    *,
    ffmpeg_bin: str,
    config: Any | None = None,
) -> dict[str, Any]:
    from scripts.narrow_onset_cleanup import (
        NarrowConfig,
        _build_detection_windows,
        _load_samples,
        _narrow_windows,
        _pre_silence_windows,
    )

    if config is None:
        config = NarrowConfig(
            fade_ms=6,
            floor_gain=0.0,
            narrow_max_ms=110,
            narrow_min_ms=40,
            max_window_end_after_silence_ms=170,
            pre_silence_keep_tail_ms=45,
            pre_silence_floor_gain=0.0,
            pre_silence_fade_ms=0,
        )
    low_silences, high_silences, raw_windows = _build_detection_windows(
        input_wav,
        ffmpeg_bin=ffmpeg_bin,
        config=config,
    )
    samples, sample_rate = _load_samples(input_wav)
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

    shutil.copy2(input_wav, output_wav)
    if pre_silence_windows:
        duck_audio_file_in_place(
            output_wav,
            windows=pre_silence_windows,
            floor_gain=config.pre_silence_floor_gain,
            fade_ms=config.pre_silence_fade_ms,
        )
    if narrow_windows:
        duck_audio_file_in_place(
            output_wav,
            windows=narrow_windows,
            floor_gain=config.floor_gain,
            fade_ms=config.fade_ms,
        )

    return {
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
        "pre_silence_windows": [
            {
                "start_seconds": round(window.start_seconds, 3),
                "end_seconds": round(window.end_seconds, 3),
                "duration_ms": round((window.end_seconds - window.start_seconds) * 1000.0, 1),
            }
            for window in pre_silence_windows
        ],
        "refined_items": refined_items,
        "pre_silence_items": pre_silence_items,
        "low_threshold_silence_count": len(low_silences),
        "high_threshold_silence_count": len(high_silences),
        "config": asdict(config),
    }


def _legacy_reference_preview_a_config() -> Any:
    from scripts.narrow_onset_cleanup import NarrowConfig

    return NarrowConfig(
        low_threshold_db=-52.0,
        high_threshold_db=-38.0,
        silence_min_duration=0.10,
        fade_ms=6,
        floor_gain=0.05,
        narrow_max_ms=90,
        narrow_min_ms=45,
        max_window_end_after_silence_ms=160,
        min_preceding_silence_ms=160,
        pre_silence_keep_tail_ms=35,
        pre_silence_min_duration_ms=200,
        pre_silence_floor_gain=0.0,
        pre_silence_fade_ms=0,
    )


def _legacy_reference_preview_b_config() -> Any:
    from scripts.narrow_onset_cleanup import NarrowConfig

    return NarrowConfig(
        low_threshold_db=-50.0,
        high_threshold_db=-36.0,
        silence_min_duration=0.12,
        fade_ms=8,
        floor_gain=0.08,
        narrow_max_ms=90,
        narrow_min_ms=45,
        max_window_end_after_silence_ms=160,
        min_preceding_silence_ms=160,
        pre_silence_keep_tail_ms=45,
        pre_silence_min_duration_ms=220,
        pre_silence_floor_gain=0.0,
        pre_silence_fade_ms=0,
    )


def _legacy_reference_extreme_config() -> Any:
    from scripts.narrow_onset_cleanup import NarrowConfig

    return NarrowConfig(
        low_threshold_db=-48.0,
        high_threshold_db=-34.0,
        silence_min_duration=0.08,
        min_breath_ms=40,
        max_breath_ms=220,
        fade_ms=2,
        floor_gain=0.0,
        narrow_max_ms=110,
        narrow_min_ms=35,
        max_window_end_after_silence_ms=190,
        min_preceding_silence_ms=120,
        pre_silence_keep_tail_ms=20,
        pre_silence_min_duration_ms=140,
        pre_silence_floor_gain=0.0,
        pre_silence_fade_ms=0,
    )


def _legacy_reference_hardmute_config() -> HardMuteCleanupConfig:
    return HardMuteCleanupConfig(
        silence_threshold_db=-46.0,
        silence_min_duration=0.05,
        trim_start_seconds=0.02,
        trim_end_seconds=0.02,
        min_window_seconds=0.05,
        fade_ms=0.0,
    )


def _legacy_reference_bridge_config() -> BridgeCleanupConfig:
    return BridgeCleanupConfig(
        silence_threshold_db=-42.0,
        silence_min_duration=0.02,
        min_seed_silence_duration=0.03,
        require_multi_seed_window=False,
        tiny_gap_merge_seconds=0.06,
        bridge_gap_seconds=0.14,
        bridge_peak_db=-14.0,
        bridge_rms_db=-22.0,
        protected_gap_seconds=0.075,
        protected_bridge_peak_db=-22.0,
        protected_bridge_rms_db=-30.0,
        short_trim_seconds=0.004,
        long_trim_seconds=0.008,
        trim_switch_seconds=0.18,
        min_window_seconds=0.05,
        fade_ms=0.0,
    )


def _processing_order_for_mode(mode: WorkflowMode) -> list[str]:
    steps = [
        "Input source overview",
        "Respiro-en breath detection",
        "SpectraMini-style breath control and mouth de-click",
        "DeepFilterNet primary denoise",
        "FFmpeg mastering and loudnorm",
    ]
    if mode.name == "reference-legacy":
        steps.extend(
            [
                "Legacy preview A narrow and pre-silence cleanup",
                "Legacy preview B narrow and pre-silence cleanup",
                "Legacy extreme pause cleanup",
                "Legacy hard mute cleanup",
                "Legacy aggressive bridge cleanup",
                "Transcript-ready export and spectrogram artifacts",
            ]
        )
        return steps
    steps.extend(
        [
            "Optional narrow onset and pre-silence cleanup",
            "Optional pause-thread bridge cleanup",
            "Transcript-ready export and spectrogram artifacts",
        ]
    )
    return steps


def run_skill_workflow(
    *,
    input_path: Path,
    mode_name: str,
    output_dir: Path | None,
    recursive: bool,
    target_lufs_override: float | None,
    attenuation_db_override: float | None,
    ffmpeg_bin: str,
    ffprobe_bin: str,
    python_executable: str,
    noise_windows: Sequence[str],
    focus_windows: Sequence[FocusWindow],
    exact_mute_windows: Sequence[NoiseWindow],
    exact_duck_windows: Sequence[ExactDuckWindow],
    skip_spectrograms: bool,
    skip_bridge_cleanup: bool,
    delivery_dir: Path | None,
    delivery_prefix: str,
    keep_intermediate_audio: bool,
    spectrogram_start: float,
    spectrogram_duration: float,
) -> dict[str, Any]:
    from .cli import main as audio_cleanup_main

    mode = resolve_mode(mode_name)
    run_slug = utc_timestamp_slug()
    workflow_root = (
        output_dir
        if output_dir is not None
        else PROJECT_ROOT / "output" / f"skill-{run_slug}_{_slug_label(input_path.stem or input_path.name)}"
    )
    workflow_root.mkdir(parents=True, exist_ok=True)
    resolved_delivery_dir = delivery_dir if delivery_dir is not None else PROJECT_ROOT / "output" / "修音成品"
    resolved_delivery_dir.mkdir(parents=True, exist_ok=True)

    cli_args = [
        "clean",
        str(input_path),
        "--preset",
        mode.preset_name,
        "--output-dir",
        str(workflow_root),
        "--attenuation-db",
        str(attenuation_db_override if attenuation_db_override is not None else mode.attenuation_db),
        "--ffmpeg-bin",
        ffmpeg_bin,
        "--ffprobe-bin",
        ffprobe_bin,
        "--python-executable",
        python_executable,
    ]
    resolved_target_lufs = target_lufs_override if target_lufs_override is not None else mode.target_lufs
    if resolved_target_lufs is not None:
        cli_args.extend(["--target-lufs", str(resolved_target_lufs)])
    if recursive:
        cli_args.append("--recursive")
    for noise_window in noise_windows:
        cli_args.extend(["--noise-window", noise_window])

    lower_level_stdout = io.StringIO()
    lower_level_stderr = io.StringIO()
    with redirect_stdout(lower_level_stdout), redirect_stderr(lower_level_stderr):
        exit_code = audio_cleanup_main(cli_args)
    if exit_code != 0:
        details = "\n".join(
            part.strip()
            for part in (lower_level_stdout.getvalue(), lower_level_stderr.getvalue())
            if part.strip()
        )
        if details:
            raise RuntimeError(f"audio cleanup workflow failed with exit code {exit_code}\n{details}")
        raise RuntimeError(f"audio cleanup workflow failed with exit code {exit_code}")

    batch_summary_path = workflow_root / "batch-summary.json"
    batch_summary = json.loads(batch_summary_path.read_text(encoding="utf-8"))
    files: list[dict[str, Any]] = []
    for report_item in batch_summary["reports"]:
        core_report_path = Path(report_item["report_json"])
        core_report = json.loads(core_report_path.read_text(encoding="utf-8"))
        files.append(
            _finalize_workflow_file(
                core_report=core_report,
                mode=mode,
                ffmpeg_bin=ffmpeg_bin,
                reference_target_lufs=resolved_target_lufs if resolved_target_lufs is not None else -24.7,
                focus_windows=focus_windows,
                exact_mute_windows=exact_mute_windows,
                exact_duck_windows=exact_duck_windows,
                skip_spectrograms=skip_spectrograms,
                skip_bridge_cleanup=skip_bridge_cleanup or not mode.apply_bridge_cleanup,
                delivery_dir=resolved_delivery_dir,
                delivery_prefix=delivery_prefix,
                spectrogram_start=spectrogram_start,
                spectrogram_duration=spectrogram_duration,
                run_slug=run_slug,
            )
        )

    summary = {
        "mode": asdict(mode),
        "workflow_root": str(workflow_root),
        "delivery_dir": str(resolved_delivery_dir),
        "input_path": str(input_path),
        "recursive": recursive,
        "intermediate_audio_retained": keep_intermediate_audio,
        "lower_level_stdout": lower_level_stdout.getvalue(),
        "lower_level_stderr": lower_level_stderr.getvalue(),
        "processing_order": _processing_order_for_mode(mode),
        "files": files,
        "core_batch_summary": batch_summary,
    }
    if keep_intermediate_audio:
        summary["intermediate_audio_removed"] = []
    else:
        summary["intermediate_audio_removed"] = _prune_intermediate_audio(files)

    (workflow_root / "skill-workflow-summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (workflow_root / "skill-workflow-summary.md").write_text(
        render_skill_workflow_markdown(summary),
        encoding="utf-8",
    )
    return summary


def render_skill_workflow_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Repo Local Audio Skill Workflow",
        "",
        f"- Mode: `{summary['mode']['name']}`",
        f"- Base preset: `{summary['mode']['preset_name']}`",
        f"- Input path: `{summary['input_path']}`",
        f"- Workflow root: `{summary['workflow_root']}`",
        f"- Final delivery dir: `{summary['delivery_dir']}`",
        f"- Intermediate audio retained: `{summary['intermediate_audio_retained']}`",
        "",
        "## Processing Order",
    ]
    lines.extend(f"- {step}" for step in summary["processing_order"])
    lines.append("")
    lines.append("## Files")
    for item in summary["files"]:
        lines.extend(
            [
                f"- `{item['input_file']}`",
                f"  - final wav: `{item['deliverables']['wav']}`",
                f"  - final mp3: `{item['deliverables']['mp3']}`",
                f"  - workflow report: `{item['workflow_report_json']}`",
            ]
        )
    return "\n".join(lines)


def command_describe_modes(mode_name: str | None) -> int:
    print(json.dumps(describe_modes(mode_name), indent=2, ensure_ascii=False))
    return 0


def command_run(args: argparse.Namespace) -> int:
    summary = run_skill_workflow(
        input_path=Path(args.input_path),
        mode_name=args.mode,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        recursive=args.recursive,
        target_lufs_override=args.target_lufs,
        attenuation_db_override=args.attenuation_db,
        ffmpeg_bin=args.ffmpeg_bin,
        ffprobe_bin=args.ffprobe_bin,
        python_executable=args.python_executable,
        noise_windows=args.noise_window,
        focus_windows=[parse_focus_window(value) for value in args.focus_window],
        exact_mute_windows=[parse_exact_mute_window(value) for value in args.exact_mute_window],
        exact_duck_windows=[parse_exact_duck_window(value) for value in args.exact_duck_window],
        skip_spectrograms=args.skip_spectrograms,
        skip_bridge_cleanup=args.skip_bridge_cleanup,
        delivery_dir=Path(args.delivery_dir) if args.delivery_dir else None,
        delivery_prefix=args.delivery_prefix,
        keep_intermediate_audio=args.keep_intermediate_audio,
        spectrogram_start=args.spectrogram_start,
        spectrogram_duration=args.spectrogram_duration,
    )
    print(f"Workflow root: {summary['workflow_root']}")
    print(f"Final delivery dir: {summary['delivery_dir']}")
    for item in summary["files"]:
        print(f"- {item['input_file']} -> {item['deliverables']['wav']}")
    if not summary["intermediate_audio_retained"]:
        print(f"Removed intermediate audio files: {len(summary['intermediate_audio_removed'])}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "describe-modes":
        return command_describe_modes(args.mode)
    if args.command == "run":
        return command_run(args)
    raise ValueError(f"Unhandled command: {args.command}")


def _export_mp3_from_wav(source_wav: Path, output_mp3: Path, *, ffmpeg_bin: str) -> None:
    ffmpeg = ensure_tool(ffmpeg_bin)
    output_mp3.parent.mkdir(parents=True, exist_ok=True)
    completed = run_command(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-nostdin",
            "-i",
            str(source_wav),
            "-c:a",
            "libmp3lame",
            "-b:a",
            "192k",
            str(output_mp3),
        ]
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "mp3 export failed")


def _finalize_workflow_file(
    *,
    core_report: dict[str, Any],
    mode: WorkflowMode,
    ffmpeg_bin: str,
    reference_target_lufs: float,
    focus_windows: Sequence[FocusWindow],
    exact_mute_windows: Sequence[NoiseWindow],
    exact_duck_windows: Sequence[ExactDuckWindow],
    skip_spectrograms: bool,
    skip_bridge_cleanup: bool,
    delivery_dir: Path,
    delivery_prefix: str,
    spectrogram_start: float,
    spectrogram_duration: float,
    run_slug: str,
) -> dict[str, Any]:
    if mode.name == "reference-legacy":
        return _finalize_reference_legacy_file(
            core_report=core_report,
            mode=mode,
            ffmpeg_bin=ffmpeg_bin,
            reference_target_lufs=reference_target_lufs,
            focus_windows=focus_windows,
            exact_mute_windows=exact_mute_windows,
            exact_duck_windows=exact_duck_windows,
            skip_spectrograms=skip_spectrograms,
            delivery_dir=delivery_dir,
            delivery_prefix=delivery_prefix,
            spectrogram_start=spectrogram_start,
            spectrogram_duration=spectrogram_duration,
            run_slug=run_slug,
        )

    input_file = Path(core_report["input_file"])
    outputs = core_report["outputs"]
    raw_wav = Path(outputs["raw_wav"])
    denoised_wav = Path(outputs["denoised_wav"])
    clean_wav = Path(outputs["clean_wav"])
    transcript_mp3 = Path(outputs["transcript_mp3"])
    preprocess_dir = Path(outputs["preprocess_dir"])
    transcript_dir = Path(outputs["transcript_dir"])
    workflow_dir = preprocess_dir / "workflow_artifacts"
    workflow_dir.mkdir(parents=True, exist_ok=True)
    delivery_label = _delivery_label(input_file.stem, mode.suffix, run_slug)
    final_wav, final_mp3 = _reserve_delivery_paths(
        delivery_dir,
        input_file.stem,
        delivery_prefix=delivery_prefix,
    )

    delivered_wav = clean_wav
    internal_mp3 = transcript_mp3
    narrow_cleanup_report: dict[str, Any] | None = None
    bridge_cleanup_report: dict[str, Any] | None = None
    residue_cleanup_report: dict[str, Any] | None = None
    exact_cleanup_report: dict[str, Any] | None = None
    if mode.apply_narrow_cleanup:
        narrow_cleanup_output = preprocess_dir / f"{delivery_label}_narrow.wav"
        narrow_cleanup_report = apply_narrow_cleanup_to_file(
            clean_wav,
            narrow_cleanup_output,
            ffmpeg_bin=ffmpeg_bin,
        )
        delivered_wav = narrow_cleanup_output
    if not skip_bridge_cleanup:
        bridge_source = delivered_wav
        delivered_wav = preprocess_dir / f"{delivery_label}_bridge.wav"
        bridge_cleanup_report = apply_bridge_cleanup_to_file(
            bridge_source,
            delivered_wav,
            ffmpeg_bin=ffmpeg_bin,
            config=BridgeCleanupConfig(),
        )
    if mode.apply_narrow_cleanup:
        residue_source = delivered_wav
        delivered_wav = preprocess_dir / f"{delivery_label}.wav"
        residue_cleanup_report = apply_residue_cleanup_to_file(
            residue_source,
            delivered_wav,
            ffmpeg_bin=ffmpeg_bin,
            config=ResidueCleanupConfig(),
        )
    if exact_mute_windows or exact_duck_windows:
        exact_source = delivered_wav
        delivered_wav = preprocess_dir / f"{delivery_label}_精修版.wav"
        exact_cleanup_report = apply_exact_cleanup_to_file(
            exact_source,
            delivered_wav,
            mute_windows=exact_mute_windows,
            duck_windows=exact_duck_windows,
        )
    shutil.copy2(delivered_wav, final_wav)
    _export_mp3_from_wav(final_wav, final_mp3, ffmpeg_bin=ffmpeg_bin)
    delivered_wav = final_wav
    delivered_mp3 = final_mp3

    spectrograms: dict[str, str] = {}
    if not skip_spectrograms:
        overview_dir = workflow_dir / "spectrograms" / "overview"
        focus_dir = workflow_dir / "spectrograms" / "focus"
        step_sources = [
            ("input", input_file),
            ("respiro_spectra", raw_wav),
            ("deepfilternet", denoised_wav),
            ("mastered", clean_wav),
            ("delivered", delivered_wav),
        ]
        for label, source in step_sources:
            output_path = overview_dir / f"{label}_{_window_suffix(spectrogram_start, spectrogram_duration)}.png"
            render_spectrogram_png(
                source,
                output_path,
                ffmpeg_bin=ffmpeg_bin,
                start_seconds=spectrogram_start,
                duration_seconds=spectrogram_duration,
            )
            spectrograms[label] = str(output_path)
        for window in focus_windows:
            output_path = focus_dir / (
                f"{window.label}_{_window_suffix(window.start_seconds, window.duration_seconds)}.png"
            )
            render_spectrogram_png(
                delivered_wav,
                output_path,
                ffmpeg_bin=ffmpeg_bin,
                start_seconds=window.start_seconds,
                duration_seconds=window.duration_seconds,
            )
            spectrograms[f"focus:{window.label}"] = str(output_path)

    metrics = {
        "input": measure_audio_metrics(
            input_file,
            ffmpeg_bin=ffmpeg_bin,
            measurement_target_lufs=reference_target_lufs,
        ),
        "respiro_spectra": measure_audio_metrics(
            raw_wav,
            ffmpeg_bin=ffmpeg_bin,
            measurement_target_lufs=reference_target_lufs,
        ),
        "deepfilternet": measure_audio_metrics(
            denoised_wav,
            ffmpeg_bin=ffmpeg_bin,
            measurement_target_lufs=reference_target_lufs,
        ),
        "mastered": measure_audio_metrics(
            clean_wav,
            ffmpeg_bin=ffmpeg_bin,
            measurement_target_lufs=reference_target_lufs,
        ),
        "delivered": measure_audio_metrics(
            delivered_wav,
            ffmpeg_bin=ffmpeg_bin,
            measurement_target_lufs=reference_target_lufs,
        ),
    }

    workflow_file_report = {
        "input_file": str(input_file),
        "deliverables": {
            "wav": str(delivered_wav),
            "mp3": str(delivered_mp3),
            "delivery_dir": str(delivery_dir),
            "internal_mp3": str(internal_mp3),
        },
        "core_outputs": outputs,
        "spectrograms": spectrograms,
        "metrics": metrics,
        "core_report_json": outputs["report_json"],
        "core_report_md": outputs["report_md"],
        "narrow_cleanup": narrow_cleanup_report,
        "bridge_cleanup": bridge_cleanup_report,
        "residue_cleanup": residue_cleanup_report,
        "exact_cleanup": exact_cleanup_report,
    }
    workflow_report_json = workflow_dir / "skill-file-report.json"
    workflow_report_md = workflow_dir / "skill-file-report.md"
    workflow_report_json.write_text(
        json.dumps(workflow_file_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    workflow_report_md.write_text(
        _render_file_workflow_markdown(workflow_file_report),
        encoding="utf-8",
    )
    workflow_file_report["workflow_report_json"] = str(workflow_report_json)
    workflow_file_report["workflow_report_md"] = str(workflow_report_md)
    return workflow_file_report


def _finalize_reference_legacy_file(
    *,
    core_report: dict[str, Any],
    mode: WorkflowMode,
    ffmpeg_bin: str,
    reference_target_lufs: float,
    focus_windows: Sequence[FocusWindow],
    exact_mute_windows: Sequence[NoiseWindow],
    exact_duck_windows: Sequence[ExactDuckWindow],
    skip_spectrograms: bool,
    delivery_dir: Path,
    delivery_prefix: str,
    spectrogram_start: float,
    spectrogram_duration: float,
    run_slug: str,
) -> dict[str, Any]:
    input_file = Path(core_report["input_file"])
    outputs = core_report["outputs"]
    raw_wav = Path(outputs["raw_wav"])
    denoised_wav = Path(outputs["denoised_wav"])
    clean_wav = Path(outputs["clean_wav"])
    preprocess_dir = Path(outputs["preprocess_dir"])
    transcript_dir = Path(outputs["transcript_dir"])
    workflow_dir = preprocess_dir / "workflow_artifacts"
    workflow_dir.mkdir(parents=True, exist_ok=True)
    delivery_label = _delivery_label(input_file.stem, mode.suffix, run_slug)
    final_wav, final_mp3 = _reserve_delivery_paths(
        delivery_dir,
        input_file.stem,
        delivery_prefix=delivery_prefix,
    )

    preview_a_output = preprocess_dir / f"{input_file.stem}_clean_停顿残留加强版A.wav"
    preview_b_output = preprocess_dir / f"{input_file.stem}_clean_停顿残留加强版B.wav"
    extreme_output = preprocess_dir / f"{input_file.stem}_clean_极限停顿清理版.wav"
    hardmute_output = preprocess_dir / "hardmute_pause_tmp.wav"
    bridge_tmp_output = preprocess_dir / "bridgeclean_tmp.wav"
    delivered_wav = preprocess_dir / f"{delivery_label}.wav"
    exact_cleanup_report: dict[str, Any] | None = None

    narrow_cleanup_a_report = apply_narrow_cleanup_to_file(
        clean_wav,
        preview_a_output,
        ffmpeg_bin=ffmpeg_bin,
        config=_legacy_reference_preview_a_config(),
    )
    _write_json_report(preprocess_dir / "narrow_cleanup_A.report.json", narrow_cleanup_a_report)

    narrow_cleanup_b_report = apply_narrow_cleanup_to_file(
        clean_wav,
        preview_b_output,
        ffmpeg_bin=ffmpeg_bin,
        config=_legacy_reference_preview_b_config(),
    )
    _write_json_report(preprocess_dir / "narrow_cleanup_B.report.json", narrow_cleanup_b_report)

    extreme_cleanup_report = apply_narrow_cleanup_to_file(
        clean_wav,
        extreme_output,
        ffmpeg_bin=ffmpeg_bin,
        config=_legacy_reference_extreme_config(),
    )
    _write_json_report(preprocess_dir / "extreme_cleanup_preview.report.json", extreme_cleanup_report)
    _write_json_report(preprocess_dir / f"{extreme_output.stem}.report.json", extreme_cleanup_report)

    hardmute_cleanup_report = apply_hardmute_cleanup_to_file(
        extreme_output,
        hardmute_output,
        ffmpeg_bin=ffmpeg_bin,
        config=_legacy_reference_hardmute_config(),
    )
    _write_json_report(preprocess_dir / "hardmute_pause_tmp.report.json", hardmute_cleanup_report)

    bridge_cleanup_report = apply_bridge_cleanup_to_file(
        hardmute_output,
        bridge_tmp_output,
        ffmpeg_bin=ffmpeg_bin,
        config=_legacy_reference_bridge_config(),
    )
    bridge_cleanup_report["source"] = hardmute_output.name
    _write_json_report(preprocess_dir / "bridgeclean_tmp.report.json", bridge_cleanup_report)

    shutil.copy2(bridge_tmp_output, delivered_wav)
    if exact_mute_windows or exact_duck_windows:
        exact_output = preprocess_dir / f"{delivery_label}_精修版.wav"
        exact_cleanup_report = apply_exact_cleanup_to_file(
            delivered_wav,
            exact_output,
            mute_windows=exact_mute_windows,
            duck_windows=exact_duck_windows,
        )
        delivered_wav = exact_output
    shutil.copy2(delivered_wav, final_wav)
    _export_mp3_from_wav(final_wav, final_mp3, ffmpeg_bin=ffmpeg_bin)
    delivered_wav = final_wav
    delivered_mp3 = final_mp3

    spectrograms: dict[str, str] = {}
    if not skip_spectrograms:
        overview_dir = workflow_dir / "spectrograms" / "overview"
        focus_dir = workflow_dir / "spectrograms" / "focus"
        step_sources = [
            ("input", input_file),
            ("respiro_spectra", raw_wav),
            ("deepfilternet", denoised_wav),
            ("mastered", clean_wav),
            ("legacy_extreme", extreme_output),
            ("legacy_hardmute", hardmute_output),
            ("legacy_bridgeclean", bridge_tmp_output),
            ("delivered", delivered_wav),
        ]
        for label, source in step_sources:
            output_path = overview_dir / f"{label}_{_window_suffix(spectrogram_start, spectrogram_duration)}.png"
            render_spectrogram_png(
                source,
                output_path,
                ffmpeg_bin=ffmpeg_bin,
                start_seconds=spectrogram_start,
                duration_seconds=spectrogram_duration,
            )
            spectrograms[label] = str(output_path)
        for window in focus_windows:
            output_path = focus_dir / (
                f"{window.label}_{_window_suffix(window.start_seconds, window.duration_seconds)}.png"
            )
            render_spectrogram_png(
                delivered_wav,
                output_path,
                ffmpeg_bin=ffmpeg_bin,
                start_seconds=window.start_seconds,
                duration_seconds=window.duration_seconds,
            )
            spectrograms[f"focus:{window.label}"] = str(output_path)

    metrics = {
        "input": measure_audio_metrics(
            input_file,
            ffmpeg_bin=ffmpeg_bin,
            measurement_target_lufs=reference_target_lufs,
        ),
        "respiro_spectra": measure_audio_metrics(
            raw_wav,
            ffmpeg_bin=ffmpeg_bin,
            measurement_target_lufs=reference_target_lufs,
        ),
        "deepfilternet": measure_audio_metrics(
            denoised_wav,
            ffmpeg_bin=ffmpeg_bin,
            measurement_target_lufs=reference_target_lufs,
        ),
        "mastered": measure_audio_metrics(
            clean_wav,
            ffmpeg_bin=ffmpeg_bin,
            measurement_target_lufs=reference_target_lufs,
        ),
        "legacy_bridgeclean": measure_audio_metrics(
            bridge_tmp_output,
            ffmpeg_bin=ffmpeg_bin,
            measurement_target_lufs=reference_target_lufs,
        ),
        "delivered": measure_audio_metrics(
            delivered_wav,
            ffmpeg_bin=ffmpeg_bin,
            measurement_target_lufs=reference_target_lufs,
        ),
    }

    workflow_file_report = {
        "input_file": str(input_file),
        "deliverables": {
            "wav": str(delivered_wav),
            "mp3": str(delivered_mp3),
            "delivery_dir": str(delivery_dir),
        },
        "core_outputs": outputs,
        "spectrograms": spectrograms,
        "metrics": metrics,
        "core_report_json": outputs["report_json"],
        "core_report_md": outputs["report_md"],
        "narrow_cleanup_a": narrow_cleanup_a_report,
        "narrow_cleanup_b": narrow_cleanup_b_report,
        "extreme_cleanup": extreme_cleanup_report,
        "hardmute_cleanup": hardmute_cleanup_report,
        "bridge_cleanup": bridge_cleanup_report,
        "exact_cleanup": exact_cleanup_report,
    }
    workflow_report_json = workflow_dir / "skill-file-report.json"
    workflow_report_md = workflow_dir / "skill-file-report.md"
    workflow_report_json.write_text(
        json.dumps(workflow_file_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    workflow_report_md.write_text(
        _render_file_workflow_markdown(workflow_file_report),
        encoding="utf-8",
    )
    workflow_file_report["workflow_report_json"] = str(workflow_report_json)
    workflow_file_report["workflow_report_md"] = str(workflow_report_md)
    return workflow_file_report


def _render_file_workflow_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Skill File Report: {Path(report['input_file']).name}",
        "",
        "## Deliverables",
        f"- WAV: `{report['deliverables']['wav']}`",
        f"- MP3: `{report['deliverables']['mp3']}`",
        "",
        "## Metrics",
    ]
    for label, metrics in report["metrics"].items():
        lines.extend(
            [
                f"### {label}",
                f"- mean_volume_db: `{metrics['mean_volume_db']}`",
                f"- max_volume_db: `{metrics['max_volume_db']}`",
                f"- integrated_lufs: `{metrics['integrated_lufs']}`",
                f"- true_peak_dbtp: `{metrics['true_peak_dbtp']}`",
                f"- lra_lu: `{metrics['lra_lu']}`",
            ]
        )
    if report.get("narrow_cleanup_a"):
        lines.extend(
            [
                "",
                "## Legacy Preview A",
                f"- narrow_window_count: `{report['narrow_cleanup_a']['narrow_window_count']}`",
                f"- pre_silence_window_count: `{report['narrow_cleanup_a']['pre_silence_window_count']}`",
            ]
        )
    if report.get("narrow_cleanup_b"):
        lines.extend(
            [
                "",
                "## Legacy Preview B",
                f"- narrow_window_count: `{report['narrow_cleanup_b']['narrow_window_count']}`",
                f"- pre_silence_window_count: `{report['narrow_cleanup_b']['pre_silence_window_count']}`",
            ]
        )
    if report.get("extreme_cleanup"):
        lines.extend(
            [
                "",
                "## Legacy Extreme Cleanup",
                f"- narrow_window_count: `{report['extreme_cleanup']['narrow_window_count']}`",
                f"- pre_silence_window_count: `{report['extreme_cleanup']['pre_silence_window_count']}`",
            ]
        )
    if report.get("narrow_cleanup"):
        lines.extend(
            [
                "",
                "## Narrow Cleanup",
                f"- narrow_window_count: `{report['narrow_cleanup']['narrow_window_count']}`",
                f"- pre_silence_window_count: `{report['narrow_cleanup']['pre_silence_window_count']}`",
            ]
        )
    if report.get("bridge_cleanup"):
        lines.extend(
            [
                "",
                "## Bridge Cleanup",
                f"- mute_window_count: `{report['bridge_cleanup']['mute_window_count']}`",
                f"- silence_candidate_count: `{report['bridge_cleanup']['silence_candidate_count']}`",
            ]
        )
    if report.get("hardmute_cleanup"):
        lines.extend(
            [
                "",
                "## Hard Mute Cleanup",
                f"- window_count: `{report['hardmute_cleanup']['window_count']}`",
                f"- silence_count: `{report['hardmute_cleanup']['silence_count']}`",
                f"- total_window_ms: `{report['hardmute_cleanup']['total_window_ms']}`",
            ]
        )
    if report.get("residue_cleanup"):
        lines.extend(
            [
                "",
                "## Residue Cleanup",
                f"- residue_window_count: `{report['residue_cleanup']['residue_window_count']}`",
                f"- silence_candidate_count: `{report['residue_cleanup']['silence_candidate_count']}`",
            ]
        )
    if report.get("exact_cleanup"):
        lines.extend(
            [
                "",
                "## Exact Cleanup",
                f"- mute_window_count: `{report['exact_cleanup']['mute_window_count']}`",
                f"- duck_window_count: `{report['exact_cleanup']['duck_window_count']}`",
            ]
        )
    if report.get("spectrograms"):
        lines.extend(["", "## Spectrograms"])
        lines.extend(f"- {label}: `{path}`" for label, path in report["spectrograms"].items())
    return "\n".join(lines)


def _load_wave_samples(audio_path: Path) -> tuple[Any, array]:
    with wave.open(str(audio_path), "rb") as reader:
        params = reader.getparams()
        if params.sampwidth != 2 or params.nchannels != 1:
            raise ValueError("Expected mono 16-bit WAV audio")
        raw_frames = reader.readframes(params.nframes)
    samples = array("h")
    samples.frombytes(raw_frames)
    return params, samples


def _parse_single_float(pattern: re.Pattern[str], text: str) -> float | None:
    match = pattern.search(text)
    if not match:
        return None
    return float(match.group(1))


def _write_json_report(report_path: Path, payload: dict[str, Any]) -> None:
    report_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _format_seconds(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _window_suffix(start_seconds: float, duration_seconds: float) -> str:
    start_label = _format_seconds(start_seconds).replace(".", "_")
    duration_label = _format_seconds(duration_seconds).replace(".", "_")
    return f"{start_label}_{duration_label}"


def _delivery_label(input_stem: str, mode_suffix: str, run_slug: str) -> str:
    return f"{input_stem}_clean_{mode_suffix}_{run_slug}"


def _final_delivery_label(input_stem: str, delivery_prefix: str = "修音版") -> str:
    safe_prefix = _safe_windows_stem(delivery_prefix) or "修音版"
    safe_stem = _safe_windows_stem(input_stem) or "audio"
    return f"{safe_prefix}_{safe_stem}"


def _reserve_delivery_paths(
    delivery_dir: Path,
    input_stem: str,
    *,
    delivery_prefix: str = "修音版",
) -> tuple[Path, Path]:
    delivery_dir.mkdir(parents=True, exist_ok=True)
    base_label = _final_delivery_label(input_stem, delivery_prefix)
    for index in range(10000):
        suffix = "" if index == 0 else f"_{index:02d}"
        wav_path = delivery_dir / f"{base_label}{suffix}.wav"
        mp3_path = delivery_dir / f"{base_label}{suffix}.mp3"
        if not wav_path.exists() and not mp3_path.exists():
            return wav_path, mp3_path
    raise RuntimeError(f"Could not reserve a unique delivery filename under {delivery_dir}")


def _safe_windows_stem(value: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "-", value.strip())
    cleaned = cleaned.rstrip(" .")
    return cleaned or "audio"


def _prune_intermediate_audio(workflow_files: Sequence[dict[str, Any]]) -> list[str]:
    removed: list[str] = []
    suffixes = {".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg"}
    for item in workflow_files:
        core_outputs = item.get("core_outputs", {})
        roots = [
            Path(core_outputs["preprocess_dir"])
            for key in ("preprocess_dir",)
            if core_outputs.get(key)
        ]
        roots.extend(
            Path(core_outputs[key])
            for key in ("transcript_dir",)
            if core_outputs.get(key)
        )
        for root in roots:
            if not root.exists() or not root.is_dir():
                continue
            for path in sorted(root.rglob("*")):
                if path.is_file() and path.suffix.lower() in suffixes:
                    path.unlink()
                    removed.append(str(path))
            _remove_empty_dirs(root)
    return removed


def _remove_empty_dirs(root: Path) -> None:
    if not root.exists():
        return
    for directory in sorted((path for path in root.rglob("*") if path.is_dir()), reverse=True):
        try:
            directory.rmdir()
        except OSError:
            pass


def _slug_label(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]+", "-", value.strip())
    cleaned = re.sub(r"-+", "-", cleaned).strip("-")
    return cleaned or "window"
