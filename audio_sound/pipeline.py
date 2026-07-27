from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import tempfile
import wave
from array import array
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from statistics import median
from typing import Any, Sequence

from .config import PROJECT_ROOT, resolve_binary, resolve_repo_python
from .media_utils import (
    build_mp3_export_command,
    load_pcm16_wave as _load_wave_samples,
)
from .stereo_balance import run_stereo_balance, write_pcm16_wave

SUPPORTED_MEDIA_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".aac",
    ".ogg",
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
}
JSON_BLOCK_PATTERN = re.compile(r"\{\s*\"input_i\".*?\}", re.DOTALL)
SILENCE_START_PATTERN = re.compile(r"silence_start:\s*([0-9.]+)")
SILENCE_END_PATTERN = re.compile(r"silence_end:\s*([0-9.]+)\s*\|\s*silence_duration:\s*([0-9.]+)")
DEFAULT_BREATH_FALLBACK_CONFIG = {
    "low_threshold_db": -44,
    "high_threshold_db": -32,
    "silence_min_duration": 0.08,
    "min_breath_ms": 70,
    "max_breath_ms": 320,
    "pre_roll_ms": 25,
    "fade_ms": 14,
    "floor_gain": 0.0,
    "analysis_hop_ms": 18,
    "analysis_scan_ms": 260,
    "noise_floor_ms": 120,
    "speech_start_ratio": 0.78,
    "breath_over_noise_ratio": 2.6,
    "speech_over_breath_ratio": 2.15,
    "speech_confirm_frames": 2,
}


@dataclass
class RuntimeOptions:
    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"
    python_executable: str | None = None
    overwrite: bool = True
    dry_run: bool = False
    env_path: Path | None = None


@dataclass(frozen=True)
class NoiseWindow:
    start_seconds: float
    end_seconds: float


@dataclass
class RespiroDetectionResult:
    windows: list[NoiseWindow]
    mode: str
    assets_present: bool
    attempted: bool
    succeeded: bool
    command: list[str] | None = None
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    error: str | None = None


@dataclass(frozen=True)
class OutputLayout:
    job_name: str
    job_dir: Path
    preprocess_dir: Path
    transcript_dir: Path
    raw_wav: Path
    breath_wav: Path
    denoised_wav: Path
    noise_sample_wav: Path
    clean_wav: Path
    transcript_mp3: Path
    report_json: Path
    report_md: Path
    deepfilternet_dir: Path


def build_breath_processing_plan(
    preset: dict[str, Any],
    *,
    attenuation_db: float,
    skip_spectramini: bool,
    skip_deepfilternet: bool,
) -> list[dict[str, Any]]:
    stages = preset.get("pipeline", {}).get("stages", [])
    plan: list[dict[str, Any]] = []
    for stage in stages:
        stage_type = stage.get("type")
        if not stage.get("enabled", True):
            continue
        if stage_type == "spectramini" and skip_spectramini:
            continue
        if stage_type == "deepfilternet" and skip_deepfilternet:
            continue
        resolved = dict(stage)
        if stage_type == "respiro":
            resolved["attenuation_db"] = float(attenuation_db)
        plan.append(resolved)
    return plan


SAFE_ADAPTIVE_PROFILES = frozenset(
    {"baseline", "leveling_gentle", "noise_review", "manual_review"}
)


def select_adaptive_profile(analysis: dict[str, Any]) -> dict[str, Any]:
    """Select a bounded local processing profile from measured audio facts."""
    if int(analysis.get("clipped_sample_count", 0)) > 0:
        return {
            "profile": "manual_review",
            "confidence": 0.98,
            "reason": "Clipped samples were detected; automatic strengthening is unsafe.",
            "allow_destructive_cleanup": False,
        }
    if analysis.get("stationary_noise") and analysis.get("candidate_noise_windows"):
        return {
            "profile": "noise_review",
            "confidence": 0.82,
            "reason": "A stable noise floor was detected, but automatic denoise remains disabled until a window is reviewed.",
            "allow_destructive_cleanup": False,
        }
    if float(analysis.get("active_dynamic_range_db", 0.0)) > 12.0:
        return {
            "profile": "leveling_gentle",
            "confidence": 0.78,
            "reason": "Speech dynamics are wide enough for bounded gentle leveling.",
            "allow_destructive_cleanup": False,
        }
    return {
        "profile": "baseline",
        "confidence": 0.72,
        "reason": "No high-confidence defect requires stronger processing.",
        "allow_destructive_cleanup": False,
    }


def apply_adaptive_profile(preset: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    profile = str(decision.get("profile", "baseline"))
    if profile not in SAFE_ADAPTIVE_PROFILES:
        raise ValueError(f"Unsupported adaptive profile: {profile}")
    resolved = deepcopy(preset)
    if profile == "leveling_gentle":
        compressor = resolved.setdefault("filters", {}).setdefault("compressor", {})
        compressor["enabled"] = True
        current_threshold = float(compressor.get("threshold_db", -16.0))
        current_ratio = float(compressor.get("ratio", 1.25))
        compressor["threshold_db"] = max(-20.0, min(-18.0, current_threshold))
        compressor["ratio"] = min(1.4, max(1.35, current_ratio))
    return resolved


def apply_input_safety_overrides(
    preset_name: str,
    preset: dict[str, Any],
    diagnostics: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    resolved = preset
    adaptations: list[str] = []
    if (
        preset_name == "final"
        and not bool(diagnostics.get("stationary_noise"))
        and float(diagnostics.get("estimated_snr_db", 0.0) or 0.0) >= 35.0
    ):
        resolved = deepcopy(preset)
        secondary_denoise = resolved.setdefault("filters", {}).setdefault(
            "secondary_denoise",
            {},
        )
        secondary_denoise["enabled"] = False
        adaptations.append("skip_secondary_denoise_clean_source")
    return resolved, adaptations


def utc_timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def build_output_root(base_dir: str | Path, run_slug: str) -> Path:
    return Path(base_dir) / f"run-{run_slug}"


def _slugify_stem(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("._")
    if cleaned and re.search(r"[A-Za-z0-9]", cleaned):
        return cleaned
    digest = sha1(value.encode("utf-8")).hexdigest()[:8]
    return f"media_{digest}"


def get_pipeline_stage(preset: dict[str, Any], stage_type: str) -> dict[str, Any]:
    for stage in preset.get("pipeline", {}).get("stages", []):
        if stage.get("type") == stage_type:
            return stage
    raise KeyError(stage_type)


def attenuation_db_to_gain(attenuation_db: float) -> float:
    return 10 ** (-float(attenuation_db) / 20.0)


def build_respiro_detect_command(
    *,
    audio_path: Path,
    python_executable: str,
    repo_path: Path,
    weights_path: Path,
    threshold: float,
    min_length_ms: int,
) -> list[str]:
    script = (
        "import json, sys, torch; "
        "from pathlib import Path; "
        "repo = Path(sys.argv[1]); "
        "weights = Path(sys.argv[2]); "
        "audio = Path(sys.argv[3]); "
        "threshold = float(sys.argv[4]); "
        "min_length_ms = int(sys.argv[5]); "
        "sys.path.insert(0, str(repo)); "
        "from modules import DetectionNet, BreathDetector; "
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); "
        "model = DetectionNet().to(device); "
        "checkpoint = torch.load(str(weights), map_location=device); "
        "model.load_state_dict(checkpoint['model']); "
        "model.eval(); "
        "detector = BreathDetector(model, device=device); "
        "tree = detector(str(audio), threshold=threshold, min_length=max(1, int(round(min_length_ms / 10.0)))); "
        "intervals = [{'start_seconds': float(item.begin), 'end_seconds': float(item.end)} for item in sorted(tree)]; "
        "print(json.dumps({'intervals': intervals}))"
    )
    return [
        python_executable,
        "-c",
        script,
        str(repo_path),
        str(weights_path),
        str(audio_path),
        str(threshold),
        str(min_length_ms),
    ]


def resolve_respiro_runtime(
    *,
    respiro_repo: str | None,
    respiro_weights: str | None,
    env_values: dict[str, str] | None = None,
) -> dict[str, Path | None]:
    values = env_values or {}
    repo_value = respiro_repo or values.get("AUDIO_SOUND_RESPIRO_REPO")
    weights_value = respiro_weights or values.get("AUDIO_SOUND_RESPIRO_WEIGHTS")
    return {
        "repo_path": Path(repo_value) if repo_value else None,
        "weights_path": Path(weights_value) if weights_value else None,
    }


def apply_spectramini_style_cleanup_to_samples(
    samples: array,
    *,
    breath_windows: list[NoiseWindow],
    sample_rate: int,
    attenuation_db: float,
    mouth_declick_sensitivity: float,
    fade_ms: float,
    channels: int = 1,
) -> array:
    if channels <= 0:
        raise ValueError("channels must be positive")
    if channels > 1:
        channel_samples = [array("h") for _ in range(channels)]
        for frame_start in range(0, len(samples), channels):
            for channel_index, value in enumerate(samples[frame_start : frame_start + channels]):
                channel_samples[channel_index].append(value)
        cleaned_channels = [
            apply_spectramini_style_cleanup_to_samples(
                channel,
                breath_windows=breath_windows,
                sample_rate=sample_rate,
                attenuation_db=attenuation_db,
                mouth_declick_sensitivity=mouth_declick_sensitivity,
                fade_ms=fade_ms,
            )
            for channel in channel_samples
        ]
        cleaned = array("h")
        frame_count = max((len(channel) for channel in cleaned_channels), default=0)
        for frame_index in range(frame_count):
            for channel in cleaned_channels:
                if frame_index < len(channel):
                    cleaned.append(channel[frame_index])
        return cleaned

    cleaned = array("h", samples)
    duck_samples_for_windows(
        cleaned,
        sample_rate=sample_rate,
        windows=breath_windows,
        floor_gain=attenuation_db_to_gain(attenuation_db),
        fade_ms=fade_ms,
    )

    if float(mouth_declick_sensitivity) <= 0.0:
        return cleaned

    if len(cleaned) < 3:
        return cleaned

    threshold_scale = max(1.0, 6.0 - (float(mouth_declick_sensitivity) * 4.0))
    diffs = [abs(int(cleaned[index + 1]) - int(cleaned[index])) for index in range(len(cleaned) - 1)]
    median_diff = float(median(diffs))
    median_deviation = float(median(abs(diff - median_diff) for diff in diffs))
    click_threshold = median_diff + (threshold_scale * max(1.0, median_deviation))

    for _ in range(2):
        for index in range(1, len(cleaned) - 1):
            left_delta = abs(int(cleaned[index]) - int(cleaned[index - 1]))
            right_delta = abs(int(cleaned[index + 1]) - int(cleaned[index]))
            sample_peak = abs(int(cleaned[index]))
            is_impulsive = (
                sample_peak > click_threshold
                and min(left_delta, right_delta) > click_threshold
            )
            if is_impulsive:
                cleaned[index] = int((int(cleaned[index - 1]) + int(cleaned[index + 1])) / 2)

    return cleaned


def _clip_window_before_active_speech(
    window: NoiseWindow,
    source_analysis: dict[str, Any],
) -> NoiseWindow | None:
    frame_levels = list(source_analysis.get("_frame_rms_db") or [])
    if not frame_levels:
        return window
    frame_seconds = max(0.001, float(source_analysis.get("frame_ms", 10.0)) / 1000.0)
    speech_level = float(source_analysis.get("speech_level_dbfs", -24.0))
    # Never let cleanup authorization become less sensitive than the hard-mute
    # preservation guard. This protects quieter speech onsets that sit below
    # the representative speech level but are still active source speech.
    protection_threshold = min(speech_level - 6.0, -35.0)
    start_index = max(0, int(math.floor(window.start_seconds / frame_seconds)))
    end_index = min(len(frame_levels), int(math.ceil(window.end_seconds / frame_seconds)))
    if end_index <= start_index:
        return None

    leading = start_index
    while leading < end_index and float(frame_levels[leading]) >= protection_threshold:
        leading += 1
    trailing = end_index - 1
    while trailing >= leading and float(frame_levels[trailing]) >= protection_threshold:
        trailing -= 1
    if trailing < leading:
        return None

    first_internal_active = next(
        (
            index
            for index in range(leading, trailing + 1)
            if float(frame_levels[index]) >= protection_threshold
        ),
        None,
    )
    if first_internal_active is not None:
        trailing = first_internal_active - 1
    if trailing < leading:
        return None

    safe_start = max(window.start_seconds, leading * frame_seconds)
    safe_end = min(window.end_seconds, (trailing + 1) * frame_seconds)
    if safe_end - safe_start < 0.04:
        return None
    return NoiseWindow(round(safe_start, 6), round(safe_end, 6))


def build_effective_breath_windows(
    *,
    respiro_windows: list[NoiseWindow],
    auxiliary_windows: list[NoiseWindow],
    source_analysis: dict[str, Any],
    max_gap_seconds: float = 0.008,
) -> tuple[list[NoiseWindow], list[dict[str, Any]]]:
    tagged = [
        (window, "respiro") for window in respiro_windows
    ] + [
        (window, "auxiliary") for window in auxiliary_windows
    ]
    tagged.sort(key=lambda item: (item[0].start_seconds, item[0].end_seconds))
    safe_tagged: list[tuple[NoiseWindow, str]] = []
    rejected: list[tuple[NoiseWindow, str]] = []
    for window, source in tagged:
        safe_window = _clip_window_before_active_speech(window, source_analysis)
        if safe_window is None:
            rejected.append((window, source))
        else:
            safe_tagged.append((safe_window, source))
    tagged = safe_tagged
    groups: list[tuple[NoiseWindow, set[str]]] = []
    for window, source in tagged:
        if window.end_seconds <= window.start_seconds:
            continue
        if groups and window.start_seconds <= groups[-1][0].end_seconds + max_gap_seconds:
            previous, sources = groups[-1]
            groups[-1] = (
                NoiseWindow(
                    start_seconds=min(previous.start_seconds, window.start_seconds),
                    end_seconds=max(previous.end_seconds, window.end_seconds),
                ),
                {*sources, source},
            )
        else:
            groups.append((window, {source}))

    accepted: list[NoiseWindow] = []
    evidence: list[dict[str, Any]] = [
        {
            "start_seconds": round(window.start_seconds, 6),
            "end_seconds": round(window.end_seconds, 6),
            "sources": [source],
            "evidence_grade": "single_source_review",
            "decision": "rejected_active_speech",
        }
        for window, source in rejected
    ]
    for window, sources in groups:
        evidence.append(
            {
                "start_seconds": round(window.start_seconds, 6),
                "end_seconds": round(window.end_seconds, 6),
                "sources": sorted(sources, key=("respiro", "auxiliary").index),
                "evidence_grade": (
                    "model_aux_confirmed"
                    if sources == {"respiro", "auxiliary"}
                    else "single_source_confirmed"
                ),
                "decision": "accepted",
            }
        )
        accepted.append(window)
    evidence.sort(key=lambda item: (item["start_seconds"], item["end_seconds"]))
    return accepted, evidence


def filter_noise_like_breath_windows(
    samples: array,
    *,
    windows: list[NoiseWindow],
    sample_rate: int,
    channels: int,
    minimum_zero_crossing_rate: float = 0.04,
    minimum_roughness_ratio: float = 0.45,
) -> tuple[list[NoiseWindow], list[dict[str, Any]]]:
    accepted: list[NoiseWindow] = []
    evidence: list[dict[str, Any]] = []
    for window in windows:
        interleaved = _window_samples(
            samples,
            window,
            sample_rate=sample_rate,
            channels=channels,
        )
        mono = array("h")
        for frame_start in range(0, len(interleaved), channels):
            frame = interleaved[frame_start : frame_start + channels]
            if frame:
                mono.append(int(sum(int(value) for value in frame) / len(frame)))
        rms = _compute_rms(mono)
        differences = array(
            "h",
            [
                max(-32768, min(32767, int(mono[index]) - int(mono[index - 1])))
                for index in range(1, len(mono))
            ],
        )
        roughness_ratio = _compute_rms(differences) / max(1.0, rms)
        zero_crossings = sum(
            1
            for index in range(1, len(mono))
            if (int(mono[index - 1]) < 0 <= int(mono[index]))
            or (int(mono[index - 1]) >= 0 > int(mono[index]))
        )
        zero_crossing_rate = zero_crossings / max(1, len(mono) - 1)
        is_noise_like = (
            zero_crossing_rate >= minimum_zero_crossing_rate
            and roughness_ratio >= minimum_roughness_ratio
        )
        evidence.append(
            {
                "start_seconds": round(window.start_seconds, 6),
                "end_seconds": round(window.end_seconds, 6),
                "zero_crossing_rate": round(zero_crossing_rate, 4),
                "roughness_ratio": round(roughness_ratio, 4),
                "decision": (
                    "accepted_noise_like"
                    if is_noise_like
                    else "rejected_harmonic_content"
                ),
            }
        )
        if is_noise_like:
            accepted.append(window)
    return accepted, evidence


def _window_samples(
    samples: array,
    window: NoiseWindow,
    *,
    sample_rate: int,
    channels: int,
) -> array:
    start_frame = max(0, int(window.start_seconds * sample_rate))
    end_frame = max(start_frame, int(window.end_seconds * sample_rate))
    return samples[start_frame * channels : min(len(samples), end_frame * channels)]


def apply_adaptive_breath_cleanup(
    samples: array,
    *,
    windows: list[NoiseWindow],
    sample_rate: int,
    channels: int,
    max_attenuation_db: float,
    target_margin_db: float,
    context_ms: float,
    fade_ms: float,
    target_dbfs_override: float | None = None,
    absolute_floor_dbfs: float | None = None,
) -> tuple[array, list[dict[str, Any]]]:
    cleaned = array("h", samples)
    details: list[dict[str, Any]] = []
    context_seconds = max(0.0, context_ms / 1000.0)
    for window in merge_noise_windows(windows, max_gap_seconds=0.008):
        window_samples = _window_samples(
            samples,
            window,
            sample_rate=sample_rate,
            channels=channels,
        )
        pre_context = _window_samples(
            samples,
            NoiseWindow(
                max(0.0, window.start_seconds - context_seconds),
                window.start_seconds,
            ),
            sample_rate=sample_rate,
            channels=channels,
        )
        post_context = _window_samples(
            samples,
            NoiseWindow(
                window.end_seconds,
                window.end_seconds + context_seconds,
            ),
            sample_rate=sample_rate,
            channels=channels,
        )
        context_levels: list[float] = []
        if pre_context:
            context_levels.append(_amplitude_to_dbfs(_compute_rms(pre_context)))
        if post_context:
            context_levels.append(_amplitude_to_dbfs(_compute_rms(post_context)))
        # Prefer the quieter side so speech bleed immediately before an inhale
        # does not raise the adaptive target and skip soft but audible breaths.
        context_dbfs = min(context_levels) if context_levels else -120.0
        window_dbfs = _amplitude_to_dbfs(_compute_rms(window_samples))
        relative_target = context_dbfs + float(target_margin_db)
        if target_dbfs_override is not None:
            target_dbfs = float(target_dbfs_override)
        elif absolute_floor_dbfs is not None:
            target_dbfs = min(relative_target, float(absolute_floor_dbfs))
        else:
            target_dbfs = relative_target
        requested_db = max(
            0.0,
            min(float(max_attenuation_db), window_dbfs - target_dbfs),
        )
        if requested_db > 0.0:
            duck_samples_for_windows(
                cleaned,
                sample_rate=sample_rate,
                windows=[window],
                floor_gain=attenuation_db_to_gain(requested_db),
                fade_ms=fade_ms,
                channels=channels,
            )
        details.append(
            {
                "start_seconds": round(window.start_seconds, 6),
                "end_seconds": round(window.end_seconds, 6),
                "window_dbfs": round(window_dbfs, 3),
                "context_dbfs": round(context_dbfs, 3),
                "target_dbfs": round(target_dbfs, 3),
                "relative_target_dbfs": round(relative_target, 3),
                "absolute_floor_dbfs": (
                    None
                    if absolute_floor_dbfs is None
                    else round(float(absolute_floor_dbfs), 3)
                ),
                "requested_attenuation_db": round(requested_db, 3),
            }
        )
    return cleaned, details


def detect_first_speech_onset_seconds(
    samples: array,
    *,
    sample_rate: int,
    channels: int = 1,
    threshold_dbfs: float = -35.0,
    min_hold_ms: float = 50.0,
    frame_ms: float = 10.0,
) -> float | None:
    """Return the first sustained active-speech onset in seconds."""
    if sample_rate <= 0 or channels <= 0 or not samples:
        return None
    frame = max(1, int(sample_rate * (frame_ms / 1000.0)))
    hold_frames = max(1, int(math.ceil(min_hold_ms / max(frame_ms, 1.0))))
    total_frames = len(samples) // channels
    run = 0
    run_start_frame = 0
    for start_frame in range(0, max(0, total_frames - frame + 1), frame):
        chunk = samples[
            start_frame * channels : (start_frame + frame) * channels
        ]
        if not chunk:
            continue
        level = _amplitude_to_dbfs(_compute_rms(chunk))
        if level >= threshold_dbfs:
            if run == 0:
                run_start_frame = start_frame
            run += 1
            if run >= hold_frames:
                return run_start_frame / float(sample_rate)
        else:
            run = 0
    return None


def seal_leading_pre_speech_noise(
    reference_samples: array,
    processed_samples: array,
    *,
    sample_rate: int,
    channels: int,
    hold_pad_ms: float = 45.0,
    silence_floor_dbfs: float = -96.0,
    fade_ms: float = 12.0,
    max_attenuation_db: float = 60.0,
    source_quiet_dbfs: float = -40.0,
    processed_audible_dbfs: float = -55.0,
    max_boost_db: float = 12.0,
    min_duration_ms: float = 40.0,
    source_analysis: dict[str, Any] | None = None,
) -> tuple[array, dict[str, Any]]:
    """Duck only evidenced leading noise boosted above its source level.

    A source-speech onset alone does not authorize muting the whole prefix.
    Candidates must be quiet in the immutable source, audibly raised in the
    processed media, and clipped against the source active-speech guard.
    """
    reference_mono = _downmix_interleaved_samples(reference_samples, channels)
    processed_mono = _downmix_interleaved_samples(processed_samples, channels)
    speech_onset_seconds = detect_first_speech_onset_seconds(
        reference_mono,
        sample_rate=sample_rate,
        channels=1,
        threshold_dbfs=-35.0,
        min_hold_ms=50.0,
    )
    report: dict[str, Any] = {
        "status": "SKIP",
        "speech_onset_seconds": (
            None if speech_onset_seconds is None else round(float(speech_onset_seconds), 6)
        ),
        "sealed_end_seconds": None,
        "hold_pad_ms": round(float(hold_pad_ms), 3),
        "silence_floor_dbfs": round(float(silence_floor_dbfs), 3),
        "guard_exclusion_allowed": False,
        "candidate_windows": [],
        "applied_windows": [],
        "rejected_windows": [],
        "details": [],
    }
    if speech_onset_seconds is None:
        report["reason"] = "speech_onset_unavailable"
        return array("h", processed_samples), report
    candidates = detect_pre_speech_soft_noise_boost_windows(
        reference_mono,
        processed_mono,
        sample_rate=sample_rate,
        source_quiet_dbfs=source_quiet_dbfs,
        processed_audible_dbfs=processed_audible_dbfs,
        max_boost_db=max_boost_db,
        hold_pad_ms=hold_pad_ms,
        min_duration_ms=min_duration_ms,
    )
    report["candidate_windows"] = candidates
    if not candidates:
        report["reason"] = "no_boosted_pre_speech_noise"
        return array("h", processed_samples), report
    resolved_source_analysis = source_analysis or analyze_pcm16_samples(
        reference_mono,
        sample_rate=sample_rate,
    )
    safe_windows: list[NoiseWindow] = []
    for item in candidates:
        candidate = NoiseWindow(
            start_seconds=float(item["start_seconds"]),
            end_seconds=float(item["end_seconds"]),
        )
        safe_window = _clip_window_before_active_speech(
            candidate,
            resolved_source_analysis,
        )
        if safe_window is None:
            report["rejected_windows"].append(
                {
                    "start_seconds": candidate.start_seconds,
                    "end_seconds": candidate.end_seconds,
                    "reason": "source_active_speech_protected",
                }
            )
            continue
        safe_windows.append(safe_window)
    if not safe_windows:
        report["reason"] = "all_candidates_rejected_by_speech_guard"
        return array("h", processed_samples), report
    cleaned, details = apply_adaptive_breath_cleanup(
        processed_samples,
        windows=safe_windows,
        sample_rate=sample_rate,
        channels=channels,
        max_attenuation_db=float(max_attenuation_db),
        target_margin_db=0.0,
        context_ms=0.0,
        fade_ms=float(fade_ms),
        target_dbfs_override=float(silence_floor_dbfs),
    )
    if cleaned == processed_samples:
        report["reason"] = "no_attenuation_required"
        return cleaned, report
    report["status"] = "PASS"
    report["sealed_end_seconds"] = round(
        max(window.end_seconds for window in safe_windows),
        6,
    )
    report["applied_windows"] = [
        {
            "start_seconds": window.start_seconds,
            "end_seconds": window.end_seconds,
        }
        for window in safe_windows
    ]
    report["details"] = details
    return cleaned, report


def detect_pre_speech_soft_noise_boost_windows(
    reference_samples: array,
    processed_samples: array,
    *,
    sample_rate: int,
    frame_ms: float = 20.0,
    speech_threshold_dbfs: float = -35.0,
    source_quiet_dbfs: float = -40.0,
    processed_audible_dbfs: float = -55.0,
    max_boost_db: float = 12.0,
    hold_pad_ms: float = 45.0,
    min_duration_ms: float = 40.0,
) -> list[dict[str, Any]]:
    """Flag pre-speech regions where mastering boosted soft source noise."""
    onset = detect_first_speech_onset_seconds(
        reference_samples,
        sample_rate=sample_rate,
        channels=1,
        threshold_dbfs=speech_threshold_dbfs,
        min_hold_ms=50.0,
        frame_ms=10.0,
    )
    if onset is None:
        return []
    limit_seconds = max(0.0, float(onset) - (float(hold_pad_ms) / 1000.0))
    if limit_seconds <= 0.05:
        return []
    frame = max(1, int(sample_rate * (frame_ms / 1000.0)))
    limit_index = int(limit_seconds * sample_rate)
    mask: list[bool] = []
    for start in range(0, max(0, limit_index - frame + 1), frame):
        ref = reference_samples[start : start + frame]
        proc = processed_samples[start : start + frame]
        if not ref or not proc:
            mask.append(False)
            continue
        ref_db = _amplitude_to_dbfs(_compute_rms(ref))
        proc_db = _amplitude_to_dbfs(_compute_rms(proc))
        boost = proc_db - ref_db
        mask.append(
            ref_db <= source_quiet_dbfs
            and proc_db >= processed_audible_dbfs
            and boost >= max_boost_db
        )
    windows: list[dict[str, Any]] = []
    start_index: int | None = None
    for index, flagged in enumerate(mask + [False]):
        if flagged and start_index is None:
            start_index = index
        elif not flagged and start_index is not None:
            duration_ms = (index - start_index) * frame_ms
            if duration_ms >= min_duration_ms:
                windows.append(
                    {
                        "start_seconds": round(start_index * frame_ms / 1000.0, 3),
                        "end_seconds": round(index * frame_ms / 1000.0, 3),
                        "duration_ms": round(duration_ms, 1),
                    }
                )
            start_index = None
    return windows


def match_residual_breath_windows(
    *,
    parent_windows: list[NoiseWindow],
    residual_windows: list[NoiseWindow],
    min_duration_seconds: float = 0.03,
) -> list[NoiseWindow]:
    matched: list[NoiseWindow] = []
    for parent in parent_windows:
        for residual in residual_windows:
            start = max(parent.start_seconds, residual.start_seconds)
            end = min(parent.end_seconds, residual.end_seconds)
            if end - start >= min_duration_seconds:
                matched.append(NoiseWindow(start, end))
    return merge_noise_windows(matched, max_gap_seconds=0.008)


def trim_noise_windows_for_assessment(
    windows: list[NoiseWindow],
    *,
    edge_seconds: float,
    minimum_duration_seconds: float = 0.02,
) -> list[NoiseWindow]:
    trimmed: list[NoiseWindow] = []
    for window in windows:
        start = window.start_seconds + max(0.0, edge_seconds)
        end = window.end_seconds - max(0.0, edge_seconds)
        if end - start >= minimum_duration_seconds:
            trimmed.append(NoiseWindow(start, end))
    return trimmed


def run_respiro_or_fallback_detection(
    *,
    audio_path: Path,
    ffmpeg_bin: str,
    respiro_repo: Path | None,
    respiro_weights: Path | None,
    python_executable: str,
    threshold: float,
    min_length_ms: int,
    fallback_config: dict[str, Any],
) -> RespiroDetectionResult:
    def build_fallback_result(
        *,
        command: list[str] | None,
        returncode: int | None,
        stdout: str,
        stderr: str,
        error: str | None,
    ) -> RespiroDetectionResult:
        fallback_windows = (
            detect_breath_onset_windows(
                audio_path,
                ffmpeg_bin=ffmpeg_bin,
                config=fallback_config,
            )
            if audio_path.exists()
            else []
        )
        return RespiroDetectionResult(
            windows=fallback_windows,
            mode="fallback",
            assets_present=True,
            attempted=True,
            succeeded=False,
            command=command,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            error=error,
        )

    assets_present = bool(
        respiro_repo and respiro_weights and respiro_repo.exists() and respiro_weights.exists()
    )
    if assets_present and respiro_repo and respiro_weights:
        duration_seconds = 0.0
        if audio_path.exists():
            try:
                duration_seconds = _read_wave_duration_seconds(audio_path)
            except (wave.Error, EOFError, FileNotFoundError):
                duration_seconds = 0.0
        segment_length_seconds = 120.0
        segment_overlap_seconds = 1.0
        segment_mode = duration_seconds > segment_length_seconds

        if not segment_mode:
            command = build_respiro_detect_command(
                audio_path=audio_path,
                python_executable=python_executable,
                repo_path=respiro_repo,
                weights_path=respiro_weights,
                threshold=threshold,
                min_length_ms=min_length_ms,
            )
            completed = run_command(command)
            if completed.returncode == 0:
                try:
                    payload = json.loads(completed.stdout.strip() or "{}")
                    intervals = payload.get("intervals", [])
                    return RespiroDetectionResult(
                        windows=[
                            NoiseWindow(
                                start_seconds=float(item["start_seconds"]),
                                end_seconds=float(item["end_seconds"]),
                            )
                            for item in intervals
                        ],
                        mode="respiro",
                        assets_present=True,
                        attempted=True,
                        succeeded=True,
                        command=command,
                        returncode=completed.returncode,
                        stdout=completed.stdout.strip(),
                        stderr=completed.stderr.strip(),
                    )
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    error = f"Failed to parse Respiro-en output: {exc}"
            else:
                error = completed.stderr.strip() or completed.stdout.strip() or "Respiro-en command failed"
            return build_fallback_result(
                command=command,
                returncode=completed.returncode,
                stdout=completed.stdout.strip(),
                stderr=completed.stderr.strip(),
                error=error,
            )

        ffmpeg = ensure_tool(ffmpeg_bin)
        segment_windows: list[NoiseWindow] = []
        segment_commands: list[list[str]] = []
        segment_stdout: list[str] = []
        segment_stderr: list[str] = []
        segment_count = 0

        with tempfile.TemporaryDirectory(prefix="respiro-segments-") as tmp_dir:
            temp_root = Path(tmp_dir)
            start_seconds = 0.0
            while start_seconds < duration_seconds:
                segment_start = start_seconds
                segment_duration = min(segment_length_seconds, duration_seconds - segment_start)
                segment_wav = temp_root / f"segment_{segment_count:04d}.wav"
                extract_command = [
                    ffmpeg,
                    "-y",
                    "-hide_banner",
                    "-nostdin",
                    "-ss",
                    f"{segment_start:.3f}",
                    "-t",
                    f"{segment_duration:.3f}",
                    "-i",
                    str(audio_path),
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_s16le",
                    str(segment_wav),
                ]
                extract_completed = run_command(extract_command)
                segment_commands.append(extract_command)
                if extract_completed.returncode != 0:
                    error = extract_completed.stderr.strip() or extract_completed.stdout.strip() or "Failed to extract Respiro-en segment"
                    return build_fallback_result(
                        command=extract_command,
                        returncode=extract_completed.returncode,
                        stdout=extract_completed.stdout.strip(),
                        stderr=extract_completed.stderr.strip(),
                        error=error,
                    )

                detect_command = build_respiro_detect_command(
                    audio_path=segment_wav,
                    python_executable=python_executable,
                    repo_path=respiro_repo,
                    weights_path=respiro_weights,
                    threshold=threshold,
                    min_length_ms=min_length_ms,
                )
                detect_completed = run_command(detect_command)
                segment_commands.append(detect_command)
                segment_stdout.append(detect_completed.stdout.strip())
                segment_stderr.append(detect_completed.stderr.strip())
                if detect_completed.returncode != 0:
                    error = detect_completed.stderr.strip() or detect_completed.stdout.strip() or "Respiro-en segment command failed"
                    return build_fallback_result(
                        command=detect_command,
                        returncode=detect_completed.returncode,
                        stdout=detect_completed.stdout.strip(),
                        stderr=detect_completed.stderr.strip(),
                        error=error,
                    )
                try:
                    payload = json.loads(detect_completed.stdout.strip() or "{}")
                    intervals = payload.get("intervals", [])
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    return build_fallback_result(
                        command=detect_command,
                        returncode=detect_completed.returncode,
                        stdout=detect_completed.stdout.strip(),
                        stderr=detect_completed.stderr.strip(),
                        error=f"Failed to parse Respiro-en segment output: {exc}",
                    )

                for item in intervals:
                    adjusted_start = segment_start + float(item["start_seconds"])
                    adjusted_end = segment_start + float(item["end_seconds"])
                    segment_windows.append(
                        NoiseWindow(
                            start_seconds=adjusted_start,
                            end_seconds=adjusted_end,
                        )
                    )

                segment_count += 1
                if segment_start + segment_duration >= duration_seconds:
                    break
                start_seconds += max(1.0, segment_length_seconds - segment_overlap_seconds)

        merged_windows = merge_noise_windows(segment_windows, max_gap_seconds=0.05)
        return RespiroDetectionResult(
            windows=merged_windows,
            mode="respiro",
            assets_present=True,
            attempted=True,
            succeeded=True,
            command=segment_commands[-1] if segment_commands else None,
            returncode=0,
            stdout="\n".join(line for line in segment_stdout if line),
            stderr="\n".join(line for line in segment_stderr if line),
            error=None,
        )
    fallback_windows = (
        detect_breath_onset_windows(
            audio_path,
            ffmpeg_bin=ffmpeg_bin,
            config=fallback_config,
        )
        if audio_path.exists()
        else []
    )
    return RespiroDetectionResult(
        windows=fallback_windows,
        mode="fallback",
        assets_present=False,
        attempted=False,
        succeeded=False,
        error="Respiro-en assets not configured or missing"
        if (respiro_repo or respiro_weights)
        else None,
    )


def discover_media_files(input_path: Path, recursive: bool = False) -> list[Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_MEDIA_EXTENSIONS:
            raise ValueError(f"Unsupported media file: {input_path}")
        return [input_path]

    pattern = "**/*" if recursive else "*"
    files = [
        path
        for path in input_path.glob(pattern)
        if path.is_file() and path.suffix.lower() in SUPPORTED_MEDIA_EXTENSIONS
    ]
    return sorted(files)


def ensure_tool(tool_name: str) -> str:
    if any(separator in tool_name for separator in ("/", "\\")):
        path = Path(tool_name)
        if not path.exists():
            raise RuntimeError(f"Required tool not found: {tool_name}")
        return str(path)

    resolved = shutil.which(tool_name)
    if resolved is None:
        raise RuntimeError(f"Required tool not found on PATH: {tool_name}")
    return resolved


def run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _run_recorded_command(
    command: list[str],
    executed: list[dict[str, Any]],
) -> subprocess.CompletedProcess[str]:
    completed = run_command(command)
    executed.append(
        {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    )
    if completed.returncode != 0:
        raise RuntimeError(
            completed.stderr.strip()
            or completed.stdout.strip()
            or "command failed"
        )
    return completed


def ffprobe_media(path: Path, ffprobe_bin: str) -> dict[str, Any]:
    ffprobe = ensure_tool(ffprobe_bin)
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "stream=codec_name,codec_type,sample_rate,channels,bit_rate:format=duration,size,bit_rate",
        "-of",
        "json",
        str(path),
    ]
    completed = run_command(cmd)
    if completed.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {completed.stderr.strip()}")

    payload = json.loads(completed.stdout or "{}")
    streams = payload.get("streams", [])
    audio_stream = next((item for item in streams if item.get("codec_type") == "audio"), streams[0] if streams else {})
    format_block = payload.get("format", {})

    def parse_number(value: Any) -> float | int | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return value
        try:
            if "." in str(value):
                return float(value)
            return int(value)
        except ValueError:
            return None

    return {
        "path": str(path),
        "codec": audio_stream.get("codec_name"),
        "sample_rate": parse_number(audio_stream.get("sample_rate")),
        "channels": parse_number(audio_stream.get("channels")),
        "stream_bit_rate": parse_number(audio_stream.get("bit_rate")),
        "format_bit_rate": parse_number(format_block.get("bit_rate")),
        "duration_seconds": parse_number(format_block.get("duration")),
        "size_bytes": parse_number(format_block.get("size")),
    }


def build_output_layout(*, input_path: Path, output_root: Path, run_slug: str) -> OutputLayout:
    file_base_name = input_path.stem
    job_name = f"{run_slug}_{file_base_name}"
    job_dir = output_root / job_name
    preprocess_dir = job_dir / "audio_preprocess"
    transcript_dir = job_dir / "transcript_ready"
    return OutputLayout(
        job_name=job_name,
        job_dir=job_dir,
        preprocess_dir=preprocess_dir,
        transcript_dir=transcript_dir,
        raw_wav=preprocess_dir / f"{file_base_name}_raw.wav",
        breath_wav=preprocess_dir / f"{file_base_name}_breath.wav",
        denoised_wav=preprocess_dir / f"{file_base_name}_df.wav",
        noise_sample_wav=preprocess_dir / f"{file_base_name}_noise_sample.wav",
        clean_wav=preprocess_dir / f"{file_base_name}_clean.wav",
        transcript_mp3=transcript_dir / f"{file_base_name}_transcript.mp3",
        report_json=preprocess_dir / "audio_process_report.json",
        report_md=preprocess_dir / "audio_process_report.md",
        deepfilternet_dir=preprocess_dir / "deepfilternet_out",
    )


def resolve_processing_format(
    source_metadata: dict[str, Any] | None,
    preset: dict[str, Any],
) -> dict[str, int | str]:
    extract = preset["extract"]
    source_sample_rate = (source_metadata or {}).get("sample_rate")
    source_channels = (source_metadata or {}).get("channels")
    sample_rate = int(source_sample_rate) if source_sample_rate else int(extract["sample_rate"])
    channels = int(source_channels) if source_channels else int(extract["channels"])
    if sample_rate <= 0 or channels <= 0:
        raise ValueError("source audio format must contain positive sample_rate and channels")
    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "pcm_codec": str(extract["pcm_codec"]),
    }


def build_ffmpeg_extract_command(
    *,
    input_path: Path,
    raw_wav: Path,
    preset: dict[str, Any],
    ffmpeg_bin: str,
    processing_format: dict[str, int | str] | None = None,
) -> list[str]:
    extract = preset["extract"]
    output_format = processing_format or resolve_processing_format(None, preset)
    return [
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-nostdin",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        str(output_format["channels"]),
        "-ar",
        str(output_format["sample_rate"]),
        "-c:a",
        str(output_format.get("pcm_codec", extract["pcm_codec"])),
        str(raw_wav),
    ]


def build_deepfilternet_command(*, raw_wav: Path, output_dir: Path, preset: dict[str, Any], python_executable: str) -> list[str]:
    config = get_pipeline_stage(preset, "deepfilternet")
    command = [
        python_executable,
        "-m",
        config["module"],
        "--output-dir",
        str(output_dir),
    ]
    if config.get("post_filter", False):
        command.append("--pf")
    command.append(str(raw_wav))
    return command


def parse_noise_window(raw_value: str) -> NoiseWindow:
    parts = raw_value.split(":", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid noise window: {raw_value}")

    try:
        start_seconds = float(parts[0])
        end_seconds = float(parts[1])
    except ValueError as exc:
        raise ValueError(f"Invalid noise window: {raw_value}") from exc

    if start_seconds < 0 or end_seconds <= start_seconds:
        raise ValueError(f"Invalid noise window: {raw_value}")

    return NoiseWindow(start_seconds=start_seconds, end_seconds=end_seconds)


def _format_filter_float(value: float) -> str:
    return format(value, ".6f").rstrip("0").rstrip(".")


def build_ffmpeg_noise_sample_command(
    *,
    source_wav: Path,
    noise_sample_wav: Path,
    noise_windows: list[NoiseWindow],
    preset: dict[str, Any],
    ffmpeg_bin: str,
    processing_format: dict[str, int | str] | None = None,
) -> list[str]:
    if not noise_windows:
        raise ValueError("At least one noise window is required to build a noise sample command")

    segments: list[str] = []
    concat_inputs: list[str] = []
    for index, window in enumerate(noise_windows):
        label = f"s{index}"
        concat_inputs.append(f"[{label}]")
        segments.append(
            "[0:a]atrim="
            f"start={_format_filter_float(window.start_seconds)}:"
            f"end={_format_filter_float(window.end_seconds)},"
            f"asetpts=PTS-STARTPTS[{label}]"
        )

    segments.append(f"{''.join(concat_inputs)}concat=n={len(noise_windows)}:v=0:a=1[outa]")
    filter_complex = ";".join(segments)
    output_format = processing_format or resolve_processing_format(None, preset)
    return [
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-nostdin",
        "-i",
        str(source_wav),
        "-filter_complex",
        filter_complex,
        "-map",
        "[outa]",
        "-ar",
        str(output_format["sample_rate"]),
        "-ac",
        str(output_format["channels"]),
        "-c:a",
        str(output_format.get("pcm_codec", preset["extract"]["pcm_codec"])),
        str(noise_sample_wav),
    ]


def build_mastering_filter_chain(
    preset: dict[str, Any],
    *,
    for_noise_print: bool = False,
    include_secondary_denoise: bool = True,
    include_loudnorm: bool = True,
    include_tonal_shaping: bool = True,
    include_declick: bool = True,
    include_nonlinear_denoise: bool = True,
    include_deesser: bool = True,
    include_gate: bool = True,
    include_compressor: bool = True,
    include_speech_norm: bool = True,
) -> str:
    filters = preset["filters"]
    parts: list[str] = []

    highpass_hz = filters.get("highpass_hz")
    if include_tonal_shaping and highpass_hz:
        parts.append(f"highpass=f={highpass_hz}")

    lowpass_hz = filters.get("lowpass_hz")
    if include_tonal_shaping and lowpass_hz:
        parts.append(f"lowpass=f={lowpass_hz}")

    equalizer = filters.get("equalizer", {})
    if include_tonal_shaping and equalizer.get("enabled"):
        for band in equalizer.get("bands", []):
            parts.append(
                "equalizer="
                f"f={band['frequency_hz']}:"
                f"t={band.get('width_type', 'q')}:"
                f"w={band.get('width', 1.0)}:"
                f"g={band['gain_db']}"
            )

    declick = filters.get("declick", {})
    if include_declick and declick.get("enabled"):
        parts.append(
            "adeclick="
            f"w={declick['window']}:o={declick['overlap']}:a={declick['arorder']}:t={declick['threshold']}:b={declick['burst']}"
        )

    secondary_denoise = filters.get("secondary_denoise", {})
    if include_secondary_denoise and secondary_denoise.get("enabled"):
        part = f"afftdn=nr={secondary_denoise['nr']}:nf={secondary_denoise['nf']}"
        residual_floor = secondary_denoise.get("residual_floor")
        if residual_floor is not None:
            part += f":rf={residual_floor}"
        adaptivity = secondary_denoise.get("adaptivity")
        if adaptivity is not None:
            part += f":ad={adaptivity}"
        floor_offset = secondary_denoise.get("floor_offset")
        if floor_offset is not None:
            part += f":fo={floor_offset}"
        gain_smooth = secondary_denoise.get("gain_smooth")
        if gain_smooth is not None:
            part += f":gs={gain_smooth}"
        noise_link = secondary_denoise.get("noise_link")
        if noise_link:
            part += f":nl={noise_link}"
        if for_noise_print:
            part += ":sn=start"
        elif secondary_denoise.get("tracking"):
            part += ":tn=1"
        parts.append(part)

    nonlinear_denoise = filters.get("nonlinear_denoise", {})
    if include_nonlinear_denoise and nonlinear_denoise.get("enabled"):
        parts.append(
            "anlmdn="
            f"s={nonlinear_denoise['strength']}:p={nonlinear_denoise['patch']}:r={nonlinear_denoise['research']}:m={nonlinear_denoise['smooth']}"
        )

    deesser = filters.get("deesser", {})
    if include_deesser and deesser.get("enabled"):
        parts.append(
            "deesser="
            f"i={deesser['intensity']}:m={deesser['max_deessing']}:f={deesser['frequency']}"
        )

    gate = filters.get("gate", {})
    if include_gate and gate.get("enabled"):
        gate_parts = [
            "agate="
            f"threshold={gate['threshold']}:ratio={gate['ratio']}:"
            f"attack={gate['attack_ms']}:release={gate['release_ms']}"
        ]
        makeup_db = gate.get("makeup_db")
        if makeup_db is not None and float(makeup_db) >= 1:
            gate_parts.append(f":makeup={makeup_db}")
        parts.append("".join(gate_parts))

    compressor = filters.get("compressor", {})
    if include_compressor and compressor.get("enabled"):
        compressor_part = (
            "acompressor="
            f"threshold={compressor['threshold_db']}dB:ratio={compressor['ratio']}:"
            f"attack={compressor['attack_ms']}:release={compressor['release_ms']}"
        )
        makeup_db = compressor.get("makeup_db")
        if makeup_db is not None and float(makeup_db) >= 1:
            compressor_part += f":makeup={makeup_db}"
        parts.append(compressor_part)

    speech_norm = filters.get("speech_norm", {})
    if include_speech_norm and speech_norm.get("enabled"):
        parts.append(
            "speechnorm="
            f"e={speech_norm['expansion']}:r={speech_norm['raise']}:f={speech_norm['fall']}:t={speech_norm['threshold']}"
        )

    if include_loudnorm:
        loudnorm = filters["loudnorm"]
        parts.append(
            "loudnorm="
            f"I={loudnorm['target_i']}:LRA={loudnorm['target_lra']}:TP={loudnorm['target_tp']}:print_format=json"
        )
    return ",".join(parts)


def build_breath_ducking_filter_graph(
    *,
    input_label: str,
    output_label: str,
    preset: dict[str, Any],
    include_loudnorm: bool = True,
) -> str:
    filters = preset["filters"]
    breath_ducking = filters.get("breath_ducking", {})

    pre_duck_chain = build_mastering_filter_chain(
        preset,
        include_secondary_denoise=False,
        include_loudnorm=False,
        include_deesser=False,
        include_gate=False,
        include_compressor=False,
        include_speech_norm=False,
    )
    post_duck_chain = build_mastering_filter_chain(
        preset,
        include_secondary_denoise=False,
        include_tonal_shaping=False,
        include_declick=False,
        include_nonlinear_denoise=False,
        include_gate=False,
        include_loudnorm=include_loudnorm,
        include_speech_norm=False,
    )

    if not breath_ducking.get("enabled"):
        if pre_duck_chain and post_duck_chain:
            return f"[{input_label}]{pre_duck_chain},{post_duck_chain}[{output_label}]"
        if pre_duck_chain:
            return f"[{input_label}]{pre_duck_chain}[{output_label}]"
        if post_duck_chain:
            return f"[{input_label}]{post_duck_chain}[{output_label}]"
        return f"[{input_label}]anull[{output_label}]"

    detector_chain = ",".join(
        [
            f"highpass=f={breath_ducking['detector_highpass_hz']}",
            f"lowpass=f={breath_ducking['detector_lowpass_hz']}",
            "acompressor="
            f"threshold={breath_ducking['detector_threshold_db']}dB:"
            f"ratio={breath_ducking['detector_ratio']}:"
            f"attack={breath_ducking['detector_attack_ms']}:"
            f"release={breath_ducking['detector_release_ms']}:"
            f"makeup={breath_ducking['detector_makeup_db']}",
        ]
    )
    gate_filter = (
        "sidechaingate="
        f"threshold={breath_ducking['threshold']}:"
        f"ratio={breath_ducking['ratio']}:"
        f"attack={breath_ducking['attack_ms']}:"
        f"release={breath_ducking['release_ms']}:"
        f"range={breath_ducking['range']}:"
        f"detection={breath_ducking['detection']}:"
        f"link={breath_ducking['link']}:"
        f"level_sc={breath_ducking['level_sc']}"
    )

    source_label = input_label
    graph_parts: list[str] = []
    if pre_duck_chain:
        graph_parts.append(f"[{input_label}]{pre_duck_chain}[duck_pre]")
        source_label = "duck_pre"

    graph_parts.append(f"[{source_label}]asplit=2[duck_main][duck_sc]")
    graph_parts.append(f"[duck_sc]{detector_chain}[duck_ctl]")
    graph_parts.append(f"[duck_main][duck_ctl]{gate_filter}[ducked]")

    if post_duck_chain:
        graph_parts.append(f"[ducked]{post_duck_chain}[{output_label}]")
    else:
        graph_parts.append(f"[ducked]anull[{output_label}]")
    return ";".join(graph_parts)


def build_ffmpeg_finalize_commands(
    *,
    denoised_wav: Path,
    clean_wav: Path,
    transcript_mp3: Path,
    preset: dict[str, Any],
    ffmpeg_bin: str,
    noise_sample_wav: Path | None = None,
    noise_sample_duration: float | None = None,
    processing_format: dict[str, int | str] | None = None,
) -> list[list[str]]:
    transcript = preset["transcript_export"]
    output_format = processing_format or resolve_processing_format(None, preset)
    breath_ducking_enabled = preset.get("filters", {}).get("breath_ducking", {}).get("enabled", False)
    if noise_sample_wav is not None:
        if noise_sample_duration is None or noise_sample_duration <= 0:
            raise ValueError("noise_sample_duration must be provided when noise_sample_wav is used")
        sample_duration = _format_filter_float(noise_sample_duration)
        noise_print_chain = build_mastering_filter_chain(
            preset,
            for_noise_print=True,
            include_loudnorm=False,
            include_tonal_shaping=False,
            include_declick=False,
            include_nonlinear_denoise=False,
            include_deesser=False,
            include_gate=False,
            include_compressor=False,
            include_speech_norm=False,
        )
        if breath_ducking_enabled:
            voice_graph = build_breath_ducking_filter_graph(
                input_label="den",
                output_label="outa",
                preset=preset,
                include_loudnorm=True,
            )
            filter_complex = (
                "[0:a][1:a]concat=n=2:v=0:a=1[cat];"
                f"[cat]{noise_print_chain},atrim=start={sample_duration},asetpts=PTS-STARTPTS[den];"
                f"{voice_graph}"
            )
        else:
            voice_chain = build_mastering_filter_chain(
                preset,
                include_secondary_denoise=False,
            )
            filter_complex = (
                "[0:a][1:a]concat=n=2:v=0:a=1[cat];"
                f"[cat]{noise_print_chain},atrim=start={sample_duration},asetpts=PTS-STARTPTS[outa]"
            )
            if voice_chain:
                filter_complex = (
                    "[0:a][1:a]concat=n=2:v=0:a=1[cat];"
                    f"[cat]{noise_print_chain},atrim=start={sample_duration},asetpts=PTS-STARTPTS[den];"
                    f"[den]{voice_chain}[outa]"
                )
        clean_command = [
            ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-nostdin",
            "-i",
            str(denoised_wav),
            "-i",
            str(noise_sample_wav),
            "-filter_complex",
            filter_complex,
            "-map",
            "[outa]",
            "-ar",
            str(output_format["sample_rate"]),
            "-ac",
            str(output_format["channels"]),
            "-c:a",
            str(output_format.get("pcm_codec", preset["extract"]["pcm_codec"])),
            str(clean_wav),
        ]
    elif breath_ducking_enabled:
        filter_complex = build_breath_ducking_filter_graph(
            input_label="0:a",
            output_label="outa",
            preset=preset,
            include_loudnorm=True,
        )
        clean_command = [
            ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-nostdin",
            "-i",
            str(denoised_wav),
            "-filter_complex",
            filter_complex,
            "-map",
            "[outa]",
            "-ar",
            str(output_format["sample_rate"]),
            "-ac",
            str(output_format["channels"]),
            "-c:a",
            str(output_format.get("pcm_codec", preset["extract"]["pcm_codec"])),
            str(clean_wav),
        ]
    else:
        clean_command = [
            ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-nostdin",
            "-i",
            str(denoised_wav),
            "-af",
            build_mastering_filter_chain(preset),
            "-ar",
            str(output_format["sample_rate"]),
            "-ac",
            str(output_format["channels"]),
            "-c:a",
            str(output_format.get("pcm_codec", preset["extract"]["pcm_codec"])),
            str(clean_wav),
        ]

    return [
        clean_command,
        build_mp3_export_command(
            clean_wav,
            transcript_mp3,
            ffmpeg_bin=ffmpeg_bin,
            codec=str(transcript["codec"]),
            bitrate=str(transcript["bitrate"]),
        ),
    ]


def extract_loudnorm_summary(stderr: str) -> dict[str, Any] | None:
    matches = JSON_BLOCK_PATTERN.findall(stderr)
    if not matches:
        return None
    try:
        return json.loads(matches[-1])
    except json.JSONDecodeError:
        return None


def detect_silence_candidates(
    audio_path: Path,
    *,
    ffmpeg_bin: str,
    threshold_db: float,
    min_duration: float,
) -> list[dict[str, float]]:
    ffmpeg = ensure_tool(ffmpeg_bin)
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-i",
        str(audio_path),
        "-af",
        f"silencedetect=n={threshold_db}dB:d={min_duration}",
        "-f",
        "null",
        "-",
    ]
    completed = run_command(cmd)
    stderr = completed.stderr

    starts = [float(match.group(1)) for match in SILENCE_START_PATTERN.finditer(stderr)]
    ends = [
        (float(match.group(1)), float(match.group(2)))
        for match in SILENCE_END_PATTERN.finditer(stderr)
    ]

    candidates: list[dict[str, float]] = []
    for index, (silence_end, silence_duration) in enumerate(ends):
        silence_start = starts[index] if index < len(starts) else max(0.0, silence_end - silence_duration)
        candidates.append(
            {
                "start_seconds": round(silence_start, 3),
                "end_seconds": round(silence_end, 3),
                "duration_seconds": round(silence_duration, 3),
            }
        )
    return candidates


def merge_noise_windows(windows: list[NoiseWindow], max_gap_seconds: float = 0.01) -> list[NoiseWindow]:
    if not windows:
        return []

    merged: list[NoiseWindow] = []
    for window in sorted(windows, key=lambda item: (item.start_seconds, item.end_seconds)):
        if not merged:
            merged.append(window)
            continue
        previous = merged[-1]
        if window.start_seconds <= previous.end_seconds + max_gap_seconds:
            merged[-1] = NoiseWindow(
                start_seconds=previous.start_seconds,
                end_seconds=max(previous.end_seconds, window.end_seconds),
            )
        else:
            merged.append(window)
    return merged


def infer_breath_onset_windows(
    low_threshold_silences: list[dict[str, float]],
    high_threshold_silences: list[dict[str, float]],
    *,
    min_breath_seconds: float,
    max_breath_seconds: float,
    pre_roll_seconds: float,
) -> list[NoiseWindow]:
    candidates: list[NoiseWindow] = []
    for low_item in low_threshold_silences:
        low_end = float(low_item["end_seconds"])
        low_start = float(low_item["start_seconds"])
        for high_item in high_threshold_silences:
            high_start = float(high_item["start_seconds"])
            high_end = float(high_item["end_seconds"])
            if high_start - 0.01 <= low_end <= high_end and high_end > low_end:
                breath_duration = high_end - low_end
                if min_breath_seconds <= breath_duration <= max_breath_seconds:
                    candidates.append(
                        NoiseWindow(
                            start_seconds=max(low_start, low_end - pre_roll_seconds),
                            end_seconds=high_end,
                        )
                    )
                break
    return merge_noise_windows(candidates)


def _compute_rms(samples: array) -> float:
    if not samples:
        return 0.0
    total = 0.0
    for value in samples:
        total += float(value) * float(value)
    return (total / len(samples)) ** 0.5


def infer_breath_window_from_frame_rms(
    frame_rms: list[float],
    *,
    silence_floor_rms: float,
    hop_seconds: float,
    min_breath_seconds: float,
    max_breath_seconds: float,
    speech_start_ratio: float,
    breath_over_noise_ratio: float,
    speech_over_breath_ratio: float,
    speech_confirm_frames: int,
) -> tuple[float, float] | None:
    if not frame_rms:
        return None

    peak_rms = max(frame_rms)
    if peak_rms <= 0:
        return None

    threshold = max(peak_rms * speech_start_ratio, silence_floor_rms * breath_over_noise_ratio * 1.25)
    onset_index: int | None = None
    for index in range(len(frame_rms)):
        confirm_slice = frame_rms[index : index + speech_confirm_frames]
        if len(confirm_slice) < speech_confirm_frames:
            break
        if sum(confirm_slice) / len(confirm_slice) >= threshold:
            onset_index = index
            break

    if onset_index is None or onset_index == 0:
        return None

    breath_duration = onset_index * hop_seconds
    if breath_duration < min_breath_seconds or breath_duration > max_breath_seconds:
        return None

    breath_avg = sum(frame_rms[:onset_index]) / float(onset_index)
    speech_slice = frame_rms[onset_index : onset_index + speech_confirm_frames]
    speech_avg = sum(speech_slice) / float(len(speech_slice))
    if breath_avg <= silence_floor_rms * breath_over_noise_ratio:
        return None
    if speech_avg <= breath_avg * speech_over_breath_ratio:
        return None
    return (0.0, breath_duration)


def infer_breath_windows_from_silence_edges(
    samples: array,
    *,
    sample_rate: int,
    silences: list[dict[str, float]],
    config: dict[str, Any],
) -> list[NoiseWindow]:
    hop_samples = max(1, int(sample_rate * (float(config["analysis_hop_ms"]) / 1000.0)))
    scan_samples = max(hop_samples, int(sample_rate * (float(config["analysis_scan_ms"]) / 1000.0)))
    noise_floor_samples = max(hop_samples, int(sample_rate * (float(config["noise_floor_ms"]) / 1000.0)))

    candidates: list[NoiseWindow] = []
    for silence in silences:
        silence_end_seconds = float(silence["end_seconds"])
        silence_end_index = min(len(samples), max(0, int(silence_end_seconds * sample_rate)))
        if silence_end_index >= len(samples):
            continue

        noise_floor_start = max(0, silence_end_index - noise_floor_samples)
        silence_floor_rms = _compute_rms(samples[noise_floor_start:silence_end_index])

        frame_rms: list[float] = []
        scan_end = min(len(samples), silence_end_index + scan_samples)
        index = silence_end_index
        while index + hop_samples <= scan_end:
            frame_rms.append(_compute_rms(samples[index : index + hop_samples]))
            index += hop_samples

        inferred = infer_breath_window_from_frame_rms(
            frame_rms,
            silence_floor_rms=silence_floor_rms,
            hop_seconds=hop_samples / float(sample_rate),
            min_breath_seconds=float(config["min_breath_ms"]) / 1000.0,
            max_breath_seconds=float(config["max_breath_ms"]) / 1000.0,
            speech_start_ratio=float(config["speech_start_ratio"]),
            breath_over_noise_ratio=float(config["breath_over_noise_ratio"]),
            speech_over_breath_ratio=float(config["speech_over_breath_ratio"]),
            speech_confirm_frames=int(config["speech_confirm_frames"]),
        )
        if inferred is None:
            continue

        _, inferred_end_offset = inferred
        candidates.append(
            NoiseWindow(
                start_seconds=max(0.0, silence_end_seconds - (float(config["pre_roll_ms"]) / 1000.0)),
                end_seconds=silence_end_seconds + inferred_end_offset,
            )
        )
    return merge_noise_windows(candidates)


def detect_breath_onset_windows(
    audio_path: Path,
    *,
    ffmpeg_bin: str,
    config: dict[str, Any],
) -> list[NoiseWindow]:
    low_threshold_silences = detect_silence_candidates(
        audio_path,
        ffmpeg_bin=ffmpeg_bin,
        threshold_db=float(config["low_threshold_db"]),
        min_duration=float(config["silence_min_duration"]),
    )
    high_threshold_silences = detect_silence_candidates(
        audio_path,
        ffmpeg_bin=ffmpeg_bin,
        threshold_db=float(config["high_threshold_db"]),
        min_duration=float(config["silence_min_duration"]),
    )
    seed_windows = infer_breath_onset_windows(
        low_threshold_silences,
        high_threshold_silences,
        min_breath_seconds=float(config["min_breath_ms"]) / 1000.0,
        max_breath_seconds=float(config["max_breath_ms"]) / 1000.0,
        pre_roll_seconds=float(config["pre_roll_ms"]) / 1000.0,
    )

    with wave.open(str(audio_path), "rb") as reader:
        params = reader.getparams()
        if params.sampwidth != 2 or params.nchannels <= 0:
            raise ValueError("Breath onset detection expects 16-bit WAV audio")
        raw_frames = reader.readframes(params.nframes)
    samples = array("h")
    samples.frombytes(raw_frames)
    samples = _analysis_samples(params, samples)

    inferred_windows = infer_breath_windows_from_silence_edges(
        samples,
        sample_rate=params.framerate,
        silences=low_threshold_silences,
        config=config,
    )
    return merge_noise_windows([*seed_windows, *inferred_windows], max_gap_seconds=0.02)


def duck_samples_for_windows(
    samples: array,
    *,
    sample_rate: int,
    windows: list[NoiseWindow],
    floor_gain: float,
    fade_ms: float,
    channels: int = 1,
) -> None:
    if channels <= 0:
        raise ValueError("channels must be positive")
    merged_windows = merge_noise_windows(windows)
    fade_frames = max(0, int(sample_rate * (fade_ms / 1000.0)))
    total_frames = len(samples) // channels

    for window in merged_windows:
        start_frame = max(0, min(total_frames, int(window.start_seconds * sample_rate)))
        end_frame = max(start_frame, min(total_frames, int(window.end_seconds * sample_rate)))
        if end_frame <= start_frame:
            continue

        local_fade = min(fade_frames, max(0, (end_frame - start_frame) // 2))
        for frame_index in range(start_frame, end_frame):
            gain = floor_gain
            if local_fade > 0 and frame_index < start_frame + local_fade:
                progress = (frame_index - start_frame) / float(local_fade)
                gain = 1.0 - (1.0 - floor_gain) * progress
            elif local_fade > 0 and frame_index >= end_frame - local_fade:
                progress = (frame_index - (end_frame - local_fade)) / float(local_fade)
                gain = floor_gain + (1.0 - floor_gain) * progress
            frame_start = frame_index * channels
            for channel_index in range(channels):
                sample_index = frame_start + channel_index
                samples[sample_index] = int(samples[sample_index] * gain)


def duck_audio_file_in_place(
    audio_path: Path,
    *,
    windows: list[NoiseWindow],
    floor_gain: float,
    fade_ms: float,
) -> None:
    merged_windows = merge_noise_windows(windows)
    if not merged_windows:
        return

    with wave.open(str(audio_path), "rb") as reader:
        params = reader.getparams()
        if params.sampwidth != 2 or params.nchannels <= 0:
            raise ValueError("Breath ducking expects 16-bit WAV audio")
        raw_frames = reader.readframes(params.nframes)

    samples = array("h")
    samples.frombytes(raw_frames)
    duck_samples_for_windows(
        samples,
        sample_rate=params.framerate,
        windows=merged_windows,
        floor_gain=floor_gain,
        fade_ms=fade_ms,
        channels=params.nchannels,
    )

    temp_path = audio_path.with_name(f"{audio_path.stem}.breathduck.tmp.wav")
    with wave.open(str(temp_path), "wb") as writer:
        writer.setparams(params)
        writer.writeframes(samples.tobytes())
    temp_path.replace(audio_path)


def _amplitude_to_dbfs(amplitude: float) -> float:
    if amplitude <= 0:
        return -120.0
    return 20.0 * math.log10(amplitude / 32767.0)


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return -120.0
    ordered = sorted(values)
    position = max(0.0, min(1.0, fraction)) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + ((ordered[upper] - ordered[lower]) * weight)


def analyze_pcm16_samples(
    samples: array,
    *,
    sample_rate: int,
    frame_ms: float = 20.0,
) -> dict[str, Any]:
    frame_size = max(1, int(sample_rate * (frame_ms / 1000.0)))
    frame_levels: list[float] = []
    frame_peaks: list[float] = []
    clipped_sample_count = 0
    total_energy = 0.0
    total_peak = 0

    for start in range(0, len(samples), frame_size):
        frame = samples[start : start + frame_size]
        if not frame:
            continue
        peak = max(abs(int(value)) for value in frame)
        energy = sum(float(int(value) * int(value)) for value in frame)
        frame_levels.append(_amplitude_to_dbfs(math.sqrt(energy / len(frame))))
        frame_peaks.append(_amplitude_to_dbfs(float(peak)))
        clipped_sample_count += sum(1 for value in frame if abs(int(value)) >= 32767)
        total_energy += energy
        total_peak = max(total_peak, peak)

    if not frame_levels:
        frame_levels = [-120.0]
        frame_peaks = [-120.0]

    low_level_cutoff_dbfs = _percentile(frame_levels, 0.2)
    low_levels = [level for level in frame_levels if level <= low_level_cutoff_dbfs]
    finite_low_levels = [level for level in low_levels if level > -90.0]
    noise_floor_source = finite_low_levels or low_levels
    noise_floor_dbfs = median(noise_floor_source) if noise_floor_source else -120.0
    speech_level_dbfs = _percentile(frame_levels, 0.8)
    speech_activity_threshold_dbfs = max(
        -50.0,
        speech_level_dbfs - 30.0,
    )
    active_levels = [level for level in frame_levels if level >= speech_activity_threshold_dbfs]
    noise_variation_db = _percentile(noise_floor_source, 0.9) - _percentile(noise_floor_source, 0.1)
    pause_threshold_dbfs = min(-25.0, speech_level_dbfs - 10.0)
    pause_ratio = sum(level <= pause_threshold_dbfs for level in frame_levels) / len(frame_levels)
    estimated_snr_db = speech_level_dbfs - noise_floor_dbfs
    active_dynamic_range_db = (
        _percentile(active_levels, 0.9) - _percentile(active_levels, 0.1)
        if active_levels
        else 0.0
    )
    stationary_noise = (
        -55.0 < noise_floor_dbfs < -25.0
        and noise_variation_db <= 4.0
        and pause_ratio >= 0.05
        and estimated_snr_db >= 8.0
    )

    candidate_threshold_dbfs = min(
        -25.0,
        noise_floor_dbfs + 3.0,
        speech_level_dbfs - 10.0,
    )
    candidate_windows: list[dict[str, float]] = []
    candidate_start: int | None = None
    for index, level in enumerate(frame_levels + [0.0]):
        is_candidate = index < len(frame_levels) and level <= candidate_threshold_dbfs
        if is_candidate and candidate_start is None:
            candidate_start = index
        if not is_candidate and candidate_start is not None:
            duration_seconds = (index - candidate_start) * frame_size / float(sample_rate)
            if duration_seconds >= 0.5:
                candidate_windows.append(
                    {
                        "start_seconds": round(candidate_start * frame_size / float(sample_rate), 3),
                        "end_seconds": round(index * frame_size / float(sample_rate), 3),
                        "duration_seconds": round(duration_seconds, 3),
                    }
                )
            candidate_start = None
    candidate_windows = sorted(
        candidate_windows,
        key=lambda item: item["duration_seconds"],
        reverse=True,
    )[:3]

    recommendations: list[dict[str, Any]] = []
    if clipped_sample_count:
        recommendations.append(
            {
                "action": "manual_clipping_review",
                "priority": "high",
                "reason": "Source contains clipped samples; do not hide this with stronger denoise or gating.",
            }
        )
    if stationary_noise and candidate_windows:
        recommendations.append(
            {
                "action": "noise_window_candidate",
                "priority": "medium",
                "reason": "A stable low-level noise floor and usable pauses were detected.",
                "windows": candidate_windows,
            }
        )
    if active_dynamic_range_db > 12.0:
        recommendations.append(
            {
                "action": "gentle_leveling_review",
                "priority": "medium",
                "reason": "Active speech has a wide short-term level spread; keep leveling gentle and compare against the natural baseline.",
            }
        )
    if not recommendations:
        recommendations.append(
            {
                "action": "natural_baseline_only",
                "priority": "low",
                "reason": "No high-confidence reason was found to enable destructive cleanup automatically.",
            }
        )

    return {
        "frame_ms": frame_ms,
        "frame_count": len(frame_levels),
        "duration_seconds": round(len(samples) / float(sample_rate), 3) if sample_rate else 0.0,
        "rms_dbfs": _amplitude_to_dbfs(math.sqrt(total_energy / len(samples))) if samples else -120.0,
        "peak_dbfs": _amplitude_to_dbfs(float(total_peak)),
        "clipped_sample_count": clipped_sample_count,
        "clipped_sample_ratio": round(clipped_sample_count / float(len(samples)), 8) if samples else 0.0,
        "noise_floor_dbfs": round(noise_floor_dbfs, 2),
        "speech_level_dbfs": round(speech_level_dbfs, 2),
        "estimated_snr_db": round(estimated_snr_db, 2),
        "speech_activity_threshold_dbfs": round(speech_activity_threshold_dbfs, 2),
        "pause_ratio": round(pause_ratio, 4),
        "active_dynamic_range_db": round(active_dynamic_range_db, 2),
        "noise_variation_db": round(noise_variation_db, 2),
        "stationary_noise": stationary_noise,
        "candidate_noise_windows": candidate_windows,
        "recommendations": recommendations,
        "_frame_rms_db": frame_levels,
    }


def compare_audio_preservation(
    reference_analysis: dict[str, Any],
    processed_analysis: dict[str, Any],
    *,
    reference_format: dict[str, Any] | None = None,
    processed_format: dict[str, Any] | None = None,
    reference_samples: array | None = None,
    processed_samples: array | None = None,
    sample_rate: int | None = None,
    excluded_windows: list[NoiseWindow] | None = None,
) -> dict[str, Any]:
    reference_frames = reference_analysis.get("_frame_rms_db", [])
    processed_frames = processed_analysis.get("_frame_rms_db", [])
    active_threshold = float(reference_analysis.get("speech_activity_threshold_dbfs", -50.0))
    frame_ms = float(reference_analysis.get("frame_ms", 20.0))
    max_lag_frames = max(1, int(round(20.0 / max(frame_ms, 1.0))))
    alignment_frames = _best_frame_level_alignment(
        reference_frames,
        processed_frames,
        active_threshold=active_threshold,
        max_lag_frames=max_lag_frames,
    )
    resolved_excluded_windows = merge_noise_windows(excluded_windows or [])

    def frame_is_excluded(frame_index: int) -> bool:
        frame_start = frame_index * frame_ms / 1000.0
        frame_end = frame_start + frame_ms / 1000.0
        return any(
            frame_start < window.end_seconds
            and frame_end > window.start_seconds
            for window in resolved_excluded_windows
        )

    aligned_pairs = [
        (reference_frames[index], processed_frames[index + alignment_frames])
        for index in range(len(reference_frames))
        if 0 <= index + alignment_frames < len(processed_frames)
        and reference_frames[index] >= active_threshold
        and not frame_is_excluded(index)
    ]
    deltas = [
        processed_level - reference_level
        for reference_level, processed_level in aligned_pairs
    ]
    median_gain_db = median(deltas) if deltas else 0.0
    relative_deltas = [delta - median_gain_db for delta in deltas]
    worst_relative_attenuation_db = _percentile(relative_deltas, 0.05) if relative_deltas else 0.0
    relative_gain_spread_db = (
        _percentile(relative_deltas, 0.95) - _percentile(relative_deltas, 0.05)
        if relative_deltas
        else 0.0
    )
    active_level_correlation = 1.0
    if len(aligned_pairs) >= 3:
        reference_mean = sum(pair[0] for pair in aligned_pairs) / len(aligned_pairs)
        processed_mean = sum(pair[1] for pair in aligned_pairs) / len(aligned_pairs)
        covariance = sum(
            (reference - reference_mean) * (processed - processed_mean)
            for reference, processed in aligned_pairs
        )
        reference_energy = sum(
            (reference - reference_mean) ** 2 for reference, _ in aligned_pairs
        )
        processed_energy = sum(
            (processed - processed_mean) ** 2 for _, processed in aligned_pairs
        )
        denominator = math.sqrt(reference_energy * processed_energy)
        if denominator > 1e-12:
            active_level_correlation = max(-1.0, min(1.0, covariance / denominator))
    failures: list[str] = []
    thresholds = {
        "worst_relative_attenuation_db": -6.0,
        "relative_gain_spread_db": 10.0,
        "hard_mute_source_dbfs": -35.0,
        "hard_mute_output_dbfs": -90.0,
        "hard_mute_min_duration_ms": 20.0,
        "spectral_max_loss_db_2_8k": 3.0,
        "spectral_max_loss_db_8_12k": 4.0,
        "spectral_max_gain_db_2_12k": 3.0,
        "spectral_max_gain_db_12_16k": 3.0,
        "pre_speech_max_boost_db": 12.0,
        "pre_speech_source_quiet_dbfs": -40.0,
        "pre_speech_processed_audible_dbfs": -55.0,
    }
    reference_format = reference_format or {}
    processed_format = processed_format or {}
    sample_rate_preserved: bool | None = None
    channel_layout_preserved: bool | None = None
    if reference_format.get("sample_rate") and processed_format.get("sample_rate"):
        sample_rate_preserved = int(reference_format["sample_rate"]) == int(processed_format["sample_rate"])
        if not sample_rate_preserved:
            failures.append("sample_rate_changed")
    if reference_format.get("channels") and processed_format.get("channels"):
        channel_layout_preserved = int(reference_format["channels"]) == int(processed_format["channels"])
        if not channel_layout_preserved:
            failures.append("channel_layout_changed")
    if abs(float(reference_analysis.get("duration_seconds", 0.0)) - float(processed_analysis.get("duration_seconds", 0.0))) > 0.001:
        failures.append("duration_changed")
    if int(processed_analysis.get("clipped_sample_count", 0)) > int(reference_analysis.get("clipped_sample_count", 0)):
        failures.append("clipping_increased")
    if relative_deltas and worst_relative_attenuation_db < thresholds["worst_relative_attenuation_db"]:
        failures.append("active_speech_attenuated")
    if relative_deltas and relative_gain_spread_db > thresholds["relative_gain_spread_db"]:
        failures.append("short_term_gain_instability")

    hard_mute_windows: list[dict[str, Any]] = []
    spectral_band_deltas_db: dict[str, float] = {}
    if (
        reference_samples is not None
        and processed_samples is not None
        and sample_rate is not None
        and sample_rate > 0
    ):
        comparison_processed_samples = array("h", processed_samples)
        for window in resolved_excluded_windows:
            start_index = max(0, int(window.start_seconds * sample_rate))
            end_index = min(
                len(comparison_processed_samples),
                int(math.ceil(window.end_seconds * sample_rate)),
            )
            comparison_processed_samples[start_index:end_index] = (
                reference_samples[start_index:end_index]
            )
        hard_mute_windows = detect_source_active_hard_mute_windows(
            reference_samples,
            comparison_processed_samples,
            sample_rate=sample_rate,
            alignment_frames=alignment_frames,
            frame_ms=frame_ms,
            source_threshold_dbfs=thresholds["hard_mute_source_dbfs"],
            output_threshold_dbfs=thresholds["hard_mute_output_dbfs"],
            min_duration_ms=thresholds["hard_mute_min_duration_ms"],
        )
        if hard_mute_windows:
            failures.append("source_active_hard_mute")
        spectral_band_deltas_db = measure_spectral_band_deltas_db(
            reference_samples,
            comparison_processed_samples,
            sample_rate=sample_rate,
            median_gain_db=median_gain_db,
        )
        for band_name, delta_db in spectral_band_deltas_db.items():
            if band_name in {"2-4k", "4-6k", "6-8k"} and delta_db < -thresholds["spectral_max_loss_db_2_8k"]:
                failures.append("spectral_clarity_lost")
                break
            if band_name in {"8-10k", "10-12k", "12-16k"} and delta_db < -thresholds["spectral_max_loss_db_8_12k"]:
                failures.append("spectral_clarity_lost")
                break
        for band_name, delta_db in spectral_band_deltas_db.items():
            if band_name in {"2-4k", "4-6k", "6-8k", "8-10k", "10-12k"} and delta_db > thresholds["spectral_max_gain_db_2_12k"]:
                failures.append("spectral_harshness_increased")
                break
            if band_name == "12-16k" and delta_db > thresholds["spectral_max_gain_db_12_16k"]:
                failures.append("spectral_harshness_increased")
                break
        # Evaluate against the true delivery samples — authorized cleanup
        # exclusions must not wash out a loudnorm pre-speech boom.
        pre_speech_boost_windows = detect_pre_speech_soft_noise_boost_windows(
            reference_samples,
            processed_samples,
            sample_rate=sample_rate,
            source_quiet_dbfs=thresholds["pre_speech_source_quiet_dbfs"],
            processed_audible_dbfs=thresholds["pre_speech_processed_audible_dbfs"],
            max_boost_db=thresholds["pre_speech_max_boost_db"],
        )
        if pre_speech_boost_windows:
            failures.append("pre_speech_soft_noise_boosted")
    else:
        pre_speech_boost_windows = []

    failures = list(dict.fromkeys(failures))
    return {
        "evidence_level": "DIRECTLY VERIFIED",
        "status": "FAIL" if failures else "PASS",
        "checked_active_frames": len(deltas),
        "alignment_frames": alignment_frames,
        "alignment_ms": round(alignment_frames * frame_ms, 3),
        "median_gain_db": round(median_gain_db, 2),
        "worst_relative_attenuation_db": round(worst_relative_attenuation_db, 2),
        "relative_gain_spread_db": round(relative_gain_spread_db, 2),
        "active_level_correlation": round(active_level_correlation, 4),
        "excluded_window_count": len(resolved_excluded_windows),
        "thresholds": thresholds,
        "hard_mute_windows": hard_mute_windows,
        "pre_speech_boost_windows": pre_speech_boost_windows,
        "spectral_band_deltas_db": spectral_band_deltas_db,
        "failures": failures,
        "release_blocked": bool(failures),
        "reference_format": reference_format,
        "processed_format": processed_format,
        "sample_rate_preserved": sample_rate_preserved,
        "channel_layout_preserved": channel_layout_preserved,
    }


def detect_source_active_hard_mute_windows(
    reference_samples: array,
    processed_samples: array,
    *,
    sample_rate: int,
    alignment_frames: int = 0,
    frame_ms: float = 10.0,
    source_threshold_dbfs: float = -35.0,
    output_threshold_dbfs: float = -90.0,
    min_duration_ms: float = 20.0,
) -> list[dict[str, Any]]:
    frame_size = max(1, int(sample_rate * (frame_ms / 1000.0)))
    reference_frames = [
        _amplitude_to_dbfs(
            math.sqrt(
                sum(float(int(value) * int(value)) for value in reference_samples[start : start + frame_size])
                / max(1, len(reference_samples[start : start + frame_size]))
            )
        )
        for start in range(0, len(reference_samples), frame_size)
        if reference_samples[start : start + frame_size]
    ]
    processed_frames = [
        _amplitude_to_dbfs(
            math.sqrt(
                sum(float(int(value) * int(value)) for value in processed_samples[start : start + frame_size])
                / max(1, len(processed_samples[start : start + frame_size]))
            )
        )
        for start in range(0, len(processed_samples), frame_size)
        if processed_samples[start : start + frame_size]
    ]
    mask: list[bool] = []
    for index, reference_level in enumerate(reference_frames):
        processed_index = index + alignment_frames
        if processed_index < 0 or processed_index >= len(processed_frames):
            mask.append(False)
            continue
        mask.append(
            reference_level >= source_threshold_dbfs
            and processed_frames[processed_index] <= output_threshold_dbfs
        )
    windows: list[dict[str, Any]] = []
    start_index: int | None = None
    for index, is_hard_mute in enumerate(mask + [False]):
        if is_hard_mute and start_index is None:
            start_index = index
        elif not is_hard_mute and start_index is not None:
            duration_ms = (index - start_index) * frame_ms
            if duration_ms >= min_duration_ms:
                segment = reference_frames[start_index:index]
                windows.append(
                    {
                        "start_seconds": round(start_index * frame_ms / 1000.0, 3),
                        "end_seconds": round(index * frame_ms / 1000.0, 3),
                        "duration_ms": round(duration_ms, 1),
                        "source_peak_dbfs": round(max(segment), 2),
                        "source_median_dbfs": round(median(segment), 2),
                    }
                )
            start_index = None
    return windows


def measure_spectral_band_deltas_db(
    reference_samples: array,
    processed_samples: array,
    *,
    sample_rate: int,
    median_gain_db: float = 0.0,
) -> dict[str, float]:
    bands = (
        ("2-4k", 2000.0, 4000.0),
        ("4-6k", 4000.0, 6000.0),
        ("6-8k", 6000.0, 8000.0),
        ("8-10k", 8000.0, 10000.0),
        ("10-12k", 10000.0, 12000.0),
        ("12-16k", 12000.0, 16000.0),
    )
    usable_bands = [(name, low, high) for name, low, high in bands if high <= sample_rate / 2]
    if not usable_bands:
        return {}
    window_size = min(2048, len(reference_samples), len(processed_samples))
    if window_size < 64:
        return {}
    reference_energies = {name: 0.0 for name, _, _ in usable_bands}
    processed_energies = {name: 0.0 for name, _, _ in usable_bands}
    hop = max(window_size, 1)
    limit = min(len(reference_samples), len(processed_samples)) - window_size + 1
    sampled = 0
    for start in range(0, max(limit, 1), hop):
        reference_window = reference_samples[start : start + window_size]
        processed_window = processed_samples[start : start + window_size]
        if len(reference_window) < window_size or len(processed_window) < window_size:
            continue
        reference_rms = math.sqrt(
            sum(float(int(value) * int(value)) for value in reference_window) / window_size
        )
        if _amplitude_to_dbfs(reference_rms) < -45.0:
            continue
        reference_spectrum = _windowed_power_spectrum(reference_window)
        processed_spectrum = _windowed_power_spectrum(processed_window)
        for name, low, high in usable_bands:
            reference_energies[name] += _band_energy(reference_spectrum, sample_rate, low, high)
            processed_energies[name] += _band_energy(processed_spectrum, sample_rate, low, high)
        sampled += 1
        if sampled >= 24:
            break
    if sampled == 0:
        return {name: 0.0 for name, _, _ in usable_bands}
    gain_linear = 10 ** (median_gain_db / 20.0)
    deltas: dict[str, float] = {}
    for name, _, _ in usable_bands:
        reference_energy = max(reference_energies[name], 1e-20)
        processed_energy = max(processed_energies[name] / max(gain_linear * gain_linear, 1e-20), 1e-20)
        deltas[name] = round(10.0 * math.log10(processed_energy / reference_energy), 2)
    return deltas


def _windowed_power_spectrum(samples: array) -> list[float]:
    count = len(samples)
    if count <= 0:
        return [0.0]
    windowed = [
        float(int(value)) * (0.5 - 0.5 * math.cos(2.0 * math.pi * index / max(count - 1, 1)))
        for index, value in enumerate(samples)
    ]
    half = count // 2 + 1
    spectrum: list[float] = []
    for frequency_bin in range(half):
        real = 0.0
        imag = 0.0
        angle_step = -2.0 * math.pi * frequency_bin / count
        for index, sample in enumerate(windowed):
            angle = angle_step * index
            real += sample * math.cos(angle)
            imag += sample * math.sin(angle)
        spectrum.append(real * real + imag * imag)
    return spectrum


def _band_energy(spectrum: list[float], sample_rate: int, low_hz: float, high_hz: float) -> float:
    if not spectrum:
        return 0.0
    bin_hz = sample_rate / max((len(spectrum) - 1) * 2, 1)
    total = 0.0
    for index, power in enumerate(spectrum):
        frequency = index * bin_hz
        if low_hz <= frequency < high_hz:
            total += power
    return total


def _best_frame_level_alignment(
    reference_frames: list[float],
    processed_frames: list[float],
    *,
    active_threshold: float,
    max_lag_frames: int = 3,
) -> int:
    if not reference_frames or not processed_frames:
        return 0

    best_lag = 0
    best_score = float("-inf")
    for lag in range(-max_lag_frames, max_lag_frames + 1):
        pairs = [
            (reference_frames[index], processed_frames[index + lag])
            for index in range(len(reference_frames))
            if 0 <= index + lag < len(processed_frames)
            and reference_frames[index] >= active_threshold
        ]
        if len(pairs) < 3:
            continue
        reference_mean = sum(item[0] for item in pairs) / len(pairs)
        processed_mean = sum(item[1] for item in pairs) / len(pairs)
        covariance = sum(
            (reference_level - reference_mean) * (processed_level - processed_mean)
            for reference_level, processed_level in pairs
        )
        reference_energy = sum((item[0] - reference_mean) ** 2 for item in pairs)
        processed_energy = sum((item[1] - processed_mean) ** 2 for item in pairs)
        if reference_energy <= 1e-12 or processed_energy <= 1e-12:
            score = 0.0 if lag == 0 else float("-inf")
        else:
            score = covariance / math.sqrt(reference_energy * processed_energy)
        if score > best_score or (math.isclose(score, best_score) and abs(lag) < abs(best_lag)):
            best_lag = lag
            best_score = score
    return best_lag


def public_audio_diagnostics(analysis: dict[str, Any]) -> dict[str, Any]:
    return {
        "evidence_level": "MODEL-INFERRED",
        **{
            key: value
            for key, value in analysis.items()
            if not key.startswith("_")
        },
    }


def measure_segment_levels(
    samples: array,
    *,
    start_index: int,
    end_index: int,
) -> tuple[float, float]:
    start = max(0, min(len(samples), start_index))
    end = max(start, min(len(samples), end_index))
    if end <= start:
        return -120.0, -120.0

    peak = 0
    energy = 0.0
    count = end - start
    for index in range(start, end):
        value = abs(int(samples[index]))
        if value > peak:
            peak = value
        energy += float(value * value)

    rms = math.sqrt(energy / float(count)) if count else 0.0
    return _amplitude_to_dbfs(float(peak)), _amplitude_to_dbfs(rms)


def _pause_core_pads(
    window: NoiseWindow,
    *,
    core_pad_seconds: float,
    leading_trailing_pad_seconds: float,
    total_duration_seconds: float | None,
) -> tuple[float, float]:
    """Use tighter pads for leading/trailing silence; keep hold pad for mid-phrase pauses."""
    start_pad = core_pad_seconds
    end_pad = core_pad_seconds
    if total_duration_seconds is None or total_duration_seconds <= 0.0:
        return start_pad, end_pad

    edge_tolerance = max(leading_trailing_pad_seconds, 0.01)
    if window.start_seconds <= edge_tolerance:
        start_pad = leading_trailing_pad_seconds
    if window.end_seconds >= (total_duration_seconds - edge_tolerance):
        end_pad = leading_trailing_pad_seconds
    return start_pad, end_pad


def infer_pause_residual_cleanup_windows(
    samples: array,
    *,
    sample_rate: int,
    channels: int = 1,
    silence_candidates: list[dict[str, float]],
    min_neighbor_silence_duration: float,
    bridge_max_duration: float,
    bridge_peak_db: float,
    bridge_rms_db: float,
    core_pad_seconds: float,
    leading_trailing_pad_seconds: float | None = None,
    total_duration_seconds: float | None = None,
    allow_bridge_windows: bool = True,
) -> list[NoiseWindow]:
    windows: list[NoiseWindow] = []
    silence_windows = [
        NoiseWindow(
            start_seconds=float(item["start_seconds"]),
            end_seconds=float(item["end_seconds"]),
        )
        for item in silence_candidates
    ]
    edge_pad_seconds = (
        float(leading_trailing_pad_seconds)
        if leading_trailing_pad_seconds is not None
        else float(core_pad_seconds)
    )
    if total_duration_seconds is None and samples and sample_rate > 0:
        total_duration_seconds = len(samples) / float(sample_rate * max(1, channels))

    for window in silence_windows:
        start_pad, end_pad = _pause_core_pads(
            window,
            core_pad_seconds=core_pad_seconds,
            leading_trailing_pad_seconds=edge_pad_seconds,
            total_duration_seconds=total_duration_seconds,
        )
        duration = window.end_seconds - window.start_seconds
        if duration <= (start_pad + end_pad):
            continue
        core_start = window.start_seconds + start_pad
        core_end = window.end_seconds - end_pad
        if core_end > core_start:
            windows.append(NoiseWindow(start_seconds=core_start, end_seconds=core_end))

    # Speech-safe AutoGate disables unpadded bridge cleanup: bridges sit outside
    # confirmed silence cores and can clip word onsets/offsets.
    if allow_bridge_windows:
        for previous, current in zip(silence_windows, silence_windows[1:]):
            previous_duration = previous.end_seconds - previous.start_seconds
            current_duration = current.end_seconds - current.start_seconds
            if (
                previous_duration < min_neighbor_silence_duration
                or current_duration < min_neighbor_silence_duration
            ):
                continue

            bridge_start = previous.end_seconds
            bridge_end = current.start_seconds
            bridge_duration = bridge_end - bridge_start
            if bridge_duration <= 0.0 or bridge_duration > bridge_max_duration:
                continue

            peak_db, rms_db = measure_segment_levels(
                samples,
                start_index=int(bridge_start * sample_rate) * max(1, channels),
                end_index=int(bridge_end * sample_rate) * max(1, channels),
            )
            if peak_db <= bridge_peak_db and rms_db <= bridge_rms_db:
                windows.append(
                    NoiseWindow(start_seconds=bridge_start, end_seconds=bridge_end)
                )

    return merge_noise_windows(windows, max_gap_seconds=0.02)


def _ensure_directories(layout: OutputLayout) -> None:
    layout.preprocess_dir.mkdir(parents=True, exist_ok=True)
    layout.transcript_dir.mkdir(parents=True, exist_ok=True)
    layout.deepfilternet_dir.mkdir(parents=True, exist_ok=True)


def _find_single_wav(directory: Path) -> Path:
    candidates = sorted(directory.glob("*.wav"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one WAV output from DeepFilterNet in {directory}, found {len(candidates)}"
        )
    return candidates[0]


def _run_stereo_balance_stage(
    *,
    raw_wav: Path,
    samples: array,
    sample_rate: int,
    channels: int,
    policy: dict[str, Any] | None,
) -> tuple[array, dict[str, Any]]:
    """Balance stereo ear mismatch without changing channel count."""
    balanced, report = run_stereo_balance(
        samples,
        sample_rate=sample_rate,
        channels=channels,
        policy=policy,
    )
    if report.get("status") == "PASS" and report.get("plan", {}).get("applied"):
        write_pcm16_wave(
            raw_wav,
            balanced,
            sample_rate=sample_rate,
            channels=channels,
        )
        return balanced, report
    return samples, report


def _processing_steps(
    preset: dict[str, Any],
    *,
    skip_spectramini: bool = False,
    skip_deepfilternet: bool = False,
) -> list[str]:
    steps = ["Extract source audio to WAV"]
    filters = preset["filters"]
    if filters.get("stereo_balance", {}).get("enabled"):
        steps.append("Stereo channel balance while preserving channel layout")
    stage_types = [stage.get("type") for stage in preset.get("pipeline", {}).get("stages", []) if stage.get("enabled", True)]
    if "respiro" in stage_types:
        steps.append("Breath detection via Respiro-en")
    if "spectramini" in stage_types and not skip_spectramini:
        steps.append("SpectraMini-style breath control")
        spectramini = get_pipeline_stage(preset, "spectramini")
        if float(spectramini.get("mouth_declick_sensitivity", 0.0)) > 0.0:
            steps.append("SpectraMini-style mouth de-click")
    if "deepfilternet" in stage_types and not skip_deepfilternet:
        steps.append("Primary denoise via DeepFilterNet")
    if filters.get("equalizer", {}).get("enabled"):
        steps.append("Clarity shaping via parametric equalizer")
    if filters.get("declick", {}).get("enabled"):
        steps.append("Mouth-click reduction via adeclick")
    if filters.get("breath_ducking", {}).get("enabled"):
        steps.append("Speech-presence breath ducking via sidechaingate")
    if filters.get("secondary_denoise", {}).get("enabled"):
        steps.append("Secondary FFmpeg denoise via afftdn")
    if filters.get("nonlinear_denoise", {}).get("enabled"):
        steps.append("Non-local denoise via anlmdn")
    if filters.get("deesser", {}).get("enabled"):
        steps.append("Breath and sibilance control via deesser")
    if filters.get("gate", {}).get("enabled"):
        steps.append("Dynamic gate via agate")
    if filters.get("compressor", {}).get("enabled"):
        steps.append("Voice compression via acompressor")
    if filters.get("speech_norm", {}).get("enabled"):
        steps.append("Speech leveling via speechnorm")
    if filters.get("breath_onset_cleanup", {}).get("enabled"):
        steps.append("Breath-onset cleanup before speech entries")
    if filters.get("pause_residual_cleanup", {}).get("enabled"):
        if "silence_floor_dbfs" in filters.get("pause_residual_cleanup", {}):
            steps.append("Speech-safe AutoGate pause cleanup to silence floor")
        else:
            steps.append("Residual pause cleanup in long silences")
    steps.append("Loudness normalization via loudnorm")
    steps.append("Transcript-ready MP3 export")
    return steps


def _noise_print_processing_steps(
    preset: dict[str, Any],
    *,
    skip_spectramini: bool = False,
    skip_deepfilternet: bool = False,
) -> list[str]:
    steps = ["Extract source audio to WAV"]
    filters = preset["filters"]
    if filters.get("stereo_balance", {}).get("enabled"):
        steps.append("Stereo channel balance while preserving channel layout")
    stage_types = [stage.get("type") for stage in preset.get("pipeline", {}).get("stages", []) if stage.get("enabled", True)]
    if "respiro" in stage_types:
        steps.append("Breath detection via Respiro-en")
    if "spectramini" in stage_types and not skip_spectramini:
        steps.append("SpectraMini-style breath control")
        steps.append("SpectraMini-style mouth de-click")
    if "deepfilternet" in stage_types and not skip_deepfilternet:
        steps.append("Primary denoise via DeepFilterNet")
    steps.extend(
        [
            "Capture noise sample from selected windows",
            "Noise-print denoise via afftdn sample capture",
        ]
    )
    if filters.get("equalizer", {}).get("enabled"):
        steps.append("Clarity shaping via parametric equalizer")
    if filters.get("declick", {}).get("enabled"):
        steps.append("Mouth-click reduction via adeclick")
    if filters.get("breath_ducking", {}).get("enabled"):
        steps.append("Speech-presence breath ducking via sidechaingate")
    if filters.get("nonlinear_denoise", {}).get("enabled"):
        steps.append("Non-local denoise via anlmdn")
    if filters.get("deesser", {}).get("enabled"):
        steps.append("Breath and sibilance control via deesser")
    if filters.get("gate", {}).get("enabled"):
        steps.append("Dynamic gate via agate")
    if filters.get("compressor", {}).get("enabled"):
        steps.append("Voice compression via acompressor")
    if filters.get("speech_norm", {}).get("enabled"):
        steps.append("Speech leveling via speechnorm")
    if filters.get("breath_onset_cleanup", {}).get("enabled"):
        steps.append("Breath-onset cleanup before speech entries")
    if filters.get("pause_residual_cleanup", {}).get("enabled"):
        if "silence_floor_dbfs" in filters.get("pause_residual_cleanup", {}):
            steps.append("Speech-safe AutoGate pause cleanup to silence floor")
        else:
            steps.append("Residual pause cleanup in long silences")
    steps.append("Loudness normalization via loudnorm")
    steps.append("Transcript-ready MP3 export")
    return steps


def _noise_sample_duration(noise_windows: list[NoiseWindow]) -> float:
    return sum(window.end_seconds - window.start_seconds for window in noise_windows)


def _normalize_noise_windows(noise_windows: list[NoiseWindow] | None) -> list[NoiseWindow]:
    return list(noise_windows or [])


def _downmix_interleaved_samples(samples: array, channels: int) -> array:
    if channels <= 0:
        raise ValueError("channels must be positive")
    if channels == 1:
        return array("h", samples)
    mono = array("h")
    for frame_start in range(0, len(samples), channels):
        frame = samples[frame_start : frame_start + channels]
        if not frame:
            continue
        average = round(sum(int(value) for value in frame) / len(frame))
        mono.append(max(-32768, min(32767, average)))
    return mono


def _analysis_samples(params: Any, samples: array) -> array:
    return _downmix_interleaved_samples(samples, int(getattr(params, "nchannels", 1)))


def _window_rms_dbfs(
    samples: array,
    windows: Sequence[NoiseWindow],
    *,
    sample_rate: int,
    channels: int,
) -> float | None:
    total_energy = 0.0
    sample_count = 0
    for window in windows:
        start = max(0, int(window.start_seconds * sample_rate) * channels)
        end = min(len(samples), int(window.end_seconds * sample_rate) * channels)
        if end <= start:
            continue
        for value in samples[start:end]:
            total_energy += float(int(value) * int(value))
            sample_count += 1
    if sample_count == 0:
        return None
    return _amplitude_to_dbfs(math.sqrt(total_energy / sample_count))


def _read_wave_duration_seconds(audio_path: Path) -> float:
    with wave.open(str(audio_path), "rb") as reader:
        frame_rate = reader.getframerate()
        if frame_rate <= 0:
            return 0.0
        return reader.getnframes() / float(frame_rate)


def _write_wave_samples(audio_path: Path, params: Any, samples: array) -> None:
    with wave.open(str(audio_path), "wb") as writer:
        writer.setparams(params)
        writer.writeframes(samples.tobytes())


def repair_deepfilternet_speech_dropouts(
    reference_samples: array,
    processed_samples: array,
    *,
    sample_rate: int,
    window_seconds: float = 0.06,
    hop_seconds: float = 0.01,
    reference_peak_db_min: float = -24.0,
    reference_rms_db_min: float = -38.0,
    processed_peak_db_max: float = -34.0,
    processed_rms_db_max: float = -46.0,
    copy_padding_seconds: float = 0.008,
    max_repair_duration_seconds: float = 0.16,
    context_window_seconds: float = 0.22,
    context_gap_seconds: float = 0.02,
    context_peak_db_min: float = -20.0,
    context_rms_db_min: float = -32.0,
) -> tuple[array, list[NoiseWindow]]:
    if len(reference_samples) != len(processed_samples):
        raise ValueError("reference_samples and processed_samples must have the same length")
    if not reference_samples:
        return array("h", processed_samples), []

    window_size = max(1, int(window_seconds * sample_rate))
    hop_size = max(1, int(hop_seconds * sample_rate))
    copy_padding = max(0, int(copy_padding_seconds * sample_rate))
    max_repair_samples = max(window_size, int(max_repair_duration_seconds * sample_rate))
    context_window_size = max(1, int(context_window_seconds * sample_rate))
    context_gap_size = max(0, int(context_gap_seconds * sample_rate))
    total_samples = len(reference_samples)

    candidate_windows: list[NoiseWindow] = []

    def has_voiced_context(start_index: int, end_index: int) -> bool:
        left_end = max(0, start_index - context_gap_size)
        left_start = max(0, left_end - context_window_size)
        right_start = min(total_samples, end_index + context_gap_size)
        right_end = min(total_samples, right_start + context_window_size)
        if left_end <= left_start or right_end <= right_start:
            return False

        left_peak_db, left_rms_db = measure_segment_levels(
            reference_samples,
            start_index=left_start,
            end_index=left_end,
        )
        right_peak_db, right_rms_db = measure_segment_levels(
            reference_samples,
            start_index=right_start,
            end_index=right_end,
        )
        return (
            left_peak_db >= context_peak_db_min
            and left_rms_db >= context_rms_db_min
            and right_peak_db >= context_peak_db_min
            and right_rms_db >= context_rms_db_min
        )

    index = 0
    while index + window_size <= total_samples:
        ref_peak_db, ref_rms_db = measure_segment_levels(
            reference_samples,
            start_index=index,
            end_index=index + window_size,
        )
        proc_peak_db, proc_rms_db = measure_segment_levels(
            processed_samples,
            start_index=index,
            end_index=index + window_size,
        )
        if (
            ref_peak_db >= reference_peak_db_min
            and ref_rms_db >= reference_rms_db_min
            and proc_peak_db <= processed_peak_db_max
            and proc_rms_db <= processed_rms_db_max
        ):
            start = index
            end = index + window_size
            index += hop_size
            while index + window_size <= total_samples:
                ref_peak_db, ref_rms_db = measure_segment_levels(
                    reference_samples,
                    start_index=index,
                    end_index=index + window_size,
                )
                proc_peak_db, proc_rms_db = measure_segment_levels(
                    processed_samples,
                    start_index=index,
                    end_index=index + window_size,
                )
                if not (
                    ref_peak_db >= reference_peak_db_min
                    and ref_rms_db >= reference_rms_db_min
                    and proc_peak_db <= processed_peak_db_max
                    and proc_rms_db <= processed_rms_db_max
                ):
                    break
                end = index + window_size
                index += hop_size
            if (end - start) <= max_repair_samples and has_voiced_context(start, end):
                candidate_windows.append(
                    NoiseWindow(
                        start_seconds=max(0.0, (start - copy_padding) / float(sample_rate)),
                        end_seconds=min(total_samples / float(sample_rate), (end + copy_padding) / float(sample_rate)),
                    )
                )
            continue
        index += hop_size

    merged_windows = merge_noise_windows(candidate_windows, max_gap_seconds=max(0.01, hop_seconds * 2.0))
    repaired = array("h", processed_samples)
    for window in merged_windows:
        start_index = max(0, min(total_samples, int(window.start_seconds * sample_rate)))
        end_index = max(start_index, min(total_samples, int(window.end_seconds * sample_rate)))
        repaired[start_index:end_index] = reference_samples[start_index:end_index]
    return repaired, merged_windows


def _run_leading_pre_speech_seal(
    *,
    target_wav: Path,
    raw_wav: Path,
    report: dict[str, Any],
    report_key: str,
    stage: str,
    source_analysis: dict[str, Any] | None = None,
    hold_pad_ms: float = 45.0,
    silence_floor_dbfs: float = -96.0,
    fade_ms: float = 12.0,
) -> bool:
    """Seal only measured post-processing boosts in pre-speech soft noise."""
    if not target_wav.exists() or not raw_wav.exists():
        report[report_key] = {
            "status": "SKIP",
            "stage": stage,
            "reason": "seal_inputs_missing",
        }
        return False
    onset_params, onset_samples = _load_wave_samples(raw_wav)
    target_params, target_samples = _load_wave_samples(target_wav)
    if (
        target_params.framerate != onset_params.framerate
        or target_params.nchannels != onset_params.nchannels
    ):
        report[report_key] = {
            "status": "SKIP",
            "stage": stage,
            "reason": "seal_format_mismatch",
        }
        return False
    sealed_samples, seal_report = seal_leading_pre_speech_noise(
        onset_samples,
        target_samples,
        sample_rate=target_params.framerate,
        channels=target_params.nchannels,
        source_analysis=source_analysis,
        hold_pad_ms=hold_pad_ms,
        silence_floor_dbfs=silence_floor_dbfs,
        fade_ms=fade_ms,
    )
    report[report_key] = {
        "stage": stage,
        **seal_report,
    }
    if seal_report.get("status") != "PASS" or sealed_samples == target_samples:
        return False
    _write_wave_samples(target_wav, target_params, sealed_samples)
    return True


def _run_pause_cleanup(
    *,
    layout: OutputLayout,
    active_preset: dict[str, Any],
    input_analysis: dict[str, Any],
    runtime: RuntimeOptions,
    report: dict[str, Any],
    authorized_cleanup_windows: list[NoiseWindow],
    commands: list[list[str]],
    executed: list[dict[str, Any]],
) -> None:
    pause_residual_cleanup = active_preset.get("filters", {}).get(
        "pause_residual_cleanup",
        {},
    )
    if pause_residual_cleanup.get("enabled"):
        processed_silence_candidates = detect_silence_candidates(
            layout.clean_wav,
            ffmpeg_bin=runtime.ffmpeg_bin,
            threshold_db=float(pause_residual_cleanup["silence_threshold_db"]),
            min_duration=float(pause_residual_cleanup["silence_min_duration"]),
        )
        source_silence_candidates = (
            detect_silence_candidates(
                layout.raw_wav,
                ffmpeg_bin=runtime.ffmpeg_bin,
                threshold_db=float(pause_residual_cleanup["silence_threshold_db"]),
                min_duration=float(pause_residual_cleanup["silence_min_duration"]),
            )
            if layout.raw_wav.exists()
            else []
        )
        merged_silence_windows = merge_noise_windows(
            [
                NoiseWindow(
                    start_seconds=float(item["start_seconds"]),
                    end_seconds=float(item["end_seconds"]),
                )
                for item in [*source_silence_candidates, *processed_silence_candidates]
            ],
            max_gap_seconds=0.02,
        )
        silence_candidates = [
            {
                "start_seconds": window.start_seconds,
                "end_seconds": window.end_seconds,
                "duration_seconds": window.end_seconds - window.start_seconds,
            }
            for window in merged_silence_windows
        ]
        report["pause_cleanup"]["source_silence_candidate_count"] = len(
            source_silence_candidates
        )
        report["pause_cleanup"]["processed_silence_candidate_count"] = len(
            processed_silence_candidates
        )
        params, samples = _load_wave_samples(layout.clean_wav)
        total_duration_seconds = (
            len(samples) / float(params.framerate * max(1, params.nchannels))
            if params.framerate
            else 0.0
        )
        core_pad_seconds = float(pause_residual_cleanup["core_pad_ms"]) / 1000.0
        leading_trailing_pad_seconds = (
            float(
                pause_residual_cleanup.get(
                    "leading_trailing_pad_ms",
                    pause_residual_cleanup["core_pad_ms"],
                )
            )
            / 1000.0
        )
        allow_bridge_windows = bool(
            pause_residual_cleanup.get(
                "allow_bridge_windows",
                "silence_floor_dbfs" not in pause_residual_cleanup,
            )
        )
        inferred_pause_windows = infer_pause_residual_cleanup_windows(
            samples,
            sample_rate=params.framerate,
            channels=params.nchannels,
            silence_candidates=silence_candidates,
            min_neighbor_silence_duration=float(pause_residual_cleanup["min_neighbor_silence_duration"]),
            bridge_max_duration=float(pause_residual_cleanup["bridge_max_duration"]),
            bridge_peak_db=float(pause_residual_cleanup["bridge_peak_db"]),
            bridge_rms_db=float(pause_residual_cleanup["bridge_rms_db"]),
            core_pad_seconds=core_pad_seconds,
            leading_trailing_pad_seconds=leading_trailing_pad_seconds,
            total_duration_seconds=total_duration_seconds,
            allow_bridge_windows=allow_bridge_windows,
        )
        pause_windows, pause_window_evidence = build_effective_breath_windows(
            respiro_windows=[],
            auxiliary_windows=inferred_pause_windows,
            source_analysis=input_analysis,
        )
        # Speech-safe AutoGate floor: confirmed silence cores target absolute silence,
        # not "slightly below the measured noise floor".
        if "silence_floor_dbfs" in pause_residual_cleanup:
            target_dbfs = float(pause_residual_cleanup["silence_floor_dbfs"])
            pause_mode = "speech_safe_autogate"
        else:
            target_dbfs = float(input_analysis["noise_floor_dbfs"]) + float(
                pause_residual_cleanup["target_margin_db"]
            )
            pause_mode = "noise_floor_margin"
        residual_measure_max_db = max(
            36.0,
            float(pause_residual_cleanup["max_attenuation_db"]),
            float(pause_residual_cleanup["final_pass_max_attenuation_db"]),
        )
        cleaned_pause_samples, pause_cleanup_details = (
            apply_adaptive_breath_cleanup(
                samples,
                windows=pause_windows,
                sample_rate=params.framerate,
                channels=params.nchannels,
                max_attenuation_db=float(
                    pause_residual_cleanup["max_attenuation_db"]
                ),
                target_margin_db=float(
                    pause_residual_cleanup["target_margin_db"]
                ),
                context_ms=0.0,
                fade_ms=float(pause_residual_cleanup["fade_ms"]),
                target_dbfs_override=target_dbfs,
            )
        )
        report["pause_cleanup"]["mode"] = pause_mode
        report["pause_cleanup"]["target_dbfs"] = round(target_dbfs, 3)
        report["pause_cleanup"]["silence_floor_dbfs"] = (
            float(pause_residual_cleanup["silence_floor_dbfs"])
            if "silence_floor_dbfs" in pause_residual_cleanup
            else None
        )
        report["pause_cleanup"]["core_pad_ms"] = round(core_pad_seconds * 1000.0, 3)
        report["pause_cleanup"]["leading_trailing_pad_ms"] = round(
            leading_trailing_pad_seconds * 1000.0,
            3,
        )
        report["pause_cleanup"]["allow_bridge_windows"] = allow_bridge_windows
        report["pause_cleanup"]["window_evidence"] = pause_window_evidence
        report["pause_cleanup"]["first_pass"] = pause_cleanup_details
        requested_attenuations = [
            float(detail.get("requested_attenuation_db", 0.0))
            for detail in pause_cleanup_details
        ]
        report["pause_cleanup"]["attenuation_stats"] = {
            "first_pass_window_count": len(requested_attenuations),
            "first_pass_max_attenuation_db": (
                round(max(requested_attenuations), 3) if requested_attenuations else 0.0
            ),
            "first_pass_mean_attenuation_db": (
                round(sum(requested_attenuations) / len(requested_attenuations), 3)
                if requested_attenuations
                else 0.0
            ),
        }
        report["pause_residual_cleanup_windows"] = [
            {"start_seconds": window.start_seconds, "end_seconds": window.end_seconds}
            for window in pause_windows
        ]
        if cleaned_pause_samples != samples:
            _write_wave_samples(layout.clean_wav, params, cleaned_pause_samples)
            authorized_cleanup_windows.extend(pause_windows)
            transcript_command = commands[-1]
            _run_recorded_command(transcript_command, executed)

        _, pause_residual_details = apply_adaptive_breath_cleanup(
            cleaned_pause_samples,
            windows=pause_windows,
            sample_rate=params.framerate,
            channels=params.nchannels,
            max_attenuation_db=residual_measure_max_db,
            target_margin_db=float(pause_residual_cleanup["target_margin_db"]),
            context_ms=0.0,
            fade_ms=0.0,
            target_dbfs_override=target_dbfs,
        )
        residual_minimum_db = float(
            pause_residual_cleanup.get("residual_min_excess_db", 3.0)
        )
        pause_residual_windows = [
            {
                "start_seconds": detail["start_seconds"],
                "end_seconds": detail["end_seconds"],
                "remaining_excess_db": detail["requested_attenuation_db"],
            }
            for detail in pause_residual_details
            if float(detail["requested_attenuation_db"]) >= residual_minimum_db
        ]
        if pause_residual_windows:
            residual_noise_windows = [
                NoiseWindow(
                    start_seconds=float(item["start_seconds"]),
                    end_seconds=float(item["end_seconds"]),
                )
                for item in pause_residual_windows
            ]
            second_pause_samples, second_pause_details = (
                apply_adaptive_breath_cleanup(
                    cleaned_pause_samples,
                    windows=residual_noise_windows,
                    sample_rate=params.framerate,
                    channels=params.nchannels,
                    max_attenuation_db=float(
                        pause_residual_cleanup[
                            "second_pass_max_attenuation_db"
                        ]
                    ),
                    target_margin_db=float(
                        pause_residual_cleanup["target_margin_db"]
                    ),
                    context_ms=0.0,
                    fade_ms=float(pause_residual_cleanup["fade_ms"]) / 2.0,
                    target_dbfs_override=target_dbfs,
                )
            )
            report["pause_cleanup"]["second_pass"] = second_pause_details
            if second_pause_samples != cleaned_pause_samples:
                _write_wave_samples(
                    layout.clean_wav,
                    params,
                    second_pause_samples,
                )
                transcript_command = commands[-1]
                _run_recorded_command(transcript_command, executed)
            _, pause_residual_details = apply_adaptive_breath_cleanup(
                second_pause_samples,
                windows=residual_noise_windows,
                sample_rate=params.framerate,
                channels=params.nchannels,
                max_attenuation_db=residual_measure_max_db,
                target_margin_db=float(
                    pause_residual_cleanup["target_margin_db"]
                ),
                context_ms=0.0,
                fade_ms=0.0,
                target_dbfs_override=target_dbfs,
            )
            pause_residual_windows = [
                {
                    "start_seconds": detail["start_seconds"],
                    "end_seconds": detail["end_seconds"],
                    "remaining_excess_db": detail[
                        "requested_attenuation_db"
                    ],
                }
                for detail in pause_residual_details
                if float(detail["requested_attenuation_db"])
                >= residual_minimum_db
            ]
        if pause_residual_windows:
            final_pause_windows = [
                NoiseWindow(
                    start_seconds=float(item["start_seconds"]),
                    end_seconds=float(item["end_seconds"]),
                )
                for item in pause_residual_windows
            ]
            final_params, final_samples = _load_wave_samples(
                layout.clean_wav
            )
            final_pause_samples, final_pause_details = (
                apply_adaptive_breath_cleanup(
                    final_samples,
                    windows=final_pause_windows,
                    sample_rate=final_params.framerate,
                    channels=final_params.nchannels,
                    max_attenuation_db=float(
                        pause_residual_cleanup[
                            "final_pass_max_attenuation_db"
                        ]
                    ),
                    target_margin_db=float(
                        pause_residual_cleanup["target_margin_db"]
                    ),
                    context_ms=0.0,
                    fade_ms=float(
                        pause_residual_cleanup["final_fade_ms"]
                    ),
                    target_dbfs_override=target_dbfs,
                )
            )
            report["pause_cleanup"]["final_pass"] = final_pause_details
            if final_pause_samples != final_samples:
                _write_wave_samples(
                    layout.clean_wav,
                    final_params,
                    final_pause_samples,
                )
                transcript_command = commands[-1]
                _run_recorded_command(transcript_command, executed)
            assessment_windows = trim_noise_windows_for_assessment(
                final_pause_windows,
                edge_seconds=float(
                    pause_residual_cleanup[
                        "residual_assessment_edge_ms"
                    ]
                )
                / 1000.0,
            )
            report["pause_cleanup"]["assessment_windows"] = [
                {
                    "start_seconds": window.start_seconds,
                    "end_seconds": window.end_seconds,
                }
                for window in assessment_windows
            ]
            # Trimmed assessment windows must not create a false PASS: if every
            # residual core was too short to assess, fail closed with prior evidence.
            if not assessment_windows:
                report["pause_cleanup"]["failures"] = [
                    "confirmed_pause_residual_after_cleanup",
                    "empty_assessment_windows",
                ]
            else:
                _, pause_residual_details = apply_adaptive_breath_cleanup(
                    final_pause_samples,
                    windows=assessment_windows,
                    sample_rate=final_params.framerate,
                    channels=final_params.nchannels,
                    max_attenuation_db=residual_measure_max_db,
                    target_margin_db=float(
                        pause_residual_cleanup["target_margin_db"]
                    ),
                    context_ms=0.0,
                    fade_ms=0.0,
                    target_dbfs_override=target_dbfs,
                )
                pause_residual_windows = [
                    {
                        "start_seconds": detail["start_seconds"],
                        "end_seconds": detail["end_seconds"],
                        "remaining_excess_db": detail[
                            "requested_attenuation_db"
                        ],
                    }
                    for detail in pause_residual_details
                    if float(detail["requested_attenuation_db"])
                    >= residual_minimum_db
                ]
        report["pause_cleanup"]["final_residual_windows"] = (
            pause_residual_windows
        )
        pause_failures = list(report["pause_cleanup"].get("failures") or [])
        if pause_residual_windows and pause_residual_cleanup.get(
            "block_on_confirmed_residual",
            True,
        ):
            pause_failures.append("confirmed_pause_residual_after_cleanup")
        if pause_failures:
            report["pause_cleanup"]["status"] = "FAIL"
            report["pause_cleanup"]["failures"] = list(dict.fromkeys(pause_failures))
        else:
            report["pause_cleanup"]["status"] = "PASS"



def _run_breath_residual_cleanup(
    *,
    breath_windows: list[NoiseWindow],
    breath_cleanup_policy: dict[str, Any],
    layout: OutputLayout,
    runtime: RuntimeOptions,
    fallback_config: dict[str, Any],
    input_analysis: dict[str, Any],
    report: dict[str, Any],
    commands: list[list[str]],
    executed: list[dict[str, Any]],
) -> list[NoiseWindow]:
    authorized_cleanup_windows = (
        list(breath_windows) if breath_cleanup_policy.get("enabled") else []
    )
    if breath_cleanup_policy.get("enabled") and not breath_windows:
        report["breath_cleanup"]["status"] = "PASS"
        report["breath_cleanup"]["reason"] = "no_authorized_breath_windows"
        return authorized_cleanup_windows
    if breath_cleanup_policy.get("enabled") and breath_windows:
        residual_windows = detect_breath_onset_windows(
            layout.clean_wav,
            ffmpeg_bin=runtime.ffmpeg_bin,
            config=fallback_config,
        )
        residual_params, residual_samples = _load_wave_samples(layout.clean_wav)
        residual_windows, residual_spectral_evidence = (
            filter_noise_like_breath_windows(
                residual_samples,
                windows=residual_windows,
                sample_rate=residual_params.framerate,
                channels=residual_params.nchannels,
            )
        )
        report["breath_cleanup"]["residual_spectral_evidence"] = (
            residual_spectral_evidence
        )
        second_pass_windows, second_pass_evidence = build_effective_breath_windows(
            respiro_windows=[],
            auxiliary_windows=residual_windows,
            source_analysis=input_analysis,
        )
        report["breath_cleanup"]["residual_evidence"] = second_pass_evidence
        if second_pass_windows:
            authorized_cleanup_windows.extend(second_pass_windows)
            clean_params, clean_samples = residual_params, residual_samples
            second_cleaned, second_pass_details = apply_adaptive_breath_cleanup(
                clean_samples,
                windows=second_pass_windows,
                sample_rate=clean_params.framerate,
                channels=clean_params.nchannels,
                max_attenuation_db=float(
                    breath_cleanup_policy["second_pass_max_attenuation_db"]
                ),
                target_margin_db=float(breath_cleanup_policy["target_margin_db"]),
                context_ms=float(breath_cleanup_policy["context_ms"]),
                fade_ms=float(breath_cleanup_policy["fade_ms"]),
                absolute_floor_dbfs=(
                    None
                    if breath_cleanup_policy.get("absolute_floor_dbfs") is None
                    else float(breath_cleanup_policy["absolute_floor_dbfs"])
                ),
            )
            report["breath_cleanup"]["second_pass"] = second_pass_details
            if second_cleaned != clean_samples:
                _write_wave_samples(layout.clean_wav, clean_params, second_cleaned)
                transcript_command = commands[-1]
                _run_recorded_command(transcript_command, executed)

        final_residual_windows = detect_breath_onset_windows(
            layout.clean_wav,
            ffmpeg_bin=runtime.ffmpeg_bin,
            config=fallback_config,
        )
        final_params, final_samples = _load_wave_samples(layout.clean_wav)
        final_residual_windows, final_spectral_evidence = (
            filter_noise_like_breath_windows(
                final_samples,
                windows=final_residual_windows,
                sample_rate=final_params.framerate,
                channels=final_params.nchannels,
            )
        )
        report["breath_cleanup"]["final_spectral_evidence"] = (
            final_spectral_evidence
        )
        final_safe_residuals, _ = build_effective_breath_windows(
            respiro_windows=[],
            auxiliary_windows=final_residual_windows,
            source_analysis=input_analysis,
        )
        _, final_residual_details = apply_adaptive_breath_cleanup(
            final_samples,
            windows=final_safe_residuals,
            sample_rate=final_params.framerate,
            channels=final_params.nchannels,
            max_attenuation_db=36.0,
            target_margin_db=float(breath_cleanup_policy["target_margin_db"]),
            context_ms=float(breath_cleanup_policy["context_ms"]),
            fade_ms=0.0,
            absolute_floor_dbfs=(
                None
                if breath_cleanup_policy.get("absolute_floor_dbfs") is None
                else float(breath_cleanup_policy["absolute_floor_dbfs"])
            ),
        )
        minimum_excess_db = float(
            breath_cleanup_policy.get("residual_min_excess_db", 3.0)
        )
        confirmed_final_residuals = [
            window
            for window, detail in zip(
                final_safe_residuals,
                final_residual_details,
                strict=True,
            )
            if float(detail["requested_attenuation_db"]) >= minimum_excess_db
        ]
        if (
            confirmed_final_residuals
            and int(breath_cleanup_policy.get("max_retries", 1)) >= 2
        ):
            authorized_cleanup_windows.extend(confirmed_final_residuals)
            final_repaired, final_repair_details = apply_adaptive_breath_cleanup(
                final_samples,
                windows=confirmed_final_residuals,
                sample_rate=final_params.framerate,
                channels=final_params.nchannels,
                max_attenuation_db=36.0,
                target_margin_db=float(breath_cleanup_policy["target_margin_db"]),
                context_ms=float(breath_cleanup_policy["context_ms"]),
                fade_ms=float(breath_cleanup_policy["fade_ms"]),
                absolute_floor_dbfs=(
                    None
                    if breath_cleanup_policy.get("absolute_floor_dbfs") is None
                    else float(breath_cleanup_policy["absolute_floor_dbfs"])
                ),
            )
            report["breath_cleanup"]["final_repair_pass"] = final_repair_details
            if final_repaired != final_samples:
                _write_wave_samples(layout.clean_wav, final_params, final_repaired)
                transcript_command = commands[-1]
                _run_recorded_command(transcript_command, executed)
            final_residual_windows = detect_breath_onset_windows(
                layout.clean_wav,
                ffmpeg_bin=runtime.ffmpeg_bin,
                config=fallback_config,
            )
            final_params, final_samples = _load_wave_samples(layout.clean_wav)
            final_residual_windows, final_retry_spectral_evidence = (
                filter_noise_like_breath_windows(
                    final_samples,
                    windows=final_residual_windows,
                    sample_rate=final_params.framerate,
                    channels=final_params.nchannels,
                )
            )
            report["breath_cleanup"]["final_retry_spectral_evidence"] = (
                final_retry_spectral_evidence
            )
            final_safe_residuals, _ = build_effective_breath_windows(
                respiro_windows=[],
                auxiliary_windows=final_residual_windows,
                source_analysis=input_analysis,
            )
            _, final_residual_details = apply_adaptive_breath_cleanup(
                final_samples,
                windows=final_safe_residuals,
                sample_rate=final_params.framerate,
                channels=final_params.nchannels,
                max_attenuation_db=36.0,
                target_margin_db=float(breath_cleanup_policy["target_margin_db"]),
                context_ms=float(breath_cleanup_policy["context_ms"]),
                fade_ms=0.0,
                absolute_floor_dbfs=(
                    None
                    if breath_cleanup_policy.get("absolute_floor_dbfs") is None
                    else float(breath_cleanup_policy["absolute_floor_dbfs"])
                ),
            )
            confirmed_final_residuals = [
                window
                for window, detail in zip(
                    final_safe_residuals,
                    final_residual_details,
                    strict=True,
                )
                if float(detail["requested_attenuation_db"]) >= minimum_excess_db
            ]
        report["breath_cleanup"]["final_residual_assessment"] = (
            final_residual_details
        )
        report["breath_cleanup"]["final_residual_windows"] = [
            {
                "start_seconds": window.start_seconds,
                "end_seconds": window.end_seconds,
            }
            for window in confirmed_final_residuals
        ]
        if confirmed_final_residuals and breath_cleanup_policy.get(
            "block_on_confirmed_residual",
            True,
        ):
            report["breath_cleanup"]["status"] = "FAIL"
            report["breath_cleanup"]["failures"] = [
                "confirmed_breath_residual_after_retry"
            ]
        else:
            report["breath_cleanup"]["status"] = "PASS"

    return authorized_cleanup_windows


def process_media_file(
    input_file: Path,
    *,
    preset_name: str,
    preset: dict[str, Any],
    output_root: Path,
    runtime: RuntimeOptions,
    run_slug: str,
    input_metadata: dict[str, Any] | None = None,
    noise_windows: list[NoiseWindow] | None = None,
    respiro_repo: Path | None = None,
    respiro_weights: Path | None = None,
    attenuation_db: float = 18.0,
    respiro_threshold: float | None = None,
    respiro_min_length_ms: int | None = None,
    skip_spectramini: bool = False,
    skip_deepfilternet: bool = False,
) -> dict[str, Any]:
    metadata = input_metadata or ffprobe_media(input_file, runtime.ffprobe_bin)
    processing_format = resolve_processing_format(metadata, preset)
    layout = build_output_layout(input_path=input_file, output_root=output_root, run_slug=run_slug)
    resolved_noise_windows = _normalize_noise_windows(noise_windows)
    noise_sample_duration = _noise_sample_duration(resolved_noise_windows) if resolved_noise_windows else None
    respiro_stage = get_pipeline_stage(preset, "respiro")
    spectramini_stage = get_pipeline_stage(preset, "spectramini")
    deepfilternet_stage = get_pipeline_stage(preset, "deepfilternet")
    respiro_enabled = bool(respiro_stage.get("enabled", True))
    spectramini_enabled = bool(spectramini_stage.get("enabled", True))
    deepfilternet_enabled = bool(deepfilternet_stage.get("enabled", True))
    apply_spectramini = spectramini_enabled and not skip_spectramini
    apply_deepfilternet = deepfilternet_enabled and not skip_deepfilternet

    ffmpeg = runtime.ffmpeg_bin if runtime.dry_run else ensure_tool(runtime.ffmpeg_bin)
    python_bin = runtime.python_executable or resolve_repo_python(PROJECT_ROOT)
    if not runtime.dry_run:
        python_bin = ensure_tool(python_bin)

    commands: list[list[str]] = [
        build_ffmpeg_extract_command(
            input_path=input_file,
            raw_wav=layout.raw_wav,
            preset=preset,
            ffmpeg_bin=ffmpeg,
            processing_format=processing_format,
        ),
    ]
    if apply_deepfilternet:
        commands.append(
            build_deepfilternet_command(
                raw_wav=layout.breath_wav,
                output_dir=layout.deepfilternet_dir,
                preset=preset,
                python_executable=python_bin,
            )
        )
    if resolved_noise_windows:
        commands.append(
            build_ffmpeg_noise_sample_command(
                source_wav=layout.denoised_wav,
                noise_sample_wav=layout.noise_sample_wav,
                noise_windows=resolved_noise_windows,
                preset=preset,
                ffmpeg_bin=ffmpeg,
                processing_format=processing_format,
            )
        )
    commands.extend(
        build_ffmpeg_finalize_commands(
            denoised_wav=layout.denoised_wav,
            clean_wav=layout.clean_wav,
            transcript_mp3=layout.transcript_mp3,
            preset=preset,
            ffmpeg_bin=ffmpeg,
            noise_sample_wav=layout.noise_sample_wav if resolved_noise_windows else None,
            noise_sample_duration=noise_sample_duration,
            processing_format=processing_format,
        )
    )

    processing_steps = (
        _noise_print_processing_steps(
            preset,
            skip_spectramini=not apply_spectramini,
            skip_deepfilternet=not apply_deepfilternet,
        )
        if resolved_noise_windows
        else _processing_steps(
            preset,
            skip_spectramini=not apply_spectramini,
            skip_deepfilternet=not apply_deepfilternet,
        )
    )

    report: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "preset_name": preset_name,
        "backend": "local",
        "input_file": str(input_file),
        "input_metadata": metadata,
        "source_format": {
            "sample_rate": processing_format["sample_rate"],
            "channels": processing_format["channels"],
        },
        "processing_format": dict(processing_format),
        "processing_steps": processing_steps,
        "commands": commands,
        "dry_run": runtime.dry_run,
        "noise_windows": [
            {"start_seconds": window.start_seconds, "end_seconds": window.end_seconds}
            for window in resolved_noise_windows
        ],
        "noise_sample_duration_seconds": noise_sample_duration,
        "outputs": {
            "job_dir": str(layout.job_dir),
            "preprocess_dir": str(layout.preprocess_dir),
            "transcript_dir": str(layout.transcript_dir),
            "raw_wav": str(layout.raw_wav),
            "breath_wav": str(layout.breath_wav),
            "denoised_wav": str(layout.denoised_wav),
            "noise_sample_wav": str(layout.noise_sample_wav),
            "clean_wav": str(layout.clean_wav),
            "transcript_mp3": str(layout.transcript_mp3),
            "report_json": str(layout.report_json),
            "report_md": str(layout.report_md),
        },
        "silence_candidates": [],
        "loudnorm_summary": None,
        "notes": preset.get("notes", []),
        "respiro_detection_mode": "disabled",
        "respiro_assets_present": bool(
            respiro_repo and respiro_weights and respiro_repo.exists() and respiro_weights.exists()
        ),
        "respiro_attempted": False,
        "respiro_succeeded": False,
        "respiro_returncode": None,
        "respiro_error": None,
        "respiro_repo_path": str(respiro_repo) if respiro_repo else None,
        "respiro_weights_path": str(respiro_weights) if respiro_weights else None,
        "respiro_command": None,
        "respiro_stdout": "",
        "respiro_stderr": "",
        "respiro_breath_windows": [],
        "auxiliary_breath_windows": [],
        "breath_window_evidence": [],
        "breath_cleanup": {
            "status": "NOT_APPLICABLE",
            "first_pass": [],
            "second_pass": [],
            "final_pass": [],
            "final_residual_windows": [],
            "failures": [],
        },
        "spectramini_applied": False,
        "stage_status": {
            "respiro": {"enabled": respiro_enabled, "applied": False},
            "spectramini": {"enabled": spectramini_enabled, "applied": False},
            "deepfilternet": {"enabled": deepfilternet_enabled, "applied": False},
        },
        "pause_residual_cleanup_windows": [],
        "pause_cleanup": {
            "status": "NOT_APPLICABLE",
            "mode": None,
            "target_dbfs": None,
            "silence_floor_dbfs": None,
            "window_evidence": [],
            "first_pass": [],
            "second_pass": [],
            "final_residual_windows": [],
            "attenuation_stats": {},
            "failures": [],
        },
        "stereo_balance": {
            "status": "NOT_APPLICABLE",
            "before": None,
            "plan": None,
            "after": None,
            "preserve_channels": True,
        },
        "leading_pre_speech_seal": {
            "status": "NOT_APPLICABLE",
            "stage": None,
        },
        "leading_pre_speech_seal_post": {
            "status": "NOT_APPLICABLE",
            "stage": None,
        },
        "input_diagnostics": None,
        "output_diagnostics": None,
        "quality_guard": None,
    }

    if runtime.dry_run:
        return report

    _ensure_directories(layout)
    executed: list[dict[str, Any]] = []

    extract_command = commands[0]
    _run_recorded_command(extract_command, executed)

    raw_params, raw_samples = _load_wave_samples(layout.raw_wav)
    stereo_policy = (
        preset.get("filters", {}).get("stereo_balance")
        if isinstance(preset.get("filters"), dict)
        else None
    )
    raw_samples, stereo_balance_report = _run_stereo_balance_stage(
        raw_wav=layout.raw_wav,
        samples=raw_samples,
        sample_rate=raw_params.framerate,
        channels=raw_params.nchannels,
        policy=stereo_policy if isinstance(stereo_policy, dict) else None,
    )
    report["stereo_balance"] = stereo_balance_report
    input_mono = _analysis_samples(raw_params, raw_samples)
    input_analysis = analyze_pcm16_samples(
        input_mono,
        sample_rate=raw_params.framerate,
        frame_ms=10.0,
    )
    report["input_diagnostics"] = public_audio_diagnostics(input_analysis)
    adaptive_decision = select_adaptive_profile(input_analysis)
    report["adaptive_profile"] = adaptive_decision
    if preset_name == "natural":
        active_preset = apply_adaptive_profile(preset, adaptive_decision)
        input_adaptations: list[str] = []
    else:
        active_preset, input_adaptations = apply_input_safety_overrides(
            preset_name,
            preset,
            input_analysis,
        )
    report["input_adaptations"] = input_adaptations
    if active_preset is not preset:
        tail_start = 1 + int(apply_deepfilternet)
        adaptive_tail: list[list[str]] = []
        if resolved_noise_windows:
            adaptive_tail.append(
                build_ffmpeg_noise_sample_command(
                    source_wav=layout.denoised_wav,
                    noise_sample_wav=layout.noise_sample_wav,
                    noise_windows=resolved_noise_windows,
                    preset=active_preset,
                    ffmpeg_bin=ffmpeg,
                    processing_format=processing_format,
                )
            )
        adaptive_tail.extend(
            build_ffmpeg_finalize_commands(
                denoised_wav=layout.denoised_wav,
                clean_wav=layout.clean_wav,
                transcript_mp3=layout.transcript_mp3,
                preset=active_preset,
                ffmpeg_bin=ffmpeg,
                noise_sample_wav=layout.noise_sample_wav if resolved_noise_windows else None,
                noise_sample_duration=noise_sample_duration,
                processing_format=processing_format,
            )
        )
        commands[tail_start:] = adaptive_tail
        report["commands"] = commands
        report["processing_steps"] = (
            _noise_print_processing_steps(
                active_preset,
                skip_spectramini=not apply_spectramini,
                skip_deepfilternet=not apply_deepfilternet,
            )
            if resolved_noise_windows
            else _processing_steps(
                active_preset,
                skip_spectramini=not apply_spectramini,
                skip_deepfilternet=not apply_deepfilternet,
            )
        )

    fallback_config = {
        **DEFAULT_BREATH_FALLBACK_CONFIG,
        **preset.get("filters", {}).get("breath_onset_cleanup", {}),
    }
    breath_cleanup_policy = active_preset.get("filters", {}).get(
        "breath_cleanup_policy",
        {},
    )
    breath_windows: list[NoiseWindow] = []
    respiro_windows: list[NoiseWindow] = []
    if respiro_enabled:
        threshold = float(respiro_threshold if respiro_threshold is not None else respiro_stage.get("threshold", 0.064))
        min_length = int(respiro_min_length_ms if respiro_min_length_ms is not None else respiro_stage.get("min_length_ms", 20))
        respiro_result = run_respiro_or_fallback_detection(
            audio_path=layout.raw_wav,
            ffmpeg_bin=runtime.ffmpeg_bin,
            respiro_repo=respiro_repo,
            respiro_weights=respiro_weights,
            python_executable=python_bin,
            threshold=threshold,
            min_length_ms=min_length,
            fallback_config=fallback_config,
        )
        respiro_windows = respiro_result.windows
        breath_windows = respiro_windows
        report["respiro_detection_mode"] = respiro_result.mode
        report["respiro_assets_present"] = respiro_result.assets_present
        report["respiro_attempted"] = respiro_result.attempted
        report["respiro_succeeded"] = respiro_result.succeeded
        report["respiro_returncode"] = respiro_result.returncode
        report["respiro_error"] = respiro_result.error
        report["respiro_command"] = respiro_result.command
        report["respiro_stdout"] = respiro_result.stdout
        report["respiro_stderr"] = respiro_result.stderr
    auxiliary_windows: list[NoiseWindow] = []
    if breath_cleanup_policy.get("enabled") and breath_cleanup_policy.get(
        "merge_auxiliary_detection",
        True,
    ):
        auxiliary_windows = detect_breath_onset_windows(
            layout.raw_wav,
            ffmpeg_bin=runtime.ffmpeg_bin,
            config=fallback_config,
        )
        auxiliary_windows, auxiliary_spectral_evidence = (
            filter_noise_like_breath_windows(
                raw_samples,
                windows=auxiliary_windows,
                sample_rate=raw_params.framerate,
                channels=raw_params.nchannels,
            )
        )
        report["auxiliary_spectral_evidence"] = auxiliary_spectral_evidence
        breath_windows, breath_evidence = build_effective_breath_windows(
            respiro_windows=respiro_windows,
            auxiliary_windows=auxiliary_windows,
            source_analysis=input_analysis,
        )
        report["breath_window_evidence"] = breath_evidence
    report["respiro_breath_windows"] = [
        {"start_seconds": window.start_seconds, "end_seconds": window.end_seconds}
        for window in respiro_windows
    ]
    report["auxiliary_breath_windows"] = [
        {"start_seconds": window.start_seconds, "end_seconds": window.end_seconds}
        for window in auxiliary_windows
    ]
    report["breath_onset_windows"] = [
        {"start_seconds": window.start_seconds, "end_seconds": window.end_seconds}
        for window in breath_windows
    ]
    report["stage_status"]["respiro"]["applied"] = bool(
        respiro_enabled
        and report.get("respiro_succeeded")
        and (
            not breath_cleanup_policy.get("enabled")
            or any(
                "respiro" in item.get("sources", [])
                and item.get("decision") == "accepted"
                for item in report["breath_window_evidence"]
            )
        )
    )

    shutil.copyfile(layout.raw_wav, layout.breath_wav)
    if layout.breath_wav.exists() and apply_spectramini:
        params, samples = _load_wave_samples(layout.breath_wav)
        if breath_cleanup_policy.get("enabled"):
            cleaned, first_pass_details = apply_adaptive_breath_cleanup(
                samples,
                windows=breath_windows,
                sample_rate=params.framerate,
                channels=params.nchannels,
                max_attenuation_db=float(
                    breath_cleanup_policy["first_pass_max_attenuation_db"]
                ),
                target_margin_db=float(breath_cleanup_policy["target_margin_db"]),
                context_ms=float(breath_cleanup_policy["context_ms"]),
                fade_ms=float(breath_cleanup_policy["fade_ms"]),
                absolute_floor_dbfs=(
                    None
                    if breath_cleanup_policy.get("absolute_floor_dbfs") is None
                    else float(breath_cleanup_policy["absolute_floor_dbfs"])
                ),
            )
            report["breath_cleanup"]["first_pass"] = first_pass_details
        else:
            cleaned = apply_spectramini_style_cleanup_to_samples(
                samples,
                breath_windows=breath_windows,
                sample_rate=params.framerate,
                attenuation_db=attenuation_db,
                mouth_declick_sensitivity=float(spectramini_stage.get("mouth_declick_sensitivity", 0.55)),
                fade_ms=float(fallback_config.get("fade_ms", 14.0)),
                channels=raw_params.nchannels,
            )
        before_breath_dbfs = _window_rms_dbfs(
            samples,
            breath_windows,
            sample_rate=params.framerate,
            channels=params.nchannels,
        )
        after_breath_dbfs = _window_rms_dbfs(
            cleaned,
            breath_windows,
            sample_rate=params.framerate,
            channels=params.nchannels,
        )
        report["breath_attenuation_db"] = (
            round(before_breath_dbfs - after_breath_dbfs, 3)
            if before_breath_dbfs is not None and after_breath_dbfs is not None
            else 0.0
        )
        _write_wave_samples(layout.breath_wav, params, cleaned)
        samples_changed = cleaned != samples
        report["spectramini_applied"] = samples_changed
        report["stage_status"]["spectramini"]["applied"] = samples_changed

    report["leading_pre_speech_seal"] = {
        "stage": "pre_mastering",
        "status": "SKIP",
        "reason": "deferred_to_post_mastering_difference_evidence",
        "guard_exclusion_allowed": False,
    }

    if apply_deepfilternet:
        deepfilter_command = commands[1]
        _run_recorded_command(deepfilter_command, executed)

        detected_df_wav = _find_single_wav(layout.deepfilternet_dir)
        if detected_df_wav.resolve() != layout.denoised_wav.resolve():
            detected_df_wav.replace(layout.denoised_wav)
        repair_windows: list[NoiseWindow] = []
        if layout.breath_wav.exists() and layout.denoised_wav.exists():
            raw_params, raw_samples = _load_wave_samples(layout.breath_wav)
            denoised_params, denoised_samples = _load_wave_samples(layout.denoised_wav)
            if raw_params.nchannels == denoised_params.nchannels == 1:
                repaired_samples, repair_windows = repair_deepfilternet_speech_dropouts(
                    raw_samples,
                    denoised_samples,
                    sample_rate=denoised_params.framerate,
                )
                if repair_windows:
                    _write_wave_samples(layout.denoised_wav, denoised_params, repaired_samples)
            else:
                report["deepfilternet_dropout_repair_skipped_reason"] = "multi_channel_input_requires_channel_safe_model_repair"
        report["deepfilternet_dropout_repair_windows"] = [
            {"start_seconds": window.start_seconds, "end_seconds": window.end_seconds}
            for window in repair_windows
        ]
        report["stage_status"]["deepfilternet"]["applied"] = True
        remaining_commands = commands[2:]
    else:
        shutil.copyfile(layout.breath_wav, layout.denoised_wav)
        report["deepfilternet_dropout_repair_windows"] = []
        remaining_commands = commands[1:]
    for command in remaining_commands:
        _run_recorded_command(command, executed)
    loudnorm_execution = executed[-2]

    authorized_cleanup_windows = _run_breath_residual_cleanup(
        breath_windows=breath_windows,
        breath_cleanup_policy=breath_cleanup_policy,
        layout=layout,
        runtime=runtime,
        fallback_config=fallback_config,
        input_analysis=input_analysis,
        report=report,
        commands=commands,
        executed=executed,
    )
    breath_onset_cleanup = preset.get("filters", {}).get("breath_onset_cleanup", {})
    if breath_onset_cleanup.get("enabled"):
        breath_windows = detect_breath_onset_windows(
            layout.clean_wav,
            ffmpeg_bin=runtime.ffmpeg_bin,
            config=breath_onset_cleanup,
        )
        report["postprocess_breath_onset_windows"] = [
            {"start_seconds": window.start_seconds, "end_seconds": window.end_seconds}
            for window in breath_windows
        ]
        if breath_windows:
            duck_audio_file_in_place(
                layout.clean_wav,
                windows=breath_windows,
                floor_gain=float(breath_onset_cleanup["floor_gain"]),
                fade_ms=float(breath_onset_cleanup["fade_ms"]),
            )
    else:
        report["postprocess_breath_onset_windows"] = []

    _run_pause_cleanup(
        layout=layout,
        active_preset=active_preset,
        input_analysis=input_analysis,
        runtime=runtime,
        report=report,
        authorized_cleanup_windows=authorized_cleanup_windows,
        commands=commands,
        executed=executed,
    )

    post_seal_applied = _run_leading_pre_speech_seal(
        target_wav=layout.clean_wav,
        raw_wav=layout.raw_wav,
        report=report,
        report_key="leading_pre_speech_seal_post",
        stage="post_mastering",
        source_analysis=input_analysis,
    )
    if post_seal_applied:
        _run_recorded_command(commands[-1], executed)

    report["executed"] = executed
    report["loudnorm_summary"] = extract_loudnorm_summary(loudnorm_execution["stderr"])
    report["output_metadata"] = ffprobe_media(layout.clean_wav, runtime.ffprobe_bin)
    clean_params, clean_samples = _load_wave_samples(layout.clean_wav)
    output_mono = _analysis_samples(clean_params, clean_samples)
    output_analysis = analyze_pcm16_samples(
        output_mono,
        sample_rate=clean_params.framerate,
        frame_ms=10.0,
    )
    report["output_diagnostics"] = public_audio_diagnostics(output_analysis)
    report["quality_guard"] = {
        "evidence_level": "DIRECTLY VERIFIED",
        **compare_audio_preservation(
            input_analysis,
            output_analysis,
            reference_format=report["source_format"],
            processed_format={
                "sample_rate": report["output_metadata"].get("sample_rate"),
                "channels": report["output_metadata"].get("channels"),
            },
            reference_samples=input_mono,
            processed_samples=output_mono,
            sample_rate=raw_params.framerate,
            excluded_windows=authorized_cleanup_windows,
        ),
    }
    cleanup_failures = [
        *list(report["breath_cleanup"].get("failures") or []),
        *list(report["pause_cleanup"].get("failures") or []),
    ]
    if cleanup_failures:
        report["quality_guard"]["failures"] = list(
            dict.fromkeys(
                [*report["quality_guard"].get("failures", []), *cleanup_failures]
            )
        )
        report["quality_guard"]["status"] = "FAIL"
        report["quality_guard"]["release_blocked"] = True
    report["channel_layout_preserved"] = report["quality_guard"].get("channel_layout_preserved")

    analysis = preset.get("analysis", {})
    if analysis.get("silence_candidates"):
        report["silence_candidates"] = detect_silence_candidates(
            layout.clean_wav,
            ffmpeg_bin=runtime.ffmpeg_bin,
            threshold_db=analysis["silence_threshold_db"],
            min_duration=analysis["silence_min_duration"],
        )

    layout.report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    layout.report_md.write_text(render_markdown_report(report), encoding="utf-8")
    return report


def render_markdown_report(report: dict[str, Any]) -> str:
    silence_lines = ["- none"] if not report.get("silence_candidates") else [
        f"- {item['start_seconds']}s -> {item['end_seconds']}s ({item['duration_seconds']}s)"
        for item in report["silence_candidates"]
    ]
    noise_window_lines = ["- none"] if not report.get("noise_windows") else [
        f"- {item['start_seconds']}s -> {item['end_seconds']}s"
        for item in report["noise_windows"]
    ]
    breath_onset_lines = ["- none"] if not report.get("breath_onset_windows") else [
        f"- {item['start_seconds']}s -> {item['end_seconds']}s"
        for item in report["breath_onset_windows"]
    ]
    loudnorm_block = json.dumps(report["loudnorm_summary"], indent=2) if report.get("loudnorm_summary") else "{}"
    input_diagnostics = report.get("input_diagnostics") or {}
    output_diagnostics = report.get("output_diagnostics") or {}
    quality_guard = report.get("quality_guard") or {}
    breath_cleanup = report.get("breath_cleanup") or {}
    pause_cleanup = report.get("pause_cleanup") or {}
    recommendation_lines = [
        f"- `{item.get('priority', 'unknown')}` `{item.get('action', 'unknown')}`: {item.get('reason', '')}"
        for item in input_diagnostics.get("recommendations", [])
    ] or ["- unavailable"]
    quality_guard_block = json.dumps(quality_guard, indent=2, ensure_ascii=False) if quality_guard else "{}"
    return "\n".join(
        [
            f"# Audio Process Report: {Path(report['input_file']).name}",
            "",
            "## Summary",
            f"- Preset: `{report['preset_name']}`",
            f"- Backend: `{report['backend']}`",
            f"- Input: `{report['input_file']}`",
            f"- Clean WAV: `{report['outputs']['clean_wav']}`",
            f"- Transcript MP3: `{report['outputs']['transcript_mp3']}`",
            f"- Noise sample WAV: `{report['outputs']['noise_sample_wav']}`",
            "",
            "## Processing Steps",
            *[f"- {step}" for step in report["processing_steps"]],
            "",
            "## Input Metadata",
            f"- Duration: `{report['input_metadata'].get('duration_seconds')}`",
            f"- Sample rate: `{report['input_metadata'].get('sample_rate')}`",
            f"- Channels: `{report['input_metadata'].get('channels')}`",
            "",
            "## Automatic Diagnosis",
            f"- Evidence: `{input_diagnostics.get('evidence_level', 'UNVERIFIED')}`",
            f"- Source noise floor: `{input_diagnostics.get('noise_floor_dbfs', 'n/a')} dBFS`",
            f"- Source speech level: `{input_diagnostics.get('speech_level_dbfs', 'n/a')} dBFS`",
            f"- Estimated SNR: `{input_diagnostics.get('estimated_snr_db', 'n/a')} dB`",
            f"- Active speech range: `{input_diagnostics.get('active_dynamic_range_db', 'n/a')} dB`",
            f"- Source clipped samples: `{input_diagnostics.get('clipped_sample_count', 'n/a')}`",
            f"- Output clipped samples: `{output_diagnostics.get('clipped_sample_count', 'n/a')}`",
            "- Recommendations:",
            *recommendation_lines,
            "",
            "## Preservation Guard",
            "```json",
            quality_guard_block,
            "```",
            "",
            "## Noise Windows",
            *noise_window_lines,
            "",
            "## Noise Sample",
            f"- Duration: `{report.get('noise_sample_duration_seconds')}`",
            "",
            "## Respiro Detection",
            f"- Mode: `{report.get('respiro_detection_mode')}`",
            f"- Assets present: `{report.get('respiro_assets_present')}`",
            f"- Attempted: `{report.get('respiro_attempted')}`",
            f"- Succeeded: `{report.get('respiro_succeeded')}`",
            f"- Return code: `{report.get('respiro_returncode')}`",
            f"- Error: `{report.get('respiro_error')}`",
            "",
            "## Breath Onset Windows",
            *breath_onset_lines,
            "",
            "## Breath Cleanup Closed Loop",
            "```json",
            json.dumps(breath_cleanup, indent=2, ensure_ascii=False),
            "```",
            "",
            "## Pause Transition Cleanup",
            "```json",
            json.dumps(pause_cleanup, indent=2, ensure_ascii=False),
            "```",
            "",
            "## Loudnorm Summary",
            "```json",
            loudnorm_block,
            "```",
            "",
            "## Silence Candidates",
            *silence_lines,
            "",
            "## Commands",
            "```json",
            json.dumps(report["commands"], indent=2, ensure_ascii=False),
            "```",
        ]
    )


def build_batch_summary(
    *,
    preset_name: str,
    preset_description: str,
    input_path: Path,
    output_root: Path,
    reports: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "preset_name": preset_name,
        "preset_description": preset_description,
        "input_path": str(input_path),
        "output_root": str(output_root),
        "files_processed": len(reports),
        "backend": "local",
        "reports": [
            {
                "input_file": report["input_file"],
                "clean_wav": report["outputs"]["clean_wav"],
                "transcript_mp3": report["outputs"].get("transcript_mp3", ""),
                "report_json": report["outputs"]["report_json"],
                "report_md": report["outputs"]["report_md"],
            }
            for report in reports
        ],
    }


def render_batch_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Batch Summary",
        "",
        f"- Preset: `{summary['preset_name']}`",
        f"- Description: `{summary['preset_description']}`",
        f"- Input path: `{summary['input_path']}`",
        f"- Output root: `{summary['output_root']}`",
        f"- Files processed: `{summary['files_processed']}`",
        "",
        "## Files",
    ]
    for item in summary["reports"]:
        lines.extend(
            [
                f"- `{item['input_file']}`",
                f"  - clean wav: `{item['clean_wav']}`",
                f"  - transcript mp3: `{item['transcript_mp3']}`",
                f"  - report: `{item['report_md']}`",
            ]
        )
    return "\n".join(lines)
