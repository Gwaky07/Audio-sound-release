from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .config import PROJECT_ROOT, load_env_file, resolve_binary
from .skill_workflow import _safe_windows_stem


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "修音成品"
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_CHANNELS = 2
DEFAULT_CROSSFADE_MS = 35.0


@dataclass(frozen=True)
class CutWindow:
    start_seconds: float
    end_seconds: float | None = None


@dataclass(frozen=True)
class KeepInterval:
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class DeliveryPaths:
    wav: Path
    mp3: Path
    mp4: Path
    report: Path


@dataclass(frozen=True)
class MediaProbe:
    duration_seconds: float
    has_audio: bool
    has_video: bool


@dataclass(frozen=True)
class SpliceResult:
    samples: array
    crossfade_frames: list[int]


def parse_time_seconds(value: str | int | float) -> float:
    if isinstance(value, (int, float)):
        seconds = float(value)
    else:
        raw = str(value).strip().replace("：", ":")
        if not raw:
            raise ValueError("Time value cannot be empty")
        if ":" in raw:
            parts = raw.split(":")
            if len(parts) > 3:
                raise ValueError(f"Invalid time value: {value!r}")
            total = 0.0
            for part in parts:
                if part == "":
                    raise ValueError(f"Invalid time value: {value!r}")
                total = total * 60.0 + float(part)
            seconds = total
        else:
            seconds = float(raw)
    if seconds < 0:
        raise ValueError(f"Time value must be non-negative: {value!r}")
    return seconds


def parse_cut_spec(value: str) -> CutWindow:
    raw = value.strip().replace("，", ",")
    if "," in raw:
        start_raw, end_raw = raw.split(",", 1)
    elif "-" in raw:
        start_raw, end_raw = raw.split("-", 1)
    else:
        raise ValueError(f"Cut spec must be START,END or START-END: {value!r}")

    start_seconds = parse_time_seconds(start_raw)
    end_seconds = parse_time_seconds(end_raw) if end_raw.strip() else None
    if end_seconds is not None and end_seconds <= start_seconds:
        raise ValueError(f"Cut end must be greater than start: {value!r}")
    return CutWindow(start_seconds=start_seconds, end_seconds=end_seconds)


def parse_cut_payload(payload: Any) -> CutWindow:
    if isinstance(payload, CutWindow):
        return payload
    if isinstance(payload, str):
        return parse_cut_spec(payload)
    if not isinstance(payload, dict):
        raise ValueError(f"Cut entry must be a string or object: {payload!r}")
    if "start" not in payload:
        raise ValueError(f"Cut entry is missing 'start': {payload!r}")
    end_value = payload.get("end")
    return CutWindow(
        start_seconds=parse_time_seconds(payload["start"]),
        end_seconds=parse_time_seconds(end_value) if end_value is not None else None,
    )


def normalize_cuts(duration_seconds: float, cuts: Sequence[CutWindow]) -> list[CutWindow]:
    if duration_seconds <= 0:
        raise ValueError("Media duration must be greater than zero")
    normalized: list[tuple[float, float]] = []
    for cut in cuts:
        start_seconds = min(max(cut.start_seconds, 0.0), duration_seconds)
        end_seconds = duration_seconds if cut.end_seconds is None else min(cut.end_seconds, duration_seconds)
        if end_seconds <= start_seconds:
            continue
        normalized.append((start_seconds, end_seconds))

    normalized.sort()
    merged: list[tuple[float, float]] = []
    for start_seconds, end_seconds in normalized:
        if not merged or start_seconds > merged[-1][1]:
            merged.append((start_seconds, end_seconds))
            continue
        previous_start, previous_end = merged[-1]
        merged[-1] = (previous_start, max(previous_end, end_seconds))

    return [CutWindow(start, end) for start, end in merged]


def build_keep_intervals(duration_seconds: float, cuts: Sequence[CutWindow]) -> list[KeepInterval]:
    normalized = normalize_cuts(duration_seconds, cuts)
    keep_intervals: list[KeepInterval] = []
    cursor = 0.0
    for cut in normalized:
        end_seconds = duration_seconds if cut.end_seconds is None else cut.end_seconds
        if cut.start_seconds > cursor:
            keep_intervals.append(KeepInterval(cursor, cut.start_seconds))
        cursor = max(cursor, end_seconds)
    if cursor < duration_seconds:
        keep_intervals.append(KeepInterval(cursor, duration_seconds))
    if not keep_intervals:
        raise ValueError("Cuts remove the entire file; keep at least one audible segment")
    return keep_intervals


def reserve_delivery_paths(
    delivery_dir: Path,
    input_stem: str,
    *,
    output_stem: str | None = None,
    delivery_prefix: str = "修音版",
) -> DeliveryPaths:
    delivery_dir.mkdir(parents=True, exist_ok=True)
    safe_prefix = _safe_windows_stem(delivery_prefix) or "修音版"
    safe_stem = _safe_windows_stem(output_stem or input_stem) or "audio"
    base_label = f"{safe_prefix}_{safe_stem}"
    for index in range(10000):
        suffix = "" if index == 0 else f"_{index:02d}"
        candidate = delivery_dir / f"{base_label}{suffix}"
        paths = DeliveryPaths(
            wav=candidate.with_suffix(".wav"),
            mp3=candidate.with_suffix(".mp3"),
            mp4=candidate.with_suffix(".mp4"),
            report=delivery_dir / f"{candidate.name}_segment-removal-report.json",
        )
        if not any(path.exists() for path in (paths.wav, paths.mp3, paths.mp4, paths.report)):
            return paths
    raise RuntimeError(f"Could not reserve a unique delivery filename under {delivery_dir}")


def _decode_process_output(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return value.decode("utf-8", errors="replace")


def _run_command(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return subprocess.CompletedProcess(
            result.args,
            result.returncode,
            _decode_process_output(result.stdout),
            _decode_process_output(result.stderr),
        )
    except subprocess.CalledProcessError as exc:
        message = "\n".join(
            part
            for part in (
                f"Command failed with exit code {exc.returncode}: {' '.join(command)}",
                _decode_process_output(exc.stdout).strip(),
                _decode_process_output(exc.stderr).strip(),
            )
            if part
        )
        raise RuntimeError(message) from exc


def probe_media(path: Path, *, ffprobe_bin: str) -> MediaProbe:
    result = _run_command(
        [
            ffprobe_bin,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
    )
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    has_audio = any(stream.get("codec_type") == "audio" for stream in streams)
    has_video = any(stream.get("codec_type") == "video" for stream in streams)
    duration_values = [
        stream.get("duration")
        for stream in streams
        if stream.get("duration") not in (None, "N/A")
    ]
    format_duration = payload.get("format", {}).get("duration")
    if format_duration not in (None, "N/A"):
        duration_values.append(format_duration)
    durations = [float(value) for value in duration_values if float(value) > 0]
    if not durations:
        raise ValueError(f"Could not determine media duration: {path}")
    return MediaProbe(duration_seconds=max(durations), has_audio=has_audio, has_video=has_video)


def extract_audio_wav(
    input_path: Path,
    output_path: Path,
    *,
    ffmpeg_bin: str,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
) -> None:
    _run_command(
        [
            ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-nostdin",
            "-i",
            str(input_path),
            "-vn",
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]
    )


def _read_pcm16_wav(path: Path) -> tuple[wave._wave_params, array]:
    with wave.open(str(path), "rb") as wav_file:
        params = wav_file.getparams()
        if params.sampwidth != 2:
            raise ValueError(f"Only 16-bit PCM WAV is supported: {path}")
        samples = array("h")
        samples.frombytes(wav_file.readframes(params.nframes))
    if sys.byteorder != "little":
        samples.byteswap()
    return params, samples


def _write_pcm16_wav(path: Path, params: wave._wave_params, samples: array) -> None:
    output_samples = array("h", samples)
    if sys.byteorder != "little":
        output_samples.byteswap()
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setparams(params)
        wav_file.writeframes(output_samples.tobytes())


def _clip_pcm16(value: float) -> int:
    return max(-32768, min(32767, int(round(value))))


def splice_pcm16_samples(
    samples: array,
    *,
    channels: int,
    sample_rate: int,
    keep_intervals: Sequence[KeepInterval],
    crossfade_ms: float = DEFAULT_CROSSFADE_MS,
) -> SpliceResult:
    if channels <= 0:
        raise ValueError("Channel count must be positive")
    fade_frames_requested = max(0, int(round(sample_rate * crossfade_ms / 1000.0)))
    output = array("h")
    crossfade_frames: list[int] = []

    for interval_index, interval in enumerate(keep_intervals):
        start_frame = max(0, int(round(interval.start_seconds * sample_rate)))
        end_frame = max(start_frame, int(round(interval.end_seconds * sample_rate)))
        segment = samples[start_frame * channels : end_frame * channels]
        if not segment:
            continue
        if interval_index == 0 or not output:
            output.extend(segment)
            continue

        output_frames = len(output) // channels
        segment_frames = len(segment) // channels
        fade_frames = min(fade_frames_requested, output_frames, segment_frames)
        crossfade_frames.append(fade_frames)
        if fade_frames <= 0:
            output.extend(segment)
            continue

        prefix_sample_count = len(output) - fade_frames * channels
        prefix = output[:prefix_sample_count]
        outgoing = output[prefix_sample_count:]
        incoming = segment[: fade_frames * channels]
        mixed = array("h")
        for frame_index in range(fade_frames):
            if fade_frames == 1:
                progress = 0.5
            else:
                progress = frame_index / (fade_frames - 1)
            out_gain = math.cos(progress * math.pi / 2.0)
            in_gain = math.sin(progress * math.pi / 2.0)
            for channel_index in range(channels):
                sample_index = frame_index * channels + channel_index
                mixed.append(
                    _clip_pcm16(outgoing[sample_index] * out_gain + incoming[sample_index] * in_gain)
                )
        output = prefix
        output.extend(mixed)
        output.extend(segment[fade_frames * channels :])

    if not output:
        raise ValueError("Splice result is empty")
    return SpliceResult(samples=output, crossfade_frames=crossfade_frames)


def splice_wav_file(
    source_wav: Path,
    output_wav: Path,
    *,
    keep_intervals: Sequence[KeepInterval],
    crossfade_ms: float = DEFAULT_CROSSFADE_MS,
) -> SpliceResult:
    params, samples = _read_pcm16_wav(source_wav)
    result = splice_pcm16_samples(
        samples,
        channels=params.nchannels,
        sample_rate=params.framerate,
        keep_intervals=keep_intervals,
        crossfade_ms=crossfade_ms,
    )
    updated_params = params._replace(nframes=len(result.samples) // params.nchannels)
    _write_pcm16_wav(output_wav, updated_params, result.samples)
    return result


def measure_wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wav_file:
        return wav_file.getnframes() / float(wav_file.getframerate())


def export_mp3(source_wav: Path, output_mp3: Path, *, ffmpeg_bin: str) -> None:
    _run_command(
        [
            ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-nostdin",
            "-i",
            str(source_wav),
            "-codec:a",
            "libmp3lame",
            "-q:a",
            "2",
            str(output_mp3),
        ]
    )


def _format_seconds(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".") or "0"


def build_video_trim_filter(keep_intervals: Sequence[KeepInterval]) -> str:
    if not keep_intervals:
        raise ValueError("At least one keep interval is required")
    if len(keep_intervals) == 1:
        interval = keep_intervals[0]
        return (
            f"[0:v]trim=start={_format_seconds(interval.start_seconds)}:"
            f"end={_format_seconds(interval.end_seconds)},setpts=PTS-STARTPTS[vout]"
        )
    parts: list[str] = []
    concat_inputs: list[str] = []
    for index, interval in enumerate(keep_intervals):
        label = f"v{index}"
        parts.append(
            f"[0:v]trim=start={_format_seconds(interval.start_seconds)}:"
            f"end={_format_seconds(interval.end_seconds)},setpts=PTS-STARTPTS[{label}]"
        )
        concat_inputs.append(f"[{label}]")
    parts.append(f"{''.join(concat_inputs)}concat=n={len(keep_intervals)}:v=1:a=0[vout]")
    return ";".join(parts)


def export_synced_mp4(
    input_path: Path,
    edited_wav: Path,
    output_mp4: Path,
    *,
    keep_intervals: Sequence[KeepInterval],
    ffmpeg_bin: str,
) -> None:
    _run_command(
        [
            ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-nostdin",
            "-i",
            str(input_path),
            "-i",
            str(edited_wav),
            "-filter_complex",
            build_video_trim_filter(keep_intervals),
            "-map",
            "[vout]",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            "-movflags",
            "+faststart",
            str(output_mp4),
        ]
    )


def _cuts_to_report(cuts: Sequence[CutWindow]) -> list[dict[str, float | None]]:
    return [
        {
            "start_seconds": cut.start_seconds,
            "end_seconds": cut.end_seconds,
        }
        for cut in cuts
    ]


def _intervals_to_report(intervals: Sequence[KeepInterval]) -> list[dict[str, float]]:
    return [
        {
            "start_seconds": interval.start_seconds,
            "end_seconds": interval.end_seconds,
            "duration_seconds": interval.end_seconds - interval.start_seconds,
        }
        for interval in intervals
    ]


def process_segment_removal_job(
    input_path: Path,
    *,
    cuts: Sequence[CutWindow],
    removed_phrase: str | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    output_stem: str | None = None,
    crossfade_ms: float = DEFAULT_CROSSFADE_MS,
    keep_work: bool = False,
    ffmpeg_bin: str = "ffmpeg",
    ffprobe_bin: str = "ffprobe",
) -> dict[str, Any]:
    input_path = input_path.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input media not found: {input_path}")
    if not cuts:
        raise ValueError("At least one cut is required")

    probe = probe_media(input_path, ffprobe_bin=ffprobe_bin)
    if not probe.has_audio:
        raise ValueError(f"Input media has no audio stream: {input_path}")

    normalized_cuts = normalize_cuts(probe.duration_seconds, cuts)
    if not normalized_cuts:
        raise ValueError("No cut window intersects the input media duration")
    keep_intervals = build_keep_intervals(probe.duration_seconds, normalized_cuts)
    output_dir = output_dir.expanduser().resolve()
    delivery_paths = reserve_delivery_paths(output_dir, input_path.stem, output_stem=output_stem)
    temp_parent = PROJECT_ROOT / "output"
    temp_parent.mkdir(parents=True, exist_ok=True)
    work_root = Path(tempfile.mkdtemp(prefix="segment-removal-", dir=str(temp_parent)))
    source_wav = work_root / "source.wav"

    try:
        extract_audio_wav(input_path, source_wav, ffmpeg_bin=ffmpeg_bin)
        splice_result = splice_wav_file(
            source_wav,
            delivery_paths.wav,
            keep_intervals=keep_intervals,
            crossfade_ms=crossfade_ms,
        )
        export_mp3(delivery_paths.wav, delivery_paths.mp3, ffmpeg_bin=ffmpeg_bin)
        mp4_path: Path | None = None
        if probe.has_video:
            export_synced_mp4(
                input_path,
                delivery_paths.wav,
                delivery_paths.mp4,
                keep_intervals=keep_intervals,
                ffmpeg_bin=ffmpeg_bin,
            )
            mp4_path = delivery_paths.mp4

        edited_audio_duration = measure_wav_duration_seconds(delivery_paths.wav)
        report: dict[str, Any] = {
            "source": str(input_path),
            "removed_phrase": removed_phrase,
            "media_probe": {
                "duration_seconds": probe.duration_seconds,
                "has_audio": probe.has_audio,
                "has_video": probe.has_video,
            },
            "cuts": _cuts_to_report(normalized_cuts),
            "keep_intervals": _intervals_to_report(keep_intervals),
            "removed_duration_seconds": sum(
                (cut.end_seconds or probe.duration_seconds) - cut.start_seconds for cut in normalized_cuts
            ),
            "crossfade_ms": crossfade_ms,
            "crossfade_frames": splice_result.crossfade_frames,
            "estimated_edited_audio_duration_seconds": edited_audio_duration,
            "deliverables": {
                "wav": str(delivery_paths.wav),
                "mp3": str(delivery_paths.mp3),
                "mp4": str(mp4_path) if mp4_path else None,
            },
            "tooling": {
                "ffmpeg": ffmpeg_bin,
                "ffprobe": ffprobe_bin,
                "respiro_en_used": False,
                "deepfilternet_used": False,
            },
            "work_dir": str(work_root) if keep_work else None,
        }
        delivery_paths.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        report["report"] = str(delivery_paths.report)
        return report
    finally:
        if not keep_work:
            shutil.rmtree(work_root, ignore_errors=True)


def _load_jobs(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload, {}
    if isinstance(payload, dict) and isinstance(payload.get("jobs"), list):
        defaults = {
            key: value
            for key, value in payload.items()
            if key
            in {
                "crossfade_ms",
                "output_dir",
                "keep_work",
            }
        }
        return payload["jobs"], defaults
    raise ValueError("Batch JSON must be a list or an object with a 'jobs' list")


def _build_single_job_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "input": args.input,
        "cuts": [parse_cut_spec(cut) for cut in args.cut],
        "removed_phrase": args.removed_phrase,
        "output_stem": args.output_stem,
        "crossfade_ms": args.crossfade_ms,
        "output_dir": args.output_dir,
        "keep_work": args.keep_work,
    }


def _coerce_job(job: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    merged = {**defaults, **job}
    if "input" not in merged:
        raise ValueError(f"Job is missing input: {job!r}")
    cuts_payload = merged.get("cuts")
    if not cuts_payload:
        raise ValueError(f"Job is missing cuts: {job!r}")
    return {
        "input": merged["input"],
        "cuts": [parse_cut_payload(cut) for cut in cuts_payload],
        "removed_phrase": merged.get("removed_phrase"),
        "output_stem": merged.get("output_stem"),
        "crossfade_ms": float(merged.get("crossfade_ms", DEFAULT_CROSSFADE_MS)),
        "output_dir": merged.get("output_dir"),
        "keep_work": bool(merged.get("keep_work", False)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="remove_spoken_segments",
        description="Physically remove spoken segments from media and smooth audio joins.",
    )
    parser.add_argument("--env", default=str(PROJECT_ROOT / ".env"), help="Optional .env file with ffmpeg paths.")
    parser.add_argument("--ffmpeg-bin", default=None, help="ffmpeg executable path.")
    parser.add_argument("--ffprobe-bin", default=None, help="ffprobe executable path.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Process one media file.")
    run_parser.add_argument("input", help="Input audio/video path.")
    run_parser.add_argument(
        "--cut",
        action="append",
        required=True,
        help='Cut window as "START,END"; use blank END for tail deletion, e.g. "4.40,".',
    )
    run_parser.add_argument("--removed-phrase", default=None, help="Phrase being removed, for the report.")
    run_parser.add_argument("--crossfade-ms", type=float, default=DEFAULT_CROSSFADE_MS)
    run_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    run_parser.add_argument("--output-stem", default=None)
    run_parser.add_argument("--keep-work", action="store_true")

    batch_parser = subparsers.add_parser("run-batch", help="Process jobs from a JSON file.")
    batch_parser.add_argument("jobs_json", help="Batch job JSON path.")
    batch_parser.add_argument("--crossfade-ms", type=float, default=None)
    batch_parser.add_argument("--output-dir", default=None)
    batch_parser.add_argument("--keep-work", action="store_true")

    return parser


def run_jobs(
    jobs: Sequence[dict[str, Any]],
    *,
    defaults: dict[str, Any],
    ffmpeg_bin: str,
    ffprobe_bin: str,
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for job in jobs:
        resolved = _coerce_job(job, defaults)
        reports.append(
            process_segment_removal_job(
                Path(resolved["input"]),
                cuts=resolved["cuts"],
                removed_phrase=resolved["removed_phrase"],
                output_dir=Path(resolved["output_dir"] or DEFAULT_OUTPUT_DIR),
                output_stem=resolved["output_stem"],
                crossfade_ms=resolved["crossfade_ms"],
                keep_work=resolved["keep_work"],
                ffmpeg_bin=ffmpeg_bin,
                ffprobe_bin=ffprobe_bin,
            )
        )
    return reports


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    env_values = load_env_file(args.env)
    ffmpeg_bin = args.ffmpeg_bin or resolve_binary("AUDIO_SOUND_FFMPEG", "ffmpeg", env_values)
    ffprobe_bin = args.ffprobe_bin or resolve_binary("AUDIO_SOUND_FFPROBE", "ffprobe", env_values)

    if args.command == "run":
        jobs = [_build_single_job_from_args(args)]
        defaults: dict[str, Any] = {}
    elif args.command == "run-batch":
        jobs, defaults = _load_jobs(Path(args.jobs_json))
        if args.crossfade_ms is not None:
            defaults["crossfade_ms"] = args.crossfade_ms
        if args.output_dir is not None:
            defaults["output_dir"] = args.output_dir
        if args.keep_work:
            defaults["keep_work"] = True
    else:
        parser.error(f"Unknown command: {args.command}")

    reports = run_jobs(jobs, defaults=defaults, ffmpeg_bin=ffmpeg_bin, ffprobe_bin=ffprobe_bin)
    print(json.dumps({"jobs": reports}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
