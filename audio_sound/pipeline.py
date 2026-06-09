from __future__ import annotations

import json
import re
import shutil
import subprocess
import wave
from array import array
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT, resolve_binary, resolve_repo_python

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


@dataclass(frozen=True)
class OutputLayout:
    job_name: str
    job_dir: Path
    preprocess_dir: Path
    transcript_dir: Path
    raw_wav: Path
    denoised_wav: Path
    noise_sample_wav: Path
    clean_wav: Path
    transcript_mp3: Path
    report_json: Path
    report_md: Path
    deepfilternet_dir: Path


def utc_timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def build_output_root(base_dir: str | Path, run_slug: str) -> Path:
    return Path(base_dir) / f"run-{run_slug}"


def _slugify_stem(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("._")
    if cleaned:
        return cleaned
    digest = sha1(value.encode("utf-8")).hexdigest()[:8]
    return f"media_{digest}"


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
    job_name = f"{run_slug}_{_slugify_stem(input_path.stem)}"
    job_dir = output_root / job_name
    preprocess_dir = job_dir / "audio_preprocess"
    transcript_dir = job_dir / "transcript_ready"
    return OutputLayout(
        job_name=job_name,
        job_dir=job_dir,
        preprocess_dir=preprocess_dir,
        transcript_dir=transcript_dir,
        raw_wav=preprocess_dir / "audio_raw.wav",
        denoised_wav=preprocess_dir / "audio_df.wav",
        noise_sample_wav=preprocess_dir / "audio_noise_sample.wav",
        clean_wav=preprocess_dir / "audio_clean.wav",
        transcript_mp3=transcript_dir / "audio.mp3",
        report_json=preprocess_dir / "audio_process_report.json",
        report_md=preprocess_dir / "audio_process_report.md",
        deepfilternet_dir=preprocess_dir / "deepfilternet_out",
    )


def build_ffmpeg_extract_command(*, input_path: Path, raw_wav: Path, preset: dict[str, Any], ffmpeg_bin: str) -> list[str]:
    extract = preset["extract"]
    return [
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-nostdin",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        str(extract["channels"]),
        "-ar",
        str(extract["sample_rate"]),
        "-c:a",
        extract["pcm_codec"],
        str(raw_wav),
    ]


def build_deepfilternet_command(*, raw_wav: Path, output_dir: Path, preset: dict[str, Any], python_executable: str) -> list[str]:
    config = preset["deepfilternet"]
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
        str(preset["extract"]["sample_rate"]),
        "-ac",
        str(preset["extract"]["channels"]),
        "-c:a",
        preset["extract"]["pcm_codec"],
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
        parts.append(
            "acompressor="
            f"threshold={compressor['threshold_db']}dB:ratio={compressor['ratio']}:"
            f"attack={compressor['attack_ms']}:release={compressor['release_ms']}:"
            f"makeup={compressor['makeup_db']}"
        )

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
) -> list[list[str]]:
    transcript = preset["transcript_export"]
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
            str(preset["extract"]["sample_rate"]),
            "-ac",
            str(preset["extract"]["channels"]),
            "-c:a",
            preset["extract"]["pcm_codec"],
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
            str(preset["extract"]["sample_rate"]),
            "-ac",
            str(preset["extract"]["channels"]),
            "-c:a",
            preset["extract"]["pcm_codec"],
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
            str(preset["extract"]["sample_rate"]),
            "-ac",
            str(preset["extract"]["channels"]),
            "-c:a",
            preset["extract"]["pcm_codec"],
            str(clean_wav),
        ]

    return [
        clean_command,
        [
            ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-nostdin",
            "-i",
            str(clean_wav),
            "-c:a",
            transcript["codec"],
            "-b:a",
            transcript["bitrate"],
            str(transcript_mp3),
        ],
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
        if params.sampwidth != 2 or params.nchannels != 1:
            raise ValueError("Breath onset detection currently expects mono 16-bit WAV audio")
        raw_frames = reader.readframes(params.nframes)
    samples = array("h")
    samples.frombytes(raw_frames)

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
) -> None:
    merged_windows = merge_noise_windows(windows)
    fade_samples = max(0, int(sample_rate * (fade_ms / 1000.0)))
    total_samples = len(samples)

    for window in merged_windows:
        start_index = max(0, min(total_samples, int(window.start_seconds * sample_rate)))
        end_index = max(start_index, min(total_samples, int(window.end_seconds * sample_rate)))
        if end_index <= start_index:
            continue

        local_fade = min(fade_samples, max(0, (end_index - start_index) // 2))
        for index in range(start_index, end_index):
            gain = floor_gain
            if local_fade > 0 and index < start_index + local_fade:
                progress = (index - start_index) / float(local_fade)
                gain = 1.0 - (1.0 - floor_gain) * progress
            elif local_fade > 0 and index >= end_index - local_fade:
                progress = (index - (end_index - local_fade)) / float(local_fade)
                gain = floor_gain + (1.0 - floor_gain) * progress
            samples[index] = int(samples[index] * gain)


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
        if params.sampwidth != 2 or params.nchannels != 1:
            raise ValueError("Breath ducking currently expects mono 16-bit WAV audio")
        raw_frames = reader.readframes(params.nframes)

    samples = array("h")
    samples.frombytes(raw_frames)
    duck_samples_for_windows(
        samples,
        sample_rate=params.framerate,
        windows=merged_windows,
        floor_gain=floor_gain,
        fade_ms=fade_ms,
    )

    temp_path = audio_path.with_name(f"{audio_path.stem}.breathduck.tmp.wav")
    with wave.open(str(temp_path), "wb") as writer:
        writer.setparams(params)
        writer.writeframes(samples.tobytes())
    temp_path.replace(audio_path)


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


def _processing_steps(preset: dict[str, Any]) -> list[str]:
    steps = [
        "Extract source audio to WAV",
        "Primary denoise via DeepFilterNet",
    ]
    filters = preset["filters"]
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
    steps.append("Loudness normalization via loudnorm")
    steps.append("Transcript-ready MP3 export")
    return steps


def _noise_print_processing_steps(preset: dict[str, Any]) -> list[str]:
    steps = [
        "Extract source audio to WAV",
        "Primary denoise via DeepFilterNet",
        "Capture noise sample from selected windows",
        "Noise-print denoise via afftdn sample capture",
    ]
    filters = preset["filters"]
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
    steps.append("Loudness normalization via loudnorm")
    steps.append("Transcript-ready MP3 export")
    return steps


def _noise_sample_duration(noise_windows: list[NoiseWindow]) -> float:
    return sum(window.end_seconds - window.start_seconds for window in noise_windows)


def _normalize_noise_windows(noise_windows: list[NoiseWindow] | None) -> list[NoiseWindow]:
    return list(noise_windows or [])


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
) -> dict[str, Any]:
    metadata = input_metadata or ffprobe_media(input_file, runtime.ffprobe_bin)
    layout = build_output_layout(input_path=input_file, output_root=output_root, run_slug=run_slug)
    resolved_noise_windows = _normalize_noise_windows(noise_windows)
    noise_sample_duration = _noise_sample_duration(resolved_noise_windows) if resolved_noise_windows else None

    ffmpeg = runtime.ffmpeg_bin if runtime.dry_run else ensure_tool(runtime.ffmpeg_bin)
    python_bin = runtime.python_executable or resolve_repo_python(PROJECT_ROOT)
    if not runtime.dry_run:
        python_bin = ensure_tool(python_bin)

    commands: list[list[str]] = [
        build_ffmpeg_extract_command(input_path=input_file, raw_wav=layout.raw_wav, preset=preset, ffmpeg_bin=ffmpeg),
        build_deepfilternet_command(raw_wav=layout.raw_wav, output_dir=layout.deepfilternet_dir, preset=preset, python_executable=python_bin),
    ]
    if resolved_noise_windows:
        commands.append(
            build_ffmpeg_noise_sample_command(
                source_wav=layout.denoised_wav,
                noise_sample_wav=layout.noise_sample_wav,
                noise_windows=resolved_noise_windows,
                preset=preset,
                ffmpeg_bin=ffmpeg,
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
        )
    )

    processing_steps = (
        _noise_print_processing_steps(preset)
        if resolved_noise_windows
        else _processing_steps(preset)
    )

    report: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "preset_name": preset_name,
        "backend": "local",
        "input_file": str(input_file),
        "input_metadata": metadata,
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
    }

    if runtime.dry_run:
        return report

    _ensure_directories(layout)
    executed: list[dict[str, Any]] = []

    for command in commands[:2]:
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
            raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "command failed")

    detected_df_wav = _find_single_wav(layout.deepfilternet_dir)
    if detected_df_wav.resolve() != layout.denoised_wav.resolve():
        detected_df_wav.replace(layout.denoised_wav)

    remaining_commands = commands[2:]
    for command in remaining_commands:
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
            raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "command failed")

    breath_onset_cleanup = preset.get("filters", {}).get("breath_onset_cleanup", {})
    if breath_onset_cleanup.get("enabled"):
        breath_windows = detect_breath_onset_windows(
            layout.clean_wav,
            ffmpeg_bin=runtime.ffmpeg_bin,
            config=breath_onset_cleanup,
        )
        report["breath_onset_windows"] = [
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
        report["breath_onset_windows"] = []

    report["executed"] = executed
    report["loudnorm_summary"] = extract_loudnorm_summary(executed[-2]["stderr"])
    report["output_metadata"] = ffprobe_media(layout.clean_wav, runtime.ffprobe_bin)

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
            "## Noise Windows",
            *noise_window_lines,
            "",
            "## Noise Sample",
            f"- Duration: `{report.get('noise_sample_duration_seconds')}`",
            "",
            "## Breath Onset Windows",
            *breath_onset_lines,
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
