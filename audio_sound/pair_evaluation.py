from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from .pipeline import (
    _analysis_samples,
    _load_wave_samples,
    analyze_pcm16_samples,
    compare_audio_preservation,
    ensure_tool,
    ffprobe_media,
    run_command,
)


def _extract_pcm_wav(
    source: Path,
    output_wav: Path,
    *,
    ffmpeg_bin: str,
    ffprobe_bin: str,
    sample_rate: int | None = None,
    channels: int | None = None,
) -> Path:
    ffmpeg = ensure_tool(ffmpeg_bin)
    metadata = ffprobe_media(source, ffprobe_bin)
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-nostdin",
        "-i",
        str(source),
        "-vn",
        "-ac",
        str(channels or metadata.get("channels") or 1),
        "-ar",
        str(sample_rate or metadata.get("sample_rate") or 48000),
        "-c:a",
        "pcm_s16le",
        str(output_wav),
    ]
    completed = run_command(command)
    if completed.returncode != 0:
        raise RuntimeError(
            completed.stderr.strip() or completed.stdout.strip() or "extract failed"
        )
    return output_wav


def evaluate_audio_pair(
    *,
    source: Path,
    processed: Path,
    ffmpeg_bin: str = "ffmpeg",
    ffprobe_bin: str = "ffprobe",
) -> dict[str, Any]:
    source_meta = ffprobe_media(source, ffprobe_bin)
    processed_meta = ffprobe_media(processed, ffprobe_bin)
    with tempfile.TemporaryDirectory(prefix="audio-pair-") as tmp_dir:
        root = Path(tmp_dir)
        source_wav = source if source.suffix.lower() == ".wav" else root / "source.wav"
        processed_wav = (
            processed if processed.suffix.lower() == ".wav" else root / "processed.wav"
        )
        if source_wav != source:
            _extract_pcm_wav(
                source,
                source_wav,
                ffmpeg_bin=ffmpeg_bin,
                ffprobe_bin=ffprobe_bin,
                sample_rate=source_meta.get("sample_rate"),
                channels=source_meta.get("channels"),
            )
        if processed_wav != processed:
            _extract_pcm_wav(
                processed,
                processed_wav,
                ffmpeg_bin=ffmpeg_bin,
                ffprobe_bin=ffprobe_bin,
                sample_rate=processed_meta.get("sample_rate"),
                channels=processed_meta.get("channels"),
            )
        reference_params, reference_samples = _load_wave_samples(source_wav)
        processed_params, processed_samples = _load_wave_samples(processed_wav)
        reference_mono = _analysis_samples(reference_params, reference_samples)
        processed_mono = _analysis_samples(processed_params, processed_samples)
        reference_analysis = analyze_pcm16_samples(
            reference_mono,
            sample_rate=reference_params.framerate,
            frame_ms=10.0,
        )
        processed_analysis = analyze_pcm16_samples(
            processed_mono,
            sample_rate=processed_params.framerate,
            frame_ms=10.0,
        )
        guard = compare_audio_preservation(
            reference_analysis,
            processed_analysis,
            reference_format={
                "sample_rate": reference_params.framerate,
                "channels": reference_params.nchannels,
            },
            processed_format={
                "sample_rate": processed_params.framerate,
                "channels": processed_params.nchannels,
            },
            reference_samples=reference_mono,
            processed_samples=processed_mono,
            sample_rate=reference_params.framerate,
        )
    return {
        "source": {"path": str(source), "metadata": source_meta},
        "processed": {"path": str(processed), "metadata": processed_meta},
        "input_diagnostics": {
            key: value
            for key, value in reference_analysis.items()
            if not str(key).startswith("_")
        },
        "output_diagnostics": {
            key: value
            for key, value in processed_analysis.items()
            if not str(key).startswith("_")
        },
        "quality_guard": guard,
    }
