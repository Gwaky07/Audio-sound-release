from __future__ import annotations

import hashlib
import shutil
import subprocess
import wave
from array import array
from pathlib import Path
from typing import Any, Sequence


def load_pcm16_wave(audio_path: Path) -> tuple[Any, array]:
    with wave.open(str(audio_path), "rb") as reader:
        params = reader.getparams()
        if params.sampwidth != 2 or params.nchannels <= 0:
            raise ValueError("Expected 16-bit WAV audio with at least one channel")
        raw_frames = reader.readframes(params.nframes)
    samples = array("h")
    samples.frombytes(raw_frames)
    return params, samples


def format_seconds(value: float, *, precision: int = 3) -> str:
    if precision < 0:
        raise ValueError("precision must be non-negative")
    return f"{value:.{precision}f}".rstrip("0").rstrip(".") or "0"


def sha256_file(path: Path, *, uppercase: bool = False) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    return value.upper() if uppercase else value


def build_mp3_export_command(
    source_wav: Path,
    output_mp3: Path,
    *,
    ffmpeg_bin: str,
    codec: str = "libmp3lame",
    bitrate: str | None = None,
    quality: int | None = None,
) -> list[str]:
    if bitrate is not None and quality is not None:
        raise ValueError("Specify either bitrate or quality, not both")
    command = [
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-nostdin",
        "-i",
        str(source_wav),
        "-c:a",
        codec,
    ]
    if bitrate is not None:
        command.extend(["-b:a", bitrate])
    if quality is not None:
        command.extend(["-q:a", str(quality)])
    command.append(str(output_mp3))
    return command


def export_mp3(
    source_wav: Path,
    output_mp3: Path,
    *,
    ffmpeg_bin: str,
    codec: str = "libmp3lame",
    bitrate: str | None = None,
    quality: int | None = None,
) -> subprocess.CompletedProcess[str]:
    resolved_ffmpeg = shutil.which(ffmpeg_bin)
    if resolved_ffmpeg is None and Path(ffmpeg_bin).is_file():
        resolved_ffmpeg = str(Path(ffmpeg_bin))
    if resolved_ffmpeg is None:
        raise FileNotFoundError(f"Required tool not found: {ffmpeg_bin}")
    output_mp3.parent.mkdir(parents=True, exist_ok=True)
    command: Sequence[str] = build_mp3_export_command(
        source_wav,
        output_mp3,
        ffmpeg_bin=resolved_ffmpeg,
        codec=codec,
        bitrate=bitrate,
        quality=quality,
    )
    completed = subprocess.run(
        list(command),
        capture_output=True,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            completed.stderr.strip()
            or completed.stdout.strip()
            or "mp3 export failed"
        )
    return completed
