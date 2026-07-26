from __future__ import annotations

import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

from audio_sound.media_utils import (
    build_mp3_export_command,
    export_mp3,
    format_seconds,
    load_pcm16_wave,
    sha256_file,
)


class MediaUtilsTests(unittest.TestCase):
    def test_load_pcm16_wave_preserves_interleaved_channels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_path = Path(tmp_dir) / "stereo.wav"
            with wave.open(str(audio_path), "wb") as writer:
                writer.setnchannels(2)
                writer.setsampwidth(2)
                writer.setframerate(16000)
                writer.writeframes(b"\x01\x00\x02\x00\x03\x00\x04\x00")

            params, samples = load_pcm16_wave(audio_path)

        self.assertEqual(params.nchannels, 2)
        self.assertEqual(list(samples), [1, 2, 3, 4])

    def test_format_seconds_supports_call_site_precision(self) -> None:
        self.assertEqual(format_seconds(1.2301), "1.23")
        self.assertEqual(format_seconds(1.230001, precision=6), "1.230001")
        self.assertEqual(format_seconds(0.0, precision=6), "0")

    def test_sha256_file_supports_standard_and_uppercase_forms(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "payload.bin"
            path.write_bytes(b"audio-sound")
            lowercase = sha256_file(path)
            uppercase = sha256_file(path, uppercase=True)

        self.assertEqual(len(lowercase), 64)
        self.assertEqual(uppercase, lowercase.upper())

    def test_mp3_command_rejects_conflicting_rate_modes(self) -> None:
        with self.assertRaises(ValueError):
            build_mp3_export_command(
                Path("source.wav"),
                Path("output.mp3"),
                ffmpeg_bin="ffmpeg",
                bitrate="192k",
                quality=2,
            )

    def test_export_mp3_uses_shared_checked_runner(self) -> None:
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="",
            stderr="",
        )
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            mock.patch(
                "audio_sound.media_utils.shutil.which",
                return_value="C:/ffmpeg/bin/ffmpeg.exe",
            ),
            mock.patch(
                "audio_sound.media_utils.subprocess.run",
                return_value=completed,
            ) as run_mock,
        ):
            output = Path(tmp_dir) / "nested" / "output.mp3"
            export_mp3(
                Path("source.wav"),
                output,
                ffmpeg_bin="ffmpeg",
                bitrate="192k",
            )
            output_parent_created = output.parent.is_dir()

        command = run_mock.call_args.args[0]
        self.assertIn("C:/ffmpeg/bin/ffmpeg.exe", command)
        self.assertIn("192k", command)
        self.assertTrue(output_parent_created)


if __name__ == "__main__":
    unittest.main()
