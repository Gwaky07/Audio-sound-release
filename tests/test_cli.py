from __future__ import annotations

import argparse
import io
import unittest
from contextlib import redirect_stdout
from unittest import mock

from audio_sound import cli


class CliTests(unittest.TestCase):
    def test_clean_parser_accepts_repeated_noise_windows(self) -> None:
        parser = cli.build_parser()
        args = parser.parse_args(
            [
                "clean",
                "sample.wav",
                "--noise-window",
                "143.089208:144.093687",
                "--noise-window",
                "161.583729:169.578792",
            ]
        )
        self.assertEqual(
            args.noise_window,
            ["143.089208:144.093687", "161.583729:169.578792"],
        )

    def test_describe_preset_outputs_json(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            exit_code = cli.command_describe_preset("safe")
        output = buffer.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn('"name": "safe"', output)
        self.assertIn('"deepfilternet"', output)

    def test_doctor_wires_standalone_runtime_report(self) -> None:
        args = argparse.Namespace(ffmpeg_bin="ffmpeg", ffprobe_bin="ffprobe", python_executable="python")
        with mock.patch("audio_sound.cli.detect_runtime") as detect_runtime:
            detect_runtime.return_value = {"python": {"ok": True}, "ffmpeg": {"ok": True}, "ffprobe": {"ok": True}}
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = cli.command_doctor(args)
        self.assertEqual(exit_code, 0)
        self.assertIn('"python"', buffer.getvalue())

    def test_list_presets_entrypoint(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            exit_code = cli.main(["list-presets"])
        self.assertEqual(exit_code, 0)
        output = buffer.getvalue()
        self.assertIn("safe", output)
        self.assertIn("review", output)

    def test_process_alias_invokes_clean(self) -> None:
        with mock.patch("audio_sound.cli.command_clean") as command_clean:
            command_clean.return_value = 0
            cli.main(["process", "sample.wav"])
        command_clean.assert_called_once()


if __name__ == "__main__":
    unittest.main()
