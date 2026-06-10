from __future__ import annotations

import argparse
import io
import unittest
from contextlib import redirect_stdout
from unittest import mock

from audio_sound import cli


class CliTests(unittest.TestCase):
    def test_setup_respiro_parser_accepts_tools_dir(self) -> None:
        parser = cli.build_parser()
        args = parser.parse_args(["setup-respiro", "--tools-dir", "D:/audio-tools"])
        self.assertEqual(args.tools_dir, "D:/audio-tools")

    def test_clean_parser_accepts_respiro_and_spectra_overrides(self) -> None:
        parser = cli.build_parser()
        args = parser.parse_args(
            [
                "clean",
                "sample.wav",
                "--attenuation-db",
                "18",
                "--respiro-threshold",
                "0.5",
                "--respiro-min-length-ms",
                "30",
                "--respiro-repo",
                "D:/models/Respiro-en",
                "--respiro-weights",
                "D:/models/respiro-en.pt",
                "--enable-legacy-breath-filters",
                "--skip-spectramini",
                "--skip-deepfilternet",
            ]
        )
        self.assertEqual(args.attenuation_db, 18.0)
        self.assertEqual(args.respiro_threshold, 0.5)
        self.assertEqual(args.respiro_min_length_ms, 30)
        self.assertEqual(args.respiro_repo, "D:/models/Respiro-en")
        self.assertEqual(args.respiro_weights, "D:/models/respiro-en.pt")
        self.assertTrue(args.enable_legacy_breath_filters)
        self.assertTrue(args.skip_spectramini)
        self.assertTrue(args.skip_deepfilternet)

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
        self.assertIn('"pipeline"', output)

    def test_describe_workflow_mode_outputs_json(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            exit_code = cli.command_describe_workflow_mode("reference-style")
        output = buffer.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn('"name": "reference-style"', output)
        self.assertIn('"preset_name": "fast"', output)

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

    def test_describe_workflow_mode_entrypoint(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            exit_code = cli.main(["describe-workflow-mode", "reference-style"])
        self.assertEqual(exit_code, 0)
        output = buffer.getvalue()
        self.assertIn('"name": "reference-style"', output)

    def test_process_alias_invokes_clean(self) -> None:
        with mock.patch("audio_sound.cli.command_clean") as command_clean:
            command_clean.return_value = 0
            cli.main(["process", "sample.wav"])
        command_clean.assert_called_once()

    def test_clean_repo_dispatches_workspace_cleanup(self) -> None:
        with mock.patch("audio_sound.cli.prune_workspace") as prune_workspace:
            prune_workspace.return_value = {"dry_run": True, "removed_count": 0, "targets": []}
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = cli.main(["clean-repo", "--dry-run"])
        self.assertEqual(exit_code, 0)
        prune_workspace.assert_called_once()
        self.assertIn('"dry_run": true', buffer.getvalue().lower())

    def test_pyproject_exposes_audio_skill_workflow_entrypoint(self) -> None:
        pyproject_text = (cli.PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('audio-skill-workflow = "audio_sound.skill_workflow:main"', pyproject_text)


if __name__ == "__main__":
    unittest.main()
