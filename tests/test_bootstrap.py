from __future__ import annotations

import unittest

from audio_sound.bootstrap import build_install_commands, format_runtime_report


class BootstrapTests(unittest.TestCase):
    def test_build_install_commands_include_deepfilternet_stack(self) -> None:
        commands = build_install_commands(repo_root="S:/Agent/Auto jianji/Audio-sound", python_executable="python")
        flattened = [" ".join(command) for command in commands]
        self.assertTrue(any("pytest>=8.0" in command for command in flattened))
        self.assertTrue(any("torch==2.3.1" in command for command in flattened))
        self.assertTrue(any("torchaudio==2.3.1" in command for command in flattened))
        self.assertTrue(any("deepfilternet" in command for command in flattened))

    def test_format_runtime_report_returns_json_string(self) -> None:
        payload = {"python": {"ok": True}, "ffmpeg": {"ok": False}}
        rendered = format_runtime_report(payload)
        self.assertIn('"python"', rendered)
        self.assertIn('"ffmpeg"', rendered)


if __name__ == "__main__":
    unittest.main()
