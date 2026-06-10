from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from audio_sound.bootstrap import (
    build_install_commands,
    build_respiro_setup_commands,
    format_runtime_report,
    prune_workspace,
)


class BootstrapTests(unittest.TestCase):
    def test_build_install_commands_include_breath_first_stack(self) -> None:
        commands = build_install_commands(repo_root="S:/Agent/Auto jianji/Audio-sound", python_executable="python")
        flattened = [" ".join(command) for command in commands]
        self.assertTrue(any("pytest>=8.0" in command for command in flattened))
        self.assertTrue(any("numpy" in command for command in flattened))
        self.assertTrue(any("librosa" in command for command in flattened))
        self.assertTrue(any("soundfile" in command for command in flattened))
        self.assertTrue(any("scipy" in command for command in flattened))
        self.assertTrue(any("intervaltree" in command for command in flattened))
        self.assertTrue(any("torch==2.2.2" in command for command in flattened))
        self.assertTrue(any("torchaudio==2.2.2" in command for command in flattened))
        self.assertTrue(any("deepfilternet" in command for command in flattened))

    def test_format_runtime_report_returns_json_string(self) -> None:
        payload = {"python": {"ok": True}, "ffmpeg": {"ok": False}}
        rendered = format_runtime_report(payload)
        self.assertIn('"python"', rendered)
        self.assertIn('"ffmpeg"', rendered)

    def test_prune_workspace_removes_generated_state_and_keeps_repo_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "output" / "run-1").mkdir(parents=True)
            (root / "output" / "run-1" / "audio.wav").write_bytes(b"wav")
            (root / "scratch").mkdir()
            (root / "scratch" / "draft.wav").write_bytes(b"draft")
            (root / "audio_sound" / "__pycache__").mkdir(parents=True)
            (root / "audio_sound" / "__pycache__" / "pipeline.cpython-311.pyc").write_bytes(b"cache")
            (root / ".venv" / "Lib").mkdir(parents=True)
            (root / ".venv" / "Lib" / "keep.pyc").write_bytes(b"venv")
            (root / "README.md").write_text("keep", encoding="utf-8")

            payload = prune_workspace(repo_root=root, dry_run=False)

            self.assertFalse((root / "output").exists())
            self.assertFalse((root / "scratch").exists())
            self.assertFalse((root / "audio_sound" / "__pycache__").exists())
            self.assertFalse((root / ".venv").exists())
            self.assertTrue((root / "README.md").exists())
            self.assertGreaterEqual(payload["removed_count"], 4)
            self.assertGreater(payload["bytes_reclaimed"], 0)

    def test_build_respiro_setup_commands_include_clone_and_download(self) -> None:
        commands = build_respiro_setup_commands(
            repo_root="S:/Agent/Auto jianji/Audio-sound",
            tools_dir="S:/Agent/Auto jianji/Audio-sound/tools",
        )
        flattened = [" ".join(command) for command in commands]
        self.assertTrue(any("git clone https://github.com/ydqmkkx/Respiro-en.git" in command for command in flattened))
        self.assertTrue(any("https://huggingface.co/ydqmkkx/respiro-en/resolve/main/respiro-en.pt" in command for command in flattened))


if __name__ == "__main__":
    unittest.main()
