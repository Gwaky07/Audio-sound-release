from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from audio_sound import __version__
from audio_sound.bootstrap import (
    build_install_commands,
    build_respiro_setup_commands,
    detect_runtime,
    format_runtime_report,
    is_supported_python_version,
    prune_workspace,
)
from audio_sound.config import PROJECT_ROOT


class BootstrapTests(unittest.TestCase):
    def test_package_version_matches_project_metadata(self) -> None:
        project = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        match = re.search(r'^version = "([^"]+)"$', project, flags=re.MULTILINE)
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.group(1), __version__)

    def test_requirement_torch_pins_match_bootstrap_commands(self) -> None:
        requirements = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
        commands = build_install_commands(repo_root=PROJECT_ROOT, python_executable="python")
        flattened = "\n".join(" ".join(command) for command in commands)
        for package in ("torch", "torchaudio"):
            match = re.search(rf"^{package}==([^\s]+)$", requirements, flags=re.MULTILINE)
            self.assertIsNotNone(match)
            assert match is not None
            self.assertIn(f"{package}=={match.group(1)}", flattened)

    def test_clean_source_quick_start_does_not_claim_bundled_runtime_assets(self) -> None:
        guide = (PROJECT_ROOT / "RELEASE_QUICK_START.md").read_text(encoding="utf-8")
        self.assertIn("不包含 Python 虚拟环境、`ffmpeg/`、`tools/`、模型权重", guide)
        self.assertIn("确定性的 `final` 修音链仍可运行", guide)
        self.assertNotIn("接收端不需要安装系统级 ffmpeg", guide)

    def test_supported_python_version_rejects_unsupported_runtime(self) -> None:
        self.assertTrue(is_supported_python_version("3.10.14"))
        self.assertTrue(is_supported_python_version("3.11.9"))
        self.assertFalse(is_supported_python_version("3.12.0"))
        self.assertFalse(is_supported_python_version("3.14.3"))

    def test_detect_runtime_defaults_to_project_root(self) -> None:
        with (
            mock.patch(
                "audio_sound.bootstrap._inspect_python_runtime",
                return_value={
                    "ok": True,
                    "path": "python",
                    "version": "3.11.9",
                    "deepfilternet_ok": False,
                    "respiro_runtime_ok": False,
                    "torch_ok": False,
                },
            ),
            mock.patch("audio_sound.bootstrap.shutil.which", return_value="tool.exe"),
            mock.patch("audio_sound.bootstrap.Path.cwd", return_value=Path("S:/not-the-repo")),
        ):
            payload = detect_runtime(python_executable="python")

        expected_weights = (PROJECT_ROOT / "tools" / "respiro-en.pt").resolve()
        self.assertEqual(
            Path(payload["respiro_en"]["assets"]["weights_path"]).resolve(),
            expected_weights,
        )

    def test_detect_runtime_reports_assets_separately_from_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "tools" / "Respiro-en").mkdir(parents=True)
            (root / "tools" / "Respiro-en" / "modules.py").write_text("", encoding="utf-8")
            (root / "tools" / "respiro-en.pt").write_bytes(b"weights")
            with (
                mock.patch(
                    "audio_sound.bootstrap._inspect_python_runtime",
                    return_value={
                        "ok": True,
                        "path": "python",
                        "version": "3.11.9",
                        "deepfilternet_ok": False,
                        "respiro_runtime_ok": False,
                        "torch_ok": True,
                    },
                ),
                mock.patch("audio_sound.bootstrap.shutil.which", return_value="tool.exe"),
            ):
                payload = detect_runtime(repo_root=root, python_executable="python")

        self.assertTrue(payload["respiro_en"]["assets"]["ok"])
        self.assertFalse(payload["respiro_en"]["runtime"]["ok"])
        self.assertFalse(payload["respiro_en"]["ready"])
        self.assertFalse(payload["deepfilternet"]["ready"])
        self.assertEqual(payload["python"]["supported"], True)

    def test_build_install_commands_include_breath_first_stack(self) -> None:
        commands = build_install_commands(repo_root="S:/Projects/Audio-sound", python_executable="python")
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
            repo_root="S:/Projects/Audio-sound",
            tools_dir="S:/Projects/Audio-sound/tools",
        )
        flattened = [" ".join(command) for command in commands]
        self.assertTrue(any("git clone https://github.com/ydqmkkx/Respiro-en.git" in command for command in flattened))
        self.assertTrue(any("https://huggingface.co/ydqmkkx/respiro-en/resolve/main/respiro-en.pt" in command for command in flattened))


if __name__ == "__main__":
    unittest.main()
