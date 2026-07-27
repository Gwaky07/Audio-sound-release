from __future__ import annotations

import ast
import inspect
import re
import unittest
from pathlib import Path

from audio_sound import pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIO_PACKAGE = PROJECT_ROOT / "audio_sound"


class ArchitectureContractTests(unittest.TestCase):
    def test_setuptools_does_not_install_generic_scripts_package(self) -> None:
        text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        match = re.search(r"^packages\s*=\s*\[([^\]]*)\]", text, flags=re.MULTILINE)
        self.assertIsNotNone(match)
        packages = [
            item.strip().strip("'\"")
            for item in str(match.group(1)).split(",")
            if item.strip()
        ]
        self.assertEqual(packages, ["audio_sound", "audio_sound.presets"])
        self.assertFalse((PROJECT_ROOT / "scripts" / "__init__.py").exists())
        # Guard against reintroducing a top-level package that would collide with
        # the generic name "scripts" in user environments / site-packages.
        self.assertNotRegex(
            text,
            r'packages\s*=\s*\[[^\]]*["\']scripts["\']',
        )
        self.assertTrue(
            (PROJECT_ROOT / "audio_sound" / "presets" / "__init__.py").exists()
        )

    def test_process_media_file_remains_an_orchestrator(self) -> None:
        source_lines, _ = inspect.getsourcelines(pipeline.process_media_file)
        self.assertLessEqual(
            len(source_lines),
            600,
            "process_media_file must delegate phases instead of returning to a "
            "thousand-line implementation",
        )
        source = "".join(source_lines)
        self.assertIn("_run_breath_residual_cleanup(", source)
        self.assertIn("_run_pause_cleanup(", source)
        self.assertIn("_run_recorded_command(", source)

    def test_shared_media_implementations_are_not_duplicated(self) -> None:
        forbidden_definitions = {
            "_load_wave_samples",
            "_format_seconds",
            "_sha256_file",
            "_export_mp3_from_wav",
        }
        observed: dict[str, list[str]] = {
            name: [] for name in forbidden_definitions
        }
        mp3_implementation_files: list[str] = []
        for path in AUDIO_PACKAGE.glob("*.py"):
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name in observed:
                        observed[node.name].append(path.name)
            if "libmp3lame" in source:
                mp3_implementation_files.append(path.name)

        self.assertEqual(
            observed,
            {name: [] for name in forbidden_definitions},
        )
        self.assertEqual(mp3_implementation_files, ["media_utils.py"])


if __name__ == "__main__":
    unittest.main()
