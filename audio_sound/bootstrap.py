from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def build_install_commands(*, repo_root: str | Path, python_executable: str | None = None) -> list[list[str]]:
    python_bin = python_executable or sys.executable
    return [
        [python_bin, "-m", "pip", "install", "pytest>=8.0"],
        [python_bin, "-m", "pip", "install", "torch==2.3.1"],
        [python_bin, "-m", "pip", "install", "torchaudio==2.3.1"],
        [python_bin, "-m", "pip", "install", "deepfilternet"],
    ]


def _inspect_python_runtime(python_executable: str) -> dict[str, Any]:
    script = (
        "import importlib.util, json, sys; "
        "spec = importlib.util.find_spec('df.enhance') if importlib.util.find_spec('df') else None; "
        "print(json.dumps({'ok': True, 'path': sys.executable, 'version': sys.version.split()[0], 'deepfilternet_ok': spec is not None}))"
    )
    completed = subprocess.run(
        [python_executable, "-c", script],
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        return {
            "ok": False,
            "path": python_executable,
            "version": "",
            "deepfilternet_ok": False,
            "error": (completed.stderr or completed.stdout or "").strip(),
        }
    try:
        return json.loads(completed.stdout.strip())
    except json.JSONDecodeError:
        return {
            "ok": False,
            "path": python_executable,
            "version": "",
            "deepfilternet_ok": False,
            "error": completed.stdout.strip(),
        }


def detect_runtime(*, python_executable: str | None = None, ffmpeg_bin: str = "ffmpeg", ffprobe_bin: str = "ffprobe") -> dict[str, Any]:
    python_info = _inspect_python_runtime(python_executable or sys.executable)
    ffmpeg_path = shutil.which(ffmpeg_bin)
    ffprobe_path = shutil.which(ffprobe_bin)
    return {
        "python": {
            "ok": bool(python_info.get("ok")),
            "path": str(python_info.get("path") or (python_executable or sys.executable)),
            "version": str(python_info.get("version") or ""),
        },
        "ffmpeg": {
            "ok": bool(ffmpeg_path),
            "path": ffmpeg_path or "",
        },
        "ffprobe": {
            "ok": bool(ffprobe_path),
            "path": ffprobe_path or "",
        },
        "deepfilternet": {
            "ok": bool(python_info.get("deepfilternet_ok")),
            "module": "df.enhance",
        },
    }


def run_install(*, repo_root: str | Path, python_executable: str | None = None) -> dict[str, Any]:
    root = Path(repo_root)
    cargo_home = root / ".cargo-home"
    cargo_home.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CARGO_HOME"] = str(cargo_home)

    steps: list[dict[str, Any]] = []
    for command in build_install_commands(repo_root=root, python_executable=python_executable):
        completed = subprocess.run(
            command,
            cwd=str(root),
            capture_output=True,
            check=False,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        step = {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
        steps.append(step)
        if completed.returncode != 0:
            reason = step["stderr"] or step["stdout"] or "install failed"
            return {"ok": False, "code": "install_failed", "reason": reason, "data": {"steps": steps}}

    return {"ok": True, "code": "ok", "data": {"steps": steps, "runtime": detect_runtime(python_executable=python_executable)}}


def format_runtime_report(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def format_install_report(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def prune_workspace(*, repo_root: str | Path, dry_run: bool = False) -> dict[str, Any]:
    root = Path(repo_root)
    top_level_dirs = [
        root / "output",
        root / "scratch",
        root / ".venv",
        root / ".pytest_cache",
        root / ".omx",
        root / ".worktrees",
        root / ".cargo-home",
    ]
    top_level_targets = {path.resolve() for path in top_level_dirs if path.exists()}
    removable_dirs = [
        path.resolve()
        for path in root.rglob("__pycache__")
        if path.is_dir() and not any(parent in top_level_targets for parent in path.resolve().parents)
    ]
    removable_files = list(root.rglob("*.pyc"))
    directory_targets = {path for path in removable_dirs if path.exists()} | top_level_targets
    file_targets = {
        path.resolve()
        for path in removable_files
        if path.exists() and not any(parent in directory_targets for parent in path.resolve().parents)
    }
    targets = sorted([*directory_targets, *file_targets], key=lambda item: str(item))

    bytes_reclaimed = 0
    for path in targets:
        if path.is_file():
            bytes_reclaimed += path.stat().st_size
            if not dry_run:
                path.unlink()
            continue

        for child in path.rglob("*"):
            if child.is_file():
                bytes_reclaimed += child.stat().st_size
        if not dry_run:
            shutil.rmtree(path)

    return {
        "dry_run": dry_run,
        "removed_count": len(targets),
        "bytes_reclaimed": bytes_reclaimed,
        "targets": [str(path) for path in targets],
        "top_level_targets": [str(path) for path in sorted(top_level_targets, key=lambda item: str(item))],
    }
