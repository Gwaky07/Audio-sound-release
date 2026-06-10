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
        [python_bin, "-m", "pip", "install", "numpy==1.23.0", "librosa==0.10.0", "soundfile", "scipy", "intervaltree==3.1.0"],
        [python_bin, "-m", "pip", "install", "torch==2.2.2"],
        [python_bin, "-m", "pip", "install", "torchaudio==2.2.2"],
        [python_bin, "-m", "pip", "install", "deepfilternet"],
    ]


def build_respiro_setup_commands(*, repo_root: str | Path, tools_dir: str | Path) -> list[list[str]]:
    root = Path(repo_root)
    tools_path = Path(tools_dir)
    repo_path = tools_path / "Respiro-en"
    weights_path = tools_path / "respiro-en.pt"
    return [
        ["git", "clone", "https://github.com/ydqmkkx/Respiro-en.git", str(repo_path)],
        [
            "powershell",
            "-NoProfile",
            "-Command",
            f"Invoke-WebRequest -UseBasicParsing 'https://huggingface.co/ydqmkkx/respiro-en/resolve/main/respiro-en.pt' -OutFile '{weights_path}'",
        ],
    ]


def _inspect_python_runtime(python_executable: str) -> dict[str, Any]:
    script = (
        "import importlib.util, json, sys; "
        "respiro_ok = importlib.util.find_spec('intervaltree') is not None and importlib.util.find_spec('librosa') is not None; "
        "spec = importlib.util.find_spec('df.enhance') if importlib.util.find_spec('df') else None; "
        "print(json.dumps({'ok': True, 'path': sys.executable, 'version': sys.version.split()[0], 'deepfilternet_ok': spec is not None, 'respiro_runtime_ok': respiro_ok}))"
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
            "respiro_runtime_ok": False,
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
            "respiro_runtime_ok": False,
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
        "respiro_en": {
            "ok": bool(python_info.get("respiro_runtime_ok")),
            "notes": "Requires local Respiro-en repository and respiro-en.pt weights path configuration.",
        },
        "spectramini": {
            "ok": bool(python_info.get("respiro_runtime_ok")),
            "notes": "SpectraMini-style breath and mouth-click stages are embedded locally via librosa/scipy helpers.",
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


def run_respiro_setup(*, repo_root: str | Path, tools_dir: str | Path) -> dict[str, Any]:
    root = Path(repo_root)
    tools_path = Path(tools_dir)
    tools_path.mkdir(parents=True, exist_ok=True)

    steps: list[dict[str, Any]] = []
    repo_path = tools_path / "Respiro-en"
    weights_path = tools_path / "respiro-en.pt"
    commands = build_respiro_setup_commands(repo_root=root, tools_dir=tools_path)
    for index, command in enumerate(commands):
        if index == 0 and repo_path.exists():
            steps.append({"command": command, "returncode": 0, "stdout": "repo exists", "stderr": ""})
            continue
        if index == 1 and weights_path.exists():
            steps.append({"command": command, "returncode": 0, "stdout": "weights exists", "stderr": ""})
            continue

        completed = subprocess.run(
            command,
            cwd=str(root),
            capture_output=True,
            check=False,
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
            reason = step["stderr"] or step["stdout"] or "respiro setup failed"
            return {"ok": False, "code": "respiro_setup_failed", "reason": reason, "data": {"steps": steps}}

    env_path = root / ".env"
    env_lines: list[str] = []
    if env_path.exists():
        env_lines = env_path.read_text(encoding="utf-8").splitlines()

    updated: dict[str, str] = {}
    for line in env_lines:
        if "=" in line:
            key, value = line.split("=", 1)
            updated[key.strip()] = value.strip()
    updated["AUDIO_SOUND_RESPIRO_REPO"] = str(repo_path)
    updated["AUDIO_SOUND_RESPIRO_WEIGHTS"] = str(weights_path)

    serialized = "\n".join(f"{key}={value}" for key, value in updated.items()) + "\n"
    env_path.write_text(serialized, encoding="utf-8")
    return {
        "ok": True,
        "code": "ok",
        "data": {
            "steps": steps,
            "repo_path": str(repo_path),
            "weights_path": str(weights_path),
            "env_path": str(env_path),
        },
    }


def format_runtime_report(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def format_install_report(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)
