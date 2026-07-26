from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def is_supported_python_version(version: str) -> bool:
    try:
        major, minor, *_ = (int(part) for part in str(version).split("."))
    except (TypeError, ValueError):
        return False
    return major == 3 and minor in {10, 11}


def build_install_commands(*, repo_root: str | Path, python_executable: str | None = None) -> list[list[str]]:
    python_bin = python_executable or sys.executable
    return [
        [python_bin, "-m", "pip", "install", "pytest>=8.0"],
        [python_bin, "-m", "pip", "install", "numpy>=1.24,<2", "librosa==0.10.0", "soundfile", "scipy", "intervaltree==3.1.0"],
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
        "torch_ok = importlib.util.find_spec('torch') is not None; "
        "respiro_ok = importlib.util.find_spec('intervaltree') is not None and importlib.util.find_spec('librosa') is not None; "
        "spec = importlib.util.find_spec('df.enhance') if importlib.util.find_spec('df') else None; "
        "print(json.dumps({'ok': True, 'path': sys.executable, 'version': sys.version.split()[0], 'torch_ok': torch_ok, 'deepfilternet_ok': spec is not None, 'respiro_runtime_ok': respiro_ok and torch_ok}))"
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
            "torch_ok": False,
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
            "torch_ok": False,
            "error": completed.stdout.strip(),
        }


def _inspect_respiro_weights(python_executable: str, weights_path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            python_executable,
            "-c",
            (
                "import json, sys, torch; "
                "payload = torch.load(sys.argv[1], map_location='cpu'); "
                "print(json.dumps({'ok': isinstance(payload, dict) and 'model' in payload}))"
            ),
            str(weights_path),
        ],
        capture_output=True,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        return {
            "ok": False,
            "error": (completed.stderr or completed.stdout or "weight load failed").strip(),
        }
    try:
        return json.loads(completed.stdout.strip())
    except json.JSONDecodeError:
        return {"ok": False, "error": completed.stdout.strip()}


def detect_runtime(
    *,
    repo_root: str | Path | None = None,
    python_executable: str | None = None,
    ffmpeg_bin: str = "ffmpeg",
    ffprobe_bin: str = "ffprobe",
) -> dict[str, Any]:
    from .config import PROJECT_ROOT

    root = Path(repo_root) if repo_root else PROJECT_ROOT
    python_bin = python_executable or sys.executable
    python_info = _inspect_python_runtime(python_executable or sys.executable)
    python_supported = is_supported_python_version(str(python_info.get("version") or ""))
    ffmpeg_path = shutil.which(ffmpeg_bin)
    ffprobe_path = shutil.which(ffprobe_bin)
    respiro_repo = root / "tools" / "Respiro-en"
    respiro_modules = respiro_repo / "modules.py"
    respiro_weights = root / "tools" / "respiro-en.pt"
    assets_ok = respiro_modules.exists() and respiro_weights.exists()
    runtime_ok = bool(python_info.get("respiro_runtime_ok")) and python_supported
    weights_check = (
        _inspect_respiro_weights(python_bin, respiro_weights)
        if assets_ok and runtime_ok
        else {"ok": False, "skipped": True}
    )
    respiro_ready = assets_ok and runtime_ok and bool(weights_check.get("ok"))
    deepfilter_runtime_ok = bool(python_info.get("deepfilternet_ok")) and python_supported
    return {
        "python": {
            "ok": bool(python_info.get("ok")),
            "path": str(python_info.get("path") or python_bin),
            "version": str(python_info.get("version") or ""),
            "supported": python_supported,
            "supported_versions": ["3.10", "3.11"],
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
            "ok": deepfilter_runtime_ok,
            "ready": deepfilter_runtime_ok,
            "module": "df.enhance",
            "runtime": {
                "ok": deepfilter_runtime_ok,
                "importable": bool(python_info.get("deepfilternet_ok")),
            },
        },
        "respiro_en": {
            "ok": respiro_ready,
            "ready": respiro_ready,
            "assets": {
                "ok": assets_ok,
                "repo_path": str(respiro_repo),
                "modules_path": str(respiro_modules),
                "weights_path": str(respiro_weights),
            },
            "runtime": {
                "ok": runtime_ok,
                "torch_importable": bool(python_info.get("torch_ok")),
                "librosa_intervaltree_importable": bool(
                    python_info.get("respiro_runtime_ok")
                ),
            },
            "weights": weights_check,
        },
        "spectramini": {
            "ok": runtime_ok,
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

    return {
        "ok": True,
        "code": "ok",
        "data": {
            "steps": steps,
            "runtime": detect_runtime(
                repo_root=root,
                python_executable=python_executable,
            ),
        },
    }


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
