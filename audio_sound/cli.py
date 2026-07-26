from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .bootstrap import (
    detect_runtime,
    format_install_report,
    format_runtime_report,
    prune_workspace,
    run_install,
    run_respiro_setup,
)
from .config import PROJECT_ROOT, apply_runtime_overrides, list_presets, load_env_file, load_preset, resolve_repo_python
from .pipeline import (
    NoiseWindow,
    RuntimeOptions,
    build_batch_summary,
    build_output_root,
    discover_media_files,
    ffprobe_media,
    parse_noise_window,
    process_media_file,
    resolve_respiro_runtime,
    render_batch_summary_markdown,
    utc_timestamp_slug,
)
from .skill_workflow import describe_modes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone audio processing pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-presets", help="List available presets.")

    describe_parser = subparsers.add_parser("describe-preset", help="Print one preset as JSON.")
    describe_parser.add_argument("preset_name", choices=list_presets())

    inspect_parser = subparsers.add_parser("inspect", help="Inspect one media file with ffprobe.")
    inspect_parser.add_argument("input_path")
    inspect_parser.add_argument("--ffprobe-bin", default="ffprobe")

    doctor_parser = subparsers.add_parser("doctor", help="Inspect local runtime availability.")
    doctor_parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    doctor_parser.add_argument("--ffprobe-bin", default="ffprobe")
    doctor_parser.add_argument("--python-executable", default=resolve_repo_python(PROJECT_ROOT))

    setup_parser = subparsers.add_parser("setup", help="Install optional local runtime dependencies.")
    setup_parser.add_argument("--python-executable", default=resolve_repo_python(PROJECT_ROOT))

    clean_repo_parser = subparsers.add_parser("clean-repo", help="Remove generated outputs and local runtime state from the repository.")
    clean_repo_parser.add_argument("--dry-run", action="store_true")

    setup_respiro_parser = subparsers.add_parser("setup-respiro", help="Clone Respiro-en and download weights locally.")
    setup_respiro_parser.add_argument("--tools-dir", default=str(PROJECT_ROOT / "tools"))

    describe_workflow_parser = subparsers.add_parser(
        "describe-workflow-mode",
        help="Print one repo-local stable workflow mode as JSON.",
    )
    describe_workflow_parser.add_argument("mode_name", nargs="?", choices=sorted(describe_modes().keys()))

    clean_parser = subparsers.add_parser("clean", help="Process one file or one directory.")
    _add_clean_arguments(clean_parser)

    process_parser = subparsers.add_parser("process", help="Alias of clean.")
    _add_clean_arguments(process_parser)

    return parser


def _add_clean_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input_path")
    parser.add_argument("--preset", default="safe", choices=list_presets())
    parser.add_argument("--attenuation-db", type=float, default=18.0)
    parser.add_argument("--respiro-threshold", type=float)
    parser.add_argument("--respiro-min-length-ms", type=int)
    parser.add_argument("--respiro-repo")
    parser.add_argument("--respiro-weights")
    parser.add_argument("--enable-legacy-breath-filters", action="store_true")
    parser.add_argument("--skip-spectramini", action="store_true")
    parser.add_argument("--skip-deepfilternet", action="store_true")
    parser.add_argument(
        "--noise-window",
        action="append",
        default=[],
        help="Noise-only sample region in seconds, formatted as start:end. Repeat for multiple regions.",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--target-lufs", type=float)
    parser.add_argument("--denoise-strength", choices=["light", "medium", "aggressive"])
    parser.add_argument("--disable-gate", action="store_true")
    parser.add_argument("--enable-silence-report", action="store_true")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--ffprobe-bin", default="ffprobe")
    parser.add_argument(
        "--python-executable",
        default=resolve_repo_python(PROJECT_ROOT),
    )
    parser.add_argument("--dry-run", action="store_true")


def command_list_presets() -> int:
    for preset_name in list_presets():
        print(preset_name)
    return 0


def command_describe_preset(preset_name: str) -> int:
    print(json.dumps(load_preset(preset_name), indent=2, ensure_ascii=False))
    return 0


def command_inspect(input_path: str, ffprobe_bin: str) -> int:
    payload = ffprobe_media(Path(input_path), ffprobe_bin)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def command_doctor(args: argparse.Namespace) -> int:
    payload = detect_runtime(
        repo_root=PROJECT_ROOT,
        python_executable=args.python_executable,
        ffmpeg_bin=args.ffmpeg_bin,
        ffprobe_bin=args.ffprobe_bin,
    )
    print(format_runtime_report(payload))
    return 0


def command_setup(args: argparse.Namespace) -> int:
    payload = run_install(repo_root=PROJECT_ROOT, python_executable=args.python_executable)
    print(format_install_report(payload))
    return 0 if payload.get("ok") else 1


def command_clean_repo(args: argparse.Namespace) -> int:
    payload = prune_workspace(repo_root=PROJECT_ROOT, dry_run=args.dry_run)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def command_setup_respiro(args: argparse.Namespace) -> int:
    payload = run_respiro_setup(repo_root=PROJECT_ROOT, tools_dir=args.tools_dir)
    print(format_install_report(payload))
    return 0 if payload.get("ok") else 1


def command_describe_workflow_mode(mode_name: str | None) -> int:
    print(json.dumps(describe_modes(mode_name), indent=2, ensure_ascii=False))
    return 0


def command_clean(args: argparse.Namespace) -> int:
    input_path = Path(args.input_path)
    files = discover_media_files(input_path, recursive=args.recursive)
    if not files:
        raise FileNotFoundError(f"No supported media files found under: {input_path}")
    noise_windows: list[NoiseWindow] = [parse_noise_window(value) for value in args.noise_window]

    raw_preset = load_preset(args.preset)
    resolved_preset = apply_runtime_overrides(
        raw_preset,
        target_lufs=args.target_lufs,
        denoise_strength=args.denoise_strength,
        disable_gate=args.disable_gate,
        enable_silence_report=args.enable_silence_report,
        enable_legacy_breath_filters=args.enable_legacy_breath_filters,
    )

    run_slug = utc_timestamp_slug()
    output_root = (
        Path(args.output_dir)
        if args.output_dir
        else build_output_root(PROJECT_ROOT / "output", run_slug)
    )

    runtime = RuntimeOptions(
        ffmpeg_bin=args.ffmpeg_bin,
        ffprobe_bin=args.ffprobe_bin,
        python_executable=args.python_executable,
        dry_run=args.dry_run,
    )
    env_values = load_env_file(PROJECT_ROOT / ".env")
    respiro_runtime = resolve_respiro_runtime(
        respiro_repo=args.respiro_repo,
        respiro_weights=args.respiro_weights,
        env_values=env_values,
    )

    reports = [
        process_media_file(
            file_path,
            preset_name=args.preset,
            preset=resolved_preset,
            output_root=output_root,
            runtime=runtime,
            run_slug=run_slug,
            noise_windows=noise_windows,
            respiro_repo=respiro_runtime["repo_path"],
            respiro_weights=respiro_runtime["weights_path"],
            attenuation_db=args.attenuation_db,
            respiro_threshold=args.respiro_threshold,
            respiro_min_length_ms=args.respiro_min_length_ms,
            skip_spectramini=args.skip_spectramini,
            skip_deepfilternet=args.skip_deepfilternet,
        )
        for file_path in files
    ]

    if args.dry_run:
        print(json.dumps({"output_root": str(output_root), "reports": reports}, indent=2, ensure_ascii=False))
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    summary = build_batch_summary(
        preset_name=args.preset,
        preset_description=resolved_preset["description"],
        input_path=input_path,
        output_root=output_root,
        reports=reports,
    )
    (output_root / "batch-summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_root / "batch-summary.md").write_text(render_batch_summary_markdown(summary), encoding="utf-8")

    print(f"Processed {len(reports)} file(s).")
    print(f"Output root: {output_root}")
    for report in reports:
        print(f"- {report['input_file']} -> {report['outputs']['clean_wav']}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "list-presets":
        return command_list_presets()
    if args.command == "describe-preset":
        return command_describe_preset(args.preset_name)
    if args.command == "inspect":
        return command_inspect(args.input_path, args.ffprobe_bin)
    if args.command == "doctor":
        return command_doctor(args)
    if args.command == "setup":
        return command_setup(args)
    if args.command == "clean-repo":
        return command_clean_repo(args)
    if args.command == "setup-respiro":
        return command_setup_respiro(args)
    if args.command == "describe-workflow-mode":
        return command_describe_workflow_mode(args.mode_name)
    if args.command in {"clean", "process"}:
        return command_clean(args)

    raise ValueError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
