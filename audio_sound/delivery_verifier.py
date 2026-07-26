from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from .pair_evaluation import evaluate_audio_pair
from .pipeline import ffprobe_media


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_file(path: Path, label: str, failures: list[str]) -> None:
    if not path.is_file():
        failures.append(f"missing_{label}")


def _core_report(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize skill-file-report / skill-workflow-summary / core report payloads."""
    for key in ("core_report", "audio_process_report", "selected_report"):
        embedded = payload.get(key)
        if isinstance(embedded, dict) and (
            "quality_guard" in embedded
            or "breath_cleanup" in embedded
            or "pause_cleanup" in embedded
        ):
            return embedded
    files = payload.get("files")
    if isinstance(files, list):
        for item in files:
            if not isinstance(item, dict):
                continue
            nested = item.get("core_report") or item.get("audio_process_report") or item
            if isinstance(nested, dict) and (
                "quality_guard" in nested
                or "breath_cleanup" in nested
                or "pause_cleanup" in nested
            ):
                return nested
    return payload


def _paths_refer_same_file(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return Path(left).as_posix() == Path(right).as_posix()


def _report_deliverable_path(report: dict[str, Any], key: str) -> Path | None:
    deliverables = report.get("deliverables")
    if isinstance(deliverables, dict):
        value = deliverables.get(key)
        if value:
            return Path(str(value))
    value = report.get(key)
    if value and key in {"final_wav", "final_mp3", "wav", "mp3"}:
        return Path(str(value))
    return None


def _repair_intent_context(
    report: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    auto_selection = payload.get("auto_selection")
    if not isinstance(auto_selection, dict):
        auto_selection = report.get("auto_selection")
    if not isinstance(auto_selection, dict):
        auto_selection = {}
    incomplete = (
        report.get("delivery_incomplete_for_repair_intent")
        if "delivery_incomplete_for_repair_intent" in report
        else payload.get("delivery_incomplete_for_repair_intent")
    )
    if incomplete is None:
        incomplete = auto_selection.get("delivery_incomplete_for_repair_intent")
    scorecard = report.get("repair_scorecard")
    if not isinstance(scorecard, dict):
        scorecard = payload.get("repair_scorecard")
    if not isinstance(scorecard, dict):
        scorecard = auto_selection.get("repair_scorecard")
    return {
        "delivery_incomplete_for_repair_intent": bool(incomplete),
        "repair_scorecard": scorecard if isinstance(scorecard, dict) else None,
    }


def _report_gate_failures(
    report: dict[str, Any],
    *,
    payload: dict[str, Any] | None = None,
    repair_intent: bool = False,
) -> list[str]:
    failures: list[str] = []
    quality_guard = report.get("quality_guard") or {}
    breath_cleanup = report.get("breath_cleanup") or {}
    pause_cleanup = report.get("pause_cleanup") or {}

    if str(quality_guard.get("status", "")).upper() != "PASS":
        failures.append("report_quality_guard_not_pass")
    if bool(quality_guard.get("release_blocked", True)):
        failures.append("report_release_blocked")
    if quality_guard.get("failures"):
        failures.append("report_quality_guard_failures")
    if str(breath_cleanup.get("status", "")).upper() != "PASS":
        failures.append("report_breath_cleanup_not_pass")
    if breath_cleanup.get("final_residual_windows"):
        failures.append("report_breath_residuals_present")
    if str(pause_cleanup.get("status", "")).upper() != "PASS":
        failures.append("report_pause_cleanup_not_pass")
    if pause_cleanup.get("mode") != "speech_safe_autogate":
        failures.append("report_pause_mode_invalid")
    if pause_cleanup.get("allow_bridge_windows") is not False:
        failures.append("report_pause_bridge_not_disabled")
    if pause_cleanup.get("final_residual_windows"):
        failures.append("report_pause_residuals_present")
    if "empty_assessment_windows" in (pause_cleanup.get("failures") or []):
        failures.append("report_empty_assessment_windows")
    if repair_intent:
        context = _repair_intent_context(report, payload or {})
        if context["delivery_incomplete_for_repair_intent"]:
            failures.append("delivery_incomplete_for_repair_intent")
        scorecard = context["repair_scorecard"]
        if not isinstance(scorecard, dict):
            failures.append("missing_repair_scorecard")
        elif str(scorecard.get("status", "")).upper() != "PASS":
            failures.append("repair_scorecard_incomplete")
    return failures


def _report_gate_summary(report: dict[str, Any]) -> dict[str, Any]:
    quality_guard = report.get("quality_guard") or {}
    breath_cleanup = report.get("breath_cleanup") or {}
    pause_cleanup = report.get("pause_cleanup") or {}
    return {
        "preset_name": report.get("preset_name"),
        "quality_guard": {
            "status": quality_guard.get("status"),
            "release_blocked": quality_guard.get("release_blocked"),
            "failures": quality_guard.get("failures") or [],
        },
        "breath_cleanup": {
            "status": breath_cleanup.get("status"),
            "final_residual_count": len(
                breath_cleanup.get("final_residual_windows") or []
            ),
            "failures": breath_cleanup.get("failures") or [],
        },
        "pause_cleanup": {
            "status": pause_cleanup.get("status"),
            "mode": pause_cleanup.get("mode"),
            "allow_bridge_windows": pause_cleanup.get("allow_bridge_windows"),
            "final_residual_count": len(
                pause_cleanup.get("final_residual_windows") or []
            ),
            "failures": pause_cleanup.get("failures") or [],
        },
    }


def verify_delivery(
    *,
    source: Path,
    final_wav: Path,
    report_json: Path,
    final_mp3: Path | None = None,
    ffmpeg_bin: str = "ffmpeg",
    ffprobe_bin: str = "ffprobe",
    repair_intent: bool = False,
    pair_evaluator: Callable[..., dict[str, Any]] = evaluate_audio_pair,
    probe: Callable[[Path, str], dict[str, Any]] = ffprobe_media,
) -> dict[str, Any]:
    failures: list[str] = []
    _require_file(source, "source", failures)
    _require_file(final_wav, "final_wav", failures)
    _require_file(report_json, "report_json", failures)
    if final_mp3 is not None:
        _require_file(final_mp3, "final_mp3", failures)
    if failures:
        return {
            "evidence_level": "DIRECTLY VERIFIED",
            "status": "FAIL",
            "failures": failures,
            "release_blocked": True,
        }

    report_payload = json.loads(report_json.read_text(encoding="utf-8"))
    report = _core_report(report_payload)
    failures.extend(
        _report_gate_failures(
            report,
            payload=report_payload,
            repair_intent=repair_intent,
        )
    )

    report_wav = _report_deliverable_path(report, "wav") or _report_deliverable_path(
        report, "final_wav"
    )
    if report_wav is not None and not _paths_refer_same_file(report_wav, final_wav):
        failures.append("report_final_wav_mismatch")
    if final_mp3 is not None:
        report_mp3 = _report_deliverable_path(report, "mp3") or _report_deliverable_path(
            report, "final_mp3"
        )
        if report_mp3 is not None and not _paths_refer_same_file(report_mp3, final_mp3):
            failures.append("report_final_mp3_mismatch")

    source_meta = probe(source, ffprobe_bin)
    wav_meta = probe(final_wav, ffprobe_bin)
    mp3_meta = probe(final_mp3, ffprobe_bin) if final_mp3 is not None else None
    if source_meta.get("sample_rate") != wav_meta.get("sample_rate"):
        failures.append("sample_rate_changed")
    if source_meta.get("channels") != wav_meta.get("channels"):
        failures.append("channel_layout_changed")
    source_duration = float(source_meta.get("duration_seconds") or 0.0)
    wav_duration = float(wav_meta.get("duration_seconds") or 0.0)
    if abs(source_duration - wav_duration) > 0.02:
        failures.append("wav_duration_changed")
    if mp3_meta is not None:
        if source_meta.get("sample_rate") != mp3_meta.get("sample_rate"):
            failures.append("mp3_sample_rate_changed")
        if source_meta.get("channels") != mp3_meta.get("channels"):
            failures.append("mp3_channel_layout_changed")
        mp3_duration = float(mp3_meta.get("duration_seconds") or 0.0)
        if abs(wav_duration - mp3_duration) > 0.1:
            failures.append("mp3_duration_mismatch")

    pair_report = pair_evaluator(
        source=source,
        processed=final_wav,
        ffmpeg_bin=ffmpeg_bin,
        ffprobe_bin=ffprobe_bin,
    )
    pair_guard = pair_report.get("quality_guard") or {}
    if str(pair_guard.get("status", "")).upper() != "PASS":
        failures.append("independent_quality_guard_not_pass")
    if bool(pair_guard.get("release_blocked", True)):
        failures.append("independent_release_blocked")
    for failure in pair_guard.get("failures") or []:
        failures.append(f"independent_{failure}")

    artifacts: dict[str, Any] = {
        "source": {
            "path": str(source.resolve()),
            "sha256": _sha256_file(source),
            "metadata": source_meta,
        },
        "final_wav": {
            "path": str(final_wav.resolve()),
            "sha256": _sha256_file(final_wav),
            "metadata": wav_meta,
        },
        "report_json": {
            "path": str(report_json.resolve()),
            "sha256": _sha256_file(report_json),
        },
    }
    if final_mp3 is not None:
        artifacts["final_mp3"] = {
            "path": str(final_mp3.resolve()),
            "sha256": _sha256_file(final_mp3),
            "metadata": mp3_meta,
        }

    unique_failures = list(dict.fromkeys(failures))
    return {
        "evidence_level": "DIRECTLY VERIFIED",
        "status": "FAIL" if unique_failures else "PASS",
        "artifacts": artifacts,
        "report_gates": _report_gate_summary(report),
        "independent_pair": pair_report,
        "repair_intent": repair_intent,
        "failures": unique_failures,
        "release_blocked": bool(unique_failures),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bind final media hashes to report gates and an independent source comparison."
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--final-wav", required=True)
    parser.add_argument("--final-mp3")
    parser.add_argument(
        "--report",
        required=True,
        help="audio_process_report.json, skill-file-report.json, or skill-workflow-summary.json",
    )
    parser.add_argument("--output", required=True, help="Verification manifest JSON")
    parser.add_argument(
        "--repair-intent",
        action="store_true",
        help="Fail when delivery_incomplete_for_repair_intent or repair_scorecard is incomplete",
    )
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--ffprobe-bin", default="ffprobe")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = verify_delivery(
        source=Path(args.source),
        final_wav=Path(args.final_wav),
        final_mp3=Path(args.final_mp3) if args.final_mp3 else None,
        report_json=Path(args.report),
        repair_intent=bool(args.repair_intent),
        ffmpeg_bin=args.ffmpeg_bin,
        ffprobe_bin=args.ffprobe_bin,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(output)
    return 1 if manifest["release_blocked"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
