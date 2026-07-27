from __future__ import annotations

from typing import Any, Mapping


def build_capability_plan(
    diagnostics: Mapping[str, Any],
    *,
    runtime_capabilities: Mapping[str, Any] | None = None,
    confirmed_mouth_noise: bool = False,
) -> dict[str, Any]:
    """Plan fixed repair capabilities from diagnostics. No free-form DSP params."""
    runtime = runtime_capabilities or {}
    pause_ratio = float(diagnostics.get("pause_ratio", 0.0) or 0.0)
    estimated_snr = diagnostics.get("estimated_snr_db")
    snr_value = float(estimated_snr) if estimated_snr is not None else None
    stationary_noise = bool(diagnostics.get("stationary_noise"))
    windows = diagnostics.get("candidate_noise_windows")
    has_noise_windows = isinstance(windows, list) and bool(windows)
    clean_source = (
        not stationary_noise and snr_value is not None and snr_value >= 35.0
    )
    respiro_ready = bool((runtime.get("respiro_en") or {}).get("ready"))
    deepfilter_ready = bool((runtime.get("deepfilternet") or {}).get("ready"))

    breath_needed = pause_ratio >= 0.05
    pause_needed = pause_ratio >= 0.03 or breath_needed
    denoise_needed = stationary_noise and has_noise_windows and (
        snr_value is None or snr_value <= 35.0
    )
    if clean_source:
        denoise_needed = False

    capabilities = {
        "breath_cleanup": {
            "needed": breath_needed,
            "skipped_reason": None
            if breath_needed
            else "insufficient_pause_ratio_for_breath_evidence",
            "execution": "final.breath_cleanup_policy",
            "optional_model": "respiro" if respiro_ready else None,
        },
        "pause_cleanup": {
            "needed": pause_needed,
            "skipped_reason": None
            if pause_needed
            else "insufficient_silence_transition_evidence",
            "execution": "final.pause_residual_cleanup",
        },
        "denoise": {
            "needed": denoise_needed,
            "skipped_reason": (
                "skip_secondary_denoise_clean_source"
                if clean_source
                else (
                    None
                    if denoise_needed
                    else "no_stationary_noise_window_evidence"
                )
            ),
            "execution": (
                "deepfilter_denoise_safe_or_conservative_afftdn"
                if denoise_needed
                else "skipped"
            ),
            "optional_model": "deepfilternet"
            if denoise_needed and deepfilter_ready
            else None,
        },
        "clarity_eq": {
            "needed": True,
            "skipped_reason": None,
            "execution": "final.equalizer",
        },
        "deess": {
            "needed": True,
            "skipped_reason": None,
            "execution": "final.deesser",
        },
        "leveling": {
            "needed": True,
            "skipped_reason": None,
            "execution": "final.compressor",
        },
        "loudness": {
            "needed": True,
            "skipped_reason": None,
            "execution": "final.loudnorm",
        },
        "mouth_declick": {
            "needed": confirmed_mouth_noise,
            "skipped_reason": None
            if confirmed_mouth_noise
            else "unconfirmed_mouth_noise_keeps_global_declick_off",
            "execution": "narrow_window_only_when_confirmed",
        },
    }
    return {
        "intent": "best_repair",
        "primary_candidate": "final_repair_best",
        "clean_source": clean_source,
        "capabilities": capabilities,
    }


def _capability_status(
    *,
    needed: bool,
    applied: bool,
    failed: bool = False,
    skipped_reason: str | None = None,
) -> str:
    if failed:
        return "FAIL"
    if not needed:
        return "SKIP"
    if applied:
        return "PASS"
    if skipped_reason:
        return "SKIP"
    return "FAIL"


def build_repair_scorecard(
    *,
    core_report: Mapping[str, Any],
    quality_guard: Mapping[str, Any],
    capability_plan: Mapping[str, Any],
    candidate_id: str,
) -> dict[str, Any]:
    capabilities = capability_plan.get("capabilities") or {}
    stage_status = core_report.get("stage_status") or {}
    processing_steps = [
        str(item).casefold() for item in (core_report.get("processing_steps") or [])
    ]
    breath_cleanup = core_report.get("breath_cleanup") or {}
    pause_cleanup = core_report.get("pause_cleanup") or {}
    input_adaptations = {
        str(item) for item in (core_report.get("input_adaptations") or [])
    }
    failures = {
        str(item) for item in (quality_guard.get("failures") or [])
    }
    spectral = quality_guard.get("spectral_band_deltas_db") or {}

    breath_applied = bool(breath_cleanup) and str(
        breath_cleanup.get("status", "")
    ).upper() in {"PASS", "FAIL"}
    breath_failed = str(breath_cleanup.get("status", "")).upper() == "FAIL" or (
        "confirmed_breath_residual_after_retry" in failures
    )
    pause_applied = bool(pause_cleanup) and str(
        pause_cleanup.get("status", "")
    ).upper() in {"PASS", "FAIL"}
    pause_mode = str(pause_cleanup.get("mode") or "")
    pause_needed = bool((capabilities.get("pause_cleanup") or {}).get("needed"))
    pause_mode_invalid = (
        pause_needed
        and pause_applied
        and pause_mode != "speech_safe_autogate"
    )
    pause_failed = (
        str(pause_cleanup.get("status", "")).upper() == "FAIL"
        or "confirmed_pause_residual_after_cleanup" in failures
        or "empty_assessment_windows" in failures
        or pause_mode_invalid
    )

    denoise_applied = bool((stage_status.get("deepfilternet") or {}).get("applied")) or any(
        "denoise" in step or "afftdn" in step for step in processing_steps
    )
    if "skip_secondary_denoise_clean_source" in input_adaptations:
        denoise_applied = False

    clarity_applied = any("equalizer" in step or "clarity" in step for step in processing_steps)
    deess_applied = any("deesser" in step or "de-ess" in step for step in processing_steps)
    leveling_applied = any("compress" in step for step in processing_steps)
    loudness_applied = any("loudnorm" in step or "loudness" in step for step in processing_steps)

    no_swallow = not any(
        code in failures
        for code in (
            "active_speech_attenuated",
            "source_active_hard_mute",
            "model_active_speech_attenuated",
        )
    )
    no_muffle = "spectral_clarity_lost" not in failures and "model_spectral_clarity_lost" not in failures
    no_harshness = "spectral_harshness_increased" not in failures

    scorecard = {
        "breath_cleanup": {
            "status": _capability_status(
                needed=bool((capabilities.get("breath_cleanup") or {}).get("needed")),
                applied=breath_applied and not breath_failed,
                failed=breath_failed,
                skipped_reason=(capabilities.get("breath_cleanup") or {}).get(
                    "skipped_reason"
                ),
            ),
            "detail": {
                "status": breath_cleanup.get("status"),
                "final_residual_windows": breath_cleanup.get("final_residual_windows"),
            },
        },
        "pause_cleanup": {
            "status": _capability_status(
                needed=pause_needed,
                applied=pause_applied and not pause_failed,
                failed=pause_failed,
                skipped_reason=(capabilities.get("pause_cleanup") or {}).get(
                    "skipped_reason"
                ),
            ),
            "detail": {
                "status": pause_cleanup.get("status"),
                "mode": pause_cleanup.get("mode"),
                "final_residual_windows": pause_cleanup.get("final_residual_windows"),
                "mode_invalid": pause_mode_invalid,
            },
        },
        "denoise": {
            "status": _capability_status(
                needed=bool((capabilities.get("denoise") or {}).get("needed")),
                applied=denoise_applied,
                skipped_reason=(capabilities.get("denoise") or {}).get("skipped_reason"),
            ),
            "detail": {
                "deepfilternet_applied": bool(
                    (stage_status.get("deepfilternet") or {}).get("applied")
                ),
                "input_adaptations": sorted(input_adaptations),
            },
        },
        "clarity_eq": {
            "status": _capability_status(
                needed=bool((capabilities.get("clarity_eq") or {}).get("needed", True)),
                applied=clarity_applied or candidate_id == "final_repair_best",
            ),
        },
        "deess": {
            "status": _capability_status(
                needed=bool((capabilities.get("deess") or {}).get("needed", True)),
                applied=deess_applied or candidate_id == "final_repair_best",
            ),
        },
        "leveling": {
            "status": _capability_status(
                needed=bool((capabilities.get("leveling") or {}).get("needed", True)),
                applied=leveling_applied or candidate_id in {
                    "final_repair_best",
                    "clarity_leveling_safe",
                    "natural_baseline",
                },
            ),
        },
        "loudness": {
            "status": _capability_status(
                needed=bool((capabilities.get("loudness") or {}).get("needed", True)),
                applied=loudness_applied or candidate_id in {
                    "final_repair_best",
                    "clarity_leveling_safe",
                    "natural_baseline",
                },
            ),
        },
        "no_swallow": {
            "status": "PASS" if no_swallow else "FAIL",
            "detail": {
                "worst_relative_attenuation_db": quality_guard.get(
                    "worst_relative_attenuation_db"
                ),
                "active_level_correlation": quality_guard.get("active_level_correlation"),
            },
        },
        "no_muffle": {
            "status": "PASS" if no_muffle else "FAIL",
            "detail": {"spectral_band_deltas_db": spectral},
        },
        "no_harshness": {
            "status": "PASS" if no_harshness else "FAIL",
            "detail": {"spectral_band_deltas_db": spectral},
        },
    }

    hard_fail_keys = (
        "no_swallow",
        "no_muffle",
        "no_harshness",
        "breath_cleanup",
        "pause_cleanup",
    )
    failed_items = [
        key for key in hard_fail_keys if scorecard[key]["status"] == "FAIL"
    ]
    return {
        "candidate_id": candidate_id,
        "items": scorecard,
        "failed_items": failed_items,
        "status": "FAIL" if failed_items else "PASS",
    }


def delivery_completeness(
    *,
    candidate_id: str,
    repair_intent: bool = True,
) -> dict[str, Any]:
    incomplete = bool(repair_intent and candidate_id == "natural_baseline")
    return {
        "repair_intent": repair_intent,
        "delivery_incomplete_for_repair_intent": incomplete,
        "completion_status": "INCOMPLETE" if incomplete else "COMPLETE",
        "reason": (
            "natural_baseline_is_not_best_repair_completion"
            if incomplete
            else "best_repair_or_explicit_natural_path"
        ),
    }
