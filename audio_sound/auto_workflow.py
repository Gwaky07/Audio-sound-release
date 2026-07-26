from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Mapping, Sequence

from .release_audit import normalize_audit_text


@dataclass(frozen=True)
class AutoCandidateSpec:
    candidate_id: str
    preset_name: str
    workflow_mode: str
    requires_asr: bool
    noise_windows: tuple[str, ...] = ()
    unavailable_reason: str | None = None


AUTO_CANDIDATE_RECIPES: dict[str, dict[str, Any]] = {
    "natural_baseline": {
        "preset_name": "natural",
        "workflow_mode": "natural",
        "requires_asr": False,
        "lowpass_hz": None,
        "gate_enabled": False,
        "hard_mute_enabled": False,
        "deepfilternet_enabled": False,
        "respiro_enabled": False,
    },
    "final_repair_best": {
        "preset_name": "final",
        "workflow_mode": "final",
        "requires_asr": False,
        "best_repair_candidate": True,
        "lowpass_hz": None,
        "gate_enabled": False,
        "hard_mute_enabled": False,
        "deepfilternet_enabled": False,
        "respiro_enabled": False,
        "post_filter_enabled": False,
    },
    "clarity_leveling_safe": {
        "preset_name": "clarity-leveling-safe",
        "workflow_mode": "clarity-leveling-safe",
        "requires_asr": True,
        "lowpass_hz": None,
        "gate_enabled": False,
        "hard_mute_enabled": False,
        "deepfilternet_enabled": False,
        "respiro_enabled": False,
    },
    "noise_cleanup_safe": {
        "preset_name": "noise-cleanup-safe",
        "workflow_mode": "noise-cleanup-safe",
        "requires_asr": True,
        "lowpass_hz": None,
        "gate_enabled": False,
        "hard_mute_enabled": False,
        "deepfilternet_enabled": False,
        "respiro_enabled": False,
    },
    "local_defect_safe": {
        "preset_name": "natural",
        "workflow_mode": "natural",
        "requires_asr": True,
        "lowpass_hz": None,
        "gate_enabled": False,
        "hard_mute_enabled": False,
        "deepfilternet_enabled": False,
        "respiro_enabled": False,
        "minimum_floor_gain": 0.25,
        "minimum_fade_ms": 8,
    },
    "respiro_breath_safe": {
        "preset_name": "respiro-breath-safe",
        "workflow_mode": "respiro-breath-safe",
        "requires_asr": False,
        "model_candidate": True,
        "lowpass_hz": None,
        "gate_enabled": False,
        "hard_mute_enabled": False,
        "deepfilternet_enabled": False,
        "respiro_enabled": True,
    },
    "deepfilter_denoise_safe": {
        "preset_name": "deepfilter-denoise-safe",
        "workflow_mode": "deepfilter-denoise-safe",
        "requires_asr": False,
        "model_candidate": True,
        "lowpass_hz": None,
        "gate_enabled": False,
        "hard_mute_enabled": False,
        "deepfilternet_enabled": True,
        "respiro_enabled": False,
        "post_filter_enabled": False,
    },
    "model_combined_review": {
        "preset_name": "model-combined-review",
        "workflow_mode": "model-combined-review",
        "requires_asr": True,
        "model_candidate": True,
        "lowpass_hz": None,
        "gate_enabled": False,
        "hard_mute_enabled": False,
        "deepfilternet_enabled": True,
        "respiro_enabled": True,
        "post_filter_enabled": False,
    },
}


def validate_auto_recipe(recipe: Mapping[str, Any]) -> None:
    forbidden_flags = ("gate_enabled", "hard_mute_enabled")
    enabled = [name for name in forbidden_flags if bool(recipe.get(name))]
    if (
        recipe.get("deepfilternet_enabled") or recipe.get("respiro_enabled")
    ) and not recipe.get("model_candidate"):
        enabled.append("unscoped_model_stage")
    if recipe.get("lowpass_hz") is not None:
        enabled.append("lowpass_hz")
    if enabled:
        raise ValueError(f"Auto recipe enables destructive cleanup: {', '.join(enabled)}")
    floor_gain = recipe.get("minimum_floor_gain")
    if floor_gain is not None and float(floor_gain) < 0.25:
        raise ValueError("Auto local cleanup floor gain must be at least 0.25")
    fade_ms = recipe.get("minimum_fade_ms")
    if fade_ms is not None and int(fade_ms) < 8:
        raise ValueError("Auto local cleanup fade must be at least 8 ms")


def build_candidate_specs(
    diagnostics: Mapping[str, Any],
    *,
    include_local_defect: bool = False,
    runtime_capabilities: Mapping[str, Any] | None = None,
    repair_intent: bool = True,
) -> list[AutoCandidateSpec]:
    candidate_ids = ["natural_baseline"]
    if repair_intent:
        candidate_ids.append("final_repair_best")
    candidate_ids.append("clarity_leveling_safe")
    windows = diagnostics.get("candidate_noise_windows")
    if diagnostics.get("stationary_noise") and isinstance(windows, list) and windows:
        candidate_ids.append("noise_cleanup_safe")
    if include_local_defect:
        candidate_ids.append("local_defect_safe")
    runtime = runtime_capabilities or {}
    model_reasons: dict[str, str | None] = {}
    pause_ratio = float(diagnostics.get("pause_ratio", 0.0) or 0.0)
    estimated_snr = diagnostics.get("estimated_snr_db")
    if runtime_capabilities is not None and pause_ratio >= 0.05:
        candidate_ids.append("respiro_breath_safe")
        if not bool((runtime.get("respiro_en") or {}).get("ready")):
            model_reasons["respiro_breath_safe"] = "runtime_unavailable"
    if (
        runtime_capabilities is not None
        and diagnostics.get("stationary_noise")
        and isinstance(windows, list)
        and windows
        and estimated_snr is not None
        and float(estimated_snr) <= 35.0
    ):
        candidate_ids.append("deepfilter_denoise_safe")
        if not bool((runtime.get("deepfilternet") or {}).get("ready")):
            model_reasons["deepfilter_denoise_safe"] = "runtime_unavailable"

    specs: list[AutoCandidateSpec] = []
    for candidate_id in candidate_ids:
        recipe = AUTO_CANDIDATE_RECIPES[candidate_id]
        validate_auto_recipe(recipe)
        noise_windows: tuple[str, ...] = ()
        if candidate_id == "noise_cleanup_safe":
            noise_windows = tuple(
                f"{float(window['start_seconds'])}:{float(window['end_seconds'])}"
                for window in windows
                if isinstance(window, Mapping)
                and window.get("start_seconds") is not None
                and window.get("end_seconds") is not None
            )
        specs.append(
            AutoCandidateSpec(
                candidate_id=candidate_id,
                preset_name=str(recipe["preset_name"]),
                workflow_mode=str(recipe["workflow_mode"]),
                requires_asr=bool(recipe["requires_asr"]),
                noise_windows=noise_windows,
                unavailable_reason=model_reasons.get(candidate_id),
            )
        )
    return specs


def _payload_text(payload: Mapping[str, Any]) -> str:
    segments = payload.get("segments")
    if not isinstance(segments, list):
        raise ValueError("ASR payload must contain a segments list")
    return normalize_audit_text(
        "".join(
            str(segment.get("text", ""))
            for segment in segments
            if isinstance(segment, Mapping)
        )
    )


def _edit_distance(source: str, candidate: str) -> int:
    if not source:
        return len(candidate)
    previous = list(range(len(candidate) + 1))
    for source_index, source_character in enumerate(source, start=1):
        current = [source_index]
        for candidate_index, candidate_character in enumerate(candidate, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[candidate_index] + 1,
                    previous[candidate_index - 1]
                    + (source_character != candidate_character),
                )
            )
        previous = current
    return previous[-1]


def _omitted_characters(source: str, candidate: str) -> int:
    omitted = 0
    for tag, source_start, source_end, candidate_start, candidate_end in SequenceMatcher(
        None,
        source,
        candidate,
        autojunk=False,
    ).get_opcodes():
        if tag == "delete":
            omitted += source_end - source_start
        elif tag == "replace":
            omitted += max(0, (source_end - source_start) - (candidate_end - candidate_start))
    return omitted


def build_asr_preservation_guard(
    source_payloads: Sequence[Mapping[str, Any]],
    candidate_payloads: Sequence[Mapping[str, Any]],
    *,
    expected_media_sha256: str | None = None,
    minimum_engines: int = 2,
    maximum_cer_increase: float = 0.005,
) -> dict[str, Any]:
    source_by_engine = {
        str(payload.get("engine", "")).strip().casefold(): payload
        for payload in source_payloads
        if str(payload.get("engine", "")).strip()
    }
    candidate_by_engine = {
        str(payload.get("engine", "")).strip().casefold(): payload
        for payload in candidate_payloads
        if str(payload.get("engine", "")).strip()
    }
    common_engines = sorted(set(source_by_engine) & set(candidate_by_engine))
    failures: list[str] = []
    if len(common_engines) < minimum_engines:
        failures.append("insufficient_independent_asr_engines")

    expected_hash = expected_media_sha256.upper() if expected_media_sha256 else None
    engine_checks: list[dict[str, Any]] = []
    for engine in common_engines:
        source_text = _payload_text(source_by_engine[engine])
        candidate_payload = candidate_by_engine[engine]
        candidate_text = _payload_text(candidate_payload)
        distance = _edit_distance(source_text, candidate_text)
        cer = distance / max(1, len(source_text))
        omitted = _omitted_characters(source_text, candidate_text)
        bound_hash = str(candidate_payload.get("media_sha256", "")).upper() or None
        hash_matches = expected_hash is None or bound_hash == expected_hash
        engine_checks.append(
            {
                "engine": engine,
                "source_characters": len(source_text),
                "candidate_characters": len(candidate_text),
                "character_error_rate": round(cer, 6),
                "omitted_characters": omitted,
                "media_sha256": bound_hash,
                "media_hash_matches": hash_matches,
            }
        )
        if omitted:
            failures.append("asr_character_omission")
        if cer > maximum_cer_increase:
            failures.append("asr_character_error_increased")
        if not hash_matches:
            failures.append("asr_media_hash_mismatch")

    failures = list(dict.fromkeys(failures))
    unavailable = "insufficient_independent_asr_engines" in failures
    return {
        "evidence_level": "MODEL-INFERRED",
        "status": "UNAVAILABLE" if unavailable else ("FAIL" if failures else "PASS"),
        "minimum_engines": minimum_engines,
        "distinct_engines": common_engines,
        "maximum_cer_increase": maximum_cer_increase,
        "engine_checks": engine_checks,
        "failures": failures,
        "release_blocked": bool(failures),
    }


def resolve_candidate_asr_guard(
    *,
    candidate_id: str,
    requires_asr: bool,
    source_asr_payloads: Sequence[Mapping[str, Any]] | None = None,
    candidate_asr_payloads: Sequence[Mapping[str, Any]] | None = None,
    expected_media_sha256: str | None = None,
) -> dict[str, Any]:
    if not requires_asr:
        return {
            "evidence_level": "DIRECTLY VERIFIED",
            "status": "NOT_REQUIRED",
            "failures": [],
            "release_blocked": False,
        }
    source_payloads = list(source_asr_payloads or [])
    candidate_payloads = list(candidate_asr_payloads or [])
    if not source_payloads or not candidate_payloads:
        return {
            "evidence_level": "UNVERIFIED",
            "status": "UNAVAILABLE",
            "failures": ["asr_evidence_missing"],
            "release_blocked": True,
        }
    return build_asr_preservation_guard(
        source_payloads,
        candidate_payloads,
        expected_media_sha256=expected_media_sha256,
    )


def score_candidate(
    *,
    candidate_id: str,
    quality_guard: Mapping[str, Any],
    asr_guard: Mapping[str, Any],
) -> float:
    if quality_guard.get("release_blocked") or asr_guard.get("release_blocked"):
        return -1.0
    score = 100.0
    worst_attenuation = float(quality_guard.get("worst_relative_attenuation_db", 0.0))
    gain_spread = float(quality_guard.get("relative_gain_spread_db", 0.0))
    score -= max(0.0, -worst_attenuation) * 4.0
    score -= max(0.0, gain_spread - 2.0) * 2.0
    spectral = quality_guard.get("spectral_band_deltas_db") or {}
    for band_name, weight in (
        ("2-4k", 5.0),
        ("4-6k", 5.0),
        ("6-8k", 4.0),
        ("8-10k", 3.0),
        ("10-12k", 2.0),
    ):
        delta = float(spectral.get(band_name, 0.0))
        if delta < 0:
            score -= abs(delta) * weight
    if candidate_id == "final_repair_best":
        score += 12.0
    elif candidate_id == "noise_cleanup_safe":
        score += 4.0
    elif candidate_id == "clarity_leveling_safe":
        score += 2.0
    elif candidate_id == "natural_baseline":
        score += 1.0
    asr_checks = asr_guard.get("engine_checks") or []
    if asr_checks:
        mean_cer = sum(float(item.get("character_error_rate", 0.0)) for item in asr_checks) / len(
            asr_checks
        )
        score -= mean_cer * 200.0
    return round(score, 3)


def _model_evidence(
    candidate_id: str,
    core_report: Mapping[str, Any],
) -> dict[str, Any]:
    stage_status = core_report.get("stage_status") or {}
    respiro_applied = bool((stage_status.get("respiro") or {}).get("applied"))
    deepfilter_applied = bool((stage_status.get("deepfilternet") or {}).get("applied"))
    uses_respiro = candidate_id in {"respiro_breath_safe", "model_combined_review"}
    uses_deepfilter = candidate_id in {"deepfilter_denoise_safe", "model_combined_review"}
    model_applied = (not uses_respiro or respiro_applied) and (
        not uses_deepfilter or deepfilter_applied
    )
    respiro_succeeded = bool(core_report.get("respiro_succeeded"))
    deepfilter_succeeded = deepfilter_applied
    model_succeeded = model_applied and (
        not uses_respiro or respiro_succeeded
    ) and (not uses_deepfilter or deepfilter_succeeded)
    fallback_used = uses_respiro and str(
        core_report.get("respiro_detection_mode", "")
    ) == "fallback"

    input_diagnostics = core_report.get("input_diagnostics") or {}
    output_diagnostics = core_report.get("output_diagnostics") or {}
    snr_improvement_db = float(output_diagnostics.get("estimated_snr_db", 0.0) or 0.0) - float(
        input_diagnostics.get("estimated_snr_db", 0.0) or 0.0
    )
    breath_attenuation_db = float(core_report.get("breath_attenuation_db", 0.0) or 0.0)
    breath_windows = len(core_report.get("breath_onset_windows") or [])
    benefit_score = max(0.0, snr_improvement_db) * 5.0
    if breath_windows and breath_attenuation_db >= 1.0:
        benefit_score += min(20.0, breath_attenuation_db * 2.0)
    return {
        "model_applied": model_applied,
        "model_succeeded": model_succeeded,
        "fallback_used": fallback_used,
        "benefit_metrics": {
            "snr_improvement_db": round(snr_improvement_db, 3),
            "breath_attenuation_db": round(breath_attenuation_db, 3),
            "breath_window_count": breath_windows,
        },
        "benefit_score": round(benefit_score, 3),
    }


def _apply_model_guards(
    quality_guard: Mapping[str, Any],
    model_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    guard = dict(quality_guard)
    failures = list(guard.get("failures") or [])
    if not model_evidence.get("model_applied"):
        failures.append("model_not_applied")
    if model_evidence.get("fallback_used"):
        failures.append("model_fallback_used")
    if not model_evidence.get("model_succeeded"):
        failures.append("model_execution_failed")
    if float(model_evidence.get("benefit_score", 0.0)) < 1.0:
        failures.append("no_measurable_model_benefit")

    worst_attenuation = float(guard.get("worst_relative_attenuation_db", 0.0) or 0.0)
    gain_spread = float(guard.get("relative_gain_spread_db", 0.0) or 0.0)
    correlation = float(guard.get("active_level_correlation", 1.0) or 0.0)
    if worst_attenuation < -3.0:
        failures.append("model_active_speech_attenuated")
    if gain_spread > 6.0:
        failures.append("model_gain_instability")
    if correlation < 0.92:
        failures.append("model_local_correlation_lost")
    spectral = guard.get("spectral_band_deltas_db") or {}
    if any(
        float(spectral.get(band, 0.0)) < limit
        for band, limit in (
            ("2-4k", -2.0),
            ("4-6k", -2.0),
            ("6-8k", -2.0),
            ("8-10k", -3.0),
            ("10-12k", -3.0),
        )
    ):
        failures.append("model_spectral_clarity_lost")
    guard["failures"] = list(dict.fromkeys(failures))
    guard["release_blocked"] = bool(guard["failures"])
    return guard


def _harm_score(quality_guard: Mapping[str, Any]) -> float:
    score = max(
        0.0,
        -float(quality_guard.get("worst_relative_attenuation_db", 0.0) or 0.0),
    ) * 4.0
    score += max(
        0.0,
        float(quality_guard.get("relative_gain_spread_db", 0.0) or 0.0) - 2.0,
    ) * 2.0
    for delta in (quality_guard.get("spectral_band_deltas_db") or {}).values():
        score += max(0.0, -float(delta))
    return round(score, 3)


def build_evaluated_candidate(
    *,
    candidate_id: str,
    requires_asr: bool,
    quality_guard: Mapping[str, Any],
    source_asr_payloads: Sequence[Mapping[str, Any]] | None = None,
    candidate_asr_payloads: Sequence[Mapping[str, Any]] | None = None,
    expected_media_sha256: str | None = None,
    recipe: Mapping[str, Any] | None = None,
    applied_stages: Sequence[str] | None = None,
    deliverables: Mapping[str, Any] | None = None,
    core_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_recipe = dict(recipe or AUTO_CANDIDATE_RECIPES.get(candidate_id, {}))
    model_evidence: dict[str, Any] = {
        "model_applied": False,
        "model_succeeded": False,
        "fallback_used": False,
        "benefit_metrics": {},
        "benefit_score": 0.0,
    }
    resolved_quality_guard = dict(quality_guard)
    if resolved_recipe.get("model_candidate"):
        model_evidence = _model_evidence(candidate_id, core_report or {})
        resolved_quality_guard = _apply_model_guards(
            resolved_quality_guard,
            model_evidence,
        )
    asr_guard = resolve_candidate_asr_guard(
        candidate_id=candidate_id,
        requires_asr=requires_asr,
        source_asr_payloads=source_asr_payloads,
        candidate_asr_payloads=candidate_asr_payloads,
        expected_media_sha256=expected_media_sha256,
    )
    return {
        "candidate_id": candidate_id,
        "recipe": resolved_recipe,
        "applied_stages": list(applied_stages or []),
        "quality_guard": resolved_quality_guard,
        "asr_guard": asr_guard,
        "score": score_candidate(
            candidate_id=candidate_id,
            quality_guard=resolved_quality_guard,
            asr_guard=asr_guard,
        )
        + float(model_evidence.get("benefit_score", 0.0)),
        "harm_score": _harm_score(resolved_quality_guard),
        **model_evidence,
        "deliverables": dict(deliverables or {}),
        "core_report": dict(core_report or {}),
    }


def select_auto_candidate(
    candidates: Sequence[Mapping[str, Any]],
    *,
    repair_intent: bool = True,
) -> dict[str, Any]:
    eligible: list[Mapping[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for candidate in candidates:
        quality_guard = candidate.get("quality_guard") or {}
        asr_guard = candidate.get("asr_guard") or {}
        failures = [
            *quality_guard.get("failures", []),
            *asr_guard.get("failures", []),
        ]
        if quality_guard.get("release_blocked") or asr_guard.get("release_blocked"):
            rejected.append(
                {
                    "candidate_id": candidate.get("candidate_id"),
                    "failures": list(dict.fromkeys(str(item) for item in failures)),
                    "score": candidate.get("score"),
                }
            )
        else:
            eligible.append(candidate)
    if not eligible:
        raise RuntimeError("No auto candidate passed all release guards")

    def _rank(item: Mapping[str, Any]) -> tuple[int, float]:
        candidate_id = str(item.get("candidate_id"))
        priority = {
            "final_repair_best": 4,
            "model_combined_review": 3,
            "respiro_breath_safe": 2,
            "deepfilter_denoise_safe": 2,
            "noise_cleanup_safe": 2,
            "clarity_leveling_safe": 1,
            "natural_baseline": 0,
        }.get(candidate_id, 0)
        return (priority, float(item.get("score", 0.0)))

    winner = max(eligible, key=_rank)
    candidate_id = str(winner["candidate_id"])
    rejected_enhancements = any(
        item["candidate_id"] != "natural_baseline" for item in rejected
    )
    if candidate_id == "final_repair_best":
        selection_reason = "best_repair_selected"
    elif candidate_id == "natural_baseline" and rejected_enhancements:
        selection_reason = (
            "safe_fallback_incomplete" if repair_intent else "safe_fallback"
        )
    elif candidate_id == "natural_baseline" and repair_intent:
        selection_reason = "safe_fallback_incomplete"
    else:
        selection_reason = "highest_safe_score"
    incomplete = bool(repair_intent and candidate_id == "natural_baseline")
    return {
        **dict(winner),
        "selection_reason": selection_reason,
        "rejected_candidates": rejected,
        "delivery_incomplete_for_repair_intent": incomplete,
        "manual_review_required": any(
            "asr_character_omission" in item.get("failures", [])
            or "asr_character_error_increased" in item.get("failures", [])
            for item in rejected
        )
        and candidate_id == "natural_baseline",
    }
