from __future__ import annotations

import argparse
import hashlib
import json
import math
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_CUT_TOLERANCE_SECONDS = 0.12
DEFAULT_SOURCE_ASR_TOLERANCE_SECONDS = 0.6
DEFAULT_TIMESTAMP_WINDOW_SECONDS = 2.0
MAX_TRANSCRIPT_JOIN_GAP_SECONDS = 1.5

TRADITIONAL_AUDIT_TRANSLATION = str.maketrans(
    {
        "實": "实",
        "踐": "践",
        "這": "这",
        "個": "个",
        "為": "为",
        "與": "与",
        "說": "说",
        "話": "话",
        "時": "时",
        "後": "后",
        "來": "来",
        "裡": "里",
        "裏": "里",
        "還": "还",
        "過": "过",
        "會": "会",
        "從": "从",
        "當": "当",
        "們": "们",
        "歷": "历",
        "敗": "败",
        "發": "发",
        "國": "国",
        "軍": "军",
        "學": "学",
        "業": "业",
        "體": "体",
        "現": "现",
        "開": "开",
        "關": "关",
        "係": "系",
        "樣": "样",
        "種": "种",
        "點": "点",
        "應": "应",
        "對": "对",
    }
)


@dataclass(frozen=True)
class TranscriptUnit:
    start_seconds: float
    end_seconds: float
    raw_text: str
    normalized_text: str


@dataclass(frozen=True)
class TranscriptStream:
    text: str
    start_seconds: tuple[float, ...]
    end_seconds: tuple[float, ...]


def _normalize_piece(text: str, replacements: Mapping[str, str] | None = None) -> str:
    normalized = unicodedata.normalize("NFKC", text).translate(TRADITIONAL_AUDIT_TRANSLATION)
    if replacements:
        for source, replacement in replacements.items():
            normalized = normalized.replace(str(source), str(replacement))
    return "".join(
        character
        for character in normalized
        if not character.isspace() and not unicodedata.category(character).startswith("P")
    )


def normalize_audit_text(text: str, replacements: Mapping[str, str] | None = None) -> str:
    return _normalize_piece(str(text), replacements)


def _coerce_seconds(value: Any, *, field_name: str) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"{field_name} must be a finite non-negative number")
    return seconds


def _transcript_units(
    payload: Mapping[str, Any],
    replacements: Mapping[str, str],
    *,
    channel: str,
) -> list[TranscriptUnit]:
    segments = payload.get("segments")
    if not isinstance(segments, list):
        raise ValueError("ASR payload must contain a 'segments' list")
    units: list[TranscriptUnit] = []
    for segment_index, segment in enumerate(segments):
        if not isinstance(segment, Mapping):
            continue
        words = segment.get("words")
        if channel == "word_timestamps":
            entries = words if isinstance(words, list) else []
        elif channel == "segment_text":
            entries = [segment]
        else:
            raise ValueError(f"Unsupported transcript channel: {channel}")
        for entry_index, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                continue
            raw_text = str(entry.get("word", entry.get("text", "")))
            normalized_text = normalize_audit_text(raw_text, replacements)
            if not normalized_text:
                continue
            start_value = entry.get("start", segment.get("start"))
            end_value = entry.get("end", segment.get("end"))
            if start_value is None or end_value is None:
                raise ValueError(
                    f"ASR segment {segment_index} entry {entry_index} is missing start/end timestamps"
                )
            start_seconds = _coerce_seconds(start_value, field_name="ASR start")
            end_seconds = _coerce_seconds(end_value, field_name="ASR end")
            if end_seconds < start_seconds:
                raise ValueError("ASR end timestamp must not precede start timestamp")
            units.append(TranscriptUnit(start_seconds, end_seconds, raw_text, normalized_text))
    return units


def _build_transcript_stream(units: Sequence[TranscriptUnit]) -> TranscriptStream:
    characters: list[str] = []
    starts: list[float] = []
    ends: list[float] = []
    previous_end: float | None = None
    for unit in units:
        if previous_end is not None and unit.start_seconds - previous_end > MAX_TRANSCRIPT_JOIN_GAP_SECONDS:
            characters.append("\0")
            starts.append(previous_end)
            ends.append(unit.start_seconds)
        duration = max(0.0, unit.end_seconds - unit.start_seconds)
        character_duration = duration / max(1, len(unit.normalized_text))
        for index, character in enumerate(unit.normalized_text):
            characters.append(character)
            starts.append(unit.start_seconds + character_duration * index)
            ends.append(unit.start_seconds + character_duration * (index + 1))
        previous_end = unit.end_seconds
    return TranscriptStream("".join(characters), tuple(starts), tuple(ends))


def _find_phrase_spans(stream: TranscriptStream, phrase: str) -> list[tuple[int, int]]:
    if not phrase:
        return []
    spans: list[tuple[int, int]] = []
    start_index = 0
    while True:
        match_index = stream.text.find(phrase, start_index)
        if match_index < 0:
            break
        spans.append((match_index, match_index + len(phrase)))
        start_index = match_index + 1
    return spans


def _span_to_evidence(
    stream: TranscriptStream,
    span: tuple[int, int],
    *,
    engine: str,
    matched_text: str,
    channel: str,
) -> dict[str, Any]:
    start_index, end_index = span
    return {
        "engine": engine,
        "channel": channel,
        "matched_text": matched_text,
        "start_seconds": stream.start_seconds[start_index],
        "end_seconds": stream.end_seconds[end_index - 1],
    }


def _find_forbidden_hits(
    payload_name: str,
    payload: Mapping[str, Any],
    *,
    target_phrases: Sequence[str],
    allowed_contexts: Sequence[str],
    replacements: Mapping[str, str],
) -> tuple[list[dict[str, Any]], list[TranscriptUnit]]:
    hits: list[dict[str, Any]] = []
    units_by_channel = [
        (channel, _transcript_units(payload, replacements, channel=channel))
        for channel in ("word_timestamps", "segment_text")
    ]
    units_by_channel = [(channel, units) for channel, units in units_by_channel if units]
    for channel, units in units_by_channel:
        stream = _build_transcript_stream(units)
        allowed_spans = [
            span
            for allowed_context in allowed_contexts
            for span in _find_phrase_spans(stream, allowed_context)
        ]
        seen_spans: set[tuple[int, int]] = set()
        for target_phrase in target_phrases:
            for span in _find_phrase_spans(stream, target_phrase):
                if span in seen_spans:
                    continue
                if any(
                    allowed_start <= span[0] and span[1] <= allowed_end
                    for allowed_start, allowed_end in allowed_spans
                ):
                    continue
                seen_spans.add(span)
                hits.append(
                    _span_to_evidence(
                        stream,
                        span,
                        engine=payload_name,
                        matched_text=target_phrase,
                        channel=channel,
                    )
                )
    hits.sort(key=lambda item: (item["start_seconds"], item["end_seconds"], item["engine"]))
    timestamp_units = next(
        (units for channel, units in units_by_channel if channel == "word_timestamps"),
        units_by_channel[0][1] if units_by_channel else [],
    )
    return hits, timestamp_units


def _find_context_hits(
    payload_name: str,
    payload: Mapping[str, Any],
    *,
    contexts: Sequence[str],
    replacements: Mapping[str, str],
) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for channel in ("word_timestamps", "segment_text"):
        units = _transcript_units(payload, replacements, channel=channel)
        if not units:
            continue
        stream = _build_transcript_stream(units)
        seen_spans: set[tuple[int, int]] = set()
        for context in contexts:
            for span in _find_phrase_spans(stream, context):
                if span in seen_spans:
                    continue
                seen_spans.add(span)
                hits.append(
                    _span_to_evidence(
                        stream,
                        span,
                        engine=payload_name,
                        matched_text=context,
                        channel=channel,
                    )
                )
    return hits


def _extract_cuts(payload: Mapping[str, Any]) -> list[dict[str, float]]:
    raw_cuts = payload.get("cuts")
    if not isinstance(raw_cuts, list):
        raise ValueError("Segment-removal report must contain a 'cuts' list")
    cuts: list[dict[str, float]] = []
    for index, raw_cut in enumerate(raw_cuts):
        if not isinstance(raw_cut, Mapping):
            raise ValueError(f"Segment-removal cut {index} must be an object")
        start_seconds = _coerce_seconds(raw_cut.get("start_seconds"), field_name="cut start_seconds")
        end_value = raw_cut.get("end_seconds")
        if end_value is None:
            end_seconds = math.inf
        else:
            end_seconds = _coerce_seconds(end_value, field_name="cut end_seconds")
        if end_seconds <= start_seconds:
            raise ValueError(f"Segment-removal cut {index} end must be greater than start")
        cuts.append({"start_seconds": start_seconds, "end_seconds": end_seconds})
    return cuts


def _cut_covers_occurrence(
    cut: Mapping[str, float],
    occurrence: Mapping[str, Any],
    *,
    tolerance_seconds: float,
) -> bool:
    start_seconds = _coerce_seconds(occurrence.get("start_seconds"), field_name="occurrence start_seconds")
    end_seconds = _coerce_seconds(occurrence.get("end_seconds"), field_name="occurrence end_seconds")
    return (
        cut["start_seconds"] <= start_seconds + tolerance_seconds
        and cut["end_seconds"] >= end_seconds - tolerance_seconds
    )


def _cut_overlaps_occurrence(
    cut: Mapping[str, float],
    occurrence: Mapping[str, Any],
    *,
    tolerance_seconds: float,
) -> bool:
    start_seconds = _coerce_seconds(occurrence.get("start_seconds"), field_name="occurrence start_seconds")
    end_seconds = _coerce_seconds(occurrence.get("end_seconds"), field_name="occurrence end_seconds")
    return (
        cut["start_seconds"] < end_seconds - tolerance_seconds
        and cut["end_seconds"] > start_seconds + tolerance_seconds
    )


def _deduplicate_hits(hits: Sequence[Mapping[str, Any]], *, tolerance_seconds: float) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for hit in sorted(hits, key=lambda item: (float(item["start_seconds"]), str(item["engine"]))):
        start_seconds = float(hit["start_seconds"])
        matching_group = next(
            (
                group
                for group in groups
                if abs(float(group["start_seconds"]) - start_seconds) <= tolerance_seconds
            ),
            None,
        )
        if matching_group is None:
            groups.append(
                {
                    "start_seconds": start_seconds,
                    "end_seconds": float(hit["end_seconds"]),
                    "observations": [dict(hit)],
                }
            )
        else:
            matching_group["end_seconds"] = max(
                float(matching_group["end_seconds"]),
                float(hit["end_seconds"]),
            )
            matching_group["observations"].append(dict(hit))
    return groups


def _normalization_replacements(
    requirements_payload: Mapping[str, Any],
    requirement: Mapping[str, Any],
) -> dict[str, str]:
    replacements: dict[str, str] = {}
    for owner in (requirements_payload, requirement):
        raw_replacements = owner.get("normalization_replacements", {})
        if not isinstance(raw_replacements, Mapping):
            raise ValueError("normalization_replacements must be an object")
        replacements.update({str(source): str(target) for source, target in raw_replacements.items()})
    return replacements


def _normalized_phrases(
    values: Sequence[Any],
    *,
    replacements: Mapping[str, str],
    field_name: str,
) -> list[str]:
    phrases: list[str] = []
    for value in values:
        normalized = normalize_audit_text(str(value), replacements)
        if not normalized:
            raise ValueError(f"{field_name} must not contain empty normalized text")
        if normalized not in phrases:
            phrases.append(normalized)
    return phrases


def _timestamp_checks(
    units_by_engine: Sequence[tuple[str, Sequence[TranscriptUnit]]],
    timestamps: Sequence[Any],
    *,
    window_seconds: float,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for timestamp_value in timestamps:
        timestamp = _coerce_seconds(timestamp_value, field_name="user-reported final timestamp")
        engine_checks: list[dict[str, Any]] = []
        for engine, units in units_by_engine:
            nearby_units = [
                unit
                for unit in units
                if unit.end_seconds >= timestamp - window_seconds
                and unit.start_seconds <= timestamp + window_seconds
            ]
            engine_checks.append(
                {
                    "engine": engine,
                    "checked": bool(nearby_units),
                    "nearby_text": "".join(unit.raw_text for unit in nearby_units),
                    "window_start_seconds": max(0.0, timestamp - window_seconds),
                    "window_end_seconds": timestamp + window_seconds,
                }
            )
        checks.append(
            {
                "timestamp_seconds": timestamp,
                "checked": bool(engine_checks) and all(item["checked"] for item in engine_checks),
                "engines": engine_checks,
            }
        )
    return checks


def _source_asr_matches_occurrence(
    occurrence: Mapping[str, Any],
    hits: Sequence[Mapping[str, Any]],
    *,
    tolerance_seconds: float,
) -> list[dict[str, Any]]:
    start_seconds = _coerce_seconds(occurrence.get("start_seconds"), field_name="occurrence start_seconds")
    end_seconds = _coerce_seconds(occurrence.get("end_seconds"), field_name="occurrence end_seconds")
    return [
        dict(hit)
        for hit in hits
        if float(hit["end_seconds"]) >= start_seconds - tolerance_seconds
        and float(hit["start_seconds"]) <= end_seconds + tolerance_seconds
    ]


def _audit_requirement(
    requirements_payload: Mapping[str, Any],
    requirement: Mapping[str, Any],
    *,
    source_asr_payloads: Sequence[tuple[str, Mapping[str, Any]]],
    final_asr_payloads: Sequence[tuple[str, Mapping[str, Any]]],
    cuts: Sequence[Mapping[str, float]],
) -> dict[str, Any]:
    requirement_id = str(requirement.get("id", "")).strip()
    if not requirement_id:
        raise ValueError("Each requirement must have a non-empty id")
    target = str(requirement.get("target", "")).strip()
    if not target:
        raise ValueError(f"Requirement {requirement_id!r} must have a non-empty target")
    replacements = _normalization_replacements(requirements_payload, requirement)
    raw_variants = requirement.get("variants", [])
    raw_allowed_contexts = requirement.get("allowed_contexts", [])
    if not isinstance(raw_variants, list) or not isinstance(raw_allowed_contexts, list):
        raise ValueError(f"Requirement {requirement_id!r} variants and allowed_contexts must be lists")
    target_phrases = _normalized_phrases(
        [target, *raw_variants],
        replacements=replacements,
        field_name=f"Requirement {requirement_id!r} target/variants",
    )
    allowed_contexts = _normalized_phrases(
        raw_allowed_contexts,
        replacements=replacements,
        field_name=f"Requirement {requirement_id!r} allowed_contexts",
    )

    all_source_hits: list[dict[str, Any]] = []
    for engine, payload in source_asr_payloads:
        hits, _ = _find_forbidden_hits(
            engine,
            payload,
            target_phrases=target_phrases,
            allowed_contexts=allowed_contexts,
            replacements=replacements,
        )
        all_source_hits.extend(hits)
    source_hits = _deduplicate_hits(
        all_source_hits,
        tolerance_seconds=float(requirement.get("source_hit_dedup_seconds", 0.5)),
    )

    all_final_hits: list[dict[str, Any]] = []
    all_allowed_context_hits: list[dict[str, Any]] = []
    final_units_by_engine: list[tuple[str, Sequence[TranscriptUnit]]] = []
    for engine, payload in final_asr_payloads:
        hits, units = _find_forbidden_hits(
            engine,
            payload,
            target_phrases=target_phrases,
            allowed_contexts=allowed_contexts,
            replacements=replacements,
        )
        all_final_hits.extend(hits)
        all_allowed_context_hits.extend(
            _find_context_hits(
                engine,
                payload,
                contexts=allowed_contexts,
                replacements=replacements,
            )
        )
        final_units_by_engine.append((engine, units))
    final_hits = _deduplicate_hits(
        all_final_hits,
        tolerance_seconds=float(requirement.get("final_hit_dedup_seconds", 0.5)),
    )
    allowed_context_hits = _deduplicate_hits(
        all_allowed_context_hits,
        tolerance_seconds=float(requirement.get("final_hit_dedup_seconds", 0.5)),
    )

    raw_occurrences = requirement.get("source_occurrences")
    if not isinstance(raw_occurrences, list) or not raw_occurrences:
        raise ValueError(f"Requirement {requirement_id!r} must enumerate source_occurrences")
    cut_tolerance = float(requirement.get("cut_tolerance_seconds", DEFAULT_CUT_TOLERANCE_SECONDS))
    source_asr_tolerance = float(
        requirement.get("source_asr_tolerance_seconds", DEFAULT_SOURCE_ASR_TOLERANCE_SECONDS)
    )
    covered_occurrences: list[dict[str, Any]] = []
    uncovered_occurrences: list[dict[str, Any]] = []
    retained_occurrences: list[dict[str, Any]] = []
    invalid_retained_occurrences: list[dict[str, Any]] = []
    source_asr_unverified_occurrences: list[dict[str, Any]] = []
    for occurrence_index, raw_occurrence in enumerate(raw_occurrences):
        if not isinstance(raw_occurrence, Mapping):
            raise ValueError(f"Requirement {requirement_id!r} source occurrence {occurrence_index} must be an object")
        occurrence = dict(raw_occurrence)
        occurrence["occurrence_index"] = occurrence_index
        action = str(occurrence.get("action", "remove")).lower()
        matching_cuts = [
            dict(cut)
            for cut in cuts
            if _cut_covers_occurrence(cut, occurrence, tolerance_seconds=cut_tolerance)
        ]
        overlapping_cuts = [
            dict(cut)
            for cut in cuts
            if _cut_overlaps_occurrence(cut, occurrence, tolerance_seconds=cut_tolerance)
        ]
        source_matches = _source_asr_matches_occurrence(
            occurrence,
            source_hits,
            tolerance_seconds=source_asr_tolerance,
        )
        occurrence["matching_source_asr_hits"] = source_matches
        occurrence["matching_cuts"] = matching_cuts
        occurrence["overlapping_cuts"] = overlapping_cuts
        if not source_matches:
            source_asr_unverified_occurrences.append(dict(occurrence))
        if action == "remove":
            (covered_occurrences if matching_cuts else uncovered_occurrences).append(occurrence)
        elif action == "retain":
            retained_occurrences.append(occurrence)
            if overlapping_cuts:
                invalid_retained_occurrences.append(occurrence)
        else:
            raise ValueError(
                f"Requirement {requirement_id!r} source occurrence {occurrence_index} has invalid action {action!r}"
            )

    undeclared_source_hits = [
        dict(hit)
        for hit in source_hits
        if not any(
            _source_asr_matches_occurrence(
                occurrence,
                [hit],
                tolerance_seconds=source_asr_tolerance,
            )
            for occurrence in raw_occurrences
            if isinstance(occurrence, Mapping)
        )
    ]

    raw_timestamps = requirement.get("user_reported_final_timestamps", [])
    if not isinstance(raw_timestamps, list):
        raise ValueError(f"Requirement {requirement_id!r} user_reported_final_timestamps must be a list")
    timestamp_window = float(requirement.get("timestamp_window_seconds", DEFAULT_TIMESTAMP_WINDOW_SECONDS))
    timestamp_checks = _timestamp_checks(
        final_units_by_engine,
        raw_timestamps,
        window_seconds=timestamp_window,
    )
    expected_final_occurrences = int(requirement.get("expected_final_occurrences", 0))
    minimum_allowed_context_occurrences = int(requirement.get("minimum_allowed_context_occurrences", 0))
    if minimum_allowed_context_occurrences < 0:
        raise ValueError("minimum_allowed_context_occurrences must be non-negative")
    failures: list[str] = []
    if uncovered_occurrences:
        failures.append("not_every_removed_source_occurrence_is_covered_by_an_actual_cut")
    if invalid_retained_occurrences:
        failures.append("a_retained_source_occurrence_overlaps_an_actual_cut")
    if source_asr_unverified_occurrences:
        failures.append("a_declared_source_occurrence_lacks_local_asr_evidence")
    if undeclared_source_hits:
        failures.append("source_asr_contains_an_undeclared_target_occurrence")
    if len(final_hits) != expected_final_occurrences:
        failures.append("final_asr_occurrence_count_does_not_match_requirement")
    if len(allowed_context_hits) < minimum_allowed_context_occurrences:
        failures.append("required_allowed_context_is_missing_from_final_asr")
    if any(not check["checked"] for check in timestamp_checks):
        failures.append("a_user_reported_final_timestamp_was_not_checked_by_every_asr_pass")

    return {
        "id": requirement_id,
        "target": target,
        "normalized_targets": target_phrases,
        "allowed_contexts": allowed_contexts,
        "status": "FAIL" if failures else "PASS",
        "evidence_level": "MODEL-INFERRED",
        "failures": failures,
        "source_asr_occurrences": source_hits,
        "covered_source_occurrences": covered_occurrences,
        "uncovered_source_occurrences": uncovered_occurrences,
        "retained_source_occurrences": retained_occurrences,
        "invalid_retained_source_occurrences": invalid_retained_occurrences,
        "source_asr_unverified_occurrences": source_asr_unverified_occurrences,
        "undeclared_source_asr_occurrences": undeclared_source_hits,
        "expected_final_occurrences": expected_final_occurrences,
        "forbidden_final_hits": final_hits,
        "minimum_allowed_context_occurrences": minimum_allowed_context_occurrences,
        "allowed_context_final_hits": allowed_context_hits,
        "timestamp_checks": timestamp_checks,
    }


def audit_release(
    *,
    requirements_payload: Mapping[str, Any],
    source_asr_payloads: Sequence[tuple[str, Mapping[str, Any]]],
    final_asr_payloads: Sequence[tuple[str, Mapping[str, Any]]],
    segment_report_payload: Mapping[str, Any],
) -> dict[str, Any]:
    source_document_revision = requirements_payload.get("source_document_revision")
    if source_document_revision is None or str(source_document_revision).strip() == "":
        raise ValueError("Requirements payload must identify source_document_revision")
    raw_requirements = requirements_payload.get("requirements")
    if not isinstance(raw_requirements, list) or not raw_requirements:
        raise ValueError("Requirements payload must contain a non-empty 'requirements' list")
    if not source_asr_payloads:
        raise ValueError("At least one source ASR payload is required")
    if not final_asr_payloads:
        raise ValueError("At least one final ASR payload is required")
    cuts = _extract_cuts(segment_report_payload)
    requirement_results = [
        _audit_requirement(
            requirements_payload,
            requirement,
            source_asr_payloads=source_asr_payloads,
            final_asr_payloads=final_asr_payloads,
            cuts=cuts,
        )
        for requirement in raw_requirements
        if isinstance(requirement, Mapping)
    ]
    if len(requirement_results) != len(raw_requirements):
        raise ValueError("Every requirement entry must be an object")
    status = "PASS" if all(result["status"] == "PASS" for result in requirement_results) else "FAIL"
    return {
        "status": status,
        "evidence_level": "MODEL-INFERRED",
        "source_document_revision": source_document_revision,
        "source_asr_engines": [name for name, _ in source_asr_payloads],
        "final_asr_engines": [name for name, _ in final_asr_payloads],
        "actual_cuts": cuts,
        "requirements": requirement_results,
        "release_blocked": status != "PASS",
        "limitations": [
            "ASR, cut-table, and timestamp evidence do not constitute direct auditory review.",
            "A PASS is invalid if the source document or final media changes after this audit.",
        ],
    }


def _load_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _named_payloads(paths: Sequence[str]) -> list[tuple[str, Mapping[str, Any]]]:
    payloads: list[tuple[str, Mapping[str, Any]]] = []
    for path_value in paths:
        path = Path(path_value).expanduser().resolve()
        payload = _load_json(path)
        engine = str(payload.get("engine", path.stem))
        payloads.append((engine, payload))
    return payloads


def _hash_media(paths: Sequence[str]) -> list[dict[str, Any]]:
    media: list[dict[str, Any]] = []
    for path_value in paths:
        path = Path(path_value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Final media not found: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        media.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": digest.hexdigest().upper(),
            }
        )
    return media


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _bind_final_asr_to_media(
    asr_paths: Sequence[str],
    final_media: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    final_hashes = {str(item["sha256"]).upper(): str(item["path"]) for item in final_media}
    bindings: list[dict[str, Any]] = []
    for path_value in asr_paths:
        asr_path = Path(path_value).expanduser().resolve()
        payload = _load_json(asr_path)
        declared_hash = payload.get("media_sha256")
        declared_media = payload.get("media")
        resolved_media_path: Path | None = None
        observed_hash: str | None = None
        failure: str | None = None
        if declared_hash is not None:
            observed_hash = str(declared_hash).strip().upper()
            if not observed_hash:
                failure = "empty_media_sha256"
        if declared_media is not None:
            resolved_media_path = Path(str(declared_media)).expanduser()
            if not resolved_media_path.is_absolute():
                resolved_media_path = (asr_path.parent / resolved_media_path).resolve()
            else:
                resolved_media_path = resolved_media_path.resolve()
            if not resolved_media_path.is_file():
                failure = "declared_media_file_not_found"
            else:
                media_file_hash = _sha256_file(resolved_media_path)
                if observed_hash is not None and observed_hash != media_file_hash:
                    failure = "declared_media_sha256_mismatch"
                else:
                    observed_hash = media_file_hash
        elif declared_hash is None:
            failure = "missing_media_or_media_sha256"
        matched_final_media = final_hashes.get(observed_hash or "")
        if failure is None and matched_final_media is None:
            failure = "asr_media_hash_does_not_match_any_final_media"
        bindings.append(
            {
                "asr": str(asr_path),
                "engine": str(payload.get("engine", asr_path.stem)),
                "declared_media": str(declared_media) if declared_media is not None else None,
                "resolved_media": str(resolved_media_path) if resolved_media_path is not None else None,
                "media_sha256": observed_hash,
                "matched_final_media": matched_final_media,
                "bound": failure is None,
                "failure": failure,
            }
        )
    return bindings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed release audit for repeated or forbidden spoken phrases."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="Audit final ASR and physical cuts against requirements.")
    run_parser.add_argument("--requirements", required=True, help="Requirement-to-cut mapping JSON.")
    run_parser.add_argument(
        "--source-asr",
        action="append",
        required=True,
        help="Source-timeline ASR JSON; repeat for independent passes.",
    )
    run_parser.add_argument(
        "--final-asr",
        action="append",
        required=True,
        help="Final-media ASR JSON; repeat for independent passes.",
    )
    run_parser.add_argument("--segment-report", required=True, help="Segment-removal report JSON.")
    run_parser.add_argument(
        "--final-media",
        action="append",
        required=True,
        help="Exact final media being audited; repeat for MP4/WAV/MP3.",
    )
    run_parser.add_argument(
        "--minimum-final-asr-engines",
        type=int,
        default=2,
        help="Minimum number of distinct final-ASR engine labels required (default: 2).",
    )
    run_parser.add_argument("--output", required=True, help="Machine-readable audit report JSON.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "run":
        raise ValueError(f"Unsupported command: {args.command}")
    requirements_path = Path(args.requirements).expanduser().resolve()
    segment_report_path = Path(args.segment_report).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    result = audit_release(
        requirements_payload=_load_json(requirements_path),
        source_asr_payloads=_named_payloads(args.source_asr),
        final_asr_payloads=_named_payloads(args.final_asr),
        segment_report_payload=_load_json(segment_report_path),
    )
    result["final_media"] = _hash_media(args.final_media)
    result["final_asr_media_bindings"] = _bind_final_asr_to_media(
        args.final_asr,
        result["final_media"],
    )
    if args.minimum_final_asr_engines < 1:
        raise ValueError("minimum-final-asr-engines must be at least 1")
    distinct_final_asr_engines = sorted(
        {
            str(binding["engine"]).strip().casefold()
            for binding in result["final_asr_media_bindings"]
            if str(binding["engine"]).strip()
        }
    )
    result["minimum_final_asr_engines"] = args.minimum_final_asr_engines
    result["distinct_final_asr_engines"] = distinct_final_asr_engines
    if any(not binding["bound"] for binding in result["final_asr_media_bindings"]):
        result["status"] = "FAIL"
        result["release_blocked"] = True
        result.setdefault("failures", []).append("a_final_asr_pass_is_not_bound_to_an_exact_final_media")
    if len(distinct_final_asr_engines) < args.minimum_final_asr_engines:
        result["status"] = "FAIL"
        result["release_blocked"] = True
        result.setdefault("failures", []).append("insufficient_independent_final_asr_engines")
    result["inputs"] = {
        "requirements": str(requirements_path),
        "source_asr": [str(Path(path).expanduser().resolve()) for path in args.source_asr],
        "final_asr": [str(Path(path).expanduser().resolve()) for path in args.final_asr],
        "segment_report": str(segment_report_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
