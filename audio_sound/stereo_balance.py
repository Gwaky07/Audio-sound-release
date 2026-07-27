"""Stereo channel balance for spoken-word sources.

Preserves the source channel count (stereo stays stereo). When left/right
active levels or correlation show a headphone-hostile mismatch, apply a
bounded repair:

- Correlated level mismatch: raise the quieter channel toward the louder one.
- Decorrelated / dual-mic mismatch: duplicate the louder channel to both ears
  (dual-mono layout). Boosting a quiet, dissimilar mic into the downmix tends
  to raise harsh high bands and fails spectral preservation guards.
"""

from __future__ import annotations

import math
import wave
from array import array
from pathlib import Path
from typing import Any


DEFAULT_STEREO_BALANCE_POLICY: dict[str, Any] = {
    "enabled": True,
    "min_imbalance_db": 1.5,
    "max_channel_gain_db": 12.0,
    "target": "louder",
    "correlation_trigger": 0.85,
    "mismatch_mode": "dual_mono_louder",
    "decorrelated_mode": "dual_mono_louder",
    "mid_blend_when_decorrelated": 0.0,
    "active_rms_threshold": 500.0,
    "frame_ms": 20.0,
}


def analyze_stereo_balance(
    samples: array,
    *,
    sample_rate: int,
    channels: int,
    frame_ms: float = 20.0,
    active_rms_threshold: float = 500.0,
) -> dict[str, Any]:
    if channels <= 1:
        return {
            "channels": channels,
            "applicable": False,
            "left_rms_dbfs": None,
            "right_rms_dbfs": None,
            "imbalance_db": 0.0,
            "correlation": None,
            "side_to_mid_ratio": None,
            "active_frame_count": 0,
        }
    if channels != 2:
        return {
            "channels": channels,
            "applicable": False,
            "left_rms_dbfs": None,
            "right_rms_dbfs": None,
            "imbalance_db": 0.0,
            "correlation": None,
            "side_to_mid_ratio": None,
            "active_frame_count": 0,
            "reason": "only_stereo_pair_supported",
        }

    left = samples[0::2]
    right = samples[1::2]
    frame = max(1, int(sample_rate * (frame_ms / 1000.0)))
    left_active: list[float] = []
    right_active: list[float] = []
    deltas: list[float] = []
    for index in range(0, min(len(left), len(right)) - frame + 1, frame):
        left_chunk = left[index : index + frame]
        right_chunk = right[index : index + frame]
        left_rms = math.sqrt(
            sum(float(value) * value for value in left_chunk) / frame
        )
        right_rms = math.sqrt(
            sum(float(value) * value for value in right_chunk) / frame
        )
        if max(left_rms, right_rms) < active_rms_threshold:
            continue
        left_active.extend(float(value) for value in left_chunk)
        right_active.extend(float(value) for value in right_chunk)
        deltas.append(20.0 * math.log10((left_rms + 1e-12) / (right_rms + 1e-12)))

    def _rms_db(values: list[float]) -> float | None:
        if not values:
            return None
        mean_square = sum(value * value for value in values) / len(values)
        return 20.0 * math.log10(math.sqrt(mean_square) / 32768.0 + 1e-12)

    left_rms_db = _rms_db(left_active)
    right_rms_db = _rms_db(right_active)
    imbalance_db = 0.0
    if left_rms_db is not None and right_rms_db is not None:
        imbalance_db = left_rms_db - right_rms_db

    correlation = None
    side_to_mid = None
    if left_active and right_active:
        mean_left = sum(left_active) / len(left_active)
        mean_right = sum(right_active) / len(right_active)
        numerator = sum(
            (left_value - mean_left) * (right_value - mean_right)
            for left_value, right_value in zip(left_active, right_active)
        )
        denom_left = math.sqrt(
            sum((value - mean_left) ** 2 for value in left_active)
        )
        denom_right = math.sqrt(
            sum((value - mean_right) ** 2 for value in right_active)
        )
        correlation = numerator / (denom_left * denom_right + 1e-12)
        mid_energy = 0.0
        side_energy = 0.0
        for left_value, right_value in zip(left_active, right_active):
            mid = 0.5 * (left_value + right_value)
            side = 0.5 * (left_value - right_value)
            mid_energy += mid * mid
            side_energy += side * side
        side_to_mid = side_energy / (mid_energy + 1e-12)

    deltas_sorted = sorted(deltas)

    def _pct(fraction: float) -> float | None:
        if not deltas_sorted:
            return None
        return deltas_sorted[int((len(deltas_sorted) - 1) * fraction)]

    return {
        "channels": 2,
        "applicable": True,
        "left_rms_dbfs": None if left_rms_db is None else round(left_rms_db, 3),
        "right_rms_dbfs": None if right_rms_db is None else round(right_rms_db, 3),
        "imbalance_db": round(imbalance_db, 3),
        "correlation": None if correlation is None else round(correlation, 4),
        "side_to_mid_ratio": None if side_to_mid is None else round(side_to_mid, 4),
        "active_frame_count": len(deltas),
        "active_imbalance_p10_db": None if _pct(0.1) is None else round(_pct(0.1), 3),
        "active_imbalance_p50_db": None if _pct(0.5) is None else round(_pct(0.5), 3),
        "active_imbalance_p90_db": None if _pct(0.9) is None else round(_pct(0.9), 3),
    }


def plan_stereo_balance(
    analysis: dict[str, Any],
    *,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    merged = {**DEFAULT_STEREO_BALANCE_POLICY, **(policy or {})}
    if not merged.get("enabled", True):
        return {
            "needed": False,
            "applied": False,
            "reason": "disabled",
            "output_mode": "none",
            "left_gain_db": 0.0,
            "right_gain_db": 0.0,
            "mid_blend": 0.0,
            "source_channel": None,
        }
    if not analysis.get("applicable"):
        return {
            "needed": False,
            "applied": False,
            "reason": analysis.get("reason") or "not_stereo",
            "output_mode": "none",
            "left_gain_db": 0.0,
            "right_gain_db": 0.0,
            "mid_blend": 0.0,
            "source_channel": None,
        }

    imbalance_db = float(analysis.get("imbalance_db") or 0.0)
    correlation = analysis.get("correlation")
    min_imbalance = float(merged["min_imbalance_db"])
    max_gain = float(merged["max_channel_gain_db"])
    correlation_trigger = float(merged["correlation_trigger"])
    decorrelated = (
        correlation is not None and float(correlation) < correlation_trigger
    )
    level_mismatch = abs(imbalance_db) >= min_imbalance
    if not level_mismatch and not decorrelated:
        return {
            "needed": False,
            "applied": False,
            "reason": "within_balance_budget",
            "output_mode": "none",
            "left_gain_db": 0.0,
            "right_gain_db": 0.0,
            "mid_blend": 0.0,
            "source_channel": None,
        }

    mismatch_mode = str(
        merged.get("mismatch_mode")
        or merged.get("decorrelated_mode")
        or "dual_mono_louder"
    )
    if (level_mismatch or decorrelated) and mismatch_mode == "dual_mono_louder":
        source_channel = "left" if imbalance_db >= 0.0 else "right"
        return {
            "needed": True,
            "applied": False,
            "reason": "stereo_channel_mismatch",
            "output_mode": "dual_mono_louder",
            "left_gain_db": 0.0,
            "right_gain_db": 0.0,
            "mid_blend": 0.0,
            "source_channel": source_channel,
            "level_mismatch": level_mismatch,
            "decorrelated": True,
            "preserve_channels": True,
        }

    left_gain_db = 0.0
    right_gain_db = 0.0
    if level_mismatch:
        target = str(merged.get("target") or "louder")
        if target == "mid":
            left_gain_db = -0.5 * imbalance_db
            right_gain_db = 0.5 * imbalance_db
        elif imbalance_db > 0:
            right_gain_db = min(max_gain, imbalance_db)
        else:
            left_gain_db = min(max_gain, -imbalance_db)
        left_gain_db = max(-max_gain, min(max_gain, left_gain_db))
        right_gain_db = max(-max_gain, min(max_gain, right_gain_db))

    mid_blend = 0.0
    if decorrelated:
        mid_blend = float(merged.get("mid_blend_when_decorrelated") or 0.0)
        mid_blend = max(0.0, min(1.0, mid_blend))

    return {
        "needed": True,
        "applied": False,
        "reason": "stereo_channel_mismatch",
        "output_mode": "channel_gains",
        "left_gain_db": round(left_gain_db, 3),
        "right_gain_db": round(right_gain_db, 3),
        "mid_blend": round(mid_blend, 3),
        "source_channel": None,
        "level_mismatch": level_mismatch,
        "decorrelated": decorrelated,
        "preserve_channels": True,
    }


def apply_stereo_balance(
    samples: array,
    *,
    channels: int,
    left_gain_db: float = 0.0,
    right_gain_db: float = 0.0,
    mid_blend: float = 0.0,
    output_mode: str = "channel_gains",
    source_channel: str | None = None,
) -> array:
    if channels != 2:
        return array("h", samples)

    if output_mode == "dual_mono_louder":
        channel = (source_channel or "left").lower()
        balanced = array("h")
        for index in range(0, len(samples) - 1, 2):
            value = samples[index] if channel == "left" else samples[index + 1]
            balanced.append(int(value))
            balanced.append(int(value))
        return balanced

    left_gain = 10 ** (float(left_gain_db) / 20.0)
    right_gain = 10 ** (float(right_gain_db) / 20.0)
    blend = max(0.0, min(1.0, float(mid_blend)))
    self_weight = 1.0 - (blend / 2.0)
    cross_weight = blend / 2.0
    balanced = array("h")
    for index in range(0, len(samples) - 1, 2):
        left = float(samples[index]) * left_gain
        right = float(samples[index + 1]) * right_gain
        out_left = self_weight * left + cross_weight * right
        out_right = cross_weight * left + self_weight * right
        balanced.append(int(max(-32768, min(32767, round(out_left)))))
        balanced.append(int(max(-32768, min(32767, round(out_right)))))
    return balanced


def write_pcm16_wave(
    path: Path,
    samples: array,
    *,
    sample_rate: int,
    channels: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(samples.tobytes())


def run_stereo_balance(
    samples: array,
    *,
    sample_rate: int,
    channels: int,
    policy: dict[str, Any] | None = None,
) -> tuple[array, dict[str, Any]]:
    merged = {**DEFAULT_STEREO_BALANCE_POLICY, **(policy or {})}
    before = analyze_stereo_balance(
        samples,
        sample_rate=sample_rate,
        channels=channels,
        frame_ms=float(merged.get("frame_ms", 20.0)),
        active_rms_threshold=float(merged.get("active_rms_threshold", 500.0)),
    )
    plan = plan_stereo_balance(before, policy=merged)
    report: dict[str, Any] = {
        "status": "NOT_APPLICABLE",
        "before": before,
        "plan": plan,
        "after": None,
        "preserve_channels": True,
    }
    if not plan.get("needed"):
        report["status"] = "SKIP"
        return array("h", samples), report

    balanced = apply_stereo_balance(
        samples,
        channels=channels,
        left_gain_db=float(plan["left_gain_db"]),
        right_gain_db=float(plan["right_gain_db"]),
        mid_blend=float(plan["mid_blend"]),
        output_mode=str(plan.get("output_mode") or "channel_gains"),
        source_channel=plan.get("source_channel"),
    )
    after = analyze_stereo_balance(
        balanced,
        sample_rate=sample_rate,
        channels=channels,
        frame_ms=float(merged.get("frame_ms", 20.0)),
        active_rms_threshold=float(merged.get("active_rms_threshold", 500.0)),
    )
    plan["applied"] = True
    report["status"] = "PASS"
    report["plan"] = plan
    report["after"] = after
    return balanced, report
