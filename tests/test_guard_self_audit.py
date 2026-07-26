"""Mutation-style self-audit for the preservation guard.

The guard is the only thing standing between a damaged render and delivery, and
its spectral checks run on a hand-rolled DFT. A bug there fails silently in the
dangerous direction: damage stops being detected and everything still looks like
a PASS. These tests inject known, individually-named damage into synthetic
speech and assert the matching failure code fires, so "the guard works" is a
fact the suite proves rather than a claim a reader has to trust.
"""

from __future__ import annotations

import math
import unittest
from array import array

from audio_sound.pipeline import (
    analyze_pcm16_samples,
    compare_audio_preservation,
)

# 16 kHz keeps the 2-4k / 4-6k / 6-8k guard bands usable (bands above Nyquist
# are skipped by the guard). The duration stays short on purpose: the spectral
# comparison uses a 2048-point hand-rolled DFT at O(n^2), so every extra second
# is a real cost multiplied across these tests.
SAMPLE_RATE = 16000
DURATION_SECONDS = 1.4
FRAME_MS = 20.0


def _speech_like_samples(
    *,
    sample_rate: int = SAMPLE_RATE,
    duration_seconds: float = DURATION_SECONDS,
) -> array:
    """Voiced segments with harmonics up to ~7 kHz, separated by quiet pauses.

    Real harmonic content matters: a pure tone or DC block would leave the
    high bands empty, and a low-pass would then be undetectable by design.
    """
    samples = array("h")
    total = int(sample_rate * duration_seconds)
    for index in range(total):
        time_s = index / sample_rate
        voiced = 0.15 <= time_s < 0.65 or 0.85 <= time_s < 1.30
        if voiced:
            envelope = 0.6 + 0.3 * math.sin(2 * math.pi * 4.0 * time_s) ** 2
            value = envelope * (
                0.50 * math.sin(2 * math.pi * 180 * time_s)
                + 0.26 * math.sin(2 * math.pi * 540 * time_s)
                + 0.16 * math.sin(2 * math.pi * 1500 * time_s)
                + 0.12 * math.sin(2 * math.pi * 3000 * time_s)
                + 0.09 * math.sin(2 * math.pi * 5000 * time_s)
                + 0.07 * math.sin(2 * math.pi * 7000 * time_s)
            )
            samples.append(int(max(-1.0, min(1.0, value)) * 11000))
        else:
            samples.append(int(120 * math.sin(2 * math.pi * 61 * time_s)))
    return samples


def _one_pole_lowpass(samples: array, *, cutoff_hz: float, sample_rate: int) -> array:
    """Cascaded one-pole low-pass: removes high-band energy the way a real
    over-aggressive denoise chain would."""
    dt = 1.0 / sample_rate
    rc = 1.0 / (2 * math.pi * cutoff_hz)
    alpha = dt / (rc + dt)
    result = array("h")
    stage_one = 0.0
    stage_two = 0.0
    for value in samples:
        stage_one += alpha * (value - stage_one)
        stage_two += alpha * (stage_one - stage_two)
        result.append(int(max(-32768, min(32767, round(stage_two)))))
    return result


def _high_shelf_boost(
    samples: array, *, cutoff_hz: float, gain_db: float, sample_rate: int
) -> array:
    """Boost content above ``cutoff_hz`` by adding back a scaled high-pass
    residual — the sibilance/harshness failure mode."""
    dt = 1.0 / sample_rate
    rc = 1.0 / (2 * math.pi * cutoff_hz)
    alpha = dt / (rc + dt)
    gain = 10 ** (gain_db / 20.0) - 1.0
    result = array("h")
    low = 0.0
    for value in samples:
        low += alpha * (value - low)
        high = value - low
        result.append(int(max(-32768, min(32767, round(value + gain * high)))))
    return result


def _apply_uniform_gain(samples: array, *, gain_db: float) -> array:
    gain = 10 ** (gain_db / 20.0)
    return array(
        "h",
        (int(max(-32768, min(32767, round(value * gain)))) for value in samples),
    )


def _decimate(samples: array, *, factor: int) -> array:
    return array("h", samples[::factor])


def _guard(
    reference: array,
    processed: array,
    *,
    sample_rate: int = SAMPLE_RATE,
    reference_format: dict[str, object] | None = None,
    processed_format: dict[str, object] | None = None,
    sample_level: bool = False,
) -> dict[str, object]:
    """Run the real analyzer, then the real guard.

    ``sample_level`` mirrors how the guard is actually called: PCM is handed over
    only when the sample-level checks (hard mute, spectral bands) are needed.
    Those checks run a 2048-point O(n^2) DFT, so tests that only exercise
    frame- or format-level failures deliberately leave it off.
    """
    reference_analysis = analyze_pcm16_samples(
        reference, sample_rate=sample_rate, frame_ms=FRAME_MS
    )
    processed_analysis = analyze_pcm16_samples(
        processed, sample_rate=sample_rate, frame_ms=FRAME_MS
    )
    return compare_audio_preservation(
        reference_analysis,
        processed_analysis,
        reference_format=reference_format,
        processed_format=processed_format,
        reference_samples=reference if sample_level else None,
        processed_samples=processed if sample_level else None,
        sample_rate=sample_rate if sample_level else None,
    )


class GuardBaselineTests(unittest.TestCase):
    """The guard must not cry wolf, or every real failure gets ignored."""

    def test_identical_audio_passes(self) -> None:
        """Covers the sample-level path too: a false positive here would make
        every other guard result meaningless."""
        samples = _speech_like_samples()

        guard = _guard(samples, array("h", samples), sample_level=True)

        self.assertEqual(guard["status"], "PASS")
        self.assertFalse(guard["release_blocked"])
        self.assertEqual(guard["failures"], [])

    def test_pure_loudness_change_is_not_treated_as_damage(self) -> None:
        samples = _speech_like_samples()

        guard = _guard(samples, _apply_uniform_gain(samples, gain_db=4.0))

        self.assertEqual(guard["status"], "PASS")
        self.assertEqual(guard["failures"], [])

    def test_guard_reports_directly_verified_evidence_level(self) -> None:
        samples = _speech_like_samples()

        guard = _guard(samples, array("h", samples))

        self.assertEqual(guard["evidence_level"], "DIRECTLY VERIFIED")


class InjectedDamageDetectionTests(unittest.TestCase):
    """One named defect per test. Each must block release."""

    def test_lowpass_is_detected_as_spectral_clarity_lost(self) -> None:
        reference = _speech_like_samples()
        processed = _one_pole_lowpass(
            reference, cutoff_hz=2200.0, sample_rate=SAMPLE_RATE
        )

        guard = _guard(reference, processed, sample_level=True)

        self.assertEqual(guard["status"], "FAIL")
        self.assertTrue(guard["release_blocked"])
        self.assertIn("spectral_clarity_lost", guard["failures"])

    def test_high_shelf_boost_is_detected_as_spectral_harshness(self) -> None:
        reference = _speech_like_samples()
        processed = _high_shelf_boost(
            reference, cutoff_hz=4000.0, gain_db=9.0, sample_rate=SAMPLE_RATE
        )

        guard = _guard(reference, processed, sample_level=True)

        self.assertEqual(guard["status"], "FAIL")
        self.assertTrue(guard["release_blocked"])
        self.assertIn("spectral_harshness_increased", guard["failures"])

    def test_hard_muted_speech_window_is_detected(self) -> None:
        reference = _speech_like_samples()
        processed = array("h", reference)
        start = int(0.30 * SAMPLE_RATE)
        end = int(0.50 * SAMPLE_RATE)
        for index in range(start, end):
            processed[index] = 0

        guard = _guard(reference, processed, sample_level=True)

        self.assertEqual(guard["status"], "FAIL")
        self.assertTrue(guard["release_blocked"])
        self.assertIn("source_active_hard_mute", guard["failures"])
        self.assertGreaterEqual(len(guard["hard_mute_windows"]), 1)

    def test_sample_rate_change_is_detected(self) -> None:
        reference = _speech_like_samples()

        guard = _guard(
            reference,
            array("h", reference),
            reference_format={"sample_rate": 44100, "channels": 1},
            processed_format={"sample_rate": 22050, "channels": 1},
        )

        self.assertEqual(guard["status"], "FAIL")
        self.assertIn("sample_rate_changed", guard["failures"])

    def test_channel_downmix_is_detected(self) -> None:
        reference = _speech_like_samples()

        guard = _guard(
            reference,
            array("h", reference),
            reference_format={"sample_rate": 44100, "channels": 2},
            processed_format={"sample_rate": 44100, "channels": 1},
        )

        self.assertEqual(guard["status"], "FAIL")
        self.assertIn("channel_layout_changed", guard["failures"])

    def test_truncated_delivery_is_detected_as_duration_change(self) -> None:
        reference = _speech_like_samples()
        processed = array("h", reference[: int(len(reference) * 0.75)])

        guard = _guard(reference, processed)

        self.assertEqual(guard["status"], "FAIL")
        self.assertIn("duration_changed", guard["failures"])

    def test_introduced_clipping_is_detected(self) -> None:
        reference = _speech_like_samples()
        processed = _apply_uniform_gain(reference, gain_db=14.0)

        guard = _guard(reference, processed)

        self.assertEqual(guard["status"], "FAIL")
        self.assertIn("clipping_increased", guard["failures"])

    def test_attenuated_speech_is_detected(self) -> None:
        reference = _speech_like_samples()
        processed = array("h", reference)
        start = int(0.90 * SAMPLE_RATE)
        end = int(1.28 * SAMPLE_RATE)
        for index in range(start, end):
            processed[index] = int(processed[index] * 0.15)

        guard = _guard(reference, processed)

        self.assertEqual(guard["status"], "FAIL")
        self.assertTrue(guard["release_blocked"])
        self.assertIn("active_speech_attenuated", guard["failures"])

    def test_unstable_short_term_gain_is_detected(self) -> None:
        reference = _speech_like_samples()
        processed = array("h", reference)
        for index in range(len(processed)):
            phase = math.sin(2 * math.pi * 1.5 * index / SAMPLE_RATE)
            gain = 10 ** ((7.0 * phase) / 20.0)
            processed[index] = int(
                max(-32768, min(32767, round(processed[index] * gain)))
            )

        guard = _guard(reference, processed)

        self.assertEqual(guard["status"], "FAIL")
        self.assertIn("short_term_gain_instability", guard["failures"])


class GuardBlindSpotBoundaryTests(unittest.TestCase):
    """Damage just under the documented thresholds must still pass, so the
    guard is provably threshold-driven and not accidentally blocking
    everything that is merely different."""

    def test_mild_high_shelf_within_budget_still_passes(self) -> None:
        reference = _speech_like_samples()
        processed = _high_shelf_boost(
            reference, cutoff_hz=4000.0, gain_db=1.0, sample_rate=SAMPLE_RATE
        )

        guard = _guard(reference, processed, sample_level=True)

        self.assertNotIn("spectral_harshness_increased", guard["failures"])

    def test_low_frequency_highpass_does_not_trip_clarity_loss(self) -> None:
        """A 55 Hz high-pass is part of the documented natural baseline. It must
        not register as clarity loss, because the guard bands start at 2 kHz."""
        reference = _speech_like_samples()
        dt = 1.0 / SAMPLE_RATE
        rc = 1.0 / (2 * math.pi * 55.0)
        alpha = rc / (rc + dt)
        processed = array("h")
        previous_input = 0
        previous_output = 0.0
        for value in reference:
            output = alpha * (previous_output + value - previous_input)
            previous_input = value
            previous_output = output
            processed.append(int(max(-32768, min(32767, round(output)))))

        guard = _guard(reference, processed, sample_level=True)

        self.assertNotIn("spectral_clarity_lost", guard["failures"])
        self.assertNotIn("spectral_harshness_increased", guard["failures"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
