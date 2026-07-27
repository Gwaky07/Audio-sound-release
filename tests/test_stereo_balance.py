from __future__ import annotations

import math
import unittest
from array import array

from audio_sound.stereo_balance import (
    analyze_stereo_balance,
    apply_stereo_balance,
    plan_stereo_balance,
    run_stereo_balance,
)


def _tone(freq: float, *, sample_rate: int, duration: float, amplitude: float) -> array:
    samples = array("h")
    total = int(sample_rate * duration)
    for index in range(total):
        value = amplitude * math.sin(2 * math.pi * freq * index / sample_rate)
        samples.append(int(max(-32768, min(32767, round(value)))))
    return samples


def _stereo(left: array, right: array) -> array:
    interleaved = array("h")
    for left_value, right_value in zip(left, right):
        interleaved.append(left_value)
        interleaved.append(right_value)
    return interleaved


class StereoBalanceTests(unittest.TestCase):
    def test_mono_is_not_applicable(self) -> None:
        mono = _tone(220, sample_rate=8000, duration=0.4, amplitude=8000)
        analysis = analyze_stereo_balance(mono, sample_rate=8000, channels=1)
        plan = plan_stereo_balance(analysis)
        self.assertFalse(analysis["applicable"])
        self.assertFalse(plan["needed"])

    def test_balanced_stereo_is_skipped(self) -> None:
        voice = _tone(180, sample_rate=8000, duration=0.5, amplitude=9000)
        samples = _stereo(voice, array("h", voice))
        analysis = analyze_stereo_balance(samples, sample_rate=8000, channels=2)
        plan = plan_stereo_balance(analysis)
        self.assertTrue(analysis["applicable"])
        self.assertLess(abs(float(analysis["imbalance_db"])), 0.5)
        self.assertFalse(plan["needed"])

    def test_correlated_level_mismatch_can_raise_quiet_channel(self) -> None:
        left = _tone(180, sample_rate=8000, duration=0.6, amplitude=10000)
        right = _tone(180, sample_rate=8000, duration=0.6, amplitude=2500)
        samples = _stereo(left, right)
        balanced, report = run_stereo_balance(
            samples,
            sample_rate=8000,
            channels=2,
            policy={
                "mid_blend_when_decorrelated": 0.0,
                "correlation_trigger": 0.0,
                "mismatch_mode": "channel_gains",
                "decorrelated_mode": "channel_gains",
            },
        )
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["plan"]["applied"])
        self.assertEqual(report["plan"]["output_mode"], "channel_gains")
        self.assertGreater(float(report["plan"]["right_gain_db"]), 6.0)
        after = report["after"]
        self.assertIsNotNone(after)
        assert after is not None
        self.assertLess(abs(float(after["imbalance_db"])), 1.5)
        self.assertEqual(len(balanced), len(samples))

    def test_default_level_mismatch_uses_identical_dual_mono_channels(self) -> None:
        left = _tone(180, sample_rate=8000, duration=0.6, amplitude=10000)
        right = _tone(180, sample_rate=8000, duration=0.6, amplitude=2500)
        samples = _stereo(left, right)

        balanced, report = run_stereo_balance(
            samples,
            sample_rate=8000,
            channels=2,
        )

        self.assertEqual(report["plan"]["output_mode"], "dual_mono_louder")
        self.assertEqual(report["plan"]["source_channel"], "left")
        self.assertEqual(list(balanced[0::2]), list(balanced[1::2]))
        self.assertEqual(len(balanced), len(samples))

    def test_decorrelated_uses_dual_mono_louder(self) -> None:
        left = _tone(180, sample_rate=8000, duration=0.5, amplitude=11000)
        right = _tone(420, sample_rate=8000, duration=0.5, amplitude=3000)
        samples = _stereo(left, right)
        balanced, report = run_stereo_balance(
            samples,
            sample_rate=8000,
            channels=2,
        )
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["plan"]["output_mode"], "dual_mono_louder")
        self.assertEqual(report["plan"]["source_channel"], "left")
        after = report["after"]
        self.assertIsNotNone(after)
        assert after is not None
        self.assertLess(abs(float(after["imbalance_db"])), 0.25)
        self.assertGreater(float(after["correlation"] or 0.0), 0.99)
        # Both ears equal the louder (left) channel.
        self.assertEqual(list(balanced[0::2]), list(left))
        self.assertEqual(list(balanced[1::2]), list(left))

    def test_preserve_two_channels_after_mid_blend(self) -> None:
        left = _tone(180, sample_rate=8000, duration=0.5, amplitude=11000)
        right = _tone(420, sample_rate=8000, duration=0.5, amplitude=3000)
        samples = _stereo(left, right)
        balanced = apply_stereo_balance(
            samples,
            channels=2,
            left_gain_db=0.0,
            right_gain_db=8.0,
            mid_blend=0.4,
            output_mode="channel_gains",
        )
        self.assertEqual(len(balanced) % 2, 0)
        self.assertEqual(len(balanced), len(samples))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
