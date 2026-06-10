from __future__ import annotations

import unittest
from array import array

from audio_sound.pipeline import NoiseWindow
from audio_sound.skill_workflow import (
    BridgeCleanupConfig,
    ExactDuckWindow,
    HardMuteCleanupConfig,
    ResidueCleanupConfig,
    build_parser,
    parse_exact_duck_window,
    parse_exact_mute_window,
    build_hardmute_cleanup_windows_from_silences,
    build_bridge_cleanup_windows_from_silences,
    build_residue_cleanup_windows_from_silences,
    describe_modes,
    parse_focus_window,
    resolve_mode,
)


class SkillWorkflowTests(unittest.TestCase):
    def test_describe_modes_returns_reference_style(self) -> None:
        payload = describe_modes("reference-style")
        self.assertEqual(payload["name"], "reference-style")
        self.assertEqual(payload["preset_name"], "fast")
        self.assertTrue(payload["apply_narrow_cleanup"])
        self.assertTrue(payload["apply_bridge_cleanup"])

    def test_resolve_mode_returns_known_mode(self) -> None:
        mode = resolve_mode("voice-isolate")
        self.assertEqual(mode.preset_name, "voice-isolate")
        self.assertFalse(mode.apply_narrow_cleanup)
        self.assertFalse(mode.apply_bridge_cleanup)

    def test_describe_modes_returns_reference_legacy(self) -> None:
        payload = describe_modes("reference-legacy")
        self.assertEqual(payload["name"], "reference-legacy")
        self.assertEqual(payload["suffix"], "细丝桥接清理版")
        self.assertFalse(payload["apply_narrow_cleanup"])
        self.assertFalse(payload["apply_bridge_cleanup"])

    def test_run_parser_defaults_to_reference_legacy(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run", "demo.wav"])
        self.assertEqual(args.mode, "reference-legacy")

    def test_parse_focus_window_accepts_label_start_duration(self) -> None:
        window = parse_focus_window("pause_a,12,3")
        self.assertEqual(window.label, "pause_a")
        self.assertEqual(window.start_seconds, 12.0)
        self.assertEqual(window.duration_seconds, 3.0)

    def test_parse_exact_mute_window_accepts_start_end(self) -> None:
        window = parse_exact_mute_window("234.452,235.457")
        self.assertEqual(window, NoiseWindow(start_seconds=234.452, end_seconds=235.457))

    def test_parse_exact_duck_window_accepts_start_end_gain_fade(self) -> None:
        window = parse_exact_duck_window("528.648,530.824,0,6")
        self.assertEqual(
            window,
            ExactDuckWindow(
                start_seconds=528.648,
                end_seconds=530.824,
                floor_gain=0.0,
                fade_ms=6,
            ),
        )

    def test_build_bridge_cleanup_windows_merges_tiny_gap(self) -> None:
        sample_rate = 10
        samples = array("h", [0] * 40)
        silence_candidates = [
            {"start_seconds": 0.0, "end_seconds": 1.0, "duration_seconds": 1.0},
            {"start_seconds": 1.04, "end_seconds": 2.0, "duration_seconds": 0.96},
        ]

        windows, merge_debug = build_bridge_cleanup_windows_from_silences(
            silence_candidates,
            samples=samples,
            sample_rate=sample_rate,
            config=BridgeCleanupConfig(
                tiny_gap_merge_seconds=0.05,
                bridge_gap_seconds=0.12,
                min_seed_silence_duration=0.07,
                short_trim_seconds=0.01,
                long_trim_seconds=0.01,
                trim_switch_seconds=0.2,
                min_window_seconds=0.05,
            ),
        )

        self.assertEqual(len(windows), 1)
        self.assertLessEqual(windows[0].start_seconds, 0.02)
        self.assertGreaterEqual(windows[0].end_seconds, 1.98)
        self.assertTrue(any(item["merged"] for item in merge_debug))

    def test_build_bridge_cleanup_windows_merges_low_level_bridge(self) -> None:
        sample_rate = 10
        samples = array("h", [0] * 50)
        # tiny bridge between two silent sections
        samples[10] = 20
        samples[11] = -20
        silence_candidates = [
            {"start_seconds": 0.0, "end_seconds": 1.0, "duration_seconds": 1.0},
            {"start_seconds": 1.1, "end_seconds": 2.0, "duration_seconds": 0.9},
        ]

        windows, _ = build_bridge_cleanup_windows_from_silences(
            silence_candidates,
            samples=samples,
            sample_rate=sample_rate,
            config=BridgeCleanupConfig(
                tiny_gap_merge_seconds=0.05,
                bridge_gap_seconds=0.2,
                bridge_peak_db=-18.0,
                bridge_rms_db=-20.0,
                min_seed_silence_duration=0.07,
                short_trim_seconds=0.0,
                long_trim_seconds=0.0,
                trim_switch_seconds=0.2,
                min_window_seconds=0.05,
            ),
        )

        self.assertEqual(windows, [NoiseWindow(start_seconds=0.0, end_seconds=2.0)])

    def test_build_bridge_cleanup_windows_skips_single_seed_silence_by_default(self) -> None:
        sample_rate = 10
        samples = array("h", [0] * 30)
        silence_candidates = [
            {"start_seconds": 0.0, "end_seconds": 2.0, "duration_seconds": 2.0},
        ]

        windows, merge_debug = build_bridge_cleanup_windows_from_silences(
            silence_candidates,
            samples=samples,
            sample_rate=sample_rate,
            config=BridgeCleanupConfig(
                min_seed_silence_duration=0.07,
                short_trim_seconds=0.0,
                long_trim_seconds=0.0,
                min_window_seconds=0.05,
            ),
        )

        self.assertEqual(windows, [])
        self.assertEqual(merge_debug, [])

    def test_build_bridge_cleanup_windows_can_keep_single_seed_when_explicitly_enabled(self) -> None:
        sample_rate = 10
        samples = array("h", [0] * 30)
        silence_candidates = [
            {"start_seconds": 0.0, "end_seconds": 2.0, "duration_seconds": 2.0},
        ]

        windows, _ = build_bridge_cleanup_windows_from_silences(
            silence_candidates,
            samples=samples,
            sample_rate=sample_rate,
            config=BridgeCleanupConfig(
                require_multi_seed_window=False,
                min_seed_silence_duration=0.07,
                short_trim_seconds=0.0,
                long_trim_seconds=0.0,
                min_window_seconds=0.05,
            ),
        )

        self.assertEqual(windows, [NoiseWindow(start_seconds=0.0, end_seconds=2.0)])

    def test_build_hardmute_cleanup_windows_trims_silence_edges(self) -> None:
        windows = build_hardmute_cleanup_windows_from_silences(
            [
                {"start_seconds": 0.0, "end_seconds": 1.0, "duration_seconds": 1.0},
                {"start_seconds": 2.0, "end_seconds": 2.04, "duration_seconds": 0.04},
            ],
            config=HardMuteCleanupConfig(
                trim_start_seconds=0.02,
                trim_end_seconds=0.02,
                min_window_seconds=0.05,
            ),
        )

        self.assertEqual(windows, [NoiseWindow(start_seconds=0.02, end_seconds=0.98)])

    def test_build_residue_cleanup_windows_accepts_short_low_level_gap_after_long_silence(self) -> None:
        sample_rate = 10
        samples = array("h", [0] * 60)
        samples[20] = 200
        samples[21] = -200
        silence_candidates = [
            {"start_seconds": 0.0, "end_seconds": 2.0, "duration_seconds": 2.0},
            {"start_seconds": 2.4, "end_seconds": 4.0, "duration_seconds": 1.6},
        ]

        windows, debug_items = build_residue_cleanup_windows_from_silences(
            silence_candidates,
            samples=samples,
            sample_rate=sample_rate,
            config=ResidueCleanupConfig(
                min_preceding_silence_seconds=0.5,
                max_residue_duration_seconds=0.6,
                residue_peak_db=-18.0,
                residue_rms_db=-28.0,
                trim_start_seconds=0.0,
                trim_end_seconds=0.0,
                min_window_seconds=0.05,
            ),
        )

        self.assertEqual(windows, [NoiseWindow(start_seconds=2.0, end_seconds=2.4)])
        self.assertTrue(debug_items[0]["accepted"])

    def test_build_residue_cleanup_windows_rejects_loud_gap(self) -> None:
        sample_rate = 10
        samples = array("h", [0] * 60)
        samples[20] = 15000
        samples[21] = -15000
        silence_candidates = [
            {"start_seconds": 0.0, "end_seconds": 2.0, "duration_seconds": 2.0},
            {"start_seconds": 2.4, "end_seconds": 4.0, "duration_seconds": 1.6},
        ]

        windows, debug_items = build_residue_cleanup_windows_from_silences(
            silence_candidates,
            samples=samples,
            sample_rate=sample_rate,
            config=ResidueCleanupConfig(
                min_preceding_silence_seconds=0.5,
                max_residue_duration_seconds=0.6,
                residue_peak_db=-18.0,
                residue_rms_db=-28.0,
                trim_start_seconds=0.0,
                trim_end_seconds=0.0,
                min_window_seconds=0.05,
            ),
        )

        self.assertEqual(windows, [])
        self.assertFalse(debug_items[0]["accepted"])


if __name__ == "__main__":
    unittest.main()
