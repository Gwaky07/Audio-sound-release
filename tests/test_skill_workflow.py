from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from array import array
from pathlib import Path

from audio_sound.pipeline import NoiseWindow
from audio_sound.skill_workflow import (
    BridgeCleanupConfig,
    ExactDuckWindow,
    HardMuteCleanupConfig,
    ResidueCleanupConfig,
    build_parser,
    build_workflow_dry_run,
    parse_exact_duck_window,
    parse_exact_mute_window,
    build_hardmute_cleanup_windows_from_silences,
    build_bridge_cleanup_windows_from_silences,
    build_residue_cleanup_windows_from_silences,
    describe_modes,
    _delivery_label,
    _build_final_delivery_guard,
    _finalize_workflow_file,
    _final_delivery_label,
    _prune_intermediate_audio,
    _reserve_delivery_paths,
    parse_focus_window,
    resolve_mode,
)


class SkillWorkflowTests(unittest.TestCase):
    def test_describe_modes_emits_utf8_under_legacy_parent_encoding(self) -> None:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "cp1252"
        code = (
            "from audio_sound.skill_workflow import main; "
            "raise SystemExit(main(['describe-modes','--mode','auto']))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            capture_output=True,
        )
        error_text = result.stderr.decode("utf-8", errors="replace")
        self.assertEqual(result.returncode, 0, error_text)
        payload = json.loads(result.stdout.decode("utf-8"))
        self.assertEqual(payload["suffix"], describe_modes("auto")["suffix"])

    def test_auto_dry_run_loads_required_candidates_and_capability_plan(self) -> None:
        payload = build_workflow_dry_run(Path("placeholder.wav"), "auto")

        self.assertTrue(payload["dry_run"])
        self.assertIn("natural_baseline", payload["candidate_ids"])
        self.assertIn("final_repair_best", payload["candidate_ids"])
        self.assertEqual(payload["capability_plan"]["primary_candidate"], "final_repair_best")

    def test_final_delivery_guard_blocks_local_speech_loss(self) -> None:
        import tempfile
        import wave
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            reference_wav = root / "reference.wav"
            delivered_wav = root / "delivered.wav"

            def write_wav(path: Path, samples: array) -> None:
                with wave.open(str(path), "wb") as writer:
                    writer.setnchannels(1)
                    writer.setsampwidth(2)
                    writer.setframerate(1000)
                    writer.writeframes(samples.tobytes())

            write_wav(reference_wav, array("h", [6000] * 2000))
            write_wav(
                delivered_wav,
                array("h", [6000] * 500 + [1000] * 500 + [10000] * 500 + [6000] * 500),
            )

            guard = _build_final_delivery_guard(reference_wav, delivered_wav)

            self.assertEqual(guard["status"], "FAIL")
            self.assertTrue(guard["release_blocked"])

    def test_final_delivery_guard_accepts_preserved_stereo_format(self) -> None:
        import tempfile
        import wave
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            reference_wav = root / "reference.wav"
            delivered_wav = root / "delivered.wav"
            stereo_samples = array("h", [6000, 3000] * 2000)

            for path in (reference_wav, delivered_wav):
                with wave.open(str(path), "wb") as writer:
                    writer.setnchannels(2)
                    writer.setsampwidth(2)
                    writer.setframerate(44100)
                    writer.writeframes(stereo_samples.tobytes())

            guard = _build_final_delivery_guard(reference_wav, delivered_wav)

            self.assertEqual(guard["status"], "PASS")
            self.assertEqual(guard["processed_format"]["channels"], 2)
            self.assertEqual(guard["processed_format"]["sample_rate"], 44100)

    def test_finalize_workflow_blocks_failed_preservation_guard_before_delivery(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "active_speech_attenuated"):
            _finalize_workflow_file(
                core_report={
                    "input_file": "D:/audio/voice.wav",
                    "quality_guard": {
                        "status": "FAIL",
                        "release_blocked": True,
                        "failures": ["active_speech_attenuated"],
                    },
                },
                mode=resolve_mode("natural"),
                ffmpeg_bin="ffmpeg",
                reference_target_lufs=-19.0,
                focus_windows=[],
                exact_mute_windows=[],
                exact_duck_windows=[],
                skip_spectrograms=True,
                skip_bridge_cleanup=True,
                delivery_dir=__import__("pathlib").Path("D:/delivery"),
                delivery_prefix="修音版",
                spectrogram_start=0.0,
                spectrogram_duration=30.0,
                run_slug="20260723-120000",
            )

    def test_describe_modes_returns_reference_style(self) -> None:
        payload = describe_modes("reference-style")
        self.assertEqual(payload["name"], "reference-style")
        self.assertEqual(payload["preset_name"], "fast")
        self.assertTrue(payload["apply_narrow_cleanup"])
        self.assertTrue(payload["apply_bridge_cleanup"])

    def test_describe_modes_returns_natural_default(self) -> None:
        payload = describe_modes("natural")
        self.assertEqual(payload["name"], "natural")
        self.assertEqual(payload["preset_name"], "natural")
        self.assertEqual(payload["attenuation_db"], 0.0)
        self.assertFalse(payload["apply_narrow_cleanup"])
        self.assertFalse(payload["apply_bridge_cleanup"])

    def test_describe_modes_returns_bounded_enhanced_final(self) -> None:
        payload = describe_modes("final")
        self.assertEqual(payload["preset_name"], "final")
        self.assertEqual(payload["attenuation_db"], 3.0)
        self.assertEqual(payload["suffix"], "增强修音终版")
        self.assertIn("clarity EQ", payload["description"])

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

    def test_describe_modes_returns_guarded_auto_mode(self) -> None:
        payload = describe_modes("auto")
        self.assertEqual(payload["preset_name"], "natural")
        self.assertEqual(payload["suffix"], "自动优选版")
        self.assertIn("candidate", payload["description"])

    def test_run_parser_defaults_to_auto(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run", "demo.wav"])
        self.assertEqual(args.mode, "auto")

    def test_auto_selection_falls_back_when_enhancement_blocked(self) -> None:
        from audio_sound.auto_workflow import build_evaluated_candidate, select_auto_candidate

        baseline = build_evaluated_candidate(
            candidate_id="natural_baseline",
            requires_asr=False,
            quality_guard={
                "release_blocked": False,
                "failures": [],
                "worst_relative_attenuation_db": -0.4,
                "relative_gain_spread_db": 1.5,
                "spectral_band_deltas_db": {},
            },
        )
        clarity = build_evaluated_candidate(
            candidate_id="clarity_leveling_safe",
            requires_asr=True,
            quality_guard={
                "release_blocked": False,
                "failures": [],
                "worst_relative_attenuation_db": -0.2,
                "relative_gain_spread_db": 1.0,
                "spectral_band_deltas_db": {},
            },
        )
        selected = select_auto_candidate([baseline, clarity], repair_intent=True)
        self.assertEqual(selected["candidate_id"], "natural_baseline")
        self.assertEqual(selected["selection_reason"], "safe_fallback_incomplete")
        self.assertTrue(selected["delivery_incomplete_for_repair_intent"])

    def test_parse_focus_window_accepts_label_start_duration(self) -> None:
        window = parse_focus_window("pause_a,12,3")
        self.assertEqual(window.label, "pause_a")
        self.assertEqual(window.start_seconds, 12.0)
        self.assertEqual(window.duration_seconds, 3.0)

    def test_delivery_label_preserves_chinese_name_and_adds_run_slug(self) -> None:
        label = _delivery_label("女生（测试2）", "细丝桥接清理版", "20260610-162806")
        self.assertEqual(label, "女生（测试2）_clean_细丝桥接清理版_20260610-162806")

    def test_final_delivery_label_uses_repair_prefix_and_original_name(self) -> None:
        label = _final_delivery_label("女生（测试2）")
        self.assertEqual(label, "修音版_女生（测试2）")

    def test_reserve_delivery_paths_increments_when_final_name_exists(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp_dir:
            delivery_dir = Path(tmp_dir)
            (delivery_dir / "修音版_女生（测试2）.wav").write_bytes(b"old wav")
            (delivery_dir / "修音版_女生（测试2）.mp3").write_bytes(b"old mp3")

            wav_path, mp3_path = _reserve_delivery_paths(delivery_dir, "女生（测试2）")

            self.assertEqual(wav_path.name, "修音版_女生（测试2）_01.wav")
            self.assertEqual(mp3_path.name, "修音版_女生（测试2）_01.mp3")

    def test_prune_intermediate_audio_removes_internal_audio_but_keeps_reports(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            preprocess_dir = root / "audio_preprocess"
            transcript_dir = root / "transcript_ready"
            preprocess_dir.mkdir()
            transcript_dir.mkdir()
            report_path = preprocess_dir / "audio_process_report.json"
            for path in (
                preprocess_dir / "raw.wav",
                preprocess_dir / "clean.wav",
                transcript_dir / "preview.mp3",
            ):
                path.write_bytes(b"audio")
            report_path.write_text("{}", encoding="utf-8")

            removed = _prune_intermediate_audio(
                [
                    {
                        "core_outputs": {
                            "preprocess_dir": str(preprocess_dir),
                            "transcript_dir": str(transcript_dir),
                        }
                    }
                ]
            )

            self.assertEqual(len(removed), 3)
            self.assertFalse((preprocess_dir / "raw.wav").exists())
            self.assertFalse((preprocess_dir / "clean.wav").exists())
            self.assertFalse((transcript_dir / "preview.mp3").exists())
            self.assertTrue(report_path.exists())

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

        windows, merge_debug, seed_debug = build_bridge_cleanup_windows_from_silences(
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
        self.assertFalse(any(item["protected"] for item in seed_debug))

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

        windows, _, _ = build_bridge_cleanup_windows_from_silences(
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

    def test_build_bridge_cleanup_windows_protects_longer_gap_near_possible_speech_onset(self) -> None:
        sample_rate = 100
        samples = array("h", [0] * 400)
        # 90ms bridge that is low level, but not quiet enough for the stricter protected-gap rule
        for index in range(100, 109):
            samples[index] = 2600
        silence_candidates = [
            {"start_seconds": 0.0, "end_seconds": 1.0, "duration_seconds": 1.0},
            {"start_seconds": 1.09, "end_seconds": 2.0, "duration_seconds": 0.91},
        ]

        windows, merge_debug, _ = build_bridge_cleanup_windows_from_silences(
            silence_candidates,
            samples=samples,
            sample_rate=sample_rate,
            config=BridgeCleanupConfig(
                tiny_gap_merge_seconds=0.05,
                bridge_gap_seconds=0.14,
                bridge_peak_db=-14.0,
                bridge_rms_db=-22.0,
                protected_gap_seconds=0.075,
                protected_bridge_peak_db=-22.0,
                protected_bridge_rms_db=-30.0,
                require_multi_seed_window=False,
                min_seed_silence_duration=0.03,
                short_trim_seconds=0.0,
                long_trim_seconds=0.0,
                trim_switch_seconds=0.2,
                min_window_seconds=0.05,
            ),
        )

        self.assertEqual(
            windows,
            [
                NoiseWindow(start_seconds=0.0, end_seconds=1.0),
                NoiseWindow(start_seconds=1.09, end_seconds=2.0),
            ],
        )
        self.assertFalse(merge_debug[0]["merged"])
        self.assertEqual(merge_debug[0]["bridge_peak_limit_db"], -22.0)

    def test_build_bridge_cleanup_windows_skips_single_seed_silence_by_default(self) -> None:
        sample_rate = 10
        samples = array("h", [0] * 30)
        silence_candidates = [
            {"start_seconds": 0.0, "end_seconds": 2.0, "duration_seconds": 2.0},
        ]

        windows, merge_debug, _ = build_bridge_cleanup_windows_from_silences(
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

        windows, _, _ = build_bridge_cleanup_windows_from_silences(
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

    def test_build_bridge_cleanup_windows_protects_short_seed_with_voiced_context(self) -> None:
        sample_rate = 100
        samples = array("h", [0] * 400)
        for index in range(88, 100):
            samples[index] = 4500
        for index in range(120, 132):
            samples[index] = 5200
        silence_candidates = [
            {"start_seconds": 0.0, "end_seconds": 1.1, "duration_seconds": 1.1},
            {"start_seconds": 1.0, "end_seconds": 1.2, "duration_seconds": 0.2},
            {"start_seconds": 1.32, "end_seconds": 2.0, "duration_seconds": 0.68},
        ]

        windows, merge_debug, seed_debug = build_bridge_cleanup_windows_from_silences(
            silence_candidates,
            samples=samples,
            sample_rate=sample_rate,
            config=BridgeCleanupConfig(
                require_multi_seed_window=False,
                min_seed_silence_duration=0.03,
                protected_silence_max_seconds=0.25,
                protected_context_probe_seconds=0.12,
                protected_context_peak_db=-18.0,
                protected_context_rms_db=-30.0,
                short_trim_seconds=0.0,
                long_trim_seconds=0.0,
                min_window_seconds=0.05,
            ),
        )

        self.assertEqual(
            windows,
            [
                NoiseWindow(start_seconds=0.0, end_seconds=1.1),
                NoiseWindow(start_seconds=1.32, end_seconds=2.0),
            ],
        )
        self.assertEqual(len(merge_debug), 1)
        self.assertTrue(seed_debug[1]["protected"])
        self.assertEqual(seed_debug[1]["protection_reason"], "continuous_speech_context")

    def test_build_hardmute_cleanup_windows_trims_silence_edges(self) -> None:
        windows, debug_items = build_hardmute_cleanup_windows_from_silences(
            [
                {"start_seconds": 0.0, "end_seconds": 1.0, "duration_seconds": 1.0},
                {"start_seconds": 2.0, "end_seconds": 2.04, "duration_seconds": 0.04},
            ],
            samples=array("h", [0] * 40),
            sample_rate=10,
            config=HardMuteCleanupConfig(
                trim_start_seconds=0.02,
                trim_end_seconds=0.02,
                min_window_seconds=0.05,
            ),
        )

        self.assertEqual(windows, [NoiseWindow(start_seconds=0.02, end_seconds=0.98)])
        self.assertFalse(any(item["protected"] for item in debug_items))

    def test_build_hardmute_cleanup_windows_protects_short_silence_between_two_voiced_islands(self) -> None:
        sample_rate = 100
        samples = array("h", [0] * 500)
        for index in range(100, 125):
            samples[index] = 4500
        for index in range(165, 195):
            samples[index] = 5200

        windows, debug_items = build_hardmute_cleanup_windows_from_silences(
            [
                {"start_seconds": 0.0, "end_seconds": 1.0, "duration_seconds": 1.0},
                {"start_seconds": 1.25, "end_seconds": 1.65, "duration_seconds": 0.4},
                {"start_seconds": 1.95, "end_seconds": 3.0, "duration_seconds": 1.05},
            ],
            samples=samples,
            sample_rate=sample_rate,
            config=HardMuteCleanupConfig(
                trim_start_seconds=0.02,
                trim_end_seconds=0.02,
                min_window_seconds=0.05,
                protected_silence_max_seconds=0.42,
                protected_neighbor_min_seconds=0.08,
                protected_neighbor_max_seconds=0.75,
                protected_neighbor_peak_db=-18.0,
                protected_neighbor_rms_db=-30.0,
            ),
        )

        self.assertEqual(
            windows,
            [
                NoiseWindow(start_seconds=0.02, end_seconds=0.98),
                NoiseWindow(start_seconds=1.97, end_seconds=2.98),
            ],
        )
        protected_item = debug_items[1]
        self.assertTrue(protected_item["protected"])
        self.assertEqual(protected_item["protection_reason"], "continuous_speech_neighbors")

    def test_build_hardmute_cleanup_windows_keeps_short_silence_when_neighbors_are_not_voiced_enough(self) -> None:
        sample_rate = 100
        samples = array("h", [0] * 500)
        for index in range(100, 125):
            samples[index] = 500
        for index in range(165, 195):
            samples[index] = 600

        windows, debug_items = build_hardmute_cleanup_windows_from_silences(
            [
                {"start_seconds": 0.0, "end_seconds": 1.0, "duration_seconds": 1.0},
                {"start_seconds": 1.25, "end_seconds": 1.65, "duration_seconds": 0.4},
                {"start_seconds": 1.95, "end_seconds": 3.0, "duration_seconds": 1.05},
            ],
            samples=samples,
            sample_rate=sample_rate,
            config=HardMuteCleanupConfig(
                trim_start_seconds=0.02,
                trim_end_seconds=0.02,
                min_window_seconds=0.05,
                protected_silence_max_seconds=0.42,
                protected_neighbor_min_seconds=0.08,
                protected_neighbor_max_seconds=0.75,
                protected_neighbor_peak_db=-18.0,
                protected_neighbor_rms_db=-30.0,
            ),
        )

        self.assertEqual(len(windows), 3)
        self.assertFalse(debug_items[1]["protected"])

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
