from __future__ import annotations

import math
import random
import tempfile
import unittest
import wave
from array import array
from hashlib import sha1
from pathlib import Path
from unittest import mock

from audio_sound.config import apply_runtime_overrides, load_preset
from audio_sound.pipeline import (
    NoiseWindow,
    RuntimeOptions,
    analyze_pcm16_samples,
    apply_input_safety_overrides,
    apply_adaptive_breath_cleanup,
    apply_spectramini_style_cleanup_to_samples,
    attenuation_db_to_gain,
    build_breath_processing_plan,
    build_mastering_filter_chain,
    build_batch_summary,
    build_deepfilternet_command,
    build_ffmpeg_extract_command,
    build_ffmpeg_finalize_commands,
    build_ffmpeg_noise_sample_command,
    build_output_layout,
    build_output_root,
    build_respiro_detect_command,
    compare_audio_preservation,
    build_effective_breath_windows,
    filter_noise_like_breath_windows,
    infer_pause_residual_cleanup_windows,
    match_residual_breath_windows,
    measure_segment_levels,
    repair_deepfilternet_speech_dropouts,
    resolve_respiro_runtime,
    run_respiro_or_fallback_detection,
    duck_samples_for_windows,
    discover_media_files,
    infer_breath_window_from_frame_rms,
    infer_breath_onset_windows,
    parse_noise_window,
    process_media_file,
    render_batch_summary_markdown,
    resolve_processing_format,
    select_adaptive_profile,
    trim_noise_windows_for_assessment,
)


class PipelineTests(unittest.TestCase):
    def test_effective_breath_windows_merge_respiro_and_auxiliary_without_crossing_speech(self) -> None:
        analysis = {
            "_frame_rms_db": [-70.0] * 20 + [-18.0] * 10 + [-70.0] * 20,
            "frame_ms": 10.0,
            "speech_level_dbfs": -18.0,
        }

        windows, evidence = build_effective_breath_windows(
            respiro_windows=[NoiseWindow(0.05, 0.14)],
            auxiliary_windows=[
                NoiseWindow(0.10, 0.19),
                NoiseWindow(0.18, 0.32),
                NoiseWindow(0.35, 0.43),
            ],
            source_analysis=analysis,
        )

        self.assertEqual(windows, [NoiseWindow(0.05, 0.19), NoiseWindow(0.35, 0.43)])
        self.assertEqual(evidence[0]["sources"], ["respiro", "auxiliary"])
        self.assertEqual(evidence[1]["sources"], ["auxiliary"])
        self.assertEqual(evidence[1]["decision"], "rejected_active_speech")
        self.assertEqual(evidence[2]["decision"], "accepted")

    def test_effective_cleanup_rejects_quiet_source_speech_at_hard_mute_threshold(self) -> None:
        analysis = {
            "_frame_rms_db": [-70.0] * 4 + [-32.0] * 4 + [-70.0] * 4,
            "frame_ms": 10.0,
            "speech_level_dbfs": -18.0,
        }

        windows, evidence = build_effective_breath_windows(
            respiro_windows=[],
            auxiliary_windows=[NoiseWindow(0.04, 0.08)],
            source_analysis=analysis,
        )

        self.assertEqual(windows, [])
        self.assertEqual(evidence[0]["decision"], "rejected_active_speech")

    def test_effective_cleanup_trims_leading_and_trailing_active_speech(self) -> None:
        analysis = {
            "_frame_rms_db": [-18.0] * 5 + [-70.0] * 20 + [-18.0] * 5,
            "frame_ms": 10.0,
            "speech_level_dbfs": -18.0,
        }

        windows, evidence = build_effective_breath_windows(
            respiro_windows=[],
            auxiliary_windows=[NoiseWindow(0.0, 0.30)],
            source_analysis=analysis,
        )

        self.assertEqual(windows, [NoiseWindow(0.05, 0.25)])
        self.assertEqual(evidence[0]["decision"], "accepted")

    def test_adaptive_breath_cleanup_targets_context_floor_with_cap(self) -> None:
        samples = array("h", [100] * 100 + [4000] * 100 + [6000] * 100)

        cleaned, details = apply_adaptive_breath_cleanup(
            samples,
            windows=[NoiseWindow(0.1, 0.2)],
            sample_rate=1000,
            channels=1,
            max_attenuation_db=12.0,
            target_margin_db=3.0,
            context_ms=80.0,
            fade_ms=0.0,
        )

        self.assertAlmostEqual(details[0]["requested_attenuation_db"], 12.0)
        self.assertLess(max(cleaned[100:200]), 1100)
        self.assertEqual(cleaned[:100], samples[:100])
        self.assertEqual(cleaned[200:], samples[200:])

    def test_adaptive_cleanup_can_target_measured_global_noise_floor(self) -> None:
        samples = array("h", [2000] * 100)

        cleaned, details = apply_adaptive_breath_cleanup(
            samples,
            windows=[NoiseWindow(0.0, 0.1)],
            sample_rate=1000,
            channels=1,
            max_attenuation_db=24.0,
            target_margin_db=3.0,
            context_ms=80.0,
            fade_ms=0.0,
            target_dbfs_override=-60.0,
        )

        self.assertEqual(details[0]["target_dbfs"], -60.0)
        self.assertEqual(details[0]["requested_attenuation_db"], 24.0)
        self.assertLess(max(cleaned), 150)

    def test_residual_second_pass_is_limited_to_narrower_parent_intersection(self) -> None:
        matched = match_residual_breath_windows(
            parent_windows=[NoiseWindow(1.0, 1.3)],
            residual_windows=[
                NoiseWindow(0.8, 1.15),
                NoiseWindow(2.0, 2.2),
            ],
        )

        self.assertEqual(matched, [NoiseWindow(1.0, 1.15)])

    def test_residual_assessment_trims_protected_fade_edges(self) -> None:
        trimmed = trim_noise_windows_for_assessment(
            [NoiseWindow(1.0, 1.055)],
            edge_seconds=0.006,
        )

        self.assertEqual(trimmed, [NoiseWindow(1.006, 1.049)])

    def test_auxiliary_breath_filter_keeps_noise_and_rejects_voiced_tone(self) -> None:
        sample_rate = 8000
        rng = random.Random(7)
        tone = [
            int(1800 * math.sin(2 * math.pi * 220 * index / sample_rate))
            for index in range(800)
        ]
        breath_noise = [rng.randint(-1800, 1800) for _ in range(800)]
        samples = array("h", [*tone, *breath_noise])

        accepted, evidence = filter_noise_like_breath_windows(
            samples,
            windows=[NoiseWindow(0.0, 0.1), NoiseWindow(0.1, 0.2)],
            sample_rate=sample_rate,
            channels=1,
        )

        self.assertEqual(accepted, [NoiseWindow(0.1, 0.2)])
        self.assertEqual(evidence[0]["decision"], "rejected_harmonic_content")
        self.assertEqual(evidence[1]["decision"], "accepted_noise_like")

    def test_final_clean_source_disables_harmful_secondary_denoise(self) -> None:
        preset, adaptations = apply_input_safety_overrides(
            "final",
            load_preset("final"),
            {
                "stationary_noise": False,
                "estimated_snr_db": 45.97,
            },
        )

        self.assertFalse(preset["filters"]["secondary_denoise"]["enabled"])
        self.assertIn("skip_secondary_denoise_clean_source", adaptations)

    def test_final_noisy_source_keeps_secondary_denoise(self) -> None:
        preset, adaptations = apply_input_safety_overrides(
            "final",
            load_preset("final"),
            {
                "stationary_noise": True,
                "estimated_snr_db": 18.0,
            },
        )

        self.assertTrue(preset["filters"]["secondary_denoise"]["enabled"])
        self.assertEqual(adaptations, [])

    def test_analyze_pcm16_samples_keeps_quiet_natural_audio_on_baseline(self) -> None:
        samples = array("h", [2000] * 1000 + [0] * 200 + [2600] * 1000)

        report = analyze_pcm16_samples(samples, sample_rate=1000)

        self.assertFalse(report["stationary_noise"])
        self.assertEqual(report["recommendations"][0]["action"], "natural_baseline_only")

    def test_analyze_pcm16_samples_finds_stable_noise_window(self) -> None:
        samples = array("h", [7000] * 1000 + [1000] * 600 + [6000] * 1000)

        report = analyze_pcm16_samples(samples, sample_rate=1000)

        self.assertTrue(report["stationary_noise"])
        self.assertEqual(report["recommendations"][0]["action"], "noise_window_candidate")
        self.assertEqual(report["candidate_noise_windows"][0]["start_seconds"], 1.0)

    def test_analyze_pcm16_samples_flags_clipping_for_manual_review(self) -> None:
        samples = array("h", [12000] * 1000 + [32767] * 20)

        report = analyze_pcm16_samples(samples, sample_rate=1000)

        self.assertEqual(report["clipped_sample_count"], 20)
        self.assertEqual(report["recommendations"][0]["action"], "manual_clipping_review")

    def test_compare_audio_preservation_accepts_uniform_gain(self) -> None:
        reference = analyze_pcm16_samples(array("h", [4000] * 2000), sample_rate=1000)
        processed = analyze_pcm16_samples(array("h", [5000] * 2000), sample_rate=1000)

        guard = compare_audio_preservation(reference, processed)

        self.assertEqual(guard["status"], "PASS")
        self.assertFalse(guard["release_blocked"])

    def test_compare_audio_preservation_aligns_small_fixed_filter_delay(self) -> None:
        reference_levels = [-18.0, -24.0, -16.0, -30.0, -20.0, -14.0, -28.0, -19.0]
        processed_levels = [-80.0, *[level - 0.2 for level in reference_levels[:-1]]]
        reference = {
            "_frame_rms_db": reference_levels,
            "speech_activity_threshold_dbfs": -50.0,
            "duration_seconds": 1.0,
            "clipped_sample_count": 0,
            "frame_ms": 20.0,
        }
        processed = {
            "_frame_rms_db": processed_levels,
            "speech_activity_threshold_dbfs": -50.0,
            "duration_seconds": 1.0,
            "clipped_sample_count": 0,
            "frame_ms": 20.0,
        }

        guard = compare_audio_preservation(reference, processed)

        self.assertEqual(guard["status"], "PASS")
        self.assertEqual(guard["alignment_frames"], 1)
        self.assertEqual(guard["alignment_ms"], 20.0)

    def test_compare_audio_preservation_blocks_swallowed_speech_and_level_swings(self) -> None:
        reference = analyze_pcm16_samples(array("h", [6000] * 2000), sample_rate=1000)
        processed = analyze_pcm16_samples(
            array("h", [6000] * 500 + [1000] * 500 + [10000] * 500 + [6000] * 500),
            sample_rate=1000,
        )

        guard = compare_audio_preservation(reference, processed)

        self.assertEqual(guard["status"], "FAIL")
        self.assertTrue(guard["release_blocked"])
        self.assertIn("active_speech_attenuated", guard["failures"])
        self.assertIn("short_term_gain_instability", guard["failures"])

    def test_compare_audio_preservation_blocks_channel_or_sample_rate_changes(self) -> None:
        reference = analyze_pcm16_samples(array("h", [6000] * 2000), sample_rate=1000)
        processed = analyze_pcm16_samples(array("h", [6000] * 2000), sample_rate=1000)

        guard = compare_audio_preservation(
            reference,
            processed,
            reference_format={"sample_rate": 44100, "channels": 2},
            processed_format={"sample_rate": 48000, "channels": 1},
        )

        self.assertEqual(guard["status"], "FAIL")
        self.assertIn("sample_rate_changed", guard["failures"])
        self.assertIn("channel_layout_changed", guard["failures"])

    def test_compare_audio_preservation_blocks_source_active_hard_mute(self) -> None:
        sample_rate = 1000
        reference_samples = array("h", [8000] * 2000)
        processed_samples = array("h", [8000] * 1000 + [0] * 40 + [8000] * 960)
        reference = analyze_pcm16_samples(reference_samples, sample_rate=sample_rate, frame_ms=10.0)
        processed = analyze_pcm16_samples(processed_samples, sample_rate=sample_rate, frame_ms=10.0)

        guard = compare_audio_preservation(
            reference,
            processed,
            reference_samples=reference_samples,
            processed_samples=processed_samples,
            sample_rate=sample_rate,
        )

        self.assertEqual(guard["status"], "FAIL")
        self.assertIn("source_active_hard_mute", guard["failures"])
        self.assertGreaterEqual(len(guard["hard_mute_windows"]), 1)
        self.assertGreaterEqual(guard["hard_mute_windows"][0]["duration_ms"], 20)

    def test_compare_audio_preservation_blocks_high_band_clarity_loss(self) -> None:
        sample_rate = 8000
        duration = sample_rate  # 1 second
        reference_samples = array("h")
        processed_samples = array("h")
        for index in range(duration):
            t = index / sample_rate
            low = 0.35 * math.sin(2 * math.pi * 400 * t)
            high = 0.35 * math.sin(2 * math.pi * 2800 * t)
            reference_samples.append(int(12000 * (low + high)))
            processed_samples.append(int(12000 * (low + 0.05 * high)))
        reference = analyze_pcm16_samples(reference_samples, sample_rate=sample_rate, frame_ms=20.0)
        processed = analyze_pcm16_samples(processed_samples, sample_rate=sample_rate, frame_ms=20.0)

        guard = compare_audio_preservation(
            reference,
            processed,
            reference_samples=reference_samples,
            processed_samples=processed_samples,
            sample_rate=sample_rate,
        )

        self.assertEqual(guard["status"], "FAIL")
        self.assertIn("spectral_clarity_lost", guard["failures"])
        self.assertIn("2-4k", guard["spectral_band_deltas_db"])
        self.assertLess(guard["spectral_band_deltas_db"]["2-4k"], -3.0)

    def test_compare_audio_preservation_blocks_excessive_high_band_gain(self) -> None:
        sample_rate = 8000
        duration = sample_rate
        reference_samples = array("h")
        processed_samples = array("h")
        for index in range(duration):
            t = index / sample_rate
            low = 0.35 * math.sin(2 * math.pi * 400 * t)
            high = 0.2 * math.sin(2 * math.pi * 2800 * t)
            reference_samples.append(int(12000 * (low + high)))
            processed_samples.append(int(12000 * (low + 1.9 * high)))
        reference = analyze_pcm16_samples(reference_samples, sample_rate=sample_rate, frame_ms=20.0)
        processed = analyze_pcm16_samples(processed_samples, sample_rate=sample_rate, frame_ms=20.0)

        guard = compare_audio_preservation(
            reference,
            processed,
            reference_samples=reference_samples,
            processed_samples=processed_samples,
            sample_rate=sample_rate,
        )

        self.assertEqual(guard["status"], "FAIL")
        self.assertIn("spectral_harshness_increased", guard["failures"])
        self.assertGreater(guard["spectral_band_deltas_db"]["2-4k"], 3.0)

    def test_compare_audio_preservation_excludes_authorized_cleanup_windows(self) -> None:
        sample_rate = 8000
        reference_samples = array("h")
        processed_samples = array("h")
        for index in range(sample_rate * 2):
            t = index / sample_rate
            if index < sample_rate:
                value = int(
                    9000
                    * (
                        0.35 * math.sin(2 * math.pi * 400 * t)
                        + 0.35 * math.sin(2 * math.pi * 2800 * t)
                    )
                )
                reference_samples.append(value)
                processed_samples.append(value)
            else:
                reference_samples.append(
                    int(6000 * math.sin(2 * math.pi * 2800 * t))
                )
                processed_samples.append(0)
        reference = analyze_pcm16_samples(reference_samples, sample_rate=sample_rate)
        processed = analyze_pcm16_samples(processed_samples, sample_rate=sample_rate)

        guard = compare_audio_preservation(
            reference,
            processed,
            reference_samples=reference_samples,
            processed_samples=processed_samples,
            sample_rate=sample_rate,
            excluded_windows=[NoiseWindow(1.0, 2.0)],
        )

        self.assertEqual(guard["status"], "PASS")
        self.assertEqual(guard["excluded_window_count"], 1)

    def test_select_adaptive_profile_uses_only_safe_whitelisted_profiles(self) -> None:
        cases = [
            ({"clipped_sample_count": 5}, "manual_review"),
            (
                {
                    "clipped_sample_count": 0,
                    "stationary_noise": True,
                    "candidate_noise_windows": [{}],
                },
                "noise_review",
            ),
            (
                {
                    "clipped_sample_count": 0,
                    "stationary_noise": False,
                    "active_dynamic_range_db": 15.0,
                },
                "leveling_gentle",
            ),
            (
                {
                    "clipped_sample_count": 0,
                    "stationary_noise": False,
                    "active_dynamic_range_db": 5.0,
                },
                "baseline",
            ),
        ]

        for analysis, expected in cases:
            decision = select_adaptive_profile(analysis)
            self.assertEqual(decision["profile"], expected)
            self.assertIn(
                decision["profile"],
                {"baseline", "leveling_gentle", "noise_review", "manual_review"},
            )
            self.assertFalse(decision["allow_destructive_cleanup"])

    def test_resolve_respiro_runtime_prefers_cli_over_env(self) -> None:
        runtime = resolve_respiro_runtime(
            respiro_repo="D:/cli/repo",
            respiro_weights="D:/cli/respiro-en.pt",
            env_values={
                "AUDIO_SOUND_RESPIRO_REPO": "D:/env/repo",
                "AUDIO_SOUND_RESPIRO_WEIGHTS": "D:/env/respiro-en.pt",
            },
        )
        self.assertEqual(runtime["repo_path"], Path("D:/cli/repo"))
        self.assertEqual(runtime["weights_path"], Path("D:/cli/respiro-en.pt"))

    def test_run_respiro_or_fallback_detection_uses_fallback_when_assets_missing(self) -> None:
        result = run_respiro_or_fallback_detection(
            audio_path=Path("D:/out/audio_clean.wav"),
            ffmpeg_bin="ffmpeg",
            respiro_repo=None,
            respiro_weights=None,
            python_executable="python",
            threshold=0.064,
            min_length_ms=20,
            fallback_config={
                "low_threshold_db": -44,
                "high_threshold_db": -32,
                "silence_min_duration": 0.08,
                "min_breath_ms": 70,
                "max_breath_ms": 320,
                "pre_roll_ms": 25,
                "fade_ms": 14,
                "floor_gain": 0.0,
                "analysis_hop_ms": 18,
                "analysis_scan_ms": 260,
                "noise_floor_ms": 120,
                "speech_start_ratio": 0.78,
                "breath_over_noise_ratio": 2.6,
                "speech_over_breath_ratio": 2.15,
                "speech_confirm_frames": 2,
            },
        )
        self.assertEqual(result.windows, [])
        self.assertEqual(result.mode, "fallback")
        self.assertFalse(result.assets_present)
        self.assertFalse(result.attempted)
        self.assertFalse(result.succeeded)

    def test_run_respiro_or_fallback_detection_uses_respiro_when_command_succeeds(self) -> None:
        repo = Path("D:/models/Respiro-en")
        weights = Path("D:/models/respiro-en.pt")

        class Result:
            returncode = 0
            stdout = '{"intervals":[{"start_seconds":1.25,"end_seconds":1.6}]}'
            stderr = ""

        with mock.patch("pathlib.Path.exists", return_value=True), mock.patch(
            "audio_sound.pipeline.run_command",
            return_value=Result(),
        ):
            result = run_respiro_or_fallback_detection(
                audio_path=Path("D:/out/audio_raw.wav"),
                ffmpeg_bin="ffmpeg",
                respiro_repo=repo,
                respiro_weights=weights,
                python_executable="python",
                threshold=0.064,
                min_length_ms=20,
                fallback_config={},
            )

        self.assertEqual(result.mode, "respiro")
        self.assertTrue(result.assets_present)
        self.assertTrue(result.attempted)
        self.assertTrue(result.succeeded)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.windows, [NoiseWindow(start_seconds=1.25, end_seconds=1.6)])
        self.assertEqual(result.command[-5:], ["D:\\models\\Respiro-en", "D:\\models\\respiro-en.pt", "D:\\out\\audio_raw.wav", "0.064", "20"])

    def test_run_respiro_or_fallback_detection_falls_back_when_respiro_command_fails(self) -> None:
        repo = Path("D:/models/Respiro-en")
        weights = Path("D:/models/respiro-en.pt")
        fallback_windows = [NoiseWindow(start_seconds=0.5, end_seconds=0.8)]

        class Result:
            returncode = 1
            stdout = ""
            stderr = "boom"

        with mock.patch("pathlib.Path.exists", return_value=True), mock.patch(
            "audio_sound.pipeline.run_command",
            return_value=Result(),
        ), mock.patch(
            "audio_sound.pipeline.detect_breath_onset_windows",
            return_value=fallback_windows,
        ):
            result = run_respiro_or_fallback_detection(
                audio_path=Path("D:/out/audio_raw.wav"),
                ffmpeg_bin="ffmpeg",
                respiro_repo=repo,
                respiro_weights=weights,
                python_executable="python",
                threshold=0.064,
                min_length_ms=20,
                fallback_config={},
            )

        self.assertEqual(result.mode, "fallback")
        self.assertTrue(result.assets_present)
        self.assertTrue(result.attempted)
        self.assertFalse(result.succeeded)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.windows, fallback_windows)
        self.assertEqual(result.error, "boom")

    def test_run_respiro_or_fallback_detection_segments_long_audio(self) -> None:
        repo = Path("D:/models/Respiro-en")
        weights = Path("D:/models/respiro-en.pt")

        class Result:
            def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
                self.returncode = returncode
                self.stdout = stdout
                self.stderr = stderr

        calls: list[list[str]] = []
        exists_map = {
            "D:\\out\\audio_raw.wav",
            "D:\\models\\Respiro-en",
            "D:\\models\\respiro-en.pt",
        }

        def fake_exists(path_obj: Path) -> bool:
            path_str = str(path_obj)
            return path_str in exists_map or "segment_" in path_str

        def fake_run(command: list[str]):
            calls.append(command)
            if command[0] == "ffmpeg":
                return Result(0)
            segment_id = 0 if "segment_0000.wav" in command[-3] else 1
            stdout = (
                '{"intervals":[{"start_seconds":1.0,"end_seconds":1.2}]}'
                if segment_id == 0
                else '{"intervals":[{"start_seconds":2.0,"end_seconds":2.3}]}'
            )
            return Result(0, stdout=stdout)

        with mock.patch("pathlib.Path.exists", fake_exists), mock.patch(
            "audio_sound.pipeline._read_wave_duration_seconds",
            return_value=130.0,
        ), mock.patch(
            "audio_sound.pipeline.ensure_tool",
            side_effect=lambda value: value,
        ), mock.patch(
            "audio_sound.pipeline.run_command",
            side_effect=fake_run,
        ):
            result = run_respiro_or_fallback_detection(
                audio_path=Path("D:/out/audio_raw.wav"),
                ffmpeg_bin="ffmpeg",
                respiro_repo=repo,
                respiro_weights=weights,
                python_executable="python",
                threshold=0.064,
                min_length_ms=20,
                fallback_config={},
            )

        self.assertEqual(result.mode, "respiro")
        self.assertTrue(result.succeeded)
        self.assertEqual(
            result.windows,
            [
                NoiseWindow(start_seconds=1.0, end_seconds=1.2),
                NoiseWindow(start_seconds=121.0, end_seconds=121.3),
            ],
        )
        ffmpeg_calls = [command for command in calls if command[0] == "ffmpeg"]
        detect_calls = [command for command in calls if command[0] == "python"]
        self.assertEqual(len(ffmpeg_calls), 2)
        self.assertEqual(len(detect_calls), 2)

    def test_apply_spectramini_style_cleanup_to_samples_reduces_click_and_breath_frames(self) -> None:
        from array import array

        samples = array("h", [0, 100, 120, 8000, -8000, 120, 110, 100, 90, 80, 70, 60])
        cleaned = apply_spectramini_style_cleanup_to_samples(
            samples,
            breath_windows=[NoiseWindow(start_seconds=0.0, end_seconds=0.0001)],
            sample_rate=48000,
            attenuation_db=18.0,
            mouth_declick_sensitivity=0.6,
            fade_ms=0.0,
        )
        self.assertLess(max(abs(value) for value in cleaned), max(abs(value) for value in samples))

    def test_apply_spectramini_style_cleanup_preserves_regular_tone_without_breath_windows(self) -> None:
        from array import array

        sample_rate = 48000
        samples = array(
            "h",
            [
                int(12000 * math.sin(2 * math.pi * 1000 * index / sample_rate))
                for index in range(sample_rate // 10)
            ],
        )

        cleaned = apply_spectramini_style_cleanup_to_samples(
            samples,
            breath_windows=[],
            sample_rate=sample_rate,
            attenuation_db=18.0,
            mouth_declick_sensitivity=0.6,
            fade_ms=0.0,
        )

        self.assertEqual(cleaned, samples)

    def test_spectramini_zero_mouth_sensitivity_disables_global_declick(self) -> None:
        samples = array("h", [0, 100, 120, 8000, -8000, 120, 110, 100])

        cleaned = apply_spectramini_style_cleanup_to_samples(
            samples,
            breath_windows=[],
            sample_rate=48000,
            attenuation_db=3.0,
            mouth_declick_sensitivity=0.0,
            fade_ms=14.0,
        )

        self.assertEqual(cleaned, samples)

    def test_repair_deepfilternet_speech_dropouts_restores_short_real_speech_window(self) -> None:
        from array import array

        sample_rate = 100
        reference = array("h", [0] * 90)
        for index in range(0, 20):
            reference[index] = 5200
        for index in range(20, 26):
            reference[index] = 4200
        for index in range(26, 50):
            reference[index] = 5200
        processed = array("h", reference)
        for index in range(20, 26):
            processed[index] = 120

        repaired, windows = repair_deepfilternet_speech_dropouts(
            reference,
            processed,
            sample_rate=sample_rate,
            window_seconds=0.06,
            hop_seconds=0.01,
            reference_peak_db_min=-24.0,
            reference_rms_db_min=-38.0,
            processed_peak_db_max=-34.0,
            processed_rms_db_max=-46.0,
            copy_padding_seconds=0.0,
            max_repair_duration_seconds=0.16,
            context_window_seconds=0.18,
            context_gap_seconds=0.0,
            context_peak_db_min=-20.0,
            context_rms_db_min=-32.0,
        )

        self.assertEqual(windows, [NoiseWindow(start_seconds=0.2, end_seconds=0.26)])
        self.assertEqual(repaired[20:26], reference[20:26])

    def test_repair_deepfilternet_speech_dropouts_skips_isolated_leading_artifact(self) -> None:
        from array import array

        sample_rate = 100
        reference = array("h", [0] * 90)
        for index in range(20, 26):
            reference[index] = 4200
        for index in range(50, 80):
            reference[index] = 5200
        processed = array("h", reference)
        for index in range(20, 26):
            processed[index] = 120

        repaired, windows = repair_deepfilternet_speech_dropouts(
            reference,
            processed,
            sample_rate=sample_rate,
            window_seconds=0.06,
            hop_seconds=0.01,
            reference_peak_db_min=-24.0,
            reference_rms_db_min=-38.0,
            processed_peak_db_max=-34.0,
            processed_rms_db_max=-46.0,
            copy_padding_seconds=0.0,
            max_repair_duration_seconds=0.16,
            context_window_seconds=0.18,
            context_gap_seconds=0.0,
            context_peak_db_min=-20.0,
            context_rms_db_min=-32.0,
        )

        self.assertEqual(windows, [])
        self.assertEqual(repaired, processed)

    def test_repair_deepfilternet_speech_dropouts_skips_long_low_energy_regions(self) -> None:
        from array import array

        sample_rate = 100
        reference = array("h", [0] * 120)
        for index in range(10, 40):
            reference[index] = 3200
        processed = array("h", [0] * 120)

        repaired, windows = repair_deepfilternet_speech_dropouts(
            reference,
            processed,
            sample_rate=sample_rate,
            window_seconds=0.06,
            hop_seconds=0.01,
            reference_peak_db_min=-24.0,
            reference_rms_db_min=-38.0,
            processed_peak_db_max=-34.0,
            processed_rms_db_max=-46.0,
            copy_padding_seconds=0.0,
            max_repair_duration_seconds=0.16,
        )

        self.assertEqual(windows, [])
        self.assertEqual(repaired, processed)

    def test_attenuation_db_to_gain_converts_db_to_linear_floor(self) -> None:
        self.assertAlmostEqual(attenuation_db_to_gain(18.0), 0.12589254117941673)

    def test_build_respiro_detect_command_passes_repo_weights_and_threshold(self) -> None:
        command = build_respiro_detect_command(
            audio_path=Path("D:/out/audio_raw.wav"),
            python_executable="python",
            repo_path=Path("D:/models/Respiro-en"),
            weights_path=Path("D:/models/respiro-en.pt"),
            threshold=0.5,
            min_length_ms=30,
        )
        self.assertEqual(command[0:2], ["python", "-c"])
        self.assertEqual(command[-5:], ["D:\\models\\Respiro-en", "D:\\models\\respiro-en.pt", "D:\\out\\audio_raw.wav", "0.5", "30"])

    def test_build_breath_processing_plan_prefers_respiro_then_spectra_then_deepfilternet(self) -> None:
        preset = load_preset("safe")
        plan = build_breath_processing_plan(
            preset,
            attenuation_db=18.0,
            skip_spectramini=False,
            skip_deepfilternet=False,
        )
        self.assertEqual([step["type"] for step in plan], ["respiro", "spectramini", "deepfilternet"])
        self.assertEqual(plan[0]["attenuation_db"], 18.0)

    def test_build_breath_processing_plan_can_skip_optional_stages(self) -> None:
        preset = load_preset("safe")
        plan = build_breath_processing_plan(
            preset,
            attenuation_db=12.0,
            skip_spectramini=True,
            skip_deepfilternet=True,
        )
        self.assertEqual([step["type"] for step in plan], ["respiro"])

    def test_natural_preset_disables_automatic_voice_modification_stages(self) -> None:
        preset = load_preset("natural")
        plan = build_breath_processing_plan(
            preset,
            attenuation_db=0.0,
            skip_spectramini=False,
            skip_deepfilternet=False,
        )

        self.assertEqual(plan, [])

    def test_discover_media_files_supports_audio_and_video(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "voice.wav").write_bytes(b"")
            (root / "clip.mp4").write_bytes(b"")
            nested = root / "nested"
            nested.mkdir()
            (nested / "lesson.mov").write_bytes(b"")
            (nested / "notes.txt").write_text("ignore", encoding="utf-8")

            top_only = discover_media_files(root, recursive=False)
            recursive = discover_media_files(root, recursive=True)

            self.assertEqual([item.name for item in top_only], ["clip.mp4", "voice.wav"])
            self.assertEqual([item.name for item in recursive], ["clip.mp4", "lesson.mov", "voice.wav"])

    def test_build_output_layout_uses_source_name_suffixes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir) / "output"
            layout = build_output_layout(
                input_path=Path("D:/media/My Clip 01.mp4"),
                output_root=output_root,
                run_slug="20260527-120000",
            )

            self.assertEqual(layout.job_name, "20260527-120000_My Clip 01")
            self.assertTrue(str(layout.preprocess_dir).endswith("audio_preprocess"))
            self.assertTrue(str(layout.transcript_dir).endswith("transcript_ready"))
            self.assertEqual(layout.raw_wav.name, "My Clip 01_raw.wav")
            self.assertEqual(layout.breath_wav.name, "My Clip 01_breath.wav")
            self.assertEqual(layout.denoised_wav.name, "My Clip 01_df.wav")
            self.assertEqual(layout.noise_sample_wav.name, "My Clip 01_noise_sample.wav")
            self.assertEqual(layout.clean_wav.name, "My Clip 01_clean.wav")
            self.assertEqual(layout.transcript_mp3.name, "My Clip 01_transcript.mp3")
            self.assertEqual(layout.report_json.name, "audio_process_report.json")

    def test_build_output_layout_preserves_unicode_source_name_for_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir) / "output"
            source = Path("C:/Users/Guanghe/Downloads/?????????-??.wav")
            layout = build_output_layout(
                input_path=source,
                output_root=output_root,
                run_slug="20260527-120000",
            )

            self.assertEqual(layout.job_name, "20260527-120000_?????????-??")
            self.assertEqual(layout.raw_wav.name, "?????????-??_raw.wav")
            self.assertEqual(layout.clean_wav.name, "?????????-??_clean.wav")
            self.assertEqual(layout.transcript_mp3.name, "?????????-??_transcript.mp3")

    def test_build_ffmpeg_extract_command_uses_preset_values(self) -> None:
        preset = load_preset("safe")
        command = build_ffmpeg_extract_command(
            input_path=Path("D:/input/source.mp4"),
            raw_wav=Path("D:/out/audio_raw.wav"),
            preset=preset,
            ffmpeg_bin="ffmpeg",
        )
        self.assertEqual(
            command,
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-nostdin",
                "-i",
                "D:\\input\\source.mp4",
                "-vn",
                "-ac",
                "1",
                "-ar",
                "48000",
                "-c:a",
                "pcm_s16le",
                "D:\\out\\audio_raw.wav",
            ],
        )

    def test_resolve_processing_format_preserves_source_stereo_and_sample_rate(self) -> None:
        preset = load_preset("natural")

        resolved = resolve_processing_format(
            {"sample_rate": 44100, "channels": 2},
            preset,
        )

        self.assertEqual(resolved["sample_rate"], 44100)
        self.assertEqual(resolved["channels"], 2)
        self.assertEqual(resolved["pcm_codec"], "pcm_s16le")

    def test_build_ffmpeg_extract_command_can_preserve_source_format(self) -> None:
        preset = load_preset("natural")
        command = build_ffmpeg_extract_command(
            input_path=Path("D:/input/source.mp3"),
            raw_wav=Path("D:/out/audio_raw.wav"),
            preset=preset,
            ffmpeg_bin="ffmpeg",
            processing_format={"sample_rate": 44100, "channels": 2, "pcm_codec": "pcm_s16le"},
        )

        self.assertEqual(command[command.index("-ac") + 1], "2")
        self.assertEqual(command[command.index("-ar") + 1], "44100")

    def test_build_deepfilternet_command_includes_post_filter(self) -> None:
        preset = load_preset("safe")
        command = build_deepfilternet_command(
            raw_wav=Path("D:/out/audio_raw.wav"),
            output_dir=Path("D:/out/df"),
            preset=preset,
            python_executable="python",
        )
        self.assertEqual(
            command,
            [
                "python",
                "-m",
                "df.enhance",
                "--output-dir",
                "D:\\out\\df",
                "--pf",
                "D:\\out\\audio_raw.wav",
            ],
        )

    def test_build_ffmpeg_finalize_commands_include_clean_and_transcript_exports(self) -> None:
        preset = load_preset("review")
        commands = build_ffmpeg_finalize_commands(
            denoised_wav=Path("D:/out/audio_df.wav"),
            clean_wav=Path("D:/out/audio_clean.wav"),
            transcript_mp3=Path("D:/out/transcript_ready/audio.mp3"),
            preset=preset,
            ffmpeg_bin="ffmpeg",
        )
        self.assertEqual(len(commands), 2)
        self.assertIn("loudnorm=I=-16.8:LRA=7.0:TP=-3.0:print_format=json", commands[0][7])
        self.assertEqual(commands[0][8:10], ["-ar", "48000"])
        self.assertEqual(commands[1][-2:], ["192k", "D:\\out\\transcript_ready\\audio.mp3"])

    def test_build_ffmpeg_finalize_commands_can_preserve_source_format(self) -> None:
        preset = load_preset("natural")
        commands = build_ffmpeg_finalize_commands(
            denoised_wav=Path("D:/out/audio_df.wav"),
            clean_wav=Path("D:/out/audio_clean.wav"),
            transcript_mp3=Path("D:/out/transcript_ready/audio.mp3"),
            preset=preset,
            ffmpeg_bin="ffmpeg",
            processing_format={"sample_rate": 44100, "channels": 2, "pcm_codec": "pcm_s16le"},
        )

        self.assertEqual(commands[0][commands[0].index("-ac") + 1], "2")
        self.assertEqual(commands[0][commands[0].index("-ar") + 1], "44100")

    def test_measure_segment_levels_reports_peak_and_rms_dbfs(self) -> None:
        from array import array

        samples = array("h", [0, 0, 1000, -1000])
        peak_db, rms_db = measure_segment_levels(samples, start_index=0, end_index=len(samples))
        self.assertLess(peak_db, -20.0)
        self.assertLess(rms_db, peak_db)

    def test_infer_pause_residual_cleanup_windows_captures_core_silence_and_tiny_bridge(self) -> None:
        from array import array

        sample_rate = 10
        samples = array("h", [0] * 40)
        samples[18] = 20
        samples[19] = -20
        silence_candidates = [
            {"start_seconds": 0.0, "end_seconds": 1.8, "duration_seconds": 1.8},
            {"start_seconds": 2.0, "end_seconds": 4.0, "duration_seconds": 2.0},
        ]

        windows = infer_pause_residual_cleanup_windows(
            samples,
            sample_rate=sample_rate,
            silence_candidates=silence_candidates,
            min_neighbor_silence_duration=0.6,
            bridge_max_duration=0.3,
            bridge_peak_db=-50.0,
            bridge_rms_db=-55.0,
            core_pad_seconds=0.1,
            total_duration_seconds=4.0,
        )

        self.assertEqual(
            windows,
            [
                NoiseWindow(start_seconds=0.1, end_seconds=1.7),
                NoiseWindow(start_seconds=1.8, end_seconds=2.0),
                NoiseWindow(start_seconds=2.1, end_seconds=3.9),
            ],
        )

    def test_infer_pause_residual_cleanup_windows_uses_tighter_leading_trailing_pad(self) -> None:
        from array import array

        sample_rate = 100
        samples = array("h", [0] * 1000)
        silence_candidates = [
            {"start_seconds": 0.0, "end_seconds": 2.0, "duration_seconds": 2.0},
            {"start_seconds": 4.0, "end_seconds": 6.0, "duration_seconds": 2.0},
            {"start_seconds": 8.0, "end_seconds": 10.0, "duration_seconds": 2.0},
        ]

        windows = infer_pause_residual_cleanup_windows(
            samples,
            sample_rate=sample_rate,
            silence_candidates=silence_candidates,
            min_neighbor_silence_duration=1.0,
            bridge_max_duration=0.1,
            bridge_peak_db=-50.0,
            bridge_rms_db=-55.0,
            core_pad_seconds=0.045,
            leading_trailing_pad_seconds=0.025,
            total_duration_seconds=10.0,
        )

        self.assertEqual(
            windows,
            [
                NoiseWindow(start_seconds=0.025, end_seconds=1.955),
                NoiseWindow(start_seconds=4.045, end_seconds=5.955),
                NoiseWindow(start_seconds=8.045, end_seconds=9.975),
            ],
        )

    def test_infer_pause_residual_cleanup_uses_frame_count_for_stereo_trailing_pad(self) -> None:
        from array import array

        sample_rate = 100
        samples = array("h", [0] * 2000)

        windows = infer_pause_residual_cleanup_windows(
            samples,
            sample_rate=sample_rate,
            channels=2,
            silence_candidates=[
                {"start_seconds": 8.0, "end_seconds": 10.0, "duration_seconds": 2.0},
            ],
            min_neighbor_silence_duration=1.0,
            bridge_max_duration=0.1,
            bridge_peak_db=-50.0,
            bridge_rms_db=-55.0,
            core_pad_seconds=0.045,
            leading_trailing_pad_seconds=0.025,
        )

        self.assertEqual(
            windows,
            [NoiseWindow(start_seconds=8.045, end_seconds=9.975)],
        )

    def test_speech_safe_autogate_can_push_confirmed_silence_near_floor(self) -> None:
        from array import array
        import math

        # ~ -60 dBFS tone burst that should be driven near -96 dBFS floor.
        amplitude = int(round(32768 * (10 ** (-60.0 / 20.0))))
        samples = array("h", [0] * 100 + [amplitude] * 200 + [1800] * 100)

        cleaned, details = apply_adaptive_breath_cleanup(
            samples,
            windows=[NoiseWindow(0.1, 0.3)],
            sample_rate=1000,
            channels=1,
            max_attenuation_db=48.0,
            target_margin_db=-6.0,
            context_ms=0.0,
            fade_ms=12.0,
            target_dbfs_override=-96.0,
        )

        core = cleaned[120:280]
        peak = max(abs(sample) for sample in core)
        rms = math.sqrt(sum(sample * sample for sample in core) / len(core))
        peak_db = 20.0 * math.log10(peak / 32768.0) if peak else -120.0
        rms_db = 20.0 * math.log10(rms / 32768.0) if rms else -120.0

        self.assertGreaterEqual(details[0]["requested_attenuation_db"], 30.0)
        self.assertEqual(details[0]["target_dbfs"], -96.0)
        self.assertLessEqual(peak_db, -93.0)
        self.assertLessEqual(rms_db, -93.0)
        self.assertEqual(cleaned[300:], samples[300:])

    def test_autogate_mode_disables_unpadded_bridge_windows(self) -> None:
        from array import array

        samples = array("h", [0] * 40)
        samples[18] = 20
        samples[19] = -20
        silence_candidates = [
            {"start_seconds": 0.0, "end_seconds": 1.8, "duration_seconds": 1.8},
            {"start_seconds": 2.0, "end_seconds": 4.0, "duration_seconds": 2.0},
        ]

        windows = infer_pause_residual_cleanup_windows(
            samples,
            sample_rate=10,
            silence_candidates=silence_candidates,
            min_neighbor_silence_duration=0.6,
            bridge_max_duration=0.3,
            bridge_peak_db=-50.0,
            bridge_rms_db=-55.0,
            core_pad_seconds=0.1,
            total_duration_seconds=4.0,
            allow_bridge_windows=False,
        )

        self.assertEqual(
            windows,
            [
                NoiseWindow(start_seconds=0.1, end_seconds=1.7),
                NoiseWindow(start_seconds=2.1, end_seconds=3.9),
            ],
        )

    def test_trim_assessment_can_empty_short_residual_windows(self) -> None:
        trimmed = trim_noise_windows_for_assessment(
            [NoiseWindow(1.0, 1.03)],
            edge_seconds=0.006,
            minimum_duration_seconds=0.02,
        )
        self.assertEqual(trimmed, [])

    def test_parse_noise_window_accepts_start_end_seconds(self) -> None:
        window = parse_noise_window("161.583729:169.578792")
        self.assertEqual(window, NoiseWindow(start_seconds=161.583729, end_seconds=169.578792))

    def test_parse_noise_window_rejects_invalid_ranges(self) -> None:
        with self.assertRaises(ValueError):
            parse_noise_window("3.0:3.0")

        with self.assertRaises(ValueError):
            parse_noise_window("bad-value")

    def test_infer_breath_onset_windows_finds_short_bridge_between_low_and_high_silence(self) -> None:
        windows = infer_breath_onset_windows(
            [
                {"start_seconds": 12.0, "end_seconds": 12.92, "duration_seconds": 0.92},
            ],
            [
                {"start_seconds": 12.9, "end_seconds": 13.18, "duration_seconds": 0.28},
            ],
            min_breath_seconds=0.08,
            max_breath_seconds=0.45,
            pre_roll_seconds=0.03,
        )
        self.assertEqual(windows, [NoiseWindow(start_seconds=12.89, end_seconds=13.18)])

    def test_duck_samples_for_windows_attentuates_target_region(self) -> None:
        from array import array

        samples = array("h", [1000] * 20)
        duck_samples_for_windows(
            samples,
            sample_rate=10,
            windows=[NoiseWindow(start_seconds=0.5, end_seconds=1.0)],
            floor_gain=0.0,
            fade_ms=0.0,
        )
        self.assertEqual(samples[4], 1000)
        self.assertEqual(samples[5], 0)
        self.assertEqual(samples[9], 0)
        self.assertEqual(samples[10], 1000)

    def test_infer_breath_window_from_frame_rms_detects_breath_before_speech(self) -> None:
        inferred = infer_breath_window_from_frame_rms(
            [8.0, 11.0, 14.0, 38.0, 45.0],
            silence_floor_rms=2.0,
            hop_seconds=0.02,
            min_breath_seconds=0.04,
            max_breath_seconds=0.14,
            speech_start_ratio=0.75,
            breath_over_noise_ratio=3.0,
            speech_over_breath_ratio=2.2,
            speech_confirm_frames=2,
        )
        self.assertEqual(inferred, (0.0, 0.06))

    def test_infer_breath_window_from_frame_rms_rejects_near_silence(self) -> None:
        inferred = infer_breath_window_from_frame_rms(
            [2.1, 2.2, 2.0, 2.3, 2.2],
            silence_floor_rms=2.0,
            hop_seconds=0.02,
            min_breath_seconds=0.04,
            max_breath_seconds=0.14,
            speech_start_ratio=0.75,
            breath_over_noise_ratio=3.0,
            speech_over_breath_ratio=2.2,
            speech_confirm_frames=2,
        )
        self.assertIsNone(inferred)

    def test_build_ffmpeg_noise_sample_command_concatenates_multiple_segments(self) -> None:
        preset = load_preset("voice-isolate")
        command = build_ffmpeg_noise_sample_command(
            source_wav=Path("D:/out/audio_df.wav"),
            noise_sample_wav=Path("D:/out/audio_noise_sample.wav"),
            noise_windows=[
                NoiseWindow(start_seconds=143.089208, end_seconds=144.093687),
                NoiseWindow(start_seconds=161.583729, end_seconds=169.578792),
            ],
            preset=preset,
            ffmpeg_bin="ffmpeg",
        )
        self.assertEqual(command[:8], ["ffmpeg", "-y", "-hide_banner", "-nostdin", "-i", "D:\\out\\audio_df.wav", "-filter_complex", command[7]])
        self.assertIn("[0:a]atrim=start=143.089208:end=144.093687,asetpts=PTS-STARTPTS[s0]", command[7])
        self.assertIn("[0:a]atrim=start=161.583729:end=169.578792,asetpts=PTS-STARTPTS[s1]", command[7])
        self.assertIn("[s0][s1]concat=n=2:v=0:a=1[outa]", command[7])
        self.assertEqual(command[-1], "D:\\out\\audio_noise_sample.wav")

    def test_build_ffmpeg_finalize_commands_use_noise_sample_profile_when_present(self) -> None:
        preset = load_preset("voice-isolate")
        commands = build_ffmpeg_finalize_commands(
            denoised_wav=Path("D:/out/audio_df.wav"),
            clean_wav=Path("D:/out/audio_clean.wav"),
            transcript_mp3=Path("D:/out/transcript_ready/audio.mp3"),
            preset=preset,
            ffmpeg_bin="ffmpeg",
            noise_sample_wav=Path("D:/out/audio_noise_sample.wav"),
            noise_sample_duration=8.5,
        )
        self.assertEqual(len(commands), 2)
        filter_complex_index = commands[0].index("-filter_complex") + 1
        filter_complex = commands[0][filter_complex_index]
        self.assertIn("afftdn=", filter_complex)
        self.assertIn("sn=start", filter_complex)
        self.assertIn("atrim=start=8.5", filter_complex)
        self.assertIn("adeclick=", filter_complex)
        self.assertIn("deesser=", filter_complex)
        self.assertNotIn("sidechaingate=", filter_complex)
        self.assertNotIn("asplit=2", filter_complex)
        self.assertEqual(filter_complex.count("afftdn="), 1)
        self.assertEqual(filter_complex.count("loudnorm="), 1)
        self.assertEqual(filter_complex.count("adeclick="), 1)
        self.assertEqual(filter_complex.count("sidechaingate="), 0)
        self.assertIn("-map", commands[0])
        self.assertEqual(commands[0][-1], "D:\\out\\audio_clean.wav")

    def test_build_ffmpeg_finalize_commands_can_opt_into_legacy_breath_filters(self) -> None:
        preset = apply_runtime_overrides(
            load_preset("voice-isolate"),
            enable_legacy_breath_filters=True,
        )
        commands = build_ffmpeg_finalize_commands(
            denoised_wav=Path("D:/out/audio_df.wav"),
            clean_wav=Path("D:/out/audio_clean.wav"),
            transcript_mp3=Path("D:/out/transcript_ready/audio.mp3"),
            preset=preset,
            ffmpeg_bin="ffmpeg",
            noise_sample_wav=Path("D:/out/audio_noise_sample.wav"),
            noise_sample_duration=8.5,
        )
        filter_complex_index = commands[0].index("-filter_complex") + 1
        filter_complex = commands[0][filter_complex_index]
        self.assertIn("sidechaingate=", filter_complex)
        self.assertIn("asplit=2", filter_complex)

    def test_build_mastering_filter_chain_omits_invalid_zero_makeup(self) -> None:
        preset = load_preset("safe")
        filter_chain = build_mastering_filter_chain(preset)
        self.assertIn("agate=", filter_chain)
        self.assertNotIn("makeup=0", filter_chain)

    def test_repair_filter_chain_contains_declick_and_deesser(self) -> None:
        preset = load_preset("repair")
        filter_chain = build_mastering_filter_chain(preset)
        self.assertIn("adeclick=", filter_chain)
        self.assertIn("deesser=", filter_chain)
        self.assertIn("speechnorm=", filter_chain)

    def test_repair_soft_filter_chain_keeps_declick_but_relaxes_processing(self) -> None:
        preset = load_preset("repair-soft")
        filter_chain = build_mastering_filter_chain(preset)
        self.assertIn("adeclick=", filter_chain)
        self.assertIn("deesser=", filter_chain)
        self.assertNotIn("speechnorm=", filter_chain)

    def test_final_filter_chain_keeps_deesser_without_declick_lowpass_or_speechnorm(self) -> None:
        preset = load_preset("final")
        filter_chain = build_mastering_filter_chain(preset)
        self.assertNotIn("adeclick=", filter_chain)
        self.assertIn("deesser=", filter_chain)
        self.assertIn("equalizer=f=3200", filter_chain)
        self.assertIn("equalizer=f=7500", filter_chain)
        self.assertNotIn("lowpass=", filter_chain)
        self.assertNotIn("agate=", filter_chain)
        self.assertIn("afftdn=nr=5:nf=-55:tn=1", filter_chain)
        self.assertNotIn("speechnorm=", filter_chain)

    def test_process_media_file_dry_run_reports_standalone_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "voice.wav"
            source.write_bytes(b"fake")
            preset = load_preset("safe")
            runtime = RuntimeOptions(dry_run=True)
            report = process_media_file(
                source,
                preset_name="safe",
                preset=preset,
                output_root=build_output_root(root / "output", "20260527-120000"),
                runtime=runtime,
                run_slug="20260527-120000",
                input_metadata={"duration_seconds": 10.0, "sample_rate": 48000, "channels": 1},
            )

            self.assertTrue(report["dry_run"])
            self.assertEqual(report["backend"], "local")
            self.assertEqual(len(report["commands"]), 4)
            self.assertIn("Breath detection via Respiro-en", report["processing_steps"])
            self.assertIn("SpectraMini-style breath control", report["processing_steps"])
            self.assertIn("Primary denoise via DeepFilterNet", report["processing_steps"])
            self.assertTrue(report["outputs"]["clean_wav"].endswith("voice_clean.wav"))

    def test_process_media_file_dry_run_natural_omits_automatic_voice_modification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "voice.wav"
            source.write_bytes(b"fake")
            preset = load_preset("natural")
            report = process_media_file(
                source,
                preset_name="natural",
                preset=preset,
                output_root=build_output_root(root / "output", "20260527-120000"),
                runtime=RuntimeOptions(dry_run=True),
                run_slug="20260527-120000",
                input_metadata={"duration_seconds": 10.0, "sample_rate": 48000, "channels": 1},
            )

            self.assertEqual(len(report["commands"]), 3)
            self.assertNotIn("Breath detection via Respiro-en", report["processing_steps"])
            self.assertNotIn("SpectraMini-style breath control", report["processing_steps"])
            self.assertNotIn("Primary denoise via DeepFilterNet", report["processing_steps"])
            self.assertNotIn("Secondary FFmpeg denoise via afftdn", report["processing_steps"])
            self.assertIn("Voice compression via acompressor", report["processing_steps"])
            self.assertIn("Loudness normalization via loudnorm", report["processing_steps"])
            self.assertEqual(
                report["stage_status"],
                {
                    "respiro": {"enabled": False, "applied": False},
                    "spectramini": {"enabled": False, "applied": False},
                    "deepfilternet": {"enabled": False, "applied": False},
                },
            )

    def test_process_media_file_dry_run_with_noise_windows_adds_noise_sample_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "voice.wav"
            source.write_bytes(b"fake")
            preset = load_preset("voice-isolate")
            runtime = RuntimeOptions(dry_run=True)
            report = process_media_file(
                source,
                preset_name="voice-isolate",
                preset=preset,
                output_root=build_output_root(root / "output", "20260527-120000"),
                runtime=runtime,
                run_slug="20260527-120000",
                input_metadata={"duration_seconds": 10.0, "sample_rate": 48000, "channels": 1},
                noise_windows=[NoiseWindow(start_seconds=1.0, end_seconds=2.5)],
            )

            self.assertTrue(report["dry_run"])
            self.assertEqual(len(report["commands"]), 5)
            self.assertEqual(report["noise_sample_duration_seconds"], 1.5)
            self.assertEqual(report["noise_windows"][0]["start_seconds"], 1.0)
            self.assertIn("Breath detection via Respiro-en", report["processing_steps"])
            self.assertIn("SpectraMini-style breath control", report["processing_steps"])
            self.assertIn("Capture noise sample from selected windows", report["processing_steps"])
            self.assertIn("Noise-print denoise via afftdn sample capture", report["processing_steps"])
            self.assertNotIn("Speech-presence breath ducking via sidechaingate", report["processing_steps"])
            self.assertNotIn("Breath-onset cleanup before speech entries", report["processing_steps"])
            self.assertNotIn("Secondary FFmpeg denoise via afftdn", report["processing_steps"])

    def test_process_media_file_dry_run_can_opt_into_legacy_breath_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "voice.wav"
            source.write_bytes(b"fake")
            preset = apply_runtime_overrides(
                load_preset("voice-isolate"),
                enable_legacy_breath_filters=True,
            )
            runtime = RuntimeOptions(dry_run=True)
            report = process_media_file(
                source,
                preset_name="voice-isolate",
                preset=preset,
                output_root=build_output_root(root / "output", "20260527-120000"),
                runtime=runtime,
                run_slug="20260527-120000",
                input_metadata={"duration_seconds": 10.0, "sample_rate": 48000, "channels": 1},
                noise_windows=[NoiseWindow(start_seconds=1.0, end_seconds=2.5)],
            )

            self.assertIn("Speech-presence breath ducking via sidechaingate", report["processing_steps"])
            self.assertIn("Breath-onset cleanup before speech entries", report["processing_steps"])

    def test_process_media_file_executes_noise_sample_after_df_output_is_placed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "voice.wav"
            source.write_bytes(b"fake")
            output_root = build_output_root(root / "output", "20260527-120000")
            preset = load_preset("voice-isolate")
            layout = build_output_layout(
                input_path=source,
                output_root=output_root,
                run_slug="20260527-120000",
            )
            df_out = layout.deepfilternet_dir / "audio_raw.wav"

            def write_fake_wav(path: Path) -> None:
                path.parent.mkdir(parents=True, exist_ok=True)
                with wave.open(str(path), "wb") as writer:
                    writer.setnchannels(1)
                    writer.setsampwidth(2)
                    writer.setframerate(48000)
                    writer.writeframes(b"\x00\x00" * 4800)

            def fake_run(command: list[str]):
                cmd_text = " ".join(command)
                if str(layout.noise_sample_wav) in cmd_text:
                    self.assertTrue(layout.denoised_wav.exists())
                if str(layout.deepfilternet_dir) in cmd_text:
                    layout.deepfilternet_dir.mkdir(parents=True, exist_ok=True)
                    write_fake_wav(df_out)
                elif str(layout.raw_wav) in cmd_text:
                    write_fake_wav(layout.raw_wav)
                elif str(layout.noise_sample_wav) in cmd_text:
                    write_fake_wav(layout.noise_sample_wav)
                elif str(layout.clean_wav) in cmd_text:
                    write_fake_wav(layout.clean_wav)
                elif str(layout.transcript_mp3) in cmd_text:
                    layout.transcript_dir.mkdir(parents=True, exist_ok=True)
                    layout.transcript_mp3.write_bytes(b"mp3")

                class Result:
                    returncode = 0
                    stdout = ""
                    stderr = ""

                return Result()

            from unittest import mock

            with mock.patch("audio_sound.pipeline.ensure_tool", side_effect=lambda value: value), mock.patch(
                "audio_sound.pipeline.run_command",
                side_effect=fake_run,
            ), mock.patch(
                "audio_sound.pipeline.ffprobe_media",
                return_value={"duration_seconds": 10.0, "sample_rate": 48000, "channels": 1},
            ):
                report = process_media_file(
                    source,
                    preset_name="voice-isolate",
                    preset=preset,
                    output_root=output_root,
                    runtime=RuntimeOptions(ffmpeg_bin="ffmpeg", ffprobe_bin="ffprobe", python_executable="python"),
                    run_slug="20260527-120000",
                    input_metadata={"duration_seconds": 10.0, "sample_rate": 48000, "channels": 1},
                    noise_windows=[NoiseWindow(start_seconds=1.0, end_seconds=2.5)],
                )

            self.assertTrue(layout.denoised_wav.exists())
            self.assertEqual(report["noise_sample_duration_seconds"], 1.5)
            self.assertEqual(report["quality_guard"]["status"], "PASS")
            self.assertNotIn("_frame_rms_db", report["input_diagnostics"])
            self.assertNotIn("_frame_rms_db", report["output_diagnostics"])

    def test_build_batch_summary_renders_markdown(self) -> None:
        summary = build_batch_summary(
            preset_name="safe",
            preset_description="spoken word",
            input_path=Path("D:/batch"),
            output_root=Path("D:/out"),
            reports=[
                {
                    "input_file": "D:/batch/a.wav",
                    "outputs": {
                        "clean_wav": "D:/out/20260527-120000_a/audio_preprocess/audio_clean.wav",
                        "transcript_mp3": "D:/out/20260527-120000_a/transcript_ready/audio.mp3",
                        "report_md": "D:/out/20260527-120000_a/audio_preprocess/audio_process_report.md",
                        "report_json": "D:/out/20260527-120000_a/audio_preprocess/audio_process_report.json",
                    },
                }
            ],
        )
        markdown = render_batch_summary_markdown(summary)
        self.assertEqual(summary["files_processed"], 1)
        self.assertIn("audio_clean.wav", markdown)
        self.assertIn("spoken word", markdown)


if __name__ == "__main__":
    unittest.main()
