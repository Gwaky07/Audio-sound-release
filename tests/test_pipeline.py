from __future__ import annotations

import tempfile
import unittest
import wave
from hashlib import sha1
from pathlib import Path
from unittest import mock

from audio_sound.config import apply_runtime_overrides, load_preset
from audio_sound.pipeline import (
    NoiseWindow,
    RuntimeOptions,
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
    infer_pause_residual_cleanup_windows,
    measure_segment_levels,
    resolve_respiro_runtime,
    run_respiro_or_fallback_detection,
    duck_samples_for_windows,
    discover_media_files,
    infer_breath_window_from_frame_rms,
    infer_breath_onset_windows,
    parse_noise_window,
    process_media_file,
    render_batch_summary_markdown,
)


class PipelineTests(unittest.TestCase):
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
        )

        self.assertEqual(
            windows,
            [
                NoiseWindow(start_seconds=0.1, end_seconds=1.7),
                NoiseWindow(start_seconds=1.8, end_seconds=2.0),
                NoiseWindow(start_seconds=2.1, end_seconds=3.9),
            ],
        )

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

    def test_final_filter_chain_keeps_declick_and_deesser_without_speechnorm(self) -> None:
        preset = load_preset("final")
        filter_chain = build_mastering_filter_chain(preset)
        self.assertIn("adeclick=", filter_chain)
        self.assertIn("deesser=", filter_chain)
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
