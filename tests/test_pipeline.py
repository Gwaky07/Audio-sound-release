from __future__ import annotations

import tempfile
import unittest
import wave
from hashlib import sha1
from pathlib import Path

from audio_sound.config import load_preset
from audio_sound.pipeline import (
    NoiseWindow,
    RuntimeOptions,
    build_mastering_filter_chain,
    build_batch_summary,
    build_deepfilternet_command,
    build_ffmpeg_extract_command,
    build_ffmpeg_finalize_commands,
    build_ffmpeg_noise_sample_command,
    build_output_layout,
    build_output_root,
    duck_samples_for_windows,
    discover_media_files,
    infer_breath_window_from_frame_rms,
    infer_breath_onset_windows,
    parse_noise_window,
    process_media_file,
    render_batch_summary_markdown,
)


class PipelineTests(unittest.TestCase):
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

    def test_build_output_layout_uses_ascii_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir) / "output"
            layout = build_output_layout(
                input_path=Path("D:/media/My Clip 01.mp4"),
                output_root=output_root,
                run_slug="20260527-120000",
            )

            self.assertEqual(layout.job_name, "20260527-120000_my_clip_01")
            self.assertTrue(str(layout.preprocess_dir).endswith("audio_preprocess"))
            self.assertTrue(str(layout.transcript_dir).endswith("transcript_ready"))
            self.assertEqual(layout.raw_wav.name, "audio_raw.wav")
            self.assertEqual(layout.denoised_wav.name, "audio_df.wav")
            self.assertEqual(layout.noise_sample_wav.name, "audio_noise_sample.wav")
            self.assertEqual(layout.clean_wav.name, "audio_clean.wav")
            self.assertEqual(layout.report_json.name, "audio_process_report.json")

    def test_build_output_layout_uses_hashed_ascii_fallback_for_non_ascii_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir) / "output"
            source = Path("C:/Users/Guanghe/Downloads/渡荆门送别精讲.wav")
            layout = build_output_layout(
                input_path=source,
                output_root=output_root,
                run_slug="20260527-120000",
            )

            expected_suffix = sha1(source.stem.encode("utf-8")).hexdigest()[:8]
            self.assertEqual(layout.job_name, f"20260527-120000_media_{expected_suffix}")

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
        self.assertIn("loudnorm=I=-16.0:LRA=7.0:TP=-1.5:print_format=json", commands[0][7])
        self.assertEqual(commands[0][8:10], ["-ar", "48000"])
        self.assertEqual(commands[1][-2:], ["192k", "D:\\out\\transcript_ready\\audio.mp3"])

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
        self.assertIn("sidechaingate=", filter_complex)
        self.assertIn("asplit=2", filter_complex)
        self.assertEqual(filter_complex.count("afftdn="), 1)
        self.assertEqual(filter_complex.count("loudnorm="), 1)
        self.assertEqual(filter_complex.count("adeclick="), 1)
        self.assertEqual(filter_complex.count("sidechaingate="), 1)
        self.assertIn("-map", commands[0])
        self.assertEqual(commands[0][-1], "D:\\out\\audio_clean.wav")

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
            self.assertIn("Primary denoise via DeepFilterNet", report["processing_steps"])
            self.assertTrue(report["outputs"]["clean_wav"].endswith("audio_clean.wav"))

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
            self.assertIn("Capture noise sample from selected windows", report["processing_steps"])
            self.assertIn("Noise-print denoise via afftdn sample capture", report["processing_steps"])
            self.assertIn("Speech-presence breath ducking via sidechaingate", report["processing_steps"])
            self.assertIn("Breath-onset cleanup before speech entries", report["processing_steps"])
            self.assertNotIn("Secondary FFmpeg denoise via afftdn", report["processing_steps"])

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
