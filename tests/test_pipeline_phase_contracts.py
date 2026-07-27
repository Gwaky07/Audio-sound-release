from __future__ import annotations

import inspect
import tempfile
import unittest
import wave
from array import array
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from audio_sound.pipeline import (
    NoiseWindow,
    OutputLayout,
    RuntimeOptions,
    _run_breath_residual_cleanup,
    _run_pause_cleanup,
    process_media_file,
)


def _write_pcm16_wav(path: Path, *, sample_rate: int = 1000, channels: int = 1) -> None:
    samples = array("h", [0] * sample_rate * channels)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(samples.tobytes())


def _layout(root: Path) -> OutputLayout:
    preprocess = root / "audio_preprocess"
    transcript = root / "transcript_ready"
    preprocess.mkdir(parents=True, exist_ok=True)
    transcript.mkdir(parents=True, exist_ok=True)
    clean_wav = preprocess / "clean.wav"
    _write_pcm16_wav(clean_wav)
    return OutputLayout(
        job_name="phase",
        job_dir=root,
        preprocess_dir=preprocess,
        transcript_dir=transcript,
        raw_wav=preprocess / "raw.wav",
        breath_wav=preprocess / "breath.wav",
        denoised_wav=preprocess / "denoised.wav",
        noise_sample_wav=preprocess / "noise.wav",
        clean_wav=clean_wav,
        transcript_mp3=transcript / "clean.mp3",
        report_json=preprocess / "report.json",
        report_md=preprocess / "report.md",
        deepfilternet_dir=preprocess / "deepfilternet",
    )


class PipelinePhaseContractTests(unittest.TestCase):
    def test_process_media_file_reads_legacy_breath_onset_from_original_preset(self) -> None:
        source = inspect.getsource(process_media_file)
        self.assertIn(
            'breath_onset_cleanup = preset.get("filters", {}).get("breath_onset_cleanup", {})',
            source,
        )
        self.assertNotIn(
            'breath_onset_cleanup = active_preset.get("filters", {}).get("breath_onset_cleanup"',
            source,
        )

    def test_pause_cleanup_fails_closed_when_assessment_windows_are_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            layout = _layout(Path(tmp_dir))
            report = {
                "pause_cleanup": {
                    "status": "NOT_APPLICABLE",
                    "mode": None,
                    "target_dbfs": None,
                    "silence_floor_dbfs": None,
                    "window_evidence": [],
                    "first_pass": [],
                    "second_pass": [],
                    "final_residual_windows": [],
                    "attenuation_stats": {},
                    "failures": [],
                },
                "pause_residual_cleanup_windows": [],
            }
            active_preset = {
                "filters": {
                    "pause_residual_cleanup": {
                        "enabled": True,
                        "silence_threshold_db": -50.0,
                        "silence_min_duration": 0.12,
                        "min_neighbor_silence_duration": 0.2,
                        "bridge_max_duration": 0.1,
                        "bridge_peak_db": -50.0,
                        "bridge_rms_db": -55.0,
                        "core_pad_ms": 45.0,
                        "leading_trailing_pad_ms": 25.0,
                        "allow_bridge_windows": False,
                        "silence_floor_dbfs": -96.0,
                        "target_margin_db": -6.0,
                        "max_attenuation_db": 48.0,
                        "second_pass_max_attenuation_db": 24.0,
                        "final_pass_max_attenuation_db": 36.0,
                        "fade_ms": 12.0,
                        "final_fade_ms": 6.0,
                        "residual_assessment_edge_ms": 6.0,
                        "residual_min_excess_db": 3.0,
                        "block_on_confirmed_residual": True,
                    }
                }
            }
            pause_window = NoiseWindow(0.2, 0.8)
            residual_detail = {
                "start_seconds": 0.2,
                "end_seconds": 0.8,
                "requested_attenuation_db": 12.0,
            }
            cleanup_calls = [
                (array("h", [1, 2, 3]), [residual_detail]),
                (None, [residual_detail]),
                (array("h", [1, 2, 3]), [residual_detail]),
                (None, [residual_detail]),
                (array("h", [1, 2, 3]), [residual_detail]),
            ]

            with (
                mock.patch(
                    "audio_sound.pipeline.detect_silence_candidates",
                    return_value=[
                        {
                            "start_seconds": 0.0,
                            "end_seconds": 1.0,
                            "duration_seconds": 1.0,
                        }
                    ],
                ),
                mock.patch(
                    "audio_sound.pipeline.infer_pause_residual_cleanup_windows",
                    return_value=[pause_window],
                ),
                mock.patch(
                    "audio_sound.pipeline.build_effective_breath_windows",
                    return_value=([pause_window], [{"decision": "accepted"}]),
                ),
                mock.patch(
                    "audio_sound.pipeline.apply_adaptive_breath_cleanup",
                    side_effect=cleanup_calls,
                ),
                mock.patch(
                    "audio_sound.pipeline.trim_noise_windows_for_assessment",
                    return_value=[],
                ),
                mock.patch(
                    "audio_sound.pipeline._write_wave_samples",
                ),
                mock.patch(
                    "audio_sound.pipeline._run_recorded_command",
                ),
            ):
                _run_pause_cleanup(
                    layout=layout,
                    active_preset=active_preset,
                    input_analysis={
                        "noise_floor_dbfs": -70.0,
                        "_frame_rms_db": [-70.0] * 100,
                        "frame_ms": 10.0,
                        "speech_level_dbfs": -18.0,
                    },
                    runtime=RuntimeOptions(ffmpeg_bin="ffmpeg"),
                    report=report,
                    authorized_cleanup_windows=[],
                    commands=[["ffmpeg", "mp3"]],
                    executed=[],
                )

            self.assertEqual(report["pause_cleanup"]["status"], "FAIL")
            self.assertIn(
                "empty_assessment_windows",
                report["pause_cleanup"]["failures"],
            )
            self.assertIn(
                "confirmed_pause_residual_after_cleanup",
                report["pause_cleanup"]["failures"],
            )
            self.assertEqual(report["pause_cleanup"]["mode"], "speech_safe_autogate")

    def test_breath_residual_cleanup_fails_closed_after_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            layout = _layout(Path(tmp_dir))
            report = {
                "breath_cleanup": {
                    "status": "NOT_APPLICABLE",
                    "first_pass": [],
                    "second_pass": [],
                    "final_pass": [],
                    "final_residual_windows": [],
                    "failures": [],
                }
            }
            residual = NoiseWindow(0.1, 0.2)
            residual_detail = {
                "start_seconds": 0.1,
                "end_seconds": 0.2,
                "requested_attenuation_db": 8.0,
            }
            params = SimpleNamespace(framerate=1000, nchannels=1)
            samples = array("h", [0] * 1000)

            with (
                mock.patch(
                    "audio_sound.pipeline.detect_breath_onset_windows",
                    return_value=[residual],
                ),
                mock.patch(
                    "audio_sound.pipeline._load_wave_samples",
                    return_value=(params, samples),
                ),
                mock.patch(
                    "audio_sound.pipeline.filter_noise_like_breath_windows",
                    return_value=([residual], [{"noise_like": True}]),
                ),
                mock.patch(
                    "audio_sound.pipeline.build_effective_breath_windows",
                    return_value=([residual], [{"decision": "accepted"}]),
                ),
                mock.patch(
                    "audio_sound.pipeline.apply_adaptive_breath_cleanup",
                    side_effect=[
                        (samples, [residual_detail]),
                        (None, [residual_detail]),
                        (samples, [residual_detail]),
                        (None, [residual_detail]),
                    ],
                ),
                mock.patch("audio_sound.pipeline._write_wave_samples"),
                mock.patch("audio_sound.pipeline._run_recorded_command"),
            ):
                authorized = _run_breath_residual_cleanup(
                    breath_windows=[residual],
                    breath_cleanup_policy={
                        "enabled": True,
                        "second_pass_max_attenuation_db": 12.0,
                        "target_margin_db": -6.0,
                        "context_ms": 80.0,
                        "fade_ms": 12.0,
                        "residual_min_excess_db": 3.0,
                        "max_retries": 2,
                        "block_on_confirmed_residual": True,
                    },
                    layout=layout,
                    runtime=RuntimeOptions(ffmpeg_bin="ffmpeg"),
                    fallback_config={},
                    input_analysis={
                        "_frame_rms_db": [-70.0] * 100,
                        "frame_ms": 10.0,
                        "speech_level_dbfs": -18.0,
                    },
                    report=report,
                    commands=[["ffmpeg", "mp3"]],
                    executed=[],
                )

            self.assertEqual(report["breath_cleanup"]["status"], "FAIL")
            self.assertEqual(
                report["breath_cleanup"]["failures"],
                ["confirmed_breath_residual_after_retry"],
            )
            self.assertIn(residual, authorized)


if __name__ == "__main__":
    unittest.main()
