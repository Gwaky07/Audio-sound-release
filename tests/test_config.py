from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from audio_sound.config import (
    PACKAGE_PRESETS_DIR,
    REPOSITORY_PRESETS_DIR,
    apply_runtime_overrides,
    list_presets,
    load_env_file,
    load_preset,
    resolve_repo_python,
)


class ConfigTests(unittest.TestCase):
    def test_packaged_presets_match_repository_presets(self) -> None:
        repository_files = {
            path.name: path.read_bytes()
            for path in REPOSITORY_PRESETS_DIR.glob("*.json")
        }
        packaged_files = {
            path.name: path.read_bytes()
            for path in PACKAGE_PRESETS_DIR.glob("*.json")
        }
        self.assertTrue(repository_files)
        self.assertEqual(packaged_files, repository_files)

    def test_resolve_repo_python_can_require_repository_venv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            expected = Path(tmp_dir) / ".venv" / "Scripts" / "python.exe"
            self.assertEqual(
                resolve_repo_python(tmp_dir, require_venv=True),
                str(expected),
            )

    def test_list_presets(self) -> None:
        self.assertEqual(
            set(list_presets()),
            {
                "fast",
                "safe",
                "review",
                "repair",
                "repair-soft",
                "final",
                "natural",
                "voice-isolate",
                "clarity-leveling-safe",
                "noise-cleanup-safe",
                "respiro-breath-safe",
                "deepfilter-denoise-safe",
                "model-combined-review",
            },
        )

    def test_safe_auto_presets_forbid_destructive_cleanup(self) -> None:
        for preset_name in ("clarity-leveling-safe", "noise-cleanup-safe"):
            preset = load_preset(preset_name)
            enabled_stages = [
                stage["type"]
                for stage in preset["pipeline"]["stages"]
                if stage.get("enabled", True)
            ]
            self.assertEqual(enabled_stages, [])
            self.assertIsNone(preset["filters"]["lowpass_hz"])
            self.assertFalse(preset["filters"]["gate"]["enabled"])
            self.assertFalse(preset["filters"]["breath_onset_cleanup"].get("enabled", False))
            self.assertFalse(preset["filters"]["pause_residual_cleanup"].get("enabled", False))

    def test_natural_preset_prioritizes_unmodified_voice(self) -> None:
        preset = load_preset("natural")
        enabled_stages = [
            stage["type"]
            for stage in preset["pipeline"]["stages"]
            if stage.get("enabled", True)
        ]

        self.assertEqual(enabled_stages, [])
        self.assertFalse(preset["filters"]["secondary_denoise"]["enabled"])
        self.assertFalse(preset["filters"]["gate"]["enabled"])
        self.assertTrue(preset["filters"]["compressor"]["enabled"])
        self.assertEqual(preset["filters"]["compressor"]["ratio"], 1.25)

    def test_final_preset_avoids_global_click_processing_and_excessive_air_boost(self) -> None:
        preset = load_preset("final")
        spectramini = next(
            stage
            for stage in preset["pipeline"]["stages"]
            if stage["type"] == "spectramini"
        )
        bands = {
            band["frequency_hz"]: band["gain_db"]
            for band in preset["filters"]["equalizer"]["bands"]
        }

        self.assertEqual(spectramini["mouth_declick_sensitivity"], 0.0)
        self.assertIsNone(preset["filters"]["lowpass_hz"])
        self.assertLessEqual(bands[3200], 0.8)
        self.assertLessEqual(bands[7500], 0.2)
        policy = preset["filters"]["breath_cleanup_policy"]
        self.assertTrue(policy["enabled"])
        self.assertEqual(policy["first_pass_max_attenuation_db"], 24.0)
        self.assertEqual(policy["second_pass_max_attenuation_db"], 12.0)
        self.assertEqual(policy["max_retries"], 2)
        self.assertTrue(policy["final_residual_check"])
        pause_policy = preset["filters"]["pause_residual_cleanup"]
        self.assertTrue(pause_policy["enabled"])
        self.assertEqual(pause_policy["silence_floor_dbfs"], -96.0)
        self.assertFalse(pause_policy["allow_bridge_windows"])
        self.assertEqual(pause_policy["target_margin_db"], -6.0)
        self.assertEqual(pause_policy["core_pad_ms"], 45.0)
        self.assertEqual(pause_policy["leading_trailing_pad_ms"], 25.0)
        self.assertEqual(pause_policy["max_attenuation_db"], 48.0)
        self.assertEqual(pause_policy["second_pass_max_attenuation_db"], 24.0)
        self.assertEqual(pause_policy["final_pass_max_attenuation_db"], 36.0)
        self.assertEqual(pause_policy["fade_ms"], 12.0)
        self.assertEqual(pause_policy["final_fade_ms"], 6.0)
        self.assertEqual(pause_policy["residual_assessment_edge_ms"], 6.0)
        self.assertEqual(pause_policy["residual_min_excess_db"], 3.0)
        self.assertTrue(pause_policy["block_on_confirmed_residual"])

    def test_load_preset_returns_structured_sections(self) -> None:
        preset = load_preset("safe")
        self.assertEqual(preset["extract"]["sample_rate"], 48000)
        self.assertEqual(preset["pipeline"]["stages"][0]["type"], "respiro")
        self.assertEqual(preset["filters"]["loudnorm"]["target_i"], -20.0)
        self.assertEqual(preset["filters"]["loudnorm"]["target_tp"], -9.0)
        self.assertTrue(preset["filters"]["pause_residual_cleanup"]["enabled"])
        self.assertEqual(preset["transcript_export"]["codec"], "libmp3lame")

    def test_apply_runtime_overrides_updates_nested_values(self) -> None:
        preset = apply_runtime_overrides(
            load_preset("fast"),
            target_lufs=-14.0,
            denoise_strength="aggressive",
            disable_gate=True,
            enable_silence_report=True,
        )
        self.assertEqual(preset["filters"]["loudnorm"]["target_i"], -14.0)
        self.assertEqual(preset["filters"]["secondary_denoise"]["nr"], 16)
        self.assertFalse(preset["filters"]["gate"]["enabled"])
        self.assertTrue(preset["analysis"]["silence_candidates"])

    def test_apply_runtime_overrides_can_opt_into_legacy_breath_filters(self) -> None:
        preset = apply_runtime_overrides(
            load_preset("voice-isolate"),
            enable_legacy_breath_filters=True,
        )
        self.assertTrue(preset["filters"]["breath_ducking"]["enabled"])
        self.assertTrue(preset["filters"]["breath_onset_cleanup"]["enabled"])

    def test_load_env_file_parses_simple_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "# comment",
                        "AUDIO_SOUND_FFMPEG=ffmpeg-custom",
                        "AUDIO_SOUND_PYTHON='python-custom'",
                        "AUDIO_SOUND_RESPIRO_REPO='D:/models/Respiro-en'",
                        "AUDIO_SOUND_RESPIRO_WEIGHTS='D:/models/respiro-en.pt'",
                    ]
                ),
                encoding="utf-8",
            )
            values = load_env_file(env_path)
        self.assertEqual(values["AUDIO_SOUND_FFMPEG"], "ffmpeg-custom")
        self.assertEqual(values["AUDIO_SOUND_PYTHON"], "python-custom")
        self.assertEqual(values["AUDIO_SOUND_RESPIRO_REPO"], "D:/models/Respiro-en")
        self.assertEqual(values["AUDIO_SOUND_RESPIRO_WEIGHTS"], "D:/models/respiro-en.pt")

    def test_preset_files_remain_json_serializable(self) -> None:
        preset = load_preset("review")
        json.dumps(preset)

    def test_repair_preset_enables_click_and_breath_controls(self) -> None:
        preset = load_preset("repair")
        self.assertTrue(preset["filters"]["declick"]["enabled"])
        self.assertTrue(preset["filters"]["deesser"]["enabled"])
        self.assertGreater(preset["filters"]["gate"]["threshold"], load_preset("safe")["filters"]["gate"]["threshold"])

    def test_repair_soft_is_less_aggressive_than_repair(self) -> None:
        soft = load_preset("repair-soft")
        hard = load_preset("repair")
        self.assertTrue(soft["filters"]["declick"]["enabled"])
        self.assertLess(soft["filters"]["gate"]["threshold"], hard["filters"]["gate"]["threshold"])
        self.assertLess(soft["filters"]["secondary_denoise"]["nr"], hard["filters"]["secondary_denoise"]["nr"])
        self.assertLess(soft["filters"]["deesser"]["intensity"], hard["filters"]["deesser"]["intensity"])

    def test_final_preset_is_complete_but_safer_than_repair_presets(self) -> None:
        final = load_preset("final")
        soft = load_preset("repair-soft")
        self.assertFalse(final["filters"]["declick"]["enabled"])
        self.assertTrue(final["filters"]["equalizer"]["enabled"])
        self.assertTrue(final["filters"]["secondary_denoise"]["enabled"])
        self.assertLess(final["filters"]["secondary_denoise"]["nr"], soft["filters"]["secondary_denoise"]["nr"])
        self.assertFalse(final["filters"]["gate"]["enabled"])
        self.assertIsNone(final["filters"]["lowpass_hz"])
        self.assertFalse(final["pipeline"]["stages"][2]["enabled"])


if __name__ == "__main__":
    unittest.main()
