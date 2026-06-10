from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from audio_sound.config import apply_runtime_overrides, list_presets, load_env_file, load_preset


class ConfigTests(unittest.TestCase):
    def test_list_presets(self) -> None:
        self.assertEqual(set(list_presets()), {"fast", "safe", "review", "repair", "repair-soft", "final", "voice-isolate"})

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

    def test_final_preset_sits_between_repair_and_repair_soft(self) -> None:
        final = load_preset("final")
        soft = load_preset("repair-soft")
        hard = load_preset("repair")
        self.assertTrue(final["filters"]["declick"]["enabled"])
        self.assertGreater(final["filters"]["secondary_denoise"]["nr"], soft["filters"]["secondary_denoise"]["nr"])
        self.assertLess(final["filters"]["secondary_denoise"]["nr"], hard["filters"]["secondary_denoise"]["nr"])
        self.assertGreater(final["filters"]["gate"]["threshold"], soft["filters"]["gate"]["threshold"])
        self.assertLess(final["filters"]["gate"]["threshold"], hard["filters"]["gate"]["threshold"])


if __name__ == "__main__":
    unittest.main()
