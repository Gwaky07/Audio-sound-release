from __future__ import annotations

import unittest

from audio_sound.auto_workflow import (
    AUTO_CANDIDATE_RECIPES,
    build_asr_preservation_guard,
    build_candidate_specs,
    build_evaluated_candidate,
    select_auto_candidate,
    validate_auto_recipe,
)
from audio_sound.config import list_presets


class AutoWorkflowTests(unittest.TestCase):
    def test_auto_recipes_forbid_destructive_cleanup(self) -> None:
        for recipe in AUTO_CANDIDATE_RECIPES.values():
            validate_auto_recipe(recipe)
            self.assertIsNone(recipe.get("lowpass_hz"))
            self.assertFalse(recipe.get("gate_enabled"))
            self.assertFalse(recipe.get("hard_mute_enabled"))
            if recipe.get("deepfilternet_enabled") or recipe.get("respiro_enabled"):
                self.assertTrue(recipe.get("model_candidate"))

    def test_auto_recipe_presets_exist_on_disk(self) -> None:
        available = set(list_presets())
        for candidate_id, recipe in AUTO_CANDIDATE_RECIPES.items():
            preset_name = str(recipe["preset_name"])
            self.assertIn(
                preset_name,
                available,
                msg=f"{candidate_id} references missing preset {preset_name}",
            )

    def test_candidate_specs_only_add_noise_cleanup_with_verified_noise_windows(self) -> None:
        baseline_only = build_candidate_specs(
            {
                "stationary_noise": False,
                "candidate_noise_windows": [],
                "active_dynamic_range_db": 4.0,
            }
        )
        with_noise = build_candidate_specs(
            {
                "stationary_noise": True,
                "candidate_noise_windows": [
                    {"start_seconds": 2.0, "end_seconds": 2.8},
                ],
                "active_dynamic_range_db": 4.0,
            }
        )

        self.assertEqual(
            [item.candidate_id for item in baseline_only],
            ["natural_baseline", "final_repair_best", "clarity_leveling_safe"],
        )
        self.assertEqual(
            [item.candidate_id for item in with_noise],
            [
                "natural_baseline",
                "final_repair_best",
                "clarity_leveling_safe",
                "noise_cleanup_safe",
            ],
        )
        self.assertEqual(with_noise[-1].noise_windows, ("2.0:2.8",))

    def test_repair_intent_includes_final_repair_best_without_asr(self) -> None:
        specs = build_candidate_specs(
            {
                "stationary_noise": False,
                "candidate_noise_windows": [],
                "pause_ratio": 0.1,
            },
            repair_intent=True,
        )
        final_spec = next(
            item for item in specs if item.candidate_id == "final_repair_best"
        )
        self.assertEqual(final_spec.preset_name, "final")
        self.assertFalse(final_spec.requires_asr)

    def test_final_repair_best_outscores_natural_when_both_pass(self) -> None:
        selected = select_auto_candidate(
            [
                {
                    "candidate_id": "natural_baseline",
                    "quality_guard": {"release_blocked": False},
                    "asr_guard": {"status": "NOT_REQUIRED", "release_blocked": False},
                    "score": 80.0,
                },
                {
                    "candidate_id": "final_repair_best",
                    "quality_guard": {"release_blocked": False},
                    "asr_guard": {"status": "NOT_REQUIRED", "release_blocked": False},
                    "score": 90.0,
                },
            ],
            repair_intent=True,
        )
        self.assertEqual(selected["candidate_id"], "final_repair_best")
        self.assertEqual(selected["selection_reason"], "best_repair_selected")
        self.assertFalse(selected["delivery_incomplete_for_repair_intent"])

    def test_final_repair_best_priority_beats_higher_model_score(self) -> None:
        selected = select_auto_candidate(
            [
                {
                    "candidate_id": "final_repair_best",
                    "quality_guard": {"release_blocked": False},
                    "asr_guard": {"status": "NOT_REQUIRED", "release_blocked": False},
                    "score": 74.0,
                },
                {
                    "candidate_id": "respiro_breath_safe",
                    "quality_guard": {"release_blocked": False},
                    "asr_guard": {"status": "NOT_REQUIRED", "release_blocked": False},
                    "score": 90.0,
                },
            ],
            repair_intent=True,
        )
        self.assertEqual(selected["candidate_id"], "final_repair_best")
        self.assertEqual(selected["selection_reason"], "best_repair_selected")


    def test_natural_fallback_marks_repair_intent_incomplete(self) -> None:
        selected = select_auto_candidate(
            [
                {
                    "candidate_id": "natural_baseline",
                    "quality_guard": {"release_blocked": False},
                    "asr_guard": {"status": "NOT_REQUIRED", "release_blocked": False},
                    "score": 70.0,
                },
                {
                    "candidate_id": "final_repair_best",
                    "quality_guard": {
                        "release_blocked": True,
                        "failures": ["confirmed_breath_residual_after_retry"],
                    },
                    "asr_guard": {"status": "NOT_REQUIRED", "release_blocked": False},
                    "score": 95.0,
                },
            ],
            repair_intent=True,
        )
        self.assertEqual(selected["candidate_id"], "natural_baseline")
        self.assertTrue(selected["delivery_incomplete_for_repair_intent"])
        self.assertEqual(selected["selection_reason"], "safe_fallback_incomplete")

    def test_candidate_specs_add_available_model_candidates_from_diagnostics(self) -> None:
        specs = build_candidate_specs(
            {
                "stationary_noise": True,
                "candidate_noise_windows": [
                    {"start_seconds": 2.0, "end_seconds": 2.8},
                ],
                "pause_ratio": 0.2,
                "estimated_snr_db": 14.0,
            },
            runtime_capabilities={
                "respiro_en": {"ready": True},
                "deepfilternet": {"ready": True},
            },
        )

        self.assertIn("respiro_breath_safe", [item.candidate_id for item in specs])
        self.assertIn("deepfilter_denoise_safe", [item.candidate_id for item in specs])
        self.assertFalse(
            next(item for item in specs if item.candidate_id == "respiro_breath_safe").requires_asr
        )

    def test_candidate_specs_record_unavailable_model_reasons(self) -> None:
        specs = build_candidate_specs(
            {
                "stationary_noise": True,
                "candidate_noise_windows": [{"start_seconds": 2.0, "end_seconds": 2.8}],
                "pause_ratio": 0.2,
                "estimated_snr_db": 14.0,
            },
            runtime_capabilities={
                "respiro_en": {"ready": False},
                "deepfilternet": {"ready": False},
            },
        )

        unavailable = {
            item.candidate_id: item.unavailable_reason
            for item in specs
            if item.unavailable_reason
        }
        self.assertEqual(unavailable["respiro_breath_safe"], "runtime_unavailable")
        self.assertEqual(unavailable["deepfilter_denoise_safe"], "runtime_unavailable")

    def test_selector_falls_back_to_natural_when_enhancement_lacks_asr(self) -> None:
        selected = select_auto_candidate(
            [
                {
                    "candidate_id": "natural_baseline",
                    "quality_guard": {"release_blocked": False},
                    "asr_guard": {"status": "NOT_REQUIRED", "release_blocked": False},
                    "score": 70.0,
                },
                {
                    "candidate_id": "clarity_leveling_safe",
                    "quality_guard": {"release_blocked": False},
                    "asr_guard": {"status": "UNAVAILABLE", "release_blocked": True},
                    "score": 95.0,
                },
            ]
        )

        self.assertEqual(selected["candidate_id"], "natural_baseline")
        self.assertEqual(selected["selection_reason"], "safe_fallback_incomplete")
        self.assertTrue(selected["delivery_incomplete_for_repair_intent"])

    def test_hard_guard_cannot_be_overridden_by_score(self) -> None:
        selected = select_auto_candidate(
            [
                {
                    "candidate_id": "natural_baseline",
                    "quality_guard": {"release_blocked": False},
                    "asr_guard": {"status": "NOT_REQUIRED", "release_blocked": False},
                    "score": 50.0,
                },
                {
                    "candidate_id": "clarity_leveling_safe",
                    "quality_guard": {
                        "release_blocked": True,
                        "failures": ["spectral_clarity_lost"],
                    },
                    "asr_guard": {"status": "PASS", "release_blocked": False},
                    "score": 100.0,
                },
            ]
        )

        self.assertEqual(selected["candidate_id"], "natural_baseline")
        self.assertEqual(selected["rejected_candidates"][0]["candidate_id"], "clarity_leveling_safe")

    def test_asr_guard_blocks_new_character_omissions(self) -> None:
        source = [
            {
                "engine": "engine-a",
                "segments": [{"start": 0.0, "end": 2.0, "text": "密度计算等体积换材料问题"}],
            },
            {
                "engine": "engine-b",
                "segments": [{"start": 0.0, "end": 2.0, "text": "密度计算等体积换材料问题"}],
            },
        ]
        candidate = [
            {
                "engine": "engine-a",
                "segments": [{"start": 0.0, "end": 2.0, "text": "密度计算体积换材料问题"}],
            },
            {
                "engine": "engine-b",
                "segments": [{"start": 0.0, "end": 2.0, "text": "密度计算体积换材料问题"}],
            },
        ]

        guard = build_asr_preservation_guard(source, candidate)

        self.assertEqual(guard["status"], "FAIL")
        self.assertTrue(guard["release_blocked"])
        self.assertIn("asr_character_omission", guard["failures"])

    def test_asr_guard_requires_two_independent_engines(self) -> None:
        payload = [
            {
                "engine": "same-engine",
                "segments": [{"start": 0.0, "end": 1.0, "text": "完整语音"}],
            },
        ]

        guard = build_asr_preservation_guard(payload, payload)

        self.assertEqual(guard["status"], "UNAVAILABLE")
        self.assertTrue(guard["release_blocked"])
        self.assertIn("insufficient_independent_asr_engines", guard["failures"])

    def test_build_evaluated_candidate_marks_missing_asr_unavailable(self) -> None:
        candidate = build_evaluated_candidate(
            candidate_id="clarity_leveling_safe",
            requires_asr=True,
            quality_guard={"release_blocked": False, "failures": []},
        )
        self.assertEqual(candidate["asr_guard"]["status"], "UNAVAILABLE")
        self.assertTrue(candidate["asr_guard"]["release_blocked"])
        self.assertLess(candidate["score"], 0)

    def test_build_evaluated_candidate_allows_natural_without_asr(self) -> None:
        candidate = build_evaluated_candidate(
            candidate_id="natural_baseline",
            requires_asr=False,
            quality_guard={
                "release_blocked": False,
                "failures": [],
                "worst_relative_attenuation_db": -0.5,
                "relative_gain_spread_db": 1.0,
                "spectral_band_deltas_db": {"2-4k": -0.2, "4-6k": -0.1},
            },
        )
        self.assertEqual(candidate["asr_guard"]["status"], "NOT_REQUIRED")
        self.assertFalse(candidate["asr_guard"]["release_blocked"])
        self.assertGreater(candidate["score"], 0)

    def test_model_candidate_is_rejected_without_measurable_benefit(self) -> None:
        candidate = build_evaluated_candidate(
            candidate_id="deepfilter_denoise_safe",
            requires_asr=False,
            recipe=AUTO_CANDIDATE_RECIPES["deepfilter_denoise_safe"],
            quality_guard={
                "release_blocked": False,
                "failures": [],
                "worst_relative_attenuation_db": -0.5,
                "relative_gain_spread_db": 1.0,
                "spectral_band_deltas_db": {},
                "active_level_correlation": 0.99,
            },
            core_report={
                "stage_status": {"deepfilternet": {"enabled": True, "applied": True}},
                "input_diagnostics": {"estimated_snr_db": 16.0},
                "output_diagnostics": {"estimated_snr_db": 16.1},
            },
        )

        self.assertTrue(candidate["quality_guard"]["release_blocked"])
        self.assertIn("no_measurable_model_benefit", candidate["quality_guard"]["failures"])
        self.assertTrue(candidate["model_applied"])
        self.assertTrue(candidate["model_succeeded"])

    def test_respiro_fallback_is_not_eligible_as_model_success(self) -> None:
        candidate = build_evaluated_candidate(
            candidate_id="respiro_breath_safe",
            requires_asr=False,
            recipe=AUTO_CANDIDATE_RECIPES["respiro_breath_safe"],
            quality_guard={"release_blocked": False, "failures": []},
            core_report={
                "respiro_attempted": True,
                "respiro_succeeded": False,
                "respiro_detection_mode": "fallback",
                "stage_status": {"respiro": {"enabled": True, "applied": True}},
            },
        )

        self.assertTrue(candidate["fallback_used"])
        self.assertFalse(candidate["model_succeeded"])
        self.assertIn("model_fallback_used", candidate["quality_guard"]["failures"])

    def test_model_candidate_strict_guard_blocks_speech_attenuation(self) -> None:
        candidate = build_evaluated_candidate(
            candidate_id="deepfilter_denoise_safe",
            requires_asr=False,
            recipe=AUTO_CANDIDATE_RECIPES["deepfilter_denoise_safe"],
            quality_guard={
                "release_blocked": False,
                "failures": [],
                "worst_relative_attenuation_db": -4.0,
                "relative_gain_spread_db": 2.0,
                "active_level_correlation": 0.98,
                "spectral_band_deltas_db": {},
            },
            core_report={
                "stage_status": {"deepfilternet": {"enabled": True, "applied": True}},
                "input_diagnostics": {"estimated_snr_db": 10.0},
                "output_diagnostics": {"estimated_snr_db": 14.0},
            },
        )

        self.assertIn(
            "model_active_speech_attenuated",
            candidate["quality_guard"]["failures"],
        )
        self.assertTrue(candidate["quality_guard"]["release_blocked"])

    def test_combined_model_candidate_still_requires_dual_asr(self) -> None:
        candidate = build_evaluated_candidate(
            candidate_id="model_combined_review",
            requires_asr=True,
            recipe=AUTO_CANDIDATE_RECIPES["model_combined_review"],
            quality_guard={
                "release_blocked": False,
                "failures": [],
                "active_level_correlation": 0.99,
            },
            core_report={
                "respiro_succeeded": True,
                "respiro_detection_mode": "respiro",
                "breath_attenuation_db": 2.0,
                "breath_onset_windows": [{"start_seconds": 0.1, "end_seconds": 0.2}],
                "stage_status": {
                    "respiro": {"applied": True},
                    "deepfilternet": {"applied": True},
                },
                "input_diagnostics": {"estimated_snr_db": 10.0},
                "output_diagnostics": {"estimated_snr_db": 14.0},
            },
        )

        self.assertEqual(candidate["asr_guard"]["status"], "UNAVAILABLE")
        self.assertTrue(candidate["asr_guard"]["release_blocked"])


if __name__ == "__main__":
    unittest.main()
