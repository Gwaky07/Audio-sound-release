from __future__ import annotations

import unittest

from audio_sound.agent_judgment import (
    build_capability_plan,
    build_repair_scorecard,
    delivery_completeness,
)


class AgentJudgmentTests(unittest.TestCase):
    def test_clean_source_skips_denoise_but_keeps_breath_eq_loudness(self) -> None:
        plan = build_capability_plan(
            {
                "stationary_noise": False,
                "estimated_snr_db": 45.0,
                "pause_ratio": 0.2,
                "candidate_noise_windows": [],
            },
            runtime_capabilities={
                "respiro_en": {"ready": True},
                "deepfilternet": {"ready": True},
            },
        )

        capabilities = plan["capabilities"]
        self.assertTrue(plan["clean_source"])
        self.assertEqual(plan["primary_candidate"], "final_repair_best")
        self.assertTrue(capabilities["breath_cleanup"]["needed"])
        self.assertTrue(capabilities["pause_cleanup"]["needed"])
        self.assertFalse(capabilities["denoise"]["needed"])
        self.assertEqual(
            capabilities["denoise"]["skipped_reason"],
            "skip_secondary_denoise_clean_source",
        )
        self.assertTrue(capabilities["clarity_eq"]["needed"])
        self.assertTrue(capabilities["loudness"]["needed"])
        self.assertFalse(capabilities["mouth_declick"]["needed"])

    def test_scorecard_marks_natural_incomplete_for_repair_intent(self) -> None:
        completeness = delivery_completeness(
            candidate_id="natural_baseline",
            repair_intent=True,
        )
        self.assertTrue(completeness["delivery_incomplete_for_repair_intent"])
        self.assertEqual(completeness["completion_status"], "INCOMPLETE")

        complete = delivery_completeness(
            candidate_id="final_repair_best",
            repair_intent=True,
        )
        self.assertFalse(complete["delivery_incomplete_for_repair_intent"])
        self.assertEqual(complete["completion_status"], "COMPLETE")

    def test_repair_scorecard_tracks_breath_pause_and_guards(self) -> None:
        plan = build_capability_plan(
            {
                "stationary_noise": False,
                "estimated_snr_db": 40.0,
                "pause_ratio": 0.15,
            }
        )
        scorecard = build_repair_scorecard(
            core_report={
                "processing_steps": [
                    "Clarity shaping via parametric equalizer",
                    "Breath and sibilance control via deesser",
                    "Voice compression via acompressor",
                    "Loudness normalization via loudnorm",
                ],
                "breath_cleanup": {"status": "PASS", "final_residual_windows": []},
                "pause_cleanup": {
                    "status": "PASS",
                    "mode": "speech_safe_autogate",
                    "final_residual_windows": [],
                },
                "input_adaptations": ["skip_secondary_denoise_clean_source"],
                "stage_status": {"deepfilternet": {"applied": False}},
            },
            quality_guard={
                "failures": [],
                "worst_relative_attenuation_db": -1.2,
                "active_level_correlation": 0.99,
                "spectral_band_deltas_db": {"2-4k": -0.5, "12-16k": -0.4},
            },
            capability_plan=plan,
            candidate_id="final_repair_best",
        )

        self.assertEqual(scorecard["status"], "PASS")
        self.assertEqual(scorecard["items"]["breath_cleanup"]["status"], "PASS")
        self.assertEqual(scorecard["items"]["pause_cleanup"]["status"], "PASS")
        self.assertEqual(scorecard["items"]["denoise"]["status"], "SKIP")
        self.assertEqual(scorecard["items"]["no_swallow"]["status"], "PASS")
        self.assertEqual(scorecard["items"]["no_harshness"]["status"], "PASS")

    def test_repair_scorecard_rejects_pause_cleanup_without_autogate_mode(self) -> None:
        plan = build_capability_plan(
            {
                "stationary_noise": False,
                "estimated_snr_db": 40.0,
                "pause_ratio": 0.15,
            }
        )
        scorecard = build_repair_scorecard(
            core_report={
                "processing_steps": [
                    "Clarity shaping via parametric equalizer",
                    "Loudness normalization via loudnorm",
                ],
                "breath_cleanup": {"status": "PASS", "final_residual_windows": []},
                "pause_cleanup": {
                    "status": "PASS",
                    "mode": "noise_floor_margin",
                    "final_residual_windows": [],
                },
                "stage_status": {"deepfilternet": {"applied": False}},
            },
            quality_guard={"failures": []},
            capability_plan=plan,
            candidate_id="final_repair_best",
        )

        self.assertEqual(scorecard["items"]["pause_cleanup"]["status"], "FAIL")
        self.assertTrue(scorecard["items"]["pause_cleanup"]["detail"]["mode_invalid"])


if __name__ == "__main__":
    unittest.main()
