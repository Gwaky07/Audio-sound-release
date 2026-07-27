"""Contract tests that keep the enforcement layer from being quietly removed.

Every check in here exists because deleting a guard, flipping a fail-closed
default, or dropping a failure code would otherwise leave the whole suite green.
These tests assert the *presence and direction* of enforcement, not audio
behaviour, so they stay fast and have no ffmpeg dependency.
"""

from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from array import array
from pathlib import Path

from audio_sound import auto_workflow, delivery_verifier, pipeline
from audio_sound.pipeline import analyze_pcm16_samples, compare_audio_preservation

# Failure codes the preservation guard must be able to emit. Removing a check
# without updating this set is the exact silent regression we are guarding
# against. See AGENTS.md "Repository Audio Capability Contract".
REQUIRED_GUARD_FAILURE_CODES = frozenset(
    {
        "sample_rate_changed",
        "channel_layout_changed",
        "duration_changed",
        "clipping_increased",
        "active_speech_attenuated",
        "short_term_gain_instability",
        "source_active_hard_mute",
        "spectral_clarity_lost",
        "spectral_harshness_increased",
    }
)

# Report-level gates the independent delivery verifier must enforce.
REQUIRED_VERIFIER_FAILURE_CODES = frozenset(
    {
        "report_quality_guard_not_pass",
        "report_release_blocked",
        "report_quality_guard_failures",
        "report_breath_cleanup_not_pass",
        "report_breath_residuals_present",
        "report_pause_cleanup_not_pass",
        "report_pause_mode_invalid",
        "report_pause_bridge_not_disabled",
        "report_pause_residuals_present",
        "report_empty_assessment_windows",
        "independent_quality_guard_not_pass",
        "independent_release_blocked",
        "delivery_incomplete_for_repair_intent",
        "missing_repair_scorecard",
        "repair_scorecard_incomplete",
        "sample_rate_changed",
        "channel_layout_changed",
        "wav_duration_changed",
        "report_final_wav_mismatch",
    }
)

# Model-candidate disqualifiers. These are what stop "the model ran" from being
# mistaken for "the model helped".
REQUIRED_MODEL_DISQUALIFIERS = frozenset(
    {
        "model_fallback_used",
        "no_measurable_model_benefit",
        "asr_evidence_missing",
        "unscoped_model_stage",
    }
)


def _source_failure_codes(module: object) -> set[str]:
    source = inspect.getsource(module)  # type: ignore[arg-type]
    codes: set[str] = set()
    marker = 'failures.append("'
    index = source.find(marker)
    while index != -1:
        start = index + len(marker)
        end = source.find('"', start)
        if end == -1:
            break
        codes.add(source[start:end])
        index = source.find(marker, end)
    return codes


class GuardFailureCodeContractTests(unittest.TestCase):
    def test_preservation_guard_still_emits_every_required_failure_code(self) -> None:
        present = _source_failure_codes(pipeline)

        missing = REQUIRED_GUARD_FAILURE_CODES - present
        self.assertEqual(
            missing,
            set(),
            f"preservation guard lost failure code(s): {sorted(missing)}",
        )

    def test_delivery_verifier_still_emits_every_required_failure_code(self) -> None:
        present = _source_failure_codes(delivery_verifier)

        missing = REQUIRED_VERIFIER_FAILURE_CODES - present
        self.assertEqual(
            missing,
            set(),
            f"delivery verifier lost failure code(s): {sorted(missing)}",
        )

    def test_model_candidate_disqualifiers_are_still_referenced(self) -> None:
        source = inspect.getsource(auto_workflow)

        for code in sorted(REQUIRED_MODEL_DISQUALIFIERS):
            self.assertIn(
                code,
                source,
                f"auto_workflow no longer references disqualifier '{code}'",
            )


class FailClosedDefaultContractTests(unittest.TestCase):
    """A missing gate field must read as blocked, never as permission."""

    def test_verifier_treats_absent_release_blocked_as_blocked(self) -> None:
        report = {
            "quality_guard": {"status": "PASS", "failures": []},
            "breath_cleanup": {"status": "PASS", "final_residual_windows": []},
            "pause_cleanup": {
                "status": "PASS",
                "mode": "speech_safe_autogate",
                "allow_bridge_windows": False,
                "final_residual_windows": [],
            },
        }

        failures = delivery_verifier._report_gate_failures(report)

        self.assertIn("report_release_blocked", failures)

    def test_verifier_treats_absent_quality_guard_as_blocked(self) -> None:
        report = {
            "breath_cleanup": {"status": "PASS", "final_residual_windows": []},
            "pause_cleanup": {
                "status": "PASS",
                "mode": "speech_safe_autogate",
                "allow_bridge_windows": False,
                "final_residual_windows": [],
            },
        }

        failures = delivery_verifier._report_gate_failures(report)

        self.assertIn("report_release_blocked", failures)
        self.assertIn("report_quality_guard_not_pass", failures)

    def test_verifier_rejects_bridge_windows_left_unspecified(self) -> None:
        report = {
            "quality_guard": {
                "status": "PASS",
                "release_blocked": False,
                "failures": [],
            },
            "breath_cleanup": {"status": "PASS", "final_residual_windows": []},
            "pause_cleanup": {
                "status": "PASS",
                "mode": "speech_safe_autogate",
                "final_residual_windows": [],
            },
        }

        failures = delivery_verifier._report_gate_failures(report)

        self.assertIn("report_pause_bridge_not_disabled", failures)

    def test_missing_files_block_before_any_gate_is_consulted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            report = root / "report.json"
            report.write_text(json.dumps({}), encoding="utf-8")

            manifest = delivery_verifier.verify_delivery(
                source=root / "absent-source.wav",
                final_wav=root / "absent-final.wav",
                report_json=report,
            )

            self.assertEqual(manifest["status"], "FAIL")
            self.assertTrue(manifest["release_blocked"])


class GuardDirectionContractTests(unittest.TestCase):
    """Threshold direction is easy to invert in a refactor and invisible
    afterwards, so pin the sign of each comparison with a minimal case."""

    def test_attenuation_blocks_but_equal_loudness_does_not(self) -> None:
        reference = analyze_pcm16_samples(array("h", [6000] * 2000), sample_rate=1000)
        attenuated = analyze_pcm16_samples(
            array("h", [6000] * 1000 + [400] * 1000), sample_rate=1000
        )
        unchanged = analyze_pcm16_samples(array("h", [6000] * 2000), sample_rate=1000)

        blocked = compare_audio_preservation(reference, attenuated)
        allowed = compare_audio_preservation(reference, unchanged)

        self.assertTrue(blocked["release_blocked"])
        self.assertFalse(allowed["release_blocked"])

    def test_guard_status_and_release_blocked_never_disagree(self) -> None:
        reference = analyze_pcm16_samples(array("h", [6000] * 2000), sample_rate=1000)
        damaged = analyze_pcm16_samples(
            array("h", [6000] * 1000 + [200] * 1000), sample_rate=1000
        )

        for guard in (
            compare_audio_preservation(reference, reference),
            compare_audio_preservation(reference, damaged),
        ):
            with self.subTest(status=guard["status"]):
                self.assertEqual(
                    guard["status"] == "FAIL",
                    bool(guard["release_blocked"]),
                    "status and release_blocked disagree",
                )
                self.assertEqual(
                    bool(guard["failures"]),
                    bool(guard["release_blocked"]),
                    "failures list and release_blocked disagree",
                )


class PackagingContractTests(unittest.TestCase):
    """The wheel must carry the real implementations, and `scripts` must not
    become an installed top-level package again."""

    def test_logic_bearing_modules_live_in_the_package(self) -> None:
        for module_name in (
            "audio_sound.narrow_onset_cleanup",
            "audio_sound.exact_window_cleanup",
            "audio_sound.stereo_balance",
            "audio_sound.delivery_verifier",
            "audio_sound.pair_evaluation",
            "audio_sound.media_utils",
            "audio_sound.auto_workflow",
            "audio_sound.agent_judgment",
        ):
            with self.subTest(module=module_name):
                __import__(module_name)

    def test_package_never_imports_from_the_scripts_layer(self) -> None:
        package_root = Path(pipeline.__file__).resolve().parent
        offenders: list[str] = []
        for path in sorted(package_root.glob("*.py")):
            text = path.read_text(encoding="utf-8")
            if "from scripts" in text or "import scripts" in text:
                offenders.append(path.name)

        self.assertEqual(
            offenders,
            [],
            f"package modules must not depend on scripts/: {offenders}",
        )

    def test_scripts_directory_is_not_an_importable_package(self) -> None:
        repo_root = Path(pipeline.__file__).resolve().parents[1]
        scripts_dir = repo_root / "scripts"
        if not scripts_dir.is_dir():
            self.skipTest("scripts/ is absent in an installed-only checkout")

        self.assertFalse(
            (scripts_dir / "__init__.py").exists(),
            "scripts/__init__.py would make `scripts` a top-level installed package",
        )

    def test_pyproject_does_not_package_scripts(self) -> None:
        repo_root = Path(pipeline.__file__).resolve().parents[1]
        pyproject = repo_root / "pyproject.toml"
        if not pyproject.is_file():
            self.skipTest("pyproject.toml is absent in an installed-only checkout")

        text = pyproject.read_text(encoding="utf-8")
        packages_line = next(
            (line for line in text.splitlines() if line.strip().startswith("packages")),
            "",
        )

        self.assertNotIn('"scripts"', packages_line)
        self.assertIn('"audio_sound"', packages_line)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
