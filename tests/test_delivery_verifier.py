from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from audio_sound.delivery_verifier import verify_delivery


def _valid_report() -> dict[str, object]:
    return {
        "preset_name": "final",
        "quality_guard": {
            "status": "PASS",
            "release_blocked": False,
            "failures": [],
        },
        "breath_cleanup": {
            "status": "PASS",
            "final_residual_windows": [],
            "failures": [],
        },
        "pause_cleanup": {
            "status": "PASS",
            "mode": "speech_safe_autogate",
            "allow_bridge_windows": False,
            "final_residual_windows": [],
            "failures": [],
        },
    }


def _probe(_: Path, __: str) -> dict[str, object]:
    return {
        "sample_rate": 44100,
        "channels": 2,
        "duration_seconds": 10.0,
    }


def _pair_pass(**_: object) -> dict[str, object]:
    return {
        "quality_guard": {
            "status": "PASS",
            "release_blocked": False,
            "failures": [],
        }
    }


class DeliveryVerifierTests(unittest.TestCase):
    def test_verify_delivery_binds_hashes_and_passes_independent_guard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "source.wav"
            final_wav = root / "final.wav"
            final_mp3 = root / "final.mp3"
            report = root / "audio_process_report.json"
            source.write_bytes(b"source")
            final_wav.write_bytes(b"wav")
            final_mp3.write_bytes(b"mp3")
            report.write_text(
                json.dumps(_valid_report(), ensure_ascii=False),
                encoding="utf-8",
            )

            manifest = verify_delivery(
                source=source,
                final_wav=final_wav,
                final_mp3=final_mp3,
                report_json=report,
                pair_evaluator=_pair_pass,
                probe=_probe,
            )

            self.assertEqual(manifest["status"], "PASS")
            self.assertFalse(manifest["release_blocked"])
            self.assertEqual(len(manifest["artifacts"]["final_wav"]["sha256"]), 64)

    def test_verify_delivery_accepts_utf8_bom_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "source.wav"
            final_wav = root / "final.wav"
            report = root / "audio_process_report.json"
            source.write_bytes(b"source")
            final_wav.write_bytes(b"wav")
            report.write_text(
                json.dumps(_valid_report(), ensure_ascii=False),
                encoding="utf-8-sig",
            )

            manifest = verify_delivery(
                source=source,
                final_wav=final_wav,
                report_json=report,
                pair_evaluator=_pair_pass,
                probe=_probe,
            )

            self.assertEqual(manifest["status"], "PASS")

    def test_verify_delivery_fails_closed_when_release_blocked_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "source.wav"
            final_wav = root / "final.wav"
            report = root / "audio_process_report.json"
            source.write_bytes(b"source")
            final_wav.write_bytes(b"wav")
            payload = _valid_report()
            del payload["quality_guard"]["release_blocked"]  # type: ignore[index]
            report.write_text(json.dumps(payload), encoding="utf-8")

            def pair_missing_block(**_: object) -> dict[str, object]:
                return {
                    "quality_guard": {
                        "status": "PASS",
                        "failures": [],
                    }
                }

            manifest = verify_delivery(
                source=source,
                final_wav=final_wav,
                report_json=report,
                pair_evaluator=pair_missing_block,
                probe=_probe,
            )

            self.assertEqual(manifest["status"], "FAIL")
            self.assertTrue(manifest["release_blocked"])
            self.assertIn("report_release_blocked", manifest["failures"])
            self.assertIn("independent_release_blocked", manifest["failures"])

    def test_verify_delivery_blocks_invalid_pause_mode_and_pair_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "source.wav"
            final_wav = root / "final.wav"
            report = root / "audio_process_report.json"
            source.write_bytes(b"source")
            final_wav.write_bytes(b"wav")
            payload = _valid_report()
            payload["pause_cleanup"]["mode"] = "noise_floor_margin"  # type: ignore[index]
            report.write_text(json.dumps(payload), encoding="utf-8")

            def pair_fail(**_: object) -> dict[str, object]:
                return {
                    "quality_guard": {
                        "status": "FAIL",
                        "release_blocked": True,
                        "failures": ["source_active_hard_mute"],
                    }
                }

            manifest = verify_delivery(
                source=source,
                final_wav=final_wav,
                report_json=report,
                pair_evaluator=pair_fail,
                probe=_probe,
            )

            self.assertEqual(manifest["status"], "FAIL")
            self.assertIn("report_pause_mode_invalid", manifest["failures"])
            self.assertIn("independent_source_active_hard_mute", manifest["failures"])

    def test_verify_delivery_accepts_skill_file_report_wrapper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "source.wav"
            final_wav = root / "final.wav"
            report = root / "skill-file-report.json"
            source.write_bytes(b"source")
            final_wav.write_bytes(b"wav")
            core = _valid_report()
            core["deliverables"] = {"wav": str(final_wav)}
            report.write_text(
                json.dumps({"core_report": core}, ensure_ascii=False),
                encoding="utf-8",
            )

            manifest = verify_delivery(
                source=source,
                final_wav=final_wav,
                report_json=report,
                pair_evaluator=_pair_pass,
                probe=_probe,
            )

            self.assertEqual(manifest["status"], "PASS")

    def test_verify_delivery_repair_intent_requires_complete_scorecard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "source.wav"
            final_wav = root / "final.wav"
            report = root / "skill-workflow-summary.json"
            source.write_bytes(b"source")
            final_wav.write_bytes(b"wav")
            core = _valid_report()
            core["deliverables"] = {"wav": str(final_wav)}
            report.write_text(
                json.dumps(
                    {
                        "core_report": core,
                        "delivery_incomplete_for_repair_intent": True,
                        "repair_scorecard": {"status": "FAIL", "failed_items": ["pause_cleanup"]},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            manifest = verify_delivery(
                source=source,
                final_wav=final_wav,
                report_json=report,
                repair_intent=True,
                pair_evaluator=_pair_pass,
                probe=_probe,
            )

            self.assertEqual(manifest["status"], "FAIL")
            self.assertIn("delivery_incomplete_for_repair_intent", manifest["failures"])
            self.assertIn("repair_scorecard_incomplete", manifest["failures"])

    def test_verify_delivery_blocks_report_path_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "source.wav"
            final_wav = root / "final.wav"
            other_wav = root / "other.wav"
            report = root / "audio_process_report.json"
            source.write_bytes(b"source")
            final_wav.write_bytes(b"wav")
            other_wav.write_bytes(b"other")
            payload = _valid_report()
            payload["deliverables"] = {"wav": str(other_wav)}
            report.write_text(json.dumps(payload), encoding="utf-8")

            manifest = verify_delivery(
                source=source,
                final_wav=final_wav,
                report_json=report,
                pair_evaluator=_pair_pass,
                probe=_probe,
            )

            self.assertEqual(manifest["status"], "FAIL")
            self.assertIn("report_final_wav_mismatch", manifest["failures"])

    def test_verify_delivery_blocks_wrapper_deliverable_path_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "source.wav"
            final_wav = root / "final.wav"
            other_wav = root / "other.wav"
            report = root / "skill-file-report.json"
            source.write_bytes(b"source")
            final_wav.write_bytes(b"wav")
            other_wav.write_bytes(b"other")
            report.write_text(
                json.dumps(
                    {
                        "core_report": _valid_report(),
                        "deliverables": {"wav": str(other_wav)},
                    }
                ),
                encoding="utf-8",
            )

            manifest = verify_delivery(
                source=source,
                final_wav=final_wav,
                report_json=report,
                pair_evaluator=_pair_pass,
                probe=_probe,
            )

            self.assertEqual(manifest["status"], "FAIL")
            self.assertIn("report_final_wav_mismatch", manifest["failures"])


if __name__ == "__main__":
    unittest.main()
