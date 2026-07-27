from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from audio_sound.release_audit import audit_release, main, normalize_audit_text


def _asr_payload(segments: list[dict[str, object]]) -> dict[str, object]:
    return {"engine": "test-asr", "language": "zh", "duration": 700.0, "segments": segments}


def _segment(start: float, end: float, text: str) -> dict[str, object]:
    return {
        "start": start,
        "end": end,
        "text": text,
        "words": [{"start": start, "end": end, "word": text}],
    }


def _requirements() -> dict[str, object]:
    return {
        "source_document_revision": 422,
        "requirements": [
            {
                "id": "red-text-its-practice",
                "target": "它实践",
                "variants": ["它實踐", "他实践", "他實踐"],
                "allowed_contexts": ["它这个实践", "它這個實踐", "他这个实践", "他這個實踐"],
                "source_occurrences": [
                    {"start_seconds": 606.10, "end_seconds": 607.28, "action": "remove"},
                    {"start_seconds": 642.18, "end_seconds": 643.34, "action": "remove"},
                    {"start_seconds": 656.26, "end_seconds": 657.58, "action": "remove"},
                    {"start_seconds": 672.94, "end_seconds": 674.14, "action": "remove"},
                ],
                "user_reported_final_timestamps": [554.62, 568.70],
                "timestamp_window_seconds": 2.0,
                "expected_final_occurrences": 0,
                "minimum_allowed_context_occurrences": 1,
            }
        ],
    }


def _source_asr() -> dict[str, object]:
    return _asr_payload(
        [
            _segment(606.10, 607.28, "他實踐"),
            _segment(607.28, 608.30, "他這個實踐"),
            _segment(642.18, 643.34, "他實踐"),
            _segment(643.34, 645.20, "他這個實踐"),
            _segment(656.26, 657.58, "他實踐"),
            _segment(657.58, 659.20, "他這個實踐"),
            _segment(672.94, 674.14, "他實踐"),
            _segment(674.14, 676.00, "他這個實踐"),
        ]
    )


def _full_cut_report() -> dict[str, object]:
    return {
        "cuts": [
            {"start_seconds": 594.72, "end_seconds": 630.18},
            {"start_seconds": 642.18, "end_seconds": 643.34},
            {"start_seconds": 656.26, "end_seconds": 657.58},
            {"start_seconds": 672.94, "end_seconds": 674.14},
        ]
    }


def _write_cli_inputs(root: Path, final_asr: dict[str, object]) -> dict[str, Path]:
    paths = {
        "requirements": root / "requirements.json",
        "source": root / "source.json",
        "final": root / "final.json",
        "report": root / "segment-report.json",
    }
    payloads = {
        "requirements": _requirements(),
        "source": _source_asr(),
        "final": final_asr,
        "report": _full_cut_report(),
    }
    for name, path in paths.items():
        path.write_text(json.dumps(payloads[name], ensure_ascii=False), encoding="utf-8")
    return paths


def _cli_args(
    paths: dict[str, Path],
    final_media_path: Path,
    output_path: Path,
    *,
    minimum_engines: int,
) -> list[str]:
    return [
        "run",
        "--requirements",
        str(paths["requirements"]),
        "--source-asr",
        str(paths["source"]),
        "--final-asr",
        str(paths["final"]),
        "--segment-report",
        str(paths["report"]),
        "--final-media",
        str(final_media_path),
        "--minimum-final-asr-engines",
        str(minimum_engines),
        "--output",
        str(output_path),
    ]


class ReleaseAuditTests(unittest.TestCase):
    def test_normalize_audit_text_handles_traditional_punctuation_and_nfkc(self) -> None:
        self.assertEqual(normalize_audit_text(" 它，實踐！Ａ "), "它实践A")
        self.assertEqual(normalize_audit_text("它這個實踐"), "它这个实践")

    def test_gate_fails_when_only_third_repeated_occurrence_was_cut(self) -> None:
        final_asr = _asr_payload(
            [
                _segment(554.62, 555.78, "它實踐"),
                _segment(555.78, 557.20, "它這個實踐"),
                _segment(568.70, 570.02, "它實踐"),
                _segment(570.02, 571.60, "它這個實踐"),
            ]
        )
        report = {
            "cuts": [
                {"start_seconds": 594.72, "end_seconds": 630.18},
                {"start_seconds": 672.94, "end_seconds": 674.14},
            ]
        }

        result = audit_release(
            requirements_payload=_requirements(),
            source_asr_payloads=[("source-small", _source_asr())],
            final_asr_payloads=[("final-small", final_asr)],
            segment_report_payload=report,
        )

        self.assertEqual(result["status"], "FAIL")
        requirement = result["requirements"][0]
        self.assertEqual(
            [item["occurrence_index"] for item in requirement["uncovered_source_occurrences"]],
            [1, 2],
        )
        self.assertEqual(
            [round(item["start_seconds"], 2) for item in requirement["forbidden_final_hits"]],
            [554.62, 568.70],
        )
        self.assertTrue(all(item["checked"] for item in requirement["timestamp_checks"]))

    def test_gate_passes_after_all_occurrences_are_cut_and_context_remains(self) -> None:
        final_asr = _asr_payload(
            [
                _segment(554.62, 556.20, "它這個實踐"),
                _segment(568.70, 570.20, "它这个实践"),
            ]
        )
        report = {
            "cuts": [
                {"start_seconds": 594.72, "end_seconds": 630.18},
                {"start_seconds": 642.18, "end_seconds": 643.34},
                {"start_seconds": 656.26, "end_seconds": 657.58},
                {"start_seconds": 672.94, "end_seconds": 674.14},
            ]
        }

        result = audit_release(
            requirements_payload=_requirements(),
            source_asr_payloads=[("source-small", _source_asr())],
            final_asr_payloads=[("final-small", final_asr)],
            segment_report_payload=report,
        )

        self.assertEqual(result["status"], "PASS")
        requirement = result["requirements"][0]
        self.assertEqual(requirement["forbidden_final_hits"], [])
        self.assertEqual(len(requirement["source_asr_occurrences"]), 4)
        self.assertEqual(len(requirement["covered_source_occurrences"]), 4)

    def test_gate_accepts_explicitly_reviewed_homophone_only_for_allowed_context(self) -> None:
        requirements = _requirements()
        requirements["requirements"][0]["allowed_contexts"].append("他这个时间")
        final_asr = _asr_payload(
            [
                _segment(554.62, 556.20, "他这个时间"),
                _segment(568.70, 570.20, "后续内容"),
            ]
        )
        report = {
            "cuts": [
                {"start_seconds": 594.72, "end_seconds": 630.18},
                {"start_seconds": 642.18, "end_seconds": 643.34},
                {"start_seconds": 656.26, "end_seconds": 657.58},
                {"start_seconds": 672.94, "end_seconds": 674.14},
            ]
        }

        result = audit_release(
            requirements_payload=requirements,
            source_asr_payloads=[("source-small", _source_asr())],
            final_asr_payloads=[("final-small", final_asr)],
            segment_report_payload=report,
        )

        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["requirements"][0]["forbidden_final_hits"], [])

    def test_gate_fails_when_retained_context_was_swallowed(self) -> None:
        final_asr = _asr_payload([_segment(554.62, 556.20, "后续内容")])
        report = {
            "cuts": [
                {"start_seconds": 594.72, "end_seconds": 630.18},
                {"start_seconds": 642.18, "end_seconds": 643.34},
                {"start_seconds": 656.26, "end_seconds": 657.58},
                {"start_seconds": 672.94, "end_seconds": 674.14},
            ]
        }

        result = audit_release(
            requirements_payload=_requirements(),
            source_asr_payloads=[("source-small", _source_asr())],
            final_asr_payloads=[("final-small", final_asr)],
            segment_report_payload=report,
        )

        self.assertEqual(result["status"], "FAIL")
        requirement = result["requirements"][0]
        self.assertIn("required_allowed_context_is_missing_from_final_asr", requirement["failures"])

    def test_gate_scans_segment_text_when_word_timestamps_disagree(self) -> None:
        final_asr = _asr_payload(
            [
                {
                    "start": 554.62,
                    "end": 556.20,
                    "text": "它實踐",
                    "words": [
                        {"start": 554.62, "end": 556.20, "word": "它这个实践"},
                    ],
                },
                _segment(568.70, 570.20, "后续内容"),
            ]
        )
        report = {
            "cuts": [
                {"start_seconds": 594.72, "end_seconds": 630.18},
                {"start_seconds": 642.18, "end_seconds": 643.34},
                {"start_seconds": 656.26, "end_seconds": 657.58},
                {"start_seconds": 672.94, "end_seconds": 674.14},
            ]
        }

        result = audit_release(
            requirements_payload=_requirements(),
            source_asr_payloads=[("source-small", _source_asr())],
            final_asr_payloads=[("final-small", final_asr)],
            segment_report_payload=report,
        )

        self.assertEqual(result["status"], "FAIL")
        observations = result["requirements"][0]["forbidden_final_hits"][0]["observations"]
        self.assertIn("segment_text", {item["channel"] for item in observations})

    def test_gate_fails_when_source_asr_has_an_undeclared_occurrence(self) -> None:
        requirements = _requirements()
        requirements["requirements"][0]["source_occurrences"] = [
            {"start_seconds": 606.10, "end_seconds": 607.28, "action": "remove"},
            {"start_seconds": 672.94, "end_seconds": 674.14, "action": "remove"},
        ]
        final_asr = _asr_payload([_segment(554.62, 556.20, "它这个实践")])
        report = {
            "cuts": [
                {"start_seconds": 594.72, "end_seconds": 630.18},
                {"start_seconds": 672.94, "end_seconds": 674.14},
            ]
        }

        result = audit_release(
            requirements_payload=requirements,
            source_asr_payloads=[("source-small", _source_asr())],
            final_asr_payloads=[("final-small", final_asr)],
            segment_report_payload=report,
        )

        self.assertEqual(result["status"], "FAIL")
        undeclared = result["requirements"][0]["undeclared_source_asr_occurrences"]
        self.assertEqual([round(item["start_seconds"], 2) for item in undeclared], [642.18, 656.26])

    def test_gate_fails_closed_when_user_timestamp_has_no_asr_evidence(self) -> None:
        final_asr = _asr_payload([_segment(100.0, 102.0, "没有目标词")])
        report = {
            "cuts": [
                {"start_seconds": 594.72, "end_seconds": 630.18},
                {"start_seconds": 642.18, "end_seconds": 643.34},
                {"start_seconds": 656.26, "end_seconds": 657.58},
                {"start_seconds": 672.94, "end_seconds": 674.14},
            ]
        }

        result = audit_release(
            requirements_payload=_requirements(),
            source_asr_payloads=[("source-small", _source_asr())],
            final_asr_payloads=[("final-small", final_asr)],
            segment_report_payload=report,
        )

        self.assertEqual(result["status"], "FAIL")
        timestamp_checks = result["requirements"][0]["timestamp_checks"]
        self.assertTrue(all(not item["checked"] for item in timestamp_checks))

    def test_cli_writes_machine_readable_failure_and_returns_nonzero(self) -> None:
        final_asr = _asr_payload(
            [
                _segment(554.62, 555.78, "它實踐"),
                _segment(568.70, 570.02, "它實踐"),
            ]
        )
        report = {
            "cuts": [
                {"start_seconds": 594.72, "end_seconds": 630.18},
                {"start_seconds": 672.94, "end_seconds": 674.14},
            ]
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            requirements_path = root / "requirements.json"
            source_asr_path = root / "source.json"
            final_asr_path = root / "final.json"
            segment_report_path = root / "segment-report.json"
            output_path = root / "audit.json"
            final_media_path = root / "final.mp4"
            final_media_path.write_bytes(b"failed-media")
            requirements_path.write_text(json.dumps(_requirements(), ensure_ascii=False), encoding="utf-8")
            source_asr_path.write_text(json.dumps(_source_asr(), ensure_ascii=False), encoding="utf-8")
            final_asr["media_sha256"] = hashlib.sha256(b"failed-media").hexdigest()
            final_asr_path.write_text(json.dumps(final_asr, ensure_ascii=False), encoding="utf-8")
            segment_report_path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")

            exit_code = main(
                [
                    "run",
                    "--requirements",
                    str(requirements_path),
                    "--source-asr",
                    str(source_asr_path),
                    "--final-asr",
                    str(final_asr_path),
                    "--segment-report",
                    str(segment_report_path),
                    "--final-media",
                    str(final_media_path),
                    "--minimum-final-asr-engines",
                    "1",
                    "--output",
                    str(output_path),
                ]
            )

            self.assertEqual(exit_code, 1)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "FAIL")
            self.assertEqual(
                payload["final_media"][0]["sha256"],
                hashlib.sha256(b"failed-media").hexdigest().upper(),
            )

    def test_cli_accepts_windows_utf8_bom_json(self) -> None:
        final_asr = _asr_payload(
            [
                _segment(554.62, 556.20, "它这个实践"),
                _segment(568.70, 570.20, "后续内容"),
            ]
        )
        report = {
            "cuts": [
                {"start_seconds": 594.72, "end_seconds": 630.18},
                {"start_seconds": 642.18, "end_seconds": 643.34},
                {"start_seconds": 656.26, "end_seconds": 657.58},
                {"start_seconds": 672.94, "end_seconds": 674.14},
            ]
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            inputs = {
                "requirements.json": _requirements(),
                "source.json": _source_asr(),
                "final.json": final_asr,
                "segment-report.json": report,
            }
            for name, payload in inputs.items():
                (root / name).write_text(
                    json.dumps(payload, ensure_ascii=False),
                    encoding="utf-8-sig",
                )
            output_path = root / "audit.json"
            final_media_path = root / "final.mp4"
            final_media_path.write_bytes(b"passing-media")
            inputs["final.json"]["media_sha256"] = hashlib.sha256(b"passing-media").hexdigest()
            (root / "final.json").write_text(
                json.dumps(inputs["final.json"], ensure_ascii=False),
                encoding="utf-8-sig",
            )

            exit_code = main(
                [
                    "run",
                    "--requirements",
                    str(root / "requirements.json"),
                    "--source-asr",
                    str(root / "source.json"),
                    "--final-asr",
                    str(root / "final.json"),
                    "--segment-report",
                    str(root / "segment-report.json"),
                    "--final-media",
                    str(final_media_path),
                    "--minimum-final-asr-engines",
                    "1",
                    "--output",
                    str(output_path),
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(json.loads(output_path.read_text(encoding="utf-8"))["status"], "PASS")

    def test_cli_blocks_asr_generated_from_a_different_media_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            final_media_path = root / "final.wav"
            asr_media_path = root / "intermediate.wav"
            final_media_path.write_bytes(b"final-pcm")
            asr_media_path.write_bytes(b"intermediate-pcm")
            final_asr = _asr_payload(
                [
                    _segment(554.62, 556.20, "它这个实践"),
                    _segment(568.70, 570.20, "后续内容"),
                ]
            )
            final_asr["media"] = str(asr_media_path)
            inputs = {
                "requirements.json": _requirements(),
                "source.json": _source_asr(),
                "final.json": final_asr,
                "segment-report.json": {
                    "cuts": [
                        {"start_seconds": 594.72, "end_seconds": 630.18},
                        {"start_seconds": 642.18, "end_seconds": 643.34},
                        {"start_seconds": 656.26, "end_seconds": 657.58},
                        {"start_seconds": 672.94, "end_seconds": 674.14},
                    ]
                },
            }
            for name, payload in inputs.items():
                (root / name).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            output_path = root / "audit.json"

            exit_code = main(
                [
                    "run",
                    "--requirements",
                    str(root / "requirements.json"),
                    "--source-asr",
                    str(root / "source.json"),
                    "--final-asr",
                    str(root / "final.json"),
                    "--segment-report",
                    str(root / "segment-report.json"),
                    "--final-media",
                    str(final_media_path),
                    "--output",
                    str(output_path),
                ]
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 1)
            self.assertEqual(payload["status"], "FAIL")
            self.assertFalse(payload["final_asr_media_bindings"][0]["bound"])
            self.assertIn("a_final_asr_pass_is_not_bound", payload["failures"][0])

    def test_cli_blocks_mismatched_declared_media_and_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            final_media_path = root / "final.wav"
            final_media_path.write_bytes(b"final-pcm")
            final_asr = _asr_payload(
                [
                    _segment(554.62, 556.20, "它这个实践"),
                    _segment(568.70, 570.20, "后续内容"),
                ]
            )
            final_asr["media"] = str(final_media_path)
            final_asr["media_sha256"] = hashlib.sha256(b"different-pcm").hexdigest()
            paths = _write_cli_inputs(root, final_asr)
            output_path = root / "audit.json"

            exit_code = main(
                _cli_args(paths, final_media_path, output_path, minimum_engines=1)
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 1)
            self.assertEqual(
                payload["final_asr_media_bindings"][0]["failure"],
                "declared_media_sha256_mismatch",
            )

    def test_cli_blocks_duplicate_engines_from_counting_as_independent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            final_media_path = root / "final.wav"
            final_media_path.write_bytes(b"final-pcm")
            media_hash = hashlib.sha256(b"final-pcm").hexdigest()
            final_asr = _asr_payload(
                [
                    _segment(554.62, 556.20, "它这个实践"),
                    _segment(568.70, 570.20, "后续内容"),
                ]
            )
            final_asr["engine"] = "same-engine"
            final_asr["media_sha256"] = media_hash
            paths = _write_cli_inputs(root, final_asr)
            duplicate_path = root / "final-duplicate.json"
            duplicate_path.write_text(json.dumps(final_asr, ensure_ascii=False), encoding="utf-8")
            output_path = root / "audit.json"
            args = _cli_args(paths, final_media_path, output_path, minimum_engines=2)
            final_asr_index = args.index("--final-asr")
            args[final_asr_index:final_asr_index + 2] = [
                "--final-asr",
                str(paths["final"]),
                "--final-asr",
                str(duplicate_path),
            ]

            exit_code = main(args)

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 1)
            self.assertIn("insufficient_independent_final_asr_engines", payload["failures"])

    def test_gate_requires_source_document_revision(self) -> None:
        requirements = _requirements()
        del requirements["source_document_revision"]

        with self.assertRaisesRegex(ValueError, "source_document_revision"):
            audit_release(
                requirements_payload=requirements,
                source_asr_payloads=[("source-small", _source_asr())],
                final_asr_payloads=[("final-small", _asr_payload([_segment(554.62, 556.20, "后续")]))],
                segment_report_payload={"cuts": []},
            )


if __name__ == "__main__":
    unittest.main()
