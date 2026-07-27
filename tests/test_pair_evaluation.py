from __future__ import annotations

import tempfile
import unittest
import wave
from array import array
from pathlib import Path
from unittest import mock

from audio_sound.pair_evaluation import evaluate_audio_pair


def _write_pcm16(path: Path, *, sample_rate: int = 8000, channels: int = 1) -> None:
    samples = array("h", [1200] * sample_rate * channels)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(samples.tobytes())


class PairEvaluationTests(unittest.TestCase):
    def test_wav_inputs_are_normalized_to_pcm16_before_loading(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "source-float.wav"
            processed = root / "processed.wav"
            source.write_bytes(b"not directly readable by wave")
            processed.write_bytes(b"not directly readable by wave")

            def extract_side_effect(
                input_path: Path,
                output_path: Path,
                **_: object,
            ) -> Path:
                self.assertIn(input_path, {source, processed})
                _write_pcm16(output_path)
                return output_path

            with (
                mock.patch(
                    "audio_sound.pair_evaluation.ffprobe_media",
                    return_value={"sample_rate": 8000, "channels": 1},
                ),
                mock.patch(
                    "audio_sound.pair_evaluation._extract_pcm_wav",
                    side_effect=extract_side_effect,
                ) as extract_mock,
            ):
                report = evaluate_audio_pair(source=source, processed=processed)

        self.assertEqual(extract_mock.call_count, 2)
        self.assertEqual(report["quality_guard"]["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
