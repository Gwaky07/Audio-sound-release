from __future__ import annotations

import tempfile
import unittest
from array import array
from pathlib import Path

from audio_sound.segment_removal import (
    CutWindow,
    KeepInterval,
    build_join_seam_pause_ms,
    build_keep_intervals,
    build_video_trim_filter,
    parse_cut_spec,
    parse_time_seconds,
    refine_cut_windows_pcm16,
    reserve_delivery_paths,
    splice_pcm16_samples,
)


class SegmentRemovalTests(unittest.TestCase):
    def test_parse_time_seconds_accepts_fullwidth_colon(self) -> None:
        self.assertEqual(parse_time_seconds("00：03.5"), 3.5)
        self.assertEqual(parse_time_seconds("01:02:03"), 3723.0)

    def test_parse_cut_spec_accepts_tail_deletion(self) -> None:
        cut = parse_cut_spec("4.40,")
        self.assertEqual(cut, CutWindow(start_seconds=4.4, end_seconds=None))

    def test_build_keep_intervals_removes_middle_segment(self) -> None:
        intervals = build_keep_intervals(8.0, [CutWindow(2.0, 4.5)])
        self.assertEqual(
            intervals,
            [
                KeepInterval(0.0, 2.0),
                KeepInterval(4.5, 8.0),
            ],
        )

    def test_build_keep_intervals_merges_overlapping_cuts(self) -> None:
        intervals = build_keep_intervals(
            10.0,
            [
                CutWindow(2.0, 4.0),
                CutWindow(3.5, 7.0),
            ],
        )
        self.assertEqual(
            intervals,
            [
                KeepInterval(0.0, 2.0),
                KeepInterval(7.0, 10.0),
            ],
        )

    def test_build_keep_intervals_removes_tail_segment(self) -> None:
        intervals = build_keep_intervals(6.0, [CutWindow(4.4, None)])
        self.assertEqual(intervals, [KeepInterval(0.0, 4.4)])

    def test_build_keep_intervals_rejects_full_removal(self) -> None:
        with self.assertRaises(ValueError):
            build_keep_intervals(6.0, [CutWindow(0.0, None)])

    def test_build_join_seam_pause_ms_skips_leading_and_tail_cuts(self) -> None:
        pauses = build_join_seam_pause_ms(
            10.0,
            [
                CutWindow(0.0, 1.0, seam_pause_ms=80.0),
                CutWindow(3.0, 4.0, seam_pause_ms=60.0),
                CutWindow(8.0, None, seam_pause_ms=100.0),
            ],
            default_seam_pause_ms=20.0,
        )

        self.assertEqual(pauses, [60.0])

    def test_reserve_delivery_paths_increments_when_any_output_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            delivery_dir = Path(tmp_dir)
            (delivery_dir / "修音版_第二段.mp4").write_bytes(b"old video")

            paths = reserve_delivery_paths(delivery_dir, "第二段")

            self.assertEqual(paths.wav.name, "修音版_第二段_01.wav")
            self.assertEqual(paths.mp3.name, "修音版_第二段_01.mp3")
            self.assertEqual(paths.mp4.name, "修音版_第二段_01.mp4")
            self.assertEqual(paths.report.name, "修音版_第二段_01_segment-removal-report.json")

    def test_splice_pcm16_samples_crossfades_join(self) -> None:
        samples = array("h", range(100))
        result = splice_pcm16_samples(
            samples,
            channels=1,
            sample_rate=10,
            keep_intervals=[
                KeepInterval(0.0, 2.0),
                KeepInterval(4.0, 6.0),
            ],
            crossfade_ms=100.0,
        )

        self.assertEqual(result.crossfade_frames, [1])
        self.assertEqual(result.seam_pause_frames, [0])
        self.assertEqual(len(result.samples), 39)

    def test_splice_pcm16_samples_uses_fade_through_pause_for_speech_join(self) -> None:
        samples = array("h", [1000] * 20 + [-1000] * 20)
        result = splice_pcm16_samples(
            samples,
            channels=1,
            sample_rate=10,
            keep_intervals=[
                KeepInterval(0.0, 2.0),
                KeepInterval(2.0, 4.0),
            ],
            crossfade_ms=100.0,
            seam_pause_ms=200.0,
        )

        self.assertEqual(result.crossfade_frames, [0])
        self.assertEqual(result.seam_pause_frames, [2])
        self.assertEqual(len(result.samples), 42)
        self.assertEqual(result.samples[19:23].tolist(), [0, 0, 0, 0])

    def test_splice_pcm16_samples_supports_per_join_pause(self) -> None:
        samples = array("h", [1000] * 60)
        result = splice_pcm16_samples(
            samples,
            channels=1,
            sample_rate=10,
            keep_intervals=[
                KeepInterval(0.0, 2.0),
                KeepInterval(2.0, 4.0),
                KeepInterval(4.0, 6.0),
            ],
            crossfade_ms=100.0,
            seam_pause_ms_by_join=[0.0, 200.0],
        )

        self.assertEqual(result.crossfade_frames, [1, 0])
        self.assertEqual(result.seam_pause_frames, [0, 2])

    def test_refine_cut_windows_respects_per_cut_zero_search(self) -> None:
        samples = array("h", [12000] * 5000)
        samples[1940:1970] = array("h", [0] * 30)
        samples[3030:3060] = array("h", [0] * 30)

        refined = refine_cut_windows_pcm16(
            samples,
            channels=1,
            sample_rate=1000,
            cuts=[CutWindow(2.0, 3.0, boundary_search_ms=0.0)],
            search_ms=80.0,
        )

        self.assertEqual(refined, [CutWindow(2.0, 3.0, boundary_search_ms=0.0)])

    def test_refine_cut_windows_expands_toward_low_energy_boundaries(self) -> None:
        samples = array("h", [12000] * 5000)
        samples[1940:1970] = array("h", [0] * 30)
        samples[3030:3060] = array("h", [0] * 30)

        refined = refine_cut_windows_pcm16(
            samples,
            channels=1,
            sample_rate=1000,
            cuts=[CutWindow(2.0, 3.0)],
            search_ms=80.0,
        )

        self.assertLess(refined[0].start_seconds, 2.0)
        self.assertGreater(refined[0].end_seconds or 0.0, 3.0)

    def test_refine_cut_windows_keeps_requested_boundary_without_quiet_valley(self) -> None:
        samples = array("h", [12000] * 5000)

        refined = refine_cut_windows_pcm16(
            samples,
            channels=1,
            sample_rate=1000,
            cuts=[CutWindow(2.0, 3.0)],
            search_ms=80.0,
        )

        self.assertEqual(refined, [CutWindow(2.0, 3.0)])

    def test_refine_cut_windows_uses_nearest_quiet_valley(self) -> None:
        samples = array("h", [12000] * 5000)
        samples[3010:3030] = array("h", [200] * 20)
        samples[3060:3090] = array("h", [0] * 30)

        refined = refine_cut_windows_pcm16(
            samples,
            channels=1,
            sample_rate=1000,
            cuts=[CutWindow(2.0, 3.0)],
            search_ms=100.0,
        )

        self.assertLess(refined[0].end_seconds or 0.0, 3.05)

    def test_build_video_trim_filter_for_multiple_intervals(self) -> None:
        filter_complex = build_video_trim_filter(
            [
                KeepInterval(0.0, 1.25),
                KeepInterval(2.0, 3.0),
            ],
            crossfade_frames=[12],
            sample_rate=1000,
        )

        self.assertIn("[0:v]trim=start=0:end=1.238", filter_complex)
        self.assertIn("concat=n=2:v=1:a=0[vout]", filter_complex)

    def test_build_video_trim_filter_holds_frame_for_speech_pause(self) -> None:
        filter_complex = build_video_trim_filter(
            [
                KeepInterval(0.0, 1.25),
                KeepInterval(2.0, 3.0),
            ],
            seam_pause_frames=[45],
            sample_rate=1000,
        )

        self.assertIn("tpad=stop_mode=clone:stop_duration=0.045", filter_complex)


if __name__ == "__main__":
    unittest.main()
