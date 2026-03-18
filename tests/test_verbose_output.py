"""Tests for verbose terminal output in all modes and scenarios.

Verifies that main() prints configuration details and step-by-step
progress messages, especially in ``--headless`` mode, so that a user
can follow program execution without a monitor.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

cv2 = pytest.importorskip(
    "cv2", reason="OpenCV runtime dependencies are unavailable"
)

from main import _display_mode_label, _enabled_detectors_label, _parse_args


# ---------------------------------------------------------------------------
# Helper: create a tiny black MP4 video
# ---------------------------------------------------------------------------


def _make_dummy_video(path, num_frames: int = 3, size: int = 64):
    """Write *num_frames* black frames into an MP4 at *path*."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 1, (size, size))
    for _ in range(num_frames):
        writer.write(np.zeros((size, size, 3), dtype=np.uint8))
    writer.release()


# ---------------------------------------------------------------------------
# _display_mode_label / _enabled_detectors_label helpers
# ---------------------------------------------------------------------------


class TestDisplayModeLabel:
    def test_gui(self):
        args = _parse_args(["--gui"])
        assert _display_mode_label(args) == "gui"

    def test_headless(self):
        args = _parse_args(["--headless"])
        assert _display_mode_label(args) == "headless"

    def test_display_default(self):
        args = _parse_args([])
        assert _display_mode_label(args) == "display"


class TestEnabledDetectorsLabel:
    def test_defaults(self):
        args = _parse_args([])
        assert _enabled_detectors_label(args) == "AprilTag"

    def test_all_enabled(self):
        args = _parse_args(["--qr", "--laser"])
        label = _enabled_detectors_label(args)
        assert "AprilTag" in label
        assert "QR" in label
        assert "Laser" in label

    def test_apriltag_disabled(self):
        args = _parse_args(["--no-apriltag"])
        assert _enabled_detectors_label(args) == "none"

    def test_qr_only(self):
        args = _parse_args(["--no-apriltag", "--qr"])
        assert _enabled_detectors_label(args) == "QR"


class TestLaserThresholdCLI:
    """Verify --laser-threshold and --laser-threshold-max parsing."""

    def test_default_laser_threshold(self):
        args = _parse_args([])
        assert args.laser_threshold == 240

    def test_custom_laser_threshold(self):
        args = _parse_args(["--laser-threshold", "180"])
        assert args.laser_threshold == 180

    def test_default_laser_threshold_max(self):
        args = _parse_args([])
        assert args.laser_threshold_max == 255

    def test_custom_laser_threshold_max(self):
        args = _parse_args(["--laser-threshold-max", "250"])
        assert args.laser_threshold_max == 250

    def test_headless_threshold_passed_to_detector(self, capsys, tmp_path):
        """Verify threshold values are used when running headless."""
        video = tmp_path / "black.mp4"
        _make_dummy_video(video, num_frames=1)

        with patch(
            "robo_eye_sense.april_tag_detector._apriltags_available",
            return_value=False,
        ):
            from main import main

            rc = main([
                "--source", str(video),
                "--headless",
                "--laser",
                "--laser-threshold", "200",
                "--laser-threshold-max", "250",
            ])

        assert rc == 0


# ---------------------------------------------------------------------------
# Startup configuration summary (headless detection loop)
# ---------------------------------------------------------------------------


class TestStartupConfigSummary:
    """main() should print configuration lines before processing."""

    def test_headless_prints_config(self, capsys, tmp_path):
        video = tmp_path / "black.mp4"
        _make_dummy_video(video)

        with patch(
            "robo_eye_sense.april_tag_detector._apriltags_available",
            return_value=False,
        ):
            from main import main

            main(["--source", str(video), "--headless"])

        out = capsys.readouterr().out
        assert "Display mode" in out
        assert "headless" in out
        assert "Detection mode" in out
        assert "normal" in out
        assert "Detectors enabled" in out
        assert "AprilTag" in out
        assert "Source" in out
        assert "Camera opened" in out

    def test_headless_prints_detection_mode_fast(self, capsys, tmp_path):
        video = tmp_path / "black.mp4"
        _make_dummy_video(video)

        with patch(
            "robo_eye_sense.april_tag_detector._apriltags_available",
            return_value=False,
        ):
            from main import main

            main(["--source", str(video), "--headless", "--mode", "fast"])

        out = capsys.readouterr().out
        assert "fast" in out

    def test_headless_prints_scenario_when_set(self, capsys, tmp_path):
        video = tmp_path / "black.mp4"
        _make_dummy_video(video, num_frames=2)

        with patch(
            "robo_eye_sense.april_tag_detector._apriltags_available",
            return_value=False,
        ):
            from main import main

            main([
                "--source", str(video), "--headless",
                "--scenario", "offset",
            ])

        out = capsys.readouterr().out
        assert "Scenario" in out
        assert "offset" in out


# ---------------------------------------------------------------------------
# Headless detection loop verbose output
# ---------------------------------------------------------------------------


class TestHeadlessDetectionLoopOutput:
    """In headless mode each frame should print a line with frame number."""

    def test_frame_numbers_in_output(self, capsys, tmp_path):
        video = tmp_path / "black.mp4"
        _make_dummy_video(video, num_frames=3)

        with patch(
            "robo_eye_sense.april_tag_detector._apriltags_available",
            return_value=False,
        ):
            from main import main

            main(["--source", str(video), "--headless"])

        out = capsys.readouterr().out
        assert "[frame 1]" in out
        assert "[frame 2]" in out
        assert "[frame 3]" in out

    def test_no_detections_message(self, capsys, tmp_path):
        """Black frames should produce 'No detections' messages."""
        video = tmp_path / "black.mp4"
        _make_dummy_video(video, num_frames=1)

        with patch(
            "robo_eye_sense.april_tag_detector._apriltags_available",
            return_value=False,
        ):
            from main import main

            main(["--source", str(video), "--headless"])

        out = capsys.readouterr().out
        assert "No detections" in out

    def test_stream_ended_message(self, capsys, tmp_path):
        """Stream ending should print a summary message."""
        video = tmp_path / "black.mp4"
        _make_dummy_video(video, num_frames=2)

        with patch(
            "robo_eye_sense.april_tag_detector._apriltags_available",
            return_value=False,
        ):
            from main import main

            main(["--source", str(video), "--headless"])

        out = capsys.readouterr().out
        assert "Stream ended" in out
        assert "Total frames processed" in out

    def test_starting_detection_loop_message(self, capsys, tmp_path):
        video = tmp_path / "black.mp4"
        _make_dummy_video(video, num_frames=1)

        with patch(
            "robo_eye_sense.april_tag_detector._apriltags_available",
            return_value=False,
        ):
            from main import main

            main(["--source", str(video), "--headless"])

        out = capsys.readouterr().out
        assert "Starting detection loop" in out


# ---------------------------------------------------------------------------
# Offset scenario verbose output
# ---------------------------------------------------------------------------


class TestOffsetScenarioVerboseOutput:
    """Offset scenario should print step-by-step progress messages."""

    def test_offset_headless_prints_steps(self, capsys, tmp_path):
        video = tmp_path / "black.mp4"
        _make_dummy_video(video, num_frames=2)

        with patch(
            "robo_eye_sense.april_tag_detector._apriltags_available",
            return_value=False,
        ):
            from main import main

            rc = main([
                "--source", str(video), "--headless",
                "--scenario", "offset",
            ])

        assert rc == 0
        out = capsys.readouterr().out
        assert "Starting offset scenario" in out
        assert "Capturing reference frame" in out
        assert "Capturing current frame and computing offset" in out
        assert "Offset scenario finished" in out

    def test_offset_interactive_prints_steps(self, capsys, tmp_path):
        video = tmp_path / "black.mp4"
        _make_dummy_video(video, num_frames=2)

        with (
            patch("builtins.input", return_value=""),
            patch(
                "robo_eye_sense.april_tag_detector._apriltags_available",
                return_value=False,
            ),
        ):
            from main import main

            rc = main([
                "--source", str(video),
                "--scenario", "offset",
            ])

        assert rc == 0
        out = capsys.readouterr().out
        assert "Starting offset scenario" in out
        assert "Capturing reference frame" in out
        assert "Capturing current frame and computing offset" in out


# ---------------------------------------------------------------------------
# SLAM scenario verbose output
# ---------------------------------------------------------------------------


class TestSlamScenarioVerboseOutput:
    """SLAM scenario should print step-by-step progress messages."""

    def test_slam_headless_prints_steps(self, capsys, tmp_path):
        video = tmp_path / "black.mp4"
        _make_dummy_video(video, num_frames=3)

        with patch(
            "robo_eye_sense.april_tag_detector._apriltags_available",
            return_value=False,
        ):
            from main import main

            rc = main([
                "--source", str(video), "--headless",
                "--scenario", "slam",
                "--map-file", str(tmp_path / "map.json"),
            ])

        assert rc == 0
        out = capsys.readouterr().out
        assert "Starting SLAM calibration" in out
        assert "Scanning for AprilTag markers" in out
        assert "[frame 1]" in out
        assert "No markers visible" in out
        assert "Stream ended" in out
        assert "SLAM calibration result" in out


# ---------------------------------------------------------------------------
# Recording messages
# ---------------------------------------------------------------------------


class TestRecordingMessages:
    """Recording start/stop should be reported to stdout."""

    def test_recording_saved_message_headless(self, capsys, tmp_path):
        video = tmp_path / "black.mp4"
        _make_dummy_video(video, num_frames=2)

        output_file = str(tmp_path / "out.mp4")

        with patch(
            "robo_eye_sense.april_tag_detector._apriltags_available",
            return_value=False,
        ):
            from main import main

            rc = main([
                "--source", str(video), "--headless",
                "--record", output_file,
            ])

        assert rc == 0
        out = capsys.readouterr().out
        assert "Recording to" in out
        assert "Recording saved to" in out
