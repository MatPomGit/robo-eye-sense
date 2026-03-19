"""Tests for the auto-follow scenario.

The tests verify the pure computation logic in
:func:`~robo_vision.auto_scenario.compute_follow_vector` and the
:class:`~robo_vision.auto_scenario.AutoFollowScenario` wrapper
using synthetic detections (no real camera or AprilTag images required).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from robo_vision.auto_scenario import (
    AutoFollowResult,
    AutoFollowScenario,
    compute_follow_vector,
)
from robo_vision.results import Detection, DetectionType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _april(tag_id: str, center: tuple[int, int]) -> Detection:
    """Shorthand for creating an AprilTag Detection."""
    return Detection(
        detection_type=DetectionType.APRIL_TAG,
        identifier=tag_id,
        center=center,
        corners=[],
    )


def _laser(center: tuple[int, int]) -> Detection:
    """Shorthand for creating a laser-spot Detection (no identifier)."""
    return Detection(
        detection_type=DetectionType.LASER_SPOT,
        identifier=None,
        center=center,
    )


# ---------------------------------------------------------------------------
# AutoFollowResult dataclass
# ---------------------------------------------------------------------------


class TestAutoFollowResult:
    def test_default_values(self):
        r = AutoFollowResult()
        assert r.position_vector == (0.0, 0.0)
        assert r.yaw == 0.0
        assert r.target_marker_id is None
        assert r.target_found is False
        assert r.visible_marker_ids == []
        assert r.marker_positions == {}

    def test_custom_values(self):
        r = AutoFollowResult(
            position_vector=(10.0, -5.0),
            yaw=3.5,
            target_marker_id="7",
            target_found=True,
            visible_marker_ids=["5", "7"],
        )
        assert r.position_vector == (10.0, -5.0)
        assert r.yaw == 3.5
        assert r.target_marker_id == "7"
        assert r.target_found is True


# ---------------------------------------------------------------------------
# compute_follow_vector – pure function
# ---------------------------------------------------------------------------


class TestComputeFollowVector:
    """Unit tests for the stateless compute_follow_vector function."""

    def test_no_detections_returns_zero(self):
        result = compute_follow_vector([], frame_width=640, frame_height=480)
        assert result.position_vector == (0.0, 0.0)
        assert result.yaw == 0.0
        assert result.target_found is False
        assert result.visible_marker_ids == []

    def test_no_apriltags_returns_zero(self):
        result = compute_follow_vector(
            [_laser((100, 100))], frame_width=640, frame_height=480
        )
        assert result.target_found is False
        assert result.visible_marker_ids == []

    def test_single_marker_at_center(self):
        """Marker at frame centre → zero displacement, zero yaw."""
        result = compute_follow_vector(
            [_april("1", (320, 240))],
            frame_width=640,
            frame_height=480,
        )
        assert result.target_found is True
        assert result.target_marker_id == "1"
        assert result.position_vector == pytest.approx((0.0, 0.0))
        assert result.yaw == pytest.approx(0.0, abs=0.1)

    def test_marker_to_the_right(self):
        """Marker to the right of centre → negative dx, positive yaw."""
        result = compute_follow_vector(
            [_april("1", (500, 240))],
            frame_width=640,
            frame_height=480,
        )
        assert result.target_found is True
        dx, dy = result.position_vector
        assert dx < 0  # marker is right → camera needs to move left (or frame_center - marker)
        assert dy == pytest.approx(0.0)
        assert result.yaw > 0  # positive yaw = marker to the right

    def test_marker_to_the_left(self):
        """Marker to the left of centre → positive dx, negative yaw."""
        result = compute_follow_vector(
            [_april("1", (100, 240))],
            frame_width=640,
            frame_height=480,
        )
        dx, dy = result.position_vector
        assert dx > 0  # marker is left → need to move right to center
        assert result.yaw < 0

    def test_marker_above(self):
        """Marker above centre → positive dy."""
        result = compute_follow_vector(
            [_april("1", (320, 100))],
            frame_width=640,
            frame_height=480,
        )
        dx, dy = result.position_vector
        assert dx == pytest.approx(0.0)
        assert dy > 0  # marker is above, need to move down to center

    def test_marker_below(self):
        """Marker below centre → negative dy."""
        result = compute_follow_vector(
            [_april("1", (320, 400))],
            frame_width=640,
            frame_height=480,
        )
        dx, dy = result.position_vector
        assert dy < 0

    def test_target_selection_specific_id(self):
        """When target_marker_id is set, use that marker."""
        detections = [
            _april("1", (100, 100)),
            _april("5", (400, 300)),
        ]
        result = compute_follow_vector(
            detections,
            frame_width=640,
            frame_height=480,
            target_marker_id="5",
        )
        assert result.target_marker_id == "5"
        assert result.target_found is True

    def test_target_not_found_falls_back_to_first(self):
        """When target_marker_id is not visible, fall back to first."""
        detections = [
            _april("3", (100, 100)),
            _april("7", (200, 200)),
        ]
        result = compute_follow_vector(
            detections,
            frame_width=640,
            frame_height=480,
            target_marker_id="99",
        )
        assert result.target_marker_id == "3"
        assert result.target_found is True

    def test_visible_marker_ids_sorted(self):
        detections = [
            _april("5", (100, 100)),
            _april("1", (200, 200)),
            _april("3", (300, 300)),
        ]
        result = compute_follow_vector(
            detections, frame_width=640, frame_height=480
        )
        assert result.visible_marker_ids == ["1", "3", "5"]

    def test_marker_positions_populated(self):
        detections = [_april("1", (100, 200))]
        result = compute_follow_vector(
            detections, frame_width=640, frame_height=480
        )
        assert result.marker_positions == {"1": (100, 200)}

    def test_non_apriltag_ignored(self):
        """Laser spots should not affect the follow vector."""
        detections = [
            _april("1", (320, 240)),
            _laser((100, 100)),
        ]
        result = compute_follow_vector(
            detections, frame_width=640, frame_height=480
        )
        assert result.target_marker_id == "1"
        assert len(result.visible_marker_ids) == 1


# ---------------------------------------------------------------------------
# AutoFollowScenario – interactive wrapper
# ---------------------------------------------------------------------------


class TestAutoFollowScenario:
    """Tests for the AutoFollowScenario class with mocked camera/detector."""

    def test_initial_target_marker_id_is_none(self):
        cam = MagicMock()
        det = MagicMock()
        s = AutoFollowScenario(camera=cam, detector=det)
        assert s.target_marker_id is None

    def test_set_target_marker_id(self):
        cam = MagicMock()
        det = MagicMock()
        s = AutoFollowScenario(camera=cam, detector=det)
        s.target_marker_id = "5"
        assert s.target_marker_id == "5"

    def test_compute_from_detections(self):
        cam = MagicMock()
        det = MagicMock()
        s = AutoFollowScenario(
            camera=cam, detector=det,
            frame_width=640, frame_height=480,
        )
        detections = [_april("1", (200, 100))]
        result = s.compute_from_detections(detections)
        assert result.target_found is True
        assert result.target_marker_id == "1"

    def test_compute_current_uses_camera(self):
        cam = MagicMock()
        det = MagicMock()
        cam.read.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        det.process_frame.return_value = [_april("1", (320, 240))]
        s = AutoFollowScenario(
            camera=cam, detector=det,
            frame_width=640, frame_height=480,
        )
        result = s.compute_current()
        assert result.target_found is True
        cam.read.assert_called_once()
        det.process_frame.assert_called_once()

    def test_compute_current_camera_none_raises(self):
        cam = MagicMock()
        det = MagicMock()
        cam.read.return_value = None
        s = AutoFollowScenario(camera=cam, detector=det)
        with pytest.raises(RuntimeError, match="no frame"):
            s.compute_current()

    def test_target_marker_id_passed_through(self):
        cam = MagicMock()
        det = MagicMock()
        s = AutoFollowScenario(
            camera=cam, detector=det,
            frame_width=640, frame_height=480,
            target_marker_id="3",
        )
        detections = [_april("1", (100, 100)), _april("3", (400, 300))]
        result = s.compute_from_detections(detections)
        assert result.target_marker_id == "3"


# ---------------------------------------------------------------------------
# CLI --mode follow (replaces legacy --mode auto)
# ---------------------------------------------------------------------------


class TestCLIScenarioFollow:
    """Verify the --mode follow argument is accepted by the CLI parser."""

    def test_parse_scenario_follow(self):
        from main import _parse_args

        args = _parse_args(["--mode", "follow", "--source", "0"])
        assert args.mode == "follow"

    def test_parse_follow_marker(self):
        from main import _parse_args

        args = _parse_args(["--mode", "follow", "--follow-marker", "5"])
        assert args.follow_marker == "5"

    def test_parse_follow_marker_default_none(self):
        from main import _parse_args

        args = _parse_args(["--mode", "follow"])
        assert args.follow_marker is None

    def test_parse_follow_box_flag(self):
        from main import _parse_args

        args = _parse_args(["--mode", "follow", "--follow-box"])
        assert args.follow_box is True

    def test_parse_target_distance(self):
        from main import _parse_args

        args = _parse_args(["--mode", "follow", "--target-distance", "0.8"])
        assert args.target_distance == pytest.approx(0.8)

    def test_scenario_follow_headless_with_mock_video(self, capsys, tmp_path):
        """Integration: --mode follow --headless with a tiny synthetic video."""
        cv2 = pytest.importorskip("cv2")

        video_path = tmp_path / "follow_test.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, 1, (64, 64))
        for _ in range(3):
            writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
        writer.release()

        with patch(
            "robo_vision.april_tag_detector._apriltags_available",
            return_value=False,
        ):
            from main import main

            rc = main([
                "--mode", "follow",
                "--headless",
                "--source", str(video_path),
            ])

        assert rc == 0
        captured = capsys.readouterr()
        assert "follow" in captured.out.lower()

    def test_auto_mode_no_longer_valid(self):
        """--mode auto should now be rejected by argparse."""
        from main import _parse_args

        with pytest.raises(SystemExit):
            _parse_args(["--mode", "auto"])
