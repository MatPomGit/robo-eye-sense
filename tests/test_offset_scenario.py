"""Tests for the camera-offset calibration scenario.

The tests verify the pure computation logic in
:func:`~robo_eye_sense.offset_scenario.compute_offset` and the
:class:`~robo_eye_sense.offset_scenario.CameraOffsetScenario` wrapper
using synthetic detections (no real camera or AprilTag images required).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from robo_eye_sense.offset_scenario import (
    CameraOffsetScenario,
    OffsetResult,
    TAG_PHYSICAL_SIZE_CM,
    _tag_apparent_size_px,
    compute_offset,
    estimate_focal_length_px,
    estimate_tag_distance_cm,
)
from robo_eye_sense.results import Detection, DetectionType


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


def _april_with_corners(
    tag_id: str,
    center: tuple[int, int],
    half_size: int = 25,
) -> Detection:
    """AprilTag Detection with a square bounding box around *center*."""
    cx, cy = center
    corners = [
        (cx - half_size, cy - half_size),
        (cx + half_size, cy - half_size),
        (cx + half_size, cy + half_size),
        (cx - half_size, cy + half_size),
    ]
    return Detection(
        detection_type=DetectionType.APRIL_TAG,
        identifier=tag_id,
        center=center,
        corners=corners,
    )


# ---------------------------------------------------------------------------
# compute_offset – pure function
# ---------------------------------------------------------------------------


class TestComputeOffset:
    """Unit tests for the stateless compute_offset function."""

    def test_no_common_tags_returns_zero_offset(self):
        ref = [_april("1", (100, 100))]
        cur = [_april("2", (200, 200))]
        result = compute_offset(ref, cur)
        assert result.offset == (0.0, 0.0)
        assert result.matched_tags == 0

    def test_empty_reference_returns_zero_offset(self):
        result = compute_offset([], [_april("1", (50, 50))])
        assert result.offset == (0.0, 0.0)
        assert result.matched_tags == 0

    def test_empty_current_returns_zero_offset(self):
        result = compute_offset([_april("1", (50, 50))], [])
        assert result.offset == (0.0, 0.0)
        assert result.matched_tags == 0

    def test_both_empty_returns_zero_offset(self):
        result = compute_offset([], [])
        assert result.offset == (0.0, 0.0)
        assert result.matched_tags == 0

    def test_single_common_tag_offset(self):
        ref = [_april("1", (100, 200))]
        cur = [_april("1", (120, 210))]
        result = compute_offset(ref, cur)
        assert result.matched_tags == 1
        # offset = ref - cur  → (100-120, 200-210) = (-20, -10)
        assert result.offset == (-20.0, -10.0)

    def test_two_common_tags_averaged(self):
        ref = [_april("1", (100, 100)), _april("2", (200, 200))]
        cur = [_april("1", (110, 110)), _april("2", (220, 220))]
        result = compute_offset(ref, cur)
        assert result.matched_tags == 2
        # tag 1: (100-110, 100-110) = (-10, -10)
        # tag 2: (200-220, 200-220) = (-20, -20)
        # avg = (-15, -15)
        assert result.offset == pytest.approx((-15.0, -15.0))

    def test_identical_positions_yield_zero(self):
        ref = [_april("5", (300, 400))]
        cur = [_april("5", (300, 400))]
        result = compute_offset(ref, cur)
        assert result.matched_tags == 1
        assert result.offset == (0.0, 0.0)

    def test_per_tag_offsets_populated(self):
        ref = [_april("1", (10, 20)), _april("2", (30, 40))]
        cur = [_april("1", (15, 25)), _april("2", (35, 45))]
        result = compute_offset(ref, cur)
        assert "1" in result.per_tag_offsets
        assert "2" in result.per_tag_offsets
        assert result.per_tag_offsets["1"] == (-5.0, -5.0)
        assert result.per_tag_offsets["2"] == (-5.0, -5.0)

    def test_non_apriltag_detections_ignored(self):
        ref = [_april("1", (100, 100)), _laser((50, 50))]
        cur = [_april("1", (110, 110)), _laser((60, 60))]
        result = compute_offset(ref, cur)
        assert result.matched_tags == 1
        assert "1" in result.per_tag_offsets

    def test_reference_and_current_positions_recorded(self):
        ref = [_april("3", (10, 20))]
        cur = [_april("3", (30, 40))]
        result = compute_offset(ref, cur)
        assert result.reference_positions == {"3": (10, 20)}
        assert result.current_positions == {"3": (30, 40)}

    def test_partial_overlap(self):
        """Only common tags contribute to the offset."""
        ref = [_april("1", (100, 100)), _april("2", (200, 200))]
        cur = [_april("2", (210, 220)), _april("3", (300, 300))]
        result = compute_offset(ref, cur)
        assert result.matched_tags == 1
        assert result.offset == (-10.0, -20.0)
        assert "2" in result.per_tag_offsets
        assert "1" not in result.per_tag_offsets
        assert "3" not in result.per_tag_offsets


# ---------------------------------------------------------------------------
# OffsetResult dataclass
# ---------------------------------------------------------------------------


class TestOffsetResult:
    def test_default_values(self):
        r = OffsetResult()
        assert r.offset == (0.0, 0.0)
        assert r.matched_tags == 0
        assert r.per_tag_offsets == {}

    def test_custom_values(self):
        r = OffsetResult(offset=(1.5, -2.5), matched_tags=3)
        assert r.offset == (1.5, -2.5)
        assert r.matched_tags == 3


# ---------------------------------------------------------------------------
# CameraOffsetScenario – interactive wrapper
# ---------------------------------------------------------------------------


class TestCameraOffsetScenario:
    """Tests for the CameraOffsetScenario class with mocked camera/detector."""

    @staticmethod
    def _make_scenario(ref_detections, cur_detections):
        """Return a scenario with mocked camera and detector."""
        cam = MagicMock()
        det = MagicMock()

        # Camera returns a dummy frame on every read()
        cam.read.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        # First process_frame call returns ref detections,
        # second returns current detections.
        det.process_frame.side_effect = [ref_detections, cur_detections]

        return CameraOffsetScenario(camera=cam, detector=det)

    def test_has_reference_initially_false(self):
        cam = MagicMock()
        det = MagicMock()
        s = CameraOffsetScenario(camera=cam, detector=det)
        assert s.has_reference is False
        assert s.reference_detections is None

    def test_capture_reference_sets_flag(self):
        s = self._make_scenario([_april("1", (10, 10))], [])
        ref = s.capture_reference()
        assert s.has_reference is True
        assert len(ref) == 1

    def test_compute_current_offset_without_reference_raises(self):
        cam = MagicMock()
        det = MagicMock()
        s = CameraOffsetScenario(camera=cam, detector=det)
        with pytest.raises(RuntimeError, match="No reference"):
            s.compute_current_offset()

    def test_capture_reference_camera_returns_none_raises(self):
        cam = MagicMock()
        cam.read.return_value = None
        det = MagicMock()
        s = CameraOffsetScenario(camera=cam, detector=det)
        with pytest.raises(RuntimeError, match="no frame"):
            s.capture_reference()

    def test_compute_current_camera_returns_none_raises(self):
        cam = MagicMock()
        det = MagicMock()
        # First read → valid frame (reference); second → None
        cam.read.side_effect = [
            np.zeros((10, 10, 3), dtype=np.uint8),
            None,
        ]
        det.process_frame.return_value = [_april("1", (50, 50))]
        s = CameraOffsetScenario(camera=cam, detector=det)
        s.capture_reference()
        with pytest.raises(RuntimeError, match="no frame"):
            s.compute_current_offset()

    def test_full_workflow_returns_offset(self):
        ref = [_april("1", (100, 100)), _april("2", (200, 200))]
        cur = [_april("1", (110, 120)), _april("2", (210, 220))]
        s = self._make_scenario(ref, cur)
        s.capture_reference()
        result = s.compute_current_offset()
        assert result.matched_tags == 2
        assert result.offset == pytest.approx((-10.0, -20.0))

    def test_reset_clears_reference(self):
        s = self._make_scenario([_april("1", (10, 10))], [])
        s.capture_reference()
        assert s.has_reference is True
        s.reset()
        assert s.has_reference is False
        assert s.reference_detections is None


# ---------------------------------------------------------------------------
# CLI --scenario offset
# ---------------------------------------------------------------------------


class TestCLIScenarioOffset:
    """Verify the --scenario offset argument is accepted by the CLI parser."""

    def test_parse_scenario_offset(self):
        from main import _parse_args

        args = _parse_args(["--scenario", "offset", "--source", "0"])
        assert args.scenario == "offset"

    def test_parse_no_scenario(self):
        from main import _parse_args

        args = _parse_args(["--source", "0"])
        assert args.scenario is None

    def test_scenario_offset_runs_with_mock_camera(self, capsys, tmp_path):
        """Full integration: --scenario offset with a tiny synthetic video."""
        cv2 = pytest.importorskip("cv2")

        # Create a tiny 2-frame video (black frames – no real AprilTags)
        video_path = tmp_path / "offset_test.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, 1, (64, 64))
        for _ in range(2):
            writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
        writer.release()

        # Patch input() to avoid blocking and _apriltags_available
        with (
            patch("builtins.input", return_value=""),
            patch(
                "robo_eye_sense.april_tag_detector._apriltags_available",
                return_value=False,
            ),
        ):
            from main import main

            rc = main(["--scenario", "offset", "--source", str(video_path)])

        assert rc == 0
        captured = capsys.readouterr()
        assert "Camera-offset result" in captured.out
        assert "Matched AprilTags" in captured.out


# ---------------------------------------------------------------------------
# Distance estimation helpers
# ---------------------------------------------------------------------------


class TestTagApparentSizePx:
    """Unit tests for _tag_apparent_size_px."""

    def test_empty_corners_returns_zero(self):
        assert _tag_apparent_size_px([]) == 0.0

    def test_fewer_than_four_corners_returns_zero(self):
        assert _tag_apparent_size_px([(0, 0), (10, 0), (10, 10)]) == 0.0

    def test_unit_square(self):
        corners = [(0, 0), (10, 0), (10, 10), (0, 10)]
        assert _tag_apparent_size_px(corners) == pytest.approx(10.0)

    def test_rectangle(self):
        corners = [(0, 0), (20, 0), (20, 10), (0, 10)]
        # sides: 20, 10, 20, 10 → avg = 15
        assert _tag_apparent_size_px(corners) == pytest.approx(15.0)


class TestEstimateFocalLengthPx:
    """Unit tests for estimate_focal_length_px."""

    def test_60_degree_fov(self):
        # f = (640 / 2) / tan(30°) ≈ 320 / 0.57735 ≈ 554.26
        f = estimate_focal_length_px(640, 60.0)
        assert f == pytest.approx(554.256, rel=1e-2)

    def test_90_degree_fov(self):
        # f = (640 / 2) / tan(45°) = 320
        f = estimate_focal_length_px(640, 90.0)
        assert f == pytest.approx(320.0, rel=1e-6)

    def test_wider_frame(self):
        f1 = estimate_focal_length_px(640, 60.0)
        f2 = estimate_focal_length_px(1280, 60.0)
        assert f2 == pytest.approx(2 * f1, rel=1e-6)


class TestEstimateTagDistanceCm:
    """Unit tests for estimate_tag_distance_cm."""

    def test_no_corners_returns_none(self):
        assert estimate_tag_distance_cm([], 500.0) is None

    def test_fewer_than_four_corners_returns_none(self):
        assert estimate_tag_distance_cm([(0, 0), (10, 0)], 500.0) is None

    def test_known_distance(self):
        # If the tag is 50 px wide and focal length is 500 px, with 5 cm tag:
        # distance = (5 * 500) / 50 = 50 cm
        corners = [(0, 0), (50, 0), (50, 50), (0, 50)]
        dist = estimate_tag_distance_cm(corners, 500.0, tag_size_cm=5.0)
        assert dist == pytest.approx(50.0)

    def test_smaller_tag_appears_farther(self):
        # Smaller apparent size → larger distance
        corners_big = [(0, 0), (100, 0), (100, 100), (0, 100)]
        corners_small = [(0, 0), (50, 0), (50, 50), (0, 50)]
        dist_big = estimate_tag_distance_cm(corners_big, 500.0)
        dist_small = estimate_tag_distance_cm(corners_small, 500.0)
        assert dist_small > dist_big


# ---------------------------------------------------------------------------
# compute_offset with distance estimation
# ---------------------------------------------------------------------------


class TestComputeOffsetWithDistance:
    """Tests that compute_offset populates distance fields."""

    def test_per_tag_distances_populated(self):
        ref = [_april_with_corners("1", (100, 100), half_size=25)]
        cur = [_april_with_corners("1", (110, 110), half_size=25)]
        result = compute_offset(ref, cur, frame_width=640)
        assert "1" in result.per_tag_distances_cm
        assert result.per_tag_distances_cm["1"] > 0

    def test_distance_to_reference_populated(self):
        ref = [_april_with_corners("1", (100, 100), half_size=25)]
        cur = [_april_with_corners("1", (120, 130), half_size=25)]
        result = compute_offset(ref, cur, frame_width=640)
        assert result.distance_to_reference_cm is not None
        assert result.distance_to_reference_cm > 0

    def test_identical_positions_zero_distance_to_ref(self):
        ref = [_april_with_corners("1", (200, 200), half_size=30)]
        cur = [_april_with_corners("1", (200, 200), half_size=30)]
        result = compute_offset(ref, cur, frame_width=640)
        assert result.distance_to_reference_cm is not None
        assert result.distance_to_reference_cm == pytest.approx(0.0, abs=0.01)

    def test_no_corners_means_no_distances(self):
        ref = [_april("1", (100, 100))]
        cur = [_april("1", (110, 110))]
        result = compute_offset(ref, cur, frame_width=640)
        # Tags without corners can't estimate distance
        assert len(result.per_tag_distances_cm) == 0
        assert result.distance_to_reference_cm is None

    def test_no_common_tags_still_reports_distances(self):
        ref = [_april_with_corners("1", (100, 100))]
        cur = [_april_with_corners("2", (200, 200))]
        result = compute_offset(ref, cur, frame_width=640)
        assert result.matched_tags == 0
        # current tag "2" should still have a distance estimate
        assert "2" in result.per_tag_distances_cm


# ---------------------------------------------------------------------------
# CameraOffsetScenario.compute_offset_from_detections
# ---------------------------------------------------------------------------


class TestComputeOffsetFromDetections:
    """Tests for the new compute_offset_from_detections method."""

    def test_without_reference_raises(self):
        cam = MagicMock()
        det = MagicMock()
        s = CameraOffsetScenario(camera=cam, detector=det)
        with pytest.raises(RuntimeError, match="No reference"):
            s.compute_offset_from_detections([_april("1", (50, 50))])

    def test_returns_offset_result(self):
        cam = MagicMock()
        det = MagicMock()
        cam.read.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        det.process_frame.return_value = [_april("1", (100, 100))]

        s = CameraOffsetScenario(camera=cam, detector=det, frame_width=640)
        s.capture_reference()

        cur = [_april("1", (110, 120))]
        result = s.compute_offset_from_detections(cur)
        assert result.matched_tags == 1
        assert result.offset == (-10.0, -20.0)
