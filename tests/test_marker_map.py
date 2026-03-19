"""Tests for the SLAM-based marker map module.

The tests verify the data structures (:class:`MarkerPose3D`,
:class:`RobotPose3D`, :class:`MarkerMap`) and the
:class:`SlamCalibrator` using synthetic detections — no real camera or
AprilTag images are required.
"""

from __future__ import annotations

import json
import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from robo_vision.marker_map import (
    MarkerMap,
    MarkerPose3D,
    RobotPose3D,
    SlamCalibrator,
    _angle_average,
    _default_camera_matrix,
    _estimate_pose_multi_marker,
    _euler_to_rotation_matrix,
    _mean_angles,
    _rotation_matrix_to_euler,
    _solve_marker_pose,
)
from robo_vision.results import Detection, DetectionType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _april(
    tag_id: str,
    center: tuple[int, int],
    half_size: int = 25,
) -> Detection:
    """Create an AprilTag Detection with a square bounding box."""
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


def _april_no_corners(tag_id: str, center: tuple[int, int]) -> Detection:
    """AprilTag Detection without corners."""
    return Detection(
        detection_type=DetectionType.APRIL_TAG,
        identifier=tag_id,
        center=center,
        corners=[],
    )


def _laser(center: tuple[int, int]) -> Detection:
    """Laser-spot Detection."""
    return Detection(
        detection_type=DetectionType.LASER_SPOT,
        identifier=None,
        center=center,
    )


# ---------------------------------------------------------------------------
# MarkerPose3D
# ---------------------------------------------------------------------------


class TestMarkerPose3D:
    def test_defaults(self):
        m = MarkerPose3D(marker_id="1")
        assert m.marker_id == "1"
        assert m.position == (0.0, 0.0, 0.0)
        assert m.orientation == (0.0, 0.0, 0.0)
        assert m.observations == 1

    def test_custom_values(self):
        m = MarkerPose3D(
            marker_id="42",
            position=(1.0, 2.0, 3.0),
            orientation=(10.0, 20.0, 30.0),
            observations=5,
        )
        assert m.marker_id == "42"
        assert m.position == (1.0, 2.0, 3.0)
        assert m.orientation == (10.0, 20.0, 30.0)
        assert m.observations == 5


# ---------------------------------------------------------------------------
# RobotPose3D
# ---------------------------------------------------------------------------


class TestRobotPose3D:
    def test_defaults(self):
        r = RobotPose3D()
        assert r.position == (0.0, 0.0, 0.0)
        assert r.orientation == (0.0, 0.0, 0.0)
        assert r.visible_markers == 0
        assert r.reprojection_error is None

    def test_custom_values(self):
        r = RobotPose3D(
            position=(10.0, 20.0, 30.0),
            orientation=(5.0, 10.0, 15.0),
            visible_markers=3,
            reprojection_error=0.5,
        )
        assert r.position == (10.0, 20.0, 30.0)
        assert r.visible_markers == 3
        assert r.reprojection_error == 0.5


# ---------------------------------------------------------------------------
# MarkerMap — basic operations
# ---------------------------------------------------------------------------


class TestMarkerMap:
    def test_empty_map(self):
        mm = MarkerMap()
        assert len(mm) == 0
        assert mm.marker_ids == []
        assert mm.markers() == []

    def test_add_and_get(self):
        mm = MarkerMap()
        pose = MarkerPose3D(marker_id="1", position=(1.0, 2.0, 3.0))
        mm.add(pose)
        assert len(mm) == 1
        assert "1" in mm
        assert mm.get("1") is pose
        assert mm.get("999") is None

    def test_marker_ids_sorted(self):
        mm = MarkerMap()
        mm.add(MarkerPose3D(marker_id="3"))
        mm.add(MarkerPose3D(marker_id="1"))
        mm.add(MarkerPose3D(marker_id="2"))
        assert mm.marker_ids == ["1", "2", "3"]

    def test_remove(self):
        mm = MarkerMap()
        mm.add(MarkerPose3D(marker_id="1"))
        assert mm.remove("1") is True
        assert len(mm) == 0
        assert mm.remove("1") is False

    def test_clear(self):
        mm = MarkerMap()
        mm.add(MarkerPose3D(marker_id="1"))
        mm.add(MarkerPose3D(marker_id="2"))
        mm.clear()
        assert len(mm) == 0

    def test_contains(self):
        mm = MarkerMap()
        mm.add(MarkerPose3D(marker_id="5"))
        assert "5" in mm
        assert "6" not in mm


# ---------------------------------------------------------------------------
# MarkerMap — merge_observation
# ---------------------------------------------------------------------------


class TestMarkerMapMerge:
    def test_merge_new_marker(self):
        mm = MarkerMap()
        mm.merge_observation("1", (10.0, 20.0, 30.0), (0.0, 0.0, 0.0))
        assert "1" in mm
        m = mm.get("1")
        assert m is not None
        assert m.position == (10.0, 20.0, 30.0)
        assert m.observations == 1

    def test_merge_existing_marker_averages_position(self):
        mm = MarkerMap()
        mm.merge_observation("1", (10.0, 20.0, 30.0), (0.0, 0.0, 0.0))
        mm.merge_observation("1", (20.0, 40.0, 60.0), (0.0, 0.0, 0.0))
        m = mm.get("1")
        assert m is not None
        assert m.observations == 2
        # Running average: (10 * 1 + 20) / (1 + 1) = 15, (20 * 1 + 40) / (1 + 1) = 30, (30 * 1 + 60) / (1 + 1) = 45
        assert m.position == pytest.approx((15.0, 30.0, 45.0))

    def test_merge_three_observations(self):
        mm = MarkerMap()
        mm.merge_observation("1", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        mm.merge_observation("1", (3.0, 6.0, 9.0), (0.0, 0.0, 0.0))
        mm.merge_observation("1", (6.0, 12.0, 18.0), (0.0, 0.0, 0.0))
        m = mm.get("1")
        assert m is not None
        assert m.observations == 3
        # (0*1+3)/2=1.5, (1.5*2+6)/3=3.0
        assert m.position[0] == pytest.approx(3.0)

    def test_merge_preserves_id(self):
        mm = MarkerMap()
        mm.merge_observation("abc", (1.0, 2.0, 3.0), (0.0, 0.0, 0.0))
        mm.merge_observation("abc", (4.0, 5.0, 6.0), (0.0, 0.0, 0.0))
        assert mm.get("abc") is not None
        assert mm.get("abc").marker_id == "abc"


# ---------------------------------------------------------------------------
# MarkerMap — serialisation
# ---------------------------------------------------------------------------


class TestMarkerMapSerialisation:
    def test_to_dict_empty(self):
        mm = MarkerMap()
        d = mm.to_dict()
        assert d == {"markers": []}

    def test_to_dict_with_markers(self):
        mm = MarkerMap()
        mm.add(MarkerPose3D(
            marker_id="1",
            position=(1.0, 2.0, 3.0),
            orientation=(10.0, 20.0, 30.0),
            observations=5,
        ))
        d = mm.to_dict()
        assert len(d["markers"]) == 1
        assert d["markers"][0]["marker_id"] == "1"
        assert d["markers"][0]["position"] == (1.0, 2.0, 3.0)

    def test_from_dict_roundtrip(self):
        mm = MarkerMap()
        mm.add(MarkerPose3D(
            marker_id="1",
            position=(1.0, 2.0, 3.0),
            orientation=(45.0, 0.0, 90.0),
            observations=3,
        ))
        mm.add(MarkerPose3D(
            marker_id="2",
            position=(4.0, 5.0, 6.0),
        ))
        data = mm.to_dict()
        mm2 = MarkerMap.from_dict(data)
        assert len(mm2) == 2
        assert mm2.get("1").position == pytest.approx((1.0, 2.0, 3.0))
        assert mm2.get("1").observations == 3
        assert mm2.get("2").position == pytest.approx((4.0, 5.0, 6.0))

    def test_save_and_load(self, tmp_path):
        mm = MarkerMap()
        mm.add(MarkerPose3D(marker_id="42", position=(7.0, 8.0, 9.0)))
        path = tmp_path / "test_map.json"
        mm.save(path)

        mm2 = MarkerMap.load(path)
        assert len(mm2) == 1
        assert mm2.get("42").position == pytest.approx((7.0, 8.0, 9.0))

    def test_save_creates_valid_json(self, tmp_path):
        mm = MarkerMap()
        mm.add(MarkerPose3D(marker_id="1"))
        path = tmp_path / "map.json"
        mm.save(path)

        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        assert "markers" in data

    def test_from_dict_missing_observations(self):
        """Observations field may be absent in older files."""
        data = {
            "markers": [
                {
                    "marker_id": "1",
                    "position": [1, 2, 3],
                    "orientation": [0, 0, 0],
                }
            ]
        }
        mm = MarkerMap.from_dict(data)
        assert mm.get("1").observations == 1


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------


class TestRotationHelpers:
    def test_euler_identity(self):
        R = _euler_to_rotation_matrix(0, 0, 0)
        assert R == pytest.approx(np.eye(3), abs=1e-10)

    def test_euler_roundtrip(self):
        for r, p, y in [(10, 20, 30), (0, 0, 90), (-45, 0, 45)]:
            R = _euler_to_rotation_matrix(r, p, y)
            r2, p2, y2 = _rotation_matrix_to_euler(R)
            assert r2 == pytest.approx(r, abs=0.1)
            assert p2 == pytest.approx(p, abs=0.1)
            assert y2 == pytest.approx(y, abs=0.1)

    def test_rotation_matrix_determinant_is_one(self):
        R = _euler_to_rotation_matrix(45, -30, 60)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-10)

    def test_rotation_matrix_orthogonal(self):
        R = _euler_to_rotation_matrix(10, 20, 30)
        assert R @ R.T == pytest.approx(np.eye(3), abs=1e-10)


class TestAngleAverage:
    def test_same_angle(self):
        result = _angle_average(45.0, 45.0, 1)
        assert result == pytest.approx(45.0, abs=0.1)

    def test_zero_weight(self):
        """When n=0 the old value has no influence."""
        result = _angle_average(0.0, 90.0, 0)
        assert result == pytest.approx(90.0, abs=0.1)

    def test_weighted_average(self):
        result = _angle_average(0.0, 90.0, 1)
        assert result == pytest.approx(45.0, abs=0.1)

    def test_wrapping(self):
        """Average of 350° and 10° should be near 0°."""
        result = _angle_average(350.0, 10.0, 1)
        assert abs(result) < 5 or abs(result - 360) < 5


class TestMeanAngles:
    def test_empty(self):
        assert _mean_angles([]) == 0.0

    def test_single(self):
        assert _mean_angles([90.0]) == pytest.approx(90.0, abs=0.1)

    def test_symmetric(self):
        result = _mean_angles([0.0, 90.0])
        assert result == pytest.approx(45.0, abs=0.1)

    def test_wrapping(self):
        result = _mean_angles([350.0, 10.0])
        assert abs(result) < 5 or abs(result - 360) < 5


# ---------------------------------------------------------------------------
# MarkerMap.estimate_robot_pose
# ---------------------------------------------------------------------------


class TestEstimateRobotPose:
    def test_no_detections_returns_empty(self):
        mm = MarkerMap()
        mm.add(MarkerPose3D(marker_id="1", position=(10, 0, 50)))
        pose = mm.estimate_robot_pose([])
        assert pose.visible_markers == 0

    def test_non_apriltag_ignored(self):
        mm = MarkerMap()
        mm.add(MarkerPose3D(marker_id="1", position=(10, 0, 50)))
        pose = mm.estimate_robot_pose([_laser((100, 100))])
        assert pose.visible_markers == 0

    def test_unknown_marker_ignored(self):
        mm = MarkerMap()
        mm.add(MarkerPose3D(marker_id="1", position=(10, 0, 50)))
        det = _april("999", (320, 240))
        pose = mm.estimate_robot_pose([det])
        assert pose.visible_markers == 0

    def test_marker_without_corners_ignored(self):
        mm = MarkerMap()
        mm.add(MarkerPose3D(marker_id="1", position=(10, 0, 50)))
        det = _april_no_corners("1", (320, 240))
        pose = mm.estimate_robot_pose([det])
        assert pose.visible_markers == 0

    def test_known_marker_produces_pose(self):
        """A detected marker that exists in the map should produce a pose."""
        mm = MarkerMap()
        mm.add(MarkerPose3D(
            marker_id="1",
            position=(0.0, 0.0, 50.0),
            orientation=(0.0, 0.0, 0.0),
        ))
        det = _april("1", (320, 240), half_size=30)
        pose = mm.estimate_robot_pose(
            [det],
            tag_size_cm=5.0,
        )
        # We should get some kind of result (the exact values depend on
        # the solvePnP estimate with a synthetic camera matrix).
        assert pose.visible_markers == 1
        assert pose.reprojection_error is not None


# ---------------------------------------------------------------------------
# SlamCalibrator
# ---------------------------------------------------------------------------


class TestSlamCalibrator:
    def test_initial_state(self):
        sc = SlamCalibrator(tag_size_cm=5.0)
        assert sc.frame_count == 0
        assert len(sc.marker_map) == 0

    def test_first_frame_establishes_map(self):
        sc = SlamCalibrator(tag_size_cm=5.0)
        dets = [_april("1", (320, 240), half_size=30)]
        result = sc.process_detections(dets)
        assert sc.frame_count == 1
        assert len(sc.marker_map) >= 1
        assert "1" in sc.marker_map
        assert result.visible_markers >= 1

    def test_empty_detections_no_crash(self):
        sc = SlamCalibrator(tag_size_cm=5.0)
        result = sc.process_detections([])
        assert sc.frame_count == 1
        assert result.visible_markers == 0

    def test_non_april_detections_ignored(self):
        sc = SlamCalibrator(tag_size_cm=5.0)
        result = sc.process_detections([_laser((100, 100))])
        assert len(sc.marker_map) == 0

    def test_detections_without_corners_ignored(self):
        sc = SlamCalibrator(tag_size_cm=5.0)
        result = sc.process_detections([_april_no_corners("1", (100, 100))])
        assert len(sc.marker_map) == 0

    def test_multiple_frames_grow_map(self):
        sc = SlamCalibrator(tag_size_cm=5.0)
        # First frame: marker 1
        sc.process_detections([_april("1", (320, 240), half_size=30)])
        assert "1" in sc.marker_map

        # Second frame: marker 1 + marker 2
        sc.process_detections([
            _april("1", (300, 230), half_size=28),
            _april("2", (500, 300), half_size=25),
        ])
        assert sc.frame_count == 2
        assert "2" in sc.marker_map

    def test_reset_clears_state(self):
        sc = SlamCalibrator(tag_size_cm=5.0)
        sc.process_detections([_april("1", (320, 240))])
        assert len(sc.marker_map) > 0
        sc.reset()
        assert len(sc.marker_map) == 0
        assert sc.frame_count == 0

    def test_custom_camera_matrix(self):
        cam_mtx = [
            [500.0, 0.0, 320.0],
            [0.0, 500.0, 240.0],
            [0.0, 0.0, 1.0],
        ]
        sc = SlamCalibrator(tag_size_cm=5.0, camera_matrix=cam_mtx)
        dets = [_april("1", (320, 240), half_size=30)]
        result = sc.process_detections(dets)
        assert len(sc.marker_map) >= 1

    def test_marker_map_property(self):
        sc = SlamCalibrator()
        assert isinstance(sc.marker_map, MarkerMap)

    def test_re_observation_refines_pose(self):
        sc = SlamCalibrator(tag_size_cm=5.0)
        det1 = _april("1", (320, 240), half_size=30)
        sc.process_detections([det1])
        obs_before = sc.marker_map.get("1").observations

        # Second frame — same marker, slightly different position
        det2 = _april("1", (325, 245), half_size=31)
        sc.process_detections([det2])
        obs_after = sc.marker_map.get("1").observations
        assert obs_after > obs_before


# ---------------------------------------------------------------------------
# _default_camera_matrix
# ---------------------------------------------------------------------------


class TestDefaultCameraMatrix:
    def test_shape(self):
        mtx = _default_camera_matrix(640, 480)
        assert len(mtx) == 3
        assert all(len(row) == 3 for row in mtx)

    def test_principal_point_at_image_centre(self):
        mtx = _default_camera_matrix(640, 480)
        assert mtx[0][2] == pytest.approx(320.0)
        assert mtx[1][2] == pytest.approx(240.0)

    def test_focal_length_positive(self):
        mtx = _default_camera_matrix(640, 480)
        assert mtx[0][0] > 0
        assert mtx[1][1] > 0

    def test_focal_lengths_equal(self):
        mtx = _default_camera_matrix(640, 480)
        assert mtx[0][0] == pytest.approx(mtx[1][1])

    def test_scales_with_width(self):
        mtx_wide = _default_camera_matrix(1280, 720)
        mtx_narrow = _default_camera_matrix(640, 480)
        # Wider image → larger focal length (same HFOV)
        assert mtx_wide[0][0] > mtx_narrow[0][0]


# ---------------------------------------------------------------------------
# SlamCalibrator – frame_size parameter
# ---------------------------------------------------------------------------


class TestSlamCalibratorFrameSize:
    def test_frame_size_sets_camera_matrix(self):
        """Passing frame_size should compute a non-None camera matrix."""
        sc = SlamCalibrator(tag_size_cm=5.0, frame_size=(640, 480))
        assert sc._camera_matrix is not None
        assert len(sc._camera_matrix) == 3

    def test_frame_size_ignored_when_matrix_given(self):
        """Explicit camera_matrix takes priority over frame_size."""
        explicit = [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
        sc = SlamCalibrator(
            tag_size_cm=5.0,
            camera_matrix=explicit,
            frame_size=(1280, 720),
        )
        assert sc._camera_matrix is explicit

    def test_no_frame_size_camera_matrix_none(self):
        """Without frame_size or camera_matrix, _camera_matrix is None."""
        sc = SlamCalibrator(tag_size_cm=5.0)
        assert sc._camera_matrix is None

    def test_frame_size_produces_valid_detections(self):
        """SlamCalibrator with frame_size should still process detections."""
        sc = SlamCalibrator(tag_size_cm=5.0, frame_size=(640, 480))
        dets = [_april("1", (320, 240), half_size=30)]
        result = sc.process_detections(dets)
        assert len(sc.marker_map) >= 1


# ---------------------------------------------------------------------------
# SlamCalibrator – tag_size_cm used correctly
# ---------------------------------------------------------------------------


class TestSlamCalibratorTagSize:
    def test_tag_size_stored(self):
        sc = SlamCalibrator(tag_size_cm=10.0)
        assert sc._tag_size_cm == 10.0

    def test_different_tag_sizes_produce_different_poses(self):
        """Larger physical tag at same apparent size → greater estimated depth."""
        dets = [_april("1", (320, 240), half_size=30)]

        sc_small = SlamCalibrator(tag_size_cm=2.5, frame_size=(640, 480))
        sc_large = SlamCalibrator(tag_size_cm=10.0, frame_size=(640, 480))

        sc_small.process_detections(dets)
        sc_large.process_detections(dets)

        pos_small = sc_small.marker_map.get("1").position
        pos_large = sc_large.marker_map.get("1").position

        # Depth (z) should be larger for the physically larger tag
        assert pos_large[2] > pos_small[2]


# ---------------------------------------------------------------------------
# _solve_marker_pose — RANSAC PnP + LM refinement
# ---------------------------------------------------------------------------


class TestSolveMarkerPoseRansac:
    """Verify that _solve_marker_pose uses RANSAC PnP and returns valid results."""

    def test_returns_valid_pose(self):
        det = _april("1", (320, 240), half_size=30)
        rvec, tvec, err = _solve_marker_pose(det.corners, 5.0, None)
        assert rvec is not None
        assert tvec is not None
        assert err is not None
        assert err >= 0.0

    def test_too_few_corners_returns_none(self):
        rvec, tvec, err = _solve_marker_pose([(0, 0), (1, 1)], 5.0, None)
        assert rvec is None
        assert tvec is None
        assert err is None

    def test_custom_camera_matrix(self):
        cam_mtx = np.array([
            [500.0, 0.0, 320.0],
            [0.0, 500.0, 240.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        det = _april("1", (320, 240), half_size=30)
        rvec, tvec, err = _solve_marker_pose(det.corners, 5.0, cam_mtx)
        assert rvec is not None
        assert tvec is not None

    def test_rvec_tvec_shapes(self):
        det = _april("1", (320, 240), half_size=30)
        rvec, tvec, _ = _solve_marker_pose(det.corners, 5.0, None)
        assert rvec.shape == (3,)
        assert tvec.shape == (3,)

    def test_depth_scales_with_tag_size(self):
        """Larger physical tag at same pixel size → greater depth estimate."""
        det = _april("1", (320, 240), half_size=30)
        _, tvec_small, _ = _solve_marker_pose(det.corners, 2.5, None)
        _, tvec_large, _ = _solve_marker_pose(det.corners, 10.0, None)
        assert tvec_large[2] > tvec_small[2]

    def test_reprojection_error_is_low(self):
        """With ideal synthetic corners the reprojection error should be small."""
        det = _april("1", (320, 240), half_size=30)
        _, _, err = _solve_marker_pose(det.corners, 5.0, None)
        assert err < 2.0  # pixels


# ---------------------------------------------------------------------------
# _estimate_pose_multi_marker — multi-marker RANSAC PnP
# ---------------------------------------------------------------------------


class TestEstimatePoseMultiMarker:
    """Tests for the multi-marker RANSAC PnP helper."""

    def _make_map_and_dets(self):
        """Create a map with two markers and matching detections."""
        markers = {
            "1": MarkerPose3D(
                marker_id="1",
                position=(0.0, 0.0, 50.0),
                orientation=(0.0, 0.0, 0.0),
            ),
            "2": MarkerPose3D(
                marker_id="2",
                position=(30.0, 0.0, 50.0),
                orientation=(0.0, 0.0, 0.0),
            ),
        }
        dets = [
            _april("1", (250, 240), half_size=25),
            _april("2", (450, 240), half_size=25),
        ]
        return markers, dets

    def test_returns_robot_pose(self):
        markers, dets = self._make_map_and_dets()
        result = _estimate_pose_multi_marker(dets, markers, 5.0, None)
        assert result is not None
        assert isinstance(result, RobotPose3D)
        assert result.visible_markers >= 1

    def test_reprojection_error_present(self):
        markers, dets = self._make_map_and_dets()
        result = _estimate_pose_multi_marker(dets, markers, 5.0, None)
        assert result is not None
        assert result.reprojection_error is not None
        assert result.reprojection_error >= 0.0

    def test_orientation_returned(self):
        markers, dets = self._make_map_and_dets()
        result = _estimate_pose_multi_marker(dets, markers, 5.0, None)
        assert result is not None
        assert len(result.orientation) == 3

    def test_custom_camera_matrix(self):
        markers, dets = self._make_map_and_dets()
        cam_mtx = np.array([
            [500.0, 0.0, 320.0],
            [0.0, 500.0, 240.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        result = _estimate_pose_multi_marker(dets, markers, 5.0, cam_mtx)
        assert result is not None
        assert result.visible_markers >= 1


# ---------------------------------------------------------------------------
# MarkerMap.estimate_robot_pose — multi-marker path
# ---------------------------------------------------------------------------


class TestEstimateRobotPoseMultiMarker:
    """Verify estimate_robot_pose uses multi-marker RANSAC when ≥ 2 markers."""

    def test_two_markers_produces_pose(self):
        mm = MarkerMap()
        mm.add(MarkerPose3D(
            marker_id="1",
            position=(0.0, 0.0, 50.0),
            orientation=(0.0, 0.0, 0.0),
        ))
        mm.add(MarkerPose3D(
            marker_id="2",
            position=(30.0, 0.0, 50.0),
            orientation=(0.0, 0.0, 0.0),
        ))
        dets = [
            _april("1", (250, 240), half_size=25),
            _april("2", (450, 240), half_size=25),
        ]
        pose = mm.estimate_robot_pose(dets, tag_size_cm=5.0)
        assert pose.visible_markers >= 1
        assert pose.reprojection_error is not None

    def test_single_marker_still_works(self):
        """With only one known marker, fallback to single-marker PnP."""
        mm = MarkerMap()
        mm.add(MarkerPose3D(
            marker_id="1",
            position=(0.0, 0.0, 50.0),
            orientation=(0.0, 0.0, 0.0),
        ))
        det = _april("1", (320, 240), half_size=30)
        pose = mm.estimate_robot_pose([det], tag_size_cm=5.0)
        assert pose.visible_markers == 1
        assert pose.reprojection_error is not None

    def test_mixed_known_unknown_markers(self):
        """Only mapped markers contribute to the pose estimate."""
        mm = MarkerMap()
        mm.add(MarkerPose3D(
            marker_id="1",
            position=(0.0, 0.0, 50.0),
            orientation=(0.0, 0.0, 0.0),
        ))
        dets = [
            _april("1", (250, 240), half_size=25),
            _april("999", (450, 240), half_size=25),
        ]
        pose = mm.estimate_robot_pose(dets, tag_size_cm=5.0)
        assert pose.visible_markers == 1

    def test_three_markers_uses_multi_marker(self):
        """Three visible mapped markers should use multi-marker RANSAC."""
        mm = MarkerMap()
        for i, x_off in enumerate([0, 30, 60]):
            mm.add(MarkerPose3D(
                marker_id=str(i + 1),
                position=(float(x_off), 0.0, 50.0),
                orientation=(0.0, 0.0, 0.0),
            ))
        dets = [
            _april("1", (200, 240), half_size=20),
            _april("2", (350, 240), half_size=20),
            _april("3", (500, 240), half_size=20),
        ]
        pose = mm.estimate_robot_pose(dets, tag_size_cm=5.0)
        assert pose.visible_markers >= 1
        assert pose.reprojection_error is not None
