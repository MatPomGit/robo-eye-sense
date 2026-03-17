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

from robo_eye_sense.marker_map import (
    MarkerMap,
    MarkerPose3D,
    RobotPose3D,
    SlamCalibrator,
    _angle_average,
    _euler_to_rotation_matrix,
    _mean_angles,
    _rotation_matrix_to_euler,
)
from robo_eye_sense.results import Detection, DetectionType


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
        # Running average: (10*1 + 20)/2=15, (20*1 + 40)/2=30, (30*1 + 60)/2=45
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
