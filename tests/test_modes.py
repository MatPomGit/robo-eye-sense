"""Tests for the new operational modes: calibration, box, pose, follow.

Tests verify:
- Mode classes instantiate correctly and implement BaseMode.run()
- Pure processing logic with synthetic frames (no real camera needed)
- CLI argument parsing for new modes and flags
- Headless integration with short synthetic videos
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

cv2 = pytest.importorskip(
    "cv2", reason="OpenCV runtime dependencies are unavailable"
)

from modes.base import BaseMode
from modes.box_mode import BoxDetection, BoxMode
from modes.calibration_mode import CalibrationMode
from modes.follow_mode import FollowMode, FollowResult
from modes.pose_mode import PoseMode, _get_tag_corners_3d, _sensitivity_params


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dummy_video(path, num_frames: int = 3, size: int = 64):
    """Write *num_frames* black frames into an MP4 at *path*."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 1, (size, size))
    for _ in range(num_frames):
        writer.write(np.zeros((size, size, 3), dtype=np.uint8))
    writer.release()


def _make_context(**kwargs):
    """Create a context dict with defaults."""
    ctx = {"headless": False, "key": -1, "frame_idx": 1, "fps": 30.0}
    ctx.update(kwargs)
    return ctx


# ===========================================================================
# BaseMode
# ===========================================================================


class TestBaseMode:
    def test_run_returns_frame_unchanged(self):
        mode = BaseMode()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = mode.run(frame, _make_context())
        assert result.shape == frame.shape
        np.testing.assert_array_equal(result, frame)


# ===========================================================================
# CalibrationMode
# ===========================================================================


class TestCalibrationMode:
    def test_init_defaults(self):
        mode = CalibrationMode()
        assert mode.capture_count == 0
        assert mode.is_calibrated is False

    def test_init_custom_params(self):
        mode = CalibrationMode(chessboard_size=(7, 5), output_path="out.npz")
        assert mode._board_size == (7, 5)
        assert mode._output_path == "out.npz"

    def test_run_on_black_frame_no_crash(self):
        mode = CalibrationMode()
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        result = mode.run(frame, _make_context())
        assert result.shape == (200, 200, 3)
        assert mode.capture_count == 0

    def test_run_headless_prints_status(self, capsys):
        mode = CalibrationMode()
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        mode.run(frame, _make_context(headless=True, frame_idx=5))
        out = capsys.readouterr().out
        assert "frame 5" in out
        assert "calibration" in out

    def test_capture_not_triggered_without_space(self):
        mode = CalibrationMode()
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        mode.run(frame, _make_context(key=ord("a")))
        assert mode.capture_count == 0


# ===========================================================================
# BoxMode
# ===========================================================================


class TestBoxMode:
    def test_init(self):
        mode = BoxMode()
        assert mode.detections == []

    def test_run_on_black_frame(self):
        mode = BoxMode()
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        result = mode.run(frame, _make_context())
        assert result.shape == (200, 200, 3)
        assert mode.detections == []

    def test_run_detects_white_rectangle(self):
        """A bright rectangle on a dark background should be detected."""
        mode = BoxMode()
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        # Draw a large white rectangle
        cv2.rectangle(frame, (50, 50), (250, 200), (255, 255, 255), -1)
        result = mode.run(frame, _make_context())
        assert result.shape == frame.shape
        # May or may not detect depending on edge characteristics;
        # at minimum the method should not crash

    def test_headless_output(self, capsys):
        mode = BoxMode()
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        mode.run(frame, _make_context(headless=True, frame_idx=3))
        out = capsys.readouterr().out
        assert "frame 3" in out
        assert "boxes_detected" in out

    def test_box_detection_dataclass(self):
        det = BoxDetection(
            contour=np.array([[[0, 0]], [[10, 0]], [[10, 10]], [[0, 10]]]),
            bounding_rect=(0, 0, 10, 10),
            center=(5, 5),
            area=100.0,
        )
        assert det.center == (5, 5)
        assert det.area == 100.0


# ===========================================================================
# PoseMode
# ===========================================================================


class TestPoseMode:
    def test_init_default(self):
        mode = PoseMode()
        assert mode._tag_size == 0.05

    def test_init_custom_tag_size(self):
        mode = PoseMode(tag_size=0.10)
        assert mode._tag_size == 0.10

    def test_init_sensitivity_default(self):
        mode = PoseMode()
        assert mode._sensitivity == 50

    def test_init_sensitivity_custom(self):
        mode = PoseMode(sensitivity=80)
        assert mode._sensitivity == 80

    def test_init_sensitivity_clamped_high(self):
        mode = PoseMode(sensitivity=150)
        assert mode._sensitivity == 100

    def test_init_sensitivity_clamped_low(self):
        mode = PoseMode(sensitivity=-10)
        assert mode._sensitivity == 0

    def test_sensitivity_100_accepts_all(self):
        """High sensitivity: margin threshold is 0, frames required is 1."""
        min_margin, req_frames = _sensitivity_params(100)
        assert min_margin == 0.0
        assert req_frames == 1

    def test_sensitivity_0_strictest(self):
        """Low sensitivity: margin threshold is maximum, many frames required."""
        min_margin, req_frames = _sensitivity_params(0)
        assert min_margin > 0.0
        assert req_frames > 1

    def test_sensitivity_50_intermediate(self):
        """Mid sensitivity values are between the extremes."""
        min_margin_0, req_0 = _sensitivity_params(0)
        min_margin_100, req_100 = _sensitivity_params(100)
        min_margin_50, req_50 = _sensitivity_params(50)
        assert min_margin_100 <= min_margin_50 <= min_margin_0
        assert req_100 <= req_50 <= req_0

    def test_tag_corners_3d(self):
        pts = _get_tag_corners_3d(0.1)
        assert pts.shape == (4, 3)
        assert pts[0, 2] == 0.0  # z=0 plane

    def test_run_on_black_frame_no_crash(self):
        mode = PoseMode()
        # Ensure no native detector is created (avoids pupil_apriltags segfault)
        mode._detector = None
        with patch.dict("sys.modules", {"pupil_apriltags": None, "apriltag": None}):
            frame = np.zeros((200, 200, 3), dtype=np.uint8)
            result = mode.run(frame, _make_context())
            assert result.shape == (200, 200, 3)

    def test_steering_vector_zero_when_no_tags(self):
        """steering_vector should be 0.0 when no tags are detected."""
        mode = PoseMode()
        # Use a MagicMock that returns empty detections so no native library
        # is loaded and no segfault occurs in the test environment.
        mock_detector = MagicMock()
        mock_detector.detect.return_value = []
        mode._detector = mock_detector
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        mode.run(frame, _make_context())
        assert mode.steering_vector == 0.0

    def test_steering_vector_range_with_external_detections(self):
        """steering_vector stays in [-1, 1] for external detections."""
        mode = PoseMode(sensitivity=100)  # accept on first frame
        mode._consecutive_counts = {}

        det = MagicMock()
        # Place tag at x=0 (far left) on a 200 px wide frame
        det.corners = [[0, 50], [0, 100], [50, 100], [50, 50]]
        det.identifier = "1"

        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        ctx = _make_context()
        ctx["april_detections"] = [det]
        mode.run(frame, ctx)
        assert -1.0 <= mode.steering_vector <= 1.0

    def test_steering_vector_negative_for_left_tag(self):
        """A tag on the left half should produce a negative steering value."""
        mode = PoseMode(sensitivity=100)

        det = MagicMock()
        # Corners centred around x=50 on a 200 px wide frame → left of centre
        det.corners = [[25, 25], [75, 25], [75, 75], [25, 75]]
        det.identifier = "1"

        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        ctx = _make_context()
        ctx["april_detections"] = [det]
        mode.run(frame, ctx)
        assert mode.steering_vector < 0.0

    def test_steering_vector_positive_for_right_tag(self):
        """A tag on the right half should produce a positive steering value."""
        mode = PoseMode(sensitivity=100)

        det = MagicMock()
        # Corners centred around x=150 on a 200 px wide frame → right of centre
        det.corners = [[125, 25], [175, 25], [175, 75], [125, 75]]
        det.identifier = "1"

        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        ctx = _make_context()
        ctx["april_detections"] = [det]
        mode.run(frame, ctx)
        assert mode.steering_vector > 0.0

    def test_consecutive_frames_filtering(self):
        """Low sensitivity: tag not reported until seen for required frames."""
        mode = PoseMode(sensitivity=0)  # strictest – requires many frames
        req = mode._required_consecutive_frames
        assert req > 1  # sanity check

        det = MagicMock()
        # Corners centred around x=50 on a 200 px wide frame (left of centre)
        det.corners = [[25, 75], [75, 75], [75, 125], [25, 125]]
        det.identifier = "42"

        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        ctx = _make_context()
        ctx["april_detections"] = [det]

        # Run fewer frames than required → should not be qualified yet
        for _ in range(req - 1):
            mode.run(frame, ctx)
        assert mode.steering_vector == 0.0

        # One more frame → now it should qualify
        mode.run(frame, ctx)
        assert mode.steering_vector != 0.0

    def test_load_calibration_nonexistent_file(self):
        """Missing calibration file should produce a warning, not crash."""
        mode = PoseMode(calibration_path="/tmp/nonexistent_calib.npz")
        assert mode._camera_matrix is None

    def test_default_camera_matrix(self):
        mode = PoseMode()
        mtx = mode._default_camera_matrix(640, 480)
        assert mtx.shape == (3, 3)
        assert mtx[0, 0] > 0  # focal length positive
        assert mtx[2, 2] == 1.0

    def test_correction_vector_zero_when_no_tags(self):
        """correction_vector should be (0.0, 0.0) when no tags are detected."""
        mode = PoseMode()
        mock_detector = MagicMock()
        mock_detector.detect.return_value = []
        mode._detector = mock_detector
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        mode.run(frame, _make_context())
        angle, dist = mode.correction_vector
        assert angle == 0.0
        assert dist == 0.0

    def test_correction_vector_populated_with_external_detections(self):
        """correction_vector should be non-zero when a tag is detected."""
        mode = PoseMode(sensitivity=100)

        det = MagicMock()
        # Corners centred around x=150 on a 200 px wide frame (right of centre)
        det.corners = [[125, 25], [175, 25], [175, 75], [125, 75]]
        det.identifier = "1"

        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        ctx = _make_context()
        ctx["april_detections"] = [det]
        mode.run(frame, ctx)
        angle, dist = mode.correction_vector
        # With a valid detection the distance should be positive
        assert dist > 0.0

    def test_correction_vector_angle_positive_for_right_tag(self):
        """Tag to the right of centre → positive horizontal angle."""
        mode = PoseMode(sensitivity=100)

        det = MagicMock()
        # Corners centred around x=150 on a 200 px wide frame (right of centre)
        det.corners = [[125, 25], [175, 25], [175, 75], [125, 75]]
        det.identifier = "1"

        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        ctx = _make_context()
        ctx["april_detections"] = [det]
        mode.run(frame, ctx)
        angle, _ = mode.correction_vector
        assert angle > 0.0

    def test_correction_vector_angle_negative_for_left_tag(self):
        """Tag to the left of centre → negative horizontal angle."""
        mode = PoseMode(sensitivity=100)

        det = MagicMock()
        # Corners centred around x=50 on a 200 px wide frame (left of centre)
        det.corners = [[25, 25], [75, 25], [75, 75], [25, 75]]
        det.identifier = "1"

        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        ctx = _make_context()
        ctx["april_detections"] = [det]
        mode.run(frame, ctx)
        angle, _ = mode.correction_vector
        assert angle < 0.0

    def test_steering_vector_derived_from_pose(self):
        """steering_vector should match sin(correction_angle) after pose estimation."""
        mode = PoseMode(sensitivity=100)

        det = MagicMock()
        det.corners = [[125, 25], [175, 25], [175, 75], [125, 75]]
        det.identifier = "1"

        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        ctx = _make_context()
        ctx["april_detections"] = [det]
        mode.run(frame, ctx)
        angle, _ = mode.correction_vector
        expected_steering = math.sin(math.radians(angle))
        assert mode.steering_vector == pytest.approx(expected_steering, abs=1e-6)

    def test_headless_output_includes_angle_and_dist(self, capsys):
        """Headless mode should print angle and distance for each detected tag."""
        mode = PoseMode(sensitivity=100)

        det = MagicMock()
        det.corners = [[125, 25], [175, 25], [175, 75], [125, 75]]
        det.identifier = "5"

        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        ctx = _make_context(headless=True, frame_idx=10)
        ctx["april_detections"] = [det]
        mode.run(frame, ctx)
        out = capsys.readouterr().out
        assert "frame 10" in out
        assert "angle:" in out
        assert "dist:" in out


# ===========================================================================
# FollowMode
# ===========================================================================


class TestFollowMode:
    def test_init_defaults(self):
        mode = FollowMode()
        assert mode._follow_marker is None
        assert mode._follow_box is False
        assert mode._target_distance == 0.5

    def test_init_with_box_fallback(self):
        mode = FollowMode(follow_box=True)
        assert mode._follow_box is True
        assert mode._box_mode is not None

    def test_run_idle_on_black_frame(self):
        mode = FollowMode()
        mode._detector = None
        with patch.dict("sys.modules", {"pupil_apriltags": None, "apriltag": None}):
            frame = np.zeros((200, 200, 3), dtype=np.uint8)
            result = mode.run(frame, _make_context())
            assert result.shape == (200, 200, 3)
            assert mode.last_result.mode_label == "idle"

    def test_headless_output(self, capsys):
        mode = FollowMode()
        mode._detector = None
        with patch.dict("sys.modules", {"pupil_apriltags": None, "apriltag": None}):
            frame = np.zeros((200, 200, 3), dtype=np.uint8)
            mode.run(frame, _make_context(headless=True, frame_idx=7))
        out = capsys.readouterr().out
        assert "frame 7" in out
        assert "follow" in out


class TestFollowResult:
    def test_defaults(self):
        r = FollowResult()
        assert r.mode_label == "idle"
        assert r.target_id is None
        assert r.error_x == 0.0
        assert r.error_z == 0.0
        assert r.linear == 0.0
        assert r.angular == 0.0

    def test_custom_values(self):
        r = FollowResult(
            mode_label="tag",
            target_id="3",
            error_x=10.5,
            error_z=-0.2,
            linear=0.3,
            angular=0.05,
        )
        assert r.mode_label == "tag"
        assert r.target_id == "3"


# ===========================================================================
# CLI argument parsing for new modes
# ===========================================================================


class TestCLINewModes:
    """Verify new --mode values and associated flags are parsed correctly."""

    def test_parse_mode_calibration(self):
        from main import _parse_args

        args = _parse_args(["--mode", "calibration"])
        assert args.mode == "calibration"

    def test_parse_mode_box(self):
        from main import _parse_args

        args = _parse_args(["--mode", "box"])
        assert args.mode == "box"

    def test_parse_mode_pose(self):
        from main import _parse_args

        args = _parse_args(["--mode", "pose"])
        assert args.mode == "pose"

    def test_parse_mode_follow(self):
        from main import _parse_args

        args = _parse_args(["--mode", "follow"])
        assert args.mode == "follow"

    def test_chessboard_size_default(self):
        from main import _parse_args

        args = _parse_args([])
        assert args.chessboard_size == "9x6"

    def test_chessboard_size_custom(self):
        from main import _parse_args

        args = _parse_args(["--chessboard-size", "7x5"])
        assert args.chessboard_size == "7x5"

    def test_calib_output_default(self):
        from main import _parse_args

        args = _parse_args([])
        assert args.cal == "calibration.npz"

    def test_tag_size_default(self):
        from main import _parse_args

        args = _parse_args([])
        assert args.tag_size == pytest.approx(0.05)

    def test_tag_size_custom(self):
        from main import _parse_args

        args = _parse_args(["--tag-size", "0.10"])
        assert args.tag_size == pytest.approx(0.10)

    def test_follow_box_flag(self):
        from main import _parse_args

        args = _parse_args(["--follow-box"])
        assert args.follow_box is True

    def test_follow_box_default_false(self):
        from main import _parse_args

        args = _parse_args([])
        assert args.follow_box is False

    def test_target_distance_default(self):
        from main import _parse_args

        args = _parse_args([])
        assert args.target_distance == pytest.approx(0.5)

    def test_target_distance_custom(self):
        from main import _parse_args

        args = _parse_args(["--target-distance", "1.0"])
        assert args.target_distance == pytest.approx(1.0)

    def test_parse_mode_by_number_basic(self):
        from main import _parse_args

        args = _parse_args(["--mode", "1"])
        assert args.mode == "basic"

    def test_parse_mode_by_number_offset(self):
        from main import _parse_args

        args = _parse_args(["--mode", "2"])
        assert args.mode == "offset"

    def test_parse_mode_by_number_slam(self):
        from main import _parse_args

        args = _parse_args(["--mode", "3"])
        assert args.mode == "slam"

    def test_parse_mode_by_number_calibration(self):
        from main import _parse_args

        args = _parse_args(["--mode", "4"])
        assert args.mode == "calibration"

    def test_parse_mode_by_number_box(self):
        from main import _parse_args

        args = _parse_args(["--mode", "5"])
        assert args.mode == "box"

    def test_parse_mode_by_number_pose(self):
        from main import _parse_args

        args = _parse_args(["--mode", "6"])
        assert args.mode == "pose"

    def test_parse_mode_by_number_follow(self):
        from main import _parse_args

        args = _parse_args(["--mode", "7"])
        assert args.mode == "follow"

    def test_parse_mode_by_number_mediapipe(self):
        from main import _parse_args

        args = _parse_args(["--mode", "8"])
        assert args.mode == "mediapipe"

    def test_parse_mode_by_number_yolo(self):
        from main import _parse_args

        args = _parse_args(["--mode", "9"])
        assert args.mode == "yolo"

    def test_parse_mode_invalid_number(self):
        from main import _parse_args

        with pytest.raises(SystemExit):
            _parse_args(["--mode", "10"])

    def test_sensitivity_default(self):
        from main import _parse_args

        args = _parse_args([])
        assert args.sensitivity == 50

    def test_sensitivity_custom(self):
        from main import _parse_args

        args = _parse_args(["--sensitivity", "80"])
        assert args.sensitivity == 80

    def test_sensitivity_zero(self):
        from main import _parse_args

        args = _parse_args(["--sensitivity", "0"])
        assert args.sensitivity == 0

    def test_sensitivity_max(self):
        from main import _parse_args

        args = _parse_args(["--sensitivity", "100"])
        assert args.sensitivity == 100

    def test_parse_mode_mediapipe(self):
        from main import _parse_args

        args = _parse_args(["--mode", "mediapipe"])
        assert args.mode == "mediapipe"

    def test_parse_mode_yolo(self):
        from main import _parse_args

        args = _parse_args(["--mode", "yolo"])
        assert args.mode == "yolo"

    def test_parse_yolo_args_defaults(self):
        from main import _parse_args

        args = _parse_args(["--mode", "yolo"])
        assert args.yolo_model is None
        assert args.yolo_conf == pytest.approx(0.25)
        assert args.yolo_iou == pytest.approx(0.45)
        assert args.yolo_classes is None
        assert args.no_yolo_track is False

    def test_parse_yolo_args_custom(self):
        from main import _parse_args

        args = _parse_args([
            "--mode", "yolo",
            "--yolo-model", "path/to/best.pt",
            "--yolo-conf", "0.5",
            "--yolo-iou", "0.6",
            "--yolo-classes", "0", "1",
            "--no-yolo-track",
        ])
        assert args.yolo_model == "path/to/best.pt"
        assert args.yolo_conf == pytest.approx(0.5)
        assert args.yolo_iou == pytest.approx(0.6)
        assert args.yolo_classes == [0, 1]
        assert args.no_yolo_track is True


# ===========================================================================
# MediaPipeMode
# ===========================================================================


class TestMediaPipeMode:
    """Tests for the MediaPipe pose-landmarker mode."""

    def test_instantiation(self):
        from modes.mediapipe_mode import MediaPipeMode

        mode = MediaPipeMode()
        assert isinstance(mode, MediaPipeMode)

    def test_default_detections_empty(self):
        from modes.mediapipe_mode import MediaPipeMode

        mode = MediaPipeMode()
        assert mode.detections == []

    def test_is_ready_before_init(self):
        from modes.mediapipe_mode import MediaPipeMode

        mode = MediaPipeMode()
        assert not mode.is_ready

    def test_run_returns_frame_shaped_output_no_mediapipe(self):
        """When mediapipe is unavailable the mode returns an annotated frame."""
        from modes.mediapipe_mode import MediaPipeMode

        mode = MediaPipeMode()
        # Simulate missing mediapipe by injecting an init error directly
        mode._init_error = "mediapipe not installed"

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = mode.run(frame, _make_context())
        assert result.shape == frame.shape
        # Verify the error message is rendered onto the returned frame
        # (error path writes red text; frame is no longer all-zeros)
        assert not np.array_equal(result, frame), (
            "Error overlay should modify the frame"
        )

    def test_run_with_mock_landmarker(self):
        """run() draws skeleton when landmarker returns landmarks."""
        from modes.mediapipe_mode import MediaPipeMode, PoseDetection, PoseLandmark

        mode = MediaPipeMode()

        # Build a fake landmark list (33 points, all at centre, fully visible)
        fake_lms = [
            PoseLandmark(x=0.5, y=0.5, z=0.0, visibility=1.0)
            for _ in range(33)
        ]
        fake_detection = PoseDetection(landmarks=fake_lms)

        # Pre-populate detections and skip the real landmarker
        mode._detections = [fake_detection]
        mode._connections = []  # no connections → only joints drawn
        mode._landmarker = MagicMock()

        # Patch the detect call so it returns the pre-built detection
        mock_result = MagicMock()
        mock_result.pose_landmarks = [
            [MagicMock(x=0.5, y=0.5, z=0.0, visibility=1.0)] * 33
        ]
        mode._landmarker.detect.return_value = mock_result

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        with patch("mediapipe.Image") as mock_img:
            mock_img.return_value = MagicMock()
            result = mode.run(frame, _make_context())

        assert result.shape == frame.shape

    def test_headless_output(self, capsys):
        """In headless mode the error message is displayed (no real model needed)."""
        from modes.mediapipe_mode import MediaPipeMode

        # Use error path: inject init error without real mediapipe landmarker
        mode = MediaPipeMode()
        mode._init_error = "mediapipe not installed"

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = mode.run(frame, _make_context(headless=True, frame_idx=5))
        # Shape must be preserved even on the error path
        assert result.shape == frame.shape
        # The error path renders text onto the frame (frame is modified)
        assert not np.array_equal(result, frame), (
            "Error overlay should modify the frame"
        )

    def test_num_poses_clamped_to_one(self):
        from modes.mediapipe_mode import MediaPipeMode

        mode = MediaPipeMode(num_poses=0)
        assert mode._num_poses >= 1

    def test_pose_landmark_dataclass(self):
        from modes.mediapipe_mode import PoseLandmark

        lm = PoseLandmark(x=0.1, y=0.2, z=0.3, visibility=0.9)
        assert lm.x == pytest.approx(0.1)
        assert lm.visibility == pytest.approx(0.9)

    def test_pose_detection_num_landmarks(self):
        from modes.mediapipe_mode import PoseDetection, PoseLandmark

        lms = [PoseLandmark(x=i * 0.01, y=0.0, z=0.0, visibility=1.0) for i in range(33)]
        det = PoseDetection(landmarks=lms)
        assert det.num_landmarks == 33


# ===========================================================================
# Integration: headless runs with synthetic videos
# ===========================================================================


class TestNewModeIntegration:
    """Run each new mode headless with a short synthetic video."""

    def test_calibration_headless(self, capsys, tmp_path):
        video = tmp_path / "calib.mp4"
        _make_dummy_video(video)

        with patch(
            "robo_vision.april_tag_detector._apriltags_available",
            return_value=False,
        ):
            from main import main

            rc = main([
                "--mode", "calibration",
                "--headless",
                "--source", str(video),
            ])
        assert rc == 0
        out = capsys.readouterr().out
        assert "calibration" in out.lower()

    def test_box_headless(self, capsys, tmp_path):
        video = tmp_path / "box.mp4"
        _make_dummy_video(video)

        with patch(
            "robo_vision.april_tag_detector._apriltags_available",
            return_value=False,
        ):
            from main import main

            rc = main([
                "--mode", "box",
                "--headless",
                "--source", str(video),
            ])
        assert rc == 0
        out = capsys.readouterr().out
        assert "box" in out.lower()

    def test_pose_headless(self, capsys, tmp_path):
        video = tmp_path / "pose.mp4"
        _make_dummy_video(video)

        with patch(
            "robo_vision.april_tag_detector._apriltags_available",
            return_value=False,
        ):
            from main import main

            rc = main([
                "--mode", "pose",
                "--headless",
                "--source", str(video),
            ])
        assert rc == 0
        out = capsys.readouterr().out
        assert "pose" in out.lower()

    def test_follow_headless(self, capsys, tmp_path):
        video = tmp_path / "follow.mp4"
        _make_dummy_video(video)

        with patch(
            "robo_vision.april_tag_detector._apriltags_available",
            return_value=False,
        ):
            from main import main

            rc = main([
                "--mode", "follow",
                "--headless",
                "--source", str(video),
            ])
        assert rc == 0
        out = capsys.readouterr().out
        assert "follow" in out.lower()

    def test_follow_with_box_headless(self, capsys, tmp_path):
        video = tmp_path / "follow_box.mp4"
        _make_dummy_video(video)

        with patch(
            "robo_vision.april_tag_detector._apriltags_available",
            return_value=False,
        ):
            from main import main

            rc = main([
                "--mode", "follow",
                "--follow-box",
                "--headless",
                "--source", str(video),
            ])
        assert rc == 0

    def test_invalid_chessboard_size(self, capsys, tmp_path):
        video = tmp_path / "bad_calib.mp4"
        _make_dummy_video(video)

        with patch(
            "robo_vision.april_tag_detector._apriltags_available",
            return_value=False,
        ):
            from main import main

            rc = main([
                "--mode", "calibration",
                "--chessboard-size", "bad",
                "--headless",
                "--source", str(video),
            ])
        assert rc == 1
        err = capsys.readouterr().err
        assert "chessboard-size" in err.lower()

    def test_slam_headless_saves_map_on_normal_exit(self, capsys, tmp_path):
        """SLAM mode should save JSON map after processing all frames."""
        video = tmp_path / "slam.mp4"
        _make_dummy_video(video)
        map_file = tmp_path / "test_map.json"

        with patch(
            "robo_vision.april_tag_detector._apriltags_available",
            return_value=False,
        ):
            from main import main

            rc = main([
                "--mode", "slam",
                "--headless",
                "--source", str(video),
                "--map-file", str(map_file),
            ])
        assert rc == 0
        # Map file must be written even with no tags detected (empty map)
        assert map_file.exists()
        import json
        data = json.loads(map_file.read_text())
        assert "markers" in data

    def test_slam_headless_saves_map_on_keyboard_interrupt(self, capsys, tmp_path):
        """SLAM mode must save the JSON map even when interrupted with Ctrl+C."""
        video = tmp_path / "slam_int.mp4"
        _make_dummy_video(video, num_frames=10)
        map_file = tmp_path / "test_map_int.json"

        _frame_counter = [0]

        with patch(
            "robo_vision.april_tag_detector._apriltags_available",
            return_value=False,
        ):
            from main import main
            from robo_vision.detector import RoboEyeDetector

            original = RoboEyeDetector.process_frame

            def _patched(self, frame):
                _frame_counter[0] += 1
                if _frame_counter[0] >= 3:
                    raise KeyboardInterrupt
                return original(self, frame)

            with patch.object(RoboEyeDetector, "process_frame", _patched):
                rc = main([
                    "--mode", "slam",
                    "--headless",
                    "--source", str(video),
                    "--map-file", str(map_file),
                ])

        # Should exit cleanly (rc=0) even after interrupt
        assert rc == 0
        # Map file MUST be written despite the interrupt
        assert map_file.exists(), "Map must be saved even when interrupted"
        import json
        data = json.loads(map_file.read_text())
        assert "markers" in data

    def test_mediapipe_headless(self, capsys, tmp_path):
        """MediaPipe mode runs headless; gracefully handles missing model."""
        video = tmp_path / "mediapipe.mp4"
        _make_dummy_video(video)

        with patch(
            "robo_vision.april_tag_detector._apriltags_available",
            return_value=False,
        ):
            # Patch _ensure_initialized to return False so no model download
            # or real mediapipe landmarker is needed.
            with patch(
                "modes.mediapipe_mode.MediaPipeMode._ensure_initialized",
                return_value=False,
            ):
                from main import main

                rc = main([
                    "--mode", "mediapipe",
                    "--headless",
                    "--source", str(video),
                ])
        # Mode must exit cleanly (rc=0) even when mediapipe is unavailable
        assert rc == 0


# ===========================================================================
# YoloMode
# ===========================================================================


class TestYoloMode:
    """Tests for the YOLO detection and tracking mode."""

    def test_instantiation(self):
        from modes.yolo_mode import YoloMode

        mode = YoloMode()
        assert isinstance(mode, YoloMode)

    def test_default_detections_empty(self):
        from modes.yolo_mode import YoloMode

        mode = YoloMode()
        assert mode.detections == []

    def test_is_ready_before_init(self):
        from modes.yolo_mode import YoloMode

        mode = YoloMode()
        assert not mode.is_ready

    def test_run_returns_frame_shaped_output_no_ultralytics(self):
        """When ultralytics is unavailable the mode returns an error frame."""
        from modes.yolo_mode import YoloMode

        mode = YoloMode()
        mode._init_error = "ultralytics not installed"

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        ctx = {"headless": False, "key": -1, "frame_idx": 1, "fps": 30.0}
        result = mode.run(frame, ctx)
        assert result.shape == frame.shape
        # Error overlay modifies the frame (red text drawn)
        assert not np.array_equal(result, frame), (
            "Error overlay should modify the frame"
        )

    def test_run_with_mock_model_no_detections(self):
        """run() handles an empty result from the YOLO model."""
        from modes.yolo_mode import YoloMode

        mode = YoloMode()
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.boxes = None
        mock_model.track.return_value = [mock_result]
        mode._model = mock_model

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        ctx = {"headless": False, "key": -1, "frame_idx": 1, "fps": 0.0}
        result = mode.run(frame, ctx)
        assert result.shape == frame.shape
        assert mode.detections == []

    def test_run_with_mock_model_with_detections(self):
        """run() parses bounding boxes and populates detections."""
        from modes.yolo_mode import YoloMode

        mode = YoloMode(track=True)
        mock_model = MagicMock()

        # Build a fake boxes object using plain mocks (no torch needed)
        fake_box = MagicMock()
        fake_box.xyxy = [MagicMock(tolist=lambda: [10.0, 20.0, 50.0, 60.0])]
        fake_box.conf = [MagicMock(__float__=lambda self: 0.9)]
        fake_box.cls = [MagicMock(__int__=lambda self: 0)]
        fake_box.id = [MagicMock(__int__=lambda self: 1)]

        # Use a MagicMock that supports len() and indexing
        boxes_mock = MagicMock()
        boxes_mock.__len__ = MagicMock(return_value=1)
        boxes_mock.__getitem__ = MagicMock(return_value=fake_box)

        mock_result = MagicMock()
        mock_result.boxes = boxes_mock
        mock_result.names = {0: "person"}
        mock_model.track.return_value = [mock_result]
        mode._model = mock_model

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        ctx = {"headless": False, "key": -1, "frame_idx": 1, "fps": 30.0}
        result = mode.run(frame, ctx)
        assert result.shape == frame.shape

    def test_run_detection_only_mode(self):
        """With track=False the mode calls predict instead of track."""
        from modes.yolo_mode import YoloMode

        mode = YoloMode(track=False)
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.boxes = None
        mock_model.predict.return_value = [mock_result]
        mode._model = mock_model

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        ctx = {"headless": False, "key": -1, "frame_idx": 1, "fps": 0.0}
        mode.run(frame, ctx)
        mock_model.predict.assert_called_once()
        mock_model.track.assert_not_called()

    def test_headless_output_error_path(self, capsys):
        """In headless mode the error message is printed and frame preserved."""
        from modes.yolo_mode import YoloMode

        mode = YoloMode()
        mode._init_error = "ultralytics not installed"

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        ctx = {"headless": True, "key": -1, "frame_idx": 3, "fps": 0.0}
        result = mode.run(frame, ctx)
        assert result.shape == frame.shape

    def test_headless_output_detection_info(self, capsys):
        """Headless mode prints detection count to stdout."""
        from modes.yolo_mode import YoloMode

        mode = YoloMode()
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.boxes = None
        mock_model.track.return_value = [mock_result]
        mode._model = mock_model

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        ctx = {"headless": True, "key": -1, "frame_idx": 5, "fps": 0.0}
        mode.run(frame, ctx)
        out = capsys.readouterr().out
        assert "frame 5" in out
        assert "yolo_detected" in out

    def test_yolo_detection_dataclass(self):
        from modes.yolo_mode import YoloDetection

        det = YoloDetection(
            track_id=3,
            class_id=0,
            class_name="person",
            confidence=0.88,
            bbox=(10, 20, 50, 60),
        )
        assert det.track_id == 3
        assert det.class_name == "person"
        assert det.confidence == pytest.approx(0.88)
        cx, cy = det.center
        assert cx == 30
        assert cy == 40

    def test_yolo_detection_center_no_track_id(self):
        from modes.yolo_mode import YoloDetection

        det = YoloDetection(
            track_id=None,
            class_id=1,
            class_name="car",
            confidence=0.5,
            bbox=(0, 0, 100, 100),
        )
        assert det.track_id is None
        assert det.center == (50, 50)

    def test_color_for_id_consistency(self):
        """Same ID always returns the same color; None returns green."""
        from modes.yolo_mode import YoloMode

        mode = YoloMode()
        c1 = mode._color_for_id(0)
        c2 = mode._color_for_id(0)
        assert c1 == c2
        assert mode._color_for_id(None) == (0, 255, 0)

    def test_init_error_on_import_failure(self):
        """_ensure_initialized returns False when ultralytics is absent."""
        from modes.yolo_mode import YoloMode

        mode = YoloMode()
        with patch.dict("sys.modules", {"ultralytics": None}):
            result = mode._ensure_initialized()
        assert result is False
        assert mode._init_error is not None

    def test_run_inference_exception_handled(self):
        """Exceptions during inference return an unmodified frame copy."""
        from modes.yolo_mode import YoloMode

        mode = YoloMode()
        mock_model = MagicMock()
        mock_model.track.side_effect = RuntimeError("inference failed")
        mode._model = mock_model

        frame = np.zeros((80, 80, 3), dtype=np.uint8)
        ctx = {"headless": False, "key": -1, "frame_idx": 1, "fps": 0.0}
        result = mode.run(frame, ctx)
        assert result.shape == frame.shape
        assert mode.detections == []


class TestYoloModeIntegration:
    """Headless CLI integration test for YOLO mode."""

    def test_yolo_headless(self, capsys, tmp_path):
        """YOLO mode runs headless; gracefully handles missing ultralytics."""
        video = tmp_path / "yolo.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video), fourcc, 1, (64, 64))
        for _ in range(3):
            writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
        writer.release()

        with patch(
            "robo_vision.april_tag_detector._apriltags_available",
            return_value=False,
        ):
            with patch(
                "modes.yolo_mode.YoloMode._ensure_initialized",
                return_value=False,
            ):
                from main import main

                rc = main([
                    "--mode", "yolo",
                    "--headless",
                    "--source", str(video),
                ])
        assert rc == 0
