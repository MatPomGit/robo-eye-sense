"""Tests for RoboEyeDetector orchestration.

AprilTag and QR-code sub-detectors are mocked so these tests run without
the optional pupil-apriltags / pyzbar libraries and without real images.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="OpenCV runtime dependencies are unavailable", exc_type=ImportError)

from robo_eye_sense.detector import RoboEyeDetector
from robo_eye_sense.results import Detection, DetectionMode, DetectionType


@pytest.fixture
def blank_bgr():
    return np.zeros((480, 640, 3), dtype=np.uint8)


class TestRoboEyeDetectorInit:
    def test_creates_with_defaults(self):
        with patch(
            "robo_eye_sense.april_tag_detector._apriltags_available",
            return_value=False,
        ):
            detector = RoboEyeDetector()
        assert detector._april_detector is None
        assert detector._qr_detector is not None
        assert detector._laser_detector is not None

    def test_laser_disabled(self):
        with patch(
            "robo_eye_sense.april_tag_detector._apriltags_available",
            return_value=False,
        ):
            detector = RoboEyeDetector(enable_laser=False)
        assert detector._laser_detector is None

    def test_qr_disabled(self):
        with patch(
            "robo_eye_sense.april_tag_detector._apriltags_available",
            return_value=False,
        ):
            detector = RoboEyeDetector(enable_qr=False)
        assert detector._qr_detector is None


class TestRoboEyeDetectorProcessFrame:
    def _detector_with_mocks(self, april_detections=None, qr_detections=None):
        """Return a RoboEyeDetector whose sub-detectors are replaced by mocks."""
        with patch(
            "robo_eye_sense.april_tag_detector._apriltags_available",
            return_value=False,
        ):
            d = RoboEyeDetector(enable_apriltag=False)

        if qr_detections is not None:
            mock_qr = MagicMock()
            mock_qr.detect.return_value = qr_detections
            d._qr_detector = mock_qr

        return d

    def test_process_black_frame_returns_list(self, blank_bgr):
        with patch(
            "robo_eye_sense.april_tag_detector._apriltags_available",
            return_value=False,
        ):
            detector = RoboEyeDetector()
        result = detector.process_frame(blank_bgr)
        assert isinstance(result, list)

    def test_laser_spot_detected_in_bright_frame(self):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(frame, (100, 100), 6, (255, 255, 255), -1)

        with patch(
            "robo_eye_sense.april_tag_detector._apriltags_available",
            return_value=False,
        ):
            detector = RoboEyeDetector(enable_qr=False)
        detections = detector.process_frame(frame)
        laser = [d for d in detections if d.detection_type == DetectionType.LASER_SPOT]
        assert len(laser) == 1
        assert laser[0].track_id is not None

    def test_mock_qr_detection_gets_track_id(self, blank_bgr):
        qr_d = Detection(
            detection_type=DetectionType.QR_CODE,
            identifier="hello",
            center=(100, 100),
            corners=[(90, 90), (110, 90), (110, 110), (90, 110)],
        )
        detector = self._detector_with_mocks(qr_detections=[qr_d])
        result = detector.process_frame(blank_bgr)
        assert any(d.detection_type == DetectionType.QR_CODE for d in result)
        qr = next(d for d in result if d.detection_type == DetectionType.QR_CODE)
        assert qr.track_id is not None

    def test_detections_tracked_consistently(self):
        """The same laser spot across two frames should get the same track_id."""
        with patch(
            "robo_eye_sense.april_tag_detector._apriltags_available",
            return_value=False,
        ):
            detector = RoboEyeDetector(enable_qr=False)

        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(frame, (100, 100), 6, (255, 255, 255), -1)

        r1 = detector.process_frame(frame)
        r2 = detector.process_frame(frame)

        id1 = r1[0].track_id
        id2 = r2[0].track_id
        assert id1 == id2


class TestRoboEyeDetectorDraw:
    def test_draw_returns_frame(self, blank_bgr):
        with patch(
            "robo_eye_sense.april_tag_detector._apriltags_available",
            return_value=False,
        ):
            detector = RoboEyeDetector()
        result = detector.draw_detections(blank_bgr.copy(), [])
        assert result.shape == blank_bgr.shape

    def test_draw_with_laser_detection(self):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(frame, (100, 100), 6, (255, 255, 255), -1)
        with patch(
            "robo_eye_sense.april_tag_detector._apriltags_available",
            return_value=False,
        ):
            detector = RoboEyeDetector(enable_qr=False)
        detections = detector.process_frame(frame)
        annotated = detector.draw_detections(frame.copy(), detections)
        assert annotated is not None
        assert annotated.shape == frame.shape

    def test_draw_truncates_long_qr_payload(self, blank_bgr):
        long_id = "A" * 50
        d = Detection(
            detection_type=DetectionType.QR_CODE,
            identifier=long_id,
            center=(100, 100),
            corners=[(90, 90), (110, 90), (110, 110), (90, 110)],
            track_id=0,
        )
        with patch(
            "robo_eye_sense.april_tag_detector._apriltags_available",
            return_value=False,
        ):
            detector = RoboEyeDetector()
        # Should not raise even with a very long identifier
        annotated = detector.draw_detections(blank_bgr.copy(), [d])
        assert annotated is not None


# ---------------------------------------------------------------------------
# DetectionMode – Mode 2: FAST
# ---------------------------------------------------------------------------


class TestFastMode:
    """RoboEyeDetector in FAST mode downscales frames and scales results back."""

    @staticmethod
    def _make_detector(**kwargs):
        with patch(
            "robo_eye_sense.april_tag_detector._apriltags_available",
            return_value=False,
        ):
            return RoboEyeDetector(mode=DetectionMode.FAST, enable_qr=False, **kwargs)

    def test_mode_property(self):
        d = self._make_detector()
        assert d.mode == DetectionMode.FAST

    def test_fast_mode_returns_list(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        d = self._make_detector()
        result = d.process_frame(frame)
        assert isinstance(result, list)

    def test_laser_detected_in_fast_mode(self):
        """A bright spot in a large frame should be detected in FAST mode."""
        frame = np.zeros((400, 400, 3), dtype=np.uint8)
        # Radius 12 → after 50 % downscale → radius ~6, area ~113 px (well above min)
        cv2.circle(frame, (200, 200), 12, (255, 255, 255), -1)
        d = self._make_detector()
        detections = d.process_frame(frame)
        laser = [x for x in detections if x.detection_type == DetectionType.LASER_SPOT]
        assert len(laser) == 1

    def test_fast_mode_coordinates_are_scaled_back(self):
        """Detected coordinates must be in original resolution (not downscaled)."""
        frame = np.zeros((400, 400, 3), dtype=np.uint8)
        # Place spot in lower-right quadrant so raw coordinates differ
        # significantly between 50%-scaled and original space.
        cv2.circle(frame, (300, 300), 12, (255, 255, 255), -1)
        d = self._make_detector()
        detections = d.process_frame(frame)
        laser = [x for x in detections if x.detection_type == DetectionType.LASER_SPOT]
        assert len(laser) == 1
        cx, cy = laser[0].center
        # At 50 % scale the centre would be ~(150, 150); after scaling back it
        # should be significantly larger than the scaled value.
        assert cx > 200 and cy > 200

    def test_fast_mode_tracker_max_disappeared(self):
        """FAST mode should configure tracker with a reduced max_disappeared."""
        d = self._make_detector()
        assert d._tracker.max_disappeared < 10  # FAST uses 5

    def test_mode_switch_normal_to_fast(self):
        with patch(
            "robo_eye_sense.april_tag_detector._apriltags_available",
            return_value=False,
        ):
            d = RoboEyeDetector(enable_qr=False)
        assert d.mode == DetectionMode.NORMAL
        d.mode = DetectionMode.FAST
        assert d.mode == DetectionMode.FAST

    def test_draw_shows_mode_label(self):
        """draw_detections must not raise for FAST mode."""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        d = self._make_detector()
        result = d.draw_detections(frame.copy(), [])
        assert result.shape == frame.shape


# ---------------------------------------------------------------------------
# DetectionMode – Mode 3: ROBUST
# ---------------------------------------------------------------------------


class TestRobustMode:
    """RoboEyeDetector in ROBUST mode uses sharpening and Kalman tracking."""

    @staticmethod
    def _make_detector(**kwargs):
        with patch(
            "robo_eye_sense.april_tag_detector._apriltags_available",
            return_value=False,
        ):
            return RoboEyeDetector(mode=DetectionMode.ROBUST, enable_qr=False, **kwargs)

    def test_mode_property(self):
        d = self._make_detector()
        assert d.mode == DetectionMode.ROBUST

    def test_robust_mode_returns_list(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        d = self._make_detector()
        result = d.process_frame(frame)
        assert isinstance(result, list)

    def test_laser_detected_in_robust_mode(self):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(frame, (100, 100), 6, (255, 255, 255), -1)
        d = self._make_detector()
        detections = d.process_frame(frame)
        laser = [x for x in detections if x.detection_type == DetectionType.LASER_SPOT]
        assert len(laser) == 1
        assert laser[0].track_id is not None

    def test_robust_tracker_uses_kalman(self):
        d = self._make_detector()
        assert d._tracker.use_kalman is True

    def test_robust_tracker_max_disappeared(self):
        """ROBUST mode should have a larger max_disappeared budget."""
        d = self._make_detector()
        assert d._tracker.max_disappeared > 10  # ROBUST uses 20

    def test_robust_tracker_max_distance(self):
        """ROBUST mode should have a larger max_distance budget."""
        d = self._make_detector()
        assert d._tracker.max_distance > 50  # ROBUST uses 100

    def test_track_survives_multiple_blank_frames(self):
        """In ROBUST mode a track should survive many consecutive missed frames."""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(frame, (100, 100), 6, (255, 255, 255), -1)
        blank = np.zeros((200, 200, 3), dtype=np.uint8)

        d = self._make_detector()
        # Establish the track
        r1 = d.process_frame(frame)
        first_id = r1[0].track_id

        # Feed several blank frames (up to max_disappeared - 1)
        for _ in range(15):
            d.process_frame(blank)

        # Re-detect: should keep the same track ID
        r2 = d.process_frame(frame)
        assert len(r2) > 0
        assert r2[0].track_id == first_id

    def test_mode_switch_enables_kalman(self):
        """Switching from NORMAL to ROBUST must enable Kalman filtering."""
        with patch(
            "robo_eye_sense.april_tag_detector._apriltags_available",
            return_value=False,
        ):
            d = RoboEyeDetector(enable_qr=False)
        assert d._tracker.use_kalman is False
        d.mode = DetectionMode.ROBUST
        assert d._tracker.use_kalman is True

    def test_mode_switch_disables_kalman(self):
        """Switching from ROBUST back to NORMAL must disable Kalman filtering."""
        d = self._make_detector()
        d.mode = DetectionMode.NORMAL
        assert d._tracker.use_kalman is False

    def test_draw_shows_mode_label(self):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        d = self._make_detector()
        result = d.draw_detections(frame.copy(), [])
        assert result.shape == frame.shape
