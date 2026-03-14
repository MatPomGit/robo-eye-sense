"""Tests for RoboEyeDetector orchestration.

AprilTag and QR-code sub-detectors are mocked so these tests run without
the optional pupil-apriltags / pyzbar libraries and without real images.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from robo_eye_sense.detector import RoboEyeDetector
from robo_eye_sense.results import Detection, DetectionType


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
