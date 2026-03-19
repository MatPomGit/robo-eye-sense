"""Tests for AprilTagDetector – decision-margin filtering.

These tests use a mocked pupil-apriltags backend so they run without a
real camera or printed tags.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from robo_vision.results import DetectionType


def _make_raw_result(tag_id=0, center=(100, 100), decision_margin=50.0):
    """Create an object that mimics a pupil-apriltags detection result."""
    return SimpleNamespace(
        tag_id=tag_id,
        center=center,
        corners=np.array(
            [[80, 80], [120, 80], [120, 120], [80, 120]], dtype=float
        ),
        decision_margin=decision_margin,
    )


class TestAprilTagDecisionMarginFiltering:
    """AprilTagDetector should discard low-margin detections."""

    @pytest.fixture(autouse=True)
    def _ensure_apriltags_available(self):
        """Patch the availability check so the detector can be instantiated."""
        with patch(
            "robo_vision.april_tag_detector._apriltags_available",
            return_value=True,
        ), patch(
            "robo_vision.april_tag_detector.importlib.util.find_spec",
            return_value=True,
        ):
            yield

    @staticmethod
    def _make_detector(min_decision_margin=25.0):
        mock_apriltag_module = MagicMock()
        mock_detector_instance = MagicMock()
        mock_apriltag_module.Detector.return_value = mock_detector_instance

        with patch.dict(
            "sys.modules", {"pupil_apriltags": mock_apriltag_module}
        ):
            from robo_vision.april_tag_detector import AprilTagDetector

            det = AprilTagDetector(min_decision_margin=min_decision_margin)
        return det, mock_detector_instance

    def test_high_margin_detection_accepted(self):
        det, mock_inner = self._make_detector(min_decision_margin=20.0)
        mock_inner.detect.return_value = [_make_raw_result(decision_margin=50.0)]

        gray = np.zeros((200, 200), dtype=np.uint8)
        results = det.detect(gray)
        assert len(results) == 1
        assert results[0].detection_type == DetectionType.APRIL_TAG

    def test_low_margin_detection_rejected(self):
        det, mock_inner = self._make_detector(min_decision_margin=30.0)
        mock_inner.detect.return_value = [_make_raw_result(decision_margin=10.0)]

        gray = np.zeros((200, 200), dtype=np.uint8)
        results = det.detect(gray)
        assert len(results) == 0

    def test_mixed_margins_only_high_kept(self):
        det, mock_inner = self._make_detector(min_decision_margin=25.0)
        mock_inner.detect.return_value = [
            _make_raw_result(tag_id=1, decision_margin=50.0),
            _make_raw_result(tag_id=2, decision_margin=5.0),
            _make_raw_result(tag_id=3, decision_margin=30.0),
        ]

        gray = np.zeros((200, 200), dtype=np.uint8)
        results = det.detect(gray)
        assert len(results) == 2
        ids = {r.identifier for r in results}
        assert ids == {"1", "3"}

    def test_confidence_populated_from_decision_margin(self):
        det, mock_inner = self._make_detector(min_decision_margin=10.0)
        mock_inner.detect.return_value = [_make_raw_result(decision_margin=42.5)]

        gray = np.zeros((200, 200), dtype=np.uint8)
        results = det.detect(gray)
        assert len(results) == 1
        assert results[0].confidence == pytest.approx(42.5)

    def test_default_min_decision_margin_is_25(self):
        det, _ = self._make_detector()
        assert det._min_decision_margin == 25.0

    def test_exact_margin_at_threshold_accepted(self):
        """A detection with margin exactly at the threshold should be accepted
        (the filter rejects only strictly-below-threshold values)."""
        det, mock_inner = self._make_detector(min_decision_margin=25.0)
        mock_inner.detect.return_value = [_make_raw_result(decision_margin=25.0)]

        gray = np.zeros((200, 200), dtype=np.uint8)
        results = det.detect(gray)
        assert len(results) == 1

    def test_margin_just_above_threshold_accepted(self):
        det, mock_inner = self._make_detector(min_decision_margin=25.0)
        mock_inner.detect.return_value = [
            _make_raw_result(decision_margin=25.01)
        ]

        gray = np.zeros((200, 200), dtype=np.uint8)
        results = det.detect(gray)
        assert len(results) == 1


class TestDefaultsDisabledModes:
    """Verify that QR and laser modes are disabled by default."""

    def test_qr_disabled_by_default(self):
        with patch(
            "robo_vision.april_tag_detector._apriltags_available",
            return_value=False,
        ):
            from robo_vision.detector import RoboEyeDetector

            det = RoboEyeDetector()
        assert det._qr_detector is None

    def test_laser_disabled_by_default(self):
        with patch(
            "robo_vision.april_tag_detector._apriltags_available",
            return_value=False,
        ):
            from robo_vision.detector import RoboEyeDetector

            det = RoboEyeDetector()
        assert det._laser_detector is None

    def test_qr_enabled_explicitly(self):
        with patch(
            "robo_vision.april_tag_detector._apriltags_available",
            return_value=False,
        ):
            from robo_vision.detector import RoboEyeDetector

            det = RoboEyeDetector(enable_qr=True)
        assert det._qr_detector is not None

    def test_laser_enabled_explicitly(self):
        with patch(
            "robo_vision.april_tag_detector._apriltags_available",
            return_value=False,
        ):
            from robo_vision.detector import RoboEyeDetector

            det = RoboEyeDetector(enable_laser=True)
        assert det._laser_detector is not None
