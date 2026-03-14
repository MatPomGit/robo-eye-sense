"""Tests for LaserSpotDetector using synthetic images."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from robo_eye_sense.laser_detector import LaserSpotDetector
from robo_eye_sense.results import DetectionType


class TestLaserSpotDetector:
    def _detector(self, **kwargs):
        return LaserSpotDetector(**kwargs)

    def _spot_frame(self, cx, cy, radius=6):
        """Create a black BGR frame with one bright white circle."""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(frame, (cx, cy), radius, (255, 255, 255), -1)
        return frame

    # ------------------------------------------------------------------
    # Basic detection
    # ------------------------------------------------------------------

    def test_detects_single_bright_spot(self, bright_spot_frame):
        detector = self._detector()
        results = detector.detect(bright_spot_frame)
        assert len(results) == 1

    def test_no_detection_on_black_frame(self, black_frame):
        detector = self._detector()
        results = detector.detect(black_frame)
        assert results == []

    def test_detects_two_spots(self, two_spots_frame):
        detector = self._detector()
        results = detector.detect(two_spots_frame)
        assert len(results) == 2

    # ------------------------------------------------------------------
    # Detection content
    # ------------------------------------------------------------------

    def test_detection_type(self, bright_spot_frame):
        detector = self._detector()
        d = detector.detect(bright_spot_frame)[0]
        assert d.detection_type == DetectionType.LASER_SPOT

    def test_identifier_is_none(self, bright_spot_frame):
        detector = self._detector()
        d = detector.detect(bright_spot_frame)[0]
        assert d.identifier is None

    def test_center_approximately_correct(self):
        detector = self._detector()
        frame = self._spot_frame(80, 120, radius=5)
        results = detector.detect(frame)
        assert len(results) == 1
        cx, cy = results[0].center
        assert abs(cx - 80) <= 5
        assert abs(cy - 120) <= 5

    def test_corners_are_four_points(self, bright_spot_frame):
        detector = self._detector()
        d = detector.detect(bright_spot_frame)[0]
        assert len(d.corners) == 4

    # ------------------------------------------------------------------
    # Parameter effects
    # ------------------------------------------------------------------

    def test_high_threshold_misses_dim_spot(self):
        """A dim spot (brightness 200) should be missed if threshold is 230."""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(frame, (100, 100), 6, (200, 200, 200), -1)
        detector = self._detector(brightness_threshold=230)
        results = detector.detect(frame)
        assert results == []

    def test_lower_threshold_finds_dim_spot(self):
        """Same dim spot detected when threshold is lowered."""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(frame, (100, 100), 6, (200, 200, 200), -1)
        detector = self._detector(brightness_threshold=180)
        results = detector.detect(frame)
        assert len(results) == 1

    def test_large_bright_region_filtered_by_max_area(self):
        """A large bright rectangle should not be detected as a laser spot."""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(frame, (10, 10), (100, 100), (255, 255, 255), -1)
        detector = self._detector(max_area=500)
        results = detector.detect(frame)
        assert results == []

    def test_non_circular_region_filtered_by_circularity(self):
        """A very elongated bright line should not be detected."""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        # Draw a thin horizontal line – very low circularity
        cv2.rectangle(frame, (5, 99), (195, 101), (255, 255, 255), -1)
        detector = self._detector(min_circularity=0.5, max_area=10000)
        results = detector.detect(frame)
        assert results == []
