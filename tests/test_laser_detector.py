"""Tests for LaserSpotDetector using synthetic images."""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="OpenCV runtime dependencies are unavailable", exc_type=ImportError)

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

    # ------------------------------------------------------------------
    # target_area parameter
    # ------------------------------------------------------------------

    def test_target_area_filters_spot_outside_window(self):
        """Spot whose area differs greatly from target_area at low sensitivity
        should not be detected."""
        # Large circle: area ~ π*30² ~ 2827 pixels
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(frame, (100, 100), 30, (255, 255, 255), -1)
        # target_area=10 with very low sensitivity → tight window that
        # excludes the large circle (area far from 10)
        detector = self._detector(
            target_area=10, sensitivity=0, max_area=10000
        )
        results = detector.detect(frame)
        assert results == []

    def test_target_area_accepts_matching_spot(self):
        """Spot whose area is close to target_area should be accepted."""
        # Circle with radius=6: area ~ 113 px
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(frame, (100, 100), 6, (255, 255, 255), -1)
        # target_area=100 with medium sensitivity → window includes ~113 px
        detector = self._detector(target_area=100, sensitivity=50)
        results = detector.detect(frame)
        assert len(results) == 1

    # ------------------------------------------------------------------
    # sensitivity parameter
    # ------------------------------------------------------------------

    def test_low_sensitivity_detects_fewer_spots_than_high(self):
        """Higher sensitivity should accept at least as many spots as lower
        sensitivity when both threshold and target_area are constant."""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        # Two circles with radii 6 and 20 – different areas
        cv2.circle(frame, (60, 60), 6, (255, 255, 255), -1)
        cv2.circle(frame, (140, 140), 20, (255, 255, 255), -1)

        det_low = self._detector(target_area=50, sensitivity=5, max_area=10000)
        det_high = self._detector(target_area=50, sensitivity=95, max_area=10000)
        count_low = len(det_low.detect(frame))
        count_high = len(det_high.detect(frame))
        assert count_high >= count_low

    def test_zero_sensitivity_requires_high_circularity(self):
        """At sensitivity=0 the effective circularity threshold rises to ~0.8,
        so a very elongated region should be rejected."""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        # Elongated rectangle: low circularity
        cv2.rectangle(frame, (5, 90), (195, 110), (255, 255, 255), -1)
        detector = self._detector(
            sensitivity=0, target_area=380, max_area=10000, min_circularity=0.05
        )
        results = detector.detect(frame)
        assert results == []

    # ------------------------------------------------------------------
    # last_threshold_mask attribute
    # ------------------------------------------------------------------

    def test_last_threshold_mask_set_after_detect(self, bright_spot_frame):
        """detect() should populate last_threshold_mask."""
        detector = self._detector()
        assert detector.last_threshold_mask is None
        detector.detect(bright_spot_frame)
        assert detector.last_threshold_mask is not None

    def test_last_threshold_mask_shape_matches_frame(self, bright_spot_frame):
        detector = self._detector()
        detector.detect(bright_spot_frame)
        h, w = bright_spot_frame.shape[:2]
        assert detector.last_threshold_mask.shape == (h, w)

    # ------------------------------------------------------------------
    # Parameter validation
    # ------------------------------------------------------------------

    def test_invalid_brightness_threshold_too_high(self):
        with pytest.raises(ValueError, match="brightness_threshold"):
            LaserSpotDetector(brightness_threshold=256)

    def test_invalid_brightness_threshold_negative(self):
        with pytest.raises(ValueError, match="brightness_threshold"):
            LaserSpotDetector(brightness_threshold=-1)

    def test_invalid_min_area_negative(self):
        with pytest.raises(ValueError, match="min_area"):
            LaserSpotDetector(min_area=-1)

    def test_invalid_max_area_less_than_min_area(self):
        with pytest.raises(ValueError, match="max_area"):
            LaserSpotDetector(min_area=100, max_area=50)

    def test_invalid_max_area_equal_to_min_area(self):
        with pytest.raises(ValueError, match="max_area"):
            LaserSpotDetector(min_area=100, max_area=100)

    # ------------------------------------------------------------------
    # Extracted helper methods
    # ------------------------------------------------------------------

    def test_compute_effective_area_bounds_low_sensitivity(self):
        """Low sensitivity produces a tight area window around target_area."""
        det = self._detector(target_area=100, sensitivity=0, min_area=4, max_area=1000)
        lo, hi = det._compute_effective_area_bounds()
        assert lo > 80  # tight window
        assert hi < 120

    def test_compute_effective_area_bounds_high_sensitivity(self):
        """High sensitivity produces a wide area window."""
        det = self._detector(target_area=100, sensitivity=100, min_area=4, max_area=1000)
        lo, hi = det._compute_effective_area_bounds()
        assert lo <= 10  # wide window
        assert hi >= 900

    def test_compute_effective_circularity_zero_sensitivity(self):
        """At sensitivity=0 circularity threshold should be ~0.8."""
        det = self._detector(sensitivity=0, min_circularity=0.2)
        circ = det._compute_effective_circularity()
        assert abs(circ - 0.8) < 0.01

    def test_compute_effective_circularity_full_sensitivity(self):
        """At sensitivity=100 circularity threshold should equal min_circularity."""
        det = self._detector(sensitivity=100, min_circularity=0.2)
        circ = det._compute_effective_circularity()
        assert abs(circ - 0.2) < 0.01

    # ------------------------------------------------------------------
    # channels parameter
    # ------------------------------------------------------------------

    def test_default_channels_is_bgr(self):
        """Default channels value should normalise to 'bgr'."""
        det = self._detector()
        assert det.channels == "bgr"

    def test_channels_single_red(self):
        """channels='r' should be accepted and stored."""
        det = self._detector(channels="r")
        assert det.channels == "r"

    def test_channels_two(self):
        """channels='rg' should normalise to 'gr'."""
        det = self._detector(channels="rg")
        assert det.channels == "gr"

    def test_channels_setter_normalises(self):
        """Setting channels at runtime should normalise and validate."""
        det = self._detector()
        det.channels = "BR"
        assert det.channels == "br"

    def test_channels_invalid_raises(self):
        """Invalid channel letter should raise ValueError."""
        with pytest.raises(ValueError, match="channels"):
            LaserSpotDetector(channels="x")

    def test_channels_empty_raises(self):
        """Empty channels string should raise ValueError."""
        with pytest.raises(ValueError, match="channels"):
            LaserSpotDetector(channels="")

    def test_detect_red_channel_only(self):
        """A pure-red spot should be detected when channels='r'."""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        # BGR: red = (0, 0, 255)
        cv2.circle(frame, (100, 100), 6, (0, 0, 255), -1)
        det = self._detector(channels="r")
        results = det.detect(frame)
        assert len(results) == 1

    def test_detect_red_channel_ignores_blue_spot(self):
        """A pure-blue spot should NOT be detected when channels='r'."""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        # BGR: blue = (255, 0, 0)
        cv2.circle(frame, (100, 100), 6, (255, 0, 0), -1)
        det = self._detector(channels="r")
        results = det.detect(frame)
        assert results == []

    def test_detect_blue_channel_finds_blue_spot(self):
        """A pure-blue spot should be detected when channels='b'."""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(frame, (100, 100), 6, (255, 0, 0), -1)
        det = self._detector(channels="b")
        results = det.detect(frame)
        assert len(results) == 1

    def test_detect_green_channel_finds_green_spot(self):
        """A pure-green spot should be detected when channels='g'."""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(frame, (100, 100), 6, (0, 255, 0), -1)
        det = self._detector(channels="g")
        results = det.detect(frame)
        assert len(results) == 1

    def test_detect_two_channels_rg(self):
        """A red+green spot should be detected when channels='rg'."""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        # Yellow-ish (R+G bright, B zero) → BGR (0, 255, 255)
        cv2.circle(frame, (100, 100), 6, (0, 255, 255), -1)
        det = self._detector(channels="rg")
        results = det.detect(frame)
        assert len(results) == 1

    def test_all_channels_detects_white_spot(self):
        """Default (all channels) should detect a white spot."""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(frame, (100, 100), 6, (255, 255, 255), -1)
        det = self._detector()
        results = det.detect(frame)
        assert len(results) == 1
