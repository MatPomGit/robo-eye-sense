"""Laser-spot detector based on brightness thresholding.

A laser pointer creates a very small, very bright spot in the camera image.
The detector thresholds the grayscale brightness channel, then applies size
and circularity filters to isolate genuine spots and discard large bright
objects (lamps, windows, etc.).
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np

from .results import Detection, DetectionType


class LaserSpotDetector:
    """Detect bright point-illumination spots (e.g. laser pointers) in a frame.

    Parameters
    ----------
    brightness_threshold:
        Grayscale intensity threshold (0-255).  Pixels brighter than this
        value are candidates.  Default ``240`` works well for a typical
        laser pointer; lower values detect dimmer spots at the risk of more
        false positives.
    min_area:
        Minimum contour area in pixels.  Filters out single-pixel noise.
    max_area:
        Maximum contour area in pixels.  Filters out large bright objects
        such as light fixtures.
    min_circularity:
        Minimum circularity score ``4π·area/perimeter²`` in ``[0, 1]``.
        A perfect circle scores ``1.0``; the default ``0.2`` is permissive
        enough to tolerate spots blurred by defocus or surface angle.
    """

    def __init__(
        self,
        brightness_threshold: int = 240,
        min_area: int = 4,
        max_area: int = 1000,
        min_circularity: float = 0.2,
    ) -> None:
        self.brightness_threshold = brightness_threshold
        self.min_area = min_area
        self.max_area = max_area
        self.min_circularity = min_circularity

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Return laser-spot detections found in *frame*.

        Parameters
        ----------
        frame:
            BGR image as a NumPy array (H × W × 3, dtype uint8).

        Returns
        -------
        List[Detection]
            Each detection has ``detection_type=DetectionType.LASER_SPOT``,
            ``identifier=None``, the spot centre, and a bounding-rect
            approximation of the corners.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Isolate very bright regions
        _, thresh = cv2.threshold(
            gray, self.brightness_threshold, 255, cv2.THRESH_BINARY
        )

        # Morphological close to fill small holes inside a spot
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections: List[Detection] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue

            # Circularity filter
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                circularity = 4.0 * np.pi * area / (perimeter ** 2)
                if circularity < self.min_circularity:
                    continue

            moments = cv2.moments(cnt)
            if moments["m00"] <= 0:
                continue
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])

            x, y, w, h = cv2.boundingRect(cnt)
            corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]

            detections.append(
                Detection(
                    detection_type=DetectionType.LASER_SPOT,
                    identifier=None,
                    center=(cx, cy),
                    corners=corners,
                )
            )

        return detections
