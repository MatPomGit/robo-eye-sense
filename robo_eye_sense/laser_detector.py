"""Laser-spot detector based on brightness thresholding.

A laser pointer creates a very small, very bright spot in the camera image.
The detector thresholds the grayscale brightness channel, then applies size
and circularity filters to isolate genuine spots and discard large bright
objects (lamps, windows, etc.).

The *sensitivity* parameter (0–100) provides a single knob to trade off
precision against recall: at low values only spots that closely match the
*target_area* and are highly circular are accepted; at high values the
acceptance window widens and the circularity requirement is relaxed.
"""

from __future__ import annotations

from typing import List, Optional

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
        Hard minimum contour area in pixels.  Filters out single-pixel noise.
    max_area:
        Hard maximum contour area in pixels.  Upper absolute limit; the
        effective maximum is also bounded by *target_area* and *sensitivity*.
    min_circularity:
        Minimum circularity score ``4π·area/perimeter²`` in ``[0, 1]``.
        A perfect circle scores ``1.0``; the default ``0.2`` is the most
        permissive threshold (applied at maximum *sensitivity*).
    target_area:
        Target spot area in pixels.  Together with *sensitivity* this
        controls the acceptable area window.  Default ``100`` pixels.
    sensitivity:
        Detection sensitivity in the range ``[0, 100]``.  At low values
        only spots that closely match *target_area* and have high circularity
        are accepted (fewer but more reliable detections).  At high values
        the acceptance window widens and circularity requirements are relaxed
        (more detections, potentially including weaker spots).  Default ``50``.
    """

    def __init__(
        self,
        brightness_threshold: int = 240,
        min_area: int = 4,
        max_area: int = 1000,
        min_circularity: float = 0.2,
        target_area: int = 100,
        sensitivity: int = 50,
    ) -> None:
        self.brightness_threshold = brightness_threshold
        self.min_area = min_area
        self.max_area = max_area
        self.min_circularity = min_circularity
        self.target_area = target_area
        self.sensitivity = max(0, min(100, sensitivity))
        # Set after each detect() call; used by the GUI threshold overlay.
        self.last_threshold_mask: Optional[np.ndarray] = None

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

        # Isolate very bright regions and store the raw mask for the GUI overlay
        _, thresh = cv2.threshold(
            gray, self.brightness_threshold, 255, cv2.THRESH_BINARY
        )
        self.last_threshold_mask = thresh.copy()

        # Morphological close to fill small holes inside a spot
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Compute effective area bounds from target_area and sensitivity.
        # area_spread grows from 0.1× (tight) to 10× (wide) as sensitivity
        # increases from 0 to 100, giving the acceptable area window
        # [target_area / (1 + spread), target_area * (1 + spread)].
        sens_norm = self.sensitivity / 100.0
        area_spread = 0.1 + sens_norm * 9.9
        effective_min_area = max(
            self.min_area,
            int(self.target_area / (1.0 + area_spread)),
        )
        effective_max_area = min(
            self.max_area,
            int(self.target_area * (1.0 + area_spread)),
        )

        # Circularity requirement: strictest (0.8) at sensitivity=0,
        # relaxing to min_circularity at sensitivity=100.
        high_circ_bound = max(self.min_circularity, 0.8)
        effective_min_circularity = self.min_circularity + (
            high_circ_bound - self.min_circularity
        ) * (1.0 - sens_norm)

        detections: List[Detection] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < effective_min_area or area > effective_max_area:
                continue

            # Circularity filter
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                circularity = 4.0 * np.pi * area / (perimeter ** 2)
                if circularity < effective_min_circularity:
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
