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

# Valid single-character channel identifiers and their BGR indices
_CHANNEL_BGR_INDEX = {"b": 0, "g": 1, "r": 2}


class LaserSpotDetector:
    """Detect bright point-illumination spots (e.g. laser pointers) in a frame.

    Parameters
    ----------
    brightness_threshold:
        Lower grayscale intensity threshold (0-255).  Pixels brighter than
        this value are candidates.  Default ``240`` works well for a typical
        laser pointer; lower values detect dimmer spots at the risk of more
        false positives.
    brightness_threshold_max:
        Upper grayscale intensity threshold (0-255).  Pixels brighter than
        this value are *excluded*.  Together with *brightness_threshold*
        this defines a brightness window ``[brightness_threshold,
        brightness_threshold_max]``.  Default ``255`` (no upper filtering).
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
    channels:
        Which colour channels of the BGR image to analyse.  A string
        containing any combination of ``'r'``, ``'g'``, and ``'b'``
        (order does not matter).  When all three channels are selected
        (default ``"rgb"``), standard weighted grayscale conversion is
        used.  When a subset is selected, only the chosen channels are
        averaged to produce the brightness image, making it possible to
        isolate e.g. a red laser by passing ``channels="r"``.
    """

    def __init__(
        self,
        brightness_threshold: int = 240,
        brightness_threshold_max: int = 255,
        min_area: int = 4,
        max_area: int = 1000,
        min_circularity: float = 0.2,
        target_area: int = 100,
        sensitivity: int = 50,
        channels: str = "rgb",
    ) -> None:
        if not (0 <= brightness_threshold <= 255):
            raise ValueError(
                f"brightness_threshold must be in [0, 255], got {brightness_threshold}"
            )
        if not (0 <= brightness_threshold_max <= 255):
            raise ValueError(
                f"brightness_threshold_max must be in [0, 255], got {brightness_threshold_max}"
            )
        if brightness_threshold_max < brightness_threshold:
            raise ValueError(
                f"brightness_threshold_max ({brightness_threshold_max}) must be "
                f">= brightness_threshold ({brightness_threshold})"
            )
        if min_area < 0:
            raise ValueError(f"min_area must be >= 0, got {min_area}")
        if max_area <= min_area:
            raise ValueError(
                f"max_area ({max_area}) must be greater than min_area ({min_area})"
            )
        self.brightness_threshold = brightness_threshold
        self.brightness_threshold_max = brightness_threshold_max
        self.min_area = min_area
        self.max_area = max_area
        self.min_circularity = min_circularity
        self.target_area = target_area
        self.sensitivity = max(0, min(100, sensitivity))
        self.channels = channels  # validated by the property setter
        # Set after each detect() call; used by the GUI threshold overlay.
        self.last_threshold_mask: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # channels property
    # ------------------------------------------------------------------

    @property
    def channels(self) -> str:
        """Active colour channels (a subset of ``"rgb"``)."""
        return self._channels

    @channels.setter
    def channels(self, value: str) -> None:
        normalized = "".join(sorted(set(value.lower())))
        if not normalized or not all(ch in _CHANNEL_BGR_INDEX for ch in normalized):
            raise ValueError(
                f"channels must be a non-empty subset of 'r', 'g', 'b', got {value!r}"
            )
        self._channels = normalized

    # ------------------------------------------------------------------
    # Sensitivity / area helpers
    # ------------------------------------------------------------------

    def _compute_effective_area_bounds(self) -> tuple[int, int]:
        """Compute the effective min/max area window from sensitivity.

        The *area_spread* factor grows from 0.1 (tight) at sensitivity=0
        to 10.0 (wide) at sensitivity=100, producing the acceptance window
        ``[target_area / (1 + spread), target_area * (1 + spread)]``
        clamped to ``[min_area, max_area]``.
        """
        sens_norm = self.sensitivity / 100.0
        area_spread = 0.1 + sens_norm * 9.9
        effective_min = max(
            self.min_area,
            int(self.target_area / (1.0 + area_spread)),
        )
        effective_max = min(
            self.max_area,
            int(self.target_area * (1.0 + area_spread)),
        )
        return effective_min, effective_max

    def _compute_effective_circularity(self) -> float:
        """Compute the effective circularity threshold from sensitivity.

        At sensitivity=0 the threshold is the stricter of 0.8 or
        ``min_circularity``; at sensitivity=100 it relaxes to exactly
        ``min_circularity``.
        """
        sens_norm = self.sensitivity / 100.0
        high_circ_bound = max(self.min_circularity, 0.8)
        return self.min_circularity + (
            high_circ_bound - self.min_circularity
        ) * (1.0 - sens_norm)

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
        # Build a single-channel brightness image from the selected channels.
        # When all three channels are active the standard weighted grayscale
        # conversion is used (backward-compatible).  Otherwise only the
        # selected channels are averaged so that, e.g., a red laser can be
        # isolated by setting channels="r".
        if self._channels == "bgr":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            indices = [_CHANNEL_BGR_INDEX[ch] for ch in self._channels]
            if len(indices) == 1:
                gray = frame[:, :, indices[0]]
            else:
                gray = np.mean(frame[:, :, indices], axis=2).astype(np.uint8)

        # Isolate pixels within the brightness range and store the raw mask
        # for the GUI overlay.
        if self.brightness_threshold_max < 255:
            thresh = cv2.inRange(
                gray,
                self.brightness_threshold,
                self.brightness_threshold_max,
            )
        else:
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

        effective_min_area, effective_max_area = self._compute_effective_area_bounds()
        effective_min_circularity = self._compute_effective_circularity()

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
