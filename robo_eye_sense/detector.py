"""Main orchestrator: RoboEyeDetector.

Combines AprilTag, QR-code, and laser-spot detectors into a single
pipeline that also runs the centroid tracker.  Designed for continuous
use in a ``while True`` video loop.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .april_tag_detector import AprilTagDetector, _apriltags_available
from .laser_detector import LaserSpotDetector
from .qr_detector import QRCodeDetector
from .results import Detection, DetectionType
from .tracker import CentroidTracker

# BGR colours used when drawing each detection type
_COLOURS: Dict[DetectionType, Tuple[int, int, int]] = {
    DetectionType.APRIL_TAG: (0, 255, 0),    # green
    DetectionType.QR_CODE: (255, 128, 0),    # blue-ish
    DetectionType.LASER_SPOT: (0, 255, 255), # yellow
}


class RoboEyeDetector:
    """All-in-one detector for visual markers and laser spots.

    Parameters
    ----------
    enable_apriltag:
        Enable AprilTag detection (requires *pupil-apriltags*).
    enable_qr:
        Enable QR-code detection.
    enable_laser:
        Enable laser-spot detection.
    april_families:
        AprilTag family string passed to :class:`~robo_eye_sense.april_tag_detector.AprilTagDetector`.
    april_quad_decimate:
        Down-sampling factor for AprilTag detection.  Higher values are
        faster but reduce detection range.
    laser_brightness_threshold:
        Pixel brightness threshold for laser-spot detection (0-255).
    tracker_max_disappeared:
        Frames before a lost track is removed.
    tracker_max_distance:
        Maximum centroid distance for matching unlabeled tracks.
    """

    def __init__(
        self,
        enable_apriltag: bool = True,
        enable_qr: bool = True,
        enable_laser: bool = True,
        april_families: str = "tag36h11",
        april_quad_decimate: float = 2.0,
        laser_brightness_threshold: int = 240,
        tracker_max_disappeared: int = 10,
        tracker_max_distance: int = 50,
    ) -> None:
        self._april_detector: Optional[AprilTagDetector] = None
        self._qr_detector: Optional[QRCodeDetector] = None
        self._laser_detector: Optional[LaserSpotDetector] = None

        if enable_apriltag:
            if _apriltags_available():
                self._april_detector = AprilTagDetector(
                    families=april_families,
                    quad_decimate=april_quad_decimate,
                )
            else:
                warnings.warn(
                    "pupil-apriltags not installed – AprilTag detection disabled. "
                    "Install with:  pip install pupil-apriltags",
                    RuntimeWarning,
                    stacklevel=2,
                )

        if enable_qr:
            self._qr_detector = QRCodeDetector()

        if enable_laser:
            self._laser_detector = LaserSpotDetector(
                brightness_threshold=laser_brightness_threshold
            )

        self._tracker = CentroidTracker(
            max_disappeared=tracker_max_disappeared,
            max_distance=tracker_max_distance,
        )

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> List[Detection]:
        """Run all enabled detectors on *frame* and return tracked detections.

        Parameters
        ----------
        frame:
            BGR image (H × W × 3, uint8).

        Returns
        -------
        List[Detection]
            All detections found, each with a populated ``track_id``.
        """
        detections: List[Detection] = []

        # Pre-compute grayscale once for detectors that need it
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._april_detector is not None:
            detections.extend(self._april_detector.detect(gray))

        if self._qr_detector is not None:
            detections.extend(self._qr_detector.detect(frame))

        if self._laser_detector is not None:
            detections.extend(self._laser_detector.detect(frame))

        self._tracker.update(detections)
        return detections

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def draw_detections(
        self, frame: np.ndarray, detections: List[Detection]
    ) -> np.ndarray:
        """Draw bounding polygons, centres, and labels onto *frame*.

        Modifies *frame* in-place and returns it for convenience.

        Parameters
        ----------
        frame:
            BGR image to draw on.
        detections:
            Detections returned by :meth:`process_frame`.

        Returns
        -------
        np.ndarray
            The annotated frame.
        """
        for d in detections:
            colour = _COLOURS.get(d.detection_type, (255, 255, 255))

            # Bounding polygon
            if len(d.corners) >= 2:
                pts = np.array(d.corners, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], isClosed=True, color=colour, thickness=2)

            # Centre dot
            cv2.circle(frame, d.center, 5, colour, -1)

            # Label: "<type>[:<identifier>] [#<track_id>]"
            label = d.detection_type.value.replace("_", " ").title()
            if d.identifier:
                # Truncate long QR payloads for readability
                payload = d.identifier[:20] + ("…" if len(d.identifier) > 20 else "")
                label += f": {payload}"
            if d.track_id is not None:
                label += f"  #{d.track_id}"

            text_x = d.center[0] + 8
            text_y = d.center[1] - 8
            cv2.putText(
                frame,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                colour,
                1,
                cv2.LINE_AA,
            )

        return frame
