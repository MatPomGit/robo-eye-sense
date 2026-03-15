"""Main orchestrator: RoboEyeDetector.

Combines AprilTag, QR-code, and laser-spot detectors into a single
pipeline that also runs the centroid tracker.  Designed for continuous
use in a ``while True`` video loop.
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import april_tag_detector
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

# Length of axis arrows drawn at the detection centre
_AXIS_LENGTH = 40




def _apriltags_available() -> bool:
    """Proxy helper so tests can patch either detector or april_tag module."""
    return april_tag_detector._apriltags_available()


def _compute_orientation(corners: List[Tuple[int, int]]) -> float:
    """Return orientation angle in degrees computed from bounding corners.

    The angle is measured between the first two corners projected onto the
    image X-axis.  Returns ``0.0`` when fewer than two corners are available.
    """
    if len(corners) < 2:
        return 0.0
    dx = corners[1][0] - corners[0][0]
    dy = corners[1][1] - corners[0][1]
    return math.degrees(math.atan2(dy, dx))


def _draw_axes(
    frame: np.ndarray,
    center: Tuple[int, int],
    angle_deg: float,
    length: int = _AXIS_LENGTH,
) -> None:
    """Draw a 2-D coordinate-axis glyph (X red, Y green) at *center*.

    The glyph is rotated by *angle_deg* so that it aligns with the detected
    object's local orientation.

    Parameters
    ----------
    frame:
        BGR image to draw on (modified in-place).
    center:
        ``(x, y)`` pixel origin of the glyph.
    angle_deg:
        Rotation angle in degrees (measured CCW from the positive X-axis).
    length:
        Arrow length in pixels.
    """
    cx, cy = center
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    # X axis (red) – points in the object's local "right" direction
    x_tip = (int(cx + length * cos_a), int(cy + length * sin_a))
    cv2.arrowedLine(frame, (cx, cy), x_tip, (0, 0, 255), 2, tipLength=0.25)
    cv2.putText(
        frame, "X", (x_tip[0] + 4, x_tip[1] + 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA,
    )

    # Y axis (green) – perpendicular (90° CCW from X in image coords)
    y_tip = (int(cx - length * sin_a), int(cy + length * cos_a))
    cv2.arrowedLine(frame, (cx, cy), y_tip, (0, 255, 0), 2, tipLength=0.25)
    cv2.putText(
        frame, "Y", (y_tip[0] + 4, y_tip[1] + 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA,
    )


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
    laser_target_area:
        Target laser-spot area in pixels for laser detection.
    laser_sensitivity:
        Detection sensitivity (0-100) for laser-spot detection.
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
        laser_target_area: int = 100,
        laser_sensitivity: int = 50,
        tracker_max_disappeared: int = 10,
        tracker_max_distance: int = 50,
    ) -> None:
        self._april_detector: Optional[april_tag_detector.AprilTagDetector] = None
        self._qr_detector: Optional[QRCodeDetector] = None
        self._laser_detector: Optional[LaserSpotDetector] = None

        if enable_apriltag:
            if _apriltags_available():
                self._april_detector = april_tag_detector.AprilTagDetector(
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
                brightness_threshold=laser_brightness_threshold,
                target_area=laser_target_area,
                sensitivity=laser_sensitivity,
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
        """Draw bounding polygons, coordinate axes, and info labels onto *frame*.

        Each detected object receives:

        * A bounding polygon drawn in the category colour.
        * A filled dot at the centroid.
        * A 2-D coordinate-axis glyph (X red, Y green) anchored at the
          centroid and rotated to match the object's local orientation.
        * Multi-line position / orientation annotation that moves with the
          object: pixel position ``(x, y)``, orientation angle ``θ``, the
          detection category, optional identifier, and track ID.

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

            # Compute object orientation from its corners (0° when no corners)
            angle = _compute_orientation(d.corners)

            # Draw coordinate-axis glyph at the centroid
            _draw_axes(frame, d.center, angle)

            # Build multi-line annotation that moves with the object
            cx, cy = d.center
            lines: List[str] = [
                f"X:{cx}  Y:{cy}",
                f"ang:{angle:.1f}deg",
            ]
            label = d.detection_type.value.replace("_", " ").title()
            if d.identifier:
                payload = d.identifier[:20] + ("..." if len(d.identifier) > 20 else "")
                label += f": {payload}"
            if d.track_id is not None:
                label += f"  #{d.track_id}"
            lines.append(label)

            # Render lines below and to the right of the centroid
            text_x = cx + _AXIS_LENGTH + 6
            text_y = cy - 20
            line_height = 16
            for i, line in enumerate(lines):
                cv2.putText(
                    frame,
                    line,
                    (text_x, text_y + i * line_height),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    colour,
                    1,
                    cv2.LINE_AA,
                )

        return frame
