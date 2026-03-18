"""Main orchestrator: RoboEyeDetector.

Combines AprilTag, QR-code, and laser-spot detectors into a single
pipeline that also runs the centroid tracker.  Designed for continuous
use in a ``while True`` video loop.

Three operating modes are supported (see
:class:`~robo_eye_sense.results.DetectionMode`):

* **NORMAL** – default balanced mode (original behaviour).
* **FAST** – input frame downscaled 50 % before detection so that all
  detectors process only ¼ of the original pixels.  Detected coordinates
  are scaled back to original resolution before being returned.  Suitable
  for resource-constrained hardware.
* **ROBUST** – unsharp-mask sharpening applied before detection to
  counter motion blur; the centroid tracker uses a Kalman-filter velocity
  model for predictive matching and the disappearance / distance budgets
  are widened so that a briefly lost track survives temporary blurring.
  Best used when the camera or robot moves quickly.
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
from .results import Detection, DetectionMode, DetectionType
from .tracker import CentroidTracker

# BGR colours used when drawing each detection type on the annotated frame
_COLOURS: Dict[DetectionType, Tuple[int, int, int]] = {
    DetectionType.APRIL_TAG: (0, 255, 0),    # green
    DetectionType.QR_CODE: (255, 128, 0),    # bluish
    DetectionType.LASER_SPOT: (0, 255, 255),  # yellow
}

# Mode indicator text rendered in the top-right corner of each annotated frame
_MODE_LABELS: Dict[DetectionMode, str] = {
    DetectionMode.NORMAL: "Mode: Normal",
    DetectionMode.FAST:   "Mode: Fast",
    DetectionMode.ROBUST: "Mode: Robust",
}

# Scale factor applied in FAST mode
_FAST_SCALE = 0.5

# Tracker parameter overrides per mode
_MODE_TRACKER_PARAMS: Dict[DetectionMode, Dict[str, int]] = {
    DetectionMode.NORMAL: {"max_disappeared": 10, "max_distance": 50},
    DetectionMode.FAST:   {"max_disappeared": 5,  "max_distance": 50},
    DetectionMode.ROBUST: {"max_disappeared": 20, "max_distance": 100},
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


def _sharpen_frame(frame: np.ndarray) -> np.ndarray:
    """Return a sharpened copy of *frame* using an unsharp mask.

    Subtracts a Gaussian-blurred version from the original so that edges
    and fine detail are accentuated.  This partially counteracts motion blur
    and temporary defocus, improving detection recall in ROBUST mode.
    """
    blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=3)
    return cv2.addWeighted(frame, 1.5, blurred, -0.5, 0)


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
    mode:
        Operating mode.  See :class:`~robo_eye_sense.results.DetectionMode`
        for a description of each mode.
    laser_brightness_threshold:
        Pixel brightness threshold for laser-spot detection (0-255).
    laser_brightness_threshold_max:
        Upper pixel brightness threshold for laser-spot detection (0-255).
    laser_target_area:
        Target laser-spot area in pixels for laser detection.
    laser_sensitivity:
        Detection sensitivity (0-100) for laser-spot detection.
    tag_names:
        Optional mapping of AprilTag ID strings to human-readable names
        (e.g. ``{"1": "box", "2": "table"}``).  When provided, the name
        is appended to the detection's identifier.
    tracker_max_disappeared:
        Frames before a lost track is removed (used in NORMAL mode; the
        value is adjusted automatically in FAST and ROBUST modes).
    tracker_max_distance:
        Maximum centroid distance for matching unlabeled tracks (used in
        NORMAL mode; adjusted automatically in FAST and ROBUST modes).
    """

    def __init__(
        self,
        enable_apriltag: bool = True,
        enable_qr: bool = False,
        enable_laser: bool = False,
        mode: DetectionMode = DetectionMode.NORMAL,
        laser_brightness_threshold: int = 240,
        laser_brightness_threshold_max: int = 255,
        laser_target_area: int = 100,
        laser_sensitivity: int = 50,
        tag_names: Optional[Dict[str, str]] = None,
        laser_channels: str = "rgb",
        tracker_max_disappeared: int = 10,
        tracker_max_distance: int = 50,
    ) -> None:
        self._april_detector: Optional[april_tag_detector.AprilTagDetector] = None
        self._qr_detector: Optional[QRCodeDetector] = None
        self._laser_detector: Optional[LaserSpotDetector] = None

        self._tag_names: Dict[str, str] = dict(tag_names) if tag_names else {}

        # Store the user-supplied normal-mode tracker parameters so that the
        # mode setter can restore them when switching back to NORMAL.
        self._tracker_max_disappeared_normal = tracker_max_disappeared
        self._tracker_max_distance_normal = tracker_max_distance

        if enable_apriltag:
            if _apriltags_available():
                self._april_detector = april_tag_detector.AprilTagDetector()
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
                brightness_threshold_max=laser_brightness_threshold_max,
                target_area=laser_target_area,
                sensitivity=laser_sensitivity,
                channels=laser_channels,
            )

        # Create tracker with parameters appropriate for the requested mode.
        # In NORMAL mode honour user-provided tracker values.
        tp = {
            "max_disappeared": tracker_max_disappeared,
            "max_distance": tracker_max_distance,
        } if mode == DetectionMode.NORMAL else _MODE_TRACKER_PARAMS[mode]
        self._tracker = CentroidTracker(
            max_disappeared=tp["max_disappeared"],
            max_distance=tp["max_distance"],
            use_kalman=(mode == DetectionMode.ROBUST),
        )

        # Set mode last (after tracker is ready)
        self._mode: DetectionMode = mode

    # ------------------------------------------------------------------
    # Mode property
    # ------------------------------------------------------------------

    @property
    def mode(self) -> DetectionMode:
        """Current operating mode."""
        return self._mode

    @mode.setter
    def mode(self, value: DetectionMode) -> None:
        """Switch operating mode and update tracker parameters accordingly."""
        if value == self._mode:
            return
        self._mode = value
        tp = _MODE_TRACKER_PARAMS[value]
        self._tracker.max_disappeared = (
            self._tracker_max_disappeared_normal
            if value == DetectionMode.NORMAL
            else tp["max_disappeared"]
        )
        self._tracker.max_distance = (
            self._tracker_max_distance_normal
            if value == DetectionMode.NORMAL
            else tp["max_distance"]
        )
        self._tracker.use_kalman = (value == DetectionMode.ROBUST)

    # ------------------------------------------------------------------
    # Public detector-state API
    # ------------------------------------------------------------------

    @property
    def april_enabled(self) -> bool:
        """Whether AprilTag detection is currently active."""
        return self._april_detector is not None

    @property
    def qr_enabled(self) -> bool:
        """Whether QR-code detection is currently active."""
        return self._qr_detector is not None

    @property
    def laser_enabled(self) -> bool:
        """Whether laser-spot detection is currently active."""
        return self._laser_detector is not None

    @property
    def laser_detector(self) -> Optional[LaserSpotDetector]:
        """The active :class:`LaserSpotDetector`, or ``None`` if disabled."""
        return self._laser_detector

    @property
    def tag_names(self) -> Dict[str, str]:
        """Mapping of AprilTag ID strings to human-readable names."""
        return self._tag_names

    @tag_names.setter
    def tag_names(self, value: Dict[str, str]) -> None:
        self._tag_names = dict(value) if value else {}

    def enable_april(self) -> bool:
        """Enable the AprilTag detector.  Returns ``True`` on success."""
        if self._april_detector is not None:
            return True
        if not _apriltags_available():
            return False
        self._april_detector = april_tag_detector.AprilTagDetector()
        return True

    def disable_april(self) -> None:
        """Disable the AprilTag detector."""
        self._april_detector = None

    def enable_qr(self) -> None:
        """Enable the QR-code detector."""
        if self._qr_detector is None:
            self._qr_detector = QRCodeDetector()

    def disable_qr(self) -> None:
        """Disable the QR-code detector."""
        self._qr_detector = None

    def enable_laser(
        self,
        brightness_threshold: int = 240,
        brightness_threshold_max: int = 255,
        target_area: int = 100,
        sensitivity: int = 50,
        channels: str = "rgb",
    ) -> None:
        """Enable the laser-spot detector with the given parameters."""
        if self._laser_detector is None:
            self._laser_detector = LaserSpotDetector(
                brightness_threshold=brightness_threshold,
                brightness_threshold_max=brightness_threshold_max,
                target_area=target_area,
                sensitivity=sensitivity,
                channels=channels,
            )

    def disable_laser(self) -> None:
        """Disable the laser-spot detector."""
        self._laser_detector = None

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> List[Detection]:
        """Run all enabled detectors on *frame* and return tracked detections.

        The processing pipeline is adapted to the current :attr:`mode`:

        * **NORMAL** – original pipeline unchanged.
        * **FAST** – frame is downscaled to 50 % (¼ pixels) before detection;
          all returned coordinates are scaled back to original resolution.
        * **ROBUST** – an unsharp-mask sharpening filter is applied before
          running the standard detection pipeline; the tracker uses Kalman
          filtering for predictive matching.

        Parameters
        ----------
        frame:
            BGR image (H × W × 3, uint8).

        Returns
        -------
        List[Detection]
            All detections found, each with a populated ``track_id``.
        """
        # FAST mode: downscale the frame before detection
        if self._mode == DetectionMode.FAST:
            h, w = frame.shape[:2]
            frame = cv2.resize(
                frame,
                (max(1, int(w * _FAST_SCALE)), max(1, int(h * _FAST_SCALE))),
            )

        # ROBUST mode: sharpen to counteract motion blur
        if self._mode == DetectionMode.ROBUST:
            frame = _sharpen_frame(frame)

        detections = self._run_detectors(frame)

        # FAST mode: scale coordinates back to original resolution
        if self._mode == DetectionMode.FAST:
            inv = 1.0 / _FAST_SCALE
            for d in detections:
                d.center = (int(d.center[0] * inv), int(d.center[1] * inv))
                d.corners = [(int(x * inv), int(y * inv)) for x, y in d.corners]

        self._tracker.update(detections)
        return detections

    def _run_detectors(self, frame: np.ndarray) -> List[Detection]:
        """Run all enabled sub-detectors on *frame* and return raw results.

        This is the common detection step shared by every pipeline mode.
        """
        detections: List[Detection] = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._april_detector is not None:
            april_detections = self._april_detector.detect(gray)
            # Enrich AprilTag detections with human-readable names
            if self._tag_names:
                for d in april_detections:
                    if d.identifier and d.identifier in self._tag_names:
                        d.identifier = f"{d.identifier} ({self._tag_names[d.identifier]})"
            detections.extend(april_detections)
        if self._qr_detector is not None:
            detections.extend(self._qr_detector.detect(frame))
        if self._laser_detector is not None:
            detections.extend(self._laser_detector.detect(frame))

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

        A mode indicator is rendered in the top-right corner of the frame.

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

        # Mode indicator – top-right corner
        mode_label = _MODE_LABELS.get(self._mode, "")
        if mode_label:
            (lw, lh), _ = cv2.getTextSize(
                mode_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            frame_w = frame.shape[1]
            cv2.putText(
                frame,
                mode_label,
                (frame_w - lw - 8, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

        return frame
