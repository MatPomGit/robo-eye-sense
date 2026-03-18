"""Auto-follow scenario – track and follow a selected AprilTag marker.

The camera continuously detects markers and computes the displacement
vector ``(dx, dy)`` and yaw rotation angle required to centre the
selected marker in the frame.  When multiple markers are visible the
user can choose which one to follow by its numeric ID.

The module is intentionally decoupled from camera and detector
instantiation so that the core logic is easy to unit-test with synthetic
data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .offset_scenario import estimate_focal_length_px, _DEFAULT_HFOV_DEG
from .results import Detection, DetectionType


@dataclass
class AutoFollowResult:
    """Result of a single auto-follow computation.

    Attributes
    ----------
    position_vector:
        ``(dx, dy)`` pixel displacement from the current marker position
        to the frame centre.  Positive ``dx`` means the camera must move
        *right*; positive ``dy`` means it must move *down*.
        ``(0.0, 0.0)`` when no target marker is found.
    yaw:
        Estimated yaw angle (degrees) from the camera's optical axis to
        the target marker.  Positive means the marker is to the right of
        the camera centre.  ``0.0`` when no target is found.
    target_marker_id:
        Identifier of the marker currently being followed, or ``None``.
    target_found:
        ``True`` if the target marker was found in the current frame.
    visible_marker_ids:
        List of all visible AprilTag marker IDs in the current frame.
    marker_positions:
        Mapping from marker identifier to its ``(x, y)`` pixel centre.
    """

    position_vector: Tuple[float, float] = (0.0, 0.0)
    yaw: float = 0.0
    target_marker_id: Optional[str] = None
    target_found: bool = False
    visible_marker_ids: List[str] = field(default_factory=list)
    marker_positions: Dict[str, Tuple[int, int]] = field(default_factory=dict)


def compute_follow_vector(
    detections: List[Detection],
    frame_width: int = 640,
    frame_height: int = 480,
    target_marker_id: Optional[str] = None,
    hfov_deg: float = _DEFAULT_HFOV_DEG,
) -> AutoFollowResult:
    """Compute the follow vector for a given set of detections.

    Parameters
    ----------
    detections:
        Current frame detections.
    frame_width:
        Width of the captured frame in pixels.
    frame_height:
        Height of the captured frame in pixels.
    target_marker_id:
        Identifier of the marker to follow.  When ``None`` the first
        visible AprilTag is used.
    hfov_deg:
        Horizontal field-of-view of the camera in degrees.

    Returns
    -------
    AutoFollowResult
        Follow vector, yaw, and detection metadata.
    """
    # Collect all AprilTag positions
    marker_positions: Dict[str, Tuple[int, int]] = {}
    for d in detections:
        if d.detection_type == DetectionType.APRIL_TAG and d.identifier is not None:
            marker_positions[d.identifier] = d.center

    visible_ids = sorted(marker_positions.keys())

    if not visible_ids:
        return AutoFollowResult(
            visible_marker_ids=visible_ids,
            marker_positions=marker_positions,
        )

    # Select target marker
    if target_marker_id is not None and target_marker_id in marker_positions:
        chosen_id = target_marker_id
    else:
        # Default to first visible marker
        chosen_id = visible_ids[0]

    mx, my = marker_positions[chosen_id]
    cx = frame_width / 2.0
    cy = frame_height / 2.0

    # Displacement from marker to frame centre
    dx = cx - mx
    dy = cy - my

    # Yaw angle: horizontal angle from the optical axis to the marker
    focal = estimate_focal_length_px(frame_width, hfov_deg)
    yaw = math.degrees(math.atan2(float(mx - cx), focal))

    return AutoFollowResult(
        position_vector=(dx, dy),
        yaw=yaw,
        target_marker_id=chosen_id,
        target_found=True,
        visible_marker_ids=visible_ids,
        marker_positions=marker_positions,
    )


class AutoFollowScenario:
    """Continuous auto-follow scenario.

    Wraps a :class:`~robo_eye_sense.detector.RoboEyeDetector` and a
    :class:`~robo_eye_sense.camera.Camera` to continuously compute the
    follow vector for a selected AprilTag marker.

    Parameters
    ----------
    camera:
        An opened :class:`~robo_eye_sense.camera.Camera` instance.
    detector:
        A configured :class:`~robo_eye_sense.detector.RoboEyeDetector`
        instance (AprilTag detection must be enabled).
    frame_width:
        Width of the captured frame in pixels.
    frame_height:
        Height of the captured frame in pixels.
    target_marker_id:
        Identifier of the marker to follow (``None`` to auto-select).
    """

    def __init__(
        self,
        camera: object,
        detector: object,
        frame_width: int = 640,
        frame_height: int = 480,
        target_marker_id: Optional[str] = None,
    ) -> None:
        self.camera = camera
        self.detector = detector
        self.frame_width = frame_width
        self.frame_height = frame_height
        self._target_marker_id = target_marker_id

    @property
    def target_marker_id(self) -> Optional[str]:
        """Identifier of the marker to follow (``None`` = auto)."""
        return self._target_marker_id

    @target_marker_id.setter
    def target_marker_id(self, value: Optional[str]) -> None:
        self._target_marker_id = value

    def compute_from_detections(
        self, detections: List[Detection]
    ) -> AutoFollowResult:
        """Compute the follow vector from already-obtained detections.

        This avoids an extra camera capture and is useful in the GUI
        loop where detections are already available.
        """
        return compute_follow_vector(
            detections,
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            target_marker_id=self._target_marker_id,
        )

    def compute_current(self) -> AutoFollowResult:
        """Capture a frame and compute the follow vector.

        Returns
        -------
        AutoFollowResult
            Follow vector for the current frame.

        Raises
        ------
        RuntimeError
            If the camera returns no frame.
        """
        frame = self.camera.read()  # type: ignore[attr-defined]
        if frame is None:
            raise RuntimeError("Camera returned no frame.")
        detections = self.detector.process_frame(frame)  # type: ignore[attr-defined]
        return self.compute_from_detections(detections)
