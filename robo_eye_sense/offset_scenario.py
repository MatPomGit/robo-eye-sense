"""Camera-offset calibration scenario.

Determines the pixel-space correction vector (offset) between a *reference*
camera position and its current position by comparing the positions of
AprilTag fiducial markers visible in both frames.

Workflow
--------
1. Capture a **reference frame** while the camera is at the desired position.
   AprilTags placed on the table (or other fixture) are detected and their
   pixel centres are recorded.
2. Move the camera to a new position.
3. Capture a **current frame** and detect the same AprilTags.  Because the
   camera has moved, the tags appear at different pixel coordinates.
4. For every tag visible in *both* frames the displacement vector
   ``(dx, dy)`` is computed.  The **offset** returned is the average of
   these displacement vectors – i.e. the pixel-space correction that must
   be applied to the current camera position to return it to the reference
   position.

The module is intentionally decoupled from the camera and detector
instantiation so that the core logic is easy to unit-test with synthetic
data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .results import Detection, DetectionType


@dataclass
class OffsetResult:
    """Result of the camera-offset computation.

    Attributes
    ----------
    offset:
        Average displacement ``(dx, dy)`` in pixels from the current tag
        positions back to the reference positions.  Positive ``dx`` means
        the camera must move *right* (in pixel space) to return to the
        reference position; positive ``dy`` means it must move *down*.
        ``(0.0, 0.0)`` when no common tags are found.
    matched_tags:
        Number of AprilTags that were visible in both the reference and
        the current frame and therefore contributed to the offset.
    per_tag_offsets:
        Mapping from tag identifier (string) to its individual
        ``(dx, dy)`` displacement vector.
    reference_positions:
        Mapping from tag identifier to its ``(x, y)`` centre in the
        reference frame.
    current_positions:
        Mapping from tag identifier to its ``(x, y)`` centre in the
        current frame.
    """

    offset: Tuple[float, float] = (0.0, 0.0)
    matched_tags: int = 0
    per_tag_offsets: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    reference_positions: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    current_positions: Dict[str, Tuple[int, int]] = field(default_factory=dict)


def _apriltag_positions(detections: List[Detection]) -> Dict[str, Tuple[int, int]]:
    """Extract a mapping ``{tag_id: centre}`` from a list of detections.

    Only :attr:`DetectionType.APRIL_TAG` entries with a non-``None``
    identifier are included.  If the same tag ID appears more than once
    (e.g. duplicate detection), the last occurrence wins.
    """
    positions: Dict[str, Tuple[int, int]] = {}
    for d in detections:
        if d.detection_type == DetectionType.APRIL_TAG and d.identifier is not None:
            positions[d.identifier] = d.center
    return positions


def compute_offset(
    reference_detections: List[Detection],
    current_detections: List[Detection],
) -> OffsetResult:
    """Compute the camera offset between *reference* and *current* frames.

    Parameters
    ----------
    reference_detections:
        Detections from the reference frame (camera at desired position).
    current_detections:
        Detections from the current frame (camera after being moved).

    Returns
    -------
    OffsetResult
        Aggregated offset and per-tag details.  When no common tags exist
        the offset is ``(0.0, 0.0)`` and ``matched_tags`` is ``0``.
    """
    ref_pos = _apriltag_positions(reference_detections)
    cur_pos = _apriltag_positions(current_detections)

    common_ids = sorted(set(ref_pos) & set(cur_pos))

    if not common_ids:
        return OffsetResult(
            reference_positions=ref_pos,
            current_positions=cur_pos,
        )

    per_tag: Dict[str, Tuple[float, float]] = {}
    sum_dx = 0.0
    sum_dy = 0.0
    for tag_id in common_ids:
        rx, ry = ref_pos[tag_id]
        cx, cy = cur_pos[tag_id]
        dx = float(rx - cx)
        dy = float(ry - cy)
        per_tag[tag_id] = (dx, dy)
        sum_dx += dx
        sum_dy += dy

    n = len(common_ids)
    return OffsetResult(
        offset=(sum_dx / n, sum_dy / n),
        matched_tags=n,
        per_tag_offsets=per_tag,
        reference_positions=ref_pos,
        current_positions=cur_pos,
    )


class CameraOffsetScenario:
    """Interactive camera-offset calibration scenario.

    Wraps a :class:`~robo_eye_sense.detector.RoboEyeDetector` and a
    :class:`~robo_eye_sense.camera.Camera` to guide the user through
    reference capture → camera movement → offset computation.

    Parameters
    ----------
    camera:
        An opened :class:`~robo_eye_sense.camera.Camera` instance.
    detector:
        A configured :class:`~robo_eye_sense.detector.RoboEyeDetector`
        instance (AprilTag detection must be enabled).
    """

    def __init__(self, camera: object, detector: object) -> None:
        self.camera = camera
        self.detector = detector
        self._reference_detections: Optional[List[Detection]] = None

    @property
    def has_reference(self) -> bool:
        """``True`` if a reference frame has already been captured."""
        return self._reference_detections is not None

    @property
    def reference_detections(self) -> Optional[List[Detection]]:
        """Detections stored from the reference frame, or ``None``."""
        return self._reference_detections

    def capture_reference(self) -> List[Detection]:
        """Capture a frame from the camera and store its detections as the reference.

        Returns
        -------
        List[Detection]
            Detections found in the reference frame.

        Raises
        ------
        RuntimeError
            If the camera returns no frame.
        """
        frame = self.camera.read()  # type: ignore[attr-defined]
        if frame is None:
            raise RuntimeError("Camera returned no frame for reference capture.")
        detections = self.detector.process_frame(frame)  # type: ignore[attr-defined]
        self._reference_detections = detections
        return detections

    def compute_current_offset(self) -> OffsetResult:
        """Capture a frame and compute the offset relative to the stored reference.

        Returns
        -------
        OffsetResult
            Offset between the reference and current frames.

        Raises
        ------
        RuntimeError
            If no reference has been captured yet, or the camera returns
            no frame.
        """
        if self._reference_detections is None:
            raise RuntimeError(
                "No reference frame captured. Call capture_reference() first."
            )
        frame = self.camera.read()  # type: ignore[attr-defined]
        if frame is None:
            raise RuntimeError("Camera returned no frame for current capture.")
        current_detections = self.detector.process_frame(frame)  # type: ignore[attr-defined]
        return compute_offset(self._reference_detections, current_detections)

    def reset(self) -> None:
        """Clear the stored reference so a new one can be captured."""
        self._reference_detections = None
