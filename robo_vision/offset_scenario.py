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

The module also estimates the distance from the camera to each AprilTag
using the known physical tag size (default 5 cm) and a pinhole-camera
approximation.

The module is intentionally decoupled from the camera and detector
instantiation so that the core logic is easy to unit-test with synthetic
data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .results import Detection, DetectionType

# Known physical side-length of the AprilTag markers in centimetres.
TAG_PHYSICAL_SIZE_CM: float = 5.0

# Default horizontal field-of-view (degrees) assumed for the camera when
# no calibration data is available.  60° is a reasonable approximation for
# most consumer webcams.
_DEFAULT_HFOV_DEG: float = 60.0


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
    per_tag_distances_cm:
        Estimated distance (in centimetres) from the camera to each
        currently visible AprilTag, computed from apparent pixel size
        and the known physical tag size.
    distance_to_reference_cm:
        Estimated distance (cm) from the current camera position to the
        reference position.  Computed as the magnitude of the offset
        scaled by the average pixels-per-cm ratio of the visible tags.
        ``None`` when no distance can be estimated (no matched tags with
        known corners).
    """

    offset: Tuple[float, float] = (0.0, 0.0)
    matched_tags: int = 0
    per_tag_offsets: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    reference_positions: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    current_positions: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    per_tag_distances_cm: Dict[str, float] = field(default_factory=dict)
    distance_to_reference_cm: Optional[float] = None


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


def _apriltag_corners(
    detections: List[Detection],
) -> Dict[str, List[Tuple[int, int]]]:
    """Extract ``{tag_id: corners}`` from *detections* (AprilTags only)."""
    result: Dict[str, List[Tuple[int, int]]] = {}
    for d in detections:
        if d.detection_type == DetectionType.APRIL_TAG and d.identifier is not None:
            result[d.identifier] = list(d.corners)
    return result


def estimate_focal_length_px(frame_width: int, hfov_deg: float = _DEFAULT_HFOV_DEG) -> float:
    """Estimate the horizontal focal length in pixels.

    Uses the pinhole camera model: ``f = (w / 2) / tan(hfov / 2)``.

    Parameters
    ----------
    frame_width:
        Width of the captured image in pixels.
    hfov_deg:
        Horizontal field-of-view of the camera in degrees.  Defaults to
        60.0°, a reasonable approximation for most consumer webcams.

    Returns
    -------
    float
        Estimated focal length in pixels.
    """
    half_fov_rad = math.radians(hfov_deg / 2.0)
    return (frame_width / 2.0) / math.tan(half_fov_rad)


def _tag_apparent_size_px(corners: List[Tuple[int, int]]) -> float:
    """Return the average side-length (in pixels) of a quadrilateral.

    *corners* must have at least 4 points.  Returns ``0.0`` if fewer than
    4 corners are provided.
    """
    if len(corners) < 4:
        return 0.0
    total = 0.0
    for i in range(4):
        x0, y0 = corners[i]
        x1, y1 = corners[(i + 1) % 4]
        total += math.hypot(x1 - x0, y1 - y0)
    return total / 4.0


def estimate_tag_distance_cm(
    corners: List[Tuple[int, int]],
    focal_length_px: float,
    tag_size_cm: float = TAG_PHYSICAL_SIZE_CM,
) -> Optional[float]:
    """Estimate distance from the camera to an AprilTag.

    Uses the pinhole-camera approximation:
    ``distance = (real_size * focal_length) / apparent_pixel_size``.

    Parameters
    ----------
    corners:
        Ordered corner points of the tag.
    focal_length_px:
        Camera focal length in pixels (see :func:`estimate_focal_length_px`).
    tag_size_cm:
        Physical side-length of the tag in centimetres.

    Returns
    -------
    float or None
        Estimated distance in centimetres, or ``None`` if the apparent
        size cannot be determined (fewer than 4 corners, or zero size).
    """
    apparent = _tag_apparent_size_px(corners)
    if apparent <= 0.0:
        return None
    return (tag_size_cm * focal_length_px) / apparent


def compute_offset(
    reference_detections: List[Detection],
    current_detections: List[Detection],
    frame_width: int = 640,
    hfov_deg: float = _DEFAULT_HFOV_DEG,
    tag_size_cm: float = TAG_PHYSICAL_SIZE_CM,
) -> OffsetResult:
    """Compute the camera offset between *reference* and *current* frames.

    Parameters
    ----------
    reference_detections:
        Detections from the reference frame (camera at desired position).
    current_detections:
        Detections from the current frame (camera after being moved).
    frame_width:
        Width of the captured frame in pixels (used for focal-length
        estimation when computing distances).
    hfov_deg:
        Horizontal field-of-view of the camera in degrees.
    tag_size_cm:
        Physical side-length of each AprilTag in centimetres.

    Returns
    -------
    OffsetResult
        Aggregated offset, per-tag details, and distance estimates.
        When no common tags exist the offset is ``(0.0, 0.0)`` and
        ``matched_tags`` is ``0``.
    """
    ref_pos = _apriltag_positions(reference_detections)
    cur_pos = _apriltag_positions(current_detections)
    cur_corners = _apriltag_corners(current_detections)

    focal = estimate_focal_length_px(frame_width, hfov_deg)

    # Per-tag distance estimation for all currently visible tags
    per_tag_dist: Dict[str, float] = {}
    for tag_id, corners in cur_corners.items():
        dist = estimate_tag_distance_cm(corners, focal, tag_size_cm)
        if dist is not None:
            per_tag_dist[tag_id] = dist

    common_ids = sorted(set(ref_pos) & set(cur_pos))

    if not common_ids:
        return OffsetResult(
            reference_positions=ref_pos,
            current_positions=cur_pos,
            per_tag_distances_cm=per_tag_dist,
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
    avg_offset = (sum_dx / n, sum_dy / n)

    # Estimate the real-world distance-to-reference from the pixel offset.
    # We convert the pixel displacement to centimetres using the average
    # apparent-size-to-real-size ratio of the matched tags.
    distance_to_ref: Optional[float] = None
    if per_tag_dist:
        matched_dists = [per_tag_dist[t] for t in common_ids if t in per_tag_dist]
        matched_sizes = [
            _tag_apparent_size_px(cur_corners[t])
            for t in common_ids
            if t in cur_corners and _tag_apparent_size_px(cur_corners[t]) > 0
        ]
        if matched_dists and matched_sizes:
            # Average cm-per-pixel ratio across matched tags: how many
            # real-world centimetres each pixel represents at the tags'
            # average distance from the camera.
            avg_cm_per_px = sum(
                tag_size_cm / s for s in matched_sizes
            ) / len(matched_sizes)
            px_magnitude = math.hypot(avg_offset[0], avg_offset[1])
            distance_to_ref = px_magnitude * avg_cm_per_px

    return OffsetResult(
        offset=avg_offset,
        matched_tags=n,
        per_tag_offsets=per_tag,
        reference_positions=ref_pos,
        current_positions=cur_pos,
        per_tag_distances_cm=per_tag_dist,
        distance_to_reference_cm=distance_to_ref,
    )


class CameraOffsetScenario:
    """Interactive camera-offset calibration scenario.

    Wraps a :class:`~robo_vision.detector.RoboEyeDetector` and a
    :class:`~robo_vision.camera.Camera` to guide the user through
    reference capture → camera movement → offset computation.

    Parameters
    ----------
    camera:
        An opened :class:`~robo_vision.camera.Camera` instance.
    detector:
        A configured :class:`~robo_vision.detector.RoboEyeDetector`
        instance (AprilTag detection must be enabled).
    frame_width:
        Width of the captured frame in pixels (for focal-length estimation).
    tag_size_cm:
        Physical side-length of the AprilTag markers in centimetres.
    """

    def __init__(
        self,
        camera: object,
        detector: object,
        frame_width: int = 640,
        tag_size_cm: float = TAG_PHYSICAL_SIZE_CM,
    ) -> None:
        self.camera = camera
        self.detector = detector
        self.frame_width = frame_width
        self.tag_size_cm = tag_size_cm
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

    def set_reference(self, detections: List[Detection]) -> None:
        """Set the reference detections directly (e.g. from the GUI loop).

        Parameters
        ----------
        detections:
            Detections to use as the reference frame.
        """
        self._reference_detections = list(detections)

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
        return compute_offset(
            self._reference_detections,
            current_detections,
            frame_width=self.frame_width,
            tag_size_cm=self.tag_size_cm,
        )

    def compute_offset_from_detections(
        self, current_detections: List[Detection]
    ) -> OffsetResult:
        """Compute the offset using already-obtained *current_detections*.

        This avoids an extra camera capture and is useful in the GUI loop
        where detections are already available from the frame-update cycle.

        Raises
        ------
        RuntimeError
            If no reference has been captured yet.
        """
        if self._reference_detections is None:
            raise RuntimeError(
                "No reference frame captured. Call capture_reference() first."
            )
        return compute_offset(
            self._reference_detections,
            current_detections,
            frame_width=self.frame_width,
            tag_size_cm=self.tag_size_cm,
        )

    def reset(self) -> None:
        """Clear the stored reference so a new one can be captured."""
        self._reference_detections = None
