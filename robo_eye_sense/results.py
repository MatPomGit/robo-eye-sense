"""Data structures shared across all detector and tracker modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


class DetectionType(Enum):
    """Category of a detected object."""

    APRIL_TAG = "april_tag"
    QR_CODE = "qr_code"
    LASER_SPOT = "laser_spot"


@dataclass
class Detection:
    """A single detected object within one video frame.

    Attributes
    ----------
    detection_type:
        Category of the detected object.
    identifier:
        Human-readable string identifier, e.g. the AprilTag numeric ID
        or the text payload of a QR code.  ``None`` for laser spots.
    center:
        ``(x, y)`` pixel coordinates of the object centre.
    corners:
        Ordered corner points of the bounding polygon (may be empty for
        laser spots, which use the bounding-rect corners).
    track_id:
        Persistent ID assigned by the :class:`~robo_eye_sense.tracker.CentroidTracker`.
        ``None`` until the detection has been through a tracker update.
    confidence:
        Detection confidence score in the range ``[0.0, 1.0]``.
    """

    detection_type: DetectionType
    identifier: Optional[str]
    center: Tuple[int, int]
    corners: List[Tuple[int, int]] = field(default_factory=list)
    track_id: Optional[int] = None
    confidence: float = 1.0

    def __repr__(self) -> str:  # pragma: no cover
        id_str = f"id={self.identifier!r} " if self.identifier else ""
        track_str = f"track={self.track_id} " if self.track_id is not None else ""
        return (
            f"Detection({self.detection_type.value} "
            f"{id_str}{track_str}"
            f"center={self.center})"
        )
