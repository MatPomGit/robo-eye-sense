"""AprilTag detector wrapping the *pupil-apriltags* library.

pupil-apriltags ships pre-compiled binaries for x86_64 and ARM, making it
suitable for both development machines and embedded robot hardware (e.g.
Raspberry Pi).

All four standard tag families are detected simultaneously:
``tag36h11``, ``tag25h9``, ``tag16h5``, ``tag12h10``.

Detection is intentionally single-threaded (``nthreads=1``) to keep CPU usage
predictable on resource-constrained platforms.
"""

from __future__ import annotations

import importlib
import importlib.util
import warnings
from typing import List, Optional

import numpy as np

from .base_detector import BaseDetector
from .results import Detection, DetectionType

_PUPIL_APRILTAGS_AVAILABLE: Optional[bool] = None
_DETECTOR_REFS: list[object] = []


def _apriltags_available() -> bool:
    global _PUPIL_APRILTAGS_AVAILABLE
    if _PUPIL_APRILTAGS_AVAILABLE is None:
        _PUPIL_APRILTAGS_AVAILABLE = (
            importlib.util.find_spec("pupil_apriltags") is not None
        )
    return _PUPIL_APRILTAGS_AVAILABLE


_ALL_FAMILIES = "tag36h11 tag25h9 tag16h5 tag12h10"


def retain_detector_reference(detector: object) -> object:
    """Keep a native AprilTag detector alive for the process lifetime."""
    _DETECTOR_REFS.append(detector)
    return detector


class AprilTagDetector(BaseDetector):
    """Detect and decode AprilTag fiducial markers in a grayscale frame.

    All four standard tag families (tag36h11, tag25h9, tag16h5, tag12h10)
    are detected simultaneously.

    Parameters
    ----------
    nthreads:
        Number of threads used internally by the detector.  Keep at ``1``
        on embedded platforms to avoid unpredictable CPU spikes.
    min_decision_margin:
        Minimum *decision_margin* reported by pupil-apriltags for a
        detection to be accepted.  Low-margin detections are typically
        caused by image noise or accidental patterns and should be
        discarded to avoid phantom tracks.  The default of ``25.0`` is a
        good starting point; increase for stricter filtering.

    Raises
    ------
    ImportError
        If *pupil-apriltags* is not installed.
    """

    def __init__(
        self,
        nthreads: int = 1,
        min_decision_margin: float = 25.0,
    ) -> None:
        if not _apriltags_available():
            raise ImportError(
                "pupil-apriltags is required for AprilTag detection. "
                "Install it with:  pip install pupil-apriltags"
            )
        import pupil_apriltags as apriltag  # type: ignore[import]

        self._min_decision_margin = float(min_decision_margin)
        self._detector = apriltag.Detector(
            families=_ALL_FAMILIES,
            nthreads=nthreads,
            quad_decimate=2.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )

    def get_name(self) -> str:
        """Return the detector name."""
        return "AprilTag"

    def detect(self, gray_frame: np.ndarray) -> List[Detection]:
        """Return AprilTag detections found in *gray_frame*.

        Parameters
        ----------
        gray_frame:
            Single-channel (grayscale) uint8 image.

        Returns
        -------
        List[Detection]
            One entry per detected tag.  ``identifier`` is the tag ID as a
            string; ``corners`` follow the convention of pupil-apriltags
            (bottom-left, bottom-right, top-right, top-left).
        """
        results = self._detector.detect(gray_frame)
        detections: List[Detection] = []
        for r in results:
            margin = getattr(r, "decision_margin", 0.0)
            if margin < self._min_decision_margin:
                continue
            center = (int(round(r.center[0])), int(round(r.center[1])))
            corners = [
                (int(round(c[0])), int(round(c[1]))) for c in r.corners
            ]
            detections.append(
                Detection(
                    detection_type=DetectionType.APRIL_TAG,
                    identifier=str(r.tag_id),
                    center=center,
                    corners=corners,
                    confidence=float(margin),
                )
            )
        return detections
