"""robo-eye-sense – lightweight real-time visual marker detection.

Primary public surface
----------------------
* :class:`~robo_eye_sense.detector.RoboEyeDetector` – all-in-one detector/tracker
* :class:`~robo_eye_sense.results.Detection` – per-detection data class
* :class:`~robo_eye_sense.results.DetectionType` – detection category enum
* :class:`~robo_eye_sense.camera.Camera` – camera capture helper (OpenCV-dependent)

Notes
-----
``RoboEyeDetector`` and :class:`~robo_eye_sense.camera.Camera` depend on OpenCV.
``RoboEyeDetector`` is imported lazily, which keeps lightweight modules
(e.g. :mod:`robo_eye_sense.results`) usable even when OpenCV is not
available yet.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .results import Detection, DetectionType

if TYPE_CHECKING:
    from .detector import RoboEyeDetector

__all__ = ["RoboEyeDetector", "Detection", "DetectionType"]
__version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    """Lazy-load heavy module attributes.

    This avoids importing :mod:`cv2` from :mod:`robo_eye_sense.detector`
    when callers only need data models from :mod:`robo_eye_sense.results`.
    """
    if name == "RoboEyeDetector":
        from .detector import RoboEyeDetector

        return RoboEyeDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
