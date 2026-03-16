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

__all__ = ["RoboEyeDetector", "Detection", "DetectionType", "APP_NAME"]
__version__ = "0.2.0"
APP_NAME = "RoboEyeSense"


def __getattr__(name: str) -> Any:
    """Lazy-load heavy module attributes.

    This avoids importing :mod:`cv2` from :mod:`robo_eye_sense.detector`
    when callers only need data models from :mod:`robo_eye_sense.results`.
    """
    if name == "RoboEyeDetector":
        try:
            from .detector import RoboEyeDetector
        except ImportError as exc:
            raise ImportError(
                "RoboEyeDetector could not be imported because its OpenCV (cv2) "
                "dependency is missing or failed to initialize. Ensure that "
                "OpenCV is installed and usable in this environment."
            ) from exc

        return RoboEyeDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
