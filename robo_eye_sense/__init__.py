"""robo-eye-sense – lightweight real-time visual marker detection.

Primary public surface
----------------------
* :class:`~robo_eye_sense.detector.RoboEyeDetector` – all-in-one detector/tracker
* :class:`~robo_eye_sense.results.Detection` – per-detection data class
* :class:`~robo_eye_sense.results.DetectionType` – detection category enum
* :class:`~robo_eye_sense.camera.Camera` – camera capture helper
"""

from .detector import RoboEyeDetector
from .results import Detection, DetectionType

__all__ = ["RoboEyeDetector", "Detection", "DetectionType"]
__version__ = "0.1.0"
