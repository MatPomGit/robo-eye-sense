"""robo-vision – lightweight real-time visual marker detection.

Primary public surface
----------------------
* :class:`~robo_vision.detector.RoboEyeDetector` – all-in-one detector/tracker
* :class:`~robo_vision.results.Detection` – per-detection data class
* :class:`~robo_vision.results.DetectionType` – detection-category enum
  (``APRIL_TAG``, ``QR_CODE``, ``LASER_SPOT``)
* :class:`~robo_vision.results.DetectionMode` – pipeline operating mode
  (``NORMAL``, ``FAST``, ``ROBUST``)
* :class:`~robo_vision.camera.Camera` – camera capture helper (OpenCV-dependent)

Notes
-----
``RoboEyeDetector`` and :class:`~robo_vision.camera.Camera` depend on OpenCV.
``RoboEyeDetector`` is imported lazily via :pep:`562` (module ``__getattr__``),
which keeps lightweight modules (e.g. :mod:`robo_vision.results`) usable
even when OpenCV is not installed or fails to initialise.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from .base_detector import BaseDetector
from .marker_map import MarkerMap, MarkerPose3D, RobotPose3D
from .results import Detection, DetectionMode, DetectionType


def _fix_qt_font_dir() -> None:
    """Point Qt at system fonts so OpenCV's Qt back-end doesn't warn.

    When ``opencv-python`` is built with Qt support, Qt may emit::

        QFontDatabase: Cannot find font directory …/cv2/qt/fonts

    Setting :envvar:`QT_QPA_FONTDIR` to a valid system font directory
    before ``cv2`` is imported silences the warning.
    """
    if "QT_QPA_FONTDIR" in os.environ:
        return
    for candidate in ("/usr/share/fonts", "/usr/local/share/fonts"):
        if os.path.isdir(candidate):
            os.environ["QT_QPA_FONTDIR"] = candidate
            return


def _fix_qt_platform() -> None:
    """Force xcb QPA when running in a Wayland session.

    ``opencv-python`` bundles its own Qt plug-in directory which typically
    ships only the ``xcb`` plugin.  On a Wayland desktop, Qt picks up the
    ``WAYLAND_DISPLAY`` environment variable and tries the wayland plugin
    first, printing::

        qt.qpa.plugin: Could not find the Qt platform plugin "wayland"

    Setting :envvar:`QT_QPA_PLATFORM` to ``xcb`` before ``cv2`` initialises
    the Qt back-end prevents the lookup and silences the warning.
    """
    if "QT_QPA_PLATFORM" in os.environ:
        return
    if "WAYLAND_DISPLAY" in os.environ:
        os.environ["QT_QPA_PLATFORM"] = "xcb"


_fix_qt_font_dir()
_fix_qt_platform()

if TYPE_CHECKING:
    from .auto_scenario import AutoFollowResult, AutoFollowScenario
    from .detector import RoboEyeDetector
    from .marker_map import SlamCalibrator
    from .ros2_bridge import ROS2Bridge

__all__ = [
    "APP_NAME",
    "AutoFollowResult",
    "AutoFollowScenario",
    "BaseDetector",
    "MarkerMap",
    "MarkerPose3D",
    "RobotPose3D",
    "ROS2Bridge",
    "RoboEyeDetector",
    "SlamCalibrator",
    "Detection",
    "DetectionMode",
    "DetectionType",
    "classify_tag",
    "load_tag_names_from_file",
    "get_device_status",
    "get_calibration_info",
    "print_headless_guide",
    "load_config",
    "__version__",
]
APP_NAME = "robo-vision"
__version__ = "0.5.0"


def __getattr__(name: str) -> Any:
    """Lazy-load heavy module attributes.

    This avoids importing :mod:`cv2` from :mod:`robo_vision.detector`
    when callers only need data models from :mod:`robo_vision.results`.
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

    if name == "SlamCalibrator":
        from .marker_map import SlamCalibrator

        return SlamCalibrator

    if name in ("AutoFollowResult", "AutoFollowScenario"):
        from .auto_scenario import AutoFollowResult, AutoFollowScenario

        if name == "AutoFollowResult":
            return AutoFollowResult
        return AutoFollowScenario

    if name in (
        "classify_tag",
        "load_tag_names_from_file",
        "get_device_status",
        "get_calibration_info",
        "print_headless_guide",
    ):
        from . import headless_guide as _hg

        return getattr(_hg, name)

    if name == "ROS2Bridge":
        from .ros2_bridge import ROS2Bridge

        return ROS2Bridge

    if name == "load_config":
        from .config import load_config

        return load_config

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
