"""Operational modes for robo-vision.

Each mode implements a common :class:`BaseMode` interface whose
:meth:`~BaseMode.run` method processes a single video frame and returns
the (possibly annotated) frame for display.
"""

from __future__ import annotations

from .base import BaseMode
from .box_mode import BoxMode
from .calibration_mode import CalibrationMode
from .follow_mode import FollowMode
from .live_mode import LiveMapMode, LiveMode
from .pose_mode import PoseMode

__all__ = [
    "BaseMode",
    "BoxMode",
    "CalibrationMode",
    "FollowMode",
    "LiveMapMode",
    "LiveMode",
    "PoseMode",
]
