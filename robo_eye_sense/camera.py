"""Camera capture wrapper around ``cv2.VideoCapture``.

Provides a context-manager interface for safe resource management and
convenience properties for querying and adjusting capture parameters.
"""

from __future__ import annotations

from typing import Optional, Union

import cv2
import numpy as np


class Camera:
    """Thin wrapper around ``cv2.VideoCapture``.

    Parameters
    ----------
    source:
        Camera index (int), path to a video file (str), or an RTSP/HTTP
        stream URL (str).
    width:
        Requested frame width in pixels.  The camera may not honour this
        exactly; check :attr:`actual_width` after opening.
    height:
        Requested frame height in pixels.
    fps:
        Requested frames-per-second.

    Raises
    ------
    RuntimeError
        If the capture device cannot be opened.
    """

    def __init__(
        self,
        source: Union[int, str] = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ) -> None:
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Failed to open camera/video source: {source!r}"
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    def read(self) -> Optional[np.ndarray]:
        """Grab and return the next frame, or ``None`` if the stream ended."""
        ret, frame = self._cap.read()
        if not ret:
            return None
        return frame

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def actual_width(self) -> int:
        """Actual frame width reported by the capture device."""
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def actual_height(self) -> int:
        """Actual frame height reported by the capture device."""
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def actual_fps(self) -> float:
        """Actual FPS reported by the capture device."""
        return self._cap.get(cv2.CAP_PROP_FPS)

    @property
    def is_opened(self) -> bool:
        """``True`` if the capture device is currently open."""
        return self._cap.isOpened()

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def release(self) -> None:
        """Release the underlying ``VideoCapture`` object."""
        self._cap.release()

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, *args: object) -> None:
        self.release()
