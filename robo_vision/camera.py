"""Camera capture wrapper around ``cv2.VideoCapture``.

Provides a context-manager interface for safe resource management and
convenience properties for querying and adjusting capture parameters.
Supports integer camera indices, local video files, and network stream
URLs (RTSP, HTTP).
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Optional, Union

import numpy as np

from ._cv2_compat import get_cv2

logger = logging.getLogger("robo_vision.camera")


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
    max_read_failures:
        Number of consecutive ``read()`` failures (``None`` frames) before
        an automatic reconnection attempt is triggered.  Set to ``0`` to
        disable reconnection.
    max_reconnect_attempts:
        Maximum number of reconnection attempts before giving up.

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
        max_read_failures: int = 5,
        max_reconnect_attempts: int = 5,
    ) -> None:
        self._source = source
        self._width = width
        self._height = height
        self._fps = fps
        self._max_read_failures = max_read_failures
        self._max_reconnect_attempts = max_reconnect_attempts
        self._consecutive_failures = 0

        cv2 = get_cv2()
        self._cv2 = cv2
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Failed to open camera/video source: {source!r}"
            )
        self._cap.set(self._cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(self._cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(self._cv2.CAP_PROP_FPS, fps)

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    def read(self) -> Optional[np.ndarray]:
        """Grab and return the next frame, or ``None`` if the stream ended.

        When the underlying capture device fails to deliver a frame for
        *max_read_failures* consecutive calls, the camera is automatically
        released and re-opened with exponential backoff (starting at 0.5 s,
        doubling up to 30 s).  If reconnection succeeds the failure counter
        is reset and the next frame is returned normally.  If all attempts
        are exhausted ``None`` is returned.
        """
        ret, frame = self._cap.read()
        if ret:
            self._consecutive_failures = 0
            return frame

        # Reconnection disabled or video file ended
        if self._max_read_failures <= 0:
            return None

        self._consecutive_failures += 1
        if self._consecutive_failures < self._max_read_failures:
            return None

        # Threshold reached – attempt reconnection
        logger.warning(
            "Camera source %r: %d consecutive read failures, "
            "attempting reconnection...",
            self._source, self._consecutive_failures,
        )
        if self._reconnect():
            self._consecutive_failures = 0
            ret, frame = self._cap.read()
            return frame if ret else None

        return None

    def _reconnect(self) -> bool:
        """Try to re-open the capture device with exponential backoff.

        Returns
        -------
        bool
            ``True`` if the device was successfully re-opened.
        """
        backoff = 0.5
        max_backoff = 30.0

        for attempt in range(1, self._max_reconnect_attempts + 1):
            logger.info(
                "Reconnection attempt %d/%d for source %r "
                "(waiting %.1f s)...",
                attempt, self._max_reconnect_attempts,
                self._source, backoff,
            )
            self._cap.release()
            time.sleep(backoff)

            self._cap = self._cv2.VideoCapture(self._source)
            if self._cap.isOpened():
                self._cap.set(self._cv2.CAP_PROP_FRAME_WIDTH, self._width)
                self._cap.set(self._cv2.CAP_PROP_FRAME_HEIGHT, self._height)
                self._cap.set(self._cv2.CAP_PROP_FPS, self._fps)
                logger.info(
                    "Reconnected to source %r on attempt %d.",
                    self._source, attempt,
                )
                return True

            backoff = min(backoff * 2, max_backoff)

        logger.error(
            "Failed to reconnect to source %r after %d attempts.",
            self._source, self._max_reconnect_attempts,
        )
        return False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def actual_width(self) -> int:
        """Actual frame width reported by the capture device."""
        return int(self._cap.get(self._cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def actual_height(self) -> int:
        """Actual frame height reported by the capture device."""
        return int(self._cap.get(self._cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def actual_fps(self) -> float:
        """Actual FPS reported by the capture device."""
        return self._cap.get(self._cv2.CAP_PROP_FPS)

    @property
    def backend_name(self) -> str:
        """Backend name reported by the capture device (e.g. ``V4L2``)."""
        return self._cap.getBackendName()

    @property
    def is_opened(self) -> bool:
        """``True`` if the capture device is currently open."""
        return self._cap.isOpened()

    def get_info(self) -> Dict[str, object]:
        """Return a dictionary of camera / capture-device parameters.

        Keys always present: ``width``, ``height``, ``fps``, ``backend``.
        Additional keys (``brightness``, ``contrast``, ``saturation``,
        ``exposure``, ``gain``, ``fourcc``) are included when the capture
        backend reports non-zero values.
        """
        info: Dict[str, object] = {
            "width": self.actual_width,
            "height": self.actual_height,
            "fps": self.actual_fps,
            "backend": self.backend_name,
        }

        # Optional properties – only include when the driver reports them.
        _optional: list[tuple[str, int]] = [
            ("brightness", self._cv2.CAP_PROP_BRIGHTNESS),
            ("contrast", self._cv2.CAP_PROP_CONTRAST),
            ("saturation", self._cv2.CAP_PROP_SATURATION),
            ("exposure", self._cv2.CAP_PROP_EXPOSURE),
            ("gain", self._cv2.CAP_PROP_GAIN),
        ]
        for name, prop_id in _optional:
            val = self._cap.get(prop_id)
            if val != 0.0:
                info[name] = val

        fourcc_code = int(self._cap.get(self._cv2.CAP_PROP_FOURCC))
        if fourcc_code != 0:
            fourcc_str = "".join(
                chr((fourcc_code >> (8 * i)) & 0xFF) for i in range(4)
            )
            info["fourcc"] = fourcc_str

        return info

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def set_capture_properties(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None,
    ) -> None:
        """Change capture width, height, and/or FPS at runtime.

        Updates the internal target values so that reconnection also uses
        the new settings.
        """
        if width is not None:
            self._width = width
            self._cap.set(self._cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            self._height = height
            self._cap.set(self._cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps is not None:
            self._fps = fps
            self._cap.set(self._cv2.CAP_PROP_FPS, fps)

    def release(self) -> None:
        """Release the underlying ``VideoCapture`` object."""
        self._cap.release()

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, *args: object) -> None:
        self.release()
