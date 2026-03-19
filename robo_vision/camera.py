"""Camera capture wrapper around ``cv2.VideoCapture``.

Provides a context-manager interface for safe resource management and
convenience properties for querying and adjusting capture parameters.
Supports integer camera indices, local video files, and network stream
URLs (RTSP, HTTP).
"""

from __future__ import annotations

from typing import Dict, Optional, Union

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
            ("brightness", cv2.CAP_PROP_BRIGHTNESS),
            ("contrast", cv2.CAP_PROP_CONTRAST),
            ("saturation", cv2.CAP_PROP_SATURATION),
            ("exposure", cv2.CAP_PROP_EXPOSURE),
            ("gain", cv2.CAP_PROP_GAIN),
        ]
        for name, prop_id in _optional:
            val = self._cap.get(prop_id)
            if val != 0.0:
                info[name] = val

        fourcc_code = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        if fourcc_code != 0:
            fourcc_str = "".join(
                chr((fourcc_code >> (8 * i)) & 0xFF) for i in range(4)
            )
            info["fourcc"] = fourcc_str

        return info

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
