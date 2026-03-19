"""Video recording wrapper around ``cv2.VideoWriter``.

Provides a simple start/stop interface for writing annotated (or raw) frames
to an MP4 file.  Works in GUI, headless, and plain ``cv2.imshow`` modes.

Usage::

    from robo_vision.recorder import VideoRecorder

    rec = VideoRecorder("output.mp4", width=640, height=480, fps=30.0)
    rec.start()
    for frame in frames:
        rec.write_frame(frame)
    rec.stop()

Or as a context manager::

    with VideoRecorder("output.mp4", width=640, height=480) as rec:
        for frame in frames:
            rec.write_frame(frame)
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


class VideoRecorder:
    """Record frames to an MP4 video file.

    Parameters
    ----------
    output_path:
        Destination file path (e.g. ``"recording.mp4"``).
    width:
        Frame width in pixels.
    height:
        Frame height in pixels.
    fps:
        Frames-per-second for the output video.
    fourcc:
        Four-character codec code passed to ``cv2.VideoWriter_fourcc``.
        Defaults to ``"mp4v"`` which produces ``.mp4`` files.
    """

    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float = 30.0,
        fourcc: str = "mp4v",
    ) -> None:
        self._output_path = output_path
        self._width = width
        self._height = height
        self._fps = fps
        self._fourcc = fourcc
        self._writer: Optional[cv2.VideoWriter] = None
        self._recording = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_recording(self) -> bool:
        """``True`` while recording is in progress."""
        return self._recording

    @property
    def output_path(self) -> str:
        """Destination file path."""
        return self._output_path

    # ------------------------------------------------------------------
    # Recording control
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the video writer and begin recording.

        Raises
        ------
        RuntimeError
            If the ``cv2.VideoWriter`` cannot be opened (e.g. invalid path
            or unsupported codec).
        """
        if self._recording:
            return
        fourcc = cv2.VideoWriter_fourcc(*self._fourcc)
        self._writer = cv2.VideoWriter(
            self._output_path, fourcc, self._fps,
            (self._width, self._height),
        )
        if not self._writer.isOpened():
            self._writer = None
            raise RuntimeError(
                f"Failed to open VideoWriter for {self._output_path!r}"
            )
        self._recording = True

    def stop(self) -> None:
        """Finish recording and release the video writer."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        self._recording = False

    def write_frame(self, frame: np.ndarray) -> None:
        """Write a single BGR frame to the video file.

        Frames whose dimensions do not match the writer's
        ``(width, height)`` are resized automatically.

        Does nothing when recording is not active.
        """
        if not self._recording or self._writer is None:
            return
        h, w = frame.shape[:2]
        if (w, h) != (self._width, self._height):
            frame = cv2.resize(frame, (self._width, self._height))
        self._writer.write(frame)

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "VideoRecorder":
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()
