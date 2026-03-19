"""On-screen information overlay for video frames.

Draws real-time information (timestamp, mode, status, keyboard shortcuts)
on the video frame without obstructing the main image area.
"""

from __future__ import annotations

import datetime
import time
from typing import Dict, List, Optional

import cv2
import numpy as np

from .results import Detection, DetectionMode

# Overlay layout constants
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE_SMALL = 0.40
_FONT_SCALE_NORMAL = 0.45
_FONT_THICKNESS = 1
_LINE_HEIGHT = 16
_MARGIN = 8
_BG_ALPHA = 0.5

# How long the keyboard legend is shown after startup (seconds)
_LEGEND_DISPLAY_DURATION = 10.0


class OverlayRenderer:
    """Draws informational overlay on video frames.

    Parameters
    ----------
    enabled:
        Whether the overlay is active.  When ``False`` all draw methods
        are no-ops.
    mode:
        Current operating mode name (e.g. ``"basic"``, ``"slam"``).
    quality:
        Current quality setting (e.g. ``"low"``, ``"normal"``, ``"high"``).
    enabled_detectors:
        List of enabled detector names for status messages.
    """

    def __init__(
        self,
        enabled: bool = True,
        mode: str = "basic",
        quality: str = "normal",
        enabled_detectors: Optional[List[str]] = None,
    ) -> None:
        self._enabled = enabled
        self._mode = mode
        self._quality = quality
        self._enabled_detectors = enabled_detectors or []
        self._start_time = time.monotonic()
        self._recording = False
        self._recording_start: Optional[float] = None

    @property
    def enabled(self) -> bool:
        """Whether the overlay is active."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    def set_recording(self, active: bool) -> None:
        """Update the recording state for the REC indicator."""
        if active and not self._recording:
            self._recording_start = time.monotonic()
        self._recording = active

    def draw(
        self,
        frame: np.ndarray,
        detections: Optional[List[Detection]] = None,
        fps: float = 0.0,
        extra_status: str = "",
    ) -> np.ndarray:
        """Draw the full overlay onto *frame*.

        Parameters
        ----------
        frame:
            BGR image to annotate (modified in-place).
        detections:
            Current detections (used for status messages).
        fps:
            Current frames-per-second.
        extra_status:
            Additional status text (e.g. SLAM info).

        Returns
        -------
        np.ndarray
            The annotated frame.
        """
        if not self._enabled:
            return frame

        h, w = frame.shape[:2]

        # ── Top-left: timestamp + FPS ────────────────────────────────
        now = datetime.datetime.now()
        timestamp = now.strftime("%H:%M:%S.") + f"{now.microsecond // 1000:03d}"
        self._draw_text(
            frame, f"{timestamp}  FPS: {fps:.1f}",
            (_MARGIN, _MARGIN + 14), color=(255, 255, 255),
        )

        # ── Top-right: mode + quality ────────────────────────────────
        mode_text = f"Mode: {self._mode}  Quality: {self._quality}"
        (tw, _), _ = cv2.getTextSize(mode_text, _FONT, _FONT_SCALE_NORMAL, _FONT_THICKNESS)
        self._draw_text(
            frame, mode_text,
            (w - tw - _MARGIN, _MARGIN + 14), color=(200, 200, 200),
        )

        # ── Status line (below top-left) ─────────────────────────────
        status = self._build_status(detections, extra_status)
        if status:
            self._draw_text(
                frame, status,
                (_MARGIN, _MARGIN + 32), color=(180, 220, 255),
                scale=_FONT_SCALE_SMALL,
            )

        # ── REC indicator (top-center) ───────────────────────────────
        if self._recording:
            self._draw_rec_indicator(frame, w)

        # ── Keyboard legend (bottom-left, shown briefly after start) ──
        elapsed = time.monotonic() - self._start_time
        if elapsed < _LEGEND_DISPLAY_DURATION:
            self._draw_legend(frame, h)

        return frame

    def _build_status(
        self,
        detections: Optional[List[Detection]],
        extra: str,
    ) -> str:
        """Build a one-line status message."""
        parts: list[str] = []

        if self._enabled_detectors:
            parts.append("Looking for: " + ", ".join(self._enabled_detectors))

        if detections is not None:
            if not detections:
                parts.append("Waiting for marker...")
            else:
                parts.append(f"{len(detections)} detection(s)")

        if extra:
            parts.append(extra)

        return "  |  ".join(parts)

    def _draw_rec_indicator(self, frame: np.ndarray, frame_w: int) -> None:
        """Draw a red REC indicator with elapsed time at top-center."""
        elapsed = 0.0
        if self._recording_start is not None:
            elapsed = time.monotonic() - self._recording_start

        mins, secs = divmod(int(elapsed), 60)
        hours, mins = divmod(mins, 60)
        time_str = f"{hours:02d}:{mins:02d}:{secs:02d}"
        rec_text = f"REC {time_str}"

        (tw, th), _ = cv2.getTextSize(rec_text, _FONT, _FONT_SCALE_NORMAL, _FONT_THICKNESS)
        x = (frame_w - tw) // 2
        y = _MARGIN + 14

        # Red dot
        cv2.circle(frame, (x - 10, y - 4), 5, (0, 0, 255), -1)
        # REC text
        cv2.putText(
            frame, rec_text, (x, y),
            _FONT, _FONT_SCALE_NORMAL, (0, 0, 255),
            _FONT_THICKNESS, cv2.LINE_AA,
        )

    def _draw_legend(self, frame: np.ndarray, frame_h: int) -> None:
        """Draw keyboard shortcut legend at the bottom-left."""
        lines = ["Q: quit | R: record toggle"]
        y = frame_h - _MARGIN - len(lines) * _LINE_HEIGHT
        for line in lines:
            self._draw_text(
                frame, line, (_MARGIN, y),
                color=(160, 160, 160), scale=_FONT_SCALE_SMALL,
            )
            y += _LINE_HEIGHT

    @staticmethod
    def _draw_text(
        frame: np.ndarray,
        text: str,
        position: tuple[int, int],
        color: tuple[int, int, int] = (255, 255, 255),
        scale: float = _FONT_SCALE_NORMAL,
    ) -> None:
        """Draw text with a dark outline for readability."""
        x, y = position
        # Dark outline
        cv2.putText(
            frame, text, (x, y), _FONT, scale,
            (0, 0, 0), _FONT_THICKNESS + 1, cv2.LINE_AA,
        )
        # Foreground text
        cv2.putText(
            frame, text, (x, y), _FONT, scale,
            color, _FONT_THICKNESS, cv2.LINE_AA,
        )
