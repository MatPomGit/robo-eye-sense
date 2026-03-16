"""QR-code detector.

Uses *pyzbar* when available (more robust, requires the ``libzbar0`` system
library), otherwise falls back to OpenCV's built-in
``cv2.QRCodeDetector``.  Both backends produce identical
:class:`~robo_eye_sense.results.Detection` objects and support detecting
multiple QR codes in a single frame.
"""

from __future__ import annotations

import importlib
from typing import List, Optional

import cv2
import numpy as np

from .results import Detection, DetectionType


def _pyzbar_available() -> bool:
    return importlib.util.find_spec("pyzbar") is not None


class QRCodeDetector:
    """Detect and decode QR codes in a colour frame.

    The detector automatically selects the best available backend:

    * **pyzbar** (preferred) – wraps the battle-tested ZBar library.
      Requires the ``pyzbar`` Python package and ``libzbar0`` on the system.
    * **OpenCV** (fallback) – uses ``cv2.QRCodeDetector``, always available
      with a standard OpenCV installation.

    Parameters
    ----------
    force_backend:
        ``"pyzbar"`` or ``"opencv"`` to force a specific backend.
        ``None`` (default) picks pyzbar if available.

    Raises
    ------
    ValueError
        If *force_backend* is ``"pyzbar"`` but pyzbar is not installed.
    """

    def __init__(self, force_backend: Optional[str] = None) -> None:
        if force_backend == "pyzbar":
            if not _pyzbar_available():
                raise ValueError(
                    "pyzbar is not installed. "
                    "Install it with:  pip install pyzbar"
                )
            self._backend = "pyzbar"
        elif force_backend == "opencv":
            self._backend = "opencv"
        else:
            self._backend = "pyzbar" if _pyzbar_available() else "opencv"

        if self._backend == "opencv":
            self._cv_detector = cv2.QRCodeDetector()

    @property
    def backend(self) -> str:
        """Name of the active detection backend (``'pyzbar'`` or ``'opencv'``)."""
        return self._backend

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Return QR-code detections found in *frame*.

        Parameters
        ----------
        frame:
            BGR image as a NumPy array (H × W × 3, dtype uint8).

        Returns
        -------
        List[Detection]
            One entry per detected QR code.  ``identifier`` is the decoded
            text payload; ``corners`` are the four polygon vertices.
        """
        if self._backend == "pyzbar":
            try:
                return self._detect_pyzbar(frame)
            except ImportError:
                self._backend = "opencv"
                self._cv_detector = cv2.QRCodeDetector()
        return self._detect_opencv(frame)

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _detect_pyzbar(self, frame: np.ndarray) -> List[Detection]:
        from pyzbar import pyzbar  # type: ignore[import]

        decoded_objects = pyzbar.decode(frame)
        detections: List[Detection] = []
        for obj in decoded_objects:
            if obj.type != "QRCODE":
                continue
            polygon = obj.polygon
            if not polygon:
                continue
            corners = [(p.x, p.y) for p in polygon]
            cx = sum(p[0] for p in corners) // len(corners)
            cy = sum(p[1] for p in corners) // len(corners)
            try:
                payload = obj.data.decode("utf-8")
            except UnicodeDecodeError:
                payload = obj.data.decode("latin-1")
            detections.append(
                Detection(
                    detection_type=DetectionType.QR_CODE,
                    identifier=payload,
                    center=(cx, cy),
                    corners=corners,
                )
            )
        return detections

    def _detect_opencv(self, frame: np.ndarray) -> List[Detection]:
        detections: List[Detection] = []
        retval, decoded_info, points, _ = self._cv_detector.detectAndDecodeMulti(
            frame
        )
        if not retval or points is None:
            return detections

        for data, quad in zip(decoded_info, points):
            if not data:
                continue
            corners = [(int(round(p[0])), int(round(p[1]))) for p in quad]
            cx = sum(p[0] for p in corners) // len(corners)
            cy = sum(p[1] for p in corners) // len(corners)
            detections.append(
                Detection(
                    detection_type=DetectionType.QR_CODE,
                    identifier=data,
                    center=(cx, cy),
                    corners=corners,
                )
            )
        return detections
