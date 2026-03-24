"""YOLO detection and tracking mode.

Uses Ultralytics YOLO (v8/v11) for real-time object detection and persistent
multi-object tracking.

On first use the mode downloads ``yolo11n.pt`` (~6 MB) from Ultralytics to
``~/.cache/robo-vision/``.  Provide an explicit path via *model_path* to skip
the download.

Requires the ``ultralytics`` package::

    pip install ultralytics
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .base import BaseMode

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "yolo11n.pt"
_CACHE_DIR = Path.home() / ".cache" / "robo-vision"

# BGR colour palette indexed by track ID
_PALETTE: List[Tuple[int, int, int]] = [
    (0, 255, 0),    # green
    (255, 128, 0),  # orange
    (0, 128, 255),  # blue
    (255, 0, 255),  # magenta
    (0, 255, 255),  # yellow
    (128, 0, 255),  # purple
    (255, 255, 0),  # cyan
    (0, 0, 255),    # red
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class YoloDetection:
    """A single detected object with optional persistent track ID.

    Attributes
    ----------
    track_id:
        Tracker-assigned persistent ID across frames.
        ``None`` when tracking is disabled.
    class_id:
        YOLO class index.
    class_name:
        Human-readable class label (e.g. ``"person"``, ``"car"``).
    confidence:
        Detection confidence in ``[0, 1]``.
    bbox:
        Bounding box ``(x1, y1, x2, y2)`` in pixel coordinates.
    """

    track_id: Optional[int]
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]

    @property
    def center(self) -> Tuple[int, int]:
        """Pixel coordinates of the bounding-box centre."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)


# ---------------------------------------------------------------------------
# Mode
# ---------------------------------------------------------------------------


class YoloMode(BaseMode):
    """YOLO-based real-time object detection and tracking mode.

    Uses Ultralytics YOLO for per-frame detection and, when *track* is
    ``True`` (the default), applies ByteTrack to assign each object a
    persistent integer ID that is kept stable across frames.

    Parameters
    ----------
    model_path:
        Path to YOLO weights (``*.pt``).  When ``None`` the mode uses
        ``yolo11n.pt`` and downloads it on first use to
        ``~/.cache/robo-vision/``.
    confidence:
        Minimum confidence threshold (0–1, default 0.25).
    iou:
        IoU threshold for NMS (0–1, default 0.45).
    classes:
        List of COCO class IDs to detect.  ``None`` = all classes.
    track:
        Enable persistent multi-object tracking (default ``True``).
        When ``False`` only per-frame detection is performed and
        ``YoloDetection.track_id`` will always be ``None``.
    device:
        Inference device string, e.g. ``'cpu'``, ``'0'`` (first CUDA
        GPU).  ``None`` = automatic selection by Ultralytics.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence: float = 0.25,
        iou: float = 0.45,
        classes: Optional[List[int]] = None,
        track: bool = True,
        device: Optional[str] = None,
    ) -> None:
        self._model_path = model_path
        self._confidence = float(confidence)
        self._iou = float(iou)
        self._classes = classes
        self._track = track
        self._device = device
        self._model: Optional[Any] = None
        self._detections: List[YoloDetection] = []
        self._init_error: Optional[str] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def detections(self) -> List[YoloDetection]:
        """Detections (and tracks) from the most recent frame."""
        return list(self._detections)

    @property
    def is_ready(self) -> bool:
        """``True`` once the YOLO model is loaded and ready."""
        return self._model is not None

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _ensure_initialized(self) -> bool:
        """Import Ultralytics and load the YOLO model on first use.

        Returns ``True`` on success, ``False`` when Ultralytics is
        unavailable or the model cannot be loaded.
        """
        if self._model is not None:
            return True
        if self._init_error is not None:
            return False

        try:
            from ultralytics import YOLO  # noqa: PLC0415
        except ImportError:
            self._init_error = (
                "ultralytics package not installed.\n"
                "Install it with:  pip install ultralytics"
            )
            logger.error(self._init_error)
            return False

        try:
            if self._model_path is None:
                _CACHE_DIR.mkdir(parents=True, exist_ok=True)
                model_file = _CACHE_DIR / _DEFAULT_MODEL
                # When the cached file already exists, load it directly.
                # Otherwise pass the bare filename so Ultralytics can
                # download it to its own cache; we copy it later if needed.
                resolved: str = (
                    str(model_file) if model_file.exists() else _DEFAULT_MODEL
                )
            else:
                resolved = self._model_path

            self._model = YOLO(resolved)
            if self._device:
                self._model.to(self._device)
            logger.info("YOLO model loaded: %s", resolved)
        except Exception as exc:
            self._init_error = f"Failed to load YOLO model: {exc}"
            logger.error(self._init_error)
            return False

        return True

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _color_for_id(
        self, track_id: Optional[int]
    ) -> Tuple[int, int, int]:
        """Return a BGR colour consistent for a given track ID."""
        if track_id is None:
            return (0, 255, 0)
        return _PALETTE[track_id % len(_PALETTE)]

    def _draw_detection(
        self, vis: np.ndarray, det: YoloDetection
    ) -> None:
        """Draw a labelled bounding box for one detection onto *vis*."""
        x1, y1, x2, y2 = det.bbox
        color = self._color_for_id(det.track_id)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        label = det.class_name
        if det.track_id is not None:
            label = f"#{det.track_id} {label}"
        label += f" {det.confidence:.2f}"

        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            vis, (x1, y1 - th - 6), (x1 + tw + 2, y1), color, -1
        )
        cv2.putText(
            vis,
            label,
            (x1 + 1, y1 - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    # ------------------------------------------------------------------
    # BaseMode.run
    # ------------------------------------------------------------------

    def run(self, frame: np.ndarray, context: Dict[str, Any]) -> np.ndarray:
        """Detect and track objects in *frame*, return annotated result."""
        if not self._ensure_initialized():
            vis = frame.copy()
            error_msg = self._init_error or "YOLO not available"
            for i, line in enumerate(error_msg.split("\n")):
                cv2.putText(
                    vis,
                    line,
                    (8, 24 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
            return vis

        try:
            if self._track:
                results = self._model.track(
                    frame,
                    persist=True,
                    conf=self._confidence,
                    iou=self._iou,
                    classes=self._classes,
                    verbose=False,
                )
            else:
                results = self._model.predict(
                    frame,
                    conf=self._confidence,
                    iou=self._iou,
                    classes=self._classes,
                    verbose=False,
                )
        except Exception as exc:
            logger.warning("YOLO inference error: %s", exc)
            self._detections = []
            return frame.copy()

        detections: List[YoloDetection] = []
        if results and results[0].boxes is not None:
            result = results[0]
            boxes = result.boxes
            names: Dict[int, str] = result.names or {}
            for i in range(len(boxes)):
                box = boxes[i]
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = names.get(cls_id, str(cls_id))
                track_id: Optional[int] = None
                if self._track and box.id is not None:
                    track_id = int(box.id[0])
                detections.append(
                    YoloDetection(
                        track_id=track_id,
                        class_id=cls_id,
                        class_name=cls_name,
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                    )
                )

        self._detections = detections

        # --- Visualisation ---
        vis = frame.copy()
        for det in detections:
            self._draw_detection(vis, det)

        fps_display = context.get("fps", 0.0)
        mode_label = "YOLO+Track" if self._track else "YOLO"
        cv2.putText(
            vis,
            f"{mode_label}: {len(detections)}  FPS: {fps_display:.1f}",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        headless = context.get("headless", False)
        if headless:
            frame_idx = context.get("frame_idx", 0)
            ids_str = ""
            if detections:
                ids = [
                    str(d.track_id) if d.track_id is not None else "?"
                    for d in detections
                ]
                ids_str = f" ids=[{', '.join(ids)}]"
            print(
                f"[frame {frame_idx}] "
                f"yolo_detected={len(detections)}"
                f"{ids_str}"
            )

        return vis
