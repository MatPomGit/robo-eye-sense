"""MediaPipe pose-landmarker mode – detect human body landmarks in real-time.

Uses MediaPipe's PoseLandmarker to detect up to *num_poses* sets of 33 body
key-points and draws a colour-coded skeleton overlay on each video frame.

The mode requires the ``mediapipe`` package (≥ 0.10) and a
``pose_landmarker_lite.task`` model bundle.  On first use the model is
looked for in ``~/.cache/robo-vision/``; if absent the mode attempts to
download it from Google's model zoo (~5 MB).  Provide an explicit path via
the *model_path* constructor argument to skip the auto-download logic.
"""

from __future__ import annotations

import logging
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .base import BaseMode

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/"
    "pose_landmarker_lite.task"
)
_MODEL_FILENAME = "pose_landmarker_lite.task"
_CACHE_DIR = Path.home() / ".cache" / "robo-vision"

# Landmark index → (R, G, B) colour used when drawing joints
_JOINT_COLOR = (0, 255, 128)   # green
_BONE_COLOR = (255, 200, 0)    # cyan-yellow
_JOINT_RADIUS = 4
_BONE_THICKNESS = 2

# Visibility / presence threshold for drawing a landmark
_MIN_VISIBILITY = 0.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_model_path() -> Path:
    """Return the cached model path, downloading the file if necessary."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model_path = _CACHE_DIR / _MODEL_FILENAME
    if not model_path.exists():
        logger.info(
            "Downloading MediaPipe pose model to %s …", model_path
        )
        try:
            urllib.request.urlretrieve(_MODEL_URL, str(model_path))
            logger.info("Model downloaded successfully.")
        except Exception as exc:
            logger.error("Failed to download model: %s", exc)
            raise FileNotFoundError(
                f"MediaPipe model not found at {model_path} and could not be "
                f"downloaded automatically.\n"
                f"Download it manually from:\n  {_MODEL_URL}\n"
                f"and place it at:\n  {model_path}"
            ) from exc
    return model_path


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class PoseLandmark:
    """Single 2-D body landmark with visibility score."""

    x: float        # normalised x coordinate (0–1)
    y: float        # normalised y coordinate (0–1)
    z: float        # depth estimate (relative)
    visibility: float


@dataclass
class PoseDetection:
    """A single detected human pose with all 33 landmarks."""

    landmarks: List[PoseLandmark] = field(default_factory=list)

    @property
    def num_landmarks(self) -> int:
        return len(self.landmarks)


# ---------------------------------------------------------------------------
# Mode
# ---------------------------------------------------------------------------


class MediaPipeMode(BaseMode):
    """Detect human body pose landmarks using MediaPipe PoseLandmarker.

    Parameters
    ----------
    model_path:
        Path to the ``pose_landmarker_*.task`` model bundle.
        When *None* the mode looks for the file in
        ``~/.cache/robo-vision/`` and attempts to download it on first use.
    num_poses:
        Maximum number of poses to detect per frame (default: 3).
    min_pose_detection_confidence:
        Minimum confidence score for pose detection (0–1, default: 0.5).
    min_tracking_confidence:
        Minimum confidence score for pose tracking (0–1, default: 0.5).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        num_poses: int = 3,
        min_pose_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._model_path = model_path
        self._num_poses = max(1, num_poses)
        self._min_detection_conf = float(min_pose_detection_confidence)
        self._min_tracking_conf = float(min_tracking_confidence)
        self._landmarker: Optional[Any] = None
        self._detections: List[PoseDetection] = []
        self._connections: Optional[List[Any]] = None
        self._init_error: Optional[str] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def detections(self) -> List[PoseDetection]:
        """Poses detected in the most recent frame."""
        return list(self._detections)

    @property
    def is_ready(self) -> bool:
        """True once the landmarker is initialised and ready."""
        return self._landmarker is not None

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _ensure_initialized(self) -> bool:
        """Import mediapipe and create the landmarker on first use.

        Returns True on success, False when mediapipe is unavailable or
        the model cannot be loaded.
        """
        if self._landmarker is not None:
            return True
        if self._init_error is not None:
            return False

        try:
            import mediapipe as mp  # noqa: PLC0415
        except ImportError:
            self._init_error = (
                "mediapipe package not installed.\n"
                "Install it with:  pip install mediapipe"
            )
            logger.error(self._init_error)
            return False

        try:
            if self._model_path is None:
                resolved_path = str(_default_model_path())
            else:
                resolved_path = self._model_path

            options = mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=resolved_path
                ),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_poses=self._num_poses,
                min_pose_detection_confidence=self._min_detection_conf,
                min_pose_presence_confidence=self._min_detection_conf,
                min_tracking_confidence=self._min_tracking_conf,
            )
            self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(
                options
            )
            self._connections = (
                mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS
            )
            logger.info("MediaPipe PoseLandmarker initialised.")
        except Exception as exc:
            self._init_error = f"Failed to initialise MediaPipe: {exc}"
            logger.error(self._init_error)
            return False

        return True

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _pixel(
        self, lm: PoseLandmark, width: int, height: int
    ) -> Tuple[int, int]:
        """Convert normalised coordinates to pixel coordinates."""
        return (int(lm.x * width), int(lm.y * height))

    def _draw_skeleton(
        self,
        vis: np.ndarray,
        detection: PoseDetection,
    ) -> None:
        """Draw joints and bones for one detected pose onto *vis*."""
        h, w = vis.shape[:2]
        landmarks = detection.landmarks

        # Draw bones (connections)
        if self._connections is not None:
            for conn in self._connections:
                start_idx = conn.start
                end_idx = conn.end
                if start_idx >= len(landmarks) or end_idx >= len(landmarks):
                    continue
                a = landmarks[start_idx]
                b = landmarks[end_idx]
                if (
                    a.visibility >= _MIN_VISIBILITY
                    and b.visibility >= _MIN_VISIBILITY
                ):
                    pa = self._pixel(a, w, h)
                    pb = self._pixel(b, w, h)
                    cv2.line(vis, pa, pb, _BONE_COLOR, _BONE_THICKNESS, cv2.LINE_AA)

        # Draw joints
        for lm in landmarks:
            if lm.visibility >= _MIN_VISIBILITY:
                px = self._pixel(lm, w, h)
                cv2.circle(vis, px, _JOINT_RADIUS, _JOINT_COLOR, -1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # BaseMode.run
    # ------------------------------------------------------------------

    def run(self, frame: np.ndarray, context: Dict[str, Any]) -> np.ndarray:
        """Detect human body poses and annotate *frame*."""
        if not self._ensure_initialized():
            vis = frame.copy()
            error_msg = self._init_error or "MediaPipe not available"
            for i, line in enumerate(error_msg.split("\n")):
                cv2.putText(
                    vis, line,
                    (8, 24 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA,
                )
            return vis

        try:
            import mediapipe as mp  # noqa: PLC0415

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB, data=rgb
            )
            result = self._landmarker.detect(mp_image)
        except Exception as exc:
            logger.warning("MediaPipe detection error: %s", exc)
            self._detections = []
            return frame.copy()

        # Convert result to PoseDetection objects
        poses: List[PoseDetection] = []
        for pose_lms in (result.pose_landmarks or []):
            landmarks = [
                PoseLandmark(
                    x=lm.x,
                    y=lm.y,
                    z=lm.z,
                    visibility=getattr(lm, "visibility", 1.0),
                )
                for lm in pose_lms
            ]
            poses.append(PoseDetection(landmarks=landmarks))

        self._detections = poses

        # --- Visualisation ---
        vis = frame.copy()
        for detection in poses:
            self._draw_skeleton(vis, detection)

        fps_display = context.get("fps", 0.0)
        cv2.putText(
            vis,
            f"Poses: {len(poses)}  FPS: {fps_display:.1f}",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )

        headless = context.get("headless", False)
        if headless:
            frame_idx = context.get("frame_idx", 0)
            print(
                f"[frame {frame_idx}] "
                f"poses_detected={len(poses)}"
            )

        return vis
