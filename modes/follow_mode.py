"""Follow mode – actively track an object and generate control signals.

Replaces the legacy *auto* mode with a richer tracking strategy:

1. If AprilTags are visible → track the selected tag (or the first one).
2. Else if ``--follow-box`` was given → fall back to box tracking.
3. Otherwise → idle (zero output).

A simple proportional (P) controller computes ``linear`` and ``angular``
velocity commands from the tracking error.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from robo_vision import april_tag_detector as _april_runtime

from .base import BaseMode
from .box_mode import BoxMode

logger = logging.getLogger(__name__)

# Default P-controller gains
_KX = 0.002  # angular gain (rad/px)
_KZ = 1.0    # linear gain


@dataclass
class FollowResult:
    """Result of a single follow computation.

    Attributes
    ----------
    mode_label:
        ``"tag"``, ``"box"``, or ``"idle"``.
    target_id:
        The tag ID or box index being tracked, or ``None``.
    error_x:
        Horizontal pixel error (positive = target is right of centre).
    error_z:
        Distance error in metres (positive = too far).
    linear:
        Linear velocity command.
    angular:
        Angular velocity command.
    """

    mode_label: str = "idle"
    target_id: Optional[str] = None
    error_x: float = 0.0
    error_z: float = 0.0
    linear: float = 0.0
    angular: float = 0.0


class FollowMode(BaseMode):
    """Intelligent follow mode with AprilTag → box fallback.

    Parameters
    ----------
    follow_marker:
        Specific AprilTag ID to follow.  ``None`` → first visible.
    follow_box:
        Enable box-tracking fallback when no tags are visible.
    target_distance:
        Desired distance to the target in metres.
    tag_size:
        Physical tag side length in metres (for distance estimation).
    calibration_path:
        Path to calibration ``.npz`` for accurate distance.
    """

    def __init__(
        self,
        follow_marker: Optional[str] = None,
        follow_box: bool = False,
        target_distance: float = 0.5,
        tag_size: float = 0.05,
        calibration_path: Optional[str] = None,
    ) -> None:
        self._follow_marker = follow_marker
        self._follow_box = follow_box
        self._target_distance = target_distance
        self._tag_size = tag_size

        # Box-detection sub-mode (lazy)
        self._box_mode: Optional[BoxMode] = BoxMode() if follow_box else None

        # Camera calibration (optional)
        self._camera_matrix: Optional[np.ndarray] = None
        self._dist_coeffs: Optional[np.ndarray] = None
        if calibration_path is not None:
            try:
                data = np.load(calibration_path)
                self._camera_matrix = data["camera_matrix"]
                self._dist_coeffs = data["dist_coeffs"]
            except (FileNotFoundError, KeyError) as exc:
                logger.warning("Could not load calibration: %s", exc)

        # AprilTag detector (lazy)
        self._detector: Any = None
        self._last_result = FollowResult()

        # 3-D object points of the tag
        half = tag_size / 2.0
        self._obj_pts = np.array([
            [-half, -half, 0],
            [ half, -half, 0],
            [ half,  half, 0],
            [-half,  half, 0],
        ], dtype=np.float64)

    # ------------------------------------------------------------------
    @property
    def last_result(self) -> FollowResult:
        """Most recent follow result."""
        return self._last_result

    # ------------------------------------------------------------------
    def _ensure_detector(self) -> None:
        if self._detector is not None:
            return
        if not _april_runtime._apriltags_available():
            self._detector = None
            return
        try:
            from pupil_apriltags import Detector  # type: ignore[import-untyped]
            self._detector = _april_runtime.retain_detector_reference(Detector(families="tag36h11"))
        except ImportError:
            try:
                import apriltag  # type: ignore[import-untyped]
                self._detector = apriltag.Detector()
            except ImportError:
                self._detector = None

    # ------------------------------------------------------------------
    def _default_camera_matrix(self, w: int, h: int) -> np.ndarray:
        fx = w / (2.0 * np.tan(np.radians(30)))
        return np.array([
            [fx, 0, w / 2.0],
            [0, fx, h / 2.0],
            [0, 0, 1],
        ], dtype=np.float64)

    # ------------------------------------------------------------------
    def run(self, frame: np.ndarray, context: Dict[str, Any]) -> np.ndarray:
        """Process frame: track tag or box, compute control output."""
        h, w = frame.shape[:2]
        frame_cx = w / 2.0
        frame_cy = h / 2.0
        vis = frame.copy()

        result = FollowResult()

        # --- 1. Try AprilTag tracking ---
        self._ensure_detector()
        tag_result = self._try_tag_tracking(frame, frame_cx, frame_cy)
        if tag_result is not None:
            result = tag_result
        elif self._follow_box and self._box_mode is not None:
            # --- 2. Fallback to box tracking ---
            box_result = self._try_box_tracking(frame, context, frame_cx)
            if box_result is not None:
                result = box_result

        self._last_result = result

        # --- Visualisation ---
        # Draw centre crosshair
        cv2.line(vis, (int(frame_cx), 0), (int(frame_cx), h), (128, 128, 128), 1)
        cv2.line(vis, (0, int(frame_cy)), (w, int(frame_cy)), (128, 128, 128), 1)

        info_lines = [
            f"mode: follow",
            f"target: {result.mode_label}"
            + (f" ({result.target_id})" if result.target_id else ""),
            f"error_x: {result.error_x:+.1f}",
            f"error_z: {result.error_z:+.3f}",
            f"linear: {result.linear:+.3f}",
            f"angular: {result.angular:+.4f}",
        ]
        for i, line in enumerate(info_lines):
            cv2.putText(
                vis, line, (8, 24 + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA,
            )

        headless = context.get("headless", False)
        if headless:
            frame_idx = context.get("frame_idx", 0)
            target_str = f"{result.mode_label}"
            if result.target_id:
                target_str += f"({result.target_id})"
            print(
                f"[frame {frame_idx}] "
                f"mode: follow  "
                f"target: {target_str}  "
                f"error_x: {result.error_x:+.1f}  "
                f"error_z: {result.error_z:+.3f}  "
                f"linear: {result.linear:+.3f}  "
                f"angular: {result.angular:+.4f}"
            )

        return vis

    # ------------------------------------------------------------------
    def _try_tag_tracking(
        self,
        frame: np.ndarray,
        frame_cx: float,
        frame_cy: float,
    ) -> Optional[FollowResult]:
        """Attempt to track an AprilTag.  Return ``None`` on failure."""
        if self._detector is None:
            return None

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        try:
            tags = self._detector.detect(gray)
        except Exception:
            logger.exception("Tag detection failed")
            return None

        if not tags:
            return None

        # Select target tag
        chosen = None
        for t in tags:
            tid = str(getattr(t, "tag_id", ""))
            if self._follow_marker is not None and tid == self._follow_marker:
                chosen = t
                break
        if chosen is None:
            chosen = tags[0]

        corners = getattr(chosen, "corners", None)
        if corners is None:
            return None

        corners_2d = np.array(corners, dtype=np.float64).reshape(-1, 2)
        tag_id = str(getattr(chosen, "tag_id", "?"))
        cx = float(corners_2d[:, 0].mean())
        cy = float(corners_2d[:, 1].mean())

        # Pose-based distance
        cam_mtx = (
            self._camera_matrix
            if self._camera_matrix is not None
            else self._default_camera_matrix(w, h)
        )
        dist_coeffs = (
            self._dist_coeffs
            if self._dist_coeffs is not None
            else np.zeros(5, dtype=np.float64)
        )
        current_distance = self._target_distance  # fallback
        success, _rvec, tvec = cv2.solvePnP(
            self._obj_pts, corners_2d, cam_mtx, dist_coeffs,
        )
        if success:
            current_distance = float(tvec[2][0])

        error_x = cx - frame_cx
        error_z = self._target_distance - current_distance
        angular = _KX * error_x
        linear = _KZ * error_z

        return FollowResult(
            mode_label="tag",
            target_id=tag_id,
            error_x=error_x,
            error_z=error_z,
            linear=linear,
            angular=angular,
        )

    # ------------------------------------------------------------------
    def _try_box_tracking(
        self,
        frame: np.ndarray,
        context: Dict[str, Any],
        frame_cx: float,
    ) -> Optional[FollowResult]:
        """Attempt to track the largest box.  Return ``None`` if none found."""
        if self._box_mode is None:
            return None
        self._box_mode.run(frame, context)
        boxes = self._box_mode.detections
        if not boxes:
            return None

        # Select largest box
        largest = max(boxes, key=lambda b: b.area)
        bx, by = largest.center
        error_x = bx - frame_cx
        angular = _KX * error_x

        return FollowResult(
            mode_label="box",
            target_id=None,
            error_x=error_x,
            error_z=0.0,
            linear=0.0,
            angular=angular,
        )
