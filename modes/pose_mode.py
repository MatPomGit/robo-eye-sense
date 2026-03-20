"""Pose estimation mode – estimate 6-DoF pose of AprilTags.

Loads camera calibration from a ``.npz`` file, detects AprilTags using
``pupil_apriltags``, and estimates their pose with ``cv2.solvePnP`` followed
by ``cv2.solvePnPRefineLM`` for improved accuracy.
Visualises tag borders, IDs, and 3-D axes on the frame.

The :attr:`correction_vector` property exposes the horizontal angular error
and distance to the primary tag so the robot can compute approach corrections.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from robo_vision import april_tag_detector as _april_runtime

from .base import BaseMode

logger = logging.getLogger(__name__)

# All tag families supported by pupil_apriltags
_ALL_FAMILIES = "tag36h11 tag25h9 tag16h5 tag12h10"

# Sensitivity → detection parameter bounds
_SENSITIVITY_MAX_MARGIN: float = 50.0   # decision_margin threshold at sensitivity=0
_SENSITIVITY_MAX_FRAMES: int = 10       # consecutive frames required at sensitivity=0


def _sensitivity_params(sensitivity: int) -> Tuple[float, int]:
    """Convert *sensitivity* (0–100) to low-level detection thresholds.

    Parameters
    ----------
    sensitivity:
        Value in the range [0, 100].  Higher values make detection more
        permissive; lower values make it stricter.

    Returns
    -------
    (min_decision_margin, required_consecutive_frames)
        min_decision_margin:
            Minimum ``decision_margin`` a raw detection must have to be
            accepted.  0.0 at sensitivity=100, up to
            ``_SENSITIVITY_MAX_MARGIN`` at sensitivity=0.
        required_consecutive_frames:
            How many consecutive frames a tag must appear before it is
            reported as detected.  1 at sensitivity=100, up to
            ``_SENSITIVITY_MAX_FRAMES`` at sensitivity=0.
    """
    s = max(0, min(100, sensitivity)) / 100.0
    min_margin = (1.0 - s) * _SENSITIVITY_MAX_MARGIN
    req_frames = max(1, round(1 + (1.0 - s) * (_SENSITIVITY_MAX_FRAMES - 1)))
    return min_margin, req_frames


def _load_calibration(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load camera matrix and distortion coefficients from *path*.

    Returns
    -------
    (camera_matrix, dist_coeffs)

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    KeyError
        If required keys are missing from the file.
    """
    data = np.load(path)
    return data["camera_matrix"], data["dist_coeffs"]


def _get_tag_corners_3d(tag_size: float) -> np.ndarray:
    """Return the four 3-D corners of a tag centred at origin."""
    half = tag_size / 2.0
    return np.array([
        [-half, -half, 0],
        [ half, -half, 0],
        [ half,  half, 0],
        [-half,  half, 0],
    ], dtype=np.float64)


def _draw_pose_axes_fallback(
    img: np.ndarray,
    cam_mtx: np.ndarray,
    dist: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    length: float,
) -> None:
    """Fallback axis-drawing when ``cv2.drawFrameAxes`` is unavailable."""
    points_3d = np.float32([
        [0, 0, 0],
        [length, 0, 0],
        [0, length, 0],
        [0, 0, -length],  # Z points toward camera
    ])
    try:
        points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, cam_mtx, dist)
        pts = points_2d.astype(int).reshape(-1, 2)
        origin = tuple(pts[0])
        cv2.line(img, origin, tuple(pts[1]), (0, 0, 255), 2, cv2.LINE_AA)  # X red
        cv2.line(img, origin, tuple(pts[2]), (0, 255, 0), 2, cv2.LINE_AA)  # Y green
        cv2.line(img, origin, tuple(pts[3]), (255, 0, 0), 2, cv2.LINE_AA)  # Z blue
    except cv2.error:
        logger.debug("Axis projection failed")


class PoseMode(BaseMode):
    """AprilTag 6-DoF pose estimation mode.

    Parameters
    ----------
    tag_size:
        Physical size (side length) of the tags in metres.
    calibration_path:
        Path to the calibration ``.npz`` file.  When ``None`` a default
        camera matrix is synthesised from the frame dimensions.
    sensitivity:
        Detection sensitivity in the range [0, 100].

        * **High values** (close to 100) – even weak AprilTag signals are
          accepted as valid detections (low decision-margin threshold,
          qualification on the very first frame).
        * **Low values** (close to 0) – only strong signals qualify *and*
          the tag must remain visible for several consecutive frames before
          it is reported as detected.
    """

    def __init__(
        self,
        tag_size: float = 0.05,
        calibration_path: Optional[str] = None,
        sensitivity: int = 50,
    ) -> None:
        self._tag_size = tag_size
        self._obj_pts = _get_tag_corners_3d(tag_size)
        self._camera_matrix: Optional[np.ndarray] = None
        self._dist_coeffs: Optional[np.ndarray] = None

        # Sensitivity settings
        self._sensitivity = max(0, min(100, sensitivity))
        self._min_decision_margin, self._required_consecutive_frames = (
            _sensitivity_params(self._sensitivity)
        )
        # Per-tag consecutive detection counters  (tag_id → frame count)
        self._consecutive_counts: Dict[str, int] = {}
        # Last computed horizontal steering value (normalised to [-1, 1])
        self._steering_vector: float = 0.0
        # Correction vector: (horizontal_angle_deg, distance_m)
        # horizontal_angle_deg – positive = tag is to the right, robot should turn right
        # distance_m           – distance from camera to tag centre
        self._correction_vector: Tuple[float, float] = (0.0, 0.0)

        if calibration_path is not None:
            try:
                self._camera_matrix, self._dist_coeffs = _load_calibration(
                    calibration_path,
                )
                logger.info("Loaded calibration from %s", calibration_path)
            except (FileNotFoundError, KeyError) as exc:
                logger.warning(
                    "Could not load calibration from %s: %s – "
                    "falling back to default matrix.",
                    calibration_path, exc,
                )

        # Lazy-load the AprilTag detector
        self._detector: Any = None

    # ------------------------------------------------------------------
    @property
    def steering_vector(self) -> float:
        """Horizontal steering signal for robot control, normalised to [-1, 1].

        Derived from the position of the primary detected AprilTag relative
        to the horizontal centre of the frame:

        * **+1.0** – tag is at the far-right edge.
        * **0.0**  – tag is centred (no correction needed), or no tag detected.
        * **-1.0** – tag is at the far-left edge.

        The value is updated on every call to :meth:`run`.  When no qualified
        tag is present the property returns ``0.0``.
        """
        return self._steering_vector

    @property
    def correction_vector(self) -> Tuple[float, float]:
        """Correction vector for robot approach, derived from the refined pose.

        Returns a ``(horizontal_angle_deg, distance_m)`` tuple computed from
        the 3-D translation vector of the primary detected AprilTag:

        * **horizontal_angle_deg** – signed horizontal angle (degrees) between
          the camera optical axis and the tag centre.  Positive means the tag
          is to the right; negative means left.  The robot should rotate by
          this amount to centre the tag.
        * **distance_m** – Euclidean distance from the camera to the tag
          centre in metres.  Use this to decide when to stop approaching.

        Both values are ``0.0`` when no qualified tag is visible.
        """
        return self._correction_vector

    # ------------------------------------------------------------------
    def _ensure_detector(self) -> None:
        """Create the AprilTag detector on first use."""
        if self._detector is not None:
            return
        if not _april_runtime._apriltags_available():
            self._detector = None
            return
        try:
            from pupil_apriltags import Detector  # type: ignore[import-untyped]

            self._detector = _april_runtime.retain_detector_reference(Detector(
                families=_ALL_FAMILIES,
                nthreads=1,
                quad_decimate=2.0,
                quad_sigma=0.0,
                refine_edges=1,
                decode_sharpening=0.25,
                debug=0,
            ))
            logger.info("AprilTag detector initialised (%s)", _ALL_FAMILIES)
            return
        except Exception as exc:
            logger.warning("pupil_apriltags init failed: %s", exc)

        try:
            import apriltag  # type: ignore[import-untyped]

            self._detector = apriltag.Detector()
            logger.info("AprilTag detector initialised (apriltag fallback)")
        except Exception as exc:
            logger.error("AprilTag detector not available: %s", exc)
            self._detector = None

    # ------------------------------------------------------------------
    def _default_camera_matrix(self, w: int, h: int) -> np.ndarray:
        """Synthesise a camera matrix assuming ~60° HFOV."""
        fx = w / (2.0 * np.tan(np.radians(30)))
        fy = fx
        return np.array([
            [fx, 0, w / 2.0],
            [0, fy, h / 2.0],
            [0, 0, 1],
        ], dtype=np.float64)

    # ------------------------------------------------------------------
    def run(self, frame: np.ndarray, context: Dict[str, Any]) -> np.ndarray:
        """Detect AprilTags and estimate their poses.

        When ``context["april_detections"]`` is provided (e.g. from the GUI
        which already runs an AprilTag detector) those detections are reused,
        avoiding a redundant second detector instance.  Otherwise the mode
        initialises its own detector and runs detection internally.

        The :attr:`steering_vector` property is updated on every call with
        the horizontal offset of the first qualified tag, normalised to
        [-1, 1].
        """
        h, w = frame.shape[:2]
        vis = frame.copy()

        cam_mtx = (
            self._camera_matrix
            if self._camera_matrix is not None
            else self._default_camera_matrix(w, h)
        )
        dist = (
            self._dist_coeffs
            if self._dist_coeffs is not None
            else np.zeros(5, dtype=np.float64)
        )

        # Build a list of (corners_2d, tag_id) from either the provided
        # detections or from the internal detector.
        april_detections = context.get("april_detections")
        using_own_detector = april_detections is None

        tags_data: List[Tuple[np.ndarray, Any]] = []
        # Track which IDs were seen this frame (for consecutive-count reset)
        detected_ids: Set[str] = set()

        if using_own_detector:
            self._ensure_detector()
            if self._detector is None:
                cv2.putText(
                    vis,
                    "Pose: no AprilTag detector available",
                    (8, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA,
                )
                self._steering_vector = 0.0
                return vis

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            try:
                raw_tags = self._detector.detect(gray)
            except Exception:
                logger.exception("AprilTag detection failed")
                self._steering_vector = 0.0
                return vis

            for tag in raw_tags:
                corners = getattr(tag, "corners", None)
                if corners is None:
                    continue
                # Sensitivity: filter by decision_margin when available.
                # float("inf") is the safe default: a detector that does not
                # expose decision_margin is treated as maximally confident so
                # that the margin filter is always satisfied regardless of the
                # configured threshold.
                margin = getattr(tag, "decision_margin", float("inf"))
                if margin < self._min_decision_margin:
                    continue
                corners_2d = np.array(corners, dtype=np.float64).reshape(-1, 2)
                tag_id = str(getattr(tag, "tag_id", "?"))
                detected_ids.add(tag_id)
                self._consecutive_counts[tag_id] = (
                    self._consecutive_counts.get(tag_id, 0) + 1
                )
                if (
                    self._consecutive_counts[tag_id]
                    >= self._required_consecutive_frames
                ):
                    tags_data.append((corners_2d, tag_id))
        else:
            # Use detections provided by the GUI's main detector
            for det in april_detections:
                if not det.corners:
                    continue
                corners_2d = np.array(det.corners, dtype=np.float64).reshape(-1, 2)
                tag_id = str(det.identifier)
                detected_ids.add(tag_id)
                self._consecutive_counts[tag_id] = (
                    self._consecutive_counts.get(tag_id, 0) + 1
                )
                if (
                    self._consecutive_counts[tag_id]
                    >= self._required_consecutive_frames
                ):
                    tags_data.append((corners_2d, tag_id))

        # Reset counters for tags that were not detected this frame
        for tid in tuple(self._consecutive_counts):
            if tid not in detected_ids:
                self._consecutive_counts[tid] = 0

        # Compute horizontal steering from the first qualified tag (pixel-based)
        if tags_data:
            first_corners, _ = tags_data[0]
            tag_cx = float(first_corners[:, 0].mean())
            half_w = w / 2.0
            self._steering_vector = (tag_cx - half_w) / half_w if half_w > 0 else 0.0
        else:
            self._steering_vector = 0.0
            self._correction_vector = (0.0, 0.0)

        if not tags_data:
            cv2.putText(
                vis,
                "Pose: waiting for AprilTags\u2026",
                (8, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA,
            )
            return vis

        headless = context.get("headless", False)
        primary_correction_set = False

        for corners_2d, tag_id in tags_data:
            # Draw tag border when using the internal detector (the main
            # detector already draws borders in the GUI code path).
            if using_own_detector:
                pts = corners_2d.astype(int).reshape((-1, 1, 2))
                cv2.polylines(vis, [pts], True, (0, 255, 0), 2)

            # Draw tag ID
            cx = int(corners_2d[:, 0].mean())
            cy = int(corners_2d[:, 1].mean())
            cv2.putText(
                vis, f"ID:{tag_id}", (cx - 20, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA,
            )

            # Initial pose estimation
            try:
                success, rvec, tvec = cv2.solvePnP(
                    self._obj_pts, corners_2d, cam_mtx, dist,
                )
            except cv2.error:
                logger.exception("solvePnP failed for tag %s", tag_id)
                continue

            if not success:
                continue

            # Pose refinement using Levenberg-Marquardt for higher accuracy
            try:
                rvec, tvec = cv2.solvePnPRefineLM(
                    self._obj_pts, corners_2d, cam_mtx, dist,
                    rvec, tvec,
                )
            except (cv2.error, AttributeError):
                # solvePnPRefineLM unavailable or failed – keep initial estimate
                logger.debug("solvePnPRefineLM unavailable for tag %s", tag_id)

            # Compute correction vector for the primary (first) tag
            tv = tvec.flatten()
            # Horizontal angle: positive = tag right of optical axis
            tag_angle_deg = math.degrees(math.atan2(float(tv[0]), float(tv[2])))
            # Euclidean distance in metres
            tag_distance_m = float(np.linalg.norm(tv))

            if not primary_correction_set:
                self._correction_vector = (tag_angle_deg, tag_distance_m)
                # Also update steering from pose (overrides pixel-based estimate)
                self._steering_vector = math.sin(math.radians(tag_angle_deg))
                primary_correction_set = True

            # Draw 3-D frame axes (X=red, Y=green, Z=blue)
            try:
                cv2.drawFrameAxes(
                    vis, cam_mtx, dist, rvec, tvec, self._tag_size * 0.5
                )
            except (cv2.error, AttributeError):
                # cv2.drawFrameAxes was added in OpenCV 4.1; fall back
                # to manual projection for older builds.
                _draw_pose_axes_fallback(
                    vis, cam_mtx, dist, rvec, tvec, self._tag_size * 0.5
                )

            if headless:
                frame_idx = context.get("frame_idx", 0)
                rv = rvec.flatten()
                print(
                    f"[frame {frame_idx}] "
                    f"ID: {tag_id} | "
                    f"tvec: [{tv[0]:.4f}, {tv[1]:.4f}, {tv[2]:.4f}] | "
                    f"rvec: [{rv[0]:.4f}, {rv[1]:.4f}, {rv[2]:.4f}] | "
                    f"angle: {tag_angle_deg:+.2f}° | "
                    f"dist: {tag_distance_m:.4f}m | "
                    f"steering: {self._steering_vector:+.4f}"
                )

        return vis
