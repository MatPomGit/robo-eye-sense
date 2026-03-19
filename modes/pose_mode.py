"""Pose estimation mode – estimate 6-DoF pose of AprilTags.

Loads camera calibration from a ``.npz`` file, detects AprilTags using
``pupil_apriltags``, and estimates their pose with ``cv2.solvePnP``.
Visualises tag borders, IDs, and 3-D axes on the frame.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .base import BaseMode

logger = logging.getLogger(__name__)

# All tag families supported by pupil_apriltags
_ALL_FAMILIES = "tag36h11 tag25h9 tag16h5 tag12h10"


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
    """

    def __init__(
        self,
        tag_size: float = 0.05,
        calibration_path: Optional[str] = None,
    ) -> None:
        self._tag_size = tag_size
        self._obj_pts = _get_tag_corners_3d(tag_size)
        self._camera_matrix: Optional[np.ndarray] = None
        self._dist_coeffs: Optional[np.ndarray] = None

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
    def _ensure_detector(self) -> None:
        """Create the AprilTag detector on first use."""
        if self._detector is not None:
            return
        try:
            from pupil_apriltags import Detector  # type: ignore[import-untyped]

            self._detector = Detector(
                families=_ALL_FAMILIES,
                nthreads=1,
                quad_decimate=2.0,
                quad_sigma=0.0,
                refine_edges=1,
                decode_sharpening=0.25,
                debug=0,
            )
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

        if using_own_detector:
            self._ensure_detector()
            if self._detector is None:
                cv2.putText(
                    vis,
                    "Pose: no AprilTag detector available",
                    (8, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA,
                )
                return vis

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            try:
                raw_tags = self._detector.detect(gray)
            except Exception:
                logger.exception("AprilTag detection failed")
                return vis

            for tag in raw_tags:
                corners = getattr(tag, "corners", None)
                if corners is None:
                    continue
                corners_2d = np.array(corners, dtype=np.float64).reshape(-1, 2)
                tag_id = str(getattr(tag, "tag_id", "?"))
                tags_data.append((corners_2d, tag_id))
        else:
            # Use detections provided by the GUI's main detector
            for det in april_detections:
                if not det.corners:
                    continue
                corners_2d = np.array(det.corners, dtype=np.float64).reshape(-1, 2)
                tags_data.append((corners_2d, det.identifier))

        if not tags_data:
            cv2.putText(
                vis,
                "Pose: waiting for AprilTags\u2026",
                (8, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA,
            )
            return vis

        headless = context.get("headless", False)

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

            # Pose estimation
            try:
                success, rvec, tvec = cv2.solvePnP(
                    self._obj_pts, corners_2d, cam_mtx, dist,
                )
            except cv2.error:
                logger.exception("solvePnP failed for tag %s", tag_id)
                continue

            if success:
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
                    tv = tvec.flatten()
                    rv = rvec.flatten()
                    print(
                        f"[frame {frame_idx}] "
                        f"ID: {tag_id} | "
                        f"tvec: [{tv[0]:.4f}, {tv[1]:.4f}, {tv[2]:.4f}] | "
                        f"rvec: [{rv[0]:.4f}, {rv[1]:.4f}, {rv[2]:.4f}]"
                    )

        return vis
