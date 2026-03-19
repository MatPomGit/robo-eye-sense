"""SLAM-based marker map building and robot localisation.

Provides data structures and algorithms for incrementally constructing
a **marker map** (tag map) that records the 3-D position and orientation
of every observed fiducial marker in a single, shared world frame.  Once
the map is built the same module can estimate the 6-DoF pose of the robot
(camera) within that map.

Workflow
--------
1. Create a :class:`SlamCalibrator` with the known physical marker size
   and (optionally) the camera intrinsic matrix.
2. Feed frames through :meth:`SlamCalibrator.process_detections` — this runs
   AprilTag detection, estimates the pose of every visible marker with
   ``cv2.solvePnPRansac`` (followed by Levenberg-Marquardt refinement),
   and fuses observations into the growing map.
3. After enough observations, call :meth:`SlamCalibrator.marker_map` to
   obtain the finished :class:`MarkerMap`.
4. The map can be saved to / loaded from a JSON file for later re-use.
5. At runtime, use :meth:`MarkerMap.estimate_robot_pose` to determine
   where the robot is within the map from a single set of detections.

The module is designed so that the lightweight data classes
(:class:`MarkerPose3D`, :class:`RobotPose3D`, :class:`MarkerMap`) have
**no OpenCV dependency** and can be imported without ``cv2``.  Only
:class:`SlamCalibrator` and the helper functions that wrap
``solvePnPRansac`` require OpenCV.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .results import Detection, DetectionType

# RANSAC PnP tuning parameters
_RANSAC_REPROJ_THRESHOLD = 8.0   # max reprojection error (px) to accept an inlier
_RANSAC_ITER_SINGLE = 100        # RANSAC iterations for single-marker PnP
_RANSAC_ITER_MULTI = 200         # RANSAC iterations for multi-marker PnP
_CORNERS_PER_MARKER = 4          # corners produced by each AprilTag detection


# ---------------------------------------------------------------------------
# Camera-matrix helpers (no cv2 dependency)
# ---------------------------------------------------------------------------


def _default_camera_matrix(
    width: int,
    height: int,
    hfov_deg: float = 60.0,
) -> List[List[float]]:
    """Compute a default 3×3 camera intrinsic matrix.

    Assumes a horizontal field of view of *hfov_deg* degrees and places
    the principal point at the image centre.  This is a reasonable
    first-order approximation for most web-cams and USB cameras.

    Parameters
    ----------
    width, height:
        Frame dimensions in pixels.
    hfov_deg:
        Horizontal field of view in degrees (default 60°).
    """
    fx = width / (2.0 * math.tan(math.radians(hfov_deg / 2.0)))
    cx = width / 2.0
    cy = height / 2.0
    return [
        [fx,  0.0, cx],
        [0.0, fx,  cy],
        [0.0, 0.0, 1.0],
    ]

# ---------------------------------------------------------------------------
# Data structures (no cv2 dependency)
# ---------------------------------------------------------------------------


@dataclass
class MarkerPose3D:
    """Position and orientation of a single marker in the world frame.

    Attributes
    ----------
    marker_id:
        Unique string identifier of the marker (e.g. the AprilTag ID).
    position:
        ``(x, y, z)`` translation in the world coordinate system
        (centimetres).
    orientation:
        ``(roll, pitch, yaw)`` Euler angles in degrees describing the
        marker's orientation in the world frame.
    observations:
        How many independent observations contributed to this estimate.
        More observations generally mean a more reliable pose.
    """

    marker_id: str
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    observations: int = 1


@dataclass
class RobotPose3D:
    """Estimated 6-DoF pose of the robot (camera) in the world frame.

    Attributes
    ----------
    position:
        ``(x, y, z)`` translation in the world coordinate system
        (centimetres).
    orientation:
        ``(roll, pitch, yaw)`` Euler angles in degrees.
    visible_markers:
        Number of mapped markers used to compute this estimate.
    reprojection_error:
        Average reprojection error (pixels) of the pose estimate.
        ``None`` when no valid estimate could be made.
    """

    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    visible_markers: int = 0
    reprojection_error: Optional[float] = None


# ---------------------------------------------------------------------------
# MarkerMap — serialisable collection of MarkerPose3D
# ---------------------------------------------------------------------------


class MarkerMap:
    """A collection of :class:`MarkerPose3D` entries keyed by marker ID.

    The map represents a snapshot of the known world — every entry records
    where a particular marker sits and how it is oriented in a common 3-D
    coordinate frame.

    The map can be serialised to / from a JSON file so that calibration
    does not have to be repeated every time the robot starts.
    """

    def __init__(self) -> None:
        self._markers: Dict[str, MarkerPose3D] = {}

    # -- accessors ----------------------------------------------------------

    @property
    def marker_ids(self) -> List[str]:
        """Sorted list of all marker IDs in the map."""
        return sorted(self._markers)

    def __len__(self) -> int:
        return len(self._markers)

    def __contains__(self, marker_id: str) -> bool:
        return marker_id in self._markers

    def get(self, marker_id: str) -> Optional[MarkerPose3D]:
        """Return the pose for *marker_id*, or ``None``."""
        return self._markers.get(marker_id)

    def markers(self) -> List[MarkerPose3D]:
        """Return all marker poses, ordered by marker ID."""
        return [self._markers[k] for k in sorted(self._markers)]

    # -- mutators -----------------------------------------------------------

    def add(self, pose: MarkerPose3D) -> None:
        """Add or replace a marker pose."""
        self._markers[pose.marker_id] = pose

    def remove(self, marker_id: str) -> bool:
        """Remove a marker by ID.  Returns ``True`` if it existed."""
        return self._markers.pop(marker_id, None) is not None  # type: ignore[arg-type]

    def clear(self) -> None:
        """Remove all markers from the map."""
        self._markers.clear()

    # -- merge (running average) --------------------------------------------

    def merge_observation(
        self,
        marker_id: str,
        position: Tuple[float, float, float],
        orientation: Tuple[float, float, float],
    ) -> None:
        """Merge a new observation into the map using a running average.

        If the marker already exists the stored position and orientation
        are updated as a weighted average where the weight of the existing
        estimate equals the number of prior observations.  This gives a
        simple incremental refinement effect without storing all raw data.

        If the marker is new, it is inserted directly.
        """
        existing = self._markers.get(marker_id)
        if existing is None:
            self._markers[marker_id] = MarkerPose3D(
                marker_id=marker_id,
                position=position,
                orientation=orientation,
                observations=1,
            )
            return

        n = existing.observations
        new_n = n + 1
        avg_pos = tuple(
            (old * n + new) / new_n
            for old, new in zip(existing.position, position)
        )
        avg_ori = tuple(
            _angle_average(old, new, n)
            for old, new in zip(existing.orientation, orientation)
        )
        self._markers[marker_id] = MarkerPose3D(
            marker_id=marker_id,
            position=avg_pos,  # type: ignore[arg-type]
            orientation=avg_ori,  # type: ignore[arg-type]
            observations=new_n,
        )

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "markers": [asdict(m) for m in self.markers()],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MarkerMap":
        """Construct a :class:`MarkerMap` from a dictionary (e.g. loaded JSON)."""
        mm = cls()
        for entry in data.get("markers", []):
            pose = MarkerPose3D(
                marker_id=str(entry["marker_id"]),
                position=tuple(entry["position"]),  # type: ignore[arg-type]
                orientation=tuple(entry["orientation"]),  # type: ignore[arg-type]
                observations=int(entry.get("observations", 1)),
            )
            mm.add(pose)
        return mm

    def save(self, path: str | Path) -> None:
        """Write the marker map to a JSON file at *path*."""
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "MarkerMap":
        """Load a marker map from a JSON file at *path*."""
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    # -- robot pose estimation (pure-Python, no cv2) ------------------------

    def estimate_robot_pose(
        self,
        detections: Sequence[Detection],
        camera_matrix: Optional[Sequence[Sequence[float]]] = None,
        tag_size_cm: float = 5.0,
    ) -> RobotPose3D:
        """Estimate the robot's 3-D pose given current *detections*.

        When two or more mapped markers are visible the function uses
        **multi-marker RANSAC PnP**: all 3-D ↔ 2-D point correspondences
        from every known marker are fed into a single
        ``cv2.solvePnPRansac`` call (with Levenberg-Marquardt refinement).
        RANSAC rejects outlier markers automatically, and the combined
        optimisation is more accurate than averaging per-marker estimates.

        When only one mapped marker is visible the function falls back
        to a single-marker ``solvePnPRansac`` call.

        Parameters
        ----------
        detections:
            Current frame detections (only AprilTags with ≥ 4 corners
            that appear in the map are used).
        camera_matrix:
            3×3 camera intrinsic matrix as a nested sequence.  When
            ``None`` the function falls back to a simple distance-based
            estimate using the pinhole model (less accurate).
        tag_size_cm:
            Physical side-length of each tag in centimetres.

        Returns
        -------
        RobotPose3D
            Best-effort 6-DoF pose of the robot.  ``visible_markers``
            will be ``0`` if no mapped markers were seen.
        """
        try:
            import cv2  # noqa: F811
            import numpy as np
        except ImportError:
            return RobotPose3D()

        if camera_matrix is not None:
            cam_mtx = np.array(camera_matrix, dtype=np.float64)
        else:
            cam_mtx = None

        # Collect valid detections that exist in the map
        valid_dets: List[Detection] = []
        for det in detections:
            if det.detection_type != DetectionType.APRIL_TAG:
                continue
            if det.identifier is None or det.identifier not in self._markers:
                continue
            if len(det.corners) < 4:
                continue
            valid_dets.append(det)

        if not valid_dets:
            return RobotPose3D()

        # --- Multi-marker RANSAC PnP (≥ 2 markers) ---
        if len(valid_dets) >= 2:
            result = _estimate_pose_multi_marker(
                valid_dets, self._markers, tag_size_cm, cam_mtx,
            )
            if result is not None:
                return result

        # --- Single-marker fallback ---
        positions: List[Tuple[float, float, float]] = []
        orientations: List[Tuple[float, float, float]] = []
        errors: List[float] = []

        for det in valid_dets:
            marker_world = self._markers[det.identifier]

            rvec, tvec, err = _solve_marker_pose(
                det.corners, tag_size_cm, cam_mtx,
            )
            if rvec is None or tvec is None:
                continue

            # Camera pose in marker frame: invert marker→camera transform
            R_cm, _ = cv2.Rodrigues(rvec)
            R_mc = R_cm.T
            t_mc = -R_mc @ tvec  # camera position in marker local frame

            # Transform camera position to world frame
            R_mw = _euler_to_rotation_matrix(*marker_world.orientation)
            cam_world = R_mw @ t_mc.flatten() + np.array(marker_world.position)

            # Camera orientation in world frame
            R_cw = R_mw @ R_mc.T
            r, p, y = _rotation_matrix_to_euler(R_cw)

            positions.append((float(cam_world[0]), float(cam_world[1]), float(cam_world[2])))
            orientations.append((r, p, y))
            if err is not None:
                errors.append(err)

        if not positions:
            return RobotPose3D()

        n = len(positions)
        avg_pos = tuple(sum(c) / n for c in zip(*positions))
        avg_ori = tuple(
            _mean_angles([o[i] for o in orientations]) for i in range(3)
        )
        avg_err = sum(errors) / len(errors) if errors else None

        return RobotPose3D(
            position=avg_pos,  # type: ignore[arg-type]
            orientation=avg_ori,  # type: ignore[arg-type]
            visible_markers=n,
            reprojection_error=avg_err,
        )


# ---------------------------------------------------------------------------
# SlamCalibrator — incremental map builder
# ---------------------------------------------------------------------------


class SlamCalibrator:
    """Incrementally build a :class:`MarkerMap` from camera observations.

    The calibrator implements a simplified marker-based SLAM pipeline:

    1. The **first frame** in which at least one marker is detected
       establishes the world coordinate frame.  The first marker seen is
       placed at the origin.
    2. In every subsequent frame, markers that are already in the map
       are used to estimate the current camera pose.
    3. Newly discovered markers are placed into the map using the
       estimated camera pose.
    4. Re-observed markers are refined through running-average fusion.

    Parameters
    ----------
    tag_size_cm:
        Physical side-length of each marker in centimetres.
    camera_matrix:
        3×3 camera intrinsic matrix (nested list / array).  If ``None``
        a default pinhole approximation is used.
    dist_coeffs:
        Lens distortion coefficients.  ``None`` means zero distortion.
    frame_size:
        ``(width, height)`` of the input frames in pixels.  Used to
        compute a default camera matrix when *camera_matrix* is ``None``.

    Attributes
    ----------
    marker_map : MarkerMap
        The current (possibly still growing) marker map.
    frame_count : int
        Number of frames processed so far.
    """

    def __init__(
        self,
        tag_size_cm: float = 5.0,
        camera_matrix: Optional[Sequence[Sequence[float]]] = None,
        dist_coeffs: Optional[Sequence[float]] = None,
        frame_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        self._tag_size_cm = tag_size_cm
        self._dist_coeffs = dist_coeffs
        self._map = MarkerMap()
        self._frame_count: int = 0

        # Resolve the camera matrix: explicit > computed from frame_size > None
        if camera_matrix is not None:
            self._camera_matrix: Optional[Sequence[Sequence[float]]] = camera_matrix
        elif frame_size is not None:
            w, h = frame_size
            self._camera_matrix = _default_camera_matrix(w, h)
        else:
            self._camera_matrix = None

    # -- public API ---------------------------------------------------------

    @property
    def marker_map(self) -> MarkerMap:
        """The current marker map (may still be growing)."""
        return self._map

    @property
    def frame_count(self) -> int:
        """Number of frames processed so far."""
        return self._frame_count

    def process_detections(self, detections: Sequence[Detection]) -> RobotPose3D:
        """Process a set of *detections* and update the internal map.

        Parameters
        ----------
        detections:
            Detections from a single frame (AprilTags with ≥ 4 corners).

        Returns
        -------
        RobotPose3D
            Estimated robot pose after incorporating this frame's data.
        """
        try:
            import cv2 as _cv2  # noqa: F811
            import numpy as _np  # noqa: F811
        except ImportError:
            return RobotPose3D()

        self._frame_count += 1

        april_dets = [
            d for d in detections
            if d.detection_type == DetectionType.APRIL_TAG
            and d.identifier is not None
            and len(d.corners) >= 4
        ]

        if not april_dets:
            return RobotPose3D()

        cam_mtx = (
            _np.array(self._camera_matrix, dtype=_np.float64)
            if self._camera_matrix is not None
            else None
        )

        # --- First frame: establish world origin ---
        if len(self._map) == 0:
            self._init_map_from_detections(april_dets, cam_mtx)
            return RobotPose3D(
                position=(0.0, 0.0, 0.0),
                orientation=(0.0, 0.0, 0.0),
                visible_markers=len(april_dets),
            )

        # --- Subsequent frames ---
        # 1. Estimate current camera (robot) pose from known markers
        robot_pose = self._map.estimate_robot_pose(
            april_dets,
            camera_matrix=self._camera_matrix,
            tag_size_cm=self._tag_size_cm,
        )

        # 2. Insert new markers / refine existing ones
        for det in april_dets:
            assert det.identifier is not None
            rvec, tvec, _ = _solve_marker_pose(
                det.corners, self._tag_size_cm, cam_mtx,
            )
            if rvec is None or tvec is None:
                continue

            # Marker pose relative to camera
            R_cm, _ = _cv2.Rodrigues(rvec)

            # Convert to world frame using the estimated robot pose
            R_rw = _euler_to_rotation_matrix(*robot_pose.orientation)
            t_rw = _np.array(robot_pose.position)

            # Marker position in world frame
            marker_world_pos = R_rw @ tvec.flatten() + t_rw

            # Marker orientation in world frame
            R_mw = R_rw @ R_cm
            marker_r, marker_p, marker_y = _rotation_matrix_to_euler(R_mw)

            world_pos = (
                float(marker_world_pos[0]),
                float(marker_world_pos[1]),
                float(marker_world_pos[2]),
            )
            world_ori = (marker_r, marker_p, marker_y)

            self._map.merge_observation(det.identifier, world_pos, world_ori)

        return robot_pose

    def reset(self) -> None:
        """Clear the map and frame counter."""
        self._map.clear()
        self._frame_count = 0

    # -- internal -----------------------------------------------------------

    def _init_map_from_detections(
        self,
        detections: List[Detection],
        cam_mtx: Any,
    ) -> None:
        """Place the first frame's markers into the map.

        The camera is assumed to be at the world origin ``(0, 0, 0)``
        facing along the +Z axis.  Each detected marker's 3-D pose
        relative to the camera becomes its world pose.
        """
        try:
            import cv2 as _cv2  # noqa: F811
        except ImportError:
            return

        for det in detections:
            if det.identifier is None or len(det.corners) < 4:
                continue

            rvec, tvec, _ = _solve_marker_pose(
                det.corners, self._tag_size_cm, cam_mtx,
            )
            if rvec is None or tvec is None:
                continue

            R, _ = _cv2.Rodrigues(rvec)
            roll, pitch, yaw = _rotation_matrix_to_euler(R)

            self._map.add(MarkerPose3D(
                marker_id=det.identifier,
                position=(float(tvec[0]), float(tvec[1]), float(tvec[2])),
                orientation=(roll, pitch, yaw),
                observations=1,
            ))


# ---------------------------------------------------------------------------
# Helper functions (cv2-dependent)
# ---------------------------------------------------------------------------


def _solve_marker_pose(
    corners: Sequence[Tuple[int, int]],
    tag_size_cm: float,
    camera_matrix: Any,
) -> Tuple[Any, Any, Optional[float]]:
    """Estimate the 6-DoF pose of a single marker via ``cv2.solvePnPRansac``.

    Uses RANSAC-based PnP for robustness against noisy corner detections,
    followed by Levenberg-Marquardt refinement (``cv2.solvePnPRefineLM``)
    for sub-pixel accuracy.

    Returns ``(rvec, tvec, reprojection_error)`` or ``(None, None, None)``
    on failure.
    """
    try:
        import cv2 as _cv2
        import numpy as _np
    except ImportError:
        return None, None, None

    if len(corners) < 4:
        return None, None, None

    half = tag_size_cm / 2.0
    obj_pts = _np.array([
        [-half, -half, 0.0],
        [half, -half, 0.0],
        [half,  half, 0.0],
        [-half,  half, 0.0],
    ], dtype=_np.float64)

    img_pts = _np.array(corners[:4], dtype=_np.float64)

    if camera_matrix is None:
        # Fall back to a standard camera matrix for a 640×480 sensor at 60° HFOV.
        # The principal point (cx, cy) must be the *frame* centre, not the tag
        # centroid.  Using the tag's bounding box as a proxy for focal length
        # and principal point produces large errors; this fixed approximation
        # is more reliable for typical USB cameras.
        default = _default_camera_matrix(640, 480)
        camera_matrix = _np.array(default, dtype=_np.float64)

    dist = _np.zeros(4, dtype=_np.float64)

    success, rvec, tvec, _inliers = _cv2.solvePnPRansac(
        obj_pts, img_pts, camera_matrix, dist,
        iterationsCount=_RANSAC_ITER_SINGLE,
        reprojectionError=_RANSAC_REPROJ_THRESHOLD,
        flags=_cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None, None, None

    # Levenberg-Marquardt refinement for sub-pixel accuracy
    try:
        rvec, tvec = _cv2.solvePnPRefineLM(
            obj_pts, img_pts, camera_matrix, dist, rvec, tvec,
        )
    except (_cv2.error, AttributeError):
        pass  # keep RANSAC estimate

    # Compute reprojection error
    proj, _ = _cv2.projectPoints(obj_pts, rvec, tvec, camera_matrix, dist)
    err = float(_np.mean(_np.linalg.norm(
        proj.reshape(-1, 2) - img_pts.reshape(-1, 2), axis=1,
    )))

    return rvec.flatten(), tvec.flatten(), err


def _estimate_pose_multi_marker(
    detections: List[Detection],
    markers: Dict[str, MarkerPose3D],
    tag_size_cm: float,
    camera_matrix: Any,
) -> Optional[RobotPose3D]:
    """Estimate the camera pose using all visible mapped markers at once.

    Collects 3-D ↔ 2-D correspondences from every *detection* whose
    marker ID exists in *markers*, transforms the 3-D corners to the
    shared world frame, and solves a single ``cv2.solvePnPRansac`` call
    followed by Levenberg-Marquardt refinement.

    Returns ``None`` if the solve fails (caller should fall back to the
    per-marker approach).
    """
    try:
        import cv2 as _cv2
        import numpy as _np
    except ImportError:
        return None

    half = tag_size_cm / 2.0
    local_corners = _np.array([
        [-half, -half, 0.0],
        [half, -half, 0.0],
        [half,  half, 0.0],
        [-half,  half, 0.0],
    ], dtype=_np.float64)

    all_obj: List[_np.ndarray] = []
    all_img: List[_np.ndarray] = []
    used_ids: List[str] = []

    for det in detections:
        assert det.identifier is not None
        mw = markers[det.identifier]
        R_mw = _euler_to_rotation_matrix(*mw.orientation)
        t_mw = _np.array(mw.position, dtype=_np.float64)

        for i in range(4):
            world_pt = R_mw @ local_corners[i] + t_mw
            all_obj.append(world_pt)
            all_img.append(_np.array(det.corners[i], dtype=_np.float64))
        used_ids.append(det.identifier)

    obj_pts = _np.array(all_obj, dtype=_np.float64)
    img_pts = _np.array(all_img, dtype=_np.float64)

    if camera_matrix is None:
        default = _default_camera_matrix(640, 480)
        camera_matrix = _np.array(default, dtype=_np.float64)

    dist = _np.zeros(4, dtype=_np.float64)

    success, rvec, tvec, inliers = _cv2.solvePnPRansac(
        obj_pts, img_pts, camera_matrix, dist,
        iterationsCount=_RANSAC_ITER_MULTI,
        reprojectionError=_RANSAC_REPROJ_THRESHOLD,
        flags=_cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None

    # Levenberg-Marquardt refinement
    try:
        rvec, tvec = _cv2.solvePnPRefineLM(
            obj_pts, img_pts, camera_matrix, dist, rvec, tvec,
        )
    except (_cv2.error, AttributeError):
        pass

    # Camera position and orientation in world frame
    R_wc, _ = _cv2.Rodrigues(rvec)
    cam_pos = (-R_wc.T @ tvec).flatten()
    R_cw = R_wc.T
    r, p, y = _rotation_matrix_to_euler(R_cw)

    # Reprojection error
    proj, _ = _cv2.projectPoints(obj_pts, rvec, tvec, camera_matrix, dist)
    err = float(_np.mean(_np.linalg.norm(
        proj.reshape(-1, 2) - img_pts.reshape(-1, 2), axis=1,
    )))

    n_markers = len(used_ids)
    if inliers is not None:
        n_markers = len(set(
            idx // _CORNERS_PER_MARKER for idx in inliers.flatten()
        ))

    return RobotPose3D(
        position=(float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])),
        orientation=(r, p, y),
        visible_markers=n_markers,
        reprojection_error=err,
    )


# ---------------------------------------------------------------------------
# Rotation helpers (pure Python / numpy)
# ---------------------------------------------------------------------------


def _euler_to_rotation_matrix(
    roll_deg: float, pitch_deg: float, yaw_deg: float,
) -> Any:
    """Convert Euler angles (degrees) to a 3×3 rotation matrix (numpy)."""
    import numpy as _np

    r = math.radians(roll_deg)
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)

    Rx = _np.array([
        [1, 0, 0],
        [0, math.cos(r), -math.sin(r)],
        [0, math.sin(r),  math.cos(r)],
    ])
    Ry = _np.array([
        [math.cos(p), 0, math.sin(p)],
        [0, 1, 0],
        [-math.sin(p), 0, math.cos(p)],
    ])
    Rz = _np.array([
        [math.cos(y), -math.sin(y), 0],
        [math.sin(y),  math.cos(y), 0],
        [0, 0, 1],
    ])
    return Rz @ Ry @ Rx


def _rotation_matrix_to_euler(R: Any) -> Tuple[float, float, float]:
    """Extract (roll, pitch, yaw) in degrees from a 3×3 rotation matrix."""
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0

    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def _angle_average(old_deg: float, new_deg: float, n: int) -> float:
    """Weighted circular average of two angles in degrees.

    *old_deg* has weight *n*; *new_deg* has weight 1.
    """
    old_r = math.radians(old_deg)
    new_r = math.radians(new_deg)
    sin_avg = (math.sin(old_r) * n + math.sin(new_r)) / (n + 1)
    cos_avg = (math.cos(old_r) * n + math.cos(new_r)) / (n + 1)
    return math.degrees(math.atan2(sin_avg, cos_avg))


def _mean_angles(angles_deg: List[float]) -> float:
    """Circular mean of a list of angles in degrees."""
    if not angles_deg:
        return 0.0
    sin_sum = sum(math.sin(math.radians(a)) for a in angles_deg)
    cos_sum = sum(math.cos(math.radians(a)) for a in angles_deg)
    return math.degrees(math.atan2(sin_sum, cos_sum))
