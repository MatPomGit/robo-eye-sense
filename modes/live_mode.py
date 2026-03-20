"""Live ASCII-art and 2-D map visualization modes.

LiveMode
    Renders the camera frame as ASCII art in the terminal, refreshed on every
    frame.  Detected AprilTag markers are highlighted with distinct characters
    and (where the terminal supports it) ANSI colours.

LiveMapMode
    Instead of the camera image, renders a 2-D top-down map of the space
    around the robot.  The robot is shown at the centre of the map; each
    detected AprilTag is placed at its estimated 3-D position (projected onto
    the horizontal XZ plane).  Per-tag data printed below the map includes the
    distance, horizontal angle, direction vector, and estimated rotation angle.
"""

from __future__ import annotations

import logging
import math
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from robo_vision import april_tag_detector as _april_runtime

from .base import BaseMode

logger = logging.getLogger(__name__)

# All AprilTag families supported by pupil_apriltags
_ALL_FAMILIES = "tag36h11 tag25h9 tag16h5 tag12h10"

# ASCII character set ordered from darkest (index 0) to brightest (last).
_ASCII_CHARS = " .:-=+*#%@"

# ANSI escape codes for terminal styling
_ANSI_TAG = "\033[93m"    # bright yellow – for highlighted tag pixels
_ANSI_ROBOT = "\033[96m"  # bright cyan   – for the robot marker
_ANSI_LINE = "\033[90m"   # dark grey     – for direction-vector lines on map
_ANSI_HEADER = "\033[1m"  # bold           – for header/legend text
_ANSI_RESET = "\033[0m"   # reset all attributes

# Terminal control: clear screen and move cursor to top-left
_CLEAR_SCREEN = "\033[2J\033[H"


def _ansi_supported() -> bool:
    """Return True when stdout is a real TTY (likely to support ANSI codes)."""
    return sys.stdout.isatty()


def _get_tag_corners_3d(tag_size: float) -> np.ndarray:
    """Return the four 3-D corners of an AprilTag centred at origin."""
    half = tag_size / 2.0
    return np.array([
        [-half, -half, 0],
        [ half, -half, 0],
        [ half,  half, 0],
        [-half,  half, 0],
    ], dtype=np.float64)


def _default_camera_matrix(w: int, h: int) -> np.ndarray:
    """Synthesise a camera matrix assuming ~60° HFOV."""
    fx = w / (2.0 * np.tan(np.radians(30)))
    return np.array([
        [fx, 0, w / 2.0],
        [0, fx, h / 2.0],
        [0,  0,      1.0],
    ], dtype=np.float64)


def _load_calibration(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load camera matrix and distortion coefficients from an .npz file."""
    data = np.load(path)
    return data["camera_matrix"], data["dist_coeffs"]


def _detect_and_estimate_poses(
    detector: Any,
    frame: np.ndarray,
    cam_mtx: np.ndarray,
    dist: np.ndarray,
    obj_pts: np.ndarray,
) -> List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """Detect AprilTags and estimate their 6-DoF poses.

    Parameters
    ----------
    detector:
        Initialised AprilTag detector (pupil_apriltags or apriltag).
    frame:
        BGR camera frame.
    cam_mtx:
        3×3 camera intrinsic matrix.
    dist:
        Distortion coefficients.
    obj_pts:
        4×3 array of 3-D object corners for the tag (from
        :func:`_get_tag_corners_3d`).

    Returns
    -------
    list of (tag_id, corners_2d, tvec, rvec)
        One entry per successfully localised tag.  *tag_id* is a string,
        *corners_2d* is (4, 2) float64, *tvec* and *rvec* are (3,) float64
        arrays.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    try:
        raw_tags = detector.detect(gray)
    except Exception:
        logger.exception("AprilTag detection failed")
        return []

    results: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    for tag in raw_tags:
        corners = getattr(tag, "corners", None)
        if corners is None:
            continue
        corners_2d = np.array(corners, dtype=np.float64).reshape(-1, 2)
        tag_id = str(getattr(tag, "tag_id", "?"))

        try:
            success, rvec, tvec = cv2.solvePnP(obj_pts, corners_2d, cam_mtx, dist)
        except cv2.error:
            logger.debug("solvePnP failed for tag %s", tag_id)
            continue
        if not success:
            continue

        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                obj_pts, corners_2d, cam_mtx, dist, rvec, tvec,
            )
        except (cv2.error, AttributeError):
            pass  # keep initial estimate

        results.append((tag_id, corners_2d, tvec.flatten(), rvec.flatten()))

    return results


def _tag_rotation_yaw_deg(rvec: np.ndarray) -> float:
    """Extract the yaw angle (rotation about Y axis) of a tag from its rvec."""
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    return math.degrees(math.atan2(float(R[0, 2]), float(R[2, 2])))


def _bresenham(
    x0: int, y0: int, x1: int, y1: int
) -> List[Tuple[int, int]]:
    """Yield integer points on the line from (x0, y0) to (x1, y1)."""
    points: List[Tuple[int, int]] = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    if dx > dy:
        err = dx // 2
        while x != x1:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy // 2
        while y != y1:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    points.append((x1, y1))
    return points


# ---------------------------------------------------------------------------
# LiveMode
# ---------------------------------------------------------------------------


def render_live_ascii(
    frame: np.ndarray,
    tags_data: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    cols: int = 80,
    rows: int = 30,
    use_ansi: bool = True,
) -> str:
    """Convert *frame* to an ASCII-art string, highlighting detected tags.

    Parameters
    ----------
    frame:
        BGR camera frame.
    tags_data:
        List of ``(tag_id, corners_2d, tvec, rvec)`` as returned by
        :func:`_detect_and_estimate_poses`.
    cols, rows:
        Target width and height (in characters) of the ASCII art output.
    use_ansi:
        When ``True`` ANSI colour codes are embedded for terminal colour
        support.

    Returns
    -------
    str
        Multi-line ASCII art string (no trailing newline).
    """
    h_frame, w_frame = frame.shape[:2]

    # Resize frame to the target character grid
    resized = cv2.resize(frame, (cols, rows), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Build a tag-pixel mask at the *original* frame resolution, then resize.
    tag_mask_full = np.zeros((h_frame, w_frame), dtype=np.uint8)
    for _tag_id, corners_2d, _tvec, _rvec in tags_data:
        pts = corners_2d.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(tag_mask_full, [pts], 255)

    tag_mask = cv2.resize(
        tag_mask_full, (cols, rows), interpolation=cv2.INTER_NEAREST
    )

    # Build mutable character grid: tag pixels → '#', others → brightness char.
    grid: List[List[str]] = []
    for y in range(rows):
        row: List[str] = []
        for x in range(cols):
            if tag_mask[y, x] > 0:
                row.append("#")
            else:
                px = int(gray[y, x])
                idx = int(px * (len(_ASCII_CHARS) - 1) / 255)
                row.append(_ASCII_CHARS[idx])
        grid.append(row)

    # Overlay tag-ID labels at tag centres (post-processing step so labels
    # are not truncated by the per-column loop logic).
    for tag_id, corners_2d, _tvec, _rvec in tags_data:
        cx = int(corners_2d[:, 0].mean() * cols / max(w_frame, 1))
        cy = int(corners_2d[:, 1].mean() * rows / max(h_frame, 1))
        cx = max(0, min(cols - 1, cx))
        cy = max(0, min(rows - 1, cy))
        label = f"[{tag_id}]"
        lbl_start = cx - len(label) // 2
        for i, ch in enumerate(label):
            lx = lbl_start + i
            if 0 <= lx < cols:
                grid[cy][lx] = ch

    # Convert grid to strings, applying optional ANSI tag highlighting.
    lines: List[str] = []
    for y in range(rows):
        if use_ansi:
            row_str = ""
            in_tag = False
            for x in range(cols):
                is_tag = tag_mask[y, x] > 0
                if is_tag and not in_tag:
                    row_str += _ANSI_TAG
                    in_tag = True
                elif not is_tag and in_tag:
                    row_str += _ANSI_RESET
                    in_tag = False
                row_str += grid[y][x]
            if in_tag:
                row_str += _ANSI_RESET
        else:
            row_str = "".join(grid[y])
        lines.append(row_str)

    # Build header info
    header_parts: List[str] = []
    if use_ansi:
        header_parts.append(_ANSI_HEADER)
    header_parts.append(f"[LIVE] tags={len(tags_data)}")
    if use_ansi:
        header_parts.append(_ANSI_RESET)
    for tag_id, _corners_2d, tvec, rvec in tags_data:
        dist_m = float(np.linalg.norm(tvec))
        angle_deg = math.degrees(math.atan2(float(tvec[0]), float(tvec[2])))
        yaw_deg = _tag_rotation_yaw_deg(rvec)
        header_parts.append(
            f"  tag {tag_id}: dist={dist_m:.2f}m  "
            f"angle={angle_deg:+.1f}°  rot={yaw_deg:+.1f}°"
        )

    return "\n".join([" ".join(header_parts)] + lines)


class LiveMode(BaseMode):
    """Renders the camera frame as live ASCII art in the terminal.

    Detected AprilTag markers are highlighted with distinct characters and
    (optionally) ANSI colour codes.

    Parameters
    ----------
    tag_size:
        Physical side length of AprilTags in metres.
    calibration_path:
        Path to a camera calibration ``.npz`` file.  Falls back to a
        synthetic matrix when not supplied.
    cols, rows:
        Width and height (in characters) of the ASCII art output.
    use_ansi:
        When ``None`` (default) ANSI codes are used only if stdout is a TTY.
        Pass ``True`` or ``False`` to override.
    """

    renders_to_terminal: bool = True

    def __init__(
        self,
        tag_size: float = 0.05,
        calibration_path: Optional[str] = None,
        cols: int = 80,
        rows: int = 30,
        use_ansi: Optional[bool] = None,
    ) -> None:
        self._tag_size = tag_size
        self._obj_pts = _get_tag_corners_3d(tag_size)
        self._cols = cols
        self._rows = rows
        self._use_ansi = _ansi_supported() if use_ansi is None else use_ansi

        self._camera_matrix: Optional[np.ndarray] = None
        self._dist_coeffs: Optional[np.ndarray] = None
        if calibration_path is not None:
            try:
                self._camera_matrix, self._dist_coeffs = _load_calibration(
                    calibration_path
                )
            except (FileNotFoundError, KeyError) as exc:
                logger.warning(
                    "LiveMode: could not load calibration from %s: %s",
                    calibration_path, exc,
                )

        self._detector: Any = None

    def _ensure_detector(self) -> None:
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
            return
        except Exception as exc:
            logger.warning("pupil_apriltags init failed: %s", exc)

        try:
            import apriltag  # type: ignore[import-untyped]

            self._detector = apriltag.Detector()
        except Exception as exc:
            logger.error("AprilTag detector not available: %s", exc)
            self._detector = None

    def run(self, frame: np.ndarray, context: Dict[str, Any]) -> np.ndarray:
        """Detect AprilTags, render ASCII art, and print to the terminal.

        Parameters
        ----------
        frame:
            BGR camera frame.
        context:
            Runtime metadata (``frame_idx``, ``fps``, …).

        Returns
        -------
        np.ndarray
            A copy of *frame* annotated with tag outlines, suitable for
            optional recording.
        """
        h, w = frame.shape[:2]
        cam_mtx = (
            self._camera_matrix
            if self._camera_matrix is not None
            else _default_camera_matrix(w, h)
        )
        dist = (
            self._dist_coeffs
            if self._dist_coeffs is not None
            else np.zeros(5, dtype=np.float64)
        )

        tags_data: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
        self._ensure_detector()
        if self._detector is not None:
            tags_data = _detect_and_estimate_poses(
                self._detector, frame, cam_mtx, dist, self._obj_pts
            )

        frame_idx = context.get("frame_idx", 0)
        fps = context.get("fps", 0.0)

        art = render_live_ascii(
            frame, tags_data, self._cols, self._rows, self._use_ansi
        )
        header = f"frame={frame_idx}  fps={fps:.1f}  ({w}x{h})\n"
        print(_CLEAR_SCREEN + header + art, end="", flush=True)

        # Return annotated frame for optional recording
        vis = frame.copy()
        for _tag_id, corners_2d, _tvec, _rvec in tags_data:
            pts = corners_2d.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [pts], True, (0, 230, 255), 2)
        return vis


# ---------------------------------------------------------------------------
# LiveMapMode
# ---------------------------------------------------------------------------


def render_live_map(
    tags_data: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    map_width: int = 70,
    map_height: int = 35,
    scale: float = 15.0,
    use_ansi: bool = True,
) -> str:
    """Render a 2-D ASCII top-down map with the robot and AprilTag positions.

    Parameters
    ----------
    tags_data:
        List of ``(tag_id, corners_2d, tvec, rvec)`` tuples.
    map_width, map_height:
        Dimensions of the character grid.
    scale:
        Characters per metre.  Larger values zoom in; smaller values zoom out.
    use_ansi:
        When ``True`` ANSI colour codes are embedded.

    Returns
    -------
    str
        Multi-line string containing the map, legend, and per-tag statistics.
        No trailing newline.
    """
    grid: List[List[str]] = [[" "] * map_width for _ in range(map_height)]

    # Robot position at centre of grid
    robot_col = map_width // 2
    robot_row = map_height // 2

    # Draw light crosshair axes
    for c in range(map_width):
        if grid[robot_row][c] == " ":
            grid[robot_row][c] = "─"
    for r in range(map_height):
        if grid[r][robot_col] == " ":
            grid[r][robot_col] = "│"
    grid[robot_row][robot_col] = "┼"

    # Place robot marker "(R)"
    robot_label = "(R)"
    rl_start = robot_col - len(robot_label) // 2
    for i, ch in enumerate(robot_label):
        c = rl_start + i
        if 0 <= c < map_width:
            grid[robot_row][c] = ch

    # Per-tag info lines collected below the map
    info_lines: List[str] = []

    for tag_id, _corners_2d, tvec, rvec in tags_data:
        tx, _ty, tz = float(tvec[0]), float(tvec[1]), float(tvec[2])

        # Map camera coords to grid:
        #   tx > 0  → tag is to the right  → grid col > robot_col
        #   tz > 0  → tag is in front        → grid row < robot_row (up on screen)
        tag_col = robot_col + int(round(tx * scale))
        tag_row = robot_row - int(round(tz * scale))

        # Draw direction-vector line from robot to tag
        for lc, lr in _bresenham(robot_col, robot_row, tag_col, tag_row):
            if 0 <= lr < map_height and 0 <= lc < map_width:
                if grid[lr][lc] in (" ", "─", "│", "┼"):
                    grid[lr][lc] = "·"

        # Place tag label "[ID]"
        label = f"[{tag_id}]"
        lbl_start = tag_col - len(label) // 2
        placed = False
        for i, ch in enumerate(label):
            c = lbl_start + i
            if 0 <= tag_row < map_height and 0 <= c < map_width:
                grid[tag_row][c] = ch
                placed = True

        if not placed:
            # Tag is off the map – mark the border near the edge
            border_row = max(0, min(map_height - 1, tag_row))
            border_col = max(0, min(map_width - 1, tag_col))
            grid[border_row][border_col] = "?"

        # Compute display statistics
        dist_m = float(np.linalg.norm(tvec))
        angle_deg = math.degrees(math.atan2(tx, float(tz)))
        dir_x = tx / dist_m if dist_m > 1e-6 else 0.0
        dir_z = float(tz) / dist_m if dist_m > 1e-6 else 0.0
        yaw_deg = _tag_rotation_yaw_deg(rvec)

        info_lines.append(
            f"  tag {tag_id:>4s} | "
            f"dist={dist_m:.2f} m | "
            f"angle={angle_deg:+.1f}° | "
            f"dir=({dir_x:+.2f}, {dir_z:+.2f}) | "
            f"rot={yaw_deg:+.1f}°"
        )

    # Convert grid to string (with optional ANSI colouring)
    grid_lines: List[str] = []
    for r, row in enumerate(grid):
        row_str = "".join(row)
        if use_ansi:
            # Highlight robot marker
            row_str = row_str.replace("(R)", _ANSI_ROBOT + "(R)" + _ANSI_RESET)
            # Highlight tag labels
            for tag_id, *_ in tags_data:
                lbl = f"[{tag_id}]"
                row_str = row_str.replace(
                    lbl, _ANSI_TAG + lbl + _ANSI_RESET
                )
            # Colour direction-vector dots
            row_str = row_str.replace("·", _ANSI_LINE + "·" + _ANSI_RESET)
        grid_lines.append(row_str)

    sep = "─" * map_width
    legend_line = (
        f"Scale: 1 char ≈ {1.0/scale:.2f} m  |  "
        f"(R)=robot  [ID]=tag  ·=direction  ↑=Z+ (forward)  →=X+ (right)"
    )
    if use_ansi:
        legend_line = _ANSI_HEADER + legend_line + _ANSI_RESET

    parts: List[str] = [legend_line, sep] + grid_lines + [sep]
    if info_lines:
        parts += info_lines
    else:
        parts.append("  No markers detected.")

    return "\n".join(parts)


class LiveMapMode(BaseMode):
    """Renders a live 2-D top-down ASCII map of the robot and AprilTags.

    Parameters
    ----------
    tag_size:
        Physical side length of AprilTags in metres.
    calibration_path:
        Path to a camera calibration ``.npz`` file.  Falls back to a
        synthetic matrix when not supplied.
    map_width, map_height:
        Dimensions of the character grid.
    scale:
        Characters per metre.  Controls the zoom level of the map.
    use_ansi:
        When ``None`` (default) ANSI codes are used only if stdout is a TTY.
    """

    renders_to_terminal: bool = True

    def __init__(
        self,
        tag_size: float = 0.05,
        calibration_path: Optional[str] = None,
        map_width: int = 70,
        map_height: int = 35,
        scale: float = 15.0,
        use_ansi: Optional[bool] = None,
    ) -> None:
        self._tag_size = tag_size
        self._obj_pts = _get_tag_corners_3d(tag_size)
        self._map_width = map_width
        self._map_height = map_height
        self._scale = scale
        self._use_ansi = _ansi_supported() if use_ansi is None else use_ansi

        self._camera_matrix: Optional[np.ndarray] = None
        self._dist_coeffs: Optional[np.ndarray] = None
        if calibration_path is not None:
            try:
                self._camera_matrix, self._dist_coeffs = _load_calibration(
                    calibration_path
                )
            except (FileNotFoundError, KeyError) as exc:
                logger.warning(
                    "LiveMapMode: could not load calibration from %s: %s",
                    calibration_path, exc,
                )

        self._detector: Any = None

    def _ensure_detector(self) -> None:
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
            return
        except Exception as exc:
            logger.warning("pupil_apriltags init failed: %s", exc)

        try:
            import apriltag  # type: ignore[import-untyped]

            self._detector = apriltag.Detector()
        except Exception as exc:
            logger.error("AprilTag detector not available: %s", exc)
            self._detector = None

    def run(self, frame: np.ndarray, context: Dict[str, Any]) -> np.ndarray:
        """Detect AprilTags, render the 2-D map, and print to the terminal.

        Parameters
        ----------
        frame:
            BGR camera frame (used only for detection; not rendered as image).
        context:
            Runtime metadata (``frame_idx``, ``fps``, …).

        Returns
        -------
        np.ndarray
            A copy of *frame* annotated with tag outlines, suitable for
            optional recording.
        """
        h, w = frame.shape[:2]
        cam_mtx = (
            self._camera_matrix
            if self._camera_matrix is not None
            else _default_camera_matrix(w, h)
        )
        dist = (
            self._dist_coeffs
            if self._dist_coeffs is not None
            else np.zeros(5, dtype=np.float64)
        )

        tags_data: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
        self._ensure_detector()
        if self._detector is not None:
            tags_data = _detect_and_estimate_poses(
                self._detector, frame, cam_mtx, dist, self._obj_pts
            )

        frame_idx = context.get("frame_idx", 0)
        fps = context.get("fps", 0.0)

        map_str = render_live_map(
            tags_data,
            self._map_width,
            self._map_height,
            self._scale,
            self._use_ansi,
        )
        header = f"frame={frame_idx}  fps={fps:.1f}  tags={len(tags_data)}\n"
        print(_CLEAR_SCREEN + header + map_str, end="", flush=True)

        # Return annotated frame for optional recording
        vis = frame.copy()
        for _tag_id, corners_2d, _tvec, _rvec in tags_data:
            pts = corners_2d.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [pts], True, (0, 230, 255), 2)
        return vis
