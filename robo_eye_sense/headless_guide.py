"""Comprehensive headless guide for robo-eye-sense.

Provides device status, camera discovery, calibration information,
AprilTag classification, and tag-name loading from files — all designed
for non-interactive (headless) operation.

AprilTag ID conventions
-----------------------
* **IDs < 10** – movable (dynamic) objects.
* **IDs 5–8** – packages intended for transport.
* **IDs 12–20** – markers that together define the plane of a stationary
  table (when they are detected close to one another).
* **IDs ≥ 10 (outside 12–20)** – generic static markers.

Usage
-----
The guide can be triggered from the CLI::

    python main.py --guide

Tag names can be loaded from a JSON file::

    python main.py --tag-names-file tag_names.json
"""

from __future__ import annotations

import json
import os
import platform
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ── AprilTag classification constants ────────────────────────────────────────

MOVABLE_ID_UPPER = 10
"""Tag IDs below this value represent movable (dynamic) objects."""

PACKAGE_ID_RANGE = range(5, 9)  # 5, 6, 7, 8
"""Tag IDs in this range represent packages intended for transport."""

TABLE_ID_RANGE = range(12, 21)  # 12 .. 20
"""Tag IDs in this range define the plane of a stationary table."""


# ── Tag classification ───────────────────────────────────────────────────────

def classify_tag(tag_id: int) -> str:
    """Return a human-readable category for the given AprilTag *tag_id*.

    Categories
    ----------
    * ``"package"`` – IDs 5–8 (movable, transport package).
    * ``"movable"`` – IDs 0–4 and 9 (generic movable object).
    * ``"table"``   – IDs 12–20 (stationary table plane marker).
    * ``"static"``  – all other IDs ≥ 10.
    """
    if tag_id in PACKAGE_ID_RANGE:
        return "package"
    if tag_id < MOVABLE_ID_UPPER:
        return "movable"
    if tag_id in TABLE_ID_RANGE:
        return "table"
    return "static"


# ── Tag-name file loading ────────────────────────────────────────────────────

def load_tag_names_from_file(filepath: str) -> Dict[str, str]:
    """Load a ``{id: name, …}`` mapping from a JSON file.

    The file must contain a single JSON object whose keys are AprilTag ID
    strings and whose values are human-readable names, e.g.::

        {
            "5": "package-A",
            "12": "table-left",
            "13": "table-right"
        }

    Parameters
    ----------
    filepath:
        Path to the JSON file.

    Returns
    -------
    dict
        Mapping of tag-ID strings to their custom names.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    json.JSONDecodeError
        If the file content is not valid JSON.
    TypeError
        If the top-level JSON value is not an object/dict.
    """
    with open(filepath, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise TypeError(
            f"Expected a JSON object (dict) in {filepath!r}, "
            f"got {type(data).__name__}"
        )
    return {str(k): str(v) for k, v in data.items()}


# ── Device / system status ───────────────────────────────────────────────────

def get_device_status() -> Dict[str, str]:
    """Collect basic device and runtime information.

    Returns a dict with keys such as ``platform``, ``python``,
    ``opencv``, ``machine``, and ``processor``.
    """
    info: Dict[str, str] = {
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor() or "n/a",
        "python": platform.python_version(),
    }
    try:
        import cv2  # type: ignore[import-untyped]
        info["opencv"] = cv2.__version__
    except ImportError:  # pragma: no cover
        info["opencv"] = "not installed"
    return info


# ── Camera discovery ─────────────────────────────────────────────────────────

def discover_cameras(max_index: int = 5) -> List[Dict[str, Any]]:
    """Probe camera indices ``0 .. max_index-1`` and return available ones.

    Each entry in the returned list is a dict with ``index`` and the same
    keys produced by :meth:`robo_eye_sense.camera.Camera.get_info`.

    Parameters
    ----------
    max_index:
        Number of indices to probe (default 5).

    Returns
    -------
    list[dict]
        One dict per successfully opened camera.
    """
    import cv2  # type: ignore[import-untyped]

    cameras: List[Dict[str, Any]] = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            entry: Dict[str, Any] = {
                "index": idx,
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "backend": cap.getBackendName(),
            }
            cap.release()
            cameras.append(entry)
        else:
            cap.release()
    return cameras


# ── Calibration info ─────────────────────────────────────────────────────────

def get_calibration_info(calib_path: str) -> Dict[str, Any]:
    """Check whether a calibration file exists and report its metadata.

    Parameters
    ----------
    calib_path:
        Path to a ``.npz`` calibration file (e.g. ``calibration.npz``).

    Returns
    -------
    dict
        ``exists`` (bool), and – when the file is present – ``path``,
        ``calibrated_at`` (ISO-8601 string of the file's last-modified
        time), and ``calibrated_at_local`` (human-readable local time).
    """
    result: Dict[str, Any] = {"exists": False, "path": calib_path}
    if os.path.isfile(calib_path):
        result["exists"] = True
        mtime = os.path.getmtime(calib_path)
        dt_utc = datetime.fromtimestamp(mtime, tz=timezone.utc)
        dt_local = datetime.fromtimestamp(mtime)
        result["calibrated_at"] = dt_utc.isoformat()
        result["calibrated_at_local"] = dt_local.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    return result


# ── Main guide printer ───────────────────────────────────────────────────────

def print_headless_guide(
    *,
    calib_path: str = "calibration.npz",
    tag_names_file: Optional[str] = None,
    tag_names: Optional[Dict[str, str]] = None,
    max_camera_index: int = 5,
) -> str:
    """Build and return a comprehensive headless status report.

    The report includes device information, available cameras, calibration
    status, loaded tag names, and AprilTag classification rules.

    Parameters
    ----------
    calib_path:
        Path to the calibration ``.npz`` file to inspect.
    tag_names_file:
        Optional path to a JSON file with custom tag names.
    tag_names:
        Optional pre-parsed tag-name mapping (from ``--tag-names``).
    max_camera_index:
        How many camera indices to probe (default 5).

    Returns
    -------
    str
        The full multi-line report text.
    """
    lines: list[str] = []
    sep = "=" * 60

    # ── Section: Device status ────────────────────────────────────────
    lines.append(sep)
    lines.append("  HEADLESS GUIDE – Device & Camera Report")
    lines.append(sep)
    lines.append("")

    status = get_device_status()
    lines.append("Device status:")
    for key, value in status.items():
        lines.append(f"  {key:<12s}: {value}")
    lines.append("")

    # ── Section: Available cameras ────────────────────────────────────
    lines.append(sep)
    lines.append("  Available cameras")
    lines.append(sep)

    cameras = discover_cameras(max_index=max_camera_index)
    if cameras:
        for cam in cameras:
            lines.append(
                f"  Camera {cam['index']}: "
                f"{cam['width']}x{cam['height']} "
                f"@ {cam['fps']:.1f} FPS  "
                f"(backend: {cam['backend']})"
            )
    else:
        lines.append("  No cameras detected.")
    lines.append("")

    # ── Section: Calibration ──────────────────────────────────────────
    lines.append(sep)
    lines.append("  Calibration status")
    lines.append(sep)

    calib = get_calibration_info(calib_path)
    if calib["exists"]:
        lines.append(f"  Calibration file : {calib['path']}")
        lines.append(f"  Calibrated at    : {calib['calibrated_at_local']}")
        lines.append(f"  (UTC)            : {calib['calibrated_at']}")
    else:
        lines.append(f"  No calibration found at: {calib['path']}")
    lines.append("")

    # ── Section: Tag names ────────────────────────────────────────────
    lines.append(sep)
    lines.append("  AprilTag custom names")
    lines.append(sep)

    merged_names: Dict[str, str] = dict(tag_names) if tag_names else {}
    if tag_names_file:
        if os.path.isfile(tag_names_file):
            try:
                file_names = load_tag_names_from_file(tag_names_file)
                # File entries take precedence over CLI entries
                merged_names.update(file_names)
                lines.append(f"  Loaded from file : {tag_names_file}")
            except (json.JSONDecodeError, TypeError) as exc:
                lines.append(
                    f"  WARNING: could not load tag names from "
                    f"{tag_names_file!r}: {exc}"
                )
        else:
            lines.append(f"  Tag names file not found: {tag_names_file}")

    if merged_names:
        def _sort_key(x: str) -> tuple:
            """Sort numeric IDs numerically, others lexicographically."""
            try:
                return (0, int(x))
            except ValueError:
                return (1, x)

        for tid in sorted(merged_names, key=_sort_key):
            category = ""
            if tid.isdigit():
                category = f" [{classify_tag(int(tid))}]"
            lines.append(f"  ID {tid:>3s}: {merged_names[tid]}{category}")
    else:
        lines.append("  No custom tag names defined.")
    lines.append("")

    # ── Section: Tag classification rules ─────────────────────────────
    lines.append(sep)
    lines.append("  AprilTag ID classification rules")
    lines.append(sep)
    lines.append(f"  IDs < {MOVABLE_ID_UPPER:<4d}       : movable (dynamic) objects")
    lines.append(
        f"  IDs {PACKAGE_ID_RANGE.start}–{PACKAGE_ID_RANGE.stop - 1}"
        f"         : package (transport)")
    lines.append(
        f"  IDs {TABLE_ID_RANGE.start}–{TABLE_ID_RANGE.stop - 1}"
        f"        : stationary table plane")
    lines.append(f"  IDs >= {MOVABLE_ID_UPPER} (other) : static markers")
    lines.append(sep)

    return "\n".join(lines)
