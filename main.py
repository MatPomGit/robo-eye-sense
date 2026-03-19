#!/usr/bin/env python3
"""CLI entry point for robo-vision.

robo-vision detects AprilTag fiducial markers, QR codes, and laser-pointer
spots in a live camera feed (or recorded video) and assigns each detected
object a persistent track ID across frames.

Run with a live camera (default: camera index 0, 640×480, all detectors on)::

    python main.py

Run with low quality (fast, low-power) – input is downscaled by 50 % before
detection::

    python main.py --quality low

Run with high quality (robust, motion-blur-resistant) – unsharp-mask
sharpening and Kalman-filter tracking for better performance under rapid
motion::

    python main.py --quality high

Run with the full Tkinter GUI (requires ``python3-tk``)::

    python main.py --gui

Run on a recorded video file::

    python main.py --source path/to/video.mp4

Run headless (no display window; print detections to stdout)::

    python main.py --headless

Record the live camera feed to an MP4 file::

    python main.py --record output.mp4

Record in headless mode (no display, but save video)::

    python main.py --headless --record output.mp4

Run the camera-offset calibration scenario – determine how much the camera
has moved relative to a reference position by comparing AprilTag markers::

    python main.py --mode offset

Run the scenario in headless mode (no display)::

    python main.py --mode offset --headless

Press **q** to quit the OpenCV display window (non-GUI, non-headless mode).
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import time
from pathlib import Path

from robo_vision import APP_NAME, RoboEyeDetector, __version__

from robo_vision.camera import Camera
from robo_vision.results import DetectionMode, DetectionType

import cv2  # noqa: E402  – imported after robo_vision to apply Qt font fix

# Maps the --quality CLI value to the internal DetectionMode enum.
_QUALITY_TO_DETECTION_MODE: dict[str, DetectionMode] = {
    "low": DetectionMode.FAST,
    "normal": DetectionMode.NORMAL,
    "high": DetectionMode.ROBUST,
}

logger = logging.getLogger("robo_vision")

# Ordered list of operating mode names (index+1 maps to mode name).
_MODES = ["basic", "offset", "slam", "calibration", "box", "pose", "follow"]

# Default physical side length of AprilTags in metres.
_DEFAULT_TAG_SIZE = 0.05


def _resolve_mode(value: str) -> str:
    """Convert a numeric mode selector (e.g. '1') to its string name.

    Accepts either a mode name (e.g. 'basic') or a 1-based index
    (e.g. '1' for 'basic', '2' for 'offset', …).
    """
    if value.isdigit():
        idx = int(value) - 1
        if 0 <= idx < len(_MODES):
            return _MODES[idx]
    return value

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"{APP_NAME} – real-time visual marker detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"{APP_NAME} {__version__}",
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="FILE",
        help="Path to a YAML configuration file. CLI arguments take precedence.",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index (0, 1, …) or path/URL to a video file or stream.",
    )
    parser.add_argument("--width", type=int, default=640, help="Frame width.")
    parser.add_argument("--height", type=int, default=480, help="Frame height.")
    parser.add_argument(
        "--no-apriltag", action="store_true", help="Disable AprilTag detection."
    )
    parser.add_argument(
        "--qr", action="store_true", help="Enable QR-code detection (disabled by default)."
    )
    parser.add_argument(
        "--laser", action="store_true", help="Enable laser-spot detection (disabled by default)."
    )
    parser.add_argument(
        "--laser-threshold",
        type=int,
        default=240,
        help="Lower brightness threshold for laser-spot detection (0–255).",
    )
    parser.add_argument(
        "--laser-threshold-max",
        type=int,
        default=255,
        help="Upper brightness threshold for laser-spot detection (0–255).",
    )
    parser.add_argument(
        "--laser-channels",
        default="rgb",
        help=(
            "Colour channels to analyse for laser detection. "
            "A combination of 'r', 'g', 'b' (default: 'rgb' – all channels). "
            "E.g. 'r' to detect only in the red channel."
        ),
    )
    parser.add_argument(
        "--quality",
        default="normal",
        choices=["low", "normal", "high"],
        help=(
            "Detection quality: "
            "'low' – downscales frames for low-power hardware (fast); "
            "'normal' – balanced default; "
            "'high' – sharpening + Kalman tracking for motion-blur resistance."
        ),
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without a display window; print detections to stdout.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch the full tkinter GUI (overrides --headless).",
    )
    parser.add_argument(
        "--record",
        default=None,
        metavar="FILE",
        help="Record the video feed to FILE (e.g. output.mp4).",
    )
    parser.add_argument(
        "--mode",
        type=_resolve_mode,
        choices=_MODES,
        default="basic",
        help=(
            "Operating mode – use the name or its 1-based index "
            "(1=basic, 2=offset, 3=slam, 4=calibration, 5=box, 6=pose, 7=follow).  "
            "'basic' – normal detection loop (default); "
            "'offset' – capture a reference frame, then compute the camera "
            "displacement vector after the camera has been moved. "
            "'slam' – incrementally build a marker map from the camera feed, "
            "estimating the 3-D pose of every visible AprilTag and the robot. "
            "'calibration' – compute camera intrinsics from chessboard images. "
            "'box' – detect box-like (cuboid) objects in real-time. "
            "'pose' – estimate 6-DoF pose of AprilTags using solvePnP. "
            "'follow' – actively track an AprilTag or box and generate "
            "control signals (replaces the old 'auto' mode)."
        ),
    )
    parser.add_argument(
        "--map-file",
        default=None,
        metavar="FILE",
        help=(
            "Path to a marker-map JSON file.  In 'slam' scenario mode the "
            "built map is saved here on exit.  When used outside a scenario "
            "the map is loaded for robot-pose estimation."
        ),
    )
    parser.add_argument(
        "--follow-marker",
        default=None,
        metavar="ID",
        help=(
            "In 'follow' mode, the ID of the AprilTag marker to "
            "follow.  When omitted the first visible marker is used."
        ),
    )
    parser.add_argument(
        "--follow-box",
        action="store_true",
        help=(
            "In 'follow' mode, fall back to box tracking when no "
            "AprilTags are visible."
        ),
    )
    parser.add_argument(
        "--target-distance",
        type=float,
        default=0.5,
        help="Desired distance to the target in metres (follow mode).",
    )
    parser.add_argument(
        "--chessboard-size",
        default="9x6",
        help=(
            "Inner corner dimensions of the calibration chessboard "
            "(COLSxROWS, e.g. '9x6')."
        ),
    )
    parser.add_argument(
        "--calib-output",
        default="calibration.npz",
        metavar="FILE",
        help="Output file for camera calibration data.",
    )
    parser.add_argument(
        "--tag-size",
        type=float,
        default=_DEFAULT_TAG_SIZE,
        help="Physical side length of AprilTags in metres (pose/follow modes).",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help=(
            "Print camera information (resolution, FPS, backend, and all "
            "available parameters) and exit."
        ),
    )
    parser.add_argument(
        "--tag-names",
        nargs="*",
        default=None,
        metavar="ID=NAME",
        help=(
            "Assign human-readable names to AprilTag IDs. "
            "Format: ID=NAME (e.g. --tag-names 1=box 2=table 5=wall)."
        ),
    )
    parser.add_argument(
        "--tag-names-file",
        default=None,
        metavar="FILE",
        help=(
            "Path to a JSON file with custom AprilTag names. "
            "The file must contain a JSON object mapping ID strings to "
            "names, e.g. {\"5\": \"package-A\", \"12\": \"table-left\"}. "
            "Entries from this file take precedence over --tag-names."
        ),
    )
    parser.add_argument(
        "--guide",
        action="store_true",
        help=(
            "Print a comprehensive headless guide: device status, "
            "available cameras, calibration info, AprilTag classification "
            "rules, and loaded tag names, then exit."
        ),
    )
    parser.add_argument(
        "--ros-status",
        action="store_true",
        help=(
            "Print the ROS2 connection status: whether rclpy is available, "
            "the node name, published and subscribed topics, then exit."
        ),
    )
    parser.add_argument(
        "--ros",
        action="store_true",
        help=(
            "Enable the ROS2 bridge during the detection loop. "
            "Requires rclpy (ROS2) to be installed and sourced. "
            "Publishes detections to /robo_vision/detections and robot "
            "pose to /robo_vision/robot_pose; subscribes to "
            "/robo_vision/config for live configuration overrides."
        ),
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Print execution times of key detection methods.",
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Disable the on-screen information overlay.",
    )
    return parser.parse_args(argv)


def _parse_tag_names(raw: list[str] | None) -> dict[str, str]:
    """Parse ``--tag-names`` arguments into a ``{id: name}`` dict."""
    if not raw:
        return {}
    mapping: dict[str, str] = {}
    for item in raw:
        if "=" not in item:
            logger.warning(
                "ignoring invalid --tag-names entry %r "
                "(expected ID=NAME format)", item,
            )
            continue
        tag_id, name = item.split("=", 1)
        mapping[tag_id.strip()] = name.strip()
    return mapping


def _display_mode_label(args: argparse.Namespace) -> str:
    """Return a human-readable label for the current display mode."""
    if args.gui:
        return "gui"
    if args.headless:
        return "headless"
    return "display"


def _enabled_detectors_label(args: argparse.Namespace) -> str:
    """Return a comma-separated list of enabled detector names."""
    names: list[str] = []
    if not args.no_apriltag:
        names.append("AprilTag")
    if args.qr:
        names.append("QR")
    if args.laser:
        names.append("Laser")
    return ", ".join(names) if names else "none"


def _validate_startup(args: argparse.Namespace) -> list[str]:
    """Run startup validation checks and return a list of warnings.

    Checks performed:
    1. Camera/video source accessibility (basic check).
    2. Required output directories are writable (recordings, maps).
    3. Critical parameter combinations (e.g. --mode slam without --tag-size).
    4. Display server availability when --gui is used.

    Parameters
    ----------
    args:
        Parsed CLI arguments.

    Returns
    -------
    list[str]
        List of warning messages (empty if all checks pass).
    """
    warnings_list: list[str] = []

    # 1. Check if video source file exists (for file-based sources)
    if not args.source.isdigit():
        source_path = Path(args.source)
        if not source_path.exists() and not args.source.startswith(("rtsp://", "http://", "https://")):
            warnings_list.append(
                f"Video source file not found: {args.source!r}"
            )

    # 2. Check that output directories are writable
    if args.record:
        record_dir = Path(args.record).parent
        if record_dir != Path(".") and not record_dir.exists():
            try:
                record_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Created recording directory: %s", record_dir)
            except OSError as exc:
                warnings_list.append(
                    f"Cannot create recording directory {record_dir}: {exc}"
                )
        elif record_dir.exists() and not os.access(str(record_dir), os.W_OK):
            warnings_list.append(
                f"Recording directory is not writable: {record_dir}"
            )

    if args.map_file:
        map_dir = Path(args.map_file).parent
        if map_dir != Path(".") and not map_dir.exists():
            try:
                map_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Created map directory: %s", map_dir)
            except OSError as exc:
                warnings_list.append(
                    f"Cannot create map directory {map_dir}: {exc}"
                )

    # Default maps directory
    maps_dir = Path(__file__).resolve().parent / "maps"
    if not maps_dir.exists():
        try:
            maps_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass  # Not critical – only needed if no --map-file is given

    # 3. Validate critical parameter combinations
    if args.mode == "slam" and args.tag_size == _DEFAULT_TAG_SIZE:
        logger.warning(
            "SLAM mode is active with default --tag-size (%s m). "
            "For accurate mapping, set --tag-size to the actual "
            "physical size of your AprilTags.",
            _DEFAULT_TAG_SIZE,
        )

    if args.mode in ("pose", "follow") and args.tag_size == _DEFAULT_TAG_SIZE:
        logger.warning(
            "%s mode uses default --tag-size (%s m). "
            "Set --tag-size for accurate distance estimation.",
            args.mode,
            _DEFAULT_TAG_SIZE,
        )

    # 4. Check display server availability for GUI mode
    if args.gui:
        display = os.environ.get("DISPLAY", "")
        wayland = os.environ.get("WAYLAND_DISPLAY", "")
        if not display and not wayland:
            if sys.platform.startswith("linux"):
                warnings_list.append(
                    "No display server detected (DISPLAY and WAYLAND_DISPLAY "
                    "are unset). GUI mode may fail on headless systems. "
                    "Consider using --headless instead."
                )

    return warnings_list


def main(argv: list[str] | None = None) -> int:  # noqa: C901
    args = _parse_args(argv)

    # ── Logging setup: INFO→stdout, WARNING+→stderr ───────────────────
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for h in root.handlers[:]:
        root.removeHandler(h)

    _fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    _h_out = logging.StreamHandler(sys.stdout)
    _h_out.setLevel(logging.DEBUG)
    _h_out.addFilter(lambda r: r.levelno < logging.WARNING)
    _h_out.setFormatter(_fmt)
    root.addHandler(_h_out)

    _h_err = logging.StreamHandler(sys.stderr)
    _h_err.setLevel(logging.WARNING)
    _h_err.setFormatter(_fmt)
    root.addHandler(_h_err)

    logger.info("%s %s", APP_NAME, __version__)

    # ── Profiling ─────────────────────────────────────────────────────
    if args.profile:
        from robo_vision.profiling import enable_profiling

        enable_profiling()
        logger.info("Profiling enabled")

    # ── Load YAML configuration (if provided) ────────────────────────
    if args.config:
        try:
            from robo_vision.config import load_config, merge_config_with_args

            config_data = load_config(args.config)
            cli_defaults = vars(_parse_args([]))
            args = merge_config_with_args(config_data, args, cli_defaults)
            logger.info("Config loaded     : %s", args.config)
        except (ImportError, FileNotFoundError) as exc:
            logger.warning("Could not load config: %s", exc)

    mode = _QUALITY_TO_DETECTION_MODE[args.quality]

    # ── Startup configuration summary ─────────────────────────────────
    display_label = _display_mode_label(args)
    logger.info("Display mode      : %s", display_label)
    logger.info("Quality           : %s", args.quality)
    logger.info("Detectors enabled : %s", _enabled_detectors_label(args))
    logger.info("Source            : %s", args.source)
    if args.mode and args.mode != "basic":
        logger.info("Scenario          : %s", args.mode)
    if args.record:
        logger.info("Record to         : %s", args.record)

    tag_names = _parse_tag_names(args.tag_names)

    # Merge tag names from file (file entries take precedence over CLI).
    if args.tag_names_file:
        from robo_vision.headless_guide import load_tag_names_from_file

        if os.path.isfile(args.tag_names_file):
            try:
                file_names = load_tag_names_from_file(args.tag_names_file)
                tag_names.update(file_names)
            except (json.JSONDecodeError, TypeError, OSError) as exc:
                logger.warning(
                    "could not load tag names from %r: %s",
                    args.tag_names_file, exc,
                )
        else:
            logger.warning(
                "tag names file not found: %r", args.tag_names_file,
            )

    # ── Guide mode ────────────────────────────────────────────────────
    if args.guide:
        from robo_vision.headless_guide import print_headless_guide

        report = print_headless_guide(
            calib_path=args.calib_output,
            tag_names_file=args.tag_names_file,
            tag_names=tag_names,
        )
        print(report)
        return 0

    # ── ROS status mode ───────────────────────────────────────────────
    if args.ros_status:
        from robo_vision.headless_guide import print_ros_status_report

        print(print_ros_status_report())
        return 0

    # ── Startup validation ────────────────────────────────────────────
    startup_warnings = _validate_startup(args)
    for w in startup_warnings:
        logger.warning(w)

    detector = RoboEyeDetector(
        enable_apriltag=not args.no_apriltag,
        enable_qr=args.qr,
        enable_laser=args.laser,
        mode=mode,
        laser_brightness_threshold=args.laser_threshold,
        laser_brightness_threshold_max=args.laser_threshold_max,
        tag_names=tag_names,
        laser_channels=args.laser_channels,
    )

    # Accept both integer camera indices (e.g. 0, 1) and string paths/URLs
    # (e.g. "/dev/video0", "rtsp://..."). argparse always gives us a string,
    # so we attempt an integer conversion first.
    try:
        source: int | str = int(args.source)
    except ValueError:
        source = args.source

    try:
        cam = Camera(source=source, width=args.width, height=args.height)
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1

    logger.info(
        "Camera opened     : %dx%d @ %.1f FPS",
        cam.actual_width, cam.actual_height, cam.actual_fps,
    )

    # ── Info mode ─────────────────────────────────────────────────────
    if args.info:
        info = cam.get_info()
        print(f"\n{'='*50}")
        print("Camera information")
        print(f"{'='*50}")
        for key, value in info.items():
            print(f"  {key:<16s}: {value}")
        print(f"{'='*50}")
        cam.release()
        return 0

    # ── Scenario mode ─────────────────────────────────────────────────────
    if args.mode == "offset":
        from robo_vision.offset_scenario import CameraOffsetScenario

        scenario = CameraOffsetScenario(camera=cam, detector=detector)

        # Optional recording during scenario
        recorder = None
        if args.record:
            from robo_vision.recorder import VideoRecorder

            recorder = VideoRecorder(
                args.record,
                width=cam.actual_width,
                height=cam.actual_height,
                fps=cam.actual_fps or 30.0,
            )

        try:
            with cam:
                if recorder is not None:
                    recorder.start()
                    logger.info("Recording to %s", args.record)

                logger.info("Starting offset scenario...")

                if args.headless:
                    # Headless scenario: capture reference immediately,
                    # then compute offset on next frame and exit.
                    # Type "ref" on stdin to re-capture the reference, or
                    # "quit" to exit.
                    logger.info("Capturing reference frame...")
                    ref = scenario.capture_reference()
                    april_ref = [
                        d
                        for d in ref
                        if d.detection_type == DetectionType.APRIL_TAG
                    ]
                    logger.info(
                        "Reference captured: %d AprilTag(s) detected.",
                        len(april_ref),
                    )
                    if not april_ref:
                        logger.warning(
                            "no AprilTags found in reference frame. "
                            "The offset will be (0, 0).",
                        )

                    logger.info("Capturing current frame and computing offset...")
                    result = scenario.compute_current_offset()

                    print(f"\n{'='*50}")
                    print("Camera-offset result")
                    print(f"{'='*50}")
                    print(f"Matched AprilTags : {result.matched_tags}")
                    dx, dy = result.offset
                    print(f"Offset (dx, dy)   : ({dx:+.1f}, {dy:+.1f}) px")
                    if result.per_tag_offsets:
                        print("\nPer-tag offsets:")
                        for tag_id, (tdx, tdy) in sorted(
                            result.per_tag_offsets.items()
                        ):
                            print(f"  tag {tag_id:>4s}: ({tdx:+.1f}, {tdy:+.1f}) px")
                    print(f"{'='*50}")

                    # Enter command loop for re-capture
                    print(
                        "Commands: 'ref' = new reference, "
                        "'offset' = compute offset, 'quit' = exit."
                    )
                    import select

                    try:
                        while True:
                            # Non-blocking check for stdin availability;
                            # fall through immediately when stdin is not a
                            # terminal (e.g. piped / EOF).
                            try:
                                ready, _, _ = select.select(
                                    [sys.stdin], [], [], 0.1
                                )
                            except (ValueError, OSError):
                                # stdin doesn't support fileno() (e.g. in
                                # test harness or when stdin is closed).
                                break
                            if not ready:
                                # No input available and stdin is still open –
                                # exit if there is nothing more to read
                                # (non-interactive pipe).
                                if sys.stdin.closed or not sys.stdin.isatty():
                                    break
                                continue
                            try:
                                line = sys.stdin.readline()
                            except EOFError:
                                break
                            if not line:
                                break
                            cmd = line.strip().lower()
                            if cmd in ("quit", "q", "exit"):
                                break
                            elif cmd == "ref":
                                logger.info("Capturing new reference frame...")
                                ref = scenario.capture_reference()
                                april_ref = [
                                    d
                                    for d in ref
                                    if d.detection_type == DetectionType.APRIL_TAG
                                ]
                                logger.info(
                                    "New reference captured: "
                                    "%d AprilTag(s) detected.",
                                    len(april_ref),
                                )
                            elif cmd == "offset":
                                logger.info("Computing offset...")
                                result = scenario.compute_current_offset()
                                dx, dy = result.offset
                                print(f"Matched: {result.matched_tags}  "
                                      f"Offset: ({dx:+.1f}, {dy:+.1f}) px")
                            elif cmd:
                                print(f"Unknown command: {cmd!r}")
                    except io.UnsupportedOperation:
                        # stdin may be a pseudofile in test harnesses or
                        # other environments that don't support fileno().
                        pass

                    logger.info("Offset scenario finished.")
                else:
                    input(
                        "Place the camera at the REFERENCE position with AprilTags "
                        "visible, then press Enter…"
                    )
                    logger.info("Capturing reference frame...")
                    ref = scenario.capture_reference()
                    april_ref = [
                        d
                        for d in ref
                        if d.detection_type == DetectionType.APRIL_TAG
                    ]
                    logger.info(
                        "Reference captured: %d AprilTag(s) detected.",
                        len(april_ref),
                    )
                    if not april_ref:
                        logger.warning(
                            "no AprilTags found in reference frame. "
                            "The offset will be (0, 0).",
                        )

                    input(
                        "Move the camera to the NEW position, then press Enter…"
                    )
                    logger.info("Capturing current frame and computing offset...")
                    result = scenario.compute_current_offset()

                    print(f"\n{'='*50}")
                    print("Camera-offset result")
                    print(f"{'='*50}")
                    print(f"Matched AprilTags : {result.matched_tags}")
                    dx, dy = result.offset
                    print(f"Offset (dx, dy)   : ({dx:+.1f}, {dy:+.1f}) px")
                    if result.per_tag_offsets:
                        print("\nPer-tag offsets:")
                        for tag_id, (tdx, tdy) in sorted(
                            result.per_tag_offsets.items()
                        ):
                            print(f"  tag {tag_id:>4s}: ({tdx:+.1f}, {tdy:+.1f}) px")
                    print(f"{'='*50}")
                    logger.info("Offset scenario finished.")
        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
        except RuntimeError as exc:
            logger.error("%s", exc)
            return 1
        finally:
            if recorder is not None:
                recorder.stop()
                logger.info("Recording saved to %s", args.record)
        return 0

    # ── SLAM scenario mode ────────────────────────────────────────────────
    if args.mode == "slam":
        from robo_vision.marker_map import SlamCalibrator

        # Convert tag size from metres (CLI default) to centimetres
        tag_size_cm = args.tag_size * 100.0

        # Optionally load camera calibration from a .npz file
        camera_matrix = None
        if args.calib_output:
            if os.path.isfile(args.calib_output):
                try:
                    import numpy as _np
                    _cal = _np.load(args.calib_output)
                    camera_matrix = _cal["camera_matrix"].tolist()
                    logger.info("Calibration loaded : %s", args.calib_output)
                except (OSError, KeyError) as _exc:
                    logger.warning(
                        "could not load calibration from %r: %s",
                        args.calib_output, _exc,
                    )

        calibrator = SlamCalibrator(
            tag_size_cm=tag_size_cm,
            camera_matrix=camera_matrix,
            frame_size=(cam.actual_width, cam.actual_height),
        )

        recorder = None
        if args.record:
            from robo_vision.recorder import VideoRecorder

            recorder = VideoRecorder(
                args.record,
                width=cam.actual_width,
                height=cam.actual_height,
                fps=cam.actual_fps or 30.0,
            )

        try:
            with cam:
                if recorder is not None:
                    recorder.start()
                    logger.info("Recording to %s", args.record)

                logger.info("Starting SLAM calibration...")
                print("Scanning for AprilTag markers. Press q to stop (display mode).")

                frame_idx = 0
                fps_counter = 0
                fps_display = 0.0
                t_fps = time.perf_counter()

                while True:
                    frame = cam.read()
                    if frame is None:
                        break

                    try:
                        map_size_before = len(calibrator.marker_map)
                        detections = detector.process_frame(frame)
                        robot_pose = calibrator.process_detections(detections)
                    except cv2.error as exc:
                        logger.error("OpenCV error during SLAM detection: %s", exc)
                        continue
                    except (ValueError, TypeError) as exc:
                        logger.error("SLAM detection error: %s", exc)
                        continue
                    frame_idx += 1

                    # FPS calculation
                    fps_counter += 1
                    t_now = time.perf_counter()
                    elapsed = t_now - t_fps
                    if elapsed >= 1.0:
                        fps_display = fps_counter / elapsed
                        fps_counter = 0
                        t_fps = t_now

                    new_markers = len(calibrator.marker_map) - map_size_before

                    if args.headless:
                        if robot_pose.visible_markers > 0:
                            rx, ry, rz = robot_pose.position
                            roll, pitch, yaw = robot_pose.orientation
                            err_str = (
                                f"  reproj_err={robot_pose.reprojection_error:.2f}px"
                                if robot_pose.reprojection_error is not None
                                else ""
                            )
                            new_str = (
                                f"  +{new_markers} new"
                                if new_markers > 0
                                else ""
                            )
                            # Collect visible marker IDs
                            vis_ids = sorted({
                                d.identifier
                                for d in detections
                                if d.identifier is not None
                            })
                            ids_str = (
                                f"  ids=[{', '.join(vis_ids)}]"
                                if vis_ids
                                else ""
                            )
                            print(
                                f"[frame {frame_idx}] "
                                f"robot=({rx:+.1f}, {ry:+.1f}, {rz:+.1f}) cm  "
                                f"yaw={yaw:+.1f}°  "
                                f"map={len(calibrator.marker_map)}"
                                f"{new_str}  "
                                f"visible={robot_pose.visible_markers}"
                                f"{ids_str}"
                                f"{err_str}  "
                                f"FPS: {fps_display:.1f}"
                            )
                        else:
                            print(
                                f"[frame {frame_idx}] "
                                f"No markers visible  "
                                f"map={len(calibrator.marker_map)}  "
                                f"FPS: {fps_display:.1f}"
                            )
                        if recorder is not None:
                            vis = detector.draw_detections(frame.copy(), detections)
                            recorder.write_frame(vis)
                    else:
                        vis = detector.draw_detections(frame.copy(), detections)
                        # Overlay SLAM info
                        rx, ry, rz = robot_pose.position
                        roll, pitch, yaw = robot_pose.orientation
                        info = (
                            f"SLAM: map={len(calibrator.marker_map)} tags  "
                            f"robot=({rx:+.1f},{ry:+.1f},{rz:+.1f}) cm  "
                            f"yaw={yaw:+.1f}°  "
                            f"FPS: {fps_display:.1f}"
                        )
                        cv2.putText(
                            vis, info, (8, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 255), 1, cv2.LINE_AA,
                        )
                        if recorder is not None:
                            recorder.write_frame(vis)
                        cv2.imshow("RoboEyeSense – SLAM", vis)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

                logger.info("Stream ended.")

                # Save marker map
                mmap = calibrator.marker_map
                if args.map_file:
                    map_path = args.map_file
                else:
                    maps_dir = Path(__file__).resolve().parent / "maps"
                    maps_dir.mkdir(exist_ok=True)
                    map_path = str(maps_dir / "marker_map.json")
                mmap.save(map_path)
                print(f"\n{'='*50}")
                print("SLAM calibration result")
                print(f"{'='*50}")
                print(f"Frames processed  : {calibrator.frame_count}")
                print(f"Tag size          : {tag_size_cm:.1f} cm")
                print(f"Markers in map    : {len(mmap)}")
                for m in mmap.markers():
                    px, py, pz = m.position
                    r, p, y = m.orientation
                    print(
                        f"  tag {m.marker_id:>4s}: "
                        f"pos=({px:+.1f}, {py:+.1f}, {pz:+.1f}) cm  "
                        f"ori=({r:+.1f}, {p:+.1f}, {y:+.1f})°  "
                        f"obs={m.observations}"
                    )
                print(f"Map saved to      : {map_path}")
                print(f"{'='*50}")
        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
        except RuntimeError as exc:
            logger.error("%s", exc)
            return 1
        finally:
            if recorder is not None:
                recorder.stop()
                logger.info("Recording saved to %s", args.record)
            if not args.headless:
                cv2.destroyAllWindows()
        return 0

    # ── New operational modes (calibration / box / pose / follow) ────────
    if args.mode in ("calibration", "box", "pose", "follow"):
        from modes import BoxMode, CalibrationMode, FollowMode, PoseMode

        if args.mode == "calibration":
            try:
                cols, rows = (int(x) for x in args.chessboard_size.split("x"))
            except ValueError:
                logger.error(
                    "invalid --chessboard-size %r "
                    "(expected COLSxROWS, e.g. '9x6')",
                    args.chessboard_size,
                )
                return 1
            active_mode = CalibrationMode(
                chessboard_size=(cols, rows),
                output_path=args.calib_output,
            )
        elif args.mode == "box":
            active_mode = BoxMode()
        elif args.mode == "pose":
            active_mode = PoseMode(
                tag_size=args.tag_size,
                calibration_path=args.calib_output,
            )
        elif args.mode == "follow":
            active_mode = FollowMode(
                follow_marker=args.follow_marker,
                follow_box=args.follow_box,
                target_distance=args.target_distance,
                tag_size=args.tag_size,
                calibration_path=args.calib_output,
            )

        recorder = None
        if args.record:
            from robo_vision.recorder import VideoRecorder

            recorder = VideoRecorder(
                args.record,
                width=cam.actual_width,
                height=cam.actual_height,
                fps=cam.actual_fps or 30.0,
            )

        fps_counter = 0
        fps_display = 0.0
        t_fps = time.perf_counter()
        frame_idx = 0

        logger.info("Starting %s mode...", args.mode)
        if not args.headless:
            print("Press q in the display window to quit.")

        try:
            with cam:
                if recorder is not None:
                    recorder.start()
                    logger.info("Recording to %s", args.record)

                while True:
                    frame = cam.read()
                    if frame is None:
                        break

                    frame_idx += 1

                    # FPS calculation
                    fps_counter += 1
                    t_now = time.perf_counter()
                    elapsed = t_now - t_fps
                    if elapsed >= 1.0:
                        fps_display = fps_counter / elapsed
                        fps_counter = 0
                        t_fps = t_now

                    key = -1
                    if not args.headless:
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            break

                    ctx = {
                        "headless": args.headless,
                        "key": key,
                        "frame_idx": frame_idx,
                        "fps": fps_display,
                    }

                    try:
                        vis = active_mode.run(frame, ctx)
                    except cv2.error as exc:
                        logger.error("OpenCV error in %s mode: %s", args.mode, exc)
                        continue
                    except (ValueError, TypeError) as exc:
                        logger.error("Error in %s mode: %s", args.mode, exc)
                        continue

                    if not args.headless:
                        cv2.imshow(f"RoboEyeSense – {args.mode}", vis)

                    if recorder is not None:
                        recorder.write_frame(vis)

                logger.info("Stream ended. Total frames: %d", frame_idx)
        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
        except RuntimeError as exc:
            logger.error("%s", exc)
            return 1
        finally:
            if recorder is not None:
                recorder.stop()
                logger.info("Recording saved to %s", args.record)
            if not args.headless:
                cv2.destroyAllWindows()
        return 0

    # ── GUI mode ──────────────────────────────────────────────────────────
    if args.gui:
        try:
            import tkinter as tk
        except ImportError:  # pragma: no cover
            logger.error(
                "tkinter is not available. "
                "Install it (e.g. sudo apt install python3-tk) to use --gui.",
            )
            return 1
        from robo_vision.gui import RoboEyeSenseApp

        logger.info("Launching GUI...")
        root = tk.Tk()
        with cam:
            app = RoboEyeSenseApp(
                root, cam, detector,
                initial_record_path=args.record,
            )
            app.run()
        return 0

    # ── Headless / cv2.imshow mode ────────────────────────────────────────
    recorder = None
    if args.record:
        from robo_vision.recorder import VideoRecorder

        recorder = VideoRecorder(
            args.record,
            width=cam.actual_width,
            height=cam.actual_height,
            fps=cam.actual_fps or 30.0,
        )

    # ── ROS2 bridge (optional) ────────────────────────────────────────
    ros_bridge = None
    if args.ros:
        from robo_vision.ros2_bridge import ROS2Bridge

        ros_bridge = ROS2Bridge()
        if ros_bridge.available:
            started = ros_bridge.start()
            if started:
                logger.info("ROS2 bridge started.")
            else:
                logger.warning("ROS2 bridge failed to start.")
        else:
            logger.warning(
                "rclpy not available – ROS2 bridge disabled. "
                "Install ROS2 and source the workspace to enable it."
            )

    # Build list of enabled detector names for overlay
    _det_names: list[str] = []
    if not args.no_apriltag:
        _det_names.append("AprilTags")
    if args.qr:
        _det_names.append("QR codes")
    if args.laser:
        _det_names.append("Laser spots")

    overlay = None
    if not getattr(args, "no_overlay", False):
        from robo_vision.overlay import OverlayRenderer

        overlay = OverlayRenderer(
            enabled=True,
            mode=args.mode,
            quality=args.quality,
            enabled_detectors=_det_names,
        )

    fps_counter = 0
    fps_display = 0.0
    t_fps = time.perf_counter()
    frame_total = 0
    _rec_file_counter = 0

    logger.info("Starting detection loop...")
    if not args.headless:
        logger.info("Press q to quit, r to toggle recording.")

    try:
        with cam:
            if recorder is not None:
                recorder.start()
                logger.info("Recording to %s", args.record)
                if overlay:
                    overlay.set_recording(True)

            while True:
                frame = cam.read()
                if frame is None:
                    break

                try:
                    detections = detector.process_frame(frame)
                except cv2.error as exc:
                    logger.error("OpenCV error during detection: %s", exc)
                    continue
                except (ValueError, TypeError) as exc:
                    logger.error("Detection error: %s", exc)
                    continue
                frame_total += 1

                # FPS calculation
                fps_counter += 1
                t_now = time.perf_counter()
                elapsed = t_now - t_fps
                if elapsed >= 1.0:
                    fps_display = fps_counter / elapsed
                    fps_counter = 0
                    t_fps = t_now

                # Publish to ROS2 bridge (if active)
                if ros_bridge is not None and ros_bridge.is_running:
                    ros_bridge.publish_detections(detections)
                    # Consume any pending config overrides sent via the ROS topic.
                    # Full runtime reconfiguration (quality, detector toggles, etc.)
                    # is not yet implemented – the received dict is logged so that
                    # operators can verify overrides are reaching this node.
                    # TODO: apply pending_cfg to detector/mode at runtime.
                    pending_cfg = ros_bridge.get_pending_config()
                    if pending_cfg is not None:
                        logger.info("ROS2 config override received: %s", pending_cfg)

                if args.headless:
                    if detections:
                        for d in detections:
                            print(f"[frame {frame_total}] {d}  (FPS: {fps_display:.1f})")
                    else:
                        print(
                            f"[frame {frame_total}] "
                            f"No detections  (FPS: {fps_display:.1f})"
                        )
                    if recorder is not None:
                        vis = detector.draw_detections(frame.copy(), detections)
                        recorder.write_frame(vis)
                else:
                    vis = detector.draw_detections(frame.copy(), detections)
                    if overlay:
                        overlay.draw(vis, detections, fps_display)
                    else:
                        cv2.putText(
                            vis,
                            f"FPS: {fps_display:.1f}",
                            (8, 24),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )
                    if recorder is not None:
                        recorder.write_frame(vis)
                    cv2.imshow("RoboEyeSense", vis)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == ord("r"):
                        # Toggle recording
                        if recorder is not None and recorder.is_recording:
                            recorder.stop()
                            logger.info("Recording stopped: %s", recorder.output_path)
                            if overlay:
                                overlay.set_recording(False)
                            recorder = None
                        else:
                            _rec_file_counter += 1
                            from robo_vision.recorder import VideoRecorder

                            rec_path = f"recording_{_rec_file_counter}.mp4"
                            recorder = VideoRecorder(
                                rec_path,
                                width=cam.actual_width,
                                height=cam.actual_height,
                                fps=cam.actual_fps or 30.0,
                            )
                            recorder.start()
                            logger.info("Recording started: %s", rec_path)
                            if overlay:
                                overlay.set_recording(True)

            logger.info("Stream ended. Total frames processed: %d", frame_total)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        if ros_bridge is not None and ros_bridge.is_running:
            ros_bridge.stop()
            logger.info("ROS2 bridge stopped.")
        if recorder is not None:
            recorder.stop()
            logger.info("Recording saved to %s", recorder.output_path)
        if not args.headless:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
