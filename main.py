#!/usr/bin/env python3
"""CLI entry point for robo-eye-sense.

robo-eye-sense detects AprilTag fiducial markers, QR codes, and laser-pointer
spots in a live camera feed (or recorded video) and assigns each detected
object a persistent track ID across frames.

Run with a live camera (default: camera index 0, 640×480, all detectors on)::

    python main.py

Run in fast (low-power) mode – input is downscaled by 50 % before detection::

    python main.py --mode fast

Run in robust (motion-blur-resistant) mode – unsharp-mask sharpening and
Kalman-filter tracking for better performance under rapid motion::

    python main.py --mode robust

Run with the full Tkinter GUI (requires ``python3-tk``)::

    python main.py --gui

Run on a recorded video file::

    python main.py --source path/to/video.mp4

Run headless (no display window; print detections to stdout)::

    python main.py --headless

Run the camera-offset calibration scenario – determine how much the camera
has moved relative to a reference position by comparing AprilTag markers::

    python main.py --scenario offset

Press **q** to quit the OpenCV display window (non-GUI, non-headless mode).
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2

from robo_eye_sense import APP_NAME, RoboEyeDetector, __version__
from robo_eye_sense.camera import Camera
from robo_eye_sense.results import DetectionMode, DetectionType


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
        "--no-qr", action="store_true", help="Disable QR-code detection."
    )
    parser.add_argument(
        "--no-laser", action="store_true", help="Disable laser-spot detection."
    )
    parser.add_argument(
        "--laser-threshold",
        type=int,
        default=240,
        help="Brightness threshold for laser-spot detection (0–255).",
    )
    parser.add_argument(
        "--mode",
        default="normal",
        choices=["normal", "fast", "robust"],
        help=(
            "Operating mode: "
            "'normal' – balanced default; "
            "'fast' – downscales frames for low-power hardware; "
            "'robust' – sharpening + Kalman tracking for motion-blur resistance."
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
        "--scenario",
        choices=["offset"],
        default=None,
        help=(
            "Run a predefined scenario instead of the normal detection loop. "
            "'offset' – capture a reference frame, then compute the camera "
            "displacement vector after the camera has been moved."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:  # noqa: C901
    args = _parse_args(argv)

    print(f"{APP_NAME} {__version__}")

    mode = DetectionMode(args.mode)

    detector = RoboEyeDetector(
        enable_apriltag=not args.no_apriltag,
        enable_qr=not args.no_qr,
        enable_laser=not args.no_laser,
        mode=mode,
        laser_brightness_threshold=args.laser_threshold,
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
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # ── Scenario mode ─────────────────────────────────────────────────────
    if args.scenario == "offset":
        from robo_eye_sense.offset_scenario import CameraOffsetScenario

        scenario = CameraOffsetScenario(camera=cam, detector=detector)
        try:
            with cam:
                input(
                    "Place the camera at the REFERENCE position with AprilTags "
                    "visible, then press Enter…"
                )
                ref = scenario.capture_reference()
                april_ref = [
                    d for d in ref if d.detection_type == DetectionType.APRIL_TAG
                ]
                print(
                    f"Reference captured: {len(april_ref)} AprilTag(s) detected."
                )
                if not april_ref:
                    print(
                        "WARNING: no AprilTags found in reference frame. "
                        "The offset will be (0, 0).",
                        file=sys.stderr,
                    )

                input(
                    "Move the camera to the NEW position, then press Enter…"
                )
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
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        return 0

    # ── GUI mode ──────────────────────────────────────────────────────────
    if args.gui:
        try:
            import tkinter as tk
        except ImportError:  # pragma: no cover
            print(
                "ERROR: tkinter is not available. "
                "Install it (e.g. sudo apt install python3-tk) to use --gui.",
                file=sys.stderr,
            )
            return 1
        from robo_eye_sense.gui import RoboEyeSenseApp

        root = tk.Tk()
        with cam:
            app = RoboEyeSenseApp(root, cam, detector)
            app.run()
        return 0

    # ── Headless / cv2.imshow mode ────────────────────────────────────────
    fps_counter = 0
    fps_display = 0.0
    t_fps = time.perf_counter()

    try:
        with cam:
            while True:
                frame = cam.read()
                if frame is None:
                    break

                detections = detector.process_frame(frame)

                # FPS calculation
                fps_counter += 1
                t_now = time.perf_counter()
                elapsed = t_now - t_fps
                if elapsed >= 1.0:
                    fps_display = fps_counter / elapsed
                    fps_counter = 0
                    t_fps = t_now

                if args.headless:
                    for d in detections:
                        print(d)
                else:
                    vis = detector.draw_detections(frame.copy(), detections)
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
                    cv2.imshow("RoboEyeSense", vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    finally:
        if not args.headless:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
