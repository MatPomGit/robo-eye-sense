#!/usr/bin/env python3
"""CLI entry point for robo-eye-sense.

Run with a live camera::

    python main.py

Run in fast (low-power) mode::

    python main.py --mode fast

Run in robust (motion-blur-resistant) mode::

    python main.py --mode robust

Run with the full GUI window::

    python main.py --gui

Run on a recorded video file::

    python main.py --source path/to/video.mp4

Run headless (no GUI window, print detections to stdout)::

    python main.py --headless

Press **q** to quit the display window (non-GUI mode).
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2

from robo_eye_sense import APP_NAME, RoboEyeDetector, __version__
from robo_eye_sense.camera import Camera
from robo_eye_sense.results import DetectionMode


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

    # Accept both integer camera indices and string paths/URLs
    try:
        source: int | str = int(args.source)
    except ValueError:
        source = args.source

    try:
        cam = Camera(source=source, width=args.width, height=args.height)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

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
