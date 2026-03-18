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

Record the live camera feed to an MP4 file::

    python main.py --record output.mp4

Record in headless mode (no display, but save video)::

    python main.py --headless --record output.mp4

Run the camera-offset calibration scenario – determine how much the camera
has moved relative to a reference position by comparing AprilTag markers::

    python main.py --scenario offset

Run the scenario in headless mode (no display)::

    python main.py --scenario offset --headless

Press **q** to quit the OpenCV display window (non-GUI, non-headless mode).
"""

from __future__ import annotations

import argparse
import sys
import time

from robo_eye_sense import APP_NAME, RoboEyeDetector, __version__
from robo_eye_sense.camera import Camera
from robo_eye_sense.results import DetectionMode, DetectionType

import cv2  # noqa: E402  – imported after robo_eye_sense to apply Qt font fix


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
        "--record",
        default=None,
        metavar="FILE",
        help="Record the video feed to FILE (e.g. output.mp4).",
    )
    parser.add_argument(
        "--scenario",
        choices=["offset", "slam"],
        default=None,
        help=(
            "Run a predefined scenario instead of the normal detection loop. "
            "'offset' – capture a reference frame, then compute the camera "
            "displacement vector after the camera has been moved. "
            "'slam' – incrementally build a marker map from the camera feed, "
            "estimating the 3-D pose of every visible AprilTag and the robot."
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
    return parser.parse_args(argv)


def _parse_tag_names(raw: list[str] | None) -> dict[str, str]:
    """Parse ``--tag-names`` arguments into a ``{id: name}`` dict."""
    if not raw:
        return {}
    mapping: dict[str, str] = {}
    for item in raw:
        if "=" not in item:
            print(
                f"WARNING: ignoring invalid --tag-names entry {item!r} "
                "(expected ID=NAME format)",
                file=sys.stderr,
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


def main(argv: list[str] | None = None) -> int:  # noqa: C901
    args = _parse_args(argv)

    print(f"{APP_NAME} {__version__}")

    mode = DetectionMode(args.mode)

    # ── Startup configuration summary ─────────────────────────────────
    display_label = _display_mode_label(args)
    print(f"Display mode      : {display_label}")
    print(f"Detection mode    : {args.mode}")
    print(f"Detectors enabled : {_enabled_detectors_label(args)}")
    print(f"Source            : {args.source}")
    if args.scenario:
        print(f"Scenario          : {args.scenario}")
    if args.record:
        print(f"Record to         : {args.record}")

    tag_names = _parse_tag_names(args.tag_names)

    detector = RoboEyeDetector(
        enable_apriltag=not args.no_apriltag,
        enable_qr=args.qr,
        enable_laser=args.laser,
        mode=mode,
        laser_brightness_threshold=args.laser_threshold,
        laser_brightness_threshold_max=args.laser_threshold_max,
        tag_names=tag_names,
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

    print(
        f"Camera opened     : {cam.actual_width}x{cam.actual_height} "
        f"@ {cam.actual_fps:.1f} FPS"
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
    if args.scenario == "offset":
        from robo_eye_sense.offset_scenario import CameraOffsetScenario

        scenario = CameraOffsetScenario(camera=cam, detector=detector)

        # Optional recording during scenario
        recorder = None
        if args.record:
            from robo_eye_sense.recorder import VideoRecorder

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
                    print(f"Recording to {args.record}")

                print("Starting offset scenario...")

                if args.headless:
                    # Headless scenario: capture reference immediately,
                    # then compute offset on next frame and exit.
                    # Type "ref" on stdin to re-capture the reference, or
                    # "quit" to exit.
                    print("Capturing reference frame...")
                    ref = scenario.capture_reference()
                    april_ref = [
                        d
                        for d in ref
                        if d.detection_type == DetectionType.APRIL_TAG
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

                    print("Capturing current frame and computing offset...")
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

                    while True:
                        # Non-blocking check for stdin availability;
                        # fall through immediately when stdin is not a
                        # terminal (e.g. piped / EOF).
                        ready, _, _ = select.select([sys.stdin], [], [], 0.1)
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
                            print("Capturing new reference frame...")
                            ref = scenario.capture_reference()
                            april_ref = [
                                d
                                for d in ref
                                if d.detection_type == DetectionType.APRIL_TAG
                            ]
                            print(
                                f"New reference captured: "
                                f"{len(april_ref)} AprilTag(s) detected."
                            )
                        elif cmd == "offset":
                            print("Computing offset...")
                            result = scenario.compute_current_offset()
                            dx, dy = result.offset
                            print(f"Matched: {result.matched_tags}  "
                                  f"Offset: ({dx:+.1f}, {dy:+.1f}) px")
                        elif cmd:
                            print(f"Unknown command: {cmd!r}")

                    print("Offset scenario finished.")
                else:
                    input(
                        "Place the camera at the REFERENCE position with AprilTags "
                        "visible, then press Enter…"
                    )
                    print("Capturing reference frame...")
                    ref = scenario.capture_reference()
                    april_ref = [
                        d
                        for d in ref
                        if d.detection_type == DetectionType.APRIL_TAG
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
                    print("Capturing current frame and computing offset...")
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
                    print("Offset scenario finished.")
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        finally:
            if recorder is not None:
                recorder.stop()
                print(f"Recording saved to {args.record}")
        return 0

    # ── SLAM scenario mode ────────────────────────────────────────────────
    if args.scenario == "slam":
        from robo_eye_sense.marker_map import SlamCalibrator

        calibrator = SlamCalibrator(tag_size_cm=5.0)

        recorder = None
        if args.record:
            from robo_eye_sense.recorder import VideoRecorder

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
                    print(f"Recording to {args.record}")

                print("Starting SLAM calibration...")
                print("Scanning for AprilTag markers. Press q to stop (display mode).")

                frame_idx = 0
                while True:
                    frame = cam.read()
                    if frame is None:
                        break

                    detections = detector.process_frame(frame)
                    robot_pose = calibrator.process_detections(detections)
                    frame_idx += 1

                    if args.headless:
                        if robot_pose.visible_markers > 0:
                            rx, ry, rz = robot_pose.position
                            print(
                                f"[frame {frame_idx}] "
                                f"robot=({rx:+.1f}, {ry:+.1f}, {rz:+.1f}) cm  "
                                f"markers_in_map={len(calibrator.marker_map)}  "
                                f"visible={robot_pose.visible_markers}"
                            )
                        else:
                            print(
                                f"[frame {frame_idx}] "
                                f"No markers visible  "
                                f"markers_in_map={len(calibrator.marker_map)}"
                            )
                        if recorder is not None:
                            vis = detector.draw_detections(frame.copy(), detections)
                            recorder.write_frame(vis)
                    else:
                        vis = detector.draw_detections(frame.copy(), detections)
                        # Overlay SLAM info
                        rx, ry, rz = robot_pose.position
                        info = (
                            f"SLAM: map={len(calibrator.marker_map)} tags  "
                            f"robot=({rx:+.1f},{ry:+.1f},{rz:+.1f})"
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

                print("Stream ended.")

                # Save marker map
                mmap = calibrator.marker_map
                map_path = args.map_file or "marker_map.json"
                mmap.save(map_path)
                print(f"\n{'='*50}")
                print("SLAM calibration result")
                print(f"{'='*50}")
                print(f"Frames processed  : {calibrator.frame_count}")
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
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        finally:
            if recorder is not None:
                recorder.stop()
                print(f"Recording saved to {args.record}")
            if not args.headless:
                cv2.destroyAllWindows()
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

        print("Launching GUI...")
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
        from robo_eye_sense.recorder import VideoRecorder

        recorder = VideoRecorder(
            args.record,
            width=cam.actual_width,
            height=cam.actual_height,
            fps=cam.actual_fps or 30.0,
        )

    fps_counter = 0
    fps_display = 0.0
    t_fps = time.perf_counter()
    frame_total = 0

    print("Starting detection loop...")
    if not args.headless:
        print("Press q in the display window to quit.")

    try:
        with cam:
            if recorder is not None:
                recorder.start()
                print(f"Recording to {args.record}")

            while True:
                frame = cam.read()
                if frame is None:
                    break

                detections = detector.process_frame(frame)
                frame_total += 1

                # FPS calculation
                fps_counter += 1
                t_now = time.perf_counter()
                elapsed = t_now - t_fps
                if elapsed >= 1.0:
                    fps_display = fps_counter / elapsed
                    fps_counter = 0
                    t_fps = t_now

                if args.headless:
                    if detections:
                        for d in detections:
                            print(f"[frame {frame_total}] {d}")
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
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            print(f"Stream ended. Total frames processed: {frame_total}")
    finally:
        if recorder is not None:
            recorder.stop()
            print(f"Recording saved to {args.record}")
        if not args.headless:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
