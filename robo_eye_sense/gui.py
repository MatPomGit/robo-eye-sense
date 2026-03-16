"""Tkinter-based GUI application for RoboEyeSense.

Layout
------
+-------------+---------------------------+-----------+
|  CONTROLS   |       VIDEO FEED          |   INFO    |
|  ---------- |  (annotated frame)        | --------- |
|  Mode       |                           | Camera    |
|  [combo]    |                           | FPS / W×H |
|  ---------- |                           | --------- |
|  Detectors  |                           | Objects   |
|  [x] April  |                           | (list)    |
|  [x] QR     |                           |           |
|  [x] Laser  |                           |           |
|  ---------- |                           |           |
|  Parameters |                           |           |
|  Threshold  |                           |           |
|  Target area|                           |           |
|  Sensitivity|                           |           |
|  Decimate   |                           |           |
+-------------+---------------------------+-----------+

Usage::

    import tkinter as tk
    from robo_eye_sense.camera import Camera
    from robo_eye_sense.detector import RoboEyeDetector
    from robo_eye_sense.gui import RoboEyeSenseApp

    root = tk.Tk()
    cam = Camera()
    detector = RoboEyeDetector()
    app = RoboEyeSenseApp(root, cam, detector)
    app.run()
"""

from __future__ import annotations

import time
import tkinter as tk
from tkinter import ttk
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

from .camera import Camera
from .detector import RoboEyeDetector, _compute_orientation
from .results import Detection, DetectionMode, DetectionType

# How often (milliseconds) the frame-update callback is rescheduled
_UPDATE_INTERVAL_MS = 16  # ~60 Hz ceiling; actual rate is camera-limited

# Human-readable labels shown in the mode combobox
_MODE_DISPLAY: dict[str, DetectionMode] = {
    "Normal": DetectionMode.NORMAL,
    "Fast (low-power)": DetectionMode.FAST,
    "Robust (motion-blur resistant)": DetectionMode.ROBUST,
}
_MODE_DISPLAY_INV: dict[DetectionMode, str] = {v: k for k, v in _MODE_DISPLAY.items()}


class RoboEyeSenseApp:
    """Tkinter application window for RoboEyeSense.

    Parameters
    ----------
    root:
        Tkinter root window (``tk.Tk()`` instance).
    camera:
        Opened :class:`~robo_eye_sense.camera.Camera` instance.
    detector:
        Configured :class:`~robo_eye_sense.detector.RoboEyeDetector` instance.
    """

    def __init__(
        self,
        root: tk.Tk,
        camera: Camera,
        detector: RoboEyeDetector,
    ) -> None:
        self.root = root
        self.camera = camera
        self.detector = detector

        self.root.title("RoboEyeSense")
        self.root.resizable(True, True)

        # ── State variables ──────────────────────────────────────────────
        self._running = True
        self._fps_counter = 0
        self._fps_display = 0.0
        self._t_fps = time.perf_counter()
        self._last_detections: List[Detection] = []
        self._canvas_image_id: Optional[int] = None

        # Detection-mode flags (mirror the detector's live state)
        self._enable_april = tk.BooleanVar(value=detector._april_detector is not None)
        self._enable_qr = tk.BooleanVar(value=detector._qr_detector is not None)
        self._enable_laser = tk.BooleanVar(value=detector._laser_detector is not None)

        # Program mode
        initial_mode_label = _MODE_DISPLAY_INV.get(detector.mode, "Normal")
        self._mode_var = tk.StringVar(value=initial_mode_label)

        # Tunable parameters
        _init_threshold = (
            detector._laser_detector.brightness_threshold
            if detector._laser_detector is not None
            else 240
        )
        _init_target_area = (
            detector._laser_detector.target_area
            if detector._laser_detector is not None
            else 100
        )
        _init_sensitivity = (
            detector._laser_detector.sensitivity
            if detector._laser_detector is not None
            else 50
        )
        _init_decimate = (
            detector._april_detector.quad_decimate
            if detector._april_detector is not None
            else 2.0
        )
        self._laser_threshold = tk.IntVar(value=_init_threshold)
        self._laser_target_area = tk.IntVar(value=_init_target_area)
        self._laser_sensitivity = tk.IntVar(value=_init_sensitivity)
        self._april_decimate = tk.DoubleVar(value=_init_decimate)
        self._show_threshold_overlay = tk.BooleanVar(value=False)

        # Build the UI
        self._build_ui()

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ──────────────────────────────────────────────────────────────────────
    # UI construction
    # ──────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        """Construct all widgets and layout."""
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Left control panel
        ctrl_frame = ttk.Frame(self.root, padding=8, width=200)
        ctrl_frame.grid(row=0, column=0, sticky="nsew")
        ctrl_frame.grid_propagate(False)

        # Centre video canvas
        self._canvas = tk.Canvas(self.root, bg="black")
        self._canvas.grid(row=0, column=1, sticky="nsew")

        # Right info panel
        info_frame = ttk.Frame(self.root, padding=8, width=220)
        info_frame.grid(row=0, column=2, sticky="nsew")
        info_frame.grid_propagate(False)

        self._build_controls(ctrl_frame)
        self._build_info_panel(info_frame)

        # Status bar at the bottom
        self._status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(
            self.root,
            textvariable=self._status_var,
            relief=tk.SUNKEN,
            anchor="w",
            padding=(4, 2),
        )
        status_bar.grid(row=1, column=0, columnspan=3, sticky="ew")

    def _build_controls(self, parent: ttk.Frame) -> None:
        """Build the left-side control panel."""
        ttk.Label(parent, text="CONTROLS", font=("", 10, "bold")).pack(
            anchor="w", pady=(0, 6)
        )

        # ── Program mode ──────────────────────────────────────────────────
        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=4)
        ttk.Label(parent, text="Program mode").pack(anchor="w")

        mode_combo = ttk.Combobox(
            parent,
            textvariable=self._mode_var,
            values=list(_MODE_DISPLAY.keys()),
            state="readonly",
            width=28,
        )
        mode_combo.pack(anchor="w", pady=(2, 0))
        mode_combo.bind("<<ComboboxSelected>>", self._on_mode_change)

        # ── Detection modes ───────────────────────────────────────────────
        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=4)
        ttk.Label(parent, text="Detection modes").pack(anchor="w")

        self._april_cb = ttk.Checkbutton(
            parent,
            text="AprilTag",
            variable=self._enable_april,
            command=self._on_toggle_april,
        )
        self._april_cb.pack(anchor="w")

        ttk.Checkbutton(
            parent,
            text="QR Code",
            variable=self._enable_qr,
            command=self._on_toggle_qr,
        ).pack(anchor="w")

        ttk.Checkbutton(
            parent,
            text="Laser Spot",
            variable=self._enable_laser,
            command=self._on_toggle_laser,
        ).pack(anchor="w")

        # ── Parameters ───────────────────────────────────────────────────
        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=8)
        ttk.Label(parent, text="Parameters").pack(anchor="w")

        # Laser threshold
        ttk.Label(parent, text="Laser threshold (0–255)").pack(
            anchor="w", pady=(6, 0)
        )
        self._threshold_label = ttk.Label(
            parent, text=str(self._laser_threshold.get())
        )
        self._threshold_label.pack(anchor="e")
        ttk.Scale(
            parent,
            from_=0,
            to=255,
            orient="horizontal",
            variable=self._laser_threshold,
            command=self._on_threshold_change,
        ).pack(fill="x")

        # Laser target area
        ttk.Label(parent, text="Laser target area (px)").pack(
            anchor="w", pady=(8, 0)
        )
        self._target_area_label = ttk.Label(
            parent, text=str(self._laser_target_area.get())
        )
        self._target_area_label.pack(anchor="e")
        ttk.Scale(
            parent,
            from_=4,
            to=2000,
            orient="horizontal",
            variable=self._laser_target_area,
            command=self._on_target_area_change,
        ).pack(fill="x")

        # Laser sensitivity
        ttk.Label(parent, text="Laser sensitivity (0–100)").pack(
            anchor="w", pady=(8, 0)
        )
        self._sensitivity_label = ttk.Label(
            parent, text=str(self._laser_sensitivity.get())
        )
        self._sensitivity_label.pack(anchor="e")
        ttk.Scale(
            parent,
            from_=0,
            to=100,
            orient="horizontal",
            variable=self._laser_sensitivity,
            command=self._on_sensitivity_change,
        ).pack(fill="x")

        # Threshold overlay toggle
        ttk.Checkbutton(
            parent,
            text="Show threshold overlay",
            variable=self._show_threshold_overlay,
        ).pack(anchor="w", pady=(8, 0))

        # AprilTag quad_decimate
        ttk.Label(parent, text="AprilTag decimate (1–4)").pack(
            anchor="w", pady=(8, 0)
        )
        self._decimate_label = ttk.Label(
            parent, text=f"{self._april_decimate.get():.1f}"
        )
        self._decimate_label.pack(anchor="e")
        ttk.Scale(
            parent,
            from_=1.0,
            to=4.0,
            orient="horizontal",
            variable=self._april_decimate,
            command=self._on_decimate_change,
        ).pack(fill="x")

        # ── Quit button ───────────────────────────────────────────────────
        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=12)
        ttk.Button(parent, text="Quit", command=self._on_close).pack(fill="x")

    def _build_info_panel(self, parent: ttk.Frame) -> None:
        """Build the right-side information panel."""
        ttk.Label(parent, text="INFO", font=("", 10, "bold")).pack(
            anchor="w", pady=(0, 6)
        )

        # Camera parameters
        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=4)
        ttk.Label(parent, text="Camera", font=("", 9, "bold")).pack(anchor="w")

        self._cam_fps_var = tk.StringVar(value="FPS: –")
        self._cam_res_var = tk.StringVar(value="Resolution: –")
        ttk.Label(parent, textvariable=self._cam_fps_var).pack(anchor="w")
        ttk.Label(parent, textvariable=self._cam_res_var).pack(anchor="w")

        # Detected objects list
        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=8)
        ttk.Label(parent, text="Detected objects", font=("", 9, "bold")).pack(
            anchor="w"
        )

        list_frame = ttk.Frame(parent)
        list_frame.pack(fill="both", expand=True)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical")
        self._detections_list = tk.Listbox(
            list_frame,
            yscrollcommand=scrollbar.set,
            font=("Courier", 8),
            selectmode=tk.SINGLE,
            height=20,
        )
        scrollbar.config(command=self._detections_list.yview)
        self._detections_list.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    # ──────────────────────────────────────────────────────────────────────
    # Control callbacks
    # ──────────────────────────────────────────────────────────────────────

    def _on_mode_change(self, _event: Optional[object] = None) -> None:
        """Switch the detector's operating mode."""
        label = self._mode_var.get()
        new_mode = _MODE_DISPLAY.get(label, DetectionMode.NORMAL)
        self.detector.mode = new_mode

    def _on_toggle_april(self) -> None:
        """Enable or disable the AprilTag detector."""
        from .april_tag_detector import AprilTagDetector, _apriltags_available

        if self._enable_april.get():
            if self.detector._april_detector is None:
                if _apriltags_available():
                    self.detector._april_detector = AprilTagDetector()
                else:
                    # pupil-apriltags not installed; revert the checkbox
                    self._enable_april.set(False)
        else:
            self.detector._april_detector = None

    def _on_toggle_qr(self) -> None:
        """Enable or disable the QR-code detector."""
        from .qr_detector import QRCodeDetector

        if self._enable_qr.get():
            if self.detector._qr_detector is None:
                self.detector._qr_detector = QRCodeDetector()
        else:
            self.detector._qr_detector = None

    def _on_toggle_laser(self) -> None:
        """Enable or disable the laser-spot detector."""
        from .laser_detector import LaserSpotDetector

        if self._enable_laser.get():
            if self.detector._laser_detector is None:
                self.detector._laser_detector = LaserSpotDetector(
                    brightness_threshold=self._laser_threshold.get(),
                    target_area=self._laser_target_area.get(),
                    sensitivity=self._laser_sensitivity.get(),
                )
        else:
            self.detector._laser_detector = None

    def _on_threshold_change(self, _value: Optional[str] = None) -> None:
        """Apply the new laser-threshold value."""
        val = self._laser_threshold.get()
        self._threshold_label.config(text=str(val))
        if self.detector._laser_detector is not None:
            self.detector._laser_detector.brightness_threshold = val

    def _on_target_area_change(self, _value: Optional[str] = None) -> None:
        """Apply the new laser target-area value."""
        val = self._laser_target_area.get()
        self._target_area_label.config(text=str(val))
        if self.detector._laser_detector is not None:
            self.detector._laser_detector.target_area = val

    def _on_sensitivity_change(self, _value: Optional[str] = None) -> None:
        """Apply the new laser sensitivity value."""
        val = self._laser_sensitivity.get()
        self._sensitivity_label.config(text=str(val))
        if self.detector._laser_detector is not None:
            self.detector._laser_detector.sensitivity = val

    def _on_decimate_change(self, _value: Optional[str] = None) -> None:
        """Apply the new AprilTag quad_decimate value."""
        val = round(self._april_decimate.get(), 1)
        self._decimate_label.config(text=f"{val:.1f}")
        if self.detector._april_detector is not None:
            try:
                self.detector._april_detector._detector.quad_decimate = val
            except AttributeError:
                pass  # detector doesn't expose this at runtime – ignored

    # ──────────────────────────────────────────────────────────────────────
    # Frame update loop
    # ──────────────────────────────────────────────────────────────────────

    def _update_frame(self) -> None:
        """Grab a frame, run detectors, update canvas and info panel."""
        if not self._running:
            return

        frame = self.camera.read()
        if frame is None:
            self._status_var.set("Camera stream ended.")
            return

        # Detection
        detections = self.detector.process_frame(frame)
        self._last_detections = detections

        # FPS
        self._fps_counter += 1
        t_now = time.perf_counter()
        elapsed = t_now - self._t_fps
        if elapsed >= 1.0:
            self._fps_display = self._fps_counter / elapsed
            self._fps_counter = 0
            self._t_fps = t_now

        # Draw annotations on a copy of the frame
        vis = frame.copy()

        # Threshold overlay: highlight pixels above the laser brightness
        # threshold so users can see the effect of slider adjustments in
        # real time.
        if (
            self._show_threshold_overlay.get()
            and self.detector._laser_detector is not None
        ):
            mask = self.detector._laser_detector.last_threshold_mask
            if mask is None:
                # Fallback before the first detect() call
                gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(
                    gray,
                    self.detector._laser_detector.brightness_threshold,
                    255,
                    cv2.THRESH_BINARY,
                )
            # Semi-transparent orange overlay on all above-threshold pixels
            overlay = np.zeros_like(vis)
            overlay[mask > 0] = (0, 140, 255)  # BGR orange
            cv2.addWeighted(overlay, 0.45, vis, 1.0, 0, vis)

        vis = self.detector.draw_detections(vis, detections)

        # Overlay FPS
        cv2.putText(
            vis,
            f"FPS: {self._fps_display:.1f}",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Convert BGR → RGB → PIL → ImageTk for the canvas
        rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # Resize to fit the current canvas size
        canvas_w = self._canvas.winfo_width()
        canvas_h = self._canvas.winfo_height()
        if canvas_w > 1 and canvas_h > 1:
            resample = getattr(Image, "Resampling", Image).BILINEAR
            pil_img = pil_img.resize((canvas_w, canvas_h), resample)

        self._tk_image = ImageTk.PhotoImage(image=pil_img)
        if self._canvas_image_id is None:
            self._canvas_image_id = self._canvas.create_image(
                0, 0, anchor="nw", image=self._tk_image
            )
        else:
            self._canvas.itemconfigure(self._canvas_image_id, image=self._tk_image)

        # Update info panels
        self._update_info_panel(detections)

        # Status bar
        n = len(detections)
        mode_label = _MODE_DISPLAY_INV.get(self.detector.mode, "")
        self._status_var.set(
            f"FPS: {self._fps_display:.1f}  |  Mode: {mode_label}  |  Detections: {n}"
        )

        # Schedule next update
        self.root.after(_UPDATE_INTERVAL_MS, self._update_frame)

    def _update_info_panel(self, detections: List[Detection]) -> None:
        """Refresh the camera-parameters and detections-list widgets."""
        # Camera info
        self._cam_fps_var.set(f"FPS: {self._fps_display:.1f}")
        self._cam_res_var.set(
            f"Resolution: {self.camera.actual_width}×{self.camera.actual_height}"
        )

        # Detections list
        self._detections_list.delete(0, tk.END)
        for d in detections:
            dtype = d.detection_type.value.replace("_", " ").title()
            ident = f"  id={d.identifier}" if d.identifier else ""
            track = f"  #trk={d.track_id}" if d.track_id is not None else ""
            cx, cy = d.center
            angle = _compute_orientation(d.corners)
            line = (
                f"[{dtype}]{ident}{track}\n"
                f"  X:{cx}  Y:{cy}  θ:{angle:.1f}°"
            )
            self._detections_list.insert(tk.END, line)

    # ──────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────────

    def _on_close(self) -> None:
        """Handle window-close event: stop the update loop and destroy."""
        self._running = False
        self.root.destroy()

    def run(self) -> None:
        """Start the update loop and enter the Tkinter event loop.

        This call blocks until the window is closed.
        """
        self.root.after(_UPDATE_INTERVAL_MS, self._update_frame)
        self.root.mainloop()
