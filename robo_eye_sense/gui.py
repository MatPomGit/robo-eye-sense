"""Tkinter-based GUI application for RoboEyeSense.

Layout
------
+-------------+---------------------------+-----------+
|  CONTROLS   |       VIDEO FEED          |   INFO    |
|  ---------- |  (annotated frame)        | --------- |
|  Mode       |                           | Camera    |
|  [combo]    |                           | FPS / W×H |
|  ---------- |                           | --------- |
|  Detectors  |                           | Mode      |
|  [x] April  |                           | --------- |
|  [x] QR     |                           | Objects   |
|  [x] Laser  |                           | (list)    |
|  ---------- |                           | --------- |
|  Parameters |                           | Scenario  |
|  Threshold  |                           | (text)    |
|  Target area|                           |           |
|  Sensitivity|                           |           |
|  [ ] Overlay|                           |           |
|  ---------- |                           |           |
|  Scenario   |                           |           |
|  [Start]    |                           |           |
|  [Capture ] |                           |           |
|  [Reset   ] |                           |           |
|  ---------- |                           |           |
|  [  Quit  ] |                           |           |
+-------------+---------------------------+-----------+
|  Status bar (FPS | Mode | Detections)               |
+------------------------------------------------------|

Keyboard shortcuts: Ctrl+1 → Normal, Ctrl+2 → Fast, Ctrl+3 → Robust.

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

from . import APP_NAME, __version__
from .camera import Camera
from .detector import RoboEyeDetector, _compute_orientation
from .offset_scenario import CameraOffsetScenario, OffsetResult
from .recorder import VideoRecorder
from .results import Detection, DetectionMode, DetectionType

# How often (milliseconds) the frame-update callback is rescheduled.
# 16 ms gives a ~60 Hz ceiling; the actual frame rate is limited by the camera.
_UPDATE_INTERVAL_MS = 16  # ~60 Hz ceiling; actual rate is camera-limited

# Human-readable labels shown in the mode combobox
_MODE_DISPLAY: dict[str, DetectionMode] = {
    "Normal": DetectionMode.NORMAL,
    "Fast (low-power)": DetectionMode.FAST,
    "Robust (motion-blur resistant)": DetectionMode.ROBUST,
}
_MODE_DISPLAY_INV: dict[DetectionMode, str] = {v: k for k, v in _MODE_DISPLAY.items()}

# Short descriptions displayed below the combobox when a mode is active
_MODE_DESCRIPTIONS: dict[DetectionMode, str] = {
    DetectionMode.NORMAL: "Balanced – default detection pipeline.",
    DetectionMode.FAST: "Speed-optimised – frame downscaled 50 %.",
    DetectionMode.ROBUST: "Tracking-optimised – sharpening + Kalman.",
}


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
        initial_record_path: Optional[str] = None,
    ) -> None:
        self.root = root
        self.camera = camera
        self.detector = detector

        self.root.title(f"{APP_NAME} v{__version__}")
        self.root.resizable(True, True)

        # ── State variables ──────────────────────────────────────────────
        self._running = True
        self._fps_counter = 0
        self._fps_display = 0.0
        self._t_fps = time.perf_counter()
        self._last_detections: List[Detection] = []
        self._canvas_image_id: Optional[int] = None

        # Detection-mode flags (mirror the detector's live state)
        self._enable_april = tk.BooleanVar(value=detector.april_enabled)
        self._enable_qr = tk.BooleanVar(value=detector.qr_enabled)
        self._enable_laser = tk.BooleanVar(value=detector.laser_enabled)

        # Program mode
        initial_mode_label = _MODE_DISPLAY_INV.get(detector.mode, "Normal")
        self._mode_var = tk.StringVar(value=initial_mode_label)

        # Tunable parameters
        _laser = detector.laser_detector
        _init_threshold = _laser.brightness_threshold if _laser is not None else 240
        _init_target_area = _laser.target_area if _laser is not None else 100
        _init_sensitivity = _laser.sensitivity if _laser is not None else 50
        self._laser_threshold = tk.IntVar(value=_init_threshold)
        self._laser_target_area = tk.IntVar(value=_init_target_area)
        self._laser_sensitivity = tk.IntVar(value=_init_sensitivity)
        self._show_threshold_overlay = tk.BooleanVar(value=False)

        # Scenario state
        self._scenario: Optional[CameraOffsetScenario] = None
        self._scenario_active = False
        self._last_offset_result: Optional[OffsetResult] = None

        # Recording state
        self._recorder: Optional[VideoRecorder] = None
        self._record_path: Optional[str] = initial_record_path

        # Build the UI
        self._build_ui()

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Keyboard shortcuts for quick mode switching
        self.root.bind("<Control-Key-1>", lambda _e: self._set_mode(DetectionMode.NORMAL))
        self.root.bind("<Control-Key-2>", lambda _e: self._set_mode(DetectionMode.FAST))
        self.root.bind("<Control-Key-3>", lambda _e: self._set_mode(DetectionMode.ROBUST))

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

        # Description of the currently selected mode
        initial_desc = _MODE_DESCRIPTIONS.get(self.detector.mode, "")
        self._mode_desc_var = tk.StringVar(value=initial_desc)
        self._mode_desc_label = ttk.Label(
            parent,
            textvariable=self._mode_desc_var,
            wraplength=180,
            font=("", 8, "italic"),
        )
        self._mode_desc_label.pack(anchor="w", pady=(2, 0))

        # Keyboard shortcut hint
        ttk.Label(
            parent,
            text="Ctrl+1 / 2 / 3",
            font=("", 7),
            foreground="gray",
        ).pack(anchor="w")

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

        # ── Scenario ─────────────────────────────────────────────────────
        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=8)
        ttk.Label(parent, text="Scenario", font=("", 9, "bold")).pack(anchor="w")

        self._scenario_start_btn = ttk.Button(
            parent,
            text="Start scenario",
            command=self._on_scenario_start,
        )
        self._scenario_start_btn.pack(fill="x", pady=(4, 2))

        self._scenario_capture_btn = ttk.Button(
            parent,
            text="Capture reference",
            command=self._on_scenario_capture_reference,
            state="disabled",
        )
        self._scenario_capture_btn.pack(fill="x", pady=2)

        self._scenario_reset_btn = ttk.Button(
            parent,
            text="Reset reference",
            command=self._on_scenario_reset,
            state="disabled",
        )
        self._scenario_reset_btn.pack(fill="x", pady=2)

        # ── Recording ─────────────────────────────────────────────────────
        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=8)
        ttk.Label(parent, text="Recording", font=("", 9, "bold")).pack(anchor="w")

        self._record_btn = ttk.Button(
            parent,
            text="Start recording",
            command=self._on_toggle_recording,
        )
        self._record_btn.pack(fill="x", pady=(4, 2))

        self._record_status_var = tk.StringVar(value="Not recording")
        ttk.Label(
            parent,
            textvariable=self._record_status_var,
            font=("", 8, "italic"),
        ).pack(anchor="w")

        # ── Quit button ───────────────────────────────────────────────────
        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=12)
        ttk.Button(parent, text="Quit", command=self._on_close).pack(fill="x")

    def _build_info_panel(self, parent: ttk.Frame) -> None:
        """Build the right-side information panel."""
        ttk.Label(parent, text="INFO", font=("", 10, "bold")).pack(
            anchor="w", pady=(0, 6)
        )

        # Software name and version
        ttk.Label(
            parent,
            text=f"{APP_NAME} v{__version__}",
            font=("", 9, "italic"),
        ).pack(anchor="w")

        # Camera parameters
        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=4)
        ttk.Label(parent, text="Camera", font=("", 9, "bold")).pack(anchor="w")

        self._cam_fps_var = tk.StringVar(value="FPS: –")
        self._cam_res_var = tk.StringVar(value="Resolution: –")
        ttk.Label(parent, textvariable=self._cam_fps_var).pack(anchor="w")
        ttk.Label(parent, textvariable=self._cam_res_var).pack(anchor="w")

        # Current mode
        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=4)
        ttk.Label(parent, text="Mode", font=("", 9, "bold")).pack(anchor="w")
        initial_label = _MODE_DISPLAY_INV.get(self.detector.mode, "Normal")
        self._info_mode_var = tk.StringVar(value=initial_label)
        ttk.Label(parent, textvariable=self._info_mode_var).pack(anchor="w")

        # Detected objects list
        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=8)
        ttk.Label(parent, text="Detected objects", font=("", 9, "bold")).pack(
            anchor="w"
        )

        list_frame = ttk.Frame(parent)
        list_frame.pack(fill="x")
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical")
        self._detections_list = tk.Listbox(
            list_frame,
            yscrollcommand=scrollbar.set,
            font=("Courier", 8),
            selectmode=tk.SINGLE,
            height=10,
        )
        scrollbar.config(command=self._detections_list.yview)
        self._detections_list.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Scenario information panel
        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=8)
        ttk.Label(parent, text="Scenario", font=("", 9, "bold")).pack(
            anchor="w"
        )

        scenario_frame = ttk.Frame(parent)
        scenario_frame.pack(fill="both", expand=True)
        scenario_scroll = ttk.Scrollbar(scenario_frame, orient="vertical")
        self._scenario_text = tk.Text(
            scenario_frame,
            yscrollcommand=scenario_scroll.set,
            font=("Courier", 8),
            wrap="word",
            height=10,
            state="disabled",
            bg="#f0f0f0",
        )
        scenario_scroll.config(command=self._scenario_text.yview)
        self._scenario_text.pack(side="left", fill="both", expand=True)
        scenario_scroll.pack(side="right", fill="y")
        self._set_scenario_text("Scenario not started.\nClick 'Start scenario' to begin.")

    # ──────────────────────────────────────────────────────────────────────
    # Control callbacks
    # ──────────────────────────────────────────────────────────────────────

    def _set_mode(self, mode: DetectionMode) -> None:
        """Programmatically switch to *mode* and update all UI elements."""
        label = _MODE_DISPLAY_INV.get(mode, "Normal")
        self._mode_var.set(label)
        self._on_mode_change()

    def _on_mode_change(self, _event: Optional[object] = None) -> None:
        """Switch the detector's operating mode."""
        label = self._mode_var.get()
        new_mode = _MODE_DISPLAY.get(label, DetectionMode.NORMAL)
        self.detector.mode = new_mode
        # Update the description label and info-panel indicator
        self._mode_desc_var.set(_MODE_DESCRIPTIONS.get(new_mode, ""))
        self._info_mode_var.set(label)

    def _on_toggle_april(self) -> None:
        """Enable or disable the AprilTag detector."""
        if self._enable_april.get():
            if not self.detector.enable_april():
                # pupil-apriltags not installed; revert the checkbox
                self._enable_april.set(False)
        else:
            self.detector.disable_april()

    def _on_toggle_qr(self) -> None:
        """Enable or disable the QR-code detector."""
        if self._enable_qr.get():
            self.detector.enable_qr()
        else:
            self.detector.disable_qr()

    def _on_toggle_laser(self) -> None:
        """Enable or disable the laser-spot detector."""
        if self._enable_laser.get():
            self.detector.enable_laser(
                brightness_threshold=self._laser_threshold.get(),
                target_area=self._laser_target_area.get(),
                sensitivity=self._laser_sensitivity.get(),
            )
        else:
            self.detector.disable_laser()

    def _on_threshold_change(self, _value: Optional[str] = None) -> None:
        """Apply the new laser-threshold value."""
        val = self._laser_threshold.get()
        self._threshold_label.config(text=str(val))
        laser = self.detector.laser_detector
        if laser is not None:
            laser.brightness_threshold = val

    def _on_target_area_change(self, _value: Optional[str] = None) -> None:
        """Apply the new laser target-area value."""
        val = self._laser_target_area.get()
        self._target_area_label.config(text=str(val))
        laser = self.detector.laser_detector
        if laser is not None:
            laser.target_area = val

    def _on_sensitivity_change(self, _value: Optional[str] = None) -> None:
        """Apply the new laser sensitivity value."""
        val = self._laser_sensitivity.get()
        self._sensitivity_label.config(text=str(val))
        laser = self.detector.laser_detector
        if laser is not None:
            laser.sensitivity = val

    # ── Scenario callbacks ────────────────────────────────────────────────

    def _set_scenario_text(self, text: str) -> None:
        """Replace the content of the scenario text widget."""
        self._scenario_text.config(state="normal")
        self._scenario_text.delete("1.0", tk.END)
        self._scenario_text.insert("1.0", text)
        self._scenario_text.config(state="disabled")

    def _on_scenario_start(self) -> None:
        """Start or stop the offset-calibration scenario."""
        if self._scenario_active:
            # Stop the scenario
            self._scenario_active = False
            self._scenario = None
            self._last_offset_result = None
            self._scenario_start_btn.config(text="Start scenario")
            self._scenario_capture_btn.config(state="disabled")
            self._scenario_reset_btn.config(state="disabled")
            self._set_scenario_text(
                "Scenario not started.\n"
                "Click 'Start scenario' to begin."
            )
        else:
            # Start the scenario
            self._scenario = CameraOffsetScenario(
                camera=self.camera,
                detector=self.detector,
                frame_width=self.camera.actual_width,
            )
            self._scenario_active = True
            self._last_offset_result = None
            self._scenario_start_btn.config(text="Stop scenario")
            self._scenario_capture_btn.config(state="normal")
            self._scenario_reset_btn.config(state="disabled")
            self._set_scenario_text(
                "Scenario started.\n\n"
                "STEP 1: Position the camera at\n"
                "the REFERENCE position with\n"
                "AprilTags visible.\n\n"
                "Then click 'Capture reference'."
            )

    def _on_scenario_capture_reference(self) -> None:
        """Capture the current detections as the reference frame."""
        if self._scenario is None:
            return
        # Use the last detections from the frame loop instead of
        # capturing a separate frame.  This keeps the UI responsive
        # and avoids consuming a camera frame outside the main loop.
        detections = self._last_detections
        april_count = sum(
            1
            for d in detections
            if d.detection_type == DetectionType.APRIL_TAG
        )
        # Store as reference via public API
        self._scenario.set_reference(detections)
        self._scenario_reset_btn.config(state="normal")

        if april_count == 0:
            self._set_scenario_text(
                "WARNING: No AprilTags found in\n"
                "reference frame!\n\n"
                "The offset will be (0, 0).\n"
                "Try repositioning and click\n"
                "'Reset reference' → 'Capture\n"
                "reference' again."
            )
        else:
            self._set_scenario_text(
                f"Reference captured:\n"
                f"  {april_count} AprilTag(s) found.\n\n"
                "STEP 2: Move the camera.\n"
                "Offset is computed continuously.\n\n"
                "Waiting for frames…"
            )

    def _on_scenario_reset(self) -> None:
        """Reset the scenario reference to allow re-capture."""
        if self._scenario is None:
            return
        self._scenario.reset()
        self._last_offset_result = None
        self._scenario_reset_btn.config(state="disabled")
        self._set_scenario_text(
            "Reference cleared.\n\n"
            "STEP 1: Position the camera at\n"
            "the REFERENCE position with\n"
            "AprilTags visible.\n\n"
            "Then click 'Capture reference'."
        )

    # ── Recording callbacks ───────────────────────────────────────────────

    def _on_toggle_recording(self) -> None:
        """Start or stop video recording."""
        if self._recorder is not None and self._recorder.is_recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        """Prompt for a file path and begin recording."""
        from tkinter import filedialog

        if self._record_path:
            path = self._record_path
            self._record_path = None  # use only once from CLI
        else:
            path = filedialog.asksaveasfilename(
                title="Save recording as",
                defaultextension=".mp4",
                filetypes=[("MP4 video", "*.mp4"), ("All files", "*.*")],
            )
        if not path:
            return
        try:
            self._recorder = VideoRecorder(
                path,
                width=self.camera.actual_width,
                height=self.camera.actual_height,
                fps=self.camera.actual_fps or 30.0,
            )
            self._recorder.start()
            self._record_btn.config(text="Stop recording")
            self._record_status_var.set(f"Recording: {path}")
        except RuntimeError as exc:
            self._record_status_var.set(f"Error: {exc}")

    def _stop_recording(self) -> None:
        """Finish recording and release the writer."""
        if self._recorder is not None:
            path = self._recorder.output_path
            self._recorder.stop()
            self._recorder = None
            self._record_btn.config(text="Start recording")
            self._record_status_var.set(f"Saved: {path}")

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
        laser = self.detector.laser_detector
        if self._show_threshold_overlay.get() and laser is not None:
            mask = laser.last_threshold_mask
            if mask is None:
                # Fallback before the first detect() call
                gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(
                    gray,
                    laser.brightness_threshold,
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

        # Record the annotated frame (before canvas resize)
        if self._recorder is not None and self._recorder.is_recording:
            self._recorder.write_frame(vis)

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
        rec_label = "  |  ● REC" if (self._recorder is not None and self._recorder.is_recording) else ""
        self._status_var.set(
            f"FPS: {self._fps_display:.1f}  |  Mode: {mode_label}  |  Detections: {n}{rec_label}"
        )

        # Schedule next update
        self.root.after(_UPDATE_INTERVAL_MS, self._update_frame)

    def _update_info_panel(self, detections: List[Detection]) -> None:
        """Refresh the camera-parameters, detections-list, and scenario widgets."""
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

        # Scenario: continuous offset computation
        if self._scenario_active and self._scenario is not None and self._scenario.has_reference:
            try:
                result = self._scenario.compute_offset_from_detections(detections)
                self._last_offset_result = result
                self._update_scenario_display(result)
            except RuntimeError:
                pass

    def _update_scenario_display(self, result: OffsetResult) -> None:
        """Refresh the scenario text widget with the latest offset data."""
        lines = []
        dx, dy = result.offset
        lines.append(f"OFFSET (dx, dy):")
        lines.append(f"  ({dx:+.1f}, {dy:+.1f}) px")
        lines.append(f"Matched tags: {result.matched_tags}")
        lines.append("")

        if result.distance_to_reference_cm is not None:
            lines.append("Est. distance to reference:")
            lines.append(f"  {result.distance_to_reference_cm:.1f} cm")
            lines.append("")

        if result.per_tag_offsets:
            lines.append("Per-tag offsets:")
            for tag_id in sorted(result.per_tag_offsets):
                tdx, tdy = result.per_tag_offsets[tag_id]
                lines.append(f"  tag {tag_id:>4s}: ({tdx:+.1f}, {tdy:+.1f}) px")
            lines.append("")

        if result.per_tag_distances_cm:
            lines.append("Est. distance to tags:")
            for tag_id in sorted(result.per_tag_distances_cm):
                dist = result.per_tag_distances_cm[tag_id]
                lines.append(f"  tag {tag_id:>4s}: {dist:.1f} cm")

        self._set_scenario_text("\n".join(lines))

    # ──────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────────

    def _on_close(self) -> None:
        """Handle window-close event: stop recording, update loop, and destroy."""
        self._running = False
        self._stop_recording()
        self.root.destroy()

    def run(self) -> None:
        """Start the update loop and enter the Tkinter event loop.

        This call blocks until the window is closed.
        """
        self.root.after(_UPDATE_INTERVAL_MS, self._update_frame)
        self.root.mainloop()
