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
|  Threshold  |                           |[Offset]   |
|  Target area|                           |  [SLAM]   |
|  Sensitivity|                           |  (tabs)   |
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

import math
import time
import tkinter as tk
from tkinter import ttk
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk

from . import APP_NAME, __version__
from .camera import Camera
from .detector import RoboEyeDetector, _compute_orientation
from .marker_map import MarkerPose3D, RobotPose3D, SlamCalibrator
from .offset_scenario import CameraOffsetScenario, OffsetResult
from .recorder import VideoRecorder
from .results import Detection, DetectionMode, DetectionType

# How often (milliseconds) the frame-update callback is rescheduled.
# 16 ms gives a ~60 Hz ceiling; the actual frame rate is limited by the camera.
_UPDATE_INTERVAL_MS = 16  # ~60 Hz ceiling; actual rate is camera-limited

# 3-D visualisation defaults
_VIS3D_WIDTH = 200
_VIS3D_HEIGHT = 200
_VIS3D_BG = "#1a1a2e"
_VIS3D_GRID_COLOR = "#334455"
_VIS3D_MARKER_COLOR = "#00cc66"
_VIS3D_CAMERA_COLOR = "#ff4444"

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


def render_3d_scene(
    width: int,
    height: int,
    markers: List[MarkerPose3D],
    robot: RobotPose3D,
) -> Image.Image:
    """Render a top-down 3-D scene as a PIL Image.

    The scene shows markers as green squares and the robot/camera as a red
    triangle pointing in its yaw direction.  A simple grid is drawn as
    background.

    Parameters
    ----------
    width, height:
        Output image size in pixels.
    markers:
        List of marker poses to draw.
    robot:
        Current robot pose (the camera).

    Returns
    -------
    PIL.Image.Image
        Rendered scene.
    """
    img = Image.new("RGB", (width, height), _VIS3D_BG)
    draw = ImageDraw.Draw(img)

    # Collect all positions to compute auto-scale
    all_x: List[float] = []
    all_z: List[float] = []
    for m in markers:
        all_x.append(m.position[0])
        all_z.append(m.position[2])
    if robot.visible_markers > 0:
        all_x.append(robot.position[0])
        all_z.append(robot.position[2])

    if not all_x:
        # Nothing to draw — return blank with grid
        _draw_grid(draw, width, height, 1.0, 0.0, 0.0)
        return img

    cx = (min(all_x) + max(all_x)) / 2.0
    cz = (min(all_z) + max(all_z)) / 2.0
    span = max(max(all_x) - min(all_x), max(all_z) - min(all_z), 20.0)
    scale = min(width, height) * 0.7 / span

    _draw_grid(draw, width, height, scale, cx, cz)

    # Draw markers
    for m in markers:
        sx = width / 2.0 + (m.position[0] - cx) * scale
        sy = height / 2.0 + (m.position[2] - cz) * scale
        r = max(3, int(4 * scale / 20))
        draw.rectangle([sx - r, sy - r, sx + r, sy + r], fill=_VIS3D_MARKER_COLOR)
        label = m.marker_id
        draw.text((sx + r + 2, sy - r), label, fill=_VIS3D_MARKER_COLOR)

    # Draw robot/camera as a triangle
    if robot.visible_markers > 0:
        rx = width / 2.0 + (robot.position[0] - cx) * scale
        ry = height / 2.0 + (robot.position[2] - cz) * scale
        yaw_rad = math.radians(robot.orientation[2])
        size = max(5, int(6 * scale / 20))
        # Triangle pointing in yaw direction
        pts = [
            (rx + size * math.cos(yaw_rad), ry + size * math.sin(yaw_rad)),
            (
                rx + size * math.cos(yaw_rad + 2.4),
                ry + size * math.sin(yaw_rad + 2.4),
            ),
            (
                rx + size * math.cos(yaw_rad - 2.4),
                ry + size * math.sin(yaw_rad - 2.4),
            ),
        ]
        draw.polygon(pts, fill=_VIS3D_CAMERA_COLOR)

    return img


def _draw_grid(
    draw: ImageDraw.ImageDraw,
    w: int,
    h: int,
    scale: float,
    cx: float,
    cz: float,
) -> None:
    """Draw a faint background grid on the scene."""
    grid_step = 10.0  # cm
    # Clamp grid range to prevent excessive drawing
    half_w = w / (2.0 * max(scale, 1e-6))
    half_h = h / (2.0 * max(scale, 1e-6))
    x_min = cx - half_w
    x_max = cx + half_w
    z_min = cz - half_h
    z_max = cz + half_h
    x = math.floor(x_min / grid_step) * grid_step
    while x <= x_max:
        sx = int(w / 2.0 + (x - cx) * scale)
        draw.line([(sx, 0), (sx, h)], fill=_VIS3D_GRID_COLOR, width=1)
        x += grid_step
    z = math.floor(z_min / grid_step) * grid_step
    while z <= z_max:
        sy = int(h / 2.0 + (z - cz) * scale)
        draw.line([(0, sy), (w, sy)], fill=_VIS3D_GRID_COLOR, width=1)
        z += grid_step


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
        _init_channels = _laser.channels if _laser is not None else "bgr"
        self._laser_threshold = tk.IntVar(value=_init_threshold)
        self._laser_target_area = tk.IntVar(value=_init_target_area)
        self._laser_sensitivity = tk.IntVar(value=_init_sensitivity)
        self._laser_ch_r = tk.BooleanVar(value="r" in _init_channels)
        self._laser_ch_g = tk.BooleanVar(value="g" in _init_channels)
        self._laser_ch_b = tk.BooleanVar(value="b" in _init_channels)
        self._show_threshold_overlay = tk.BooleanVar(value=False)

        # Scenario state
        self._scenario: Optional[CameraOffsetScenario] = None
        self._scenario_active = False
        self._last_offset_result: Optional[OffsetResult] = None

        # SLAM state
        self._slam_calibrator: Optional[SlamCalibrator] = None
        self._slam_active = False
        self._last_robot_pose = RobotPose3D()

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
        self.root.columnconfigure(0, minsize=200)
        self.root.columnconfigure(1, weight=1)
        self.root.columnconfigure(2, minsize=220)
        self.root.rowconfigure(0, weight=1)

        # Minimum window size so that side panels are never hidden.
        self.root.minsize(700, 400)

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

        # Laser channel selection
        ttk.Label(parent, text="Laser channels").pack(
            anchor="w", pady=(8, 0)
        )
        _ch_frame = ttk.Frame(parent)
        _ch_frame.pack(anchor="w")
        ttk.Checkbutton(
            _ch_frame, text="R", variable=self._laser_ch_r,
            command=self._on_channel_change,
        ).pack(side="left")
        ttk.Checkbutton(
            _ch_frame, text="G", variable=self._laser_ch_g,
            command=self._on_channel_change,
        ).pack(side="left")
        ttk.Checkbutton(
            _ch_frame, text="B", variable=self._laser_ch_b,
            command=self._on_channel_change,
        ).pack(side="left")

        # Threshold overlay toggle
        ttk.Checkbutton(
            parent,
            text="Show threshold overlay",
            variable=self._show_threshold_overlay,
        ).pack(anchor="w", pady=(8, 0))

        # ── Scenario ─────────────────────────────────────────────────────
        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=8)
        ttk.Label(parent, text="Offset Scenario", font=("", 9, "bold")).pack(anchor="w")

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

        # ── SLAM scenario ─────────────────────────────────────────────────
        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=8)
        ttk.Label(parent, text="SLAM Scenario", font=("", 9, "bold")).pack(anchor="w")

        self._slam_start_btn = ttk.Button(
            parent,
            text="Start SLAM",
            command=self._on_slam_start,
        )
        self._slam_start_btn.pack(fill="x", pady=(4, 2))

        self._slam_reset_btn = ttk.Button(
            parent,
            text="Reset SLAM",
            command=self._on_slam_reset,
            state="disabled",
        )
        self._slam_reset_btn.pack(fill="x", pady=2)

        self._slam_save_btn = ttk.Button(
            parent,
            text="Save map…",
            command=self._on_slam_save,
            state="disabled",
        )
        self._slam_save_btn.pack(fill="x", pady=2)

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

        # Scenario information panel — tabbed notebook
        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=8)
        ttk.Label(parent, text="Scenario", font=("", 9, "bold")).pack(
            anchor="w"
        )

        self._scenario_notebook = ttk.Notebook(parent)
        self._scenario_notebook.pack(fill="both", expand=True)

        # ── Offset tab ────────────────────────────────────────────────────
        offset_tab = ttk.Frame(self._scenario_notebook, padding=4)
        self._scenario_notebook.add(offset_tab, text="Offset")

        scenario_frame = ttk.Frame(offset_tab)
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

        # ── SLAM tab ─────────────────────────────────────────────────────
        slam_tab = ttk.Frame(self._scenario_notebook, padding=4)
        self._scenario_notebook.add(slam_tab, text="SLAM")
        self._build_slam_tab(slam_tab)

    # ──────────────────────────────────────────────────────────────────────
    # Control callbacks
    # ──────────────────────────────────────────────────────────────────────

    def _build_slam_tab(self, parent: ttk.Frame) -> None:
        """Build the SLAM scenario tab contents."""
        # Robot pose section
        ttk.Label(parent, text="Robot pose", font=("", 9, "bold")).pack(
            anchor="w"
        )
        self._slam_robot_var = tk.StringVar(
            value="Position: –\nOrientation: –"
        )
        ttk.Label(
            parent,
            textvariable=self._slam_robot_var,
            font=("Courier", 8),
            wraplength=200,
            justify="left",
        ).pack(anchor="w", pady=(0, 4))

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=4)

        # Marker list section
        ttk.Label(parent, text="Markers", font=("", 9, "bold")).pack(
            anchor="w"
        )
        marker_frame = ttk.Frame(parent)
        marker_frame.pack(fill="x")
        marker_scroll = ttk.Scrollbar(marker_frame, orient="vertical")
        self._slam_markers_list = tk.Listbox(
            marker_frame,
            yscrollcommand=marker_scroll.set,
            font=("Courier", 7),
            selectmode=tk.SINGLE,
            height=6,
        )
        marker_scroll.config(command=self._slam_markers_list.yview)
        self._slam_markers_list.pack(side="left", fill="both", expand=True)
        marker_scroll.pack(side="right", fill="y")

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=4)

        # 3-D visualisation canvas
        ttk.Label(parent, text="3D View", font=("", 9, "bold")).pack(
            anchor="w"
        )
        self._slam_3d_canvas = tk.Canvas(
            parent,
            width=_VIS3D_WIDTH,
            height=_VIS3D_HEIGHT,
            bg=_VIS3D_BG,
        )
        self._slam_3d_canvas.pack(fill="both", expand=True)
        self._slam_3d_image_id: Optional[int] = None

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
                channels=self._laser_channels_str(),
            )
        else:
            self.detector.disable_laser()

    def _laser_channels_str(self) -> str:
        """Build a channel string from the R/G/B checkbox state."""
        chs = ""
        if self._laser_ch_r.get():
            chs += "r"
        if self._laser_ch_g.get():
            chs += "g"
        if self._laser_ch_b.get():
            chs += "b"
        return chs or "rgb"  # fall back to all channels if none selected

    def _on_channel_change(self) -> None:
        """Apply the new laser channel selection."""
        laser = self.detector.laser_detector
        if laser is not None:
            laser.channels = self._laser_channels_str()

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

    # ── SLAM callbacks ────────────────────────────────────────────────────

    def _on_slam_start(self) -> None:
        """Start or stop the SLAM map-building scenario."""
        if self._slam_active:
            self._slam_active = False
            self._slam_calibrator = None
            self._last_robot_pose = RobotPose3D()
            self._slam_start_btn.config(text="Start SLAM")
            self._slam_reset_btn.config(state="disabled")
            self._slam_save_btn.config(state="disabled")
            self._slam_robot_var.set("Position: –\nOrientation: –")
            self._slam_markers_list.delete(0, tk.END)
        else:
            self._slam_calibrator = SlamCalibrator(tag_size_cm=5.0)
            self._slam_active = True
            self._last_robot_pose = RobotPose3D()
            self._slam_start_btn.config(text="Stop SLAM")
            self._slam_reset_btn.config(state="normal")
            self._slam_save_btn.config(state="normal")
            self._slam_robot_var.set("SLAM started.\nWaiting for markers…")
            # Switch to SLAM tab
            self._scenario_notebook.select(1)

    def _on_slam_reset(self) -> None:
        """Reset the SLAM calibrator (clear the marker map)."""
        if self._slam_calibrator is None:
            return
        self._slam_calibrator.reset()
        self._last_robot_pose = RobotPose3D()
        self._slam_robot_var.set("Map cleared.\nWaiting for markers…")
        self._slam_markers_list.delete(0, tk.END)

    def _on_slam_save(self) -> None:
        """Save the current marker map to a JSON file."""
        if self._slam_calibrator is None:
            return
        from tkinter import filedialog

        path = filedialog.asksaveasfilename(
            title="Save marker map as",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        self._slam_calibrator.marker_map.save(path)

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

        # Resize to fit the current canvas size while preserving aspect ratio
        canvas_w = self._canvas.winfo_width()
        canvas_h = self._canvas.winfo_height()
        if canvas_w > 1 and canvas_h > 1:
            img_w, img_h = pil_img.size
            scale = min(canvas_w / img_w, canvas_h / img_h)
            new_w = max(1, int(img_w * scale))
            new_h = max(1, int(img_h * scale))
            resample = getattr(Image, "Resampling", Image).BILINEAR
            pil_img = pil_img.resize((new_w, new_h), resample)

        self._tk_image = ImageTk.PhotoImage(image=pil_img)
        cx = canvas_w // 2
        cy = canvas_h // 2
        if self._canvas_image_id is None:
            self._canvas_image_id = self._canvas.create_image(
                cx, cy, anchor="center", image=self._tk_image
            )
        else:
            self._canvas.coords(self._canvas_image_id, cx, cy)
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

        # SLAM: continuous map building
        if self._slam_active and self._slam_calibrator is not None:
            robot_pose = self._slam_calibrator.process_detections(detections)
            self._last_robot_pose = robot_pose
            self._update_slam_display(robot_pose)

    def _update_scenario_display(self, result: OffsetResult) -> None:
        """Refresh the scenario text widget with the latest offset data."""
        lines = []
        dx, dy = result.offset
        lines.append("OFFSET (dx, dy):")
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

    def _update_slam_display(self, robot_pose: RobotPose3D) -> None:
        """Refresh the SLAM tab with robot pose, markers, and 3-D view."""
        if self._slam_calibrator is None:
            return

        # Robot pose
        rx, ry, rz = robot_pose.position
        ro, rp, ryaw = robot_pose.orientation
        vis = robot_pose.visible_markers
        if vis > 0:
            self._slam_robot_var.set(
                f"Pos: ({rx:+.1f}, {ry:+.1f}, {rz:+.1f}) cm\n"
                f"Ori: ({ro:+.1f}, {rp:+.1f}, {ryaw:+.1f})°\n"
                f"Visible markers: {vis}"
            )
        else:
            self._slam_robot_var.set(
                "No mapped markers visible.\n"
                "Move camera to see markers."
            )

        # Marker list
        mmap = self._slam_calibrator.marker_map
        self._slam_markers_list.delete(0, tk.END)
        for m in mmap.markers():
            px, py, pz = m.position
            mr, mp, my = m.orientation
            line = (
                f"{m.marker_id:>4s} "
                f"pos=({px:+.1f},{py:+.1f},{pz:+.1f}) "
                f"ori=({mr:+.1f},{mp:+.1f},{my:+.1f})° "
                f"n={m.observations}"
            )
            self._slam_markers_list.insert(tk.END, line)

        # 3-D visualisation
        cw = self._slam_3d_canvas.winfo_width()
        ch = self._slam_3d_canvas.winfo_height()
        if cw < 2 or ch < 2:
            cw, ch = _VIS3D_WIDTH, _VIS3D_HEIGHT
        scene = render_3d_scene(cw, ch, mmap.markers(), robot_pose)
        self._slam_3d_tk_image = ImageTk.PhotoImage(image=scene)
        cx = cw // 2
        cy = ch // 2
        if self._slam_3d_image_id is None:
            self._slam_3d_image_id = self._slam_3d_canvas.create_image(
                cx, cy, anchor="center", image=self._slam_3d_tk_image
            )
        else:
            self._slam_3d_canvas.coords(self._slam_3d_image_id, cx, cy)
            self._slam_3d_canvas.itemconfigure(
                self._slam_3d_image_id, image=self._slam_3d_tk_image
            )

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
