"""Tkinter-based GUI application for RoboEyeSense.

Layout (normal mode)
---------------------
+-------------+---------------------------+------------------+
|  CONTROLS   |       VIDEO FEED          | INFO/CAM/QUALITY |
|  ---------- |  (annotated frame)        | ---------------- |
|  Quality    |                           | App / FPS / Res  |
|  [combo]    |                           | Quality label    |
|  ---------- |                           | ---------------- |
|  Detectors  |                           | Detected objects |
|  [x] April  |                           | (list)           |
|  [x] QR     |                           | ---------------- |
|  [x] Laser  |                           |  Mode            |
|  ---------- |                           | [Offset][SLAM]   |
|  Parameters |                           | [Follow] (tabs)  |
|  Threshold  |                           | (info in tabs)   |
|  Target area|                           |                  |
|  Sensitivity|                           |                  |
|  [ ] Overlay|                           |                  |
|  ---------- |                           |                  |
|  Mode       |                           |                  |
|  [combo]    |                           |                  |
|  ---------- |                           |                  |
|  Recording  |                           |                  |
|  ---------- |                           |                  |
|[Toggle View]|                           |                  |
| [ ✕ Close ] |                           |                  |
+-------------+---------------------------+------------------+
|  Status bar (FPS | Quality | Detections)                   |
+------------------------------------------------------------+

Compact layout: camera column is fixed-width; info panel expands.
Keyboard shortcuts: Ctrl+1 → Low, Ctrl+2 → Normal, Ctrl+3 → High.

Usage::

    import tkinter as tk
    from robo_vision.camera import Camera
    from robo_vision.detector import RoboEyeDetector
    from robo_vision.gui import RoboEyeSenseApp

    root = tk.Tk()
    cam = Camera()
    detector = RoboEyeDetector()
    app = RoboEyeSenseApp(root, cam, detector)
    app.run()
"""

from __future__ import annotations

import datetime
import math
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk

from . import APP_NAME, __version__
from .camera import Camera
from .auto_scenario import AutoFollowResult, AutoFollowScenario
from .detector import RoboEyeDetector, _compute_orientation
from .marker_map import MarkerPose3D, RobotPose3D, SlamCalibrator
from .offset_scenario import CameraOffsetScenario, OffsetResult
from .recorder import VideoRecorder
from .results import Detection, DetectionMode, DetectionType

# How often (milliseconds) the frame-update callback is rescheduled.
# 16 ms gives a ~60 Hz ceiling; the actual frame rate is limited by the camera.
_UPDATE_INTERVAL_MS = 16  # ~60 Hz ceiling; actual rate is camera-limited

# 3-D visualisation defaults
_VIS3D_WIDTH = 320
_VIS3D_HEIGHT = 320
_VIS3D_BG = "#1a1a2e"
_VIS3D_GRID_COLOR = "#334455"
_VIS3D_MARKER_COLOR = "#00cc66"
_VIS3D_CAMERA_COLOR = "#ff4444"

# Fixed camera-column width used in compact layout mode
_COMPACT_CAMERA_WIDTH = _VIS3D_WIDTH

# Human-readable labels shown in the quality combobox
_QUALITY_DISPLAY: dict[str, DetectionMode] = {
    "Low": DetectionMode.FAST,
    "Normal": DetectionMode.NORMAL,
    "High": DetectionMode.ROBUST,
}
_QUALITY_DISPLAY_INV: dict[DetectionMode, str] = {v: k for k, v in _QUALITY_DISPLAY.items()}

# Short descriptions displayed below the combobox when a quality level is active
_QUALITY_DESCRIPTIONS: dict[DetectionMode, str] = {
    DetectionMode.NORMAL: "Balanced – default detection pipeline.",
    DetectionMode.FAST: "Speed-optimised – frame downscaled 50 %.",
    DetectionMode.ROBUST: "Tracking-optimised – sharpening + Kalman.",
}

# Available mode choices for the mode combobox
_MODE_CHOICES: list[str] = ["Basic", "Offset", "SLAM", "Follow"]


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
        Opened :class:`~robo_vision.camera.Camera` instance.
    detector:
        Configured :class:`~robo_vision.detector.RoboEyeDetector` instance.
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

        # Quality (detection mode)
        initial_quality_label = _QUALITY_DISPLAY_INV.get(detector.mode, "Normal")
        self._quality_var = tk.StringVar(value=initial_quality_label)

        # Active mode
        self._mode_var = tk.StringVar(value="Basic")

        # Tunable parameters
        _laser = detector.laser_detector
        _init_threshold = _laser.brightness_threshold if _laser is not None else 240
        _init_threshold_max = _laser.brightness_threshold_max if _laser is not None else 255
        _init_target_area = _laser.target_area if _laser is not None else 100
        _init_sensitivity = _laser.sensitivity if _laser is not None else 50
        _init_channels = _laser.channels if _laser is not None else "bgr"
        self._laser_threshold = tk.IntVar(value=_init_threshold)
        self._laser_threshold_max = tk.IntVar(value=_init_threshold_max)
        self._laser_target_area = tk.IntVar(value=_init_target_area)
        self._laser_sensitivity = tk.IntVar(value=_init_sensitivity)
        self._laser_ch_r = tk.BooleanVar(value="r" in _init_channels)
        self._laser_ch_g = tk.BooleanVar(value="g" in _init_channels)
        self._laser_ch_b = tk.BooleanVar(value="b" in _init_channels)
        self._show_threshold_overlay = tk.BooleanVar(value=False)

        # Offset mode state
        self._offset_scenario: Optional[CameraOffsetScenario] = None
        self._offset_active = False
        self._last_offset_result: Optional[OffsetResult] = None

        # Auto-follow mode state
        self._auto_scenario: Optional[AutoFollowScenario] = None
        self._auto_active = False
        self._last_auto_result: Optional[AutoFollowResult] = None

        # SLAM state
        self._slam_calibrator: Optional[SlamCalibrator] = None
        self._slam_active = False
        self._last_robot_pose = RobotPose3D()

        # Recording state
        self._recorder: Optional[VideoRecorder] = None
        self._record_path: Optional[str] = initial_record_path

        # Layout state
        self._compact_view = False

        # Build the UI
        self._build_ui()

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Keyboard shortcuts for quick quality switching
        self.root.bind("<Control-Key-1>", lambda _e: self._set_quality(DetectionMode.FAST))
        self.root.bind("<Control-Key-2>", lambda _e: self._set_quality(DetectionMode.NORMAL))
        self.root.bind("<Control-Key-3>", lambda _e: self._set_quality(DetectionMode.ROBUST))

    # ──────────────────────────────────────────────────────────────────────
    # UI construction
    # ──────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        """Construct all widgets and layout."""
        self.root.columnconfigure(0, minsize=200)
        self.root.columnconfigure(1, weight=1)
        self.root.columnconfigure(2, minsize=280)
        self.root.rowconfigure(0, weight=1)

        # Minimum window size so that side panels are never hidden.
        self.root.minsize(780, 480)

        # Left control panel
        ctrl_frame = ttk.Frame(self.root, padding=8, width=200)
        ctrl_frame.grid(row=0, column=0, sticky="nsew")
        ctrl_frame.grid_propagate(False)

        # Scrollable wrapper for control panel contents
        ctrl_canvas = tk.Canvas(ctrl_frame, highlightthickness=0)
        ctrl_scrollbar = ttk.Scrollbar(
            ctrl_frame, orient="vertical", command=ctrl_canvas.yview,
        )
        inner_frame = ttk.Frame(ctrl_canvas)

        inner_frame.bind(
            "<Configure>",
            lambda _e: ctrl_canvas.configure(scrollregion=ctrl_canvas.bbox("all")),
        )
        ctrl_canvas.create_window((0, 0), window=inner_frame, anchor="nw")
        ctrl_canvas.configure(yscrollcommand=ctrl_scrollbar.set)

        ctrl_canvas.pack(side="left", fill="both", expand=True)
        ctrl_scrollbar.pack(side="right", fill="y")

        def _on_ctrl_mousewheel(event: tk.Event) -> None:
            ctrl_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        ctrl_canvas.bind("<MouseWheel>", _on_ctrl_mousewheel)
        inner_frame.bind("<MouseWheel>", _on_ctrl_mousewheel)

        # Centre video canvas
        self._canvas = tk.Canvas(self.root, bg="black")
        self._canvas.grid(row=0, column=1, sticky="nsew")

        # Right info panel
        info_frame = ttk.Frame(self.root, padding=8, width=220)
        info_frame.grid(row=0, column=2, sticky="nsew")
        info_frame.grid_propagate(False)

        self._build_controls(inner_frame)
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

        # ── Quality (detection mode) ──────────────────────────────────────
        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=4)
        ttk.Label(parent, text="Quality").pack(anchor="w")

        quality_combo = ttk.Combobox(
            parent,
            textvariable=self._quality_var,
            values=list(_QUALITY_DISPLAY.keys()),
            state="readonly",
            width=28,
        )
        quality_combo.pack(anchor="w", pady=(2, 0))
        quality_combo.bind("<<ComboboxSelected>>", self._on_quality_change)

        # Description of the currently selected quality level
        initial_desc = _QUALITY_DESCRIPTIONS.get(self.detector.mode, "")
        self._quality_desc_var = tk.StringVar(value=initial_desc)
        self._quality_desc_label = ttk.Label(
            parent,
            textvariable=self._quality_desc_var,
            wraplength=180,
            font=("", 8, "italic"),
        )
        self._quality_desc_label.pack(anchor="w", pady=(2, 0))

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
        ttk.Label(parent, text="Laser threshold min (0–255)").pack(
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

        # Laser threshold max
        ttk.Label(parent, text="Laser threshold max (0–255)").pack(
            anchor="w", pady=(6, 0)
        )
        self._threshold_max_label = ttk.Label(
            parent, text=str(self._laser_threshold_max.get())
        )
        self._threshold_max_label.pack(anchor="e")
        ttk.Scale(
            parent,
            from_=0,
            to=255,
            orient="horizontal",
            variable=self._laser_threshold_max,
            command=self._on_threshold_max_change,
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

        # ── Mode ──────────────────────────────────────────────────────────
        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=8)
        ttk.Label(parent, text="Mode", font=("", 9, "bold")).pack(anchor="w")

        mode_combo = ttk.Combobox(
            parent,
            textvariable=self._mode_var,
            values=_MODE_CHOICES,
            state="readonly",
            width=28,
        )
        mode_combo.pack(anchor="w", pady=(2, 4))
        mode_combo.bind("<<ComboboxSelected>>", self._on_mode_change)

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

        # ── Layout toggle ─────────────────────────────────────────────────
        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=8)
        self._layout_btn_var = tk.StringVar(value="Compact view")
        self._layout_btn = ttk.Button(
            parent,
            textvariable=self._layout_btn_var,
            command=self._toggle_layout,
        )
        self._layout_btn.pack(fill="x", pady=(0, 4))

        # ── Quit button ───────────────────────────────────────────────────
        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=4)
        ttk.Button(parent, text="\u2715 Close", command=self._on_close).pack(fill="x")

    def _build_info_panel(self, parent: ttk.Frame) -> None:
        """Build the right-side information panel."""
        ttk.Label(parent, text="INFO / CAMERA / QUALITY", font=("", 10, "bold")).pack(
            anchor="w", pady=(0, 4)
        )

        # Software name, version, camera parameters and quality — merged section
        ttk.Label(
            parent,
            text=f"{APP_NAME} v{__version__}",
            font=("", 9, "italic"),
        ).pack(anchor="w")

        self._cam_fps_var = tk.StringVar(value="FPS: –")
        self._cam_res_var = tk.StringVar(value="Resolution: –")
        ttk.Label(parent, textvariable=self._cam_fps_var).pack(anchor="w")
        ttk.Label(parent, textvariable=self._cam_res_var).pack(anchor="w")

        initial_label = _QUALITY_DISPLAY_INV.get(self.detector.mode, "Normal")
        self._info_quality_var = tk.StringVar(value=initial_label)
        self._info_quality_label = ttk.Label(
            parent,
            textvariable=self._info_quality_var,
            font=("", 9, "bold"),
            foreground="#336699",
        )
        self._info_quality_label.pack(anchor="w")

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

        # Mode information panel — tabbed notebook
        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=8)
        ttk.Label(parent, text="Mode", font=("", 9, "bold")).pack(
            anchor="w"
        )

        self._mode_notebook = ttk.Notebook(parent)
        self._mode_notebook.pack(fill="both", expand=True)

        # ── Offset tab ────────────────────────────────────────────────────
        offset_tab = ttk.Frame(self._mode_notebook, padding=4)
        self._mode_notebook.add(offset_tab, text="Offset")
        self._build_offset_tab(offset_tab)
        self._offset_tab = offset_tab

        # ── SLAM tab ──────────────────────────────────────────────────────
        slam_tab = ttk.Frame(self._mode_notebook, padding=4)
        self._mode_notebook.add(slam_tab, text="SLAM")
        self._build_slam_tab(slam_tab)
        self._slam_tab = slam_tab

        # ── Follow tab ───────────────────────────────────────────────────
        auto_tab = ttk.Frame(self._mode_notebook, padding=4)
        self._mode_notebook.add(auto_tab, text="Follow")
        self._build_auto_tab(auto_tab)
        self._auto_tab = auto_tab

    def _build_offset_tab(self, parent: ttk.Frame) -> None:
        """Build the Offset mode tab contents."""
        # Controls
        self._offset_capture_btn = ttk.Button(
            parent,
            text="Capture reference",
            command=self._on_offset_capture,
            state="disabled",
        )
        self._offset_capture_btn.pack(fill="x", pady=(0, 2))

        self._offset_reset_btn = ttk.Button(
            parent,
            text="Reset reference",
            command=self._on_offset_reset,
            state="disabled",
        )
        self._offset_reset_btn.pack(fill="x", pady=2)

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=4)

        # Info text area
        mode_frame = ttk.Frame(parent)
        mode_frame.pack(fill="both", expand=True)
        mode_scroll = ttk.Scrollbar(mode_frame, orient="vertical")
        self._mode_text = tk.Text(
            mode_frame,
            yscrollcommand=mode_scroll.set,
            font=("Courier", 8),
            wrap="word",
            height=10,
            state="disabled",
            bg="#f0f0f0",
        )
        mode_scroll.config(command=self._mode_text.yview)
        self._mode_text.pack(side="left", fill="both", expand=True)
        mode_scroll.pack(side="right", fill="y")
        self._set_mode_text("Basic mode.\nSelect 'Offset' mode to begin.")

    def _build_slam_tab(self, parent: ttk.Frame) -> None:
        """Build the SLAM mode tab contents."""
        # Controls at the top
        self._slam_reset_btn = ttk.Button(
            parent,
            text="Reset SLAM",
            command=self._on_slam_reset,
            state="disabled",
        )
        self._slam_reset_btn.pack(fill="x", pady=(0, 2))

        self._slam_save_btn = ttk.Button(
            parent,
            text="Save map…",
            command=self._on_slam_save,
            state="disabled",
        )
        self._slam_save_btn.pack(fill="x", pady=2)

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=4)

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
            wraplength=260,
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

    def _build_auto_tab(self, parent: ttk.Frame) -> None:
        """Build the Follow mode tab contents."""
        # Marker-ID selector
        _marker_frame = ttk.Frame(parent)
        _marker_frame.pack(fill="x", pady=(0, 2))
        ttk.Label(_marker_frame, text="Follow ID:").pack(side="left")
        self._auto_marker_var = tk.StringVar(value="")
        self._auto_marker_entry = ttk.Entry(
            _marker_frame,
            textvariable=self._auto_marker_var,
            width=6,
        )
        self._auto_marker_entry.pack(side="left", padx=(4, 0))
        self._auto_marker_entry.bind(
            "<Return>", self._on_auto_marker_id_change,
        )

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=4)

        ttk.Label(parent, text="Follow vector", font=("", 9, "bold")).pack(
            anchor="w"
        )
        self._auto_info_var = tk.StringVar(
            value="Auto-follow not started.\nClick 'Start auto' to begin."
        )
        ttk.Label(
            parent,
            textvariable=self._auto_info_var,
            font=("Courier", 8),
            wraplength=260,
            justify="left",
        ).pack(anchor="w", pady=(0, 4))

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=4)

        ttk.Label(parent, text="Visible markers", font=("", 9, "bold")).pack(
            anchor="w"
        )
        auto_list_frame = ttk.Frame(parent)
        auto_list_frame.pack(fill="x")
        auto_scroll = ttk.Scrollbar(auto_list_frame, orient="vertical")
        self._auto_markers_list = tk.Listbox(
            auto_list_frame,
            yscrollcommand=auto_scroll.set,
            font=("Courier", 7),
            selectmode=tk.SINGLE,
            height=6,
        )
        auto_scroll.config(command=self._auto_markers_list.yview)
        self._auto_markers_list.pack(side="left", fill="both", expand=True)
        auto_scroll.pack(side="right", fill="y")


    def _set_quality(self, mode: DetectionMode) -> None:
        """Programmatically switch to quality *mode* and update all UI elements."""
        label = _QUALITY_DISPLAY_INV.get(mode, "Normal")
        self._quality_var.set(label)
        self._on_quality_change()

    def _on_quality_change(self, _event: Optional[object] = None) -> None:
        """Switch the detector's quality (operating mode)."""
        label = self._quality_var.get()
        new_mode = _QUALITY_DISPLAY.get(label, DetectionMode.NORMAL)
        self.detector.mode = new_mode
        # Update the description label and info-panel indicator
        self._quality_desc_var.set(_QUALITY_DESCRIPTIONS.get(new_mode, ""))
        self._info_quality_var.set(label)

    # ── Mode callbacks ──────────────────────────────────────────────────────

    def _on_mode_change(self, _event: Optional[object] = None) -> None:
        """Switch to the selected mode."""
        new_mode = self._mode_var.get()
        # Stop all currently active modes
        self._stop_all_modes()
        # Start the newly selected mode
        if new_mode == "Offset":
            self._start_offset_mode()
        elif new_mode == "SLAM":
            self._start_slam_mode()
        elif new_mode == "Follow":
            self._start_follow_mode()

    def _stop_all_modes(self) -> None:
        """Stop every active mode and reset UI to the idle state."""
        if self._offset_active:
            self._offset_active = False
            self._offset_scenario = None
            self._last_offset_result = None
            self._offset_capture_btn.config(state="disabled")
            self._offset_reset_btn.config(state="disabled")
        if self._slam_active:
            self._slam_active = False
            self._slam_calibrator = None
            self._last_robot_pose = RobotPose3D()
            self._slam_reset_btn.config(state="disabled")
            self._slam_save_btn.config(state="disabled")
            self._slam_robot_var.set("Position: –\nOrientation: –")
            self._slam_markers_list.delete(0, tk.END)
        if self._auto_active:
            self._auto_active = False
            self._auto_scenario = None
            self._last_auto_result = None
            self._auto_info_var.set(
                "Follow not started.\n"
                "Select 'Follow' mode to begin."
            )
            self._auto_markers_list.delete(0, tk.END)
        self._set_mode_text("Basic mode.")

    def _start_offset_mode(self) -> None:
        """Activate the offset-calibration mode."""
        self._offset_scenario = CameraOffsetScenario(
            camera=self.camera,
            detector=self.detector,
            frame_width=self.camera.actual_width,
        )
        self._offset_active = True
        self._last_offset_result = None
        self._offset_capture_btn.config(state="normal")
        self._offset_reset_btn.config(state="disabled")
        self._set_mode_text(
            "Offset mode started.\n\n"
            "STEP 1: Position the camera at\n"
            "the REFERENCE position with\n"
            "AprilTags visible.\n\n"
            "Then click 'Capture reference'."
        )
        self._mode_notebook.select(self._offset_tab)

    def _start_slam_mode(self) -> None:
        """Activate the SLAM map-building mode."""
        self._slam_calibrator = SlamCalibrator(tag_size_cm=5.0)
        self._slam_active = True
        self._last_robot_pose = RobotPose3D()
        self._slam_reset_btn.config(state="normal")
        self._slam_save_btn.config(state="normal")
        self._slam_robot_var.set("SLAM started.\nWaiting for markers…")
        self._mode_notebook.select(self._slam_tab)

    def _start_follow_mode(self) -> None:
        """Activate the auto-follow mode."""
        marker_id = self._auto_marker_var.get().strip() or None
        self._auto_scenario = AutoFollowScenario(
            camera=self.camera,
            detector=self.detector,
            frame_width=self.camera.actual_width,
            frame_height=self.camera.actual_height,
            target_marker_id=marker_id,
        )
        self._auto_active = True
        self._last_auto_result = None
        self._auto_info_var.set("Auto-follow started.\nWaiting for markers…")
        self._mode_notebook.select(self._auto_tab)

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
                brightness_threshold_max=self._laser_threshold_max.get(),
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

    def _on_threshold_max_change(self, _value: Optional[str] = None) -> None:
        """Apply the new laser threshold-max value."""
        val = self._laser_threshold_max.get()
        self._threshold_max_label.config(text=str(val))
        laser = self.detector.laser_detector
        if laser is not None:
            laser.brightness_threshold_max = val

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

    # ── Offset callbacks ──────────────────────────────────────────────────

    def _set_mode_text(self, text: str) -> None:
        """Replace the content of the mode text widget."""
        self._mode_text.config(state="normal")
        self._mode_text.delete("1.0", tk.END)
        self._mode_text.insert("1.0", text)
        self._mode_text.config(state="disabled")

    def _on_offset_capture(self) -> None:
        """Capture the current detections as the reference frame."""
        if self._offset_scenario is None:
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
        self._offset_scenario.set_reference(detections)
        self._offset_reset_btn.config(state="normal")

        if april_count == 0:
            self._set_mode_text(
                "WARNING: No AprilTags found in\n"
                "reference frame!\n\n"
                "The offset will be (0, 0).\n"
                "Try repositioning and click\n"
                "'Reset reference' → 'Capture\n"
                "reference' again."
            )
        else:
            self._set_mode_text(
                f"Reference captured:\n"
                f"  {april_count} AprilTag(s) found.\n\n"
                "STEP 2: Move the camera.\n"
                "Offset is computed continuously.\n\n"
                "Waiting for frames…"
            )

    def _on_offset_reset(self) -> None:
        """Reset the offset reference to allow re-capture."""
        if self._offset_scenario is None:
            return
        self._offset_scenario.reset()
        self._last_offset_result = None
        self._offset_reset_btn.config(state="disabled")
        self._set_mode_text(
            "Reference cleared.\n\n"
            "STEP 1: Position the camera at\n"
            "the REFERENCE position with\n"
            "AprilTags visible.\n\n"
            "Then click 'Capture reference'."
        )

    # ── SLAM callbacks ────────────────────────────────────────────────────

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

        maps_dir = Path(__file__).resolve().parent.parent / "maps"
        maps_dir.mkdir(exist_ok=True)
        path = filedialog.asksaveasfilename(
            title="Save marker map as",
            initialdir=str(maps_dir),
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        self._slam_calibrator.marker_map.save(path)

    # ── Auto-follow callbacks ─────────────────────────────────────────────

    def _on_auto_marker_id_change(self, _event: Optional[object] = None) -> None:
        """Update the target marker ID for auto-follow."""
        if self._auto_scenario is None:
            return
        marker_id = self._auto_marker_var.get().strip() or None
        self._auto_scenario.target_marker_id = marker_id

    # ── Recording callbacks ───────────────────────────────────────────────

    def _on_toggle_recording(self) -> None:
        """Start or stop video recording."""
        if self._recorder is not None and self._recorder.is_recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        """Begin recording to a timestamped file in the 'videos' directory."""
        if self._record_path:
            path = self._record_path
            self._record_path = None  # use only once from CLI
        else:
            # Auto-generate a timestamped filename inside the 'videos' folder
            # located in the project root directory (next to main.py).
            video_dir = Path(__file__).resolve().parent.parent / "videos"
            video_dir.mkdir(exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = str(video_dir / f"recording_{ts}.mp4")
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
        quality_label = _QUALITY_DISPLAY_INV.get(self.detector.mode, "")
        rec_label = "  |  ● REC" if (self._recorder is not None and self._recorder.is_recording) else ""
        self._status_var.set(
            f"FPS: {self._fps_display:.1f}  |  Quality: {quality_label}  |  Detections: {n}{rec_label}"
        )

        # Schedule next update
        self.root.after(_UPDATE_INTERVAL_MS, self._update_frame)

    def _update_info_panel(self, detections: List[Detection]) -> None:
        """Refresh the camera-parameters, detections-list, and mode widgets."""
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

        # Offset: continuous offset computation
        if self._offset_active and self._offset_scenario is not None and self._offset_scenario.has_reference:
            try:
                result = self._offset_scenario.compute_offset_from_detections(detections)
                self._last_offset_result = result
                self._update_offset_display(result)
            except RuntimeError:
                pass

        # SLAM: continuous map building
        if self._slam_active and self._slam_calibrator is not None:
            robot_pose = self._slam_calibrator.process_detections(detections)
            self._last_robot_pose = robot_pose
            self._update_slam_display(robot_pose)

        # Auto-follow: continuous vector computation
        if self._auto_active and self._auto_scenario is not None:
            result = self._auto_scenario.compute_from_detections(detections)
            self._last_auto_result = result
            self._update_auto_display(result)

    def _update_offset_display(self, result: OffsetResult) -> None:
        """Refresh the mode text widget with the latest offset data."""
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

        self._set_mode_text("\n".join(lines))

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

    def _update_auto_display(self, result: AutoFollowResult) -> None:
        """Refresh the Auto tab with the latest follow vector data."""
        dx, dy = result.position_vector
        if result.target_found:
            self._auto_info_var.set(
                f"Target: {result.target_marker_id}\n"
                f"Vector: ({dx:+.1f}, {dy:+.1f}) px\n"
                f"Yaw: {result.yaw:+.1f}°"
            )
        else:
            self._auto_info_var.set("No target marker visible.")

        self._auto_markers_list.delete(0, tk.END)
        for mid in result.visible_marker_ids:
            pos = result.marker_positions.get(mid, (0, 0))
            prefix = "→ " if mid == result.target_marker_id else "  "
            self._auto_markers_list.insert(
                tk.END,
                f"{prefix}{mid:>4s}  pos=({pos[0]},{pos[1]})",
            )

    # ──────────────────────────────────────────────────────────────────────
    # Layout management
    # ──────────────────────────────────────────────────────────────────────

    def _toggle_layout(self) -> None:
        """Toggle between normal and compact-camera layouts.

        Normal layout: camera column expands, info panel has fixed width.
        Compact layout: camera column is fixed at *_COMPACT_CAMERA_WIDTH* px,
        info panel expands to use the remaining space.
        """
        self._compact_view = not self._compact_view
        if self._compact_view:
            # Compact: camera fixed; info panel expands
            self.root.columnconfigure(1, minsize=_COMPACT_CAMERA_WIDTH, weight=0)
            self.root.columnconfigure(2, minsize=280, weight=1)
            self._layout_btn_var.set("Normal view")
        else:
            # Normal: camera expands; info panel has a fixed minimum
            self.root.columnconfigure(1, minsize=0, weight=1)
            self.root.columnconfigure(2, minsize=280, weight=0)
            self._layout_btn_var.set("Compact view")

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
