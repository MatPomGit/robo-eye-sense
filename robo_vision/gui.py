"""Tkinter-based GUI application for RoboEyeSense.

Layout (normal mode)
---------------------
+-------------+---------------------------+------------------+
|  CONTROLS   |       VIDEO FEED          |   MODE           |
|  ---------- |  (annotated frame)        | [mode combobox]  |
|  Quality    |                           | ---------------- |
|  [combo]    +---------------------------+ [Offset][SLAM]   |
|  ---------- |  INFO / CAMERA / QUALITY  | [Follow][Calib]  |
|  Detectors  |  App / FPS / Res          | [Box]  [Pose]    |
|  [x] April  |  Quality label            | (tabs, full col) |
|  [x] QR     |  Detected objects (5 rows)|                  |
|  [x] Laser  |                           |                  |
|  ---------- |                           |                  |
|  Parameters |                           |                  |
|  Threshold  |                           |                  |
|  Target area|                           |                  |
|  Sensitivity|                           |                  |
|  [ ] Overlay|                           |                  |
|  ---------- |                           |                  |
|  Recording  |                           |                  |
|  ---------- |                           |                  |
|[Toggle View]|                           |                  |
| [ ✕ Close ] |                           |                  |
+-------------+---------------------------+------------------+
|  Status bar (FPS | Quality | Detections)                   |
+------------------------------------------------------------+

Compact layout: camera is in the upper-right corner; controls, info panel, and
mode panel expand proportionally to fill the remaining space.
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
from typing import Any, List, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk

from . import APP_NAME, __version__
from .camera import Camera
from .auto_scenario import AutoFollowResult, AutoFollowScenario
from .detector import RoboEyeDetector, _compute_orientation
from .marker_map import MarkerPose3D, RobotPose3D, SlamCalibrator
from .offset_scenario import CameraOffsetScenario, OffsetResult
from .overlay import OverlayRenderer
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
_MODE_CHOICES: list[str] = ["Basic", "Offset", "SLAM", "Follow", "Calibration", "Box", "Pose", "MediaPipe"]

# Minimum number of chessboard captures before calibration can be run
_CALIB_MIN_CAPTURES: int = 15

# BGR colours for the mode-indicator crosshair lines drawn on the camera feed
_MODE_LINE_COLORS: dict[str, tuple[int, int, int]] = {
    "Basic":       (128, 128, 128),  # gray
    "Offset":      (255, 100,   0),  # blue
    "SLAM":        (  0, 200,   0),  # green
    "Follow":      (255, 255,   0),  # cyan
    "Calibration": (  0, 200, 200),  # yellow
    "Box":         (  0, 140, 255),  # orange
    "Pose":        (255,   0, 200),  # magenta
    "MediaPipe":   (  0, 255, 128),  # green
}

# UI palette/styling tuned for better readability on bright and dark camera feeds
_PANEL_BG = "#f4f7fb"
_SURFACE_BG = "#ffffff"
_ACCENT = "#1f5fbf"
_MUTED = "#5b6472"
_BORDER = "#d7deea"
_STATUS_BG = "#eaf1ff"
_STATUS_FG = "#163a70"


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

        self.root.title("robot-vision")
        self.root.resizable(True, True)
        self.root.configure(bg=_PANEL_BG)


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

        # Calibration mode state
        self._calibration_mode_obj: Optional[Any] = None
        self._calibration_active = False
        self._calib_capture_flag = False
        self._calib_run_flag = False

        # Box detection mode state
        self._box_mode_obj: Optional[Any] = None
        self._box_active = False

        # Pose estimation mode state
        self._pose_mode_obj: Optional[Any] = None
        self._pose_active = False
        self._pose_sensitivity_var = tk.IntVar(value=50)

        # MediaPipe pose mode state
        self._mediapipe_mode_obj: Optional[Any] = None
        self._mediapipe_active = False

        # Recording state
        self._recorder: Optional[VideoRecorder] = None
        self._record_path: Optional[str] = initial_record_path

        # Layout state
        self._compact_view = False

        # Notebook ↔ combobox sync flag (prevents infinite feedback loop)
        self._mode_changing = False

        # Camera resolution/FPS settings
        _res_choices = ["320x240", "640x480", "1280x720", "1920x1080"]
        _res_init = f"{camera.actual_width}x{camera.actual_height}"
        self._res_var = tk.StringVar(
            value=_res_init if _res_init in _res_choices else "640x480"
        )
        self._fps_target_var = tk.StringVar(value="30")

        # On-screen overlay renderer
        self._overlay = OverlayRenderer(
            enabled=True,
            mode=self._mode_var.get().lower(),
            quality=self._quality_var.get().lower(),
            enabled_detectors=self._initial_detector_names(),
        )

        # Build the UI
        self._configure_styles()
        self._build_ui()

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Keyboard shortcuts for quick quality switching
        self.root.bind("<Control-Key-1>", lambda _e: self._set_quality(DetectionMode.FAST))
        self.root.bind("<Control-Key-2>", lambda _e: self._set_quality(DetectionMode.NORMAL))
        self.root.bind("<Control-Key-3>", lambda _e: self._set_quality(DetectionMode.ROBUST))

    # ──────────────────────────────────────────────────────────────────────
    # Overlay helpers
    # ──────────────────────────────────────────────────────────────────────

    def _initial_detector_names(self) -> List[str]:
        """Return a list of enabled detector display names at startup."""
        names: List[str] = []
        if self._enable_april.get():
            names.append("AprilTags")
        if self._enable_qr.get():
            names.append("QR codes")
        if self._enable_laser.get():
            names.append("Laser spots")
        return names

    def _overlay_detector_names(self) -> List[str]:
        """Return the current list of enabled detector display names."""
        names: List[str] = []
        if self.detector.april_enabled:
            names.append("AprilTags")
        if self.detector.qr_enabled:
            names.append("QR codes")
        if self.detector.laser_enabled:
            names.append("Laser spots")
        return names

    def _configure_styles(self) -> None:
        """Configure a calmer ttk theme with clearer section hierarchy."""
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        self.root.option_add("*Font", "TkDefaultFont 10")
        self.root.option_add("*TCombobox*Listbox.font", "TkDefaultFont 10")

        style.configure("App.TFrame", background=_PANEL_BG)
        style.configure("Panel.TFrame", background=_SURFACE_BG, relief="solid", borderwidth=1)
        style.configure("Section.TLabelframe", background=_SURFACE_BG, relief="solid", borderwidth=1)
        style.configure("Section.TLabelframe.Label", background=_SURFACE_BG, foreground=_ACCENT, font=("", 10, "bold"))
        style.configure("Header.TLabel", background=_SURFACE_BG, foreground=_ACCENT, font=("", 11, "bold"))
        style.configure("Body.TLabel", background=_SURFACE_BG, foreground="#1f2937")
        style.configure("Hint.TLabel", background=_SURFACE_BG, foreground=_MUTED, font=("", 8, "italic"))
        style.configure("Value.TLabel", background=_SURFACE_BG, foreground="#0f172a", font=("", 9, "bold"))
        style.configure("Status.TLabel", background=_STATUS_BG, foreground=_STATUS_FG, font=("", 9, "bold"), padding=(10, 6))
        style.configure("TCheckbutton", background=_SURFACE_BG)
        style.configure("TRadiobutton", background=_SURFACE_BG)
        style.configure("TNotebook", background=_SURFACE_BG, borderwidth=0)
        style.configure("TNotebook.Tab", padding=(10, 5))
        style.configure("TButton", padding=(8, 5))
        style.map("Accent.TButton", background=[("active", "#2d74e0"), ("!disabled", _ACCENT)], foreground=[("!disabled", "white")])

    def _make_section(self, parent: ttk.Frame, title: str) -> ttk.LabelFrame:
        """Create a consistently styled section container."""
        section = ttk.LabelFrame(parent, text=title, style="Section.TLabelframe", padding=(10, 8))
        section.pack(fill="x", pady=(0, 8))
        return section

    # ──────────────────────────────────────────────────────────────────────
    # UI construction
    # ──────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        """Construct all widgets and layout."""
        # Normal mode: 3 cols (ctrl | camera+info | mode), 2 content rows
        self.root.columnconfigure(0, minsize=200, weight=0)
        self.root.columnconfigure(1, weight=1)
        self.root.columnconfigure(2, minsize=280, weight=0)
        self.root.rowconfigure(0, weight=1)   # camera row expands
        self.root.rowconfigure(1, weight=0)   # info row shrinks to content

        # Minimum window size so that side panels are never hidden.
        self.root.minsize(780, 480)

        # Left control panel (spans both content rows)
        self._ctrl_frame = ttk.Frame(self.root, padding=6, width=220, style="Panel.TFrame")
        self._ctrl_frame.grid(row=0, column=0, rowspan=2, sticky="nsew")
        self._ctrl_frame.pack_propagate(False)

        # Scrollable wrapper for control panel contents
        ctrl_canvas = tk.Canvas(self._ctrl_frame, highlightthickness=0, bg=_SURFACE_BG)
        ctrl_scrollbar = ttk.Scrollbar(
            self._ctrl_frame, orient="vertical", command=ctrl_canvas.yview,
        )
        inner_frame = ttk.Frame(ctrl_canvas, style="Panel.TFrame")

        inner_frame.bind(
            "<Configure>",
            lambda _e: ctrl_canvas.configure(scrollregion=ctrl_canvas.bbox("all")),
        )
        _ctrl_inner_win = ctrl_canvas.create_window((0, 0), window=inner_frame, anchor="nw")
        ctrl_canvas.bind(
            "<Configure>",
            lambda e: ctrl_canvas.itemconfig(_ctrl_inner_win, width=e.width),
        )
        ctrl_canvas.configure(yscrollcommand=ctrl_scrollbar.set)

        ctrl_canvas.pack(side="left", fill="both", expand=True)
        ctrl_scrollbar.pack(side="right", fill="y")

        def _on_ctrl_mousewheel(event: tk.Event) -> None:
            ctrl_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        ctrl_canvas.bind("<MouseWheel>", _on_ctrl_mousewheel)
        inner_frame.bind("<MouseWheel>", _on_ctrl_mousewheel)

        # Centre video canvas (row=0, col=1 in normal mode)
        self._canvas = tk.Canvas(self.root, bg="#10131a", highlightthickness=0)
        self._canvas.grid(row=0, column=1, sticky="nsew")

        # Info panel — below the camera in col=1 (row=1)
        self._info_frame = ttk.Frame(self.root, padding=8, style="Panel.TFrame")
        self._info_frame.grid(row=1, column=1, sticky="nsew")

        # Right mode panel — full right column (col=2, rowspan=2)
        self._mode_frame = ttk.Frame(self.root, padding=8, width=300, style="Panel.TFrame")
        self._mode_frame.grid(row=0, column=2, rowspan=2, sticky="nsew")
        self._mode_frame.pack_propagate(False)

        self._build_controls(inner_frame)
        self._build_info_panel(self._info_frame)
        self._build_mode_panel(self._mode_frame)

        # Status bar at the bottom (row=2 — below both content rows)
        self._status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(
            self.root,
            textvariable=self._status_var,
            style="Status.TLabel",
            anchor="w",
        )
        status_bar.grid(row=2, column=0, columnspan=3, sticky="ew")

    def _build_controls(self, parent: ttk.Frame) -> None:
        """Build the left-side control panel."""
        ttk.Label(parent, text="Controls", style="Header.TLabel").pack(
            anchor="w", pady=(0, 8)
        )

        quality_section = self._make_section(parent, "Detection quality")
        ttk.Label(quality_section, text="Preset", style="Body.TLabel").pack(anchor="w")

        quality_combo = ttk.Combobox(
            quality_section,
            textvariable=self._quality_var,
            values=list(_QUALITY_DISPLAY.keys()),
            state="readonly",
            width=28,
        )
        quality_combo.pack(fill="x", pady=(2, 0))
        quality_combo.bind("<<ComboboxSelected>>", self._on_quality_change)

        # Description of the currently selected quality level
        initial_desc = _QUALITY_DESCRIPTIONS.get(self.detector.mode, "")
        self._quality_desc_var = tk.StringVar(value=initial_desc)
        self._quality_desc_label = ttk.Label(
            quality_section,
            textvariable=self._quality_desc_var,
            wraplength=180,
            font=("", 8, "italic"),
        )
        self._quality_desc_label.pack(anchor="w", pady=(4, 0))

        # Keyboard shortcut hint
        ttk.Label(
            quality_section,
            text="Ctrl+1 / 2 / 3",
            style="Hint.TLabel",
        ).pack(anchor="w", pady=(2, 0))

        detector_section = self._make_section(parent, "Active detectors")
        ttk.Label(detector_section, text="Choose which algorithms are running.", style="Hint.TLabel").pack(anchor="w", pady=(0, 4))

        self._april_cb = ttk.Checkbutton(
            detector_section,
            text="AprilTag",
            variable=self._enable_april,
            command=self._on_toggle_april,
        )
        self._april_cb.pack(anchor="w")

        ttk.Checkbutton(
            detector_section,
            text="QR Code",
            variable=self._enable_qr,
            command=self._on_toggle_qr,
        ).pack(anchor="w")

        ttk.Checkbutton(
            detector_section,
            text="Laser Spot",
            variable=self._enable_laser,
            command=self._on_toggle_laser,
        ).pack(anchor="w")

        parameter_section = self._make_section(parent, "Laser parameters")
        ttk.Label(parameter_section, text="Tune the spot detector and preview thresholding.", style="Hint.TLabel").pack(anchor="w", pady=(0, 4))

        # Laser threshold
        ttk.Label(parameter_section, text="Laser threshold min (0–255)", 
        font=("", 8, "italic")
        ).pack(
            anchor="w", pady=(1, 0)
        )

        self._threshold_label = ttk.Label(
            parameter_section, text=str(self._laser_threshold.get()), style="Value.TLabel",
        font=("", 8, "italic")
        
        )
        self._threshold_label.pack(anchor="e")
        ttk.Scale(
            parameter_section,
            from_=0,
            to=255,
            orient="horizontal",
            variable=self._laser_threshold,
            command=self._on_threshold_change,
        ).pack(fill="x")

        # Laser threshold max
        ttk.Label(parameter_section, text="Laser threshold max (0–255)", 
        font=("", 8, "italic")
        ).pack(
            anchor="w", pady=(1, 0)
        )
        self._threshold_max_label = ttk.Label(
            parameter_section, text=str(self._laser_threshold_max.get()), style="Value.TLabel",
        font=("", 8, "italic")
        
        )
        self._threshold_max_label.pack(anchor="e")
        ttk.Scale(
            parameter_section,
            from_=0,
            to=255,
            orient="horizontal",
            variable=self._laser_threshold_max,
            command=self._on_threshold_max_change,
        ).pack(fill="x")

        # Laser target area
        ttk.Label(parameter_section, text="Laser target area (px)", 
        font=("", 8, "italic")
        ).pack(
            anchor="w", pady=(2, 0)
        )
        self._target_area_label = ttk.Label(
            parameter_section, text=str(self._laser_target_area.get()), style="Value.TLabel",
        font=("", 8, "italic")
        
        )
        self._target_area_label.pack(anchor="e")
        ttk.Scale(
            parameter_section,
            from_=4,
            to=2000,
            orient="horizontal",
            variable=self._laser_target_area,
            command=self._on_target_area_change,
        ).pack(fill="x")

        # Laser sensitivity
        ttk.Label(parameter_section, text="Laser sensitivity (0–100)", 
        font=("", 8, "italic")
        ).pack(
            anchor="w", pady=(2, 0)
        )
        self._sensitivity_label = ttk.Label(
            parameter_section, text=str(self._laser_sensitivity.get()), style="Value.TLabel",
        font=("", 8, "italic")
        
        )
        self._sensitivity_label.pack(anchor="e")
        ttk.Scale(
            parameter_section,
            from_=0,
            to=100,
            orient="horizontal",
            variable=self._laser_sensitivity,
            command=self._on_sensitivity_change,
        ).pack(fill="x")

        # Laser channel selection
        ttk.Label(parameter_section, text="Laser channels", style="Body.TLabel").pack(
            anchor="w", pady=(1, 0)
        )
        _ch_frame = ttk.Frame(parameter_section, style="Panel.TFrame")
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
            parameter_section,
            text="Show threshold overlay",
            variable=self._show_threshold_overlay,
        ).pack(anchor="w", pady=(1, 0))

        recording_section = self._make_section(parent, "Recording")

        self._record_btn = ttk.Button(
            recording_section,
            style="Accent.TButton",
            text="Start recording",
            command=self._on_toggle_recording,
        )
        self._record_btn.pack(fill="x", pady=(2, 1))

        self._record_status_var = tk.StringVar(value="Not recording")
        ttk.Label(
            recording_section,
            textvariable=self._record_status_var,
            style="Hint.TLabel",
            wraplength=180,
        ).pack(anchor="w")

        camera_section = self._make_section(parent, "Camera")

        res_frame = ttk.Frame(camera_section, style="Panel.TFrame")
        res_frame.pack(fill="x", pady=(1, 0))
        ttk.Label(res_frame, text="Resolution:").pack(side="left")
        ttk.Combobox(
            res_frame,
            textvariable=self._res_var,
            values=["320x240", "640x480", "1280x720", "1920x1080"],
            state="readonly",
            width=11,
        ).pack(side="left", padx=(4, 0))

        fps_frame = ttk.Frame(camera_section, style="Panel.TFrame")
        fps_frame.pack(fill="x", pady=(1, 0))
        ttk.Label(fps_frame, text="Target FPS:").pack(side="left")
        ttk.Combobox(
            fps_frame,
            textvariable=self._fps_target_var,
            values=["15", "24", "30", "60"],
            state="readonly",
            width=5,
        ).pack(side="left", padx=(4, 0))

        ttk.Button(
            camera_section,
            text="Apply camera settings",
            command=self._on_apply_camera_settings,
        ).pack(fill="x", pady=(2, 1))

        layout_section = self._make_section(parent, "Workspace")
        ttk.Label(layout_section, text="Switch between focus-on-video and full control layouts.", style="Hint.TLabel").pack(anchor="w", pady=(0, 4))
        self._layout_btn_var = tk.StringVar(value="Compact view")
        self._layout_btn = ttk.Button(
            layout_section,
            textvariable=self._layout_btn_var,
            command=self._toggle_layout,
        )
        self._layout_btn.pack(fill="x", pady=(0, 2))

        action_section = self._make_section(parent, "Application")
        ttk.Button(action_section, text="\u2715 Close", command=self._on_close).pack(fill="x")

    def _build_info_panel(self, parent: ttk.Frame) -> None:
        """Build the info / camera / quality panel (below the video in normal mode)."""
        ttk.Label(parent, text="Session overview", style="Header.TLabel").pack(
            anchor="w", pady=(0, 6)
        )

        # Software name, version, camera parameters and quality — merged section
        ttk.Label(
            parent,
            text=f"{APP_NAME} v{__version__}",
            style="Hint.TLabel",
        ).pack(anchor="w")

        self._cam_fps_var = tk.StringVar(value="FPS: –")
        self._cam_res_var = tk.StringVar(value="Resolution: –")
        summary_section = self._make_section(parent, "Camera summary")
        ttk.Label(summary_section, textvariable=self._cam_fps_var, style="Body.TLabel").pack(anchor="w")
        ttk.Label(summary_section, textvariable=self._cam_res_var, style="Body.TLabel").pack(anchor="w")

        initial_label = _QUALITY_DISPLAY_INV.get(self.detector.mode, "Normal")
        self._info_quality_var = tk.StringVar(value=initial_label)
        self._info_quality_label = ttk.Label(
            summary_section,
            textvariable=self._info_quality_var,
            style="Value.TLabel",
        )
        self._info_quality_label.pack(anchor="w", pady=(4, 0))

        list_section = self._make_section(parent, "Detected objects")
        ttk.Label(list_section, text="Latest recognitions from the current frame.", style="Hint.TLabel").pack(anchor="w", pady=(0, 4))

        list_frame = ttk.Frame(list_section, style="Panel.TFrame")
        list_frame.pack(fill="x")
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical")
        self._detections_list = tk.Listbox(
            list_frame,
            yscrollcommand=scrollbar.set,
            font=("Courier", 8),
            selectmode=tk.SINGLE,
            height=5,
        )
        scrollbar.config(command=self._detections_list.yview)
        self._detections_list.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _build_mode_panel(self, parent: ttk.Frame) -> None:
        """Build the right-side mode panel (mode selector + tabbed notebook)."""
        ttk.Label(parent, text="Operating modes", style="Header.TLabel").pack(
            anchor="w", pady=(0, 6)
        )

        mode_combo = ttk.Combobox(
            parent,
            textvariable=self._mode_var,
            values=_MODE_CHOICES,
            state="readonly",
            width=28,
        )
        mode_combo.pack(fill="x", pady=(0, 6))
        mode_combo.bind("<<ComboboxSelected>>", self._on_mode_change)

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=4)

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

        # ── Calibration tab ───────────────────────────────────────────────
        calib_tab = ttk.Frame(self._mode_notebook, padding=4)
        self._mode_notebook.add(calib_tab, text="Calibration")
        self._build_calibration_tab(calib_tab)
        self._calibration_tab = calib_tab

        # ── Box tab ────────────────────────────────────────────────────────
        box_tab = ttk.Frame(self._mode_notebook, padding=4)
        self._mode_notebook.add(box_tab, text="Box")
        self._build_box_tab(box_tab)
        self._box_tab = box_tab

        # ── Pose tab ──────────────────────────────────────────────────────
        pose_tab = ttk.Frame(self._mode_notebook, padding=4)
        self._mode_notebook.add(pose_tab, text="Pose")
        self._build_pose_tab(pose_tab)
        self._pose_tab = pose_tab

        # ── MediaPipe tab ─────────────────────────────────────────────────
        mediapipe_tab = ttk.Frame(self._mode_notebook, padding=4)
        self._mode_notebook.add(mediapipe_tab, text="MediaPipe")
        self._build_mediapipe_tab(mediapipe_tab)
        self._mediapipe_tab = mediapipe_tab

        # Bind AFTER all tabs are added so that initial tab selection
        # during construction does not trigger the handler.
        self._mode_notebook.bind(
            "<<NotebookTabChanged>>", self._on_notebook_tab_changed
        )

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

    def _build_calibration_tab(self, parent: ttk.Frame) -> None:
        """Build the Calibration mode tab contents."""
        # Chessboard size
        cb_frame = ttk.Frame(parent)
        cb_frame.pack(fill="x", pady=(0, 2))
        ttk.Label(cb_frame, text="Board (cols×rows):").pack(side="left")
        self._calib_cols_var = tk.StringVar(value="9")
        self._calib_rows_var = tk.StringVar(value="6")
        ttk.Entry(cb_frame, textvariable=self._calib_cols_var, width=3).pack(
            side="left", padx=(4, 0)
        )
        ttk.Label(cb_frame, text="×").pack(side="left")
        ttk.Entry(cb_frame, textvariable=self._calib_rows_var, width=3).pack(
            side="left"
        )

        # Output path
        out_frame = ttk.Frame(parent)
        out_frame.pack(fill="x", pady=(0, 4))
        ttk.Label(out_frame, text="Output:").pack(side="left")
        self._calib_output_var = tk.StringVar(value="calibration.npz")
        ttk.Entry(
            out_frame, textvariable=self._calib_output_var, width=18,
        ).pack(side="left", padx=(4, 0), fill="x", expand=True)

        # Action buttons
        self._calib_capture_btn = ttk.Button(
            parent,
            text="Capture frame",
            command=self._on_calib_capture,
            state="disabled",
        )
        self._calib_capture_btn.pack(fill="x", pady=(0, 2))

        self._calib_run_btn = ttk.Button(
            parent,
            text="Run calibration",
            command=self._on_calib_run,
            state="disabled",
        )
        self._calib_run_btn.pack(fill="x", pady=2)

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=4)

        # Status display
        self._calib_status_var = tk.StringVar(
            value="Select 'Calibration' mode to begin."
        )
        ttk.Label(
            parent,
            textvariable=self._calib_status_var,
            font=("Courier", 8),
            wraplength=260,
            justify="left",
        ).pack(anchor="w")


    def _build_box_tab(self, parent: ttk.Frame) -> None:
        """Build the Box detection mode tab contents."""
        ttk.Label(
            parent,
            text="Box / cuboid detection",
            font=("", 9, "bold"),
        ).pack(anchor="w", pady=(0, 4))

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=4)

        self._box_status_var = tk.StringVar(
            value="Box mode not started.\nSelect 'Box' mode to begin."
        )
        ttk.Label(
            parent,
            textvariable=self._box_status_var,
            font=("Courier", 8),
            wraplength=260,
            justify="left",
        ).pack(anchor="w")

    def _build_pose_tab(self, parent: ttk.Frame) -> None:
        """Build the Pose estimation mode tab contents."""
        ttk.Label(
            parent,
            text="AprilTag 6-DoF pose estimation",
            font=("", 9, "bold"),
        ).pack(anchor="w", pady=(0, 4))

        # Tag size
        size_frame = ttk.Frame(parent)
        size_frame.pack(fill="x", pady=(0, 4))
        ttk.Label(size_frame, text="Tag size (m):").pack(side="left")
        self._pose_tag_size_var = tk.StringVar(value="0.05")
        ttk.Entry(
            size_frame, textvariable=self._pose_tag_size_var, width=6
        ).pack(side="left", padx=(4, 0))

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=4)

        # Sensitivity slider
        sens_frame = ttk.Frame(parent)
        sens_frame.pack(fill="x", pady=(0, 4))
        ttk.Label(sens_frame, text="Sensitivity (0–100)").pack(
            side="left", anchor="w"
        )
        self._pose_sensitivity_label = ttk.Label(
            sens_frame,
            text=str(self._pose_sensitivity_var.get()),
            width=4,
            anchor="e",
        )
        self._pose_sensitivity_label.pack(side="right")
        ttk.Scale(
            parameter_section,
            from_=0,
            to=100,
            orient="horizontal",
            variable=self._pose_sensitivity_var,
            command=self._on_pose_sensitivity_change,
        ).pack(fill="x", pady=(0, 4))

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=4)

        self._pose_status_var = tk.StringVar(
            value="Pose mode not started.\nSelect 'Pose' mode to begin."
        )
        ttk.Label(
            parent,
            textvariable=self._pose_status_var,
            font=("Courier", 8),
            wraplength=260,
            justify="left",
        ).pack(anchor="w")

    def _build_mediapipe_tab(self, parent: ttk.Frame) -> None:
        """Build the MediaPipe pose-detection mode tab contents."""
        ttk.Label(
            parent,
            text="MediaPipe body pose detection",
            font=("", 9, "bold"),
        ).pack(anchor="w", pady=(0, 4))

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=4)

        # Max poses slider
        poses_frame = ttk.Frame(parent)
        poses_frame.pack(fill="x", pady=(0, 4))
        ttk.Label(poses_frame, text="Max poses (1–5):").pack(side="left")
        self._mediapipe_num_poses_var = tk.IntVar(value=3)
        self._mediapipe_num_poses_label = ttk.Label(
            poses_frame,
            text=str(self._mediapipe_num_poses_var.get()),
            width=3,
            anchor="e",
        )
        self._mediapipe_num_poses_label.pack(side="right")
        ttk.Scale(
            parent,
            from_=1,
            to=5,
            orient="horizontal",
            variable=self._mediapipe_num_poses_var,
            command=self._on_mediapipe_num_poses_change,
        ).pack(fill="x", pady=(0, 4))

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=4)

        self._mediapipe_status_var = tk.StringVar(
            value="MediaPipe mode not started.\nSelect 'MediaPipe' mode to begin."
        )
        ttk.Label(
            parent,
            textvariable=self._mediapipe_status_var,
            font=("Courier", 8),
            wraplength=260,
            justify="left",
        ).pack(anchor="w")


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
        self._overlay.quality = label.lower()

    # ── Mode callbacks ──────────────────────────────────────────────────────

    def _on_mode_change(self, _event: Optional[object] = None) -> None:
        """Switch to the selected mode."""
        new_mode = self._mode_var.get()
        self._mode_changing = True
        self._overlay.mode = new_mode.lower()
        try:
            # Stop all currently active modes
            self._stop_all_modes()
            # Start the newly selected mode
            if new_mode == "Offset":
                self._start_offset_mode()
            elif new_mode == "SLAM":
                self._start_slam_mode()
            elif new_mode == "Follow":
                self._start_follow_mode()
            elif new_mode == "Calibration":
                self._start_calibration_mode()
            elif new_mode == "Box":
                self._start_box_mode()
            elif new_mode == "Pose":
                self._start_pose_mode()
            elif new_mode == "MediaPipe":
                self._start_mediapipe_mode()
        finally:
            self._mode_changing = False

    # Ordered list of mode names corresponding to notebook tab indices 0..6
    _NOTEBOOK_TAB_MODES: List[str] = [
        "Offset", "SLAM", "Follow", "Calibration", "Box", "Pose", "MediaPipe"
    ]

    def _on_notebook_tab_changed(self, _event: Optional[object] = None) -> None:
        """Sync the mode combobox when the user manually selects a notebook tab."""
        if self._mode_changing:
            return
        try:
            tab_idx = self._mode_notebook.index(self._mode_notebook.select())
        except Exception:
            return
        if 0 <= tab_idx < len(self._NOTEBOOK_TAB_MODES):
            new_mode = self._NOTEBOOK_TAB_MODES[tab_idx]
            if new_mode != self._mode_var.get():
                self._mode_var.set(new_mode)
                self._on_mode_change()

    def _on_apply_camera_settings(self) -> None:
        """Apply the selected resolution and FPS to the camera."""
        res = self._res_var.get()
        w: Optional[int] = None
        h: Optional[int] = None
        if "x" in res:
            try:
                w, h = (int(v) for v in res.split("x", 1))
            except ValueError:
                pass
        fps: Optional[int] = None
        try:
            fps = int(self._fps_target_var.get())
        except ValueError:
            pass
        self.camera.set_capture_properties(width=w, height=h, fps=fps)

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
        if self._calibration_active:
            self._calibration_active = False
            self._calibration_mode_obj = None
            self._calib_capture_flag = False
            self._calib_run_flag = False
            self._calib_capture_btn.config(state="disabled")
            self._calib_run_btn.config(state="disabled")
            self._calib_status_var.set("Select 'Calibration' mode to begin.")
        if self._box_active:
            self._box_active = False
            self._box_mode_obj = None
            self._box_status_var.set(
                "Box mode not started.\nSelect 'Box' mode to begin."
            )
        if self._pose_active:
            self._pose_active = False
            self._pose_mode_obj = None
            self._pose_status_var.set(
                "Pose mode not started.\nSelect 'Pose' mode to begin."
            )
        if self._mediapipe_active:
            self._mediapipe_active = False
            self._mediapipe_mode_obj = None
            self._mediapipe_status_var.set(
                "MediaPipe mode not started.\nSelect 'MediaPipe' mode to begin."
            )
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

    def _start_calibration_mode(self) -> None:
        """Activate the chessboard camera-calibration mode."""
        try:
            from modes.calibration_mode import CalibrationMode as _CalibrationMode
        except ImportError:
            self._calib_status_var.set(
                "Error: CalibrationMode not available.\n"
                "Ensure 'modes' package is on the Python path."
            )
            return
        try:
            cols = int(self._calib_cols_var.get())
            rows = int(self._calib_rows_var.get())
        except ValueError:
            cols, rows = 9, 6
        output = self._calib_output_var.get().strip() or "calibration.npz"
        self._calibration_mode_obj = _CalibrationMode(
            chessboard_size=(cols, rows),
            output_path=output,
        )
        self._calibration_active = True
        self._calib_capture_btn.config(state="normal")
        self._calib_run_btn.config(state="disabled")
        status = (
            f"Calibration started (board {cols}×{rows}).\n"
            "Hold the chessboard in view and\n"
            "click 'Capture frame' to collect\n"
            f"samples (need {_CALIB_MIN_CAPTURES}/25)."
        )
        existing = self._load_calib_file_info(output)
        if existing:
            status += f"\n\nExisting calibration loaded from:\n{output}\n{existing}"
        self._calib_status_var.set(status)
        self._mode_notebook.select(self._calibration_tab)

    @staticmethod
    def _load_calib_file_info(path: str) -> str:
        """Return a human-readable summary of an existing .npz calibration file.

        Returns an empty string when the file does not exist or cannot be read.
        """
        calib_path = Path(path)
        if not calib_path.exists():
            return ""
        try:
            data = np.load(str(calib_path))
            lines: list[str] = []
            if "camera_matrix" in data:
                cm = data["camera_matrix"]
                fx = float(cm[0, 0])
                fy = float(cm[1, 1])
                cx = float(cm[0, 2])
                cy = float(cm[1, 2])
                lines.append(
                    f"fx={fx:.2f}  fy={fy:.2f}\ncx={cx:.2f}  cy={cy:.2f}"
                )
            if "dist_coeffs" in data:
                dc = data["dist_coeffs"].flatten()
                vals = "  ".join(f"{v:.4f}" for v in dc)
                lines.append(f"dist: {vals}")
            return "\n".join(lines) if lines else ""
        except Exception:
            return ""

    def _start_box_mode(self) -> None:
        """Activate box-detection mode."""
        try:
            from modes.box_mode import BoxMode as _BoxMode
        except ImportError:
            self._box_status_var.set(
                "Error: BoxMode not available.\n"
                "Ensure 'modes' package is on the Python path."
            )
            return
        self._box_mode_obj = _BoxMode()
        self._box_active = True
        self._box_status_var.set("Box mode active.\nDetected boxes: 0")
        self._mode_notebook.select(self._box_tab)

    def _start_pose_mode(self) -> None:
        """Activate pose-estimation mode."""
        try:
            from modes.pose_mode import PoseMode as _PoseMode
        except ImportError:
            self._pose_status_var.set(
                "Error: PoseMode not available.\n"
                "Ensure 'modes' package is on the Python path."
            )
            return
        try:
            tag_size = float(self._pose_tag_size_var.get())
        except ValueError:
            tag_size = 0.05
        sensitivity = self._pose_sensitivity_var.get()
        self._pose_mode_obj = _PoseMode(tag_size=tag_size, sensitivity=sensitivity)
        self._pose_active = True
        self._pose_status_var.set("Pose mode active.\nWaiting for AprilTags…")
        self._mode_notebook.select(self._pose_tab)

    def _on_pose_sensitivity_change(self, _value: Optional[str] = None) -> None:
        """Update the sensitivity label and restart pose mode if active."""
        val = self._pose_sensitivity_var.get()
        self._pose_sensitivity_label.config(text=str(val))
        if self._pose_active:
            self._start_pose_mode()

    def _start_mediapipe_mode(self) -> None:
        """Activate the MediaPipe pose-detection mode."""
        try:
            from modes.mediapipe_mode import MediaPipeMode as _MediaPipeMode
        except ImportError:
            self._mediapipe_status_var.set(
                "Error: MediaPipeMode not available.\n"
                "Ensure 'modes' package is on the Python path."
            )
            return
        num_poses = self._mediapipe_num_poses_var.get()
        self._mediapipe_mode_obj = _MediaPipeMode(num_poses=num_poses)
        self._mediapipe_active = True
        self._mediapipe_status_var.set(
            "MediaPipe mode active.\nDetected poses: 0"
        )
        self._mode_notebook.select(self._mediapipe_tab)

    def _on_mediapipe_num_poses_change(
        self, _value: Optional[str] = None
    ) -> None:
        """Update the max-poses label and restart MediaPipe mode if active."""
        val = self._mediapipe_num_poses_var.get()
        self._mediapipe_num_poses_label.config(text=str(val))
        if self._mediapipe_active:
            self._start_mediapipe_mode()

    def _on_calib_capture(self) -> None:
        """Request a chessboard-capture on the next frame."""
        self._calib_capture_flag = True

    def _on_calib_run(self) -> None:
        """Request running the calibration on the next frame."""
        self._calib_run_flag = True

    def _update_calibration_display(self) -> None:
        """Refresh the calibration tab status label."""
        if self._calibration_mode_obj is None:
            return
        count = self._calibration_mode_obj.capture_count
        if self._calibration_mode_obj.is_calibrated:
            self._calib_status_var.set(
                f"Calibration complete!\n"
                f"Saved to: {self._calib_output_var.get()}"
            )
            self._calib_capture_btn.config(state="disabled")
            self._calib_run_btn.config(state="disabled")
        elif count >= _CALIB_MIN_CAPTURES:
            self._calib_status_var.set(
                f"Captures: {count}/25\n"
                "Ready to calibrate.\n"
                "Click 'Run calibration'."
            )
            self._calib_run_btn.config(state="normal")
        else:
            self._calib_status_var.set(
                f"Captures: {count}/25\n"
                "Hold chessboard in view and\n"
                "click 'Capture frame'."
            )

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
        actual_fps = self.camera.actual_fps
        fps = float(actual_fps) if isinstance(actual_fps, (int, float)) and actual_fps > 0 else 30.0
        try:
            self._recorder = VideoRecorder(
                path,
                width=self.camera.actual_width,
                height=self.camera.actual_height,
                fps=fps,
            )
            self._recorder.start()
            self._record_btn.config(text="Stop recording")
            self._record_status_var.set(f"Recording: {path}")
        except RuntimeError as exc:
            self._recorder = None
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

        # Calibration overlay: run chessboard detection on the live frame
        if self._calibration_active and self._calibration_mode_obj is not None:
            key = -1
            if self._calib_capture_flag:
                key = ord(" ")
                self._calib_capture_flag = False
            elif self._calib_run_flag:
                key = ord("c")
                self._calib_run_flag = False
            vis = self._calibration_mode_obj.run(
                vis, {"key": key, "headless": False}
            )
            self._update_calibration_display()

        # Box detection overlay
        if self._box_active and self._box_mode_obj is not None:
            vis = self._box_mode_obj.run(
                vis, {"fps": self._fps_display, "headless": False}
            )
            detections_attr = getattr(self._box_mode_obj, "detections", None)
            box_count = len(detections_attr) if detections_attr is not None else 0
            self._box_status_var.set(
                f"Box mode active.\nDetected boxes: {box_count}"
            )

        # Pose estimation overlay — pass already-detected April tags so that
        # PoseMode does not need to run a second AprilTag detector instance.
        if self._pose_active and self._pose_mode_obj is not None:
            april_tags = [
                d for d in detections
                if d.detection_type == DetectionType.APRIL_TAG
            ]
            vis = self._pose_mode_obj.run(
                vis,
                {
                    "fps": self._fps_display,
                    "headless": False,
                    "april_detections": april_tags,
                },
            )

        # MediaPipe body-pose overlay
        if self._mediapipe_active and self._mediapipe_mode_obj is not None:
            vis = self._mediapipe_mode_obj.run(
                vis, {"fps": self._fps_display, "headless": False}
            )
            pose_detections = getattr(
                self._mediapipe_mode_obj, "detections", None
            )
            pose_count = (
                len(pose_detections) if pose_detections is not None else 0
            )
            self._mediapipe_status_var.set(
                f"MediaPipe mode active.\nDetected poses: {pose_count}"
            )

        # Mode-coloured crosshair lines drawn on top of all other annotations
        _h, _w = vis.shape[:2]
        _cx, _cy = _w // 2, _h // 2
        _line_color = _MODE_LINE_COLORS.get(self._mode_var.get(), (128, 128, 128))
        cv2.line(vis, (_cx, 0), (_cx, _h), _line_color, 1, cv2.LINE_AA)
        cv2.line(vis, (0, _cy), (_w, _cy), _line_color, 1, cv2.LINE_AA)

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

        Normal layout:  camera (top, col=1) + info (bottom, col=1),
                        mode panel occupies the full right column (col=2).
        Compact layout: camera is fixed-width in the upper-right corner (col=2);
                        controls (col=0) and mode + info (col=1) expand proportionally.
        """
        self._compact_view = not self._compact_view
        if self._compact_view:
            # ── Compact: camera → upper-right (col=2); rest proportional ──
            self._canvas.grid(row=0, column=2, sticky="nsew")
            self._info_frame.grid(row=1, column=1, sticky="nsew")
            self._mode_frame.grid(row=0, column=1, rowspan=1, sticky="nsew")
            self._ctrl_frame.grid(row=0, column=0, rowspan=2, sticky="nsew")
            self.root.columnconfigure(0, minsize=0, weight=1)
            self.root.columnconfigure(1, weight=1, minsize=0)
            self.root.columnconfigure(2, minsize=_COMPACT_CAMERA_WIDTH, weight=0)
            self.root.rowconfigure(0, weight=1)
            self.root.rowconfigure(1, weight=1)
            self._layout_btn_var.set("Normal view")
        else:
            # ── Normal: camera+info stacked in col=1; mode panel in col=2 ──
            self._canvas.grid(row=0, column=1, sticky="nsew")
            self._info_frame.grid(row=1, column=1, sticky="nsew")
            self._mode_frame.grid(row=0, column=2, rowspan=2, sticky="nsew")
            self._ctrl_frame.grid(row=0, column=0, rowspan=2, sticky="nsew")
            self.root.columnconfigure(0, minsize=200, weight=0)
            self.root.columnconfigure(1, weight=1, minsize=0)
            self.root.columnconfigure(2, minsize=280, weight=0)
            self.root.rowconfigure(0, weight=1)
            self.root.rowconfigure(1, weight=0)
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
