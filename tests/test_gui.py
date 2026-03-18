"""Tests for the new GUI helpers and enhanced draw_detections overlay.

The GUI itself (Tkinter mainloop) is not exercised here to avoid needing a
display server.  Instead, we test:

* The ``_compute_orientation`` helper.
* The ``_draw_axes`` helper (that it doesn't raise and modifies the frame).
* The enhanced ``draw_detections`` (axes + multi-line position/orientation
  annotation) on synthetic frames.
* The ``RoboEyeSenseApp`` constructor and control callbacks without running
  the main loop (using a Tk instance that is immediately destroyed).
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("cv2", reason="OpenCV runtime dependencies are unavailable", exc_type=ImportError)

from robo_eye_sense.detector import (
    RoboEyeDetector,
    _compute_orientation,
    _draw_axes,
)
from robo_eye_sense.results import Detection, DetectionType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_display() -> bool:
    """Return True when a graphical display is available for tkinter tests."""
    import os
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return True
    try:
        import tkinter as _tk
        r = _tk.Tk()
        r.destroy()
        return True
    except Exception:
        return False


_requires_display = pytest.mark.skipif(
    not _has_display(),
    reason="No display available (set DISPLAY env var or run under Xvfb)",
)


# ---------------------------------------------------------------------------
# _compute_orientation
# ---------------------------------------------------------------------------


class TestComputeOrientation:
    def test_empty_corners_returns_zero(self):
        assert _compute_orientation([]) == pytest.approx(0.0)

    def test_one_corner_returns_zero(self):
        assert _compute_orientation([(10, 20)]) == pytest.approx(0.0)

    def test_horizontal_edge(self):
        # Two corners on the same horizontal row → angle == 0°
        angle = _compute_orientation([(0, 0), (10, 0)])
        assert angle == pytest.approx(0.0)

    def test_vertical_edge(self):
        # Two corners forming a downward-pointing vertical edge → angle == 90°
        angle = _compute_orientation([(0, 0), (0, 10)])
        assert angle == pytest.approx(90.0)

    def test_diagonal_edge(self):
        # 45° diagonal
        angle = _compute_orientation([(0, 0), (10, 10)])
        assert angle == pytest.approx(45.0)

    def test_negative_angle(self):
        # Corner to the upper-right → -45°
        angle = _compute_orientation([(0, 0), (10, -10)])
        assert angle == pytest.approx(-45.0)


# ---------------------------------------------------------------------------
# _draw_axes
# ---------------------------------------------------------------------------


class TestDrawAxes:
    def test_does_not_raise(self):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        _draw_axes(frame, (100, 100), 0.0)

    def test_modifies_frame(self):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        before = frame.copy()
        _draw_axes(frame, (100, 100), 0.0)
        assert not np.array_equal(frame, before), "Frame should be modified by _draw_axes"

    def test_rotated_does_not_raise(self):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        for angle in (0, 45, 90, 135, 180, -45, -90):
            _draw_axes(frame, (100, 100), angle)


# ---------------------------------------------------------------------------
# Enhanced draw_detections
# ---------------------------------------------------------------------------


class TestDrawDetectionsEnhanced:
    @pytest.fixture
    def detector(self):
        with patch(
            "robo_eye_sense.detector._apriltags_available",
            return_value=False,
        ):
            return RoboEyeDetector(enable_qr=False, enable_laser=False)

    def test_returns_frame_with_correct_shape(self, detector):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        result = detector.draw_detections(frame.copy(), [])
        assert result.shape == frame.shape

    def test_axes_drawn_for_detection_with_corners(self, detector):
        """With a detection that has corners the frame should be modified."""
        d = Detection(
            detection_type=DetectionType.QR_CODE,
            identifier="test",
            center=(100, 100),
            corners=[(90, 90), (110, 90), (110, 110), (90, 110)],
            track_id=0,
        )
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        result = detector.draw_detections(frame.copy(), [d])
        assert not np.array_equal(result, frame), "Annotations should change the frame"

    def test_draw_laser_spot_no_corners(self, detector):
        """Laser spots have no meaningful corners; draw must not raise."""
        d = Detection(
            detection_type=DetectionType.LASER_SPOT,
            identifier=None,
            center=(50, 50),
            corners=[],
            track_id=1,
        )
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        result = detector.draw_detections(frame.copy(), [d])
        assert result is not None

    def test_long_qr_identifier_truncated(self, detector):
        long_id = "A" * 50
        d = Detection(
            detection_type=DetectionType.QR_CODE,
            identifier=long_id,
            center=(100, 100),
            corners=[(90, 90), (110, 90), (110, 110), (90, 110)],
            track_id=0,
        )
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        # Must not raise even with a very long identifier
        result = detector.draw_detections(frame.copy(), [d])
        assert result is not None


# ---------------------------------------------------------------------------
# GUI module – constructor + callbacks (no mainloop)
# ---------------------------------------------------------------------------


class TestRoboEyeSenseApp:
    """Verify RoboEyeSenseApp constructs without errors and callbacks work."""

    pytestmark = _requires_display

    @pytest.fixture
    def app(self):
        """Create a RoboEyeSenseApp with a real Tk root (withdrawn)."""
        tk = pytest.importorskip("tkinter")

        # Mock camera so no real device is needed
        cam = MagicMock()
        cam.actual_width = 640
        cam.actual_height = 480

        with patch(
            "robo_eye_sense.detector._apriltags_available",
            return_value=False,
        ):
            detector = RoboEyeDetector(enable_qr=False, enable_laser=True)

        root = tk.Tk()
        root.withdraw()  # hide window during tests

        from robo_eye_sense.gui import RoboEyeSenseApp

        app = RoboEyeSenseApp(root, cam, detector)
        yield app
        root.destroy()

    def test_constructs_without_error(self, app):
        assert app is not None

    # ── Layout / scaling ──────────────────────────────────────────────────

    def test_minimum_window_size_is_set(self, app):
        """root.minsize should prevent the window from being too small."""
        min_w = app.root.minsize()[0]
        min_h = app.root.minsize()[1]
        assert min_w >= 700
        assert min_h >= 400

    def test_canvas_image_preserves_aspect_ratio(self, app):
        """Image resized for the canvas should keep the source aspect ratio."""
        from PIL import Image as PILImage

        # Simulate a 640×480 (4:3) source image on a square canvas
        src = PILImage.new("RGB", (640, 480), color="red")

        canvas_w, canvas_h = 400, 400
        img_w, img_h = src.size
        scale = min(canvas_w / img_w, canvas_h / img_h)
        new_w = max(1, int(img_w * scale))
        new_h = max(1, int(img_h * scale))
        resized = src.resize((new_w, new_h))

        # Width is limiting: 640 is wider → width fills canvas, height is
        # smaller → pillarboxed (black bars top/bottom).
        assert resized.size[0] == 400  # width fills the canvas
        assert resized.size[1] < 400   # height is less → pillarboxed

        # Verify approximate aspect ratio (4:3)
        ratio = resized.size[0] / resized.size[1]
        assert abs(ratio - 4 / 3) < 0.02

    def test_toggle_laser_off(self, app):
        app._enable_laser.set(False)
        app._on_toggle_laser()
        assert app.detector.laser_enabled is False

    def test_toggle_laser_on(self, app):
        app._enable_laser.set(False)
        app._on_toggle_laser()
        app._enable_laser.set(True)
        app._on_toggle_laser()
        assert app.detector.laser_enabled is True

    def test_toggle_qr_off(self, app):
        app._enable_qr.set(False)
        app._on_toggle_qr()
        assert app.detector.qr_enabled is False

    def test_toggle_qr_on(self, app):
        app._enable_qr.set(False)
        app._on_toggle_qr()
        app._enable_qr.set(True)
        app._on_toggle_qr()
        assert app.detector.qr_enabled is True

    def test_threshold_change_updates_detector(self, app):
        app._laser_threshold.set(200)
        app._on_threshold_change()
        assert app.detector.laser_detector.brightness_threshold == 200

    def test_target_area_change_updates_detector(self, app):
        app._laser_target_area.set(300)
        app._on_target_area_change()
        assert app.detector.laser_detector.target_area == 300

    def test_sensitivity_change_updates_detector(self, app):
        app._laser_sensitivity.set(75)
        app._on_sensitivity_change()
        assert app.detector.laser_detector.sensitivity == 75

    def test_toggle_laser_off_and_on_preserves_params(self, app):
        """Re-enabling laser after disabling should use current slider values."""
        app._laser_target_area.set(250)
        app._laser_sensitivity.set(30)
        app._enable_laser.set(False)
        app._on_toggle_laser()
        app._enable_laser.set(True)
        app._on_toggle_laser()
        assert app.detector.laser_detector.target_area == 250
        assert app.detector.laser_detector.sensitivity == 30

    def test_on_close_sets_running_false(self, app):
        # Patch destroy so it doesn't actually destroy during test fixture
        app.root.destroy = MagicMock()
        app._on_close()
        assert app._running is False

    def test_initial_mode_is_normal(self, app):
        from robo_eye_sense.results import DetectionMode
        assert app.detector.mode == DetectionMode.NORMAL

    def test_mode_change_to_fast(self, app):
        from robo_eye_sense.results import DetectionMode
        app._mode_var.set("Fast (low-power)")
        app._on_mode_change()
        assert app.detector.mode == DetectionMode.FAST

    def test_mode_change_to_robust(self, app):
        from robo_eye_sense.results import DetectionMode
        app._mode_var.set("Robust (motion-blur resistant)")
        app._on_mode_change()
        assert app.detector.mode == DetectionMode.ROBUST
        assert app.detector._tracker.use_kalman is True

    def test_mode_change_back_to_normal(self, app):
        from robo_eye_sense.results import DetectionMode
        app._mode_var.set("Robust (motion-blur resistant)")
        app._on_mode_change()
        app._mode_var.set("Normal")
        app._on_mode_change()
        assert app.detector.mode == DetectionMode.NORMAL
        assert app.detector._tracker.use_kalman is False

    def test_mode_description_updates_on_change(self, app):
        """Switching mode should update the description label."""
        app._mode_var.set("Fast (low-power)")
        app._on_mode_change()
        assert "downscaled" in app._mode_desc_var.get().lower()

        app._mode_var.set("Robust (motion-blur resistant)")
        app._on_mode_change()
        assert "kalman" in app._mode_desc_var.get().lower()

    def test_info_panel_mode_updates_on_change(self, app):
        """The info-panel mode indicator should reflect the active mode."""
        assert app._info_mode_var.get() == "Normal"
        app._mode_var.set("Fast (low-power)")
        app._on_mode_change()
        assert app._info_mode_var.get() == "Fast (low-power)"

    def test_set_mode_helper(self, app):
        """_set_mode() should update combobox, detector, and UI labels."""
        from robo_eye_sense.results import DetectionMode

        app._set_mode(DetectionMode.ROBUST)
        assert app.detector.mode == DetectionMode.ROBUST
        assert app._mode_var.get() == "Robust (motion-blur resistant)"
        assert "kalman" in app._mode_desc_var.get().lower()
        assert app._info_mode_var.get() == "Robust (motion-blur resistant)"

    def test_keyboard_shortcut_switches_mode(self, app):
        """Ctrl+1/2/3 key bindings should switch detector mode."""
        from robo_eye_sense.results import DetectionMode

        # event_generate requires a visible, focused window
        app.root.deiconify()
        app.root.update()
        app.root.focus_force()
        app.root.update()

        app.root.event_generate("<Control-Key-2>", when="tail")
        app.root.update()
        assert app.detector.mode == DetectionMode.FAST

        app.root.event_generate("<Control-Key-3>", when="tail")
        app.root.update()
        assert app.detector.mode == DetectionMode.ROBUST

        app.root.event_generate("<Control-Key-1>", when="tail")
        app.root.update()
        assert app.detector.mode == DetectionMode.NORMAL

        app.root.withdraw()


    # ── Scenario callbacks ────────────────────────────────────────────────

    def test_scenario_initially_inactive(self, app):
        assert app._scenario_active is False
        assert app._scenario is None

    def test_scenario_start_activates(self, app):
        app._on_scenario_start()
        assert app._scenario_active is True
        assert app._scenario is not None
        assert "Stop" in app._scenario_start_btn.cget("text")

    def test_scenario_stop_deactivates(self, app):
        app._on_scenario_start()
        app._on_scenario_start()  # toggle off
        assert app._scenario_active is False
        assert app._scenario is None
        assert "Start" in app._scenario_start_btn.cget("text")

    def test_scenario_capture_stores_reference(self, app):
        app._on_scenario_start()
        # Simulate some detections being available
        from robo_eye_sense.results import Detection, DetectionType
        app._last_detections = [
            Detection(
                detection_type=DetectionType.APRIL_TAG,
                identifier="42",
                center=(100, 100),
                corners=[(75, 75), (125, 75), (125, 125), (75, 125)],
            )
        ]
        app._on_scenario_capture_reference()
        assert app._scenario.has_reference is True
        assert str(app._scenario_reset_btn.cget("state")) != "disabled"

    def test_scenario_reset_clears_reference(self, app):
        app._on_scenario_start()
        app._last_detections = []
        app._on_scenario_capture_reference()
        app._on_scenario_reset()
        assert app._scenario.has_reference is False

    def test_scenario_text_updates(self, app):
        app._set_scenario_text("Hello test")
        content = app._scenario_text.get("1.0", "end-1c")
        assert "Hello test" in content

    # ── Recording callbacks ───────────────────────────────────────────────

    def test_recording_initially_inactive(self, app):
        assert app._recorder is None
        assert "Start" in app._record_btn.cget("text")

    def test_start_recording_with_initial_path(self, app, tmp_path):
        """When initial_record_path is set the file dialog is skipped."""
        path = str(tmp_path / "test_output.mp4")
        app._record_path = path
        app._start_recording()
        assert app._recorder is not None
        assert app._recorder.is_recording is True
        assert "Stop" in app._record_btn.cget("text")
        app._stop_recording()

    def test_stop_recording(self, app, tmp_path):
        path = str(tmp_path / "test_output.mp4")
        app._record_path = path
        app._start_recording()
        app._stop_recording()
        assert app._recorder is None
        assert "Start" in app._record_btn.cget("text")

    def test_toggle_recording_starts_and_stops(self, app, tmp_path):
        path = str(tmp_path / "toggle.mp4")
        app._record_path = path
        app._on_toggle_recording()  # start
        assert app._recorder is not None
        app._on_toggle_recording()  # stop
        assert app._recorder is None

    def test_on_close_stops_recording(self, app, tmp_path):
        path = str(tmp_path / "close_test.mp4")
        app._record_path = path
        app._start_recording()
        assert app._recorder is not None
        app.root.destroy = MagicMock()
        app._on_close()
        assert app._recorder is None

    # ── SLAM callbacks ────────────────────────────────────────────────────

    def test_slam_initially_inactive(self, app):
        assert app._slam_active is False
        assert app._slam_calibrator is None

    def test_slam_start_activates(self, app):
        app._on_slam_start()
        assert app._slam_active is True
        assert app._slam_calibrator is not None
        assert "Stop" in app._slam_start_btn.cget("text")

    def test_slam_stop_deactivates(self, app):
        app._on_slam_start()
        app._on_slam_start()  # toggle off
        assert app._slam_active is False
        assert app._slam_calibrator is None
        assert "Start" in app._slam_start_btn.cget("text")

    def test_slam_reset_clears_map(self, app):
        app._on_slam_start()
        assert app._slam_calibrator is not None
        app._on_slam_reset()
        assert len(app._slam_calibrator.marker_map) == 0

    def test_slam_save_button_enabled_when_active(self, app):
        app._on_slam_start()
        assert str(app._slam_save_btn.cget("state")) != "disabled"

    def test_slam_save_button_disabled_when_inactive(self, app):
        assert str(app._slam_save_btn.cget("state")) == "disabled"

    def test_scenario_notebook_exists(self, app):
        """The info panel should contain a tabbed notebook."""
        assert hasattr(app, "_scenario_notebook")
        # There should be 3 tabs: Offset, SLAM, and Auto
        assert app._scenario_notebook.index("end") == 3

    def test_slam_start_switches_to_slam_tab(self, app):
        """Starting SLAM should switch the notebook to the SLAM tab."""
        app._on_slam_start()
        assert app._scenario_notebook.index("current") == 1

    # ── Window title ──────────────────────────────────────────────────────

    def test_window_title_is_robot_vision(self, app):
        """Window title should be 'robot-vision'."""
        assert app.root.title() == "robot-vision"

    # ── Layout toggle ─────────────────────────────────────────────────────

    def test_layout_toggle_switches_compact_view(self, app):
        """Toggling layout should change the _compact_view flag."""
        assert app._compact_view is False
        app._toggle_layout()
        assert app._compact_view is True
        app._toggle_layout()
        assert app._compact_view is False

    def test_layout_toggle_updates_button_label(self, app):
        """Toggle layout button label should reflect the current state."""
        # Initially normal view → button says "Compact view"
        assert app._layout_btn_var.get() == "Compact view"
        app._toggle_layout()
        assert app._layout_btn_var.get() == "Normal view"
        app._toggle_layout()
        assert app._layout_btn_var.get() == "Compact view"

    # ── Scenario buttons in tabs ──────────────────────────────────────────

    def test_offset_tab_has_start_button(self, app):
        """Offset scenario start button should exist (now inside the tab)."""
        assert hasattr(app, "_scenario_start_btn")

    def test_slam_tab_has_start_button(self, app):
        """SLAM start button should exist (inside SLAM tab)."""
        assert hasattr(app, "_slam_start_btn")

    def test_auto_tab_has_start_button(self, app):
        """Auto start button should exist (inside Auto tab)."""
        assert hasattr(app, "_auto_start_btn")

    def test_auto_tab_has_marker_entry(self, app):
        """Auto tab should have a Follow ID entry field."""
        assert hasattr(app, "_auto_marker_entry")

    # ── Merged INFO/CAMERA/MODE panel ────────────────────────────────────

    def test_info_panel_has_combined_fps_and_mode(self, app):
        """FPS and mode info should both be accessible in the merged panel."""
        assert hasattr(app, "_cam_fps_var")
        assert hasattr(app, "_info_mode_var")
        assert hasattr(app, "_cam_res_var")


# ---------------------------------------------------------------------------
# render_3d_scene
# ---------------------------------------------------------------------------


class TestRender3dScene:

    @pytest.fixture(autouse=True)
    def _skip_no_tk(self):
        pytest.importorskip("tkinter")

    def test_returns_pil_image(self):
        from robo_eye_sense.gui import render_3d_scene
        from robo_eye_sense.marker_map import MarkerPose3D, RobotPose3D

        img = render_3d_scene(200, 200, [], RobotPose3D())
        from PIL import Image as PILImage
        assert isinstance(img, PILImage.Image)
        assert img.size == (200, 200)

    def test_with_markers_and_robot(self):
        from robo_eye_sense.gui import render_3d_scene
        from robo_eye_sense.marker_map import MarkerPose3D, RobotPose3D

        markers = [
            MarkerPose3D(marker_id="1", position=(10.0, 0.0, 20.0)),
            MarkerPose3D(marker_id="2", position=(-5.0, 0.0, 15.0)),
        ]
        robot = RobotPose3D(
            position=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 45.0),
            visible_markers=2,
        )
        img = render_3d_scene(200, 200, markers, robot)
        assert img.size == (200, 200)

    def test_single_marker_no_robot(self):
        from robo_eye_sense.gui import render_3d_scene
        from robo_eye_sense.marker_map import MarkerPose3D, RobotPose3D

        markers = [MarkerPose3D(marker_id="42", position=(0.0, 0.0, 0.0))]
        robot = RobotPose3D()
        img = render_3d_scene(150, 100, markers, robot)
        assert img.size == (150, 100)
