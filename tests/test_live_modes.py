"""Tests for LiveMode and LiveMapMode ASCII-art terminal visualization modes.

Tests verify:
- Module-level render functions with synthetic tag data
- LiveMode and LiveMapMode class instantiation and BaseMode interface
- ASCII art content (presence of characters, tag markers)
- 2-D map content (robot marker, tag labels, direction lines)
- CLI argument parsing for --live and --live-map flags
- Headless integration with short synthetic videos
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

cv2 = pytest.importorskip(
    "cv2", reason="OpenCV runtime dependencies are unavailable"
)

from modes.base import BaseMode
from modes.live_mode import (
    LiveMapMode,
    LiveMode,
    _bresenham,
    _default_camera_matrix,
    _tag_rotation_yaw_deg,
    render_live_ascii,
    render_live_map,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dummy_video(path, num_frames: int = 3, size: int = 64):
    """Write *num_frames* black frames into an MP4 at *path*."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 1, (size, size))
    for _ in range(num_frames):
        writer.write(np.zeros((size, size, 3), dtype=np.uint8))
    writer.release()


def _make_context(**kwargs):
    ctx = {"headless": True, "key": -1, "frame_idx": 1, "fps": 30.0}
    ctx.update(kwargs)
    return ctx


def _make_tag_data(
    tag_id: str = "5",
    tx: float = 0.3,
    ty: float = 0.0,
    tz: float = 1.0,
    rot_deg: float = 10.0,
) -> tuple:
    """Return a synthetic (tag_id, corners_2d, tvec, rvec) tuple."""
    corners_2d = np.array(
        [[100, 100], [150, 100], [150, 150], [100, 150]], dtype=np.float64
    )
    tvec = np.array([tx, ty, tz], dtype=np.float64)
    rvec = np.array([0.0, math.radians(rot_deg), 0.0], dtype=np.float64)
    return (tag_id, corners_2d, tvec, rvec)


# ===========================================================================
# Utility functions
# ===========================================================================


class TestBresenham:
    def test_horizontal_line(self):
        pts = _bresenham(0, 0, 5, 0)
        assert (0, 0) in pts
        assert (5, 0) in pts
        assert len(pts) == 6

    def test_vertical_line(self):
        pts = _bresenham(0, 0, 0, 4)
        assert (0, 0) in pts
        assert (0, 4) in pts
        assert len(pts) == 5

    def test_diagonal_line(self):
        pts = _bresenham(0, 0, 3, 3)
        assert (0, 0) in pts
        assert (3, 3) in pts

    def test_single_point(self):
        pts = _bresenham(2, 2, 2, 2)
        assert pts == [(2, 2)]

    def test_reverse_direction(self):
        pts = _bresenham(5, 5, 2, 2)
        assert (5, 5) in pts
        assert (2, 2) in pts


class TestDefaultCameraMatrix:
    def test_shape(self):
        mtx = _default_camera_matrix(640, 480)
        assert mtx.shape == (3, 3)

    def test_focal_positive(self):
        mtx = _default_camera_matrix(640, 480)
        assert mtx[0, 0] > 0
        assert mtx[1, 1] > 0

    def test_principal_point(self):
        mtx = _default_camera_matrix(640, 480)
        assert mtx[0, 2] == pytest.approx(320.0)
        assert mtx[1, 2] == pytest.approx(240.0)

    def test_bottom_right_is_one(self):
        mtx = _default_camera_matrix(640, 480)
        assert mtx[2, 2] == pytest.approx(1.0)


class TestTagRotationYaw:
    def test_zero_rotation(self):
        rvec = np.zeros(3, dtype=np.float64)
        yaw = _tag_rotation_yaw_deg(rvec)
        assert isinstance(yaw, float)

    def test_90_degree_rotation(self):
        rvec = np.array([0.0, math.pi / 2.0, 0.0], dtype=np.float64)
        yaw = _tag_rotation_yaw_deg(rvec)
        assert isinstance(yaw, float)


# ===========================================================================
# render_live_ascii
# ===========================================================================


class TestRenderLiveAscii:
    def test_returns_string(self):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        result = render_live_ascii(frame, [], cols=20, rows=10, use_ansi=False)
        assert isinstance(result, str)

    def test_has_correct_number_of_lines(self):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        result = render_live_ascii(frame, [], cols=20, rows=10, use_ansi=False)
        # 1 header line + rows content lines
        lines = result.split("\n")
        assert len(lines) == 11  # header + 10 rows

    def test_contains_ascii_chars(self):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        result = render_live_ascii(frame, [], cols=20, rows=5, use_ansi=False)
        # Black frame → mostly spaces
        assert " " in result

    def test_bright_frame_uses_dense_chars(self):
        frame = np.full((64, 64, 3), 255, dtype=np.uint8)
        result = render_live_ascii(frame, [], cols=20, rows=5, use_ansi=False)
        # Bright frame → should contain '@' (brightest char)
        assert "@" in result

    def test_no_crash_with_tags(self):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        tag = _make_tag_data("3", tx=0.0, ty=0.0, tz=1.0)
        result = render_live_ascii(
            frame, [tag], cols=40, rows=20, use_ansi=False
        )
        assert isinstance(result, str)

    def test_header_shows_tag_count(self):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        tag1 = _make_tag_data("1")
        tag2 = _make_tag_data("2")
        result = render_live_ascii(
            frame, [tag1, tag2], cols=40, rows=10, use_ansi=False
        )
        assert "tags=2" in result

    def test_header_shows_tag_stats(self):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        tag = _make_tag_data("7", tx=0.1, ty=0.0, tz=0.5)
        result = render_live_ascii(
            frame, [tag], cols=40, rows=10, use_ansi=False
        )
        assert "dist=" in result
        assert "angle=" in result
        assert "rot=" in result

    def test_ansi_mode_contains_escape_codes(self):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        tag = _make_tag_data("5")
        result = render_live_ascii(
            frame, [tag], cols=40, rows=20, use_ansi=True
        )
        assert "\033[" in result

    def test_no_ansi_mode_no_escape_codes(self):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        tag = _make_tag_data("5")
        result = render_live_ascii(
            frame, [tag], cols=40, rows=20, use_ansi=False
        )
        assert "\033[" not in result

    def test_empty_frame_no_crash(self):
        frame = np.zeros((1, 1, 3), dtype=np.uint8)
        result = render_live_ascii(frame, [], cols=5, rows=3, use_ansi=False)
        assert isinstance(result, str)


# ===========================================================================
# render_live_map
# ===========================================================================


class TestRenderLiveMap:
    def test_returns_string(self):
        result = render_live_map([], map_width=40, map_height=15, use_ansi=False)
        assert isinstance(result, str)

    def test_robot_marker_present(self):
        result = render_live_map([], map_width=40, map_height=15, use_ansi=False)
        assert "(R)" in result

    def test_no_markers_message(self):
        result = render_live_map([], map_width=40, map_height=15, use_ansi=False)
        assert "No markers detected" in result

    def test_crosshair_axes_present(self):
        result = render_live_map([], map_width=40, map_height=15, use_ansi=False)
        assert "─" in result
        assert "│" in result

    def test_tag_label_present(self):
        tag = _make_tag_data("5", tx=0.0, ty=0.0, tz=0.5)
        result = render_live_map(
            [tag], map_width=70, map_height=35, scale=10.0, use_ansi=False
        )
        assert "[5]" in result

    def test_distance_in_info(self):
        tag = _make_tag_data("3", tx=0.0, ty=0.0, tz=1.0)
        result = render_live_map(
            [tag], map_width=70, map_height=35, scale=10.0, use_ansi=False
        )
        assert "dist=" in result

    def test_angle_in_info(self):
        tag = _make_tag_data("3", tx=0.3, ty=0.0, tz=1.0)
        result = render_live_map(
            [tag], map_width=70, map_height=35, scale=10.0, use_ansi=False
        )
        assert "angle=" in result

    def test_direction_vector_in_info(self):
        tag = _make_tag_data("3", tx=0.3, ty=0.0, tz=1.0)
        result = render_live_map(
            [tag], map_width=70, map_height=35, scale=10.0, use_ansi=False
        )
        assert "dir=" in result

    def test_rotation_in_info(self):
        tag = _make_tag_data("3", tx=0.3, ty=0.0, tz=1.0, rot_deg=15.0)
        result = render_live_map(
            [tag], map_width=70, map_height=35, scale=10.0, use_ansi=False
        )
        assert "rot=" in result

    def test_multiple_tags(self):
        tag1 = _make_tag_data("1", tx=0.3, ty=0.0, tz=1.0)
        tag2 = _make_tag_data("2", tx=-0.5, ty=0.0, tz=0.8)
        result = render_live_map(
            [tag1, tag2],
            map_width=70,
            map_height=35,
            scale=10.0,
            use_ansi=False,
        )
        assert "[1]" in result
        assert "[2]" in result

    def test_direction_line_dot_present_when_in_bounds(self):
        tag = _make_tag_data("9", tx=0.0, ty=0.0, tz=0.5)
        result = render_live_map(
            [tag], map_width=70, map_height=35, scale=10.0, use_ansi=False
        )
        assert "·" in result

    def test_legend_in_output(self):
        result = render_live_map([], map_width=40, map_height=15, use_ansi=False)
        assert "Scale:" in result
        assert "(R)=robot" in result

    def test_ansi_mode_contains_escape_codes(self):
        tag = _make_tag_data("5")
        result = render_live_map(
            [tag], map_width=40, map_height=15, scale=10.0, use_ansi=True
        )
        assert "\033[" in result

    def test_no_ansi_mode_no_escape_codes(self):
        tag = _make_tag_data("5")
        result = render_live_map(
            [tag], map_width=40, map_height=15, scale=10.0, use_ansi=False
        )
        assert "\033[" not in result

    def test_off_map_tag_marked(self):
        """Tags far off the map boundary should not cause IndexError."""
        tag = _make_tag_data("99", tx=100.0, ty=0.0, tz=100.0)
        result = render_live_map(
            [tag], map_width=40, map_height=15, scale=10.0, use_ansi=False
        )
        assert isinstance(result, str)


# ===========================================================================
# LiveMode class
# ===========================================================================


class TestLiveModeClass:
    def test_is_base_mode(self):
        mode = LiveMode(use_ansi=False)
        assert isinstance(mode, BaseMode)

    def test_renders_to_terminal(self):
        mode = LiveMode(use_ansi=False)
        assert mode.renders_to_terminal is True

    def test_init_defaults(self):
        mode = LiveMode()
        assert mode._tag_size == 0.05
        assert mode._cols == 80
        assert mode._rows == 30

    def test_init_custom(self):
        mode = LiveMode(tag_size=0.10, cols=60, rows=20)
        assert mode._tag_size == 0.10
        assert mode._cols == 60
        assert mode._rows == 20

    def test_run_returns_ndarray(self, capsys):
        mode = LiveMode(use_ansi=False, cols=20, rows=10)
        mode._detector = MagicMock()
        mode._detector.detect.return_value = []
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        result = mode.run(frame, _make_context())
        assert isinstance(result, np.ndarray)
        assert result.shape == frame.shape

    def test_run_prints_to_stdout(self, capsys):
        mode = LiveMode(use_ansi=False, cols=20, rows=10)
        mode._detector = MagicMock()
        mode._detector.detect.return_value = []
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        mode.run(frame, _make_context(frame_idx=5, fps=25.0))
        out = capsys.readouterr().out
        assert len(out) > 0

    def test_run_with_no_detector(self, capsys):
        mode = LiveMode(use_ansi=False, cols=20, rows=10)
        mode._detector = None
        with patch.dict("sys.modules", {"pupil_apriltags": None, "apriltag": None}):
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            result = mode.run(frame, _make_context())
        assert isinstance(result, np.ndarray)

    def test_run_frame_idx_in_output(self, capsys):
        mode = LiveMode(use_ansi=False, cols=20, rows=10)
        mode._detector = MagicMock()
        mode._detector.detect.return_value = []
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        mode.run(frame, _make_context(frame_idx=42))
        out = capsys.readouterr().out
        assert "42" in out

    def test_missing_calibration_file_no_crash(self):
        mode = LiveMode(calibration_path="/tmp/no_such_calib.npz")
        assert mode._camera_matrix is None


# ===========================================================================
# LiveMapMode class
# ===========================================================================


class TestLiveMapModeClass:
    def test_is_base_mode(self):
        mode = LiveMapMode(use_ansi=False)
        assert isinstance(mode, BaseMode)

    def test_renders_to_terminal(self):
        mode = LiveMapMode(use_ansi=False)
        assert mode.renders_to_terminal is True

    def test_init_defaults(self):
        mode = LiveMapMode()
        assert mode._tag_size == 0.05
        assert mode._map_width == 70
        assert mode._map_height == 35
        assert mode._scale == pytest.approx(15.0)

    def test_init_custom(self):
        mode = LiveMapMode(tag_size=0.10, map_width=50, map_height=25, scale=10.0)
        assert mode._tag_size == 0.10
        assert mode._map_width == 50
        assert mode._map_height == 25
        assert mode._scale == pytest.approx(10.0)

    def test_run_returns_ndarray(self, capsys):
        mode = LiveMapMode(use_ansi=False)
        mode._detector = MagicMock()
        mode._detector.detect.return_value = []
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        result = mode.run(frame, _make_context())
        assert isinstance(result, np.ndarray)
        assert result.shape == frame.shape

    def test_run_prints_robot_marker(self, capsys):
        mode = LiveMapMode(use_ansi=False)
        mode._detector = MagicMock()
        mode._detector.detect.return_value = []
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        mode.run(frame, _make_context())
        out = capsys.readouterr().out
        assert "(R)" in out

    def test_run_with_no_detector(self, capsys):
        mode = LiveMapMode(use_ansi=False)
        mode._detector = None
        with patch.dict("sys.modules", {"pupil_apriltags": None, "apriltag": None}):
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            result = mode.run(frame, _make_context())
        assert isinstance(result, np.ndarray)

    def test_run_frame_idx_in_output(self, capsys):
        mode = LiveMapMode(use_ansi=False)
        mode._detector = MagicMock()
        mode._detector.detect.return_value = []
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        mode.run(frame, _make_context(frame_idx=99))
        out = capsys.readouterr().out
        assert "99" in out

    def test_missing_calibration_file_no_crash(self):
        mode = LiveMapMode(calibration_path="/tmp/no_such_calib.npz")
        assert mode._camera_matrix is None


# ===========================================================================
# CLI argument parsing
# ===========================================================================


class TestCLILiveFlags:
    def test_live_flag_default_false(self):
        from main import _parse_args

        args = _parse_args([])
        assert args.live is False

    def test_live_flag_set(self):
        from main import _parse_args

        args = _parse_args(["--live"])
        assert args.live is True

    def test_live_map_flag_default_false(self):
        from main import _parse_args

        args = _parse_args([])
        assert args.live_map is False

    def test_live_map_flag_set(self):
        from main import _parse_args

        args = _parse_args(["--live-map"])
        assert args.live_map is True

    def test_live_cols_default(self):
        from main import _parse_args

        args = _parse_args([])
        assert args.live_cols == 80

    def test_live_cols_custom(self):
        from main import _parse_args

        args = _parse_args(["--live-cols", "120"])
        assert args.live_cols == 120

    def test_live_rows_default(self):
        from main import _parse_args

        args = _parse_args([])
        assert args.live_rows == 30

    def test_live_rows_custom(self):
        from main import _parse_args

        args = _parse_args(["--live-rows", "40"])
        assert args.live_rows == 40

    def test_map_width_default(self):
        from main import _parse_args

        args = _parse_args([])
        assert args.map_width == 70

    def test_map_width_custom(self):
        from main import _parse_args

        args = _parse_args(["--map-width", "50"])
        assert args.map_width == 50

    def test_map_height_default(self):
        from main import _parse_args

        args = _parse_args([])
        assert args.map_height == 35

    def test_map_height_custom(self):
        from main import _parse_args

        args = _parse_args(["--map-height", "25"])
        assert args.map_height == 25

    def test_map_scale_default(self):
        from main import _parse_args

        args = _parse_args([])
        assert args.map_scale == pytest.approx(15.0)

    def test_map_scale_custom(self):
        from main import _parse_args

        args = _parse_args(["--map-scale", "10.0"])
        assert args.map_scale == pytest.approx(10.0)


# ===========================================================================
# Integration: headless runs with synthetic videos
# ===========================================================================


class TestLiveModeIntegration:
    def test_live_mode_headless(self, capsys, tmp_path):
        video = tmp_path / "live.mp4"
        _make_dummy_video(video, num_frames=2)

        from modes.live_mode import LiveMode as _LiveMode

        def _noop_ensure(self):
            if self._detector is None:
                self._detector = MagicMock()
                self._detector.detect.return_value = []

        with (
            patch(
                "robo_vision.april_tag_detector._apriltags_available",
                return_value=False,
            ),
            patch.object(_LiveMode, "_ensure_detector", _noop_ensure),
        ):
            from main import main

            rc = main([
                "--live",
                "--source", str(video),
                "--live-cols", "20",
                "--live-rows", "5",
            ])
        assert rc == 0
        out = capsys.readouterr().out
        # Should contain ASCII art output
        assert len(out) > 0

    def test_live_map_mode_headless(self, capsys, tmp_path):
        video = tmp_path / "live_map.mp4"
        _make_dummy_video(video, num_frames=2)

        from modes.live_mode import LiveMapMode as _LiveMapMode

        def _noop_ensure(self):
            if self._detector is None:
                self._detector = MagicMock()
                self._detector.detect.return_value = []

        with (
            patch(
                "robo_vision.april_tag_detector._apriltags_available",
                return_value=False,
            ),
            patch.object(_LiveMapMode, "_ensure_detector", _noop_ensure),
        ):
            from main import main

            rc = main([
                "--live-map",
                "--source", str(video),
                "--map-width", "30",
                "--map-height", "10",
            ])
        assert rc == 0
        out = capsys.readouterr().out
        # Map output should contain robot marker
        assert "(R)" in out
