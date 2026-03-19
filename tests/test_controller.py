"""Tests for RoboVisionController – the programmatic control API."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

cv2 = pytest.importorskip(
    "cv2", reason="OpenCV runtime dependencies are unavailable"
)

from main import RoboVisionController


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dummy_video(path, num_frames: int = 5, size: int = 64):
    """Write *num_frames* black frames into an MP4 at *path*."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 1, (size, size))
    for _ in range(num_frames):
        writer.write(np.zeros((size, size, 3), dtype=np.uint8))
    writer.release()


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestRoboVisionControllerInit:
    """Verify __init__ validates and stores parameters correctly."""

    def test_default_parameters(self):
        ctrl = RoboVisionController()
        assert ctrl._mode == "basic"
        assert ctrl._quality == "normal"
        assert ctrl._source == 0
        assert ctrl._width == 640
        assert ctrl._height == 480
        assert ctrl._enable_apriltag is True
        assert ctrl._enable_qr is False
        assert ctrl._enable_laser is False
        assert ctrl.on_detections is None
        assert not ctrl.is_running

    def test_custom_parameters(self):
        cb = MagicMock()
        ctrl = RoboVisionController(
            source="video.mp4",
            width=320,
            height=240,
            mode="follow",
            quality="high",
            enable_apriltag=False,
            enable_qr=True,
            enable_laser=True,
            laser_brightness_threshold=100,
            laser_brightness_threshold_max=200,
            laser_channels="r",
            tag_names={"1": "box"},
            on_detections=cb,
            tag_size=0.1,
            follow_marker="5",
            follow_box=True,
            target_distance=1.0,
            calibration_path="cal.npz",
            map_file="map.json",
            record="out.mp4",
            sensitivity=80,
        )
        assert ctrl._source == "video.mp4"
        assert ctrl._mode == "follow"
        assert ctrl._quality == "high"
        assert ctrl._enable_qr is True
        assert ctrl._tag_names == {"1": "box"}
        assert ctrl.on_detections is cb
        assert ctrl._follow_marker == "5"

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be one of"):
            RoboVisionController(mode="invalid_mode")

    def test_invalid_quality_raises(self):
        with pytest.raises(ValueError, match="quality must be one of"):
            RoboVisionController(quality="ultra")

    def test_tag_names_default_empty(self):
        ctrl = RoboVisionController()
        assert ctrl._tag_names == {}

    def test_tag_names_copied(self):
        original = {"1": "box"}
        ctrl = RoboVisionController(tag_names=original)
        original["2"] = "table"
        assert "2" not in ctrl._tag_names


# ---------------------------------------------------------------------------
# is_running property
# ---------------------------------------------------------------------------


class TestIsRunning:
    """is_running reflects background thread state."""

    def test_not_running_before_start(self):
        ctrl = RoboVisionController()
        assert ctrl.is_running is False

    def test_not_running_after_stop(self, tmp_path):
        video = tmp_path / "v.mp4"
        _make_dummy_video(video)

        with patch(
            "robo_vision.april_tag_detector._apriltags_available",
            return_value=False,
        ):
            ctrl = RoboVisionController(source=str(video))
            ctrl.start()
            ctrl.stop()

        assert ctrl.is_running is False


# ---------------------------------------------------------------------------
# run() – blocking execution
# ---------------------------------------------------------------------------


class TestRun:
    """run() executes synchronously and returns an integer exit code."""

    def test_run_returns_zero_on_success(self, tmp_path):
        video = tmp_path / "v.mp4"
        _make_dummy_video(video)

        with patch(
            "robo_vision.april_tag_detector._apriltags_available",
            return_value=False,
        ):
            ctrl = RoboVisionController(source=str(video))
            rc = ctrl.run()

        assert rc == 0

    def test_run_returns_one_on_bad_source(self):
        with patch(
            "robo_vision.april_tag_detector._apriltags_available",
            return_value=False,
        ):
            ctrl = RoboVisionController(source="/no/such/file.mp4")
            rc = ctrl.run()
        assert rc == 1

    def test_run_processes_all_frames(self, tmp_path):
        video = tmp_path / "v.mp4"
        num_frames = 4
        _make_dummy_video(video, num_frames=num_frames)

        received: list[int] = []

        def cb(frame_idx, detections, fps):
            received.append(frame_idx)

        with patch(
            "robo_vision.april_tag_detector._apriltags_available",
            return_value=False,
        ):
            ctrl = RoboVisionController(source=str(video), on_detections=cb)
            ctrl.run()

        assert len(received) == num_frames
        assert received == list(range(1, num_frames + 1))


# ---------------------------------------------------------------------------
# start() / stop() – background thread
# ---------------------------------------------------------------------------


class TestStartStop:
    """start() and stop() manage a background daemon thread."""

    def test_start_raises_if_already_running(self, tmp_path):
        video = tmp_path / "v.mp4"
        _make_dummy_video(video, num_frames=30)

        with patch(
            "robo_vision.april_tag_detector._apriltags_available",
            return_value=False,
        ):
            ctrl = RoboVisionController(source=str(video))
            ctrl.start()
            try:
                with pytest.raises(RuntimeError, match="already running"):
                    ctrl.start()
            finally:
                ctrl.stop()

    def test_stop_signals_early_exit(self, tmp_path):
        video = tmp_path / "v.mp4"
        _make_dummy_video(video, num_frames=100)

        received: list[int] = []
        started = threading.Event()

        def cb(frame_idx, detections, fps):
            started.set()
            received.append(frame_idx)

        with patch(
            "robo_vision.april_tag_detector._apriltags_available",
            return_value=False,
        ):
            ctrl = RoboVisionController(source=str(video), on_detections=cb)
            ctrl.start()
            started.wait(timeout=5.0)
            ctrl.stop(timeout=5.0)

        assert len(received) < 100

    def test_background_thread_is_daemon(self, tmp_path):
        video = tmp_path / "v.mp4"
        _make_dummy_video(video, num_frames=30)

        with patch(
            "robo_vision.april_tag_detector._apriltags_available",
            return_value=False,
        ):
            ctrl = RoboVisionController(source=str(video))
            ctrl.start()
            assert ctrl._thread is not None
            assert ctrl._thread.daemon is True
            ctrl.stop()


# ---------------------------------------------------------------------------
# on_detections callback
# ---------------------------------------------------------------------------


class TestOnDetectionsCallback:
    """on_detections is called once per frame with correct arguments."""

    def test_callback_receives_frame_idx_and_fps(self, tmp_path):
        video = tmp_path / "v.mp4"
        _make_dummy_video(video, num_frames=3)

        calls: list[tuple] = []

        def cb(frame_idx, detections, fps):
            calls.append((frame_idx, detections, fps))

        with patch(
            "robo_vision.april_tag_detector._apriltags_available",
            return_value=False,
        ):
            ctrl = RoboVisionController(source=str(video), on_detections=cb)
            ctrl.run()

        assert len(calls) == 3
        for i, (frame_idx, detections, fps) in enumerate(calls, start=1):
            assert frame_idx == i
            assert isinstance(detections, list)
            assert fps >= 0.0

    def test_callback_exception_does_not_crash_loop(self, tmp_path):
        video = tmp_path / "v.mp4"
        num_frames = 3
        _make_dummy_video(video, num_frames=num_frames)

        call_count = 0

        def bad_cb(frame_idx, detections, fps):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("deliberate error in callback")

        with patch(
            "robo_vision.april_tag_detector._apriltags_available",
            return_value=False,
        ):
            ctrl = RoboVisionController(source=str(video), on_detections=bad_cb)
            rc = ctrl.run()

        assert rc == 0
        assert call_count == num_frames

    def test_no_callback_runs_without_error(self, tmp_path):
        video = tmp_path / "v.mp4"
        _make_dummy_video(video)

        with patch(
            "robo_vision.april_tag_detector._apriltags_available",
            return_value=False,
        ):
            ctrl = RoboVisionController(source=str(video))
            rc = ctrl.run()

        assert rc == 0


# ---------------------------------------------------------------------------
# All valid modes can be instantiated
# ---------------------------------------------------------------------------


class TestModeInstantiation:
    """Every valid mode string is accepted by the constructor."""

    @pytest.mark.parametrize(
        "mode",
        ["basic", "offset", "slam", "calibration", "box", "pose", "follow"],
    )
    def test_valid_modes(self, mode):
        ctrl = RoboVisionController(mode=mode)
        assert ctrl._mode == mode


# ---------------------------------------------------------------------------
# Quality / detection mode mapping
# ---------------------------------------------------------------------------


class TestQualityMapping:
    """Quality strings map to the correct DetectionMode."""

    @pytest.mark.parametrize(
        "quality",
        ["low", "normal", "high"],
    )
    def test_valid_qualities(self, quality):
        ctrl = RoboVisionController(quality=quality)
        assert ctrl._quality == quality
