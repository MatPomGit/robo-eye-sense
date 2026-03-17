"""Tests for :class:`~robo_eye_sense.recorder.VideoRecorder`."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip(
    "cv2", reason="OpenCV runtime dependencies are unavailable"
)

from robo_eye_sense.recorder import VideoRecorder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frame(width: int = 64, height: int = 64) -> np.ndarray:
    """Return a random BGR frame of the given size."""
    return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestVideoRecorderInit:
    def test_not_recording_after_construction(self, tmp_path: Path):
        rec = VideoRecorder(str(tmp_path / "out.mp4"), 64, 64)
        assert rec.is_recording is False

    def test_output_path_stored(self, tmp_path: Path):
        path = str(tmp_path / "test.mp4")
        rec = VideoRecorder(path, 64, 64)
        assert rec.output_path == path


# ---------------------------------------------------------------------------
# Start / stop
# ---------------------------------------------------------------------------


class TestStartStop:
    def test_start_sets_recording_flag(self, tmp_path: Path):
        rec = VideoRecorder(str(tmp_path / "out.mp4"), 64, 64)
        rec.start()
        try:
            assert rec.is_recording is True
        finally:
            rec.stop()

    def test_stop_clears_recording_flag(self, tmp_path: Path):
        rec = VideoRecorder(str(tmp_path / "out.mp4"), 64, 64)
        rec.start()
        rec.stop()
        assert rec.is_recording is False

    def test_double_start_is_safe(self, tmp_path: Path):
        rec = VideoRecorder(str(tmp_path / "out.mp4"), 64, 64)
        rec.start()
        rec.start()  # should not raise
        try:
            assert rec.is_recording is True
        finally:
            rec.stop()

    def test_double_stop_is_safe(self, tmp_path: Path):
        rec = VideoRecorder(str(tmp_path / "out.mp4"), 64, 64)
        rec.start()
        rec.stop()
        rec.stop()  # should not raise
        assert rec.is_recording is False

    def test_stop_without_start_is_safe(self, tmp_path: Path):
        rec = VideoRecorder(str(tmp_path / "out.mp4"), 64, 64)
        rec.stop()  # should not raise
        assert rec.is_recording is False


# ---------------------------------------------------------------------------
# Writing frames
# ---------------------------------------------------------------------------


class TestWriteFrame:
    def test_creates_output_file(self, tmp_path: Path):
        path = str(tmp_path / "out.mp4")
        rec = VideoRecorder(path, 64, 64, fps=10.0)
        rec.start()
        for _ in range(5):
            rec.write_frame(_make_frame())
        rec.stop()
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0

    def test_write_frame_when_not_recording_is_noop(self, tmp_path: Path):
        path = str(tmp_path / "out.mp4")
        rec = VideoRecorder(path, 64, 64)
        rec.write_frame(_make_frame())  # should not raise
        assert not os.path.exists(path)

    def test_mismatched_frame_size_is_resized(self, tmp_path: Path):
        """Frames with a different resolution are auto-resized."""
        path = str(tmp_path / "out.mp4")
        rec = VideoRecorder(path, 64, 64, fps=10.0)
        rec.start()
        big_frame = _make_frame(width=128, height=128)
        rec.write_frame(big_frame)  # should not raise
        rec.stop()
        assert os.path.isfile(path)

    def test_output_is_readable_video(self, tmp_path: Path):
        """The created file should be openable by cv2.VideoCapture."""
        path = str(tmp_path / "out.mp4")
        rec = VideoRecorder(path, 64, 64, fps=10.0)
        rec.start()
        for _ in range(10):
            rec.write_frame(_make_frame())
        rec.stop()

        cap = cv2.VideoCapture(path)
        try:
            assert cap.isOpened()
            ret, frame = cap.read()
            assert ret
            assert frame is not None
        finally:
            cap.release()


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


class TestContextManager:
    def test_context_manager_starts_and_stops(self, tmp_path: Path):
        path = str(tmp_path / "out.mp4")
        with VideoRecorder(path, 64, 64, fps=10.0) as rec:
            assert rec.is_recording is True
            rec.write_frame(_make_frame())
        assert rec.is_recording is False
        assert os.path.isfile(path)

    def test_context_manager_stops_on_exception(self, tmp_path: Path):
        path = str(tmp_path / "out.mp4")
        rec = VideoRecorder(path, 64, 64, fps=10.0)
        try:
            with rec:
                rec.write_frame(_make_frame())
                raise ValueError("test error")
        except ValueError:
            pass
        assert rec.is_recording is False


# ---------------------------------------------------------------------------
# CLI --record integration
# ---------------------------------------------------------------------------


class TestCLIRecordFlag:
    """Test that ``main()`` accepts ``--record`` and creates a video file."""

    def test_record_flag_creates_file_headless(self, tmp_path: Path):
        """``--headless --record <path>`` should produce an .mp4 file."""
        import subprocess
        import sys

        # Create a tiny dummy video as the camera source
        dummy_video = str(tmp_path / "source.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(dummy_video, fourcc, 1, (64, 64))
        for _ in range(3):
            writer.write(_make_frame())
        writer.release()

        output_file = str(tmp_path / "out.mp4")

        # Run in a subprocess to avoid pupil_apriltags __del__ segfault
        result = subprocess.run(
            [
                sys.executable, "-c",
                "from main import main; import sys; "
                f"sys.exit(main(['--source', {dummy_video!r}, '--headless', "
                f"'--record', {output_file!r}]))",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert os.path.isfile(output_file)
        assert os.path.getsize(output_file) > 0

    def test_record_flag_parsed(self):
        """Ensure ``--record`` is present in the parser."""
        from main import _parse_args

        args = _parse_args(["--record", "test.mp4"])
        assert args.record == "test.mp4"

    def test_record_default_is_none(self):
        from main import _parse_args

        args = _parse_args([])
        assert args.record is None
