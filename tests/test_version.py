"""Tests for version display in CLI and GUI."""

from __future__ import annotations

import pathlib
import re
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="OpenCV runtime dependencies are unavailable", exc_type=ImportError)

import robo_vision
from robo_vision import APP_NAME
from main import main


# ---------------------------------------------------------------------------
# Package version
# ---------------------------------------------------------------------------


class TestPackageVersion:
    def test_version_attribute_exists(self):
        assert hasattr(robo_vision, "__version__")

    def test_version_is_string(self):
        assert isinstance(robo_vision.__version__, str)

    def test_version_matches_semver(self):
        pattern = r"^\d+\.\d+\.\d+$"
        assert re.match(pattern, robo_vision.__version__), (
            f"__version__ {robo_vision.__version__!r} does not match semver"
        )

    def test_requires_python_matches_runtime_syntax_floor(self):
        pyproject = pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml"
        text = pyproject.read_text(encoding="utf-8")
        match = re.search(r'^requires-python = ">=([0-9]+)\.([0-9]+)"$', text, re.M)
        assert match is not None, "requires-python must be declared in pyproject.toml"
        major, minor = map(int, match.groups())
        assert (major, minor) >= (3, 10)


# ---------------------------------------------------------------------------
# CLI version flag
# ---------------------------------------------------------------------------


class TestCLIVersionFlag:
    def test_version_flag_prints_app_name_and_version(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        output = captured.out + captured.err
        assert APP_NAME in output
        assert robo_vision.__version__ in output


# ---------------------------------------------------------------------------
# CLI startup print
# ---------------------------------------------------------------------------


class TestCLIStartupPrint:
    def test_version_printed_on_startup(self, capsys, tmp_path):
        """main() should print the app name and version before processing."""
        # Use a tiny black video file to avoid real camera
        dummy_video = tmp_path / "black.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(dummy_video), fourcc, 1, (64, 64))
        writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
        writer.release()

        main(["--source", str(dummy_video), "--headless"])
        captured = capsys.readouterr()
        assert APP_NAME in captured.out
        assert robo_vision.__version__ in captured.out


# ---------------------------------------------------------------------------
# GUI version display
# ---------------------------------------------------------------------------


def _has_display() -> bool:
    """Return True when a graphical display is available."""
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


class TestGUIVersionDisplay:
    pytestmark = _requires_display

    @pytest.fixture
    def app(self):
        tk = pytest.importorskip("tkinter")

        cam = MagicMock()
        cam.actual_width = 640
        cam.actual_height = 480

        with patch(
            "robo_vision.detector._apriltags_available",
            return_value=False,
        ):
            from robo_vision.detector import RoboEyeDetector

            detector = RoboEyeDetector(enable_qr=False, enable_laser=False)

        root = tk.Tk()
        root.withdraw()

        from robo_vision.gui import RoboEyeSenseApp

        app = RoboEyeSenseApp(root, cam, detector)
        yield app
        root.destroy()

    def test_window_title_contains_app_name(self, app):
        assert APP_NAME in app.root.title()

    def test_window_title_contains_version(self, app):
        assert robo_vision.__version__ in app.root.title()
