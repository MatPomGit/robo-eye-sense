"""Tests for new features: info command, tag names, laser threshold range,
and headless offset reference re-capture."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

cv2 = pytest.importorskip(
    "cv2", reason="OpenCV runtime dependencies are unavailable"
)

from main import _parse_args, _parse_tag_names
from robo_eye_sense.laser_detector import LaserSpotDetector
from robo_eye_sense.results import Detection, DetectionType


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_dummy_video(path, num_frames: int = 3, size: int = 64):
    """Write *num_frames* black frames into an MP4 at *path*."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 1, (size, size))
    for _ in range(num_frames):
        writer.write(np.zeros((size, size, 3), dtype=np.uint8))
    writer.release()


# ===========================================================================
# 1. Info command
# ===========================================================================


class TestInfoCommand:
    """--info should print camera info and exit."""

    def test_info_flag_parsed(self):
        args = _parse_args(["--info"])
        assert args.info is True

    def test_info_flag_default_false(self):
        args = _parse_args([])
        assert args.info is False

    def test_info_prints_camera_information(self, capsys, tmp_path):
        video = tmp_path / "black.mp4"
        _make_dummy_video(video)

        with patch(
            "robo_eye_sense.april_tag_detector._apriltags_available",
            return_value=False,
        ):
            from main import main

            rc = main(["--source", str(video), "--info"])

        assert rc == 0
        out = capsys.readouterr().out
        assert "Camera information" in out
        assert "width" in out
        assert "height" in out
        assert "fps" in out
        assert "backend" in out


class TestCameraGetInfo:
    """Camera.get_info() should return a dictionary of parameters."""

    def test_get_info_returns_dict(self, tmp_path):
        video = tmp_path / "black.mp4"
        _make_dummy_video(video, num_frames=1, size=64)
        from robo_eye_sense.camera import Camera

        cam = Camera(source=str(video))
        info = cam.get_info()
        cam.release()

        assert isinstance(info, dict)
        assert "width" in info
        assert "height" in info
        assert "fps" in info
        assert "backend" in info

    def test_backend_name_property(self, tmp_path):
        video = tmp_path / "black.mp4"
        _make_dummy_video(video, num_frames=1, size=64)
        from robo_eye_sense.camera import Camera

        cam = Camera(source=str(video))
        name = cam.backend_name
        cam.release()
        assert isinstance(name, str)


# ===========================================================================
# 2. AprilTag name mapping
# ===========================================================================


class TestTagNames:
    """--tag-names should parse ID=NAME pairs."""

    def test_parse_tag_names_empty(self):
        assert _parse_tag_names(None) == {}
        assert _parse_tag_names([]) == {}

    def test_parse_tag_names_single(self):
        result = _parse_tag_names(["1=box"])
        assert result == {"1": "box"}

    def test_parse_tag_names_multiple(self):
        result = _parse_tag_names(["1=box", "2=table", "5=wall"])
        assert result == {"1": "box", "2": "table", "5": "wall"}

    def test_parse_tag_names_invalid_ignored(self, capsys):
        result = _parse_tag_names(["1=box", "invalid", "2=table"])
        assert result == {"1": "box", "2": "table"}
        err = capsys.readouterr().err
        assert "invalid" in err

    def test_cli_tag_names_parsed(self):
        args = _parse_args(["--tag-names", "1=box", "2=table"])
        assert args.tag_names == ["1=box", "2=table"]

    def test_cli_tag_names_default_none(self):
        args = _parse_args([])
        assert args.tag_names is None


class TestDetectorTagNames:
    """RoboEyeDetector should enrich AprilTag identifiers with names."""

    def test_tag_names_property(self):
        from robo_eye_sense.detector import RoboEyeDetector

        with patch(
            "robo_eye_sense.april_tag_detector._apriltags_available",
            return_value=False,
        ):
            d = RoboEyeDetector(tag_names={"1": "box", "2": "table"})
        assert d.tag_names == {"1": "box", "2": "table"}

    def test_tag_names_setter(self):
        from robo_eye_sense.detector import RoboEyeDetector

        with patch(
            "robo_eye_sense.april_tag_detector._apriltags_available",
            return_value=False,
        ):
            d = RoboEyeDetector()
        assert d.tag_names == {}
        d.tag_names = {"3": "floor"}
        assert d.tag_names == {"3": "floor"}

    def test_tag_names_enriches_identifier(self):
        """When tag_names are set, AprilTag identifiers should include the name."""
        from robo_eye_sense.detector import RoboEyeDetector

        mock_april = MagicMock()
        mock_april.detect.return_value = [
            Detection(
                detection_type=DetectionType.APRIL_TAG,
                identifier="1",
                center=(50, 50),
                corners=[(40, 40), (60, 40), (60, 60), (40, 60)],
            )
        ]

        with patch(
            "robo_eye_sense.april_tag_detector._apriltags_available",
            return_value=False,
        ):
            d = RoboEyeDetector(
                enable_apriltag=False,
                enable_qr=False,
                enable_laser=False,
                tag_names={"1": "box"},
            )
        d._april_detector = mock_april

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        results = d.process_frame(frame)
        assert len(results) == 1
        assert "box" in results[0].identifier
        assert "1" in results[0].identifier

    def test_tag_names_no_match(self):
        """When tag has no name, identifier should stay unchanged."""
        from robo_eye_sense.detector import RoboEyeDetector

        mock_april = MagicMock()
        mock_april.detect.return_value = [
            Detection(
                detection_type=DetectionType.APRIL_TAG,
                identifier="99",
                center=(50, 50),
                corners=[(40, 40), (60, 40), (60, 60), (40, 60)],
            )
        ]

        with patch(
            "robo_eye_sense.april_tag_detector._apriltags_available",
            return_value=False,
        ):
            d = RoboEyeDetector(
                enable_apriltag=False,
                enable_qr=False,
                enable_laser=False,
                tag_names={"1": "box"},
            )
        d._april_detector = mock_april

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        results = d.process_frame(frame)
        assert len(results) == 1
        assert results[0].identifier == "99"


# ===========================================================================
# 3. Laser threshold range
# ===========================================================================


class TestLaserThresholdRange:
    """LaserSpotDetector should support min/max brightness range."""

    def test_default_max_is_255(self):
        d = LaserSpotDetector()
        assert d.brightness_threshold_max == 255

    def test_custom_range(self):
        d = LaserSpotDetector(brightness_threshold=60, brightness_threshold_max=85)
        assert d.brightness_threshold == 60
        assert d.brightness_threshold_max == 85

    def test_invalid_max_below_min(self):
        with pytest.raises(ValueError, match="brightness_threshold_max"):
            LaserSpotDetector(brightness_threshold=100, brightness_threshold_max=50)

    def test_invalid_max_above_255(self):
        with pytest.raises(ValueError, match="brightness_threshold_max"):
            LaserSpotDetector(brightness_threshold_max=256)

    def test_invalid_max_negative(self):
        with pytest.raises(ValueError, match="brightness_threshold_max"):
            LaserSpotDetector(brightness_threshold_max=-1)

    def test_range_filtering_excludes_too_bright(self):
        """A very bright spot (255) should be excluded when max < 255."""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(frame, (100, 100), 6, (255, 255, 255), -1)
        d = LaserSpotDetector(brightness_threshold=60, brightness_threshold_max=200)
        results = d.detect(frame)
        assert results == []

    def test_range_filtering_includes_in_range(self):
        """A spot with brightness in range should be detected."""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(frame, (100, 100), 6, (120, 120, 120), -1)
        d = LaserSpotDetector(brightness_threshold=60, brightness_threshold_max=200)
        results = d.detect(frame)
        assert len(results) == 1

    def test_default_range_same_as_before(self):
        """Default (threshold=240, max=255) should work like original."""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(frame, (100, 100), 6, (255, 255, 255), -1)
        d = LaserSpotDetector(brightness_threshold=240, brightness_threshold_max=255)
        results = d.detect(frame)
        assert len(results) == 1

    def test_cli_threshold_max_parsed(self):
        args = _parse_args(["--laser-threshold-max", "200"])
        assert args.laser_threshold_max == 200

    def test_cli_threshold_max_default(self):
        args = _parse_args([])
        assert args.laser_threshold_max == 255


# ===========================================================================
# 4. Headless offset reference re-capture
# ===========================================================================


class TestHeadlessOffsetRecapture:
    """In headless offset mode, 'ref' command should re-capture reference."""

    def test_headless_offset_prints_commands_prompt(self, capsys, tmp_path):
        """After initial offset, the commands prompt should be printed."""
        video = tmp_path / "black.mp4"
        _make_dummy_video(video, num_frames=3)

        with patch(
            "robo_eye_sense.april_tag_detector._apriltags_available",
            return_value=False,
        ):
            from main import main

            rc = main([
                "--source", str(video), "--headless",
                "--scenario", "offset",
            ])

        assert rc == 0
        out = capsys.readouterr().out
        assert "Commands:" in out
        assert "ref" in out
        assert "offset" in out
        assert "quit" in out

    def test_headless_offset_still_prints_result(self, capsys, tmp_path):
        """Headless offset should still print camera-offset result."""
        video = tmp_path / "black.mp4"
        _make_dummy_video(video, num_frames=2)

        with patch(
            "robo_eye_sense.april_tag_detector._apriltags_available",
            return_value=False,
        ):
            from main import main

            rc = main([
                "--source", str(video), "--headless",
                "--scenario", "offset",
            ])

        assert rc == 0
        out = capsys.readouterr().out
        assert "Camera-offset result" in out
        assert "Offset scenario finished" in out


# ===========================================================================
# 5. Detector enable_laser with threshold max
# ===========================================================================


class TestDetectorEnableLaserMax:
    """RoboEyeDetector.enable_laser should accept brightness_threshold_max."""

    def test_enable_laser_with_max(self):
        from robo_eye_sense.detector import RoboEyeDetector

        with patch(
            "robo_eye_sense.april_tag_detector._apriltags_available",
            return_value=False,
        ):
            d = RoboEyeDetector(enable_laser=False)
        d.enable_laser(brightness_threshold=60, brightness_threshold_max=200)
        assert d.laser_detector is not None
        assert d.laser_detector.brightness_threshold == 60
        assert d.laser_detector.brightness_threshold_max == 200

    def test_constructor_with_max(self):
        from robo_eye_sense.detector import RoboEyeDetector

        with patch(
            "robo_eye_sense.april_tag_detector._apriltags_available",
            return_value=False,
        ):
            d = RoboEyeDetector(
                enable_laser=True,
                laser_brightness_threshold=50,
                laser_brightness_threshold_max=180,
            )
        assert d.laser_detector is not None
        assert d.laser_detector.brightness_threshold == 50
        assert d.laser_detector.brightness_threshold_max == 180
