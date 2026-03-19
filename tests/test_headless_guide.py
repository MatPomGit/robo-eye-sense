"""Tests for the headless guide module and CLI integration."""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import numpy as np
import pytest

cv2 = pytest.importorskip(
    "cv2", reason="OpenCV runtime dependencies are unavailable"
)

from robo_vision.headless_guide import (
    MOVABLE_ID_UPPER,
    PACKAGE_ID_RANGE,
    TABLE_ID_RANGE,
    classify_tag,
    discover_cameras,
    get_calibration_info,
    get_device_status,
    load_tag_names_from_file,
    print_headless_guide,
)

from main import _parse_args


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
# 1. Tag classification
# ===========================================================================


class TestClassifyTag:
    """classify_tag should return the correct category for each ID range."""

    def test_package_ids(self):
        for tag_id in range(5, 9):
            assert classify_tag(tag_id) == "package"

    def test_movable_ids_below_package(self):
        for tag_id in range(0, 5):
            assert classify_tag(tag_id) == "movable"

    def test_movable_id_9(self):
        assert classify_tag(9) == "movable"

    def test_table_ids(self):
        for tag_id in range(12, 21):
            assert classify_tag(tag_id) == "table"

    def test_static_id_10(self):
        assert classify_tag(10) == "static"

    def test_static_id_11(self):
        assert classify_tag(11) == "static"

    def test_static_id_above_table(self):
        assert classify_tag(21) == "static"
        assert classify_tag(100) == "static"

    def test_constants(self):
        assert MOVABLE_ID_UPPER == 10
        assert list(PACKAGE_ID_RANGE) == [5, 6, 7, 8]
        assert list(TABLE_ID_RANGE) == list(range(12, 21))


# ===========================================================================
# 2. Tag names from file
# ===========================================================================


class TestLoadTagNamesFromFile:
    """load_tag_names_from_file should parse JSON dicts."""

    def test_load_valid_file(self, tmp_path):
        f = tmp_path / "tags.json"
        f.write_text(json.dumps({"5": "package-A", "12": "table-left"}))
        result = load_tag_names_from_file(str(f))
        assert result == {"5": "package-A", "12": "table-left"}

    def test_keys_converted_to_strings(self, tmp_path):
        f = tmp_path / "tags.json"
        f.write_text(json.dumps({5: "box", 12: "table"}))
        result = load_tag_names_from_file(str(f))
        assert result == {"5": "box", "12": "table"}

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_tag_names_from_file(str(tmp_path / "missing.json"))

    def test_invalid_json(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("not json")
        with pytest.raises(json.JSONDecodeError):
            load_tag_names_from_file(str(f))

    def test_non_dict_json(self, tmp_path):
        f = tmp_path / "array.json"
        f.write_text(json.dumps([1, 2, 3]))
        with pytest.raises(TypeError, match="dict"):
            load_tag_names_from_file(str(f))

    def test_empty_dict(self, tmp_path):
        f = tmp_path / "empty.json"
        f.write_text("{}")
        result = load_tag_names_from_file(str(f))
        assert result == {}


# ===========================================================================
# 3. Device status
# ===========================================================================


class TestGetDeviceStatus:
    """get_device_status should return a populated dict."""

    def test_returns_dict(self):
        status = get_device_status()
        assert isinstance(status, dict)

    def test_required_keys(self):
        status = get_device_status()
        for key in ("platform", "system", "machine", "python", "opencv"):
            assert key in status, f"Missing key: {key}"

    def test_python_version_present(self):
        import platform as _platform

        status = get_device_status()
        assert status["python"] == _platform.python_version()


# ===========================================================================
# 4. Calibration info
# ===========================================================================


class TestGetCalibrationInfo:
    """get_calibration_info should detect .npz file presence and timestamp."""

    def test_missing_file(self, tmp_path):
        info = get_calibration_info(str(tmp_path / "cal.npz"))
        assert info["exists"] is False

    def test_existing_file(self, tmp_path):
        cal = tmp_path / "cal.npz"
        np.savez(str(cal), camera_matrix=np.eye(3))
        info = get_calibration_info(str(cal))
        assert info["exists"] is True
        assert "calibrated_at" in info
        assert "calibrated_at_local" in info

    def test_calibrated_at_is_iso(self, tmp_path):
        cal = tmp_path / "cal.npz"
        np.savez(str(cal), camera_matrix=np.eye(3))
        info = get_calibration_info(str(cal))
        # ISO 8601 contains 'T' separator
        assert "T" in info["calibrated_at"]

    def test_path_always_present(self, tmp_path):
        info = get_calibration_info(str(tmp_path / "missing.npz"))
        assert "path" in info


# ===========================================================================
# 5. Camera discovery
# ===========================================================================


class TestDiscoverCameras:
    """discover_cameras should return a list (may be empty in CI)."""

    def test_returns_list(self):
        cameras = discover_cameras(max_index=1)
        assert isinstance(cameras, list)

    def test_dict_entries_have_keys(self):
        """If any cameras are found, verify keys."""
        cameras = discover_cameras(max_index=1)
        for cam in cameras:
            assert "index" in cam
            assert "width" in cam
            assert "height" in cam


# ===========================================================================
# 6. Full headless guide report
# ===========================================================================


class TestPrintHeadlessGuide:
    """print_headless_guide should produce a comprehensive report."""

    def test_contains_device_section(self):
        report = print_headless_guide(max_camera_index=0)
        assert "Device status" in report
        assert "python" in report

    def test_contains_camera_section(self):
        report = print_headless_guide(max_camera_index=0)
        assert "Available cameras" in report

    def test_contains_calibration_section_missing(self, tmp_path):
        report = print_headless_guide(
            calib_path=str(tmp_path / "missing.npz"),
            max_camera_index=0,
        )
        assert "Calibration status" in report
        assert "No calibration found" in report

    def test_contains_calibration_section_present(self, tmp_path):
        cal = tmp_path / "cal.npz"
        np.savez(str(cal), camera_matrix=np.eye(3))
        report = print_headless_guide(
            calib_path=str(cal),
            max_camera_index=0,
        )
        assert "Calibration status" in report
        assert "Calibrated at" in report

    def test_contains_tag_names_from_file(self, tmp_path):
        f = tmp_path / "tags.json"
        f.write_text(json.dumps({"5": "box-5", "12": "table-12"}))
        report = print_headless_guide(
            tag_names_file=str(f),
            max_camera_index=0,
        )
        assert "box-5" in report
        assert "table-12" in report
        assert "[package]" in report
        assert "[table]" in report

    def test_contains_tag_names_from_dict(self):
        report = print_headless_guide(
            tag_names={"1": "my-tag"},
            max_camera_index=0,
        )
        assert "my-tag" in report
        assert "[movable]" in report

    def test_file_overrides_cli_names(self, tmp_path):
        f = tmp_path / "tags.json"
        f.write_text(json.dumps({"1": "from-file"}))
        report = print_headless_guide(
            tag_names={"1": "from-cli"},
            tag_names_file=str(f),
            max_camera_index=0,
        )
        assert "from-file" in report

    def test_contains_classification_rules(self):
        report = print_headless_guide(max_camera_index=0)
        assert "classification rules" in report
        assert "movable" in report
        assert "package" in report
        assert "table" in report
        assert "static" in report

    def test_missing_tag_names_file(self, tmp_path):
        report = print_headless_guide(
            tag_names_file=str(tmp_path / "no.json"),
            max_camera_index=0,
        )
        assert "not found" in report

    def test_no_tag_names(self):
        report = print_headless_guide(max_camera_index=0)
        assert "No custom tag names defined" in report


# ===========================================================================
# 7. CLI integration
# ===========================================================================


class TestCLIIntegration:
    """CLI arguments --tag-names-file and --guide should be parsed."""

    def test_tag_names_file_parsed(self):
        args = _parse_args(["--tag-names-file", "tags.json"])
        assert args.tag_names_file == "tags.json"

    def test_tag_names_file_default_none(self):
        args = _parse_args([])
        assert args.tag_names_file is None

    def test_guide_flag_parsed(self):
        args = _parse_args(["--guide"])
        assert args.guide is True

    def test_guide_flag_default_false(self):
        args = _parse_args([])
        assert args.guide is False

    def test_guide_prints_report_and_exits(self, capsys):
        with patch(
            "robo_vision.headless_guide.discover_cameras",
            return_value=[],
        ):
            from main import main

            rc = main(["--guide"])

        assert rc == 0
        out = capsys.readouterr().out
        assert "HEADLESS GUIDE" in out
        assert "Device status" in out
        assert "Available cameras" in out
        assert "Calibration status" in out
        assert "classification rules" in out

    def test_guide_with_tag_names_file(self, capsys, tmp_path):
        f = tmp_path / "tags.json"
        f.write_text(json.dumps({"6": "parcel"}))

        with patch(
            "robo_vision.headless_guide.discover_cameras",
            return_value=[],
        ):
            from main import main

            rc = main(["--guide", "--tag-names-file", str(f)])

        assert rc == 0
        out = capsys.readouterr().out
        assert "parcel" in out
        assert "[package]" in out

    def test_tag_names_file_merge_in_normal_run(self, capsys, tmp_path):
        """--tag-names-file should merge names for a normal (non-guide) run."""
        video = tmp_path / "black.mp4"
        _make_dummy_video(video)
        f = tmp_path / "tags.json"
        f.write_text(json.dumps({"1": "from-file"}))

        with patch(
            "robo_vision.april_tag_detector._apriltags_available",
            return_value=False,
        ):
            from main import main

            rc = main([
                "--source", str(video),
                "--headless",
                "--tag-names-file", str(f),
            ])

        assert rc == 0

    def test_tag_names_file_missing_warns(self, capsys, tmp_path):
        """Missing --tag-names-file should print a warning."""
        video = tmp_path / "black.mp4"
        _make_dummy_video(video)

        with patch(
            "robo_vision.april_tag_detector._apriltags_available",
            return_value=False,
        ):
            from main import main

            rc = main([
                "--source", str(video),
                "--headless",
                "--tag-names-file", str(tmp_path / "missing.json"),
            ])

        assert rc == 0
        err = capsys.readouterr().err
        assert "not found" in err
