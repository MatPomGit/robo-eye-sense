"""Tests for Detection dataclass and DetectionType enum."""

from __future__ import annotations

import pytest

from robo_vision.results import Detection, DetectionType


class TestDetectionType:
    def test_values(self):
        assert DetectionType.APRIL_TAG.value == "april_tag"
        assert DetectionType.QR_CODE.value == "qr_code"
        assert DetectionType.LASER_SPOT.value == "laser_spot"

    def test_enum_members(self):
        members = {m.name for m in DetectionType}
        assert members == {"APRIL_TAG", "QR_CODE", "LASER_SPOT"}


class TestDetection:
    def _make(self, **kwargs):
        defaults = dict(
            detection_type=DetectionType.LASER_SPOT,
            identifier=None,
            center=(10, 20),
            corners=[(0, 0), (10, 0), (10, 10), (0, 10)],
        )
        defaults.update(kwargs)
        return Detection(**defaults)

    def test_defaults(self):
        d = self._make()
        assert d.track_id is None
        assert d.confidence == 1.0

    def test_explicit_fields(self):
        d = self._make(
            detection_type=DetectionType.APRIL_TAG,
            identifier="42",
            center=(100, 200),
            track_id=7,
            confidence=0.95,
        )
        assert d.detection_type == DetectionType.APRIL_TAG
        assert d.identifier == "42"
        assert d.center == (100, 200)
        assert d.track_id == 7
        assert d.confidence == pytest.approx(0.95)

    def test_empty_corners(self):
        d = Detection(
            detection_type=DetectionType.LASER_SPOT,
            identifier=None,
            center=(5, 5),
        )
        assert d.corners == []

    def test_repr_contains_type(self):
        d = self._make(detection_type=DetectionType.QR_CODE, identifier="hello")
        r = repr(d)
        assert "qr_code" in r
        assert "hello" in r

    def test_repr_keeps_empty_identifier(self):
        d = self._make(detection_type=DetectionType.QR_CODE, identifier="")
        r = repr(d)
        assert "id=''" in r


def test_package_import_is_lazy_for_detector(monkeypatch):
    import importlib
    import sys

    # Work on a copy of sys.modules so this test does not affect other tests.
    modules_copy = sys.modules.copy()
    modules_copy.pop("robo_vision", None)
    modules_copy.pop("robo_vision.detector", None)
    monkeypatch.setattr(sys, "modules", modules_copy)

    module = importlib.import_module("robo_vision")

    assert "robo_vision.detector" not in sys.modules
    assert hasattr(module, "Detection")
    assert hasattr(module, "DetectionType")
