"""Tests for CentroidTracker."""

from __future__ import annotations

import pytest

from robo_eye_sense.results import Detection, DetectionType
from robo_eye_sense.tracker import CentroidTracker


def _laser(center, **kwargs):
    return Detection(
        detection_type=DetectionType.LASER_SPOT,
        identifier=None,
        center=center,
        **kwargs,
    )


def _tag(tag_id, center, **kwargs):
    return Detection(
        detection_type=DetectionType.APRIL_TAG,
        identifier=str(tag_id),
        center=center,
        **kwargs,
    )


def _qr(data, center, **kwargs):
    return Detection(
        detection_type=DetectionType.QR_CODE,
        identifier=data,
        center=center,
        **kwargs,
    )


class TestCentroidTrackerLabeled:
    def test_new_labeled_detection_gets_track_id(self):
        tracker = CentroidTracker()
        detections = [_tag(1, (100, 100))]
        tracker.update(detections)
        assert detections[0].track_id is not None

    def test_same_tag_keeps_same_track_id(self):
        tracker = CentroidTracker()
        d1 = [_tag(5, (100, 100))]
        tracker.update(d1)
        first_id = d1[0].track_id

        d2 = [_tag(5, (105, 102))]
        tracker.update(d2)
        assert d2[0].track_id == first_id

    def test_different_tag_ids_get_different_track_ids(self):
        tracker = CentroidTracker()
        d = [_tag(1, (100, 100)), _tag(2, (150, 150))]
        tracker.update(d)
        assert d[0].track_id != d[1].track_id

    def test_labeled_track_survives_brief_absence(self):
        tracker = CentroidTracker(max_disappeared=3)
        d1 = [_tag(3, (100, 100))]
        tracker.update(d1)
        first_id = d1[0].track_id

        # 2 absent frames
        tracker.update([])
        tracker.update([])

        # reappears
        d2 = [_tag(3, (100, 100))]
        tracker.update(d2)
        assert d2[0].track_id == first_id

    def test_labeled_track_removed_after_max_disappeared(self):
        tracker = CentroidTracker(max_disappeared=2)
        d = [_tag(7, (100, 100))]
        tracker.update(d)
        for _ in range(3):
            tracker.update([])
        assert tracker.active_track_count == 0

    def test_qr_code_labeled_tracking(self):
        tracker = CentroidTracker()
        d1 = [_qr("TARGET_A", (60, 60))]
        tracker.update(d1)
        first_id = d1[0].track_id

        d2 = [_qr("TARGET_A", (62, 61))]
        tracker.update(d2)
        assert d2[0].track_id == first_id


class TestCentroidTrackerUnlabeled:
    def test_new_laser_spot_gets_track_id(self):
        tracker = CentroidTracker()
        d = [_laser((50, 50))]
        tracker.update(d)
        assert d[0].track_id is not None

    def test_nearby_laser_spot_keeps_track_id(self):
        tracker = CentroidTracker(max_distance=30)
        d1 = [_laser((100, 100))]
        tracker.update(d1)
        first_id = d1[0].track_id

        d2 = [_laser((105, 102))]
        tracker.update(d2)
        assert d2[0].track_id == first_id

    def test_far_laser_spot_gets_new_track_id(self):
        tracker = CentroidTracker(max_distance=30)
        d1 = [_laser((10, 10))]
        tracker.update(d1)
        first_id = d1[0].track_id

        d2 = [_laser((190, 190))]
        tracker.update(d2)
        # Old track gets a new track_id because it's too far away
        assert d2[0].track_id != first_id

    def test_two_spots_tracked_independently(self):
        tracker = CentroidTracker(max_distance=30)
        d1 = [_laser((30, 30)), _laser((130, 130))]
        tracker.update(d1)
        ids_frame1 = {d1[0].track_id, d1[1].track_id}
        assert len(ids_frame1) == 2

        d2 = [_laser((31, 31)), _laser((131, 131))]
        tracker.update(d2)
        ids_frame2 = {d2[0].track_id, d2[1].track_id}
        assert ids_frame1 == ids_frame2

    def test_unlabeled_track_removed_after_max_disappeared(self):
        tracker = CentroidTracker(max_disappeared=2)
        d = [_laser((50, 50))]
        tracker.update(d)
        assert tracker.active_track_count == 1

        for _ in range(3):
            tracker.update([])
        assert tracker.active_track_count == 0

    def test_empty_update_does_not_raise(self):
        tracker = CentroidTracker()
        result = tracker.update([])
        assert result == []

    def test_mixed_labeled_and_unlabeled(self):
        tracker = CentroidTracker()
        detections = [_tag(1, (100, 100)), _laser((50, 50))]
        tracker.update(detections)
        # Both should have track IDs
        assert all(d.track_id is not None for d in detections)
        # IDs should be distinct
        assert detections[0].track_id != detections[1].track_id
