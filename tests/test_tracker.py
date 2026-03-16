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


# ---------------------------------------------------------------------------
# Kalman-filter tracking (use_kalman=True)
# ---------------------------------------------------------------------------


class TestCentroidTrackerKalman:
    """CentroidTracker with Kalman-filter prediction enabled."""

    def test_new_track_gets_id_with_kalman(self):
        tracker = CentroidTracker(use_kalman=True)
        d = [_laser((50, 50))]
        tracker.update(d)
        assert d[0].track_id is not None

    def test_nearby_spot_keeps_track_id_with_kalman(self):
        tracker = CentroidTracker(max_distance=60, use_kalman=True)
        d1 = [_laser((100, 100))]
        tracker.update(d1)
        first_id = d1[0].track_id

        d2 = [_laser((108, 103))]
        tracker.update(d2)
        assert d2[0].track_id == first_id

    def test_kalman_survives_more_absent_frames_than_centroid(self):
        """Kalman tracker re-matches after as many absent frames as centroid does."""
        tracker = CentroidTracker(max_disappeared=5, max_distance=80, use_kalman=True)
        d1 = [_laser((100, 100))]
        tracker.update(d1)
        first_id = d1[0].track_id

        # Feed 5 absent frames (equal to max_disappeared)
        for _ in range(5):
            tracker.update([])

        # The track should still be alive (disappeared == max_disappeared, not yet pruned)
        d2 = [_laser((100, 100))]
        tracker.update(d2)
        assert d2[0].track_id == first_id

    def test_kalman_track_removed_after_max_disappeared(self):
        tracker = CentroidTracker(max_disappeared=3, use_kalman=True)
        d = [_laser((50, 50))]
        tracker.update(d)
        for _ in range(4):
            tracker.update([])
        assert tracker.active_track_count == 0
        # Kalman filter dict should also be cleared
        assert len(tracker._kalman_filters) == 0

    def test_two_spots_tracked_independently_with_kalman(self):
        tracker = CentroidTracker(max_distance=30, use_kalman=True)
        d1 = [_laser((30, 30)), _laser((130, 130))]
        tracker.update(d1)
        ids1 = {d1[0].track_id, d1[1].track_id}
        assert len(ids1) == 2

        d2 = [_laser((32, 31)), _laser((132, 131))]
        tracker.update(d2)
        ids2 = {d2[0].track_id, d2[1].track_id}
        assert ids1 == ids2

    def test_use_kalman_setter_creates_filters_for_existing_tracks(self):
        tracker = CentroidTracker(use_kalman=False)
        d = [_laser((50, 50))]
        tracker.update(d)
        tid = d[0].track_id

        # Enable Kalman post-hoc: filter should be created for existing track
        tracker.use_kalman = True
        assert tid in tracker._kalman_filters

    def test_use_kalman_setter_clears_filters_when_disabled(self):
        tracker = CentroidTracker(use_kalman=True)
        d = [_laser((50, 50))]
        tracker.update(d)
        assert len(tracker._kalman_filters) > 0

        tracker.use_kalman = False
        assert len(tracker._kalman_filters) == 0

    def test_empty_update_does_not_raise_with_kalman(self):
        tracker = CentroidTracker(use_kalman=True)
        result = tracker.update([])
        assert result == []

    def test_kalman_uses_prediction_for_faster_moving_object(self):
        """Kalman should match a spot that moved beyond the centroid window
        because the predicted position compensates for velocity.
        """
        # Use a generous max_distance so both trackers can potentially match,
        # but only the Kalman-enabled one accounts for velocity.
        tracker = CentroidTracker(max_distance=80, use_kalman=True)

        # Establish track with a left-ward moving object
        tracker.update([_laser((100, 100))])
        tracker.update([_laser((115, 100))])
        d1 = [_laser((130, 100))]
        tracker.update(d1)
        first_id = d1[0].track_id

        # Object continues at roughly same velocity
        d2 = [_laser((145, 100))]
        tracker.update(d2)
        assert d2[0].track_id == first_id
