"""Centroid-based multi-object tracker.

Labeled objects (AprilTags, QR codes) are matched by their
``(detection_type, identifier)`` pair so that a physical marker always
keeps the same track ID even after a brief occlusion.

Unlabeled objects (laser spots) are matched by nearest-centroid with a
configurable maximum pixel distance.  When *use_kalman* is enabled every
unlabeled track maintains a 4-state Kalman filter
(position + velocity: ``[x, y, vx, vy]``) so that the tracker can
predict where a fast-moving or temporarily blurred object will reappear
and match it even if the detection centroid has shifted noticeably.
This Kalman mode is automatically activated in ROBUST pipeline mode.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple

import numpy as np

from .results import Detection

_SMOOTHING_ALPHA = 0.35


def _make_kalman(center: Tuple[int, int]) -> Any:
    """Return a new Kalman filter initialised at *center*.

    Requires OpenCV (cv2) and is called only when Kalman tracking is enabled.

    State vector:  ``[x, y, vx, vy]``
    Measurement:   ``[x, y]``
    Model:         constant-velocity (velocity is propagated unchanged).
    """
    import cv2

    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    kf.transitionMatrix = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=np.float32,
    )
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.05
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 5.0
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    kf.statePost = np.array([[center[0]], [center[1]], [0.0], [0.0]], dtype=np.float32)
    return kf


class CentroidTracker:
    """Assigns persistent track IDs to detected objects across video frames."""

    def __init__(
        self,
        max_disappeared: int = 10,
        max_distance: int = 50,
        use_kalman: bool = False,
    ) -> None:
        self._next_id: int = 0
        self._labeled_tracks: Dict[Tuple, int] = {}
        self._unlabeled_tracks: Dict[int, Tuple[int, int]] = {}
        self._disappeared: Dict[int, int] = {}
        self._kalman_filters: Dict[int, Any] = {}
        self._estimated_centers: Dict[int, Tuple[float, float]] = {}
        self._velocity: Dict[int, Tuple[float, float]] = {}
        self._track_age: Dict[int, int] = {}

        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self._use_kalman = use_kalman

    @property
    def use_kalman(self) -> bool:
        """Whether Kalman-filter prediction is active for unlabeled tracks."""
        return self._use_kalman

    @use_kalman.setter
    def use_kalman(self, value: bool) -> None:
        if value and not self._use_kalman:
            for tid, center in self._unlabeled_tracks.items():
                if tid not in self._kalman_filters:
                    self._kalman_filters[tid] = _make_kalman(center)
        elif not value:
            self._kalman_filters.clear()
        self._use_kalman = value

    def update(self, detections: List[Detection]) -> List[Detection]:
        """Assign ``track_id`` and quality diagnostics to *detections* in-place."""
        labeled = [d for d in detections if d.identifier is not None]
        unlabeled = [d for d in detections if d.identifier is None]
        self._update_labeled(labeled)
        self._update_unlabeled(unlabeled)
        return detections

    @property
    def active_track_count(self) -> int:
        """Total number of currently active tracks (labeled + unlabeled)."""
        return len(self._labeled_tracks) + len(self._unlabeled_tracks)

    def _new_id(self) -> int:
        track_id = self._next_id
        self._next_id += 1
        return track_id

    def _drop_track(self, track_id: int) -> None:
        self._unlabeled_tracks.pop(track_id, None)
        self._kalman_filters.pop(track_id, None)
        self._disappeared.pop(track_id, None)
        self._estimated_centers.pop(track_id, None)
        self._velocity.pop(track_id, None)
        self._track_age.pop(track_id, None)

    def _ensure_track_state(self, track_id: int, center: Tuple[int, int]) -> None:
        self._estimated_centers.setdefault(track_id, (float(center[0]), float(center[1])))
        self._velocity.setdefault(track_id, (0.0, 0.0))
        self._track_age.setdefault(track_id, 0)

    def _update_smoothed_center(
        self,
        track_id: int,
        measured_center: Tuple[int, int],
    ) -> Tuple[int, int]:
        self._ensure_track_state(track_id, measured_center)
        prev_x, prev_y = self._estimated_centers[track_id]
        new_x = prev_x + _SMOOTHING_ALPHA * (measured_center[0] - prev_x)
        new_y = prev_y + _SMOOTHING_ALPHA * (measured_center[1] - prev_y)
        self._estimated_centers[track_id] = (new_x, new_y)
        self._velocity[track_id] = (new_x - prev_x, new_y - prev_y)
        return int(round(new_x)), int(round(new_y))

    def _populate_quality_metrics(
        self,
        detection: Detection,
        track_id: int,
        measured_center: Tuple[int, int],
        match_distance: float,
    ) -> None:
        estimated_center = self._update_smoothed_center(track_id, measured_center)
        age = self._track_age.get(track_id, 1)
        frames_since_seen = self._disappeared.get(track_id, 0)
        vx, vy = self._velocity.get(track_id, (0.0, 0.0))
        motion = float(np.hypot(vx, vy))

        detector_confidence = float(np.clip(detection.confidence, 0.0, 1.0))
        if detection.confidence > 1.0:
            detector_confidence = min(1.0, detection.confidence / 100.0)
        geometry_quality = min(1.0, len(detection.corners) / 4.0) if detection.corners else 0.6
        match_quality = max(0.0, 1.0 - match_distance / max(1.0, float(self.max_distance)))
        history_quality = min(1.0, age / 5.0)
        stability_quality = max(0.0, 1.0 - motion / max(10.0, float(self.max_distance)))
        visibility_quality = max(
            0.0,
            1.0 - frames_since_seen / max(1.0, float(self.max_disappeared + 1)),
        )

        detection.estimated_center = estimated_center
        detection.track_age = age
        detection.frames_since_seen = frames_since_seen
        detection.tracking_quality = float(
            np.clip(0.45 * match_quality + 0.35 * history_quality + 0.20 * visibility_quality, 0.0, 1.0)
        )
        detection.position_quality = float(
            np.clip(0.40 * detector_confidence + 0.35 * stability_quality + 0.25 * geometry_quality, 0.0, 1.0)
        )
        detection.quality_metrics = {
            "match_distance_px": float(match_distance),
            "match_quality": float(np.clip(match_quality, 0.0, 1.0)),
            "history_quality": float(history_quality),
            "stability_quality": float(np.clip(stability_quality, 0.0, 1.0)),
            "detector_confidence": detector_confidence,
        }

    def _update_labeled(self, detections: List[Detection]) -> None:
        seen_keys: Set[Tuple] = set()
        for d in detections:
            key = (d.detection_type, d.identifier)
            seen_keys.add(key)
            if key in self._labeled_tracks:
                d.track_id = self._labeled_tracks[key]
            else:
                d.track_id = self._new_id()
                self._labeled_tracks[key] = d.track_id
            self._ensure_track_state(d.track_id, d.center)
            prev_center = self._estimated_centers[d.track_id]
            self._disappeared[d.track_id] = 0
            self._track_age[d.track_id] = self._track_age.get(d.track_id, 0) + 1
            match_distance = float(np.hypot(d.center[0] - prev_center[0], d.center[1] - prev_center[1]))
            self._populate_quality_metrics(d, d.track_id, d.center, match_distance)

        for key, track_id in list(self._labeled_tracks.items()):
            if key not in seen_keys:
                self._disappeared[track_id] = self._disappeared.get(track_id, 0) + 1
                if self._disappeared[track_id] > self.max_disappeared:
                    del self._labeled_tracks[key]
                    self._drop_track(track_id)

    def _register_unlabeled(self, d: Detection) -> int:
        track_id = self._new_id()
        d.track_id = track_id
        self._unlabeled_tracks[track_id] = d.center
        self._disappeared[track_id] = 0
        self._ensure_track_state(track_id, d.center)
        self._track_age[track_id] = 1
        if self._use_kalman:
            self._kalman_filters[track_id] = _make_kalman(d.center)
        self._populate_quality_metrics(d, track_id, d.center, 0.0)
        return track_id

    def _update_unlabeled(self, detections: List[Detection]) -> None:
        if self._use_kalman:
            self._update_unlabeled_kalman(detections)
        else:
            self._update_unlabeled_centroid(detections)

    def _update_unlabeled_centroid(self, detections: List[Detection]) -> None:
        if not detections:
            for track_id in list(self._unlabeled_tracks.keys()):
                self._disappeared[track_id] = self._disappeared.get(track_id, 0) + 1
                if self._disappeared[track_id] > self.max_disappeared:
                    self._drop_track(track_id)
            return

        if not self._unlabeled_tracks:
            for d in detections:
                self._register_unlabeled(d)
            return

        track_ids = list(self._unlabeled_tracks.keys())
        track_centers = [self._unlabeled_tracks[tid] for tid in track_ids]
        input_centers = [d.center for d in detections]
        D = np.array(
            [[np.hypot(tc[0] - ic[0], tc[1] - ic[1]) for ic in input_centers] for tc in track_centers],
            dtype=float,
        )

        used_rows: Set[int] = set()
        used_cols: Set[int] = set()
        for row, col in sorted(((r, c) for r in range(D.shape[0]) for c in range(D.shape[1])), key=lambda rc: D[rc[0], rc[1]]):
            if D[row, col] > self.max_distance:
                break
            if row in used_rows or col in used_cols:
                continue
            track_id = track_ids[row]
            detections[col].track_id = track_id
            self._unlabeled_tracks[track_id] = detections[col].center
            self._disappeared[track_id] = 0
            self._track_age[track_id] = self._track_age.get(track_id, 0) + 1
            self._populate_quality_metrics(detections[col], track_id, detections[col].center, float(D[row, col]))
            used_rows.add(row)
            used_cols.add(col)

        for row in set(range(len(track_ids))) - used_rows:
            track_id = track_ids[row]
            self._disappeared[track_id] = self._disappeared.get(track_id, 0) + 1
            if self._disappeared[track_id] > self.max_disappeared:
                self._drop_track(track_id)

        for col in set(range(len(detections))) - used_cols:
            self._register_unlabeled(detections[col])

    def _update_unlabeled_kalman(self, detections: List[Detection]) -> None:
        predicted: Dict[int, Tuple[int, int]] = {}
        for tid in list(self._unlabeled_tracks.keys()):
            kf = self._kalman_filters.get(tid)
            if kf is None:
                kf = _make_kalman(self._unlabeled_tracks[tid])
                self._kalman_filters[tid] = kf
            pred = kf.predict()
            predicted[tid] = (int(pred[0, 0]), int(pred[1, 0]))

        if not detections:
            for track_id in list(self._unlabeled_tracks.keys()):
                self._disappeared[track_id] = self._disappeared.get(track_id, 0) + 1
                if self._disappeared[track_id] > self.max_disappeared:
                    self._drop_track(track_id)
            return

        if not self._unlabeled_tracks:
            for d in detections:
                self._register_unlabeled(d)
            return

        track_ids = list(predicted.keys())
        pred_centers = [predicted[tid] for tid in track_ids]
        input_centers = [d.center for d in detections]
        D = np.array(
            [[np.hypot(pc[0] - ic[0], pc[1] - ic[1]) for ic in input_centers] for pc in pred_centers],
            dtype=float,
        )

        used_rows: Set[int] = set()
        used_cols: Set[int] = set()
        for row, col in sorted(((r, c) for r in range(D.shape[0]) for c in range(D.shape[1])), key=lambda rc: D[rc[0], rc[1]]):
            if D[row, col] > self.max_distance:
                break
            if row in used_rows or col in used_cols:
                continue

            track_id = track_ids[row]
            measurement = np.array([[detections[col].center[0]], [detections[col].center[1]]], dtype=np.float32)
            corrected = self._kalman_filters[track_id].correct(measurement)
            corrected_center = (int(corrected[0, 0]), int(corrected[1, 0]))
            self._unlabeled_tracks[track_id] = corrected_center
            detections[col].track_id = track_id
            self._disappeared[track_id] = 0
            self._track_age[track_id] = self._track_age.get(track_id, 0) + 1
            self._populate_quality_metrics(detections[col], track_id, corrected_center, float(D[row, col]))
            used_rows.add(row)
            used_cols.add(col)

        for row in set(range(len(track_ids))) - used_rows:
            track_id = track_ids[row]
            self._disappeared[track_id] = self._disappeared.get(track_id, 0) + 1
            if self._disappeared[track_id] > self.max_disappeared:
                self._drop_track(track_id)

        for col in set(range(len(detections))) - used_cols:
            self._register_unlabeled(detections[col])
